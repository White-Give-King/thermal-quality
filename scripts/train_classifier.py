"""Train a supported torchvision model to classify full thermal panels for focus quality."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from classifier import (
    ARCHITECTURE_MODEL_VERSIONS,
    ARCHITECTURES,
    CLASSES,
    DEFAULT_ARCHITECTURE,
    build_classifier,
    preprocess_panel,
)
from dataset import load_annotations, split_dataset
from image_io import load_image

LOGGER = logging.getLogger("thermal_focus_quality.train_focus_quality")


class ThermalPanelDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, input_size: int) -> None:
        self.frame = frame.reset_index(drop=True)
        self.input_size = input_size
        self.label_to_index = {label: index for index, label in enumerate(CLASSES)}

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        image = load_image(row["image_path"]).image
        return preprocess_panel(image, self.input_size), self.label_to_index[row["label"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", type=Path, help="local checkpoint to continue training from")
    parser.add_argument(
        "--architecture",
        choices=ARCHITECTURES,
        default="mobilenet_v3_large",
        help="classifier backbone (default: mobilenet_v3_large)",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="download torchvision weights (default; use --no-pretrained to disable)",
    )
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def _write_annotation_database(path: Path, frame: pd.DataFrame) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("BEGIN")
        connection.execute("DROP TABLE IF EXISTS image_annotations")
        connection.execute(
            """
            CREATE TABLE image_annotations (
                image_path TEXT PRIMARY KEY,
                label TEXT NOT NULL CHECK (label IN ('sharp', 'borderline', 'defocused')),
                source_path TEXT NOT NULL DEFAULT '',
                annotated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.executemany(
            "INSERT INTO image_annotations (image_path, label) VALUES (?, ?)",
            frame[["image_path", "label"]].itertuples(index=False, name=None),
        )


def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)
            total_loss += float(loss.item()) * labels.size(0)
            total_samples += labels.size(0)
    return total_loss / max(total_samples, 1)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.input_size <= 0 or args.learning_rate <= 0:
        raise SystemExit("epochs, batch-size, input-size, and learning-rate must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; use --device cpu")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    frame = load_annotations(args.database, image_root=args.image_root)
    try:
        splits = split_dataset(frame, seed=args.seed)
    except ValueError as exc:
        raise SystemExit(
            f"cannot create train/validation/test splits: {exc}. "
            "Add enough labeled examples for all three focus-quality classes."
        ) from exc
    args.output.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", splits.train), ("validation", splits.validation), ("test", splits.test)):
        _write_annotation_database(args.output / f"{name}_split.sqlite3", split)
        LOGGER.info("%s rows=%d labels=%s", name, len(split), split.label.value_counts().to_dict())

    train_dataset = ThermalPanelDataset(splits.train, args.input_size)
    validation_dataset = ThermalPanelDataset(splits.validation, args.input_size)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    model = build_classifier(
        pretrained=args.pretrained,
        architecture=args.architecture,
    ).to(device)
    start_epoch = 0
    resume_checkpoint = None
    if args.resume:
        try:
            resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
            if tuple(resume_checkpoint.get("classes", ())) != CLASSES:
                raise ValueError(f"checkpoint classes must be {CLASSES}")
            checkpoint_architecture = str(
                resume_checkpoint.get("architecture", DEFAULT_ARCHITECTURE)
            )
            if checkpoint_architecture != args.architecture:
                raise ValueError(
                    f"checkpoint architecture {checkpoint_architecture!r} does not match "
                    f"--architecture {args.architecture!r}"
                )
            if int(resume_checkpoint.get("input_size", args.input_size)) != args.input_size:
                raise ValueError("checkpoint input_size does not match --input-size")
            model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
            start_epoch = int(resume_checkpoint.get("epoch", 0))
        except (OSError, RuntimeError, KeyError, ValueError) as exc:
            raise SystemExit(f"could not resume from {args.resume}: {exc}") from exc
    counts = Counter(splits.train["label"])
    class_weights = torch.tensor(
        [len(splits.train) / (len(CLASSES) * counts[name]) for name in CLASSES],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if resume_checkpoint and "optimizer_state_dict" in resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
    best_validation_loss = (
        float(resume_checkpoint.get("best_validation_loss", resume_checkpoint.get("validation_loss", float("inf"))))
        if resume_checkpoint
        else float("inf")
    )
    best_path = args.output / "best.pt"
    last_path = args.output / "last.pt"
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
        validation_loss = evaluate_loss(model, validation_loader, criterion, device)
        improved = validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
        LOGGER.info(
            "epoch=%d train_loss=%.5f validation_loss=%.5f",
            epoch + 1,
            total_loss / len(train_dataset),
            validation_loss,
        )
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "classes": list(CLASSES),
            "architecture": args.architecture,
            "model_version": ARCHITECTURE_MODEL_VERSIONS[args.architecture],
            "input_size": args.input_size,
            "epoch": epoch + 1,
            "validation_loss": validation_loss,
            "best_validation_loss": best_validation_loss,
            "seed": args.seed,
        }
        torch.save(payload, last_path)
        if improved:
            torch.save(payload, best_path)
    metadata = {
        "architecture": args.architecture,
        "model_version": ARCHITECTURE_MODEL_VERSIONS[args.architecture],
        "classes": list(CLASSES),
        "input_size": args.input_size,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "pretrained": args.pretrained,
        "resume": str(args.resume) if args.resume else None,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "class_distribution": frame["label"].value_counts().to_dict(),
    }
    (args.output / "training_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("best=%s last=%s test_split_saved_only=true", best_path, last_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
