"""Evaluate a trained focus-quality classifier on labeled test data."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from thermal_focus.classifier import classify_panel, load_classifier
from thermal_focus.dataset import load_annotations
from thermal_focus.decision import decide_focus_quality
from thermal_focus.evaluation import summarize
from thermal_focus.image_io import load_image
from thermal_focus.inference import load_config

LOGGER = logging.getLogger("thermal_focus.evaluate_focus_quality")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path, help="SQLite annotation database")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--model", required=True, type=Path, help="trained focus-quality classifier checkpoint")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[1] / "configs" / "default.yaml",
    )
    parser.add_argument("--output", type=Path, default=Path("evaluation.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = load_config(args.config)
    frame = load_annotations(args.database, image_root=args.image_root)
    model, metadata = load_classifier(args.model, device=args.device)
    confidence_threshold = float(
        config.get("classifier", {}).get("confidence_threshold", 0.60)
    )
    predictions: list[str] = []
    durations_ms: list[float] = []
    for image_path in frame["image_path"]:
        image = load_image(image_path).image
        started = time.perf_counter()
        output = classify_panel(
            model, image, device=args.device, input_size=metadata["input_size"]
        )
        decision = decide_focus_quality(
            output, confidence_threshold=confidence_threshold
        )
        predictions.append(decision.status)
        durations_ms.append((time.perf_counter() - started) * 1000.0)
    labels = frame["label"].tolist()
    report = {
        "dataset": str(args.database.resolve()),
        "sample_count": len(frame),
        "task": "thermal_focus_quality",
        "class_semantics": {
            "sharp": "image details and target contours are clearly defined",
            "borderline": "focus quality is borderline and should be reviewed",
            "defocused": "image details and contours are visibly blurred or defocused",
        },
        "mean_inference_latency_ms": float(np.mean(durations_ms)),
        "p95_inference_latency_ms": float(np.percentile(durations_ms, 95)),
        "model_version": metadata["model_version"],
        "architecture": metadata["architecture"],
        "classifier": summarize(labels, predictions),
        "note": "Metrics describe this labeled set only; use an independent test set for performance claims.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("wrote focus-quality evaluation report to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
