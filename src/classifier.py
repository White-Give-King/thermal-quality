"""Torchvision image classifiers for whole-panel focus-quality classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
)

from decision import LABELS

CLASSES = LABELS
ARCHITECTURES = (
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "resnet18",
)
DEFAULT_ARCHITECTURE = "mobilenet_v3_small"
MODEL_VERSION = f"{DEFAULT_ARCHITECTURE}-thermal-focus-quality-1"
ARCHITECTURE_MODEL_VERSIONS = {
    architecture: f"{architecture}-thermal-focus-quality-1"
    for architecture in ARCHITECTURES
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_classifier(
    pretrained: bool = False, architecture: str = DEFAULT_ARCHITECTURE
) -> nn.Module:
    """Build a supported torchvision backbone; downloads occur only when requested."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unsupported architecture {architecture!r}; choose from {ARCHITECTURES}")
    model_factories = {
        "mobilenet_v3_small": (
            models.mobilenet_v3_small,
            MobileNet_V3_Small_Weights,
        ),
        "mobilenet_v3_large": (
            models.mobilenet_v3_large,
            MobileNet_V3_Large_Weights,
        ),
        "efficientnet_b0": (models.efficientnet_b0, EfficientNet_B0_Weights),
        "resnet18": (models.resnet18, ResNet18_Weights),
    }
    factory, weights_type = model_factories[architecture]
    model = factory(weights=weights_type.DEFAULT if pretrained else None)
    if architecture.startswith("mobilenet_v3_") or architecture == "efficientnet_b0":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    else:
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def preprocess_panel(roi: NDArray[np.uint8], input_size: int = 224) -> torch.Tensor:
    """Convert a full thermal panel to RGB while preserving its aspect ratio."""
    if roi.size == 0 or roi.ndim not in (2, 3):
        raise ValueError("thermal panel must be a non-empty gray or color image")
    if not np.isfinite(roi).all():
        raise ValueError("thermal panel contains non-finite values")
    if input_size <= 0:
        raise ValueError("input_size must be positive")
    if roi.ndim == 2:
        rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    elif roi.shape[2] == 1:
        rgb = cv2.cvtColor(roi[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif roi.shape[2] == 4:
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGRA2RGB)
    elif roi.shape[2] == 3:
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("thermal panel must have one, three, or four channels")
    height, width = rgb.shape[:2]
    scale = min(input_size / width, input_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(
        rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    pad_x = input_size - resized_width
    pad_y = input_size - resized_height
    left, top = pad_x // 2, pad_y // 2
    resized = cv2.copyMakeBorder(
        resized,
        top,
        pad_y - top,
        left,
        pad_x - left,
        cv2.BORDER_REPLICATE,
    )
    tensor = torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1))).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def load_classifier(
    checkpoint: str | Path, device: str = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """Restore a locally supplied checkpoint; missing or incompatible weights are errors."""
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"classifier checkpoint does not exist: {checkpoint_path}")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this Python environment")
    try:
        payload = torch.load(checkpoint_path, map_location=resolved_device, weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"could not load classifier checkpoint {checkpoint_path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise ValueError("checkpoint must contain model_state_dict and class metadata")
    classes = tuple(payload.get("classes", ()))
    if classes != CLASSES:
        raise ValueError(f"checkpoint classes must be {CLASSES}, got {classes}")
    architecture = str(payload.get("architecture", DEFAULT_ARCHITECTURE))
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"checkpoint architecture {architecture!r} is unsupported; "
            f"supported architectures are {ARCHITECTURES}"
        )
    model = build_classifier(pretrained=False, architecture=architecture)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(resolved_device).eval()
    metadata = {
        "model_version": str(payload.get("model_version", MODEL_VERSION)),
        "input_size": int(payload.get("input_size", 224)),
        "classes": list(classes),
        "architecture": architecture,
    }
    return model, metadata


@torch.inference_mode()
def classify_panel(
    model: nn.Module, roi: NDArray[np.uint8], *, device: str, input_size: int = 224
) -> dict[str, Any]:
    """Return actual softmax probabilities and the predicted class."""
    target_device = torch.device(device)
    model.to(target_device).eval()
    tensor = preprocess_panel(roi, input_size).unsqueeze(0).to(target_device)
    logits = model(tensor)
    if logits.ndim != 2 or logits.shape != (1, len(CLASSES)):
        raise ValueError(f"classifier returned unexpected logits shape {tuple(logits.shape)}")
    probabilities = torch.softmax(logits[0], dim=0).cpu().numpy()
    index = int(np.argmax(probabilities))
    return {
        "predicted_class": CLASSES[index],
        "confidence": float(probabilities[index]),
        "probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASSES)},
    }
