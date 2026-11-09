"""Whole-panel focus-quality classification after configured thermal ROI cropping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from decision import STATUS_LABELS_ZH, decide_focus_quality
from image_io import ImageData
from thermal_roi import (
    ThermalBox,
    detect_black_border_roi,
    extract_thermal_roi,
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config and reject empty or non-mapping documents."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a YAML mapping: {config_path}")
    return config


class InferenceEngine:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        classifier: Any | None = None,
        classifier_metadata: dict[str, Any] | None = None,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.classifier = classifier
        self.classifier_metadata = classifier_metadata or {}
        self.device = device

    def predict_array(
        self, image_data: ImageData, *, image_name: str
    ) -> tuple[dict[str, Any], np.ndarray, ThermalBox]:
        """Detect a dark frame or fall back to the configured panel crop."""
        thermal_box = detect_black_border_roi(image_data.image)
        if thermal_box is None:
            thermal_image, thermal_box = extract_thermal_roi(
                image_data.image, self.config
            )
        else:
            x, y, width, height = thermal_box
            thermal_image = image_data.image[y : y + height, x : x + width].copy()
        result: dict[str, Any] = {
            "image": image_name,
            "status": "uncertain",
            "focus_quality": None,
            "label_zh": STATUS_LABELS_ZH["uncertain"],
            "confidence": None,
            "model_score": None,
            "probabilities": None,
            "task": "thermal_focus_quality",
            "thermal_roi": {
                "xywh": list(thermal_box),
                "size": {
                    "width": thermal_image.shape[1],
                    "height": thermal_image.shape[0],
                },
            },
            "image_size": {"width": image_data.width, "height": image_data.height},
            "source_kind": image_data.source_kind,
            "reasons": [],
            "model_version": self.classifier_metadata.get("model_version"),
            "architecture": self.classifier_metadata.get("architecture"),
            "error": None,
        }
        if self.classifier is None:
            decision = decide_focus_quality(None)
            result["reasons"] = list(decision.reasons)
            result["error"] = "trained focus-quality classifier weights are required"
            return result, thermal_image, thermal_box

        from classifier import classify_panel

        model_output = classify_panel(
            self.classifier,
            thermal_image,
            device=self.device,
            input_size=int(self.classifier_metadata.get("input_size", 224)),
        )
        decision = decide_focus_quality(
            model_output,
            confidence_threshold=float(
                self.config.get("classifier", {}).get("confidence_threshold", 0.60)
            ),
        )
        result["status"] = decision.status
        result["focus_quality"] = decision.status
        result["label_zh"] = STATUS_LABELS_ZH[decision.status]
        result["confidence"] = model_output["confidence"]
        result["model_score"] = round(float(model_output["confidence"]) * 100, 1)
        result["probabilities"] = model_output["probabilities"]
        result["reasons"] = list(decision.reasons)
        return result, thermal_image, thermal_box
