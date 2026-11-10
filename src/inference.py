"""Whole-panel focus-quality classification after configured thermal ROI cropping."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from decision import STATUS_LABELS_ZH, decide_focus_quality
from image_io import ImageData
from thermal_roi import ThermalBox, detect_black_border_roi, extract_thermal_roi

INFERENCE_BATCH_SIZE = 16


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

    def _prepare_array(
        self, image_data: ImageData, image_name: str
    ) -> tuple[dict[str, Any], np.ndarray, ThermalBox]:
        """Extract a panel and create the per-image result metadata."""
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
        return result, thermal_image, thermal_box

    def _finish_prediction(
        self,
        prepared: tuple[dict[str, Any], np.ndarray, ThermalBox],
        model_output: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], np.ndarray, ThermalBox]:
        result, thermal_image, thermal_box = prepared
        if model_output is None:
            decision = decide_focus_quality(None)
            result["reasons"] = list(decision.reasons)
            result["error"] = "trained focus-quality classifier weights are required"
            return result, thermal_image, thermal_box
        decision = decide_focus_quality(
            model_output,
            confidence_threshold=float(
                self.config.get("classifier", {}).get("confidence_threshold", 0.60)
            ),
        )
        result["status"] = decision.status
        result["focus_quality"] = decision.status
        result["label_zh"] = STATUS_LABELS_ZH[decision.status]
        if decision.status != "uncertain" or decision.reasons == (
            "classifier_low_confidence",
        ):
            result["probabilities"] = model_output["probabilities"]
        result["reasons"] = list(decision.reasons)
        return result, thermal_image, thermal_box

    def predict_batch(
        self, inputs: Sequence[tuple[ImageData, str]]
    ) -> list[tuple[dict[str, Any], np.ndarray, ThermalBox] | Exception]:
        """Prepare panels concurrently and classify them in a single model batch.

        Outcomes stay in input order. A preparation or batch-classification
        error is returned for the affected input(s), allowing callers to keep
        processing the rest of the batch.
        """
        if not inputs:
            return []

        prepared: list[
            tuple[dict[str, Any], np.ndarray, ThermalBox] | Exception | None
        ] = [None] * len(inputs)
        if len(inputs) == 1:
            try:
                prepared[0] = self._prepare_array(*inputs[0])
            except (OSError, ValueError, RuntimeError) as exc:
                prepared[0] = exc
        else:
            with ThreadPoolExecutor(max_workers=min(4, len(inputs))) as executor:
                futures = {
                    executor.submit(self._prepare_array, image_data, image_name): index
                    for index, (image_data, image_name) in enumerate(inputs)
                }
                for future in as_completed(futures):
                    try:
                        prepared[futures[future]] = future.result()
                    except (OSError, ValueError, RuntimeError) as exc:
                        prepared[futures[future]] = exc

        valid_indices = [
            index
            for index, item in enumerate(prepared)
            if isinstance(item, tuple)
        ]
        model_outputs: list[dict[str, Any] | None] = [None] * len(inputs)
        if self.classifier is not None and valid_indices:
            from classifier import classify_tensors, preprocess_panel

            for start in range(0, len(valid_indices), INFERENCE_BATCH_SIZE):
                batch_indices = valid_indices[start : start + INFERENCE_BATCH_SIZE]
                input_indices = []
                input_tensors = []
                input_size = int(self.classifier_metadata.get("input_size", 224))
                for index in batch_indices:
                    item = prepared[index]
                    if not isinstance(item, tuple):
                        raise RuntimeError("batch preparation outcome changed unexpectedly")
                    try:
                        tensor = preprocess_panel(item[1], input_size)
                    except (OSError, ValueError, RuntimeError) as exc:
                        prepared[index] = exc
                    else:
                        input_indices.append(index)
                        input_tensors.append(tensor)
                if not input_tensors:
                    continue
                try:
                    batch_outputs = classify_tensors(
                        self.classifier,
                        input_tensors,
                        device=self.device,
                    )
                except (OSError, ValueError, RuntimeError) as exc:
                    for index in input_indices:
                        prepared[index] = exc
                else:
                    for index, model_output in zip(input_indices, batch_outputs):
                        model_outputs[index] = model_output

        outcomes: list[
            tuple[dict[str, Any], np.ndarray, ThermalBox] | Exception
        ] = []
        for index, item in enumerate(prepared):
            if isinstance(item, Exception):
                outcomes.append(item)
            elif item is None:
                raise RuntimeError("batch preparation produced no outcome")
            else:
                outcomes.append(self._finish_prediction(item, model_outputs[index]))
        return outcomes

    def predict_array(
        self, image_data: ImageData, *, image_name: str
    ) -> tuple[dict[str, Any], np.ndarray, ThermalBox]:
        """Detect a dark frame or fall back to the configured panel crop."""
        outcome = self.predict_batch([(image_data, image_name)])[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
