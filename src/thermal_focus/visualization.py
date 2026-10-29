"""Write image-level focus-quality previews and JSON results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
from numpy.typing import NDArray

from .thermal_roi import ThermalBox, draw_thermal_roi

STATUS_COLORS = {
    "sharp": (0, 180, 0),
    "borderline": (0, 190, 230),
    "defocused": (0, 0, 220),
}
FALLBACK_COLOR = (140, 140, 140)


def render_annotated_image(
    image: NDArray,
    thermal_box: ThermalBox,
    *,
    label: str,
    status: str,
    score: float | None = None,
) -> NDArray:
    """Draw the thermal box and a status caption on a copy of the original image."""
    annotated = draw_thermal_roi(image.copy(), thermal_box)
    caption = label if score is None else f"{label} | Score: {score:.1f}/100"
    cv2.putText(
        annotated,
        caption,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        STATUS_COLORS.get(status, FALLBACK_COLOR),
        2,
        cv2.LINE_AA,
    )
    return annotated


def _write_image(path: Path, image: NDArray) -> None:
    success, encoded = cv2.imencode(path.suffix.lower(), image)
    if not success:
        raise OSError(f"failed to encode visualization: {path}")
    try:
        encoded.tofile(path)
    except OSError as exc:
        raise OSError(f"failed to write visualization: {path}: {exc}") from exc


def write_error_json(
    output_dir: str | Path,
    stem: str,
    image_name: str,
    error: str,
    *,
    reason: str = "image_read_error",
) -> str:
    """Write an explicit error-state prediction when image loading fails."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{stem}.json"
    result = {
        "image": image_name,
        "status": "error",
        "focus_quality": None,
        "label_zh": "无法评估",
        "confidence": None,
        "model_score": None,
        "probabilities": None,
        "task": "thermal_focus_quality",
        "reasons": [reason],
        "error": error,
    }
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return str(path)


def write_visualizations(
    output_dir: str | Path,
    stem: str,
    image: NDArray,
    thermal_image: NDArray,
    thermal_box: ThermalBox,
    result: dict[str, Any],
) -> dict[str, str]:
    """Save original image, cropped thermal panel, and focus-quality JSON."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    original_path = target_dir / f"{stem}.annotated.jpg"
    panel_path = target_dir / f"{stem}.thermal.png"
    result_path = target_dir / f"{stem}.json"
    annotated = render_annotated_image(
        image,
        thermal_box,
        label=str(result.get("label_zh", result.get("status", "uncertain"))),
        status=str(result.get("status")),
        score=result.get("model_score")
        if isinstance(result.get("model_score"), (int, float))
        and not isinstance(result.get("model_score"), bool)
        else None,
    )
    _write_image(original_path, annotated)
    _write_image(panel_path, thermal_image)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return {
        "annotated": str(original_path),
        "thermal": str(panel_path),
        "json": str(result_path),
    }
