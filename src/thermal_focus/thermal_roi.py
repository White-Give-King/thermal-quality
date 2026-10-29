"""Fixed-canvas valid thermal-panel cropping and preview utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

ThermalBox: TypeAlias = tuple[int, int, int, int]


def _thermal_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    settings = config.get("thermal_roi", config)
    if not isinstance(settings, Mapping):
        raise ValueError("thermal_roi configuration must be a mapping")
    return settings


def _positive_integer(settings: Mapping[str, Any], name: str) -> int:
    value = settings.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"thermal_roi.{name} must be a positive integer")
    return int(value)


def _coordinate(settings: Mapping[str, Any], name: str) -> int:
    value = settings.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"thermal_roi.{name} must be a non-negative integer")
    return int(value)


def validate_thermal_roi(
    image_shape: Sequence[int], config: Mapping[str, Any]
) -> ThermalBox:
    """Validate fixed input dimensions and return the half-open XYWH crop.

    Coordinates use the original image's top-left origin. The returned box is
    ``(x, y, width, height)`` and denotes ``[x, x+width) x [y, y+height)``.
    """
    if len(image_shape) not in (2, 3):
        raise ValueError("image_shape must be (height, width) or (height, width, channels)")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image must have non-zero width and height")

    image_settings = config.get("image", {})
    if not isinstance(image_settings, Mapping):
        raise ValueError("image configuration must be a mapping")
    expected_width = image_settings.get("expected_width")
    expected_height = image_settings.get("expected_height")
    if expected_width is not None:
        if isinstance(expected_width, bool) or not isinstance(expected_width, int) or expected_width <= 0:
            raise ValueError("image.expected_width must be a positive integer")
        if width != expected_width:
            raise ValueError(
                f"input width {width} does not match configured image.expected_width {expected_width}"
            )
    if expected_height is not None:
        if isinstance(expected_height, bool) or not isinstance(expected_height, int) or expected_height <= 0:
            raise ValueError("image.expected_height must be a positive integer")
        if height != expected_height:
            raise ValueError(
                f"input height {height} does not match configured image.expected_height {expected_height}"
            )

    settings = _thermal_settings(config)
    mode = settings.get("mode", "fixed")
    if mode != "fixed":
        raise ValueError(f"unsupported thermal_roi.mode {mode!r}; only fixed mode is implemented")
    x = _coordinate(settings, "x")
    y = _coordinate(settings, "y")
    roi_width = _positive_integer(settings, "width")
    roi_height = _positive_integer(settings, "height")
    if x + roi_width > width or y + roi_height > height:
        raise ValueError(
            f"thermal ROI [{x}, {x + roi_width}) x [{y}, {y + roi_height}) "
            f"exceeds input dimensions {width}x{height}"
        )
    return (x, y, roi_width, roi_height)


def extract_thermal_roi(
    image: NDArray[np.generic], config: Mapping[str, Any]
) -> tuple[NDArray[np.generic], ThermalBox]:
    """Return a source-preserving thermal-panel crop and original-image XYWH box."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image is empty or is not a NumPy array")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3, 4)):
        raise ValueError("image must be gray, BGR, or BGRA")
    if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
        raise ValueError("image must contain finite numeric pixels")
    x, y, width, height = validate_thermal_roi(image.shape, config)
    return image[y : y + height, x : x + width].copy(), (x, y, width, height)


def draw_thermal_roi(
    image: NDArray[np.uint8], roi: Sequence[int], *, color: tuple[int, int, int] = (0, 255, 0), thickness: int = 2
) -> NDArray[np.uint8]:
    """Draw the validated half-open thermal box on a copy of the original image."""
    if image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("image must be a non-empty gray, BGR, or BGRA array")
    if len(roi) != 4:
        raise ValueError("roi must be (x, y, width, height)")
    x, y, width, height = roi
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in roi):
        raise ValueError("ROI coordinates and dimensions must be integers")
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > image.shape[1] or y + height > image.shape[0]:
        raise ValueError("thermal ROI is outside the image or has zero area")
    if thickness <= 0:
        raise ValueError("thickness must be positive")
    output = image.copy()
    cv2.rectangle(output, (int(x), int(y)), (int(x + width - 1), int(y + height - 1)), color, thickness)
    return output

