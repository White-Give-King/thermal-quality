"""Image loading that preserves source pixels and distinguishes pseudo-color data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

ImageArray = NDArray[np.uint8]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageData:
    image: ImageArray
    width: int
    height: int
    channels: int
    source_kind: str = "pseudo_color"


def normalize_image(image: np.ndarray, float_range: tuple[float, float] | None = None) -> ImageArray:
    """Convert gray/BGR/BGRA or explicitly ranged numeric pixels to BGR uint8.

    Integer pixels are converted with a dtype-range scale, never per-image
    contrast stretching. Floating point arrays require an explicit range so
    radiometric values cannot be mistaken for display RGB values.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image is empty or is not a NumPy array")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3, 4)):
        raise ValueError("expected a gray, BGR, or BGRA image")
    if not np.isfinite(image).all():
        raise ValueError("image contains non-finite values")

    if np.issubdtype(image.dtype, np.floating):
        if float_range is None:
            raise ValueError("floating-point image requires an explicit float_range")
        low, high = float_range
        if not np.isfinite([low, high]).all() or high <= low:
            raise ValueError("float_range must be finite and increasing")
        scaled = np.clip((image.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255)
        pixels = np.rint(scaled).astype(np.uint8)
    elif np.issubdtype(image.dtype, np.integer):
        if image.dtype == np.uint8:
            pixels = image.copy()
        else:
            limits = np.iinfo(image.dtype)
            scaled = (image.astype(np.float64) - limits.min) * (
                255.0 / (limits.max - limits.min)
            )
            pixels = np.rint(np.clip(scaled, 0, 255)).astype(np.uint8)
    else:
        raise ValueError(f"unsupported image dtype: {image.dtype}")

    if pixels.ndim == 2:
        bgr = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    elif pixels.shape[2] == 1:
        bgr = cv2.cvtColor(pixels[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif pixels.shape[2] == 4:
        bgr = cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
    else:
        bgr = pixels.copy()
    return np.ascontiguousarray(bgr)


def load_image(
    path: str | Path,
    max_dimension: int = 12000,
    float_range: tuple[float, float] | None = None,
) -> ImageData:
    """Decode an image without modifying the source file or stretching contrast."""
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"image does not exist: {image_path}")
    try:
        encoded = np.fromfile(image_path, dtype=np.uint8)
    except OSError as exc:
        raise OSError(f"could not read image file: {image_path}: {exc}") from exc
    raw = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if raw is None or raw.size == 0:
        raise ValueError(f"image could not be decoded: {image_path}")
    bgr = normalize_image(raw, float_range=float_range)
    height, width = bgr.shape[:2]
    if width > max_dimension or height > max_dimension:
        raise ValueError(
            f"image dimensions {width}x{height} exceed configured maximum {max_dimension}"
        )
    channels = raw.shape[2] if raw.ndim == 3 else 1
    source_kind = "grayscale" if channels == 1 else "pseudo_color_or_color_display"
    return ImageData(image=bgr, width=width, height=height, channels=channels, source_kind=source_kind)


def decode_image_bytes(
    data: bytes,
    max_dimension: int = 12000,
    float_range: tuple[float, float] | None = None,
) -> ImageData:
    """Decode uploaded image bytes using the same checks as file-based input."""
    if not data:
        raise ValueError("uploaded image is empty")
    raw = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.size == 0:
        raise ValueError("uploaded image could not be decoded")
    bgr = normalize_image(raw, float_range=float_range)
    height, width = bgr.shape[:2]
    if width > max_dimension or height > max_dimension:
        raise ValueError(f"image dimensions {width}x{height} exceed configured maximum {max_dimension}")
    channels = raw.shape[2] if raw.ndim == 3 else 1
    source_kind = "grayscale" if channels == 1 else "pseudo_color_or_color_display"
    return ImageData(bgr, width, height, channels, source_kind)


def collect_image_paths(path: str | Path) -> list[Path]:
    """Return sorted supported image files from a single file or a directory tree."""
    target = Path(path).expanduser().resolve()
    if target.is_file():
        if target.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image extension: {target.suffix}")
        return [target]
    if not target.is_dir():
        raise NotADirectoryError(f"image folder does not exist: {target}")
    return sorted(
        (
            item.resolve()
            for item in target.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda item: str(item).casefold(),
    )


def load_image_for(path: str | Path, config: Mapping[str, Any]) -> ImageData:
    """Load an image using the ``image`` section of a project configuration."""
    settings = config.get("image", {})
    return load_image(
        path,
        max_dimension=int(settings.get("max_dimension", 12000)),
        float_range=settings.get("float_range"),
    )
