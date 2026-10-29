from pathlib import Path

import cv2
import numpy as np
import pytest

from thermal_focus.image_io import load_image, normalize_image


def test_normalize_gray_bgr_and_bgra_without_mutating_source() -> None:
    gray = np.arange(100, dtype=np.uint8).reshape(10, 10)
    source = gray.copy()
    bgr = normalize_image(gray)
    bgra = np.dstack((bgr, np.full((10, 10), 255, dtype=np.uint8)))

    assert bgr.shape == (10, 10, 3)
    assert np.array_equal(bgr[:, :, 0], gray)
    assert normalize_image(bgra).shape == (10, 10, 3)
    assert np.array_equal(gray, source)


def test_float_image_requires_fixed_range() -> None:
    values = np.array([[0.0, 10.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="explicit float_range"):
        normalize_image(values)
    result = normalize_image(values, float_range=(0.0, 20.0))
    assert result[0, 1, 0] == 128


def test_rejects_nonfinite_empty_and_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="empty"):
        normalize_image(np.array([], dtype=np.uint8))
    with pytest.raises(ValueError, match="non-finite"):
        normalize_image(np.array([[np.nan]], dtype=np.float32), (0.0, 1.0))
    with pytest.raises(ValueError, match="expected"):
        normalize_image(np.zeros((2, 2, 2), dtype=np.uint8))


def test_load_image_checks_decode_and_dimensions(tmp_path: Path) -> None:
    image_path = tmp_path / "gray.png"
    assert cv2.imwrite(str(image_path), np.full((12, 15), 50, dtype=np.uint8))
    loaded = load_image(image_path)
    assert loaded.image.shape == (12, 15, 3)
    assert (loaded.width, loaded.height, loaded.channels) == (15, 12, 1)
    with pytest.raises(ValueError, match="maximum"):
        load_image(image_path, max_dimension=10)
    corrupt = tmp_path / "bad.png"
    corrupt.write_bytes(b"broken")
    with pytest.raises(ValueError, match="decoded"):
        load_image(corrupt)


def test_load_image_supports_unicode_windows_paths(tmp_path: Path) -> None:
    image_path = tmp_path / "红外图像.png"
    ok, encoded = cv2.imencode(".png", np.full((4, 6), 20, dtype=np.uint8))
    assert ok
    encoded.tofile(image_path)
    loaded = load_image(image_path)
    assert loaded.source_kind == "grayscale"
    assert loaded.image.shape == (4, 6, 3)
