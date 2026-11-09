from pathlib import Path

import cv2
import numpy as np
import pytest

from image_io import load_image
from thermal_roi import (
    detect_black_border_roi,
    draw_thermal_roi,
    extract_thermal_roi,
    validate_thermal_roi,
)


def _config(
    *,
    x: int = 10,
    y: int = 5,
    width: int = 20,
    height: int = 12,
    expected_width: int = 40,
    expected_height: int = 30,
) -> dict[str, object]:
    return {
        "image": {"expected_width": expected_width, "expected_height": expected_height},
        "thermal_roi": {
            "mode": "fixed",
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        },
    }


def test_extract_thermal_roi_uses_half_open_xywh_and_copies_source() -> None:
    image = np.arange(30 * 40 * 3, dtype=np.uint8).reshape(30, 40, 3)
    original = image.copy()
    crop, box = extract_thermal_roi(image, _config())

    assert box == (10, 5, 20, 12)
    assert crop.shape == (12, 20, 3)
    assert np.array_equal(crop, image[5:17, 10:30])
    crop[0, 0] = 0
    assert np.array_equal(image, original)


@pytest.mark.parametrize(
    ("config", "shape", "message"),
    [
        (_config(x=30, width=20), (30, 40, 3), "exceeds input dimensions"),
        (_config(y=20, height=12), (30, 40, 3), "exceeds input dimensions"),
        (_config(), (31, 40, 3), "expected_height"),
        (_config(), (30, 41, 3), "expected_width"),
        (_config(x=-1), (30, 40, 3), "non-negative integer"),
        (_config(width=0), (30, 40, 3), "positive integer"),
        ({"thermal_roi": {"mode": "auto"}}, (30, 40, 3), "supported modes are fixed and centered"),
    ],
)
def test_invalid_size_or_roi_fails_with_clear_error(
    config: dict[str, object], shape: tuple[int, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_thermal_roi(shape, config)


def test_centered_mode_centers_crop_and_ignores_xy() -> None:
    image = np.arange(30 * 40 * 3, dtype=np.uint8).reshape(30, 40, 3)
    config = {
        "thermal_roi": {
            "mode": "centered",
            "x": 0,
            "y": 0,
            "width": 20,
            "height": 12,
        }
    }
    crop, box = extract_thermal_roi(image, config)

    assert box == (10, 9, 20, 12)
    assert crop.shape == (12, 20, 3)
    assert np.array_equal(crop, image[9:21, 10:30])


@pytest.mark.parametrize("shape", [(30, 41, 3), (31, 40, 3)])
def test_centered_mode_rejects_canvas_smaller_than_crop(shape: tuple[int, ...]) -> None:
    config = {"thermal_roi": {"mode": "centered", "width": 50, "height": 12}}
    with pytest.raises(ValueError, match="does not fit"):
        validate_thermal_roi(shape, config)


def test_detect_black_border_roi_returns_non_black_content_bounds() -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[5:35, 8:52] = (20, 50, 80)

    assert detect_black_border_roi(image) == (8, 5, 44, 30)


def test_detect_black_border_roi_ignores_disconnected_content_in_landscape_image() -> None:
    image = np.zeros((240, 640, 3), dtype=np.uint8)
    image[10:230, 110:410] = (20, 80, 150)
    image[15:25, 480:620] = 255

    assert detect_black_border_roi(image) == (110, 10, 300, 220)


def test_detect_black_border_roi_ignores_small_bright_marks() -> None:
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    image[2:5, 2:6] = 255

    assert detect_black_border_roi(image) is None


@pytest.mark.parametrize("size", [30, 40])
def test_detect_black_border_roi_rejects_small_isolated_region(size: int) -> None:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[20 : 20 + size, 25 : 25 + size] = 255

    assert detect_black_border_roi(image) is None


@pytest.mark.parametrize(
    "image",
    [
        np.full((40, 60, 3), 80, dtype=np.uint8),
        np.zeros((40, 60, 3), dtype=np.uint8),
    ],
)
def test_detect_black_border_roi_returns_none_without_clear_frame(
    image: np.ndarray,
) -> None:
    assert detect_black_border_roi(image) is None


def test_empty_and_invalid_images_are_rejected() -> None:
    config = _config()
    with pytest.raises(ValueError, match="non-zero"):
        validate_thermal_roi((0, 40, 3), config)
    with pytest.raises(ValueError, match="empty"):
        extract_thermal_roi(np.array([], dtype=np.uint8), config)
    with pytest.raises(ValueError, match="finite"):
        extract_thermal_roi(np.full((30, 40, 3), np.nan), config)


def test_thermal_border_visualization() -> None:
    image = np.zeros((30, 40, 3), dtype=np.uint8)
    original = image.copy()
    drawn = draw_thermal_roi(image, (10, 5, 20, 12), thickness=1)
    assert not np.array_equal(drawn, image)
    assert np.array_equal(image, original)
    assert np.array_equal(drawn[5, 10], (0, 255, 0))
    assert np.array_equal(drawn[4, 10], (0, 0, 0))

def test_bundled_real_sample_crop_excludes_frame_scale_and_timestamp(tmp_path: Path) -> None:
    sample_dir = Path(__file__).parents[1] / "数据集" / "A7-10-19"
    sample_path = next(sample_dir.glob("*.jpg"), None)
    if sample_path is None:
        pytest.skip("bundled real-image examples are not available")

    config = {
        "image": {"expected_width": 1993, "expected_height": 1080},
        "thermal_roi": {
            "mode": "fixed",
            "x": 592,
            "y": 0,
            "width": 809,
            "height": 1080,
        },
    }
    image = load_image(sample_path).image
    thermal, box = extract_thermal_roi(image, config)

    assert box == (592, 0, 809, 1080)
    assert thermal.shape == (1080, 809, 3)
    assert np.array_equal(thermal, image[:, 592:1401])
    assert np.array_equal(thermal[540, 0], image[540, 592])
    assert np.array_equal(thermal[540, -1], image[540, 1400])
    assert np.all(image[540, 0:580] < 20)
    assert np.all(image[540, 1410:1900] < 20)

    crop_path = tmp_path / "real_thermal_panel.png"
    overlay_path = tmp_path / "real_thermal_panel_overlay.jpg"
    success, encoded = cv2.imencode(".png", thermal)
    assert success
    encoded.tofile(crop_path)
    overlay = draw_thermal_roi(image, box)
    success, encoded = cv2.imencode(".jpg", overlay)
    assert success
    encoded.tofile(overlay_path)
    assert crop_path.is_file() and overlay_path.is_file()
