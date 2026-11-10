import numpy as np
import torch

from image_io import ImageData
from inference import InferenceEngine


def _config(width: int = 64, height: int = 64) -> dict[str, object]:
    return {
        "image": {"expected_width": width, "expected_height": height},
        "thermal_roi": {
            "mode": "fixed",
            "x": 8,
            "y": 4,
            "width": width - 16,
            "height": height - 8,
        },
        "classifier": {"confidence_threshold": 0.6},
    }


def _image_data(image: np.ndarray) -> ImageData:
    height, width = image.shape[:2]
    return ImageData(image.copy(), width, height, image.shape[2] if image.ndim == 3 else 1)


class SharpPredictor(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.tensor((5.0, 0.0, 0.0), device=images.device).repeat(images.shape[0], 1)


def test_missing_classifier_is_uncertain_and_never_fabricates_focus_quality() -> None:
    image = np.full((64, 64, 3), 90, dtype=np.uint8)
    engine = InferenceEngine(_config())
    result, panel, box = engine.predict_array(
        _image_data(image), image_name="frame.png"
    )
    assert result["status"] == "uncertain"
    assert result["reasons"] == ["classifier_weights_missing"]
    assert result["focus_quality"] is None
    assert result["model_score"] is None
    assert result["error"] == "trained focus-quality classifier weights are required"
    assert box == (8, 4, 48, 56)
    assert np.array_equal(panel, image[4:60, 8:56])


def test_classifier_decides_focus_quality_on_entire_cropped_thermal_panel() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    engine = InferenceEngine(
        _config(),
        classifier=SharpPredictor(),
        classifier_metadata={
            "input_size": 224,
            "model_version": "test-model",
            "architecture": "mobilenet_v3_small",
        },
    )
    result, panel, _ = engine.predict_array(_image_data(image), image_name="frame.png")
    assert result["status"] == "sharp"
    assert result["focus_quality"] == "sharp"
    assert result["label_zh"] == "清晰，对焦正常"
    assert result["architecture"] == "mobilenet_v3_small"
    assert result["probabilities"]["sharp"] > 0.9
    assert result["model_score"] == round(result["confidence"] * 100, 1)
    assert panel.shape == (56, 48, 3)


def test_inference_detects_black_border_before_using_configured_roi() -> None:
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    image[3:57, 5:75] = (20, 50, 80)
    engine = InferenceEngine(_config(width=80, height=60))

    result, panel, box = engine.predict_array(_image_data(image), image_name="frame.png")

    assert box == (5, 3, 70, 54)
    assert result["thermal_roi"]["xywh"] == [5, 3, 70, 54]
    assert np.array_equal(panel, image[3:57, 5:75])


def test_unexpected_canvas_size_fails_before_classification() -> None:
    engine = InferenceEngine(_config(width=80))
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    try:
        engine.predict_array(_image_data(image), image_name="wrong-size.png")
    except ValueError as exc:
        assert "expected_width" in str(exc)
    else:
        raise AssertionError("unexpected input canvas dimensions were accepted")


def test_predict_batch_keeps_order_and_isolates_bad_images() -> None:
    class OrderedPredictor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes: list[int] = []

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            self.batch_sizes.append(images.shape[0])
            return torch.tensor(
                ((5.0, 0.0, 0.0), (0.0, 5.0, 0.0)), device=images.device
            )[: images.shape[0]]

    predictor = OrderedPredictor()
    engine = InferenceEngine(_config(), classifier=predictor)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    invalid_image = np.zeros((32, 32, 3), dtype=np.uint8)

    outcomes = engine.predict_batch(
        [
            (_image_data(image), "first.png"),
            (_image_data(invalid_image), "invalid.png"),
            (_image_data(image), "third.png"),
        ]
    )

    assert predictor.batch_sizes == [2]
    first = outcomes[0]
    assert not isinstance(first, Exception)
    assert first[0]["image"] == "first.png"
    assert first[0]["status"] == "sharp"
    assert isinstance(outcomes[1], ValueError)
    assert "expected_width" in str(outcomes[1])
    third = outcomes[2]
    assert not isinstance(third, Exception)
    assert third[0]["image"] == "third.png"
    assert third[0]["status"] == "borderline"


def test_predict_batch_caps_model_batches_and_keeps_all_results() -> None:
    class BatchSizePredictor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes: list[int] = []

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            self.batch_sizes.append(images.shape[0])
            return torch.tensor((5.0, 0.0, 0.0), device=images.device).repeat(
                images.shape[0], 1
            )

    predictor = BatchSizePredictor()
    engine = InferenceEngine(_config(), classifier=predictor)
    image = _image_data(np.zeros((64, 64, 3), dtype=np.uint8))
    inputs = [(image, f"{index}.png") for index in range(17)]

    outcomes = engine.predict_batch(inputs)

    assert predictor.batch_sizes == [16, 1]
    assert len(outcomes) == len(inputs)
    predicted_names = []
    for outcome in outcomes:
        assert not isinstance(outcome, Exception)
        predicted_names.append(outcome[0]["image"])
    assert predicted_names == [name for _, name in inputs]
