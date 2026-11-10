import numpy as np
import pytest
import torch

from classifier import (
    ARCHITECTURES,
    CLASSES,
    MODEL_VERSION,
    build_classifier,
    classify_panel,
    classify_panels,
    load_classifier,
    preprocess_panel,
)
from image_io import normalize_image


def test_panel_preprocessing_checkpoint_restore_and_probabilities(tmp_path) -> None:
    gray = np.tile(np.arange(48, dtype=np.uint8), (32, 1))
    tensor = preprocess_panel(gray, input_size=64)
    assert tuple(tensor.shape) == (3, 64, 64)
    assert np.isfinite(tensor.numpy()).all()
    assert torch.equal(tensor[:, 0, :], tensor[:, 1, :])

    model = build_classifier(pretrained=False)
    checkpoint_path = tmp_path / "classifier.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": list(CLASSES),
            "model_version": MODEL_VERSION,
            "input_size": 64,
        },
        checkpoint_path,
    )
    restored, metadata = load_classifier(checkpoint_path, device="cpu")
    result = classify_panel(restored, normalize_image(gray), device="cpu", input_size=64)
    assert metadata["model_version"] == MODEL_VERSION
    assert metadata["architecture"] == "mobilenet_v3_small"
    assert result["predicted_class"] in CLASSES
    assert set(result["probabilities"]) == set(CLASSES)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert 0 <= result["confidence"] <= 1


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_supported_architectures_build_and_restore_checkpoints(tmp_path, architecture) -> None:
    model = build_classifier(architecture=architecture)
    image = np.full((48, 64, 3), 80, dtype=np.uint8)
    logits = model(preprocess_panel(image, input_size=64).unsqueeze(0))
    assert tuple(logits.shape) == (1, len(CLASSES))

    checkpoint = tmp_path / f"{architecture}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": list(CLASSES),
            "architecture": architecture,
            "model_version": f"{architecture}-thermal-focus-quality-1",
            "input_size": 64,
        },
        checkpoint,
    )
    restored, metadata = load_classifier(checkpoint)
    prediction = classify_panel(
        restored,
        image,
        device="cpu",
        input_size=metadata["input_size"],
    )
    assert metadata["architecture"] == architecture
    assert prediction["predicted_class"] in CLASSES


def test_legacy_checkpoint_defaults_to_mobilenet_v3_small(tmp_path) -> None:
    model = build_classifier()
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": list(CLASSES),
            "model_version": MODEL_VERSION,
            "input_size": 64,
        },
        checkpoint,
    )

    _, metadata = load_classifier(checkpoint)

    assert metadata["architecture"] == "mobilenet_v3_small"


def test_classify_panels_uses_one_forward_pass_and_keeps_per_image_results() -> None:
    class BatchPredictor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes: list[int] = []

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            self.batch_sizes.append(images.shape[0])
            return torch.tensor(
                ((5.0, 0.0, 0.0), (0.0, 5.0, 0.0)), device=images.device
            )[: images.shape[0]]

    model = BatchPredictor()
    panels = [
        np.full((32, 48, 3), 40, dtype=np.uint8),
        np.full((48, 32, 3), 180, dtype=np.uint8),
    ]

    predictions = classify_panels(model, panels, device="cpu", input_size=64)

    assert model.batch_sizes == [2]
    assert [item["predicted_class"] for item in predictions] == [
        CLASSES[0],
        CLASSES[1],
    ]
    assert set(predictions[0]["probabilities"]) == set(CLASSES)


def test_build_classifier_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="unsupported architecture"):
        build_classifier(architecture="unknown")
