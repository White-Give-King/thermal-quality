import numpy as np
import pytest
import torch

from thermal_focus.classifier import (
    ARCHITECTURES,
    CLASSES,
    MODEL_VERSION,
    build_classifier,
    classify_panel,
    load_classifier,
    preprocess_panel,
)
from thermal_focus.image_io import normalize_image


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


def test_build_classifier_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="unsupported architecture"):
        build_classifier(architecture="unknown")
