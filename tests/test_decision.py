import pytest

from thermal_focus.decision import decide_focus_quality


def _prediction(label: str, confidence: float) -> dict[str, object]:
    remaining = [name for name in ("sharp", "borderline", "defocused") if name != label]
    probabilities = {name: (1 - confidence) / 2 for name in remaining}
    probabilities[label] = confidence
    return {
        "predicted_class": label,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def test_classifier_decides_focus_quality() -> None:
    sharp = decide_focus_quality(_prediction("sharp", 0.9))
    borderline = decide_focus_quality(_prediction("borderline", 0.9))
    defocused = decide_focus_quality(_prediction("defocused", 0.9))
    assert sharp.status == "sharp"
    assert borderline.status == "borderline"
    assert defocused.status == "defocused"


def test_missing_or_uncertain_prediction_requires_review() -> None:
    missing = decide_focus_quality(None)
    low_confidence = decide_focus_quality(_prediction("sharp", 0.55))
    assert missing.status == "uncertain"
    assert low_confidence.reasons == ("classifier_low_confidence",)


@pytest.mark.parametrize(
    "prediction",
    [
        {"predicted_class": "not_a_class", "confidence": 0.9, "probabilities": {}},
        _prediction("sharp", 0.9) | {"confidence": float("nan")},
        _prediction("sharp", 0.9)
        | {"probabilities": {"sharp": 1.0, "borderline": 0.0, "defocused": 0.0}},
    ],
)
def test_invalid_output_requires_review(prediction) -> None:
    assert decide_focus_quality(prediction).status == "uncertain"


def test_invalid_confidence_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        decide_focus_quality(None, confidence_threshold=1.1)
