import pytest

from scripts.calibrate_thresholds import calibrate_predictions
from scripts.evaluate import summarize


def test_evaluation_counts_uncertain_prediction_as_focus_miss_and_review() -> None:
    report = summarize(
        ["defocused", "sharp", "borderline"],
        ["uncertain", "defocused", "borderline"],
    )
    assert report["confusion_matrix_labels"] == [
        "sharp",
        "borderline",
        "defocused",
        "uncertain",
    ]
    assert report["confusion_matrix"] == [
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ]
    assert report["defocused_miss_rate"] == 1.0
    assert report["sharp_false_reject_rate"] == 1.0
    assert report["review_rate"] == pytest.approx(2 / 3)


def test_threshold_calibration_reports_validation_error_and_review_metrics() -> None:
    result = calibrate_predictions(
        ["sharp", "borderline", "defocused"],
        [
            {
                "predicted_class": "sharp",
                "confidence": 0.9,
                "probabilities": {"sharp": 0.9, "borderline": 0.05, "defocused": 0.05},
            },
            {
                "predicted_class": "borderline",
                "confidence": 0.55,
                "probabilities": {"sharp": 0.2, "borderline": 0.55, "defocused": 0.25},
            },
            {
                "predicted_class": "defocused",
                "confidence": 0.8,
                "probabilities": {"sharp": 0.1, "borderline": 0.1, "defocused": 0.8},
            },
        ],
        baseline_threshold=0.6,
    )

    assert result["recommended"]["metrics"]["macro_f1"] == 1.0
    assert result["recommended"]["metrics"]["defocused_miss_rate"] == 0.0
    assert result["recommended"]["metrics"]["sharp_false_reject_rate"] == 0.0
    assert result["baseline"]["metrics"]["review_rate"] == pytest.approx(1 / 3)
