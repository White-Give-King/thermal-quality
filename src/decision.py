"""Conservative image-level focus-quality decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass

LABELS = ("sharp", "borderline", "defocused")
CLASSES = frozenset(LABELS)
STATUS_LABELS_ZH = {
    "sharp": "清晰，对焦正常",
    "borderline": "临界状态",
    "defocused": "模糊或失焦",
    "uncertain": "无法确定",
}


@dataclass(frozen=True)
class Decision:
    status: str
    reasons: tuple[str, ...]


def decide_focus_quality(
    prediction: dict[str, object] | None,
    *,
    confidence_threshold: float = 0.45,
) -> Decision:
    """Interpret a three-class focus prediction using a confidence threshold."""
    if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if prediction is None:
        return Decision("uncertain", ("classifier_weights_missing",))
    predicted = prediction.get("predicted_class")
    confidence = prediction.get("confidence")
    probabilities = prediction.get("probabilities")
    if (
        not isinstance(predicted, str)
        or predicted not in CLASSES
        or not isinstance(probabilities, dict)
        or set(probabilities) != CLASSES
    ):
        return Decision("uncertain", ("classifier_output_invalid",))
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        return Decision("uncertain", ("classifier_confidence_invalid",))
    values = list(probabilities.values())
    if (
        any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in values
        )
        or not math.isclose(sum(values), 1.0, abs_tol=1e-4)
        or not math.isclose(float(probabilities[predicted]), float(confidence), abs_tol=1e-4)
    ):
        return Decision("uncertain", ("classifier_probabilities_invalid",))
    if float(confidence) < confidence_threshold:
        return Decision("uncertain", ("classifier_low_confidence",))
    return Decision(
        str(predicted),
        (f"classifier_predicted_{predicted}",),
    )
