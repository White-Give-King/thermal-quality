"""Shared metrics for focus-quality evaluation and threshold calibration."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from thermal_focus.classifier import CLASSES


def summarize(y_true: list[str], y_pred: list[str]) -> dict[str, object]:
    defocused_true = np.asarray([label == "defocused" for label in y_true], dtype=bool)
    defocused_pred = np.asarray([label == "defocused" for label in y_pred], dtype=bool)
    sharp_true = np.asarray([label == "sharp" for label in y_true], dtype=bool)
    sharp_pred = np.asarray([label == "sharp" for label in y_pred], dtype=bool)
    borderline_pred = np.asarray([label == "borderline" for label in y_pred], dtype=bool)
    uncertain = np.asarray([label == "uncertain" for label in y_pred], dtype=bool)
    report_labels = [*CLASSES, "uncertain"]
    report = classification_report(
        y_true,
        y_pred,
        labels=report_labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "confusion_matrix_labels": report_labels,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=report_labels
        ).tolist(),
        "per_class": {label: report[label] for label in CLASSES},
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=list(CLASSES), average="macro", zero_division=0)
        ),
        "defocused_miss_rate": float(
            np.sum(defocused_true & ~defocused_pred) / max(np.sum(defocused_true), 1)
        ),
        "sharp_false_reject_rate": float(
            np.sum(sharp_true & ~sharp_pred) / max(np.sum(sharp_true), 1)
        ),
        "review_rate": float(np.mean(borderline_pred | uncertain)),
    }
