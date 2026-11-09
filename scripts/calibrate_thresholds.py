"""Select a confidence threshold from a labeled validation database."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from classifier import classify_panel, load_classifier
from dataset import load_annotations
from decision import decide_focus_quality
from evaluation import summarize
from image_io import load_image
from inference import load_config

LOGGER = logging.getLogger("thermal_focus_quality.calibrate_thresholds")


def calibrate_predictions(
    labels: list[str],
    predictions: list[dict[str, Any]],
    *,
    baseline_threshold: float,
) -> dict[str, Any]:
    """Choose the validation threshold with best Macro-F1; ties prefer fewer reviews."""
    if not labels or len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same non-zero length")
    candidates = sorted(
        {0.0, 1.0, baseline_threshold, *(float(item["confidence"]) for item in predictions)}
    )
    results: list[dict[str, Any]] = []
    for threshold in candidates:
        predicted_labels = [
            decide_focus_quality(item, confidence_threshold=threshold).status
            for item in predictions
        ]
        metrics = summarize(labels, predicted_labels)
        results.append(
            {
                "confidence_threshold": threshold,
                "metrics": metrics,
            }
        )
    selected = max(
        results,
        key=lambda item: (
            item["metrics"]["macro_f1"],
            -item["metrics"]["review_rate"],
            -item["confidence_threshold"],
        ),
    )
    return {
        "selection_metric": "macro_f1",
        "tie_break": "lower review rate, then lower confidence threshold",
        "recommended": selected,
        "baseline": next(
            result for result in results
            if result["confidence_threshold"] == baseline_threshold
        ),
        "candidates": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path, help="SQLite annotation database")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--model", required=True, type=Path, help="trained focus-quality checkpoint")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[1] / "configs" / "default.yaml",
    )
    parser.add_argument("--output", type=Path, default=Path("threshold-calibration.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = load_config(args.config)
    frame = load_annotations(args.database, image_root=args.image_root)
    model, metadata = load_classifier(args.model, device=args.device)
    predictions = [
        classify_panel(
            model,
            load_image(image_path).image,
            device=args.device,
            input_size=metadata["input_size"],
        )
        for image_path in frame["image_path"]
    ]
    baseline_threshold = float(
        config.get("classifier", {}).get("confidence_threshold", 0.60)
    )
    report = {
        "dataset": str(args.database.resolve()),
        "sample_count": len(frame),
        "task": "thermal_focus_quality",
        "model_version": metadata["model_version"],
        "selection_note": (
            "Threshold selected on the supplied validation set by Macro-F1; "
            "do not use this report as an independent test result."
        ),
        **calibrate_predictions(
            frame["label"].tolist(),
            predictions,
            baseline_threshold=baseline_threshold,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info(
        "recommended confidence threshold=%.4f; wrote %s",
        report["recommended"]["confidence_threshold"],
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
