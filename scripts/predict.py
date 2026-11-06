"""Assess focus quality in pseudo-color thermal images from the command line."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from thermal_focus.image_io import collect_image_paths, load_image_for
from thermal_focus.inference import InferenceEngine, load_config
from thermal_focus.visualization import write_error_json, write_visualizations

LOGGER = logging.getLogger("thermal_focus.predict")


def _inputs(path: Path) -> list[Path]:
    return collect_image_paths(path)


def _output_location(image_path: Path, input_root: Path, output_root: Path) -> tuple[Path, str]:
    if input_root.is_dir():
        relative = image_path.relative_to(input_root)
        return output_root / relative.parent, relative.stem
    return output_root, image_path.stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assess focus quality in pseudo-color thermal images."
    )
    parser.add_argument("--input", required=True, type=Path, help="image file or directory")
    parser.add_argument("--output", required=True, type=Path, help="JSON and preview output directory")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "default.yaml",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="trained three-class focus-quality checkpoint; overrides classifier.weights in config",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    args.input = args.input.expanduser().resolve()
    try:
        config = load_config(args.config)
        checkpoint = args.model or config.get("classifier", {}).get("weights")
        if not checkpoint:
            raise ValueError("pass --model or configure classifier.weights with a trained checkpoint")
        import torch

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; choose --device cpu")
        device = (
            "cuda" if args.device == "auto" and torch.cuda.is_available()
            else "cpu" if args.device == "auto"
            else args.device
        )
        from thermal_focus.classifier import load_classifier

        classifier, metadata = load_classifier(checkpoint, device=device)
        engine = InferenceEngine(
            config,
            classifier=classifier,
            classifier_metadata=metadata,
            device=device,
        )
        paths = _inputs(args.input)
        if not paths:
            raise ValueError(f"no supported image files found in {args.input}")
        failures = 0
        for path in paths:
            output_dir, stem = _output_location(path, args.input, args.output)
            image_name = (
                path.relative_to(args.input).as_posix() if args.input.is_dir() else path.name
            )
            try:
                image_data = load_image_for(path, config)
                result, thermal_image, thermal_box = engine.predict_array(
                    image_data, image_name=image_name
                )
                visual_paths = write_visualizations(
                    output_dir,
                    stem,
                    image_data.image,
                    thermal_image,
                    thermal_box,
                    result,
                )
                LOGGER.info(
                    "%s focus_quality=%s confidence=%s json=%s",
                    path,
                    result["status"],
                    result["confidence"],
                    visual_paths["json"],
                )
                failures += int(result["error"] is not None)
            except (OSError, ValueError, RuntimeError) as exc:
                failures += 1
                error_json = write_error_json(output_dir, stem, image_name, str(exc))
                LOGGER.error("prediction failed for %s: %s (json=%s)", path, exc, error_json)
        return 1 if failures else 0
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("could not initialize prediction: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
