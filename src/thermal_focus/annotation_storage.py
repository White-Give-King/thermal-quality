"""Persistent storage for focus-quality annotation labels and cropped panels."""

from __future__ import annotations

import csv
import hashlib
import os
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .decision import LABELS

CSV_FIELDS = ["image_path", "label", "source_path"]


def save_roi_coordinates(config_path: str | Path, roi: tuple[int, int, int, int]) -> None:
    """Update fixed ROI values in YAML while preserving other settings and comments."""
    target = Path(config_path)
    text = target.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    section_start = next(
        (index for index, line in enumerate(lines) if re.match(r"^thermal_roi\s*:", line)),
        None,
    )
    if section_start is None:
        raise ValueError(f"configuration has no thermal_roi section: {target}")
    section_end = next(
        (
            index
            for index in range(section_start + 1, len(lines))
            if re.match(r"^[^\s#][^:]*\s*:", lines[index])
        ),
        len(lines),
    )
    keys = dict(zip(("x", "y", "width", "height"), roi, strict=True))
    updated: set[str] = set()
    for index in range(section_start + 1, section_end):
        line_ending = (
            "\r\n" if lines[index].endswith("\r\n")
            else "\n" if lines[index].endswith("\n")
            else ""
        )
        content = lines[index][:-len(line_ending)] if line_ending else lines[index]
        match = re.match(r"^(\s+)(x|y|width|height)(\s*:\s*)([^#]*?)(\s*#.*)?$", content)
        if match and match.group(2) in keys:
            key = match.group(2)
            comment = match.group(5) or ""
            comment_suffix = f" {comment.lstrip()}" if comment else ""
            lines[index] = (
                f"{match.group(1)}{key}{match.group(3)}{keys[key]}"
                f"{comment_suffix}{line_ending}"
            )
            updated.add(key)
    if updated != set(keys):
        raise ValueError(
            "configuration thermal_roi section must define x, y, width, height; "
            f"missing {sorted(set(keys) - updated)}"
        )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=target.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.writelines(lines)
        os.replace(temporary_path, target)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def write_png(path: str | Path, image: np.ndarray) -> None:
    """Write a PNG image, including on Windows paths containing non-ASCII characters."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise OSError(f"could not encode thermal panel: {target}")
    encoded.tofile(target)


def save_panel_label(
    output_dir: str | Path,
    source_path: str | Path,
    label: str,
    panel: np.ndarray,
) -> dict[str, str]:
    """Save a cropped panel and atomically insert or update its CSV annotation."""
    if label not in LABELS:
        raise ValueError(f"unsupported focus-quality label: {label}")
    source = str(Path(source_path).resolve())
    target_dir = Path(output_dir).resolve()
    csv_path = target_dir / "labels.csv"
    rows: list[dict[str, str]] = []
    if csv_path.is_file():
        with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))

    existing = next(
        (row for row in rows if row.get("source_path", "").casefold() == source.casefold()),
        None,
    )
    if existing:
        relative_path = Path(existing["image_path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("existing label image_path must remain inside the output directory")
        row = existing
    else:
        token = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
        relative_path = Path("images") / f"{Path(source).stem}_{token}.png"
        row = {
            "image_path": relative_path.as_posix(),
            "label": label,
            "source_path": source,
        }
        rows.append(row)
    row["image_path"] = relative_path.as_posix()
    row["label"] = label
    row["source_path"] = source
    write_png(target_dir / relative_path, panel)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", newline="", encoding="utf-8-sig", dir=csv_path.parent, delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, csv_path)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return row.copy()
