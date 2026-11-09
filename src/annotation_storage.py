"""Persistent storage for focus-quality annotation labels and cropped panels."""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np

from decision import LABELS

ROI_DATABASE_FILENAME = "annotations.sqlite3"


def _create_roi_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS image_rois (
            image_path TEXT PRIMARY KEY,
            x INTEGER NOT NULL CHECK (x >= 0),
            y INTEGER NOT NULL CHECK (y >= 0),
            width INTEGER NOT NULL CHECK (width > 0),
            height INTEGER NOT NULL CHECK (height > 0)
        )
        """
    )


def _create_annotation_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS image_annotations (
            image_path TEXT PRIMARY KEY,
            label TEXT NOT NULL CHECK (label IN ('sharp', 'borderline', 'defocused')),
            source_path TEXT NOT NULL DEFAULT '',
            annotated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def load_image_rois(
    database_path: str | Path,
) -> dict[str, tuple[int, int, int, int]]:
    """Load per-image crop boxes from a workspace database, creating it if needed."""
    target = Path(database_path)
    try:
        with closing(sqlite3.connect(target)) as connection, connection:
            _create_roi_table(connection)
            rows = connection.execute(
                "SELECT image_path, x, y, width, height FROM image_rois"
            ).fetchall()
    except sqlite3.Error as exc:
        raise OSError(f"could not read ROI database {target}: {exc}") from exc
    return {
        str(image_path): (int(x), int(y), int(width), int(height))
        for image_path, x, y, width, height in rows
    }


def save_image_roi(
    database_path: str | Path,
    image_path: str,
    roi: tuple[int, int, int, int],
) -> None:
    """Persist one image's XYWH crop box in its workspace database."""
    if not image_path:
        raise ValueError("image path key cannot be empty")
    if (
        len(roi) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in roi)
        or roi[0] < 0
        or roi[1] < 0
        or roi[2] <= 0
        or roi[3] <= 0
    ):
        raise ValueError("ROI must contain non-negative offsets and positive dimensions")
    target = Path(database_path)
    try:
        with closing(sqlite3.connect(target)) as connection, connection:
            _create_roi_table(connection)
            connection.execute(
                """
                INSERT INTO image_rois (image_path, x, y, width, height)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(image_path) DO UPDATE SET
                    x = excluded.x,
                    y = excluded.y,
                    width = excluded.width,
                    height = excluded.height
                """,
                (image_path, *roi),
            )
    except sqlite3.Error as exc:
        raise OSError(f"could not write ROI database {target}: {exc}") from exc


def save_image_annotation(
    database_path: str | Path, annotation: dict[str, str]
) -> None:
    """Persist one label and its source/crop paths in the workspace database."""
    image_path = annotation.get("image_path", "")
    label = annotation.get("label", "")
    source_path = annotation.get("source_path", "")
    relative_path = Path(image_path)
    if not image_path or relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("annotation image_path must stay inside the output directory")
    if label not in LABELS:
        raise ValueError(f"unsupported focus-quality label: {label}")
    if not source_path:
        raise ValueError("annotation source_path cannot be empty")

    target = Path(database_path)
    try:
        with closing(sqlite3.connect(target)) as connection, connection:
            _create_annotation_table(connection)
            connection.execute(
                """
                INSERT INTO image_annotations (image_path, label, source_path)
                VALUES (?, ?, ?)
                ON CONFLICT(image_path) DO UPDATE SET
                    label = excluded.label,
                    source_path = excluded.source_path,
                    annotated_at = CURRENT_TIMESTAMP
                """,
                (relative_path.as_posix(), label, str(Path(source_path).resolve())),
            )
    except sqlite3.Error as exc:
        raise OSError(f"could not write annotation database {target}: {exc}") from exc


def load_image_annotations(database_path: str | Path) -> list[dict[str, str]]:
    """Load label records from a workspace database."""
    target = Path(database_path)
    try:
        with closing(sqlite3.connect(target)) as connection, connection:
            _create_annotation_table(connection)
            rows = connection.execute(
                "SELECT image_path, label, source_path FROM image_annotations"
            ).fetchall()
    except sqlite3.Error as exc:
        raise OSError(f"could not read annotation database {target}: {exc}") from exc
    return [
        {"image_path": str(image_path), "label": str(label), "source_path": str(source_path)}
        for image_path, label, source_path in rows
    ]


def save_roi_coordinates(config_path: str | Path, roi: tuple[int, int, int, int]) -> None:
    """Write explicit ROI values into YAML and switch the mode to ``fixed``."""
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
    pattern = re.compile(r"^(\s+)(x|y|width|height|mode)(\s*:\s*)([^#]*?)(\s*#.*)?$")
    indent = "  "
    present: set[str] = set()
    for index in range(section_start + 1, section_end):
        line_ending = (
            "\r\n" if lines[index].endswith("\r\n")
            else "\n" if lines[index].endswith("\n")
            else ""
        )
        content = lines[index][:-len(line_ending)] if line_ending else lines[index]
        match = pattern.match(content)
        if not match:
            continue
        key = match.group(2)
        indent = match.group(1)
        present.add(key)
        comment = match.group(5) or ""
        comment_suffix = f" {comment.lstrip()}" if comment else ""
        value = "fixed" if key == "mode" else str(keys[key])
        lines[index] = (
            f"{match.group(1)}{key}{match.group(3)}{value}{comment_suffix}{line_ending}"
        )
    missing = [name for name in ("mode", "x", "y", "width", "height") if name not in present]
    if missing:
        if section_end > 0 and not lines[section_end - 1].endswith(("\n", "\r")):
            lines[section_end - 1] += "\n"
        inserts = [
            f"{indent}{name}: {'fixed' if name == 'mode' else keys[name]}\n"
            for name in missing
        ]
        lines[section_end:section_end] = inserts

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
    database_path: str | Path,
    source_path: str | Path,
    label: str,
    panel: np.ndarray,
) -> dict[str, str]:
    """Save a cropped panel and persist its annotation in the workspace database."""
    if label not in LABELS:
        raise ValueError(f"unsupported focus-quality label: {label}")
    source = str(Path(source_path).resolve())
    target_dir = Path(output_dir).resolve()
    token = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    relative_path = Path("images") / f"{Path(source).stem}_{token}.png"
    row = {
        "image_path": relative_path.as_posix(),
        "label": label,
        "source_path": source,
    }
    write_png(target_dir / relative_path, panel)
    save_image_annotation(database_path, row)
    return row.copy()
