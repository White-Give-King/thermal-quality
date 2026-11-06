from pathlib import Path
import sqlite3

import cv2
import numpy as np
import pandas as pd
import pytest

from thermal_focus.dataset import load_annotations, split_dataset


def _write_dataset(root: Path) -> Path:
    records = []
    for group_index in range(18):
        for label in ("sharp", "borderline", "defocused"):
            name = f"{group_index}-{label}.png"
            assert cv2.imwrite(str(root / name), np.full((8, 8), group_index, np.uint8))
            records.append((name, label))
    database_path = root / "annotations.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE image_annotations (image_path TEXT, label TEXT)"
        )
        connection.executemany(
            "INSERT INTO image_annotations (image_path, label) VALUES (?, ?)",
            records,
        )
    return database_path


def test_stratified_split_is_deterministic_and_has_no_image_leakage(tmp_path: Path) -> None:
    database_path = _write_dataset(tmp_path)
    frame = load_annotations(database_path)
    first = split_dataset(frame, seed=7)
    second = split_dataset(frame, seed=7)
    assert first.train.image_path.tolist() == second.train.image_path.tolist()
    train_images = set(first.train.image_path)
    validation_images = set(first.validation.image_path)
    test_images = set(first.test.image_path)
    assert not train_images & validation_images
    assert not train_images & test_images
    assert not validation_images & test_images
    assert set(first.train.label) == {"sharp", "borderline", "defocused"}


def test_annotation_database_loads_only_image_and_label_columns(tmp_path: Path) -> None:
    database_path = _write_dataset(tmp_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "ALTER TABLE image_annotations ADD COLUMN source_path TEXT"
        )
        connection.execute(
            "UPDATE image_annotations SET source_path = 'unused'"
        )
    frame = load_annotations(database_path)
    assert list(frame.columns) == ["image_path", "label"]
    splits = split_dataset(frame, seed=7)
    partitions = [set(split.image_path) for split in (
        splits.train, splits.validation, splits.test
    )]
    assert not partitions[0] & partitions[1]
    assert not partitions[0] & partitions[2]
    assert not partitions[1] & partitions[2]
    for split in (splits.train, splits.validation, splits.test):
        assert set(split.label) == {"sharp", "borderline", "defocused"}


def test_invalid_label_duplicate_and_too_few_groups_rejected(tmp_path: Path) -> None:
    database_path = _write_dataset(tmp_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE image_annotations SET label = 'valid_target' "
            "WHERE rowid = (SELECT MIN(rowid) FROM image_annotations)"
        )
    with pytest.raises(ValueError, match="illegal labels"):
        load_annotations(database_path)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE image_annotations SET label = 'sharp' "
            "WHERE rowid = (SELECT MIN(rowid) FROM image_annotations)"
        )

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE image_annotations SET label = 'sharp', image_path = "
            "(SELECT image_path FROM image_annotations ORDER BY rowid LIMIT 1) "
            "WHERE rowid = (SELECT rowid FROM image_annotations ORDER BY rowid LIMIT 2 OFFSET 1)"
        )
    with pytest.raises(ValueError, match="duplicate"):
        load_annotations(database_path)

    with pytest.raises(ValueError):
        split_dataset(pd.DataFrame({"label": ["sharp", "defocused"]}))
