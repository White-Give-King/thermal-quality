from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from thermal_focus.dataset import load_labels, split_dataset


def _write_dataset(root: Path) -> Path:
    records = []
    for group_index in range(18):
        for label in ("sharp", "borderline", "defocused"):
            name = f"{group_index}-{label}.png"
            assert cv2.imwrite(str(root / name), np.full((8, 8), group_index, np.uint8))
            records.append({
                "image_path": name,
                "label": label,
            })
    csv_path = root / "labels.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    return csv_path


def test_stratified_split_is_deterministic_and_has_no_image_leakage(tmp_path: Path) -> None:
    csv_path = _write_dataset(tmp_path)
    frame = load_labels(csv_path)
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


def test_minimal_csv_splits_by_image_without_group_metadata(tmp_path: Path) -> None:
    csv_path = _write_dataset(tmp_path)
    source = pd.read_csv(csv_path)
    source["group_id"] = "unused"
    source.to_csv(csv_path, index=False)
    frame = load_labels(csv_path)
    assert "group_id" not in frame
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
    csv_path = _write_dataset(tmp_path)
    frame = pd.read_csv(csv_path, dtype=str)
    frame.loc[0, "label"] = "valid_target"
    frame.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="illegal labels"):
        load_labels(csv_path)

    frame = pd.read_csv(csv_path, dtype=str)
    frame.loc[0, "label"] = "good"
    frame.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="illegal labels"):
        load_labels(csv_path)

    frame = pd.read_csv(csv_path, dtype=str)
    frame.loc[0, "label"] = "sharp"
    frame.loc[1, "image_path"] = frame.loc[0, "image_path"]
    frame.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_labels(csv_path)

    with pytest.raises(ValueError):
        split_dataset(pd.DataFrame({"label": ["sharp", "defocused"]}))
