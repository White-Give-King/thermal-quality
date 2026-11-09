"""Label validation and leakage-resistant splits for focus-quality classification."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from decision import LABELS

REQUIRED_COLUMNS = (
    "image_path",
    "label",
)
ALLOWED_LABELS = set(LABELS)


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_annotations(
    database_path: str | Path, *, image_root: str | Path | None = None
) -> pd.DataFrame:
    """Load and validate focus-quality annotations from a workspace SQLite database."""
    source = Path(database_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"annotation database does not exist: {source}")
    try:
        with sqlite3.connect(source) as connection:
            rows = connection.execute(
                "SELECT image_path, label FROM image_annotations"
            ).fetchall()
    except sqlite3.Error as exc:
        raise ValueError(f"could not read annotations from {source}: {exc}") from exc
    frame = pd.DataFrame(rows, columns=REQUIRED_COLUMNS, dtype=str)
    if frame.empty:
        raise ValueError("annotation database contains no labeled rows")
    for column in ("image_path", "label"):
        frame[column] = frame[column].str.strip()
    invalid = sorted(set(frame["label"]) - ALLOWED_LABELS)
    if invalid:
        raise ValueError(f"illegal labels: {invalid}; allowed labels are {sorted(ALLOWED_LABELS)}")
    if (frame["image_path"] == "").any():
        raise ValueError("image_path must be non-empty for every sample")
    resolved: list[str] = []
    root = Path(image_root).resolve() if image_root else source.parent
    for value in frame["image_path"]:
        path = Path(value)
        full_path = path.resolve() if path.is_absolute() else (root / path).resolve()
        if not full_path.is_file():
            raise FileNotFoundError(f"labeled image does not exist: {full_path}")
        resolved.append(str(full_path))
    frame["image_path"] = resolved
    duplicates = frame["image_path"].str.casefold().duplicated(keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, "image_path"].tolist()[:5]
        raise ValueError(f"duplicate image samples found: {examples}")
    distribution = frame["label"].value_counts().to_dict()
    frame.attrs["class_distribution"] = distribution
    return frame


def split_dataset(
    frame: pd.DataFrame, *, seed: int = 42, test_fraction: float = 0.2, validation_fraction: float = 0.2
) -> DatasetSplits:
    """Create deterministic stratified train/validation/test splits by image."""
    if "label" not in frame or frame.empty:
        raise ValueError("frame must contain labeled rows")
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("validation and test fractions must be strictly between 0 and 1")
    if set(frame["label"]) - ALLOWED_LABELS:
        raise ValueError(f"frame contains labels outside {sorted(ALLOWED_LABELS)}")
    outer = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_val_indices, test_indices = next(outer.split(frame, frame["label"]))
    train_val = frame.iloc[train_val_indices].copy()
    test = frame.iloc[test_indices].copy()
    inner_fraction = validation_fraction / (1 - test_fraction)
    inner = StratifiedShuffleSplit(n_splits=1, test_size=inner_fraction, random_state=seed + 1)
    inner_train, validation_indices = next(inner.split(train_val, train_val["label"]))
    train = train_val.iloc[inner_train].copy()
    validation = train_val.iloc[validation_indices].copy()
    partitions = {"train": train, "validation": validation, "test": test}
    for name, partition in partitions.items():
        if partition.empty:
            raise ValueError(f"split produced an empty {name} partition")
    missing_train_labels = ALLOWED_LABELS - set(train["label"])
    if missing_train_labels:
        raise ValueError(
            "training split is missing classes "
            f"{sorted(missing_train_labels)}; add examples of each focus class or change the random seed"
        )
    return DatasetSplits(train, validation, test)
