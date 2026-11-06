import sqlite3

import pandas as pd

from scripts.train_classifier import _write_annotation_database, parse_args


def test_training_cli_defaults_to_recommended_parameters(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_classifier.py",
            "--database",
            str(tmp_path / "annotations.sqlite3"),
            "--output",
            str(tmp_path / "run"),
        ],
    )

    args = parse_args()

    assert args.architecture == "mobilenet_v3_large"
    assert args.pretrained is True
    assert args.epochs == 30
    assert args.batch_size == 16
    assert args.input_size == 224
    assert args.learning_rate == 1e-4
    assert args.device == "auto"


def test_training_cli_can_disable_pretrained_weights(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_classifier.py",
            "--database",
            str(tmp_path / "annotations.sqlite3"),
            "--output",
            str(tmp_path / "run"),
            "--no-pretrained",
        ],
    )

    assert parse_args().pretrained is False


def test_write_annotation_database_replaces_previous_split(tmp_path) -> None:
    database = tmp_path / "train_split.sqlite3"
    previous = pd.DataFrame(
        [("old.png", "sharp")], columns=["image_path", "label"]
    )
    current = pd.DataFrame(
        [("new.png", "defocused")], columns=["image_path", "label"]
    )

    _write_annotation_database(database, previous)
    _write_annotation_database(database, current)

    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT image_path, label FROM image_annotations"
        ).fetchall()
    assert rows == [("new.png", "defocused")]
