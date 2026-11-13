from __future__ import annotations

import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest

from annotation_storage import save_image_annotation
from dataset import load_annotations
from dataset_versioning import DatasetVersionStore
from webui_service import WebUIService, _validate_training_options


def _write_image(path: Path, value: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((16, 16, 3), value, dtype=np.uint8))


def _config() -> dict[str, object]:
    return {
        "image": {"max_dimension": 12000, "float_range": None},
        "thermal_roi": {
            "mode": "centered",
            "width": 16,
            "height": 16,
        },
    }


def _store(tmp_path: Path) -> tuple[DatasetVersionStore, Path]:
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True)
    return DatasetVersionStore(data_root, data_root), data_root


def test_import_creates_immutable_incrementing_versions_and_rolls_back(tmp_path: Path) -> None:
    store, output = _store(tmp_path)
    original = tmp_path / "sharp.png"
    _write_image(original, 10)

    first = store.import_labeled_images(
        {"sharp": [original]}, config=_config(), max_upload_bytes=1024 * 1024
    )
    assert first["active_version"] == 2
    assert first["latest_version"] == 2
    database_v1 = store._version_dir(store.versions_dir, 1) / "annotations.sqlite3"
    database_v2 = store._version_dir(store.versions_dir, 2) / "annotations.sqlite3"
    assert sqlite3.connect(database_v1).execute(
        "SELECT COUNT(*) FROM image_annotations"
    ).fetchone()[0] == 0
    assert len(load_annotations(database_v2, image_root=store.assets_dir)) == 1

    store.activate(1)
    assert store.active_paths()[2] == 1
    next_image = tmp_path / "borderline.png"
    _write_image(next_image, 20)
    third = store.import_labeled_images(
        {"borderline": [next_image]}, config=_config(), max_upload_bytes=1024 * 1024
    )
    assert third["active_version"] == 3
    assert third["latest_version"] == 3
    assert [record["version"] for record in third["versions"]] == [1, 2, 3]
    assert len(load_annotations(store.active_paths()[0], image_root=store.assets_dir)) == 1
    assert (output / "dataset_versions" / "versions" / "v0002" / "annotations.sqlite3").is_file()


def test_first_import_preserves_existing_labeled_baseline(tmp_path: Path) -> None:
    store, output = _store(tmp_path)
    crop = output / "images" / "existing.png"
    _write_image(crop)
    database = store.source_workspace / "annotations.sqlite3"
    save_image_annotation(
        database,
        {
            "image_path": "images/existing.png",
            "label": "defocused",
            "source_path": str(crop),
        },
    )
    incoming = tmp_path / "incoming.png"
    _write_image(incoming, 200)

    status = store.import_labeled_images(
        {"sharp": [incoming]}, config=_config(), max_upload_bytes=1024 * 1024
    )

    assert status["active_version"] == 2
    database_v1 = store._version_dir(store.versions_dir, 1) / "annotations.sqlite3"
    database_v2 = store._version_dir(store.versions_dir, 2) / "annotations.sqlite3"
    assert len(load_annotations(database_v1, image_root=store.assets_dir)) == 1
    assert len(load_annotations(database_v2, image_root=store.assets_dir)) == 2


def test_import_rejects_duplicate_and_invalid_images_without_new_version(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)
    image = tmp_path / "image.png"
    _write_image(image)
    store.import_labeled_images(
        {"sharp": [image]}, config=_config(), max_upload_bytes=1024 * 1024
    )
    with pytest.raises(ValueError, match="duplicate image"):
        store.import_labeled_images(
            {"borderline": [image]}, config=_config(), max_upload_bytes=1024 * 1024
        )
    assert store.status()["latest_version"] == 2

    with pytest.raises(ValueError, match="unsupported focus-quality label"):
        store.import_labeled_images(
            {"invalid": [image]}, config=_config(), max_upload_bytes=1024 * 1024
        )
    assert store.status()["latest_version"] == 2


def test_status_initializes_the_data_directory_as_version_one(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)

    status = store.status()
    assert status["active_version"] == 1
    assert status["latest_version"] == 1
    assert status["versions"][0]["sample_count"] == 0
    assert store.root.exists()


def test_version_one_collects_existing_data_subfolder_annotations(tmp_path: Path) -> None:
    store, data_root = _store(tmp_path)
    legacy = data_root / "older-dataset"
    crop = legacy / "images" / "known.png"
    _write_image(crop)
    save_image_annotation(
        legacy / "annotations.sqlite3",
        {
            "image_path": "images/known.png",
            "label": "sharp",
            "source_path": str(crop),
        },
    )

    status = store.status()
    database, image_root, version = store.active_paths()

    assert status["active_version"] == 1
    assert status["versions"][0]["sample_count"] == 1
    assert version == 1
    assert len(load_annotations(database, image_root=image_root)) == 1


def test_version_import_publishes_new_annotations_from_data_root(tmp_path: Path) -> None:
    store, data_root = _store(tmp_path)
    store.status()
    crop = data_root / "images" / "annotated.png"
    source = data_root / "images_source" / "annotated.png"
    _write_image(crop, 20)
    _write_image(source, 30)
    save_image_annotation(
        data_root / "annotations.sqlite3",
        {
            "image_path": "images/annotated.png",
            "label": "borderline",
            "source_path": str(source),
        },
    )
    incoming = tmp_path / "new.png"
    _write_image(incoming, 40)

    status = store.import_labeled_images(
        {"sharp": [incoming]}, config=_config(), max_upload_bytes=1024 * 1024
    )

    database, image_root, version = store.active_paths()
    frame = load_annotations(database, image_root=image_root)
    assert version == 2
    assert len(frame) == 2
    assert set(frame["label"]) == {"sharp", "borderline"}
    assert status["versions"][0]["sample_count"] == 0


def test_annotation_loads_data_root_sources_and_skips_generated_folders(
    tmp_path: Path,
) -> None:
    import webui_service

    data_root = tmp_path / "data"
    _write_image(data_root / "Camera" / "images_source" / "source.png")
    _write_image(data_root / "Camera" / "images" / "crop.png")
    _write_image(data_root / "group" / "task" / "direct.png")
    _write_image(data_root / "dataset_versions" / "assets" / "images" / "version.png")

    paths = webui_service._data_source_images(data_root)

    assert {path.name for path in paths} == {"source.png", "direct.png"}


def test_training_uses_the_active_dataset_snapshot(tmp_path: Path, monkeypatch) -> None:
    import webui_service

    project = tmp_path / "project"
    monkeypatch.setattr(webui_service, "ROOT", project)
    data_root = project / "data"
    data_root.mkdir(parents=True)
    incoming = tmp_path / "incoming.png"
    _write_image(incoming)
    monkeypatch.setattr(webui_service, "load_config", lambda _: _config())
    service = WebUIService()
    monkeypatch.setattr(service.jobs, "start", lambda options, **kwargs: {
        "job_id": "test", "status": "queued", "output": options["output"]
    })
    service.import_labeled_dataset(data_root, {"sharp": [incoming]})

    options = _validate_training_options(
        {
            "dataset_root": str(data_root),
            "epochs": 1,
        }
    )

    assert options["database"] == str(
        project
        / "data"
        / "dataset_versions"
        / "versions"
        / "v0002"
        / "annotations.sqlite3"
    )
    assert options["image_root"] == str(
        project / "data" / "dataset_versions" / "assets"
    )
    assert options["output"] == str(
        (project / "data" / "runs" / "dataset-v0002").resolve()
    )


def test_import_queues_each_created_version_even_after_rollback(tmp_path: Path, monkeypatch) -> None:
    import threading
    import time
    import webui_service

    data_root = tmp_path / "data"
    data_root.mkdir()
    incoming = tmp_path / "first.png"
    later = tmp_path / "second.png"
    _write_image(incoming, 40)
    _write_image(later, 80)
    monkeypatch.setattr(webui_service, "load_config", lambda _: _config())
    gate = threading.Event()
    commands = []

    class Process:
        def wait(self):
            gate.wait(timeout=10)
            return 0

    def popen(command, **kwargs):
        commands.append(command)
        return Process()

    monkeypatch.setattr(webui_service.subprocess, "Popen", popen)
    service = WebUIService()
    first = service.import_labeled_dataset(data_root, {"sharp": [incoming]})
    service.activate_dataset_version(data_root, 1)
    second = service.import_labeled_dataset(data_root, {"sharp": [later]})
    assert first["training_job"]["status"] == "running"
    assert second["training_job"]["status"] == "queued"
    assert len(commands) == 1
    versions = service.dataset_version_status(data_root)["versions"]
    assert versions[1]["training"]["status"] == "running"
    assert versions[2]["training"]["status"] == "queued"

    service.activate_dataset_version(data_root, 1)
    gate.set()
    deadline = time.monotonic() + 5
    while len(commands) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(commands) == 2
    database = commands[1][commands[1].index("--database") + 1]
    assert Path(database) == data_root / "dataset_versions" / "versions" / "v0003" / "annotations.sqlite3"
    assert Path(second["training_job"]["output"]) == data_root / "runs" / "dataset-v0003"


def test_import_reports_when_created_version_cannot_be_queued(tmp_path: Path, monkeypatch) -> None:
    import webui_service

    data_root = tmp_path / "data"
    data_root.mkdir()
    incoming = tmp_path / "first.png"
    _write_image(incoming)
    monkeypatch.setattr(webui_service, "load_config", lambda _: _config())

    def fail(_payload):
        raise ValueError("invalid options")

    monkeypatch.setattr(webui_service, "_validate_training_options", fail)
    service = WebUIService()
    with pytest.raises(RuntimeError, match="v0002 已创建.*invalid options"):
        service.import_labeled_dataset(data_root, {"sharp": [incoming]})
    assert service.dataset_version_status(data_root)["active_version"] == 2
