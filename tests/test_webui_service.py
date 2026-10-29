from __future__ import annotations

import csv
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from thermal_focus import webui_service
from thermal_focus.classifier import load_classifier
from thermal_focus.webui_service import (
    TrainingJobs,
    WebUIService,
    _collect_annotation_images,
    _list_workspaces,
    _validate_training_options,
)


def test_annotation_folder_scan_is_recursive_and_sorted(tmp_path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    first = tmp_path / "a.png"
    second = nested / "b.jpg"
    first.write_bytes(b"not decoded by folder discovery")
    second.write_bytes(b"not decoded by folder discovery")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    assert _collect_annotation_images(tmp_path) == [first.resolve(), second.resolve()]
    with pytest.raises(NotADirectoryError, match="image folder does not exist"):
        _collect_annotation_images(tmp_path / "missing")


def test_workspace_discovery_maps_same_name_data_and_run_directories(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    monkeypatch.setattr(webui_service, "ROOT", project_root)
    workspace_root = tmp_path / "source-workspaces"
    image_workspace = workspace_root / "A7"
    image_workspace.mkdir(parents=True)
    (image_workspace / "frame.png").write_bytes(b"test")
    (workspace_root / "empty").mkdir()

    workspaces = _list_workspaces(workspace_root)

    assert len(workspaces) == 1
    assert workspaces[0]["name"] == "A7"
    assert workspaces[0]["image_dir"] == str(image_workspace.resolve())
    assert workspaces[0]["data_dir"] == str(project_root / "data" / "A7")
    assert workspaces[0]["runs_dir"] == str(project_root / "runs" / "A7")
    assert workspaces[0]["labels_csv"] == str(project_root / "data" / "A7" / "labels.csv")


def test_workspace_listing_returns_subfolders_and_rejects_invalid_root(tmp_path) -> None:
    root = tmp_path / "workspaces"
    (root / "CameraA").mkdir(parents=True)
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    assert cv2.imwrite(str(root / "CameraA" / "frame.png"), image)
    service = WebUIService()

    result = service.list_workspaces(str(root))
    assert [workspace["name"] for workspace in result["workspaces"]] == ["CameraA"]
    assert result["workspaces"][0]["image_count"] == 1
    with pytest.raises(NotADirectoryError):
        service.list_workspaces(str(root / "missing"))


def test_training_options_validate_dataset_and_parameters(tmp_path) -> None:
    labels = tmp_path / "labels.csv"
    labels.write_text("image_path,label\n", encoding="utf-8")
    options = _validate_training_options(
        {
            "labels": str(labels),
            "output": str(tmp_path / "run"),
            "epochs": "3",
            "batch_size": "4",
            "input_size": "96",
            "learning_rate": "0.001",
            "seed": "9",
            "device": "cpu",
        }
    )

    assert options["labels"] == str(labels.resolve())
    assert options["epochs"] == 3
    assert options["architecture"] == "mobilenet_v3_small"
    assert options["batch_size"] == 4
    assert options["device"] == "cpu"


def test_training_options_reject_invalid_parameters(tmp_path) -> None:
    labels = tmp_path / "labels.csv"
    labels.write_text("image_path,label\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be positive"):
        _validate_training_options(
            {"labels": str(labels), "output": str(tmp_path / "run"), "epochs": 0}
        )
    with pytest.raises(ValueError, match="device must be"):
        _validate_training_options(
            {"labels": str(labels), "output": str(tmp_path / "run"), "device": "tpu"}
        )
    with pytest.raises(ValueError, match="learning rate must be finite"):
        _validate_training_options(
            {
                "labels": str(labels),
                "output": str(tmp_path / "run"),
                "learning_rate": "nan",
            }
        )
    with pytest.raises(ValueError, match="pretrained must be a boolean"):
        _validate_training_options(
            {
                "labels": str(labels),
                "output": str(tmp_path / "run"),
                "pretrained": "false",
            }
        )
    with pytest.raises(ValueError, match="architecture must be one of"):
        _validate_training_options(
            {
                "labels": str(labels),
                "output": str(tmp_path / "run"),
                "architecture": "unknown",
            }
        )


def test_training_workspace_automatically_maps_data_and_runs(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    monkeypatch.setattr(webui_service, "ROOT", project_root)
    labels = project_root / "data" / "CameraA" / "labels.csv"
    labels.parent.mkdir(parents=True)
    labels.write_text("image_path,label\n", encoding="utf-8")

    options = _validate_training_options({"workspace_name": "CameraA"})

    assert options["labels"] == str(labels.resolve())
    assert options["output"] == str((project_root / "runs" / "CameraA").resolve())


def test_training_workspace_rejects_path_traversal(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(webui_service, "ROOT", tmp_path / "project")
    with pytest.raises(ValueError, match="single folder name"):
        _validate_training_options({"workspace_name": ".."})


def test_prediction_requires_an_existing_model(tmp_path) -> None:
    service = WebUIService()
    with pytest.raises(FileNotFoundError, match="checkpoint does not exist"):
        service.predict(
            [str(tmp_path / "image.png")],
            model=str(tmp_path / "missing.pt"),
            device="cpu",
        )


def test_list_trained_models_scans_runs_directory_newest_first(tmp_path) -> None:
    runs = tmp_path / "runs"
    older = runs / "older-workspace"
    newer = runs / "newer-workspace"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    older_best = older / "best.pt"
    newer_best = newer / "best.pt"
    older_best.write_bytes(b"weights")
    newer_best.write_bytes(b"weights")
    os.utime(older_best, (1000000, 1000000))
    os.utime(newer_best, (2000000, 2000000))
    (older / "training_config.json").write_text(
        json.dumps({"architecture": "resnet18", "input_size": 224}), encoding="utf-8"
    )

    models = webui_service.list_trained_models(runs)

    assert [Path(item["path"]) for item in models] == [
        newer_best.resolve(),
        older_best.resolve(),
    ]
    assert models[0]["workspace"] == "newer-workspace"
    assert models[0]["architecture"] == "未知"
    assert models[1]["workspace"] == "older-workspace"
    assert models[1]["architecture"] == "resnet18"
    assert models[1]["input_size"] == 224


def test_list_trained_models_returns_empty_when_runs_directory_missing(tmp_path) -> None:
    assert webui_service.list_trained_models(tmp_path / "missing") == []


def test_prediction_file_returns_real_model_result_and_previews(monkeypatch, tmp_path) -> None:
    class SharpPredictor(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return torch.tensor((5.0, 0.0, 0.0), device=images.device).repeat(
                images.shape[0], 1
            )

    config = tmp_path / "config.yaml"
    config.write_text(
        "image:\n"
        "  expected_width: 64\n"
        "  expected_height: 64\n"
        "thermal_roi:\n"
        "  mode: fixed\n"
        "  x: 8\n"
        "  y: 4\n"
        "  width: 48\n"
        "  height: 56\n"
        "classifier:\n"
        "  confidence_threshold: 0.6\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("THERMAL_FOCUS_CONFIG", str(config))
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(
        webui_service,
        "load_classifier",
        lambda _path, device: (
            SharpPredictor(),
            {"input_size": 64, "model_version": "test-model"},
        ),
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(encoded.tobytes())

    response = WebUIService().predict(
        [image_path], model=str(checkpoint), device="cpu"
    )

    result = response["results"][0]
    assert result["prediction"]["status"] == "sharp"
    assert result["prediction"]["model_version"] == "test-model"
    for key in ("annotated_image", "thermal_image"):
        preview = result[key]
        assert isinstance(preview, np.ndarray)
        assert preview.shape[2] == 3
        assert preview.dtype == np.uint8


def test_web_annotation_workspace_navigate_label_and_roi_persistence(monkeypatch, tmp_path) -> None:
    config = tmp_path / "default.yaml"
    config.write_text(
        "image:\n"
        "  expected_width: 64\n"
        "  expected_height: 64\n"
        "thermal_roi:\n"
        "  mode: fixed\n"
        "  x: 8 # keep this comment\n"
        "  y: 4\n"
        "  width: 48\n"
        "  height: 56\n"
        "classifier:\n"
        "  input_size: 64\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("THERMAL_FOCUS_CONFIG", str(config))
    project_root = tmp_path / "project"
    monkeypatch.setattr(webui_service, "ROOT", project_root)
    images = []
    for index in range(2):
        image = np.full((64, 64, 3), index * 70, dtype=np.uint8)
        cv2.rectangle(image, (10, 10), (50, 52), (240, 180, 30), 2)
        success, encoded = cv2.imencode(".png", image)
        assert success
        images.append(encoded.tobytes())
    service = WebUIService()

    workspace_root = tmp_path / "source-workspaces"
    input_dir = workspace_root / "ThermalA"
    input_dir.mkdir(parents=True)
    (input_dir / "frame-1.png").write_bytes(images[0])
    (input_dir / "frame-2.png").write_bytes(images[1])
    created = service.start_annotation(str(workspace_root), "ThermalA")
    assert created["input_dir"] == str(input_dir.resolve())
    assert created["output_dir"] == str(project_root / "data" / "ThermalA")
    output = project_root / "data" / "ThermalA"
    session_id = created["session_id"]
    current = service.render_annotation(
        service.get_annotation_session(session_id), original=False
    )
    assert current["total"] == 2
    assert current["index"] == 0
    preview = current["preview"]
    assert isinstance(preview, np.ndarray)
    assert preview.shape[2] == 3
    assert preview.dtype == np.uint8
    assert len(current["gallery"]) == 2
    assert len(current["gallery_indices"]) == 2
    assert current["preview_errors"] == []

    saved = service.save_label(session_id, label="sharp", advance=True)
    assert saved["index"] == 1
    assert saved["label"] is None

    csv_path = output / "labels.csv"
    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["label"] == "sharp"
    saved_crop = output / rows[0]["image_path"]
    assert cv2.imread(str(saved_crop)).shape[:2] == (56, 48)

    repeated = service.start_annotation(str(workspace_root), "ThermalA")
    updated = service.save_label(
        repeated["session_id"], label="borderline", advance=False
    )
    assert updated["saved"]["label"] == "borderline"
    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["label"] == "borderline"

    applied = service.update_roi(
        repeated["session_id"], x=10, y=5, width=40, height=50, save=False
    )
    assert applied["roi"] == {"x": 10, "y": 5, "width": 40, "height": 50}
    persisted = service.update_roi(
        repeated["session_id"], x=10, y=5, width=40, height=50, save=True
    )
    assert persisted["updated_labels"] == 1
    assert cv2.imread(str(saved_crop)).shape[:2] == (50, 40)
    config_text = config.read_text(encoding="utf-8")
    assert "x: 10 # keep this comment" in config_text


def test_web_annotation_session_creation_does_not_decode_entire_workspace(
    monkeypatch, tmp_path
) -> None:
    config = tmp_path / "default.yaml"
    config.write_text(
        "image:\n  max_dimension: 12000\n"
        "thermal_roi:\n  mode: fixed\n  x: 0\n  y: 0\n  width: 64\n  height: 64\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("THERMAL_FOCUS_CONFIG", str(config))
    project_root = tmp_path / "project"
    monkeypatch.setattr(webui_service, "ROOT", project_root)
    workspace = tmp_path / "source" / "ThermalA"
    workspace.mkdir(parents=True)
    for index in range(50):
        (workspace / f"frame-{index:03}.png").write_bytes(b"not decoded during startup")

    def fail_if_decoded(*_args, **_kwargs):
        pytest.fail("session startup must not decode every image")

    monkeypatch.setattr(webui_service, "load_image_for", fail_if_decoded)
    created = WebUIService().start_annotation(str(workspace.parent), workspace.name)

    assert created["total"] == 50


def test_web_annotation_rejects_invalid_label_and_image_index(monkeypatch, tmp_path) -> None:
    config = tmp_path / "default.yaml"
    config.write_text(
        "image:\n"
        "  expected_width: 64\n"
        "  expected_height: 64\n"
        "thermal_roi:\n"
        "  mode: fixed\n"
        "  x: 0\n"
        "  y: 0\n"
        "  width: 64\n"
        "  height: 64\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("THERMAL_FOCUS_CONFIG", str(config))
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    project_root = tmp_path / "project"
    monkeypatch.setattr(webui_service, "ROOT", project_root)
    workspace_root = tmp_path / "source"
    source_dir = workspace_root / "WorkspaceA"
    source_dir.mkdir(parents=True)
    assert cv2.imwrite(str(source_dir / "frame.png"), image)
    service = WebUIService()
    created = service.start_annotation(str(workspace_root), "WorkspaceA")
    session_id = created["session_id"]

    with pytest.raises(ValueError, match="label must be one of"):
        service.save_label(session_id, label="valid_target")
    with pytest.raises(ValueError, match="image index is out of range"):
        service.navigate(session_id, 5)
    assert not (project_root / "data" / "WorkspaceA" / "labels.csv").exists()


def test_training_manager_starts_only_one_job(monkeypatch, tmp_path) -> None:
    allow_exit = threading.Event()

    class FakeProcess:
        def wait(self) -> int:
            allow_exit.wait(timeout=2)
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    manager = TrainingJobs()
    options = {
        "labels": str(tmp_path / "labels.csv"),
        "output": str(tmp_path / "run"),
        "epochs": 1,
        "architecture": "resnet18",
        "batch_size": 1,
        "input_size": 32,
        "learning_rate": 0.001,
        "seed": 42,
        "device": "cpu",
        "resume": "",
        "pretrained": False,
    }

    job = manager.start(options)
    assert job["status"] == "running"
    assert manager.status(job["job_id"])["job_id"] == job["job_id"]
    with pytest.raises(RuntimeError, match="already running"):
        manager.start(options)
    allow_exit.set()


def test_webui_runs_training_subprocess_to_checkpoint(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    labels_path = tmp_path / "labels.csv"
    rows = []
    for label, blur in (("sharp", 0.0), ("borderline", 1.2), ("defocused", 4.0)):
        for index in range(15):
            image = np.full((64, 48, 3), 100, dtype=np.uint8)
            cv2.rectangle(image, (8, 8), (39, 55), (220, 160, 40), 2)
            if blur:
                kernel = 5 if blur < 2 else 13
                image = cv2.GaussianBlur(image, (kernel, kernel), blur)
            path = image_dir / f"{label}-{index}.png"
            assert cv2.imwrite(str(path), image)
            rows.append({"image_path": str(path), "label": label})
    with labels_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image_path", "label"))
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "run"
    service = WebUIService()
    job = service.start_training(
        {
            "labels": str(labels_path),
            "output": str(output),
            "epochs": 1,
            "architecture": "resnet18",
            "batch_size": 9,
            "input_size": 64,
            "learning_rate": 0.0001,
            "seed": 42,
            "device": "cpu",
        }
    )
    job_id = job["job_id"]
    deadline = time.monotonic() + 45
    status = service.training_status(job_id)
    while time.monotonic() < deadline:
        status = service.training_status(job_id)
        if status["status"] != "running":
            break
        time.sleep(0.25)

    assert status["status"] == "completed", status.get("log")
    assert (output / "best.pt").is_file()
    assert (output / "last.pt").is_file()
    assert "epoch=1" in status["log"]
    _, metadata = load_classifier(output / "best.pt", device="cpu")
    assert metadata["architecture"] == "resnet18"
