from __future__ import annotations

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
import yaml

import webui_service
from annotation_storage import (
    load_image_annotations,
    load_image_rois,
    save_roi_coordinates,
)
from classifier import load_classifier
from webui_service import (
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
    assert workspaces[0]["runs_dir"] == str(project_root / "data" / "A7" / "runs")
    assert workspaces[0]["annotation_database"] == str(
        workspace_root / "A7" / "annotations.sqlite3"
    )


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


def test_workspace_discovery_counts_workspaces_concurrently_and_keeps_order(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "workspaces"
    for name in ("CameraB", "CameraA"):
        source = root / name / "images_source"
        (source / "nested").mkdir(parents=True)
        (source / "frame.png").write_bytes(b"image")
        (source / "nested" / "frame.jpg").write_bytes(b"image")
        (source / "notes.txt").write_text("ignore", encoding="utf-8")

    barrier = threading.Barrier(2)
    original_record = webui_service._workspace_record

    def synchronized_record(workspace_root, directory):
        barrier.wait(timeout=5)
        return original_record(workspace_root, directory)

    monkeypatch.setattr(webui_service, "_workspace_record", synchronized_record)

    workspaces = _list_workspaces(root)

    assert [workspace["name"] for workspace in workspaces] == ["CameraB", "CameraA"]
    assert [workspace["image_count"] for workspace in workspaces] == [2, 2]


def test_workspace_discovery_descends_into_department_groups(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    monkeypatch.setattr(webui_service, "ROOT", project_root)
    root = tmp_path / "data"
    task = root / "A10" / "A10-2026-01-01-00-00-00"
    (task / "images_source").mkdir(parents=True)
    (task / "images_source" / "frame.png").write_bytes(b"test")
    flat = root / "Flat"
    flat.mkdir()
    (flat / "frame.png").write_bytes(b"test")
    # 标注输出目录中的裁剪图不应被当作工作区。
    output = root / "A10-2026-01-01-00-00-00"
    output.mkdir()
    (output / "images").mkdir()
    (output / "images" / "crop.png").write_bytes(b"test")
    # 同时含标注产物与原图的任务文件夹仍应被识别。
    annotated = root / "A7" / "A7-2026-10-19-08-44-50"
    (annotated / "images_source").mkdir(parents=True)
    (annotated / "images_source" / "shot.jpg").write_bytes(b"test")
    special = root / "1特殊数据集"
    special_source_dir = special / "images_source"
    special_source_dir.mkdir(parents=True)
    special_source = special_source_dir / "flat-source.jpg"
    special_source.write_bytes(b"test")
    (special / "images").mkdir()
    (special / "images" / "saved-crop.png").write_bytes(b"test")

    workspaces = _list_workspaces(root)

    assert [workspace["name"] for workspace in workspaces] == [
        "Flat",
        "A7-2026-10-19-08-44-50",
        "A10-2026-01-01-00-00-00",
        "1特殊数据集",
    ]
    task_workspace = next(
        workspace
        for workspace in workspaces
        if workspace["name"] == "A10-2026-01-01-00-00-00"
    )
    assert task_workspace["image_dir"] == str((task / "images_source").resolve())
    assert task_workspace["image_count"] == 1
    special_workspace = workspaces[-1]
    assert special_workspace["image_dir"] == str(special_source_dir.resolve())
    assert special_workspace["image_count"] == 1


def test_training_options_validate_dataset_and_parameters(tmp_path) -> None:
    import sqlite3

    database = tmp_path / "annotations.sqlite3"
    with sqlite3.connect(database):
        pass
    options = _validate_training_options(
        {
            "database": str(database),
            "image_root": str(tmp_path),
            "output": str(tmp_path / "run"),
            "epochs": "3",
            "batch_size": "4",
            "input_size": "96",
            "learning_rate": "0.001",
            "seed": "9",
            "device": "cpu",
        }
    )

    assert options["database"] == str(database.resolve())
    assert options["image_root"] == str(tmp_path.resolve())
    assert options["epochs"] == 3
    assert options["architecture"] == "mobilenet_v3_large"
    assert options["batch_size"] == 4
    assert options["learning_rate"] == 0.001
    assert options["device"] == "cpu"
    assert options["pretrained"] is True


def test_training_options_reject_invalid_parameters(tmp_path) -> None:
    import sqlite3

    database = tmp_path / "annotations.sqlite3"
    with sqlite3.connect(database):
        pass
    with pytest.raises(ValueError, match="must be positive"):
        _validate_training_options(
            {"database": str(database), "output": str(tmp_path / "run"), "epochs": 0}
        )
    with pytest.raises(ValueError, match="device must be"):
        _validate_training_options(
            {"database": str(database), "output": str(tmp_path / "run"), "device": "tpu"}
        )
    with pytest.raises(ValueError, match="learning rate must be finite"):
        _validate_training_options(
            {
                "database": str(database),
                "output": str(tmp_path / "run"),
                "learning_rate": "nan",
            }
        )
    with pytest.raises(ValueError, match="pretrained must be a boolean"):
        _validate_training_options(
            {
                "database": str(database),
                "output": str(tmp_path / "run"),
                "pretrained": "false",
            }
        )
    with pytest.raises(ValueError, match="architecture must be one of"):
        _validate_training_options(
            {
                "database": str(database),
                "output": str(tmp_path / "run"),
                "architecture": "unknown",
            }
        )


def test_training_workspace_automatically_maps_data_and_runs(tmp_path, monkeypatch) -> None:
    import sqlite3

    project_root = tmp_path / "project"
    monkeypatch.setattr(webui_service, "ROOT", project_root)
    workspace = project_root / "data" / "CameraA"
    database = workspace / "annotations.sqlite3"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database):
        pass

    options = _validate_training_options({"workspace_name": "CameraA"})

    assert options["database"] == str(database.resolve())
    assert options["image_root"] == str(workspace.resolve())
    assert options["output"] == str((workspace / "runs").resolve())


def test_training_resolves_database_for_external_workspace_root(
    tmp_path, monkeypatch
) -> None:
    import sqlite3

    project_root = tmp_path / "project"
    monkeypatch.setattr(webui_service, "ROOT", project_root)
    workspace_root = tmp_path / "source"
    workspace = workspace_root / "CameraB"
    workspace.mkdir(parents=True)
    database = workspace / "annotations.sqlite3"
    with sqlite3.connect(database):
        pass

    options = _validate_training_options(
        {"workspace_name": "CameraB", "workspace_root": str(workspace_root)}
    )

    output_dir = project_root / "data" / "CameraB"
    assert options["database"] == str(database.resolve())
    assert options["image_root"] == str(output_dir.resolve())
    assert options["output"] == str((output_dir / "runs").resolve())


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
        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes: list[int] = []

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            self.batch_sizes.append(images.shape[0])
            return torch.tensor((5.0, 0.0, 0.0), device=images.device).repeat(
                images.shape[0], 1
            )

    predictor = SharpPredictor()
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
            predictor,
            {"input_size": 64, "model_version": "test-model"},
        ),
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(encoded.tobytes())
    second_image_path = tmp_path / "frame2.png"
    second_image_path.write_bytes(encoded.tobytes())
    invalid_image_path = tmp_path / "invalid.png"
    invalid_image_path.write_bytes(b"not an image")

    response = WebUIService().predict(
        [image_path, invalid_image_path, second_image_path],
        model=str(checkpoint),
        device="cpu",
    )

    assert predictor.batch_sizes == [2]
    assert [item["filename"] for item in response["results"]] == [
        "frame.png",
        "invalid.png",
        "frame2.png",
    ]
    assert "error" in response["results"][1]
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
        "  x: 2 # keep this comment\n"
        "  y: 3\n"
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
        image = np.full((64, 64, 3), 40 + index * 70, dtype=np.uint8)
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
    assert created["roi_database_path"] == str(input_dir / "annotations.sqlite3")
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
    with pytest.raises(ValueError, match="exceeds input dimensions"):
        service.update_roi(
            session_id, x=17, y=4, width=48, height=56, save=False
        )

    saved = service.save_label(session_id, label="sharp", advance=True)
    assert saved["index"] == 1
    assert saved["label"] is None

    roi_database = input_dir / "annotations.sqlite3"
    annotations = load_image_annotations(roi_database)
    assert len(annotations) == 1
    assert annotations[0]["label"] == "sharp"
    saved_crop = output / annotations[0]["image_path"]
    assert cv2.imread(str(saved_crop)).shape[:2] == (56, 48)
    assert load_image_rois(roi_database)["frame-1.png"] == (8, 4, 48, 56)
    assert annotations[0]["source_path"] == str((input_dir / "frame-1.png").resolve())

    repeated = service.start_annotation(str(workspace_root), "ThermalA")
    assert repeated["index"] == 1
    service.navigate(repeated["session_id"], 0)
    updated = service.save_label(
        repeated["session_id"], label="borderline", advance=False
    )
    assert updated["saved"] is None
    assert updated["skipped"] is True
    assert load_image_annotations(roi_database)[0]["label"] == "sharp"

    applied = service.update_roi(
        repeated["session_id"], x=10, y=5, width=40, height=50, save=False
    )
    assert applied["roi"] == {"x": 10, "y": 5, "width": 40, "height": 50}
    assert load_image_rois(roi_database)["frame-1.png"] == (8, 4, 48, 56)
    persisted = service.update_roi(
        repeated["session_id"], x=10, y=5, width=40, height=50, save=True
    )
    assert persisted["updated_labels"] == 1
    assert cv2.imread(str(saved_crop)).shape[:2] == (50, 40)
    independently_detected = service.navigate(repeated["session_id"], 1)
    assert independently_detected["roi"] == {
        "x": 8,
        "y": 4,
        "width": 48,
        "height": 56,
    }
    saved_second_roi = service.update_roi(
        repeated["session_id"], x=12, y=6, width=30, height=40, save=True
    )
    assert saved_second_roi["updated_labels"] == 0
    assert load_image_rois(roi_database) == {
        "frame-1.png": (10, 5, 40, 50),
        "frame-2.png": (12, 6, 30, 40),
    }
    assert {
        row["source_path"]: row["label"]
        for row in load_image_annotations(roi_database)
    } == {
        str((input_dir / "frame-1.png").resolve()): "sharp",
    }

    restarted = service.start_annotation(str(workspace_root), "ThermalA")
    assert restarted["index"] == 1
    restarted_first = service.render_annotation(
        service.get_annotation_session(restarted["session_id"]), original=False
    )
    assert restarted_first["roi"] == {"x": 12, "y": 6, "width": 30, "height": 40}
    restarted_second = service.navigate(restarted["session_id"], 1)
    assert restarted_second["roi"] == {
        "x": 12,
        "y": 6,
        "width": 30,
        "height": 40,
    }
    restarted_first = service.navigate(restarted["session_id"], 0)
    assert restarted_first["roi"] == {
        "x": 10,
        "y": 5,
        "width": 40,
        "height": 50,
    }
    config_text = config.read_text(encoding="utf-8")
    assert "x: 2 # keep this comment" in config_text


def test_annotation_queue_assigns_distinct_images_across_web_sessions(
    monkeypatch, tmp_path
) -> None:
    config = tmp_path / "default.yaml"
    config.write_text(
        "image:\n  max_dimension: 12000\n"
        "thermal_roi:\n  mode: fixed\n  x: 0\n  y: 0\n  width: 32\n  height: 32\n"
        "classifier:\n  input_size: 32\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("THERMAL_FOCUS_CONFIG", str(config))
    project_root = tmp_path / "project"
    monkeypatch.setattr(webui_service, "ROOT", project_root)
    workspace_root = tmp_path / "source"
    workspace = workspace_root / "CameraA"
    workspace.mkdir(parents=True)
    for index in range(4):
        assert cv2.imwrite(
            str(workspace / f"frame-{index}.png"),
            np.full((32, 32, 3), 40 + index * 30, dtype=np.uint8),
        )

    service = WebUIService()
    first = service.start_annotation(str(workspace_root), "CameraA")
    second = service.start_annotation(str(workspace_root), "CameraA")
    assert (first["index"], second["index"]) == (0, 1)

    first_saved = service.save_label(first["session_id"], "sharp")
    assert first_saved["index"] == 2
    service.save_label(second["session_id"], "borderline", advance=False)
    service.save_label(first["session_id"], "defocused")
    first_session = service.get_annotation_session(first["session_id"])
    claim_key = service._annotation_claim_key(first_session, 3)
    service._annotation_claims[claim_key] = (
        first["session_id"],
        time.monotonic() - 1,
    )
    third = service.start_annotation(str(workspace_root), "CameraA")
    assert third["index"] == 3
    service.save_label(third["session_id"], "sharp")

    annotations = load_image_annotations(workspace / "annotations.sqlite3")
    assert len(annotations) == 4
    assert {row["label"] for row in annotations} == {
        "sharp",
        "borderline",
        "defocused",
    }


def test_annotation_detects_each_frame_and_uses_config_fallback_when_missing(
    monkeypatch, tmp_path
) -> None:
    config = tmp_path / "default.yaml"
    config.write_text(
        "image:\n  max_dimension: 12000\n"
        "thermal_roi:\n  mode: fixed\n  x: 8\n  y: 4\n  width: 48\n  height: 56\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("THERMAL_FOCUS_CONFIG", str(config))
    project_root = tmp_path / "project"
    monkeypatch.setattr(webui_service, "ROOT", project_root)
    workspace_root = tmp_path / "source"
    workspace = workspace_root / "CameraA"
    workspace.mkdir(parents=True)

    bordered_first = np.zeros((64, 64, 3), dtype=np.uint8)
    bordered_first[6:57, 8:56] = (40, 70, 90)
    unbordered_second = np.full((64, 64, 3), 35, dtype=np.uint8)
    bordered_third = np.zeros((64, 64, 3), dtype=np.uint8)
    bordered_third[3:61, 2:62] = (30, 50, 80)
    for index, image in enumerate(
        (bordered_first, unbordered_second, bordered_third), start=1
    ):
        assert cv2.imwrite(str(workspace / f"frame-{index}.png"), image)

    service = WebUIService()
    created = service.start_annotation(str(workspace_root), "CameraA")
    session = service.get_annotation_session(created["session_id"])
    first = service.render_annotation(session, original=False)
    assert first["roi"] == {"x": 8, "y": 6, "width": 48, "height": 51}
    assert service.annotation_image_size(created["session_id"]) == (64, 64)

    second = service.navigate(created["session_id"], 1)
    assert second["roi"] == {"x": 8, "y": 4, "width": 48, "height": 56}
    third = service.navigate(created["session_id"], 2)
    assert third["roi"] == {"x": 2, "y": 3, "width": 60, "height": 58}


def test_annotation_uses_config_fallback_for_each_image_orientation(
    monkeypatch, tmp_path
) -> None:
    config = tmp_path / "default.yaml"
    config.write_text(
        "image:\n  max_dimension: 12000\n"
        "thermal_roi:\n  mode: fixed\n  x: 8\n  y: 4\n  width: 40\n  height: 56\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("THERMAL_FOCUS_CONFIG", str(config))
    project_root = tmp_path / "project"
    monkeypatch.setattr(webui_service, "ROOT", project_root)
    workspace_root = tmp_path / "source"
    workspace = workspace_root / "CameraA"
    workspace.mkdir(parents=True)

    portrait = np.zeros((80, 64, 3), dtype=np.uint8)
    portrait[8:72, 10:54] = (40, 70, 90)
    landscape = np.full((64, 80, 3), 35, dtype=np.uint8)
    assert cv2.imwrite(str(workspace / "frame-1.png"), portrait)
    assert cv2.imwrite(str(workspace / "frame-2.png"), landscape)

    service = WebUIService()
    created = service.start_annotation(str(workspace_root), "CameraA")
    first = service.render_annotation(
        service.get_annotation_session(created["session_id"]), original=False
    )
    assert first["roi"] == {"x": 10, "y": 8, "width": 44, "height": 64}

    second = service.navigate(created["session_id"], 1)
    assert second["roi"] == {"x": 20, "y": 4, "width": 40, "height": 56}


def test_save_roi_coordinates_switches_centered_config_to_fixed(tmp_path) -> None:
    config = tmp_path / "default.yaml"
    config.write_text(
        "image:\n  max_dimension: 12000\n"
        "thermal_roi:\n  mode: centered\n  width: 64\n  height: 64\n"
        "classifier:\n  input_size: 64\n",
        encoding="utf-8",
    )

    save_roi_coordinates(config, (10, 5, 40, 50))

    roi = yaml.safe_load(config.read_text(encoding="utf-8"))["thermal_roi"]
    assert roi["mode"] == "fixed"
    assert (roi["x"], roi["y"], roi["width"], roi["height"]) == (10, 5, 40, 50)


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
    assert load_image_annotations(source_dir / "annotations.sqlite3") == []


def test_web_annotation_supports_grouped_workspace(monkeypatch, tmp_path) -> None:
    config = tmp_path / "default.yaml"
    config.write_text(
        "image:\n"
        "  expected_width: 64\n"
        "  expected_height: 64\n"
        "thermal_roi:\n"
        "  mode: fixed\n"
        "  x: 0\n  y: 0\n  width: 64\n  height: 64\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("THERMAL_FOCUS_CONFIG", str(config))
    project_root = tmp_path / "project"
    monkeypatch.setattr(webui_service, "ROOT", project_root)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    root = project_root / "data"
    task = root / "A10" / "A10-2026-01-01-00-00-00"
    task.mkdir(parents=True)
    assert cv2.imwrite(str(task / "frame.png"), image)
    service = WebUIService()

    created = service.start_annotation(str(root), "A10-2026-01-01-00-00-00")

    assert created["total"] == 1
    assert created["input_dir"] == str(task.resolve())
    # data/ 下的工作区：标注产物直接写入工作区文件夹。
    assert created["output_dir"] == str(task.resolve())
    with pytest.raises(ValueError, match="workspace does not exist"):
        service.start_annotation(str(root), "Missing")


def test_training_options_resolve_grouped_workspace_paths(monkeypatch, tmp_path) -> None:
    import sqlite3

    project_root = tmp_path / "project"
    monkeypatch.setattr(webui_service, "ROOT", project_root)
    task = project_root / "data" / "A10" / "A10-2026-01-01-00-00-00"
    task.mkdir(parents=True)
    database = task / "annotations.sqlite3"
    with sqlite3.connect(database):
        pass

    options = webui_service._validate_training_options(
        {"workspace_name": "A10-2026-01-01-00-00-00"}
    )

    assert options["database"] == str(database.resolve())
    assert options["image_root"] == str(task.resolve())
    assert options["output"] == str((task / "runs").resolve())


def test_list_trained_models_scans_workspace_run_dirs(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    monkeypatch.setattr(webui_service, "ROOT", project_root)
    task_runs = project_root / "data" / "A10" / "A10-2026-01-01-00-00-00" / "runs"
    task_runs.mkdir(parents=True)
    best = task_runs / "best.pt"
    best.write_bytes(b"weights")
    legacy_runs = project_root / "runs" / "legacy-ws"
    legacy_runs.mkdir(parents=True)
    legacy_best = legacy_runs / "best.pt"
    legacy_best.write_bytes(b"weights")
    os.utime(best, (2000000, 2000000))
    os.utime(legacy_best, (1000000, 1000000))

    models = webui_service.list_trained_models()

    assert [Path(item["path"]) for item in models] == [
        best.resolve(),
        legacy_best.resolve(),
    ]
    assert models[0]["workspace"] == "A10-2026-01-01-00-00-00"
    assert models[1]["workspace"] == "legacy-ws"


def test_training_manager_starts_only_one_job(monkeypatch, tmp_path) -> None:
    import sqlite3

    allow_exit = threading.Event()

    class FakeProcess:
        def wait(self) -> int:
            allow_exit.wait(timeout=2)
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    manager = TrainingJobs()
    database = tmp_path / "annotations.sqlite3"
    with sqlite3.connect(database):
        pass
    options = {
        "database": str(database),
        "image_root": str(tmp_path),
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
    import sqlite3

    image_dir = tmp_path / "images"
    image_dir.mkdir()
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
            rows.append((str(path), label))
    database_path = tmp_path / "annotations.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE image_annotations (image_path TEXT, label TEXT)"
        )
        connection.executemany(
            "INSERT INTO image_annotations (image_path, label) VALUES (?, ?)",
            rows,
        )

    output = tmp_path / "run"
    service = WebUIService()
    job = service.start_training(
        {
            "database": str(database_path),
            "image_root": str(tmp_path),
            "output": str(output),
            "epochs": 1,
            "architecture": "resnet18",
            "batch_size": 9,
            "input_size": 64,
            "learning_rate": 0.0001,
            "seed": 42,
            "device": "cpu",
            "pretrained": False,
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
