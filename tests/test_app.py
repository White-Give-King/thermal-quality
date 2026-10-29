from __future__ import annotations

from fastapi.testclient import TestClient
from gradio.routes import App

from app import _selected_gallery_index, demo
from thermal_focus import webui_service


def test_gradio_page_is_served_and_callbacks_are_registered() -> None:
    client = TestClient(App.create_app(demo))

    page = client.get("/")
    assert page.status_code == 200
    assert "红外热图清晰度与对焦质量" in page.text

    assert {"scan_workspaces", "begin_annotation", "predict_images", "train"} <= {
        item.get("api_name") for item in demo.config["dependencies"]
    }


def test_gradio_workspace_scan_callback_updates_workspace_fields(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    monkeypatch.setattr(webui_service, "ROOT", project_root)
    source_root = tmp_path / "source"
    other_dir = source_root / "CameraB"
    image_dir = source_root / "A7-10-19"
    other_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    (image_dir / "frame.png").write_bytes(b"image discovery only")
    (other_dir / "frame.png").write_bytes(b"image discovery only")

    callback = next(
        item.fn for item in demo.fns.values() if item.fn.__name__ == "scan_workspaces"
    )
    records, dropdown, message, summary = callback(str(source_root))

    assert [record["name"] for record in records] == ["A7-10-19", "CameraB"]
    assert dropdown.choices == [("A7-10-19", "A7-10-19"), ("CameraB", "CameraB")]
    assert dropdown.value == "A7-10-19"
    assert "2 个工作区" in message
    assert str(image_dir.resolve()) in summary
    assert "data/A7-10-19/" in summary
    assert "runs/A7-10-19/" in summary


def test_gallery_selection_maps_visible_thumbnail_to_absolute_image() -> None:
    indices = [3, 4, 5, 6, 7]

    assert _selected_gallery_index(indices, 2) == 5
    assert _selected_gallery_index(indices, (4, 0)) == 7


def test_gallery_selection_rejects_stale_position() -> None:
    import pytest

    with pytest.raises(ValueError, match="out of range"):
        _selected_gallery_index([10, 11], 2)


def test_annotation_gallery_provides_preview_without_separate_image_view() -> None:
    components = demo.config["components"]
    gal_images = [
        item for item in components
        if item["type"] == "gallery"
        and item["props"].get("columns") == 8
        and item["props"].get("rows") == 1
        and item["props"].get("show_label") is False
    ]
    assert len(gal_images) == 1, f"expected 1 neighbour gallery, got {len(gal_images)}"
    gallery = gal_images[0]["props"]
    assert gallery.get("allow_preview") is True
    assert gallery.get("object_fit") == "contain"
    assert gallery.get("height") == 420
    assert not [
        item
        for item in components
        if item["type"] == "image" and item["props"].get("label") == "当前图片"
    ]
    buttons = {
        item["props"]["value"]: item["props"]
        for item in components
        if item["type"] == "button"
        and item.get("props", {}).get("value") in ("← 上一张", "下一张 →")
    }
    assert buttons["← 上一张"]["size"] == "lg"
    assert buttons["下一张 →"]["size"] == "lg"
    index = next(
        item
        for item in components
        if item["type"] == "number" and item.get("props", {}).get("placeholder") == "序号"
    )
    assert index["props"]["show_label"] is False


def test_annotation_image_actions_keep_original_view_checkbox_state() -> None:
    components = {item["id"]: item for item in demo.config["components"]}
    original_view = next(
        item["id"]
        for item in components.values()
        if item["type"] == "checkbox"
        and item.get("props", {}).get("label") == "显示原图与热区框"
    )
    # Annotation-area dependencies: 3 named + 8 lambdas = 11 total
    annotation_names = {
        "begin_annotation",
        "_select_neighbour",
        "navigate_annotation",
        "lambda",
        "lambda_1",
        "lambda_2",
        "lambda_3",
        "lambda_4",
        "lambda_5",
        "lambda_6",
        "lambda_7",
    }
    annotation_actions = [
        dep for dep in demo.config["dependencies"]
        if dep.get("api_name") in annotation_names
    ]
    assert len(annotation_actions) == len(annotation_names)
    assert all(original_view in dep["inputs"] for dep in annotation_actions)
