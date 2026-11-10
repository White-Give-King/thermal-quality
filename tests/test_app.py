from __future__ import annotations

from fastapi.testclient import TestClient
from gradio.routes import App

import app as app_entrypoint
import webui_service
from webui.annotation_tab import (
    _bounded_roi,
    _roi_input_updates,
    _selected_gallery_index,
)
from webui.app import create_demo, demo


def test_main_exposes_launched_demo_for_gradio_hot_reload(monkeypatch) -> None:
    class FakeDemo:
        css = "test-css"
        head = "test-head"

        def launch(self, **kwargs) -> None:
            self.launch_kwargs = kwargs

    launched_demo = FakeDemo()
    create_demo_options = {}

    def fake_create_demo(*, inference_only=False):
        create_demo_options["inference_only"] = inference_only
        return launched_demo

    monkeypatch.setattr(
        app_entrypoint,
        "create_demo",
        fake_create_demo,
    )

    app_entrypoint.main([])

    assert app_entrypoint.demo is launched_demo
    assert create_demo_options == {"inference_only": False}
    assert launched_demo.launch_kwargs == {
        "server_name": "0.0.0.0",
        "server_port": 7860,
        "show_error": True,
        "css": "test-css",
        "head": "test-head",
    }


def test_main_accepts_custom_port(monkeypatch) -> None:
    class FakeDemo:
        css = "test-css"
        head = "test-head"

        def launch(self, **kwargs) -> None:
            self.launch_kwargs = kwargs

    launched_demo = FakeDemo()
    monkeypatch.setattr(app_entrypoint, "create_demo", lambda **_: launched_demo)

    app_entrypoint.main(["--port", "8080"])

    assert launched_demo.launch_kwargs["server_port"] == 8080


def test_gradio_page_is_served_and_callbacks_are_registered() -> None:
    client = TestClient(App.create_app(demo))

    page = client.get("/")
    assert page.status_code == 200
    assert "红外热图清晰度与对焦质量" in page.text

    assert {
        "scan_workspaces",
        "begin_annotation",
        "next_unlabeled_annotation",
        "predict_images",
        "train",
    } <= {item.get("api_name") for item in demo.config["dependencies"]}


def test_inference_controls_expose_recommended_defaults_on_hover() -> None:
    components = {
        item["props"].get("label"): item["props"]
        for item in demo.config["components"]
    }

    model_info = components["选择模型（runs 目录下已训练权重，可输入自定义路径）"]["info"]
    device_info = components["设备"]["info"]

    assert "输入尺寸优先读取权重元数据" in model_info
    assert "CUDA 可用时使用 CUDA" in device_info
    assert "809 × 1080" in model_info
    assert "12000 px" in model_info
    assert "置信度阈值 0.60" in model_info
    assert "未记录时默认 224" in model_info


def test_training_controls_use_recommended_defaults() -> None:
    components = {
        item["props"].get("label"): item["props"]
        for item in demo.config["components"]
    }

    assert components["Epoch"]["value"] == 30
    assert components["模型架构"]["value"] == "mobilenet_v3_large"
    assert components["Batch size"]["value"] == 16
    assert components["输入尺寸"]["value"] == 224
    assert components["学习率"]["value"] == 0.0001
    assert components["训练设备"]["value"] == "auto"
    assert components["显式下载 torchvision 预训练权重（需要网络）"]["value"] is True
    assert "不要设为 1.0" in components["学习率"]["info"]


def test_inference_only_ui_excludes_dataset_annotation_and_training() -> None:
    inference_demo = create_demo(inference_only=True)
    component_labels = {
        item["props"].get("label")
        for item in inference_demo.config["components"]
    }
    component_values = [
        item["props"].get("value")
        for item in inference_demo.config["components"]
    ]
    callback_names = {
        item.get("api_name") for item in inference_demo.config["dependencies"]
    }

    assert "选择图片（可多选）" in component_labels
    assert "开始推理" in component_values
    assert not {"Epoch", "工作区根目录", "图片标注"} & component_labels
    assert "predict_images" in callback_names
    assert not {"train", "scan_workspaces", "begin_annotation"} & callback_names


def test_inference_api_documentation_describes_request_and_response() -> None:
    dependency = next(
        item
        for item in demo.config["dependencies"]
        if item.get("api_name") == "predict_images"
    )

    assert dependency["api_description"] == (
        "上传一张或多张图片进行热区检测和清晰度三分类。输入依次为：图片文件列表、"
        "已训练的 .pt 模型权重路径、推理设备（auto/cuda/cpu）。返回依次为："
        "处理状态文本、推理预览图库、分类结果表（文件名、预测标签、模型分数、错误）。"
        "图库对每张成功处理的图片展示原图热区框和热区裁剪图；单张图片读取或解码失败时，"
        "错误写入该图片对应的结果行，其他图片继续处理。模型分数是未校准的最高类别 "
        "softmax 概率乘以 100，不代表准确率。"
    )


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

    assert [record["name"] for record in records] == ["CameraB", "A7-10-19"]
    assert dropdown.choices == [("CameraB", "CameraB"), ("A7-10-19", "A7-10-19")]
    assert dropdown.value == "A7-10-19"
    assert "2 个工作区" in message
    assert str(image_dir.resolve()) in summary
    assert str((project_root / "data" / "A7-10-19").resolve()) in summary
    assert str((image_dir / "annotations.sqlite3")) in summary


def test_workspace_status_components_do_not_scroll() -> None:
    components = {
        item["props"].get("elem_id"): item
        for item in demo.config["components"]
        if item["type"] == "markdown"
    }

    assert components["workspace-message"]["props"]["elem_classes"] == ["workspace-status"]
    assert components["workspace-summary"]["props"]["elem_classes"] == ["workspace-summary"]
    assert ".workspace-status,\n.workspace-summary {" in demo.css
    assert "overflow: visible !important;" in demo.css
    assert "#workspace-summary code {" in demo.css
    assert "overflow-wrap: anywhere;" in demo.css


def test_gallery_selection_maps_visible_thumbnail_to_absolute_image() -> None:
    indices = [3, 4, 5, 6, 7]

    assert _selected_gallery_index(indices, 2) == 5
    assert _selected_gallery_index(indices, (4, 0)) == 7


def test_gallery_selection_rejects_stale_position() -> None:
    import pytest

    with pytest.raises(ValueError, match="out of range"):
        _selected_gallery_index([10, 11], 2)


def test_gallery_selection_refreshes_thumbnails_and_index_mapping(monkeypatch) -> None:
    from types import SimpleNamespace

    gallery_result = {
        "gallery": ["refreshed-thumbnail"],
        "gallery_indices": [4, 5, 6],
        "index": 5,
        "total": 10,
        "labeled_count": 2,
        "label": None,
        "filename": "frame-6.png",
        "preview_errors": [],
        "roi": {"x": 0, "y": 0, "width": 1, "height": 1},
        "image_size": {"width": 1, "height": 1},
    }
    navigated_indices = []
    monkeypatch.setattr(
        webui_service.WebUIService,
        "navigate",
        lambda self, session_id, index: navigated_indices.append(index),
    )
    monkeypatch.setattr(
        webui_service.WebUIService,
        "get_annotation_session",
        lambda self, session_id: object(),
    )
    monkeypatch.setattr(
        webui_service.WebUIService,
        "render_annotation",
        lambda self, session, *, original: gallery_result,
    )
    callback = next(
        item.fn for item in demo.fns.values() if item.fn.__name__ == "_select_neighbour"
    )

    outputs = callback(
        "session",
        [3, 4, 5],
        False,
        SimpleNamespace(index=2),
    )

    assert navigated_indices == [5]
    assert outputs[0]["value"] == ["refreshed-thumbnail"]
    assert outputs[0]["selected_index"] == 1
    assert outputs[0]["preview"] is True
    assert outputs[-1] == [4, 5, 6]


def test_roi_input_limits_follow_image_edges_and_clamp_values() -> None:
    assert _bounded_roi(90, 70, 40, 30, 100, 80) == (60, 50, 40, 30)
    assert _bounded_roi(-5, -2, 0, 0, 100, 80) == (0, 0, 1, 1)

    x_update, y_update, width_update, height_update = _roi_input_updates(
        60, 50, 40, 30
    )
    assert [update["value"] for update in (x_update, y_update, width_update, height_update)] == [
        60,
        50,
        40,
        30,
    ]
    assert all(
        "minimum" not in update and "maximum" not in update
        for update in (x_update, y_update, width_update, height_update)
    )
    callbacks = [
        item
        for item in demo.fns.values()
        if item.fn.__name__ == "constrain_roi_inputs"
    ]
    assert len(callbacks) == 4
    assert all(len(callback.inputs) == 5 for callback in callbacks)
    assert all(len(callback.outputs) == 4 for callback in callbacks)


def test_roi_coordinate_accepts_values_beyond_a_previous_image_bound() -> None:
    roi_y = next(
        component
        for component in demo.blocks.values()
        if getattr(component, "label", None) == "y"
    )

    assert roi_y.maximum is None
    assert roi_y.preprocess(1975) == 1975


def test_annotation_gallery_provides_preview_without_separate_image_view() -> None:
    components = demo.config["components"]
    gal_images = [
        item for item in components
        if item["type"] == "gallery"
        and item["props"].get("columns") == 2
        and item["props"].get("show_label") is False
    ]
    assert len(gal_images) == 1, f"expected 1 neighbour gallery, got {len(gal_images)}"
    gallery = gal_images[0]["props"]
    assert gallery.get("elem_id") == "annotation-gallery"
    assert gallery.get("allow_preview") is True
    assert gallery.get("object_fit") == "contain"
    assert gallery.get("height") == "75vh"
    assert gallery.get("columns") == 2
    assert not [
        item
        for item in components
        if item["type"] == "image" and item["props"].get("label") == "当前图片"
    ]
    assert ".preview img {" in demo.css
    assert "margin: auto !important;" in demo.css
    assert "max-height: min(75vh, 900px) !important;" in demo.css
    assert "#annotation-gallery .thumbnail-small {" in demo.css
    assert "width: 84px !important;" in demo.css
    assert "#annotation-gallery .preview .media-button {" in demo.css
    assert "height: calc(100% - 104px) !important;" in demo.css
    buttons = {
        item["props"]["value"]: item["props"]
        for item in components
        if item["type"] == "button"
        and item.get("props", {}).get("value")
        in ("← 上一张", "下一张 →", "下一张未标注 →")
    }
    assert buttons["← 上一张"]["size"] == "lg"
    assert buttons["下一张 →"]["size"] == "lg"
    assert buttons["下一张未标注 →"]["size"] == "lg"
    index = next(
        item
        for item in components
        if item["type"] == "number" and item.get("props", {}).get("placeholder") == "序号"
    )
    assert index["props"]["show_label"] is False


def test_annotation_image_actions_keep_original_view_checkbox_state() -> None:
    components = {item["id"]: item for item in demo.config["components"]}
    assert not [
        item
        for item in components.values()
        if item["type"] == "textbox"
        and "标注者" in item.get("props", {}).get("label", "")
    ]
    original_view = next(
        item["id"]
        for item in components.values()
        if item["type"] == "checkbox"
        and item.get("props", {}).get("label") == "显示原图与热区框"
    )
    # Annotation-area dependencies: 4 named + 7 lambdas = 11 total
    annotation_names = {
        "begin_annotation",
        "next_annotation",
        "_select_neighbour",
        "navigate_annotation",
        "lambda",
        "lambda_1",
        "lambda_2",
        "lambda_3",
        "lambda_4",
        "lambda_5",
        "lambda_6",
    }
    annotation_actions = [
        dep for dep in demo.config["dependencies"]
        if dep.get("api_name") in annotation_names
    ]
    assert len(annotation_actions) == len(annotation_names)
    assert all(original_view in dep["inputs"] for dep in annotation_actions)


def test_next_annotation_navigates_to_the_immediately_following_image(monkeypatch) -> None:
    gallery_result = {
        "gallery": [],
        "gallery_indices": [],
        "index": 8,
        "total": 10,
        "labeled_count": 0,
        "label": None,
        "filename": "frame-9.png",
        "preview_errors": [],
        "roi": {"x": 0, "y": 0, "width": 1, "height": 1},
        "image_size": {"width": 1, "height": 1},
    }
    navigated_indices = []
    monkeypatch.setattr(
        webui_service.WebUIService,
        "navigate",
        lambda self, session_id, index: navigated_indices.append(index),
    )
    monkeypatch.setattr(
        webui_service.WebUIService,
        "get_annotation_session",
        lambda self, session_id: object(),
    )
    monkeypatch.setattr(
        webui_service.WebUIService,
        "render_annotation",
        lambda self, session, *, original: gallery_result,
    )
    callback = next(
        item.fn for item in demo.fns.values() if item.fn.__name__ == "next_annotation"
    )

    callback("session", 8, False)

    assert navigated_indices == [8]


def test_annotation_counter_has_room_for_status_text() -> None:
    counter = next(
        item
        for item in demo.config["components"]
        if item["type"] == "markdown"
        and item["props"].get("elem_id") == "annotation-counter"
    )

    assert counter["props"]["min_width"] == 260
    assert "#annotation-counter p {" in demo.css
    assert "overflow-wrap: anywhere;" in demo.css


def _write_inspection_json(path, with_picture: bool) -> None:
    import json

    payload = {
        "code": 200,
        "msg": "操作成功",
        "data": {
            "deptName": "A10分部",
            "startTime": 1767196800000,
            "inspectionDeviceTypes": [
                {
                    "name": "M1-401机房PDU",
                    "inspectionDevices": [
                        {
                            "id": "dev-1",
                            "name": "M1-401机房PDU",
                            "installationSite": "401机房",
                            "picture": "https://example.com/1.jpg"
                            if with_picture
                            else "",
                        }
                    ],
                }
            ],
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_dataset_preview_reports_grouped_path_and_skip(tmp_path) -> None:
    import json

    mixed = tmp_path / "mixed"
    mixed.mkdir()
    _write_inspection_json(mixed / "with-pic.json", with_picture=True)
    empty = mixed / "empty.json"
    payload = json.loads((mixed / "with-pic.json").read_text(encoding="utf-8"))
    payload["data"]["inspectionDeviceTypes"][0]["inspectionDevices"][0]["picture"] = ""
    empty.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    callback = next(
        item.fn for item in demo.fns.values() if item.fn.__name__ == "preview_json"
    )
    summary, table = callback([], str(mixed))

    assert "data/A10/" in summary
    assert "images_source/" in summary
    assert "无有效图片，将跳过归档与下载" in summary


def test_dataset_download_skips_json_without_valid_pictures(tmp_path, monkeypatch) -> None:
    import pytest
    import gradio as gr

    project_root = tmp_path / "project"
    monkeypatch.setattr(webui_service, "ROOT", project_root)
    empty_json = tmp_path / "empty.json"
    _write_inspection_json(empty_json, with_picture=False)

    callback = next(
        item.fn for item in demo.fns.values() if item.fn.__name__ == "start_download"
    )
    with pytest.raises(gr.Error, match="无有效图片已跳过"):
        list(callback([], str(empty_json), ""))

    # 未创建任何工作区目录（不归档、不下载）。
    assert not (project_root / "data").exists()
