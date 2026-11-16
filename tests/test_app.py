from __future__ import annotations

import gradio as gr
from fastapi.testclient import TestClient
from gradio.routes import App

import app as app_entrypoint
import webui_service
from webui.annotation_tab import (
    _bounded_roi,
    _roi_input_updates,
    _selected_gallery_index,
)
from webui.app import create_demo, create_ui

demo = create_demo(enable_annotation=True, enable_training=True)


def test_main_exposes_launched_demo_for_gradio_hot_reload(monkeypatch) -> None:
    class FakeDemo:
        css = "test-css"
        head = "test-head"

        def launch(self, **kwargs) -> None:
            self.launch_kwargs = kwargs

    launched_demo = FakeDemo()
    create_demo_options = {}

    def fake_create_demo(*, enable_training=False, enable_annotation=False):
        create_demo_options["enable_training"] = enable_training
        create_demo_options["enable_annotation"] = enable_annotation
        return launched_demo

    monkeypatch.setattr(
        app_entrypoint,
        "create_demo",
        fake_create_demo,
    )

    app_entrypoint.main([])

    assert app_entrypoint.demo is launched_demo
    assert create_demo_options == {"enable_training": False, "enable_annotation": False}
    assert launched_demo.launch_kwargs == {
        "server_name": "0.0.0.0",
        "server_port": 7860,
        "show_error": True,
        "css": "test-css",
        "head": "test-head",
        "footer_links": ["api"],
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


def test_annotation_page_and_apis_require_startup_flag() -> None:
    annotation_apis = {
        "/start_annotation", "/annotation_status", "/navigate_annotation",
        "/claim_next_unlabeled", "/save_annotation", "/update_annotation_roi",
    }
    for enabled in (False, True):
        ui = create_ui(enable_annotation=enabled)
        tabs = {
            item["props"].get("label")
            for item in ui.config["components"] if item["type"] == "tabitem"
        }
        endpoints = set(ui.get_api_info()["named_endpoints"])
        assert ("图片标注" in tabs) is enabled
        assert (annotation_apis <= endpoints) is enabled
        if not enabled:
            assert not annotation_apis & endpoints
            assert not any(item.fn.__name__ == "begin_annotation" for item in ui.fns.values())
        assert {"/training_status", "/dataset_version_status", "/predict_images"} <= endpoints
        assert "/start_training" not in endpoints


def test_annotation_and_training_startup_options_can_be_combined(monkeypatch) -> None:
    import pytest

    calls = []

    class FakeDemo:
        css = head = ""

        def launch(self, **kwargs):
            pass

    monkeypatch.setattr(app_entrypoint, "create_demo", lambda **kwargs: calls.append(kwargs) or FakeDemo())
    app_entrypoint.main(["--enable-annotation", "--enable-training"])
    assert calls == [{"enable_training": True, "enable_annotation": True}]
    with pytest.raises(SystemExit):
        app_entrypoint.main(["--inference-only"])


def test_training_page_and_manual_api_require_startup_flag() -> None:
    for training in (False, True):
        for annotation in (False, True):
            ui = create_ui(enable_training=training, enable_annotation=annotation)
            tabs = {
                item["props"].get("label")
                for item in ui.config["components"] if item["type"] == "tabitem"
            }
            endpoints = set(ui.get_api_info()["named_endpoints"])
            assert ("模型训练" in tabs) is training
            assert ("/start_training" in endpoints) is training
            assert ("/start_annotation" in endpoints) is annotation
            assert {"数据集版本", "图片推理"} <= tabs
            assert {"/import_labeled_dataset", "/training_status"} <= endpoints


def test_gradio_page_is_served_and_callbacks_are_registered() -> None:
    client = TestClient(App.create_app(demo))

    page = client.get("/")
    assert page.status_code == 200
    assert "红外热图清晰度与对焦质量" in page.text

    component_labels = {
        item["props"].get("label") for item in demo.config["components"]
    }
    assert "工作区子文件夹" not in component_labels
    assert "数据集目录（当前基础数据集）" not in component_labels
    assert {
        "/activate_dataset_version",
        "/annotation_status",
        "/claim_next_unlabeled",
        "/dataset_version_status",
        "/import_labeled_dataset",
        "/list_models",
        "/navigate_annotation",
        "/predict_images",
        "/save_annotation",
        "/start_annotation",
        "/start_training",
        "/training_status",
        "/update_annotation_roi",
        "/upload_images",
    } == set(demo.get_api_info()["named_endpoints"])


def test_inference_automatically_selects_model_and_device() -> None:
    components = {
        item["props"].get("label"): item["props"]
        for item in demo.config["components"]
    }

    assert "选择模型（runs 目录下已训练权重，可输入自定义路径）" not in components
    assert "设备" not in components

    inference_table = next(
        item["props"]
        for item in demo.config["components"]
        if str(item["props"].get("label", "")).startswith("分类结果表（")
    )
    assert inference_table["headers"] == [
        "文件名",
        "Prediction",
        "sharp",
        "borderline",
        "defocused",
        "错误",
    ]


def test_model_information_refreshes_on_load_and_inference_tab_selection() -> None:
    for training in (False, True):
        ui = create_ui(enable_training=training)
        callbacks = [
            item for item in ui.fns.values()
            if getattr(item.fn, "__name__", "") == "refresh_model_info"
        ]
        assert {event for item in callbacks for _, event in item.targets} == {"load", "select"}


def test_training_completion_refreshes_automatic_model_info(tmp_path, monkeypatch) -> None:
    run = tmp_path / "data" / "runs" / "dataset-v0002"
    run.mkdir(parents=True)
    best = run / "best.pt"
    best.write_bytes(b"weights")
    (run / "last.pt").write_bytes(b"weights")
    monkeypatch.setattr(webui_service, "ROOT", tmp_path)
    monkeypatch.setattr(
        webui_service.WebUIService, "start_training",
        lambda self, payload: {"job_id": "job", "output": str(run), "status": "running"},
    )
    monkeypatch.setattr(
        webui_service.WebUIService, "training_status",
        lambda self, job_id: {"status": "completed", "return_code": 0, "log": ""},
    )
    ui = create_ui(enable_training=True)
    train = next(
        item.fn for item in ui.fns.values()
        if getattr(item.fn, "__name__", "") == "train"
    )
    updates = list(train(str(tmp_path / "data"), 1, "resnet18", 1, 32, 0.001, 42, "cpu", "", False))
    assert str(best) in updates[-1][2]


def test_inference_gallery_explains_result_annotations() -> None:
    gallery = next(
        item["props"]
        for item in demo.config["components"]
        if item["props"].get("elem_id") == "prediction-gallery"
    )

    assert "类别和最高概率" in gallery["label"]
    assert "各类别概率" in gallery["label"]
    assert "失败图片不展示" in gallery["label"]


def test_prediction_gallery_hover_frame_covers_full_thumbnail() -> None:
    css = demo.css

    assert "#prediction-gallery .grid-wrap {" in css
    assert "padding-left: 0 !important;" in css
    assert "padding-right: 0 !important;" in css
    assert "#prediction-gallery .thumbnail-lg::after {" in css
    assert "inset: 0;" in css
    assert "pointer-events: none;" in css
    assert "#prediction-gallery .thumbnail-lg:hover::after" in css
    assert "#prediction-gallery .thumbnail-lg.selected::after" in css


def test_inference_button_is_enabled_only_when_images_are_uploaded() -> None:
    button = next(
        item["props"]
        for item in demo.config["components"]
        if item["props"].get("value") == "开始推理"
    )
    callback = next(
        item.fn
        for item in demo.fns.values()
        if item.fn.__name__ == "update_predict_button"
    )

    assert button["interactive"] is False
    assert callback(None)["interactive"] is False
    assert callback([])["interactive"] is False
    assert callback(["frame.png"])["interactive"] is True


def test_inference_results_use_english_labels_and_probability_columns(monkeypatch, tmp_path) -> None:
    annotated_image = object()
    monkeypatch.setattr(webui_service, "ROOT", tmp_path)
    checkpoint = tmp_path / "data" / "runs" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    monkeypatch.setattr(
        webui_service.WebUIService,
        "predict",
        lambda self, files, *, model, device: {
            "results": [
                {
                    "filename": "frame.png",
                    "prediction": {
                        "status": "borderline",
                        "label_zh": "临界",
                        "probabilities": {
                            "sharp": 0.1,
                            "borderline": 0.7,
                            "defocused": 0.2,
                        },
                    },
                    "error": None,
                    "annotated_image": annotated_image,
                }
            ]
        },
    )
    ui = create_ui()
    callback = next(
        item.fn for item in ui.fns.values() if item.fn.__name__ == "predict_api"
    )

    _, previews, rows = callback(["frame.png"])

    assert rows == [
        ["frame.png", "borderline", "10.0%", "70.0%", "20.0%", ""]
    ]
    assert len(previews) == 1
    assert previews[0][0] is annotated_image
    assert previews[0][1] == (
        "frame.png: 临界 (borderline) · sharp 10.0% · borderline 70.0% · defocused 20.0%"
    )


def test_page_inference_uses_latest_best_and_auto_device_each_time(tmp_path, monkeypatch) -> None:
    import os
    import pytest

    monkeypatch.setattr(webui_service, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(
        webui_service.WebUIService, "predict",
        lambda self, files, *, model, device: calls.append((model, device)) or {"results": []},
    )
    ui = create_ui()
    callback = next(item.fn for item in ui.fns.values()
                    if getattr(item.fn, "__name__", "") == "predict_automatically")
    api_callback = next(item.fn for item in ui.fns.values()
                        if getattr(item.fn, "__name__", "") == "predict_api")
    with pytest.raises(Exception, match="未发现"):
        callback(["frame.png"])
    with pytest.raises(Exception, match="未发现"):
        api_callback(["frame.png"])
    older = tmp_path / "data" / "runs" / "dataset-v0001" / "best.pt"
    older.parent.mkdir(parents=True)
    older.write_bytes(b"old")
    os.utime(older, (1000000, 1000000))
    callback(["frame.png"])
    newer = tmp_path / "data" / "runs" / "dataset-v0002" / "best.pt"
    newer.parent.mkdir(parents=True)
    newer.write_bytes(b"new")
    callback(["frame.png"])
    result = api_callback(["frame.png"])
    assert len(result) == 3
    assert calls == [(str(older), "auto"), (str(newer), "auto"), (str(newer), "auto")]


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


def test_default_ui_excludes_annotation_and_training() -> None:
    inference_demo = create_demo()
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
    endpoint = demo.get_api_info()["named_endpoints"]["/predict_images"]
    assert "自动使用最新" in endpoint["description"]

    parameter_descriptions = {
        parameter["parameter_name"]: parameter["python_type"]["description"]
        for parameter in endpoint["parameters"]
    }
    assert "图片文件列表" in parameter_descriptions["files"]
    assert "每张图片最大 40 MiB" in parameter_descriptions["files"]
    assert set(parameter_descriptions) == {"files"}

    return_labels = [item["label"] for item in endpoint["returns"]]
    assert "成功获得推理结果数 / 输入图片总数" in return_labels[0]
    assert "例如：完成：2/3 张图片获得推理结果" in return_labels[0]
    assert "类别和最高概率" in return_labels[1]
    assert "各类别概率" in return_labels[1]
    assert "失败图片不展示" in return_labels[1]
    assert "sharp/borderline/defocused/uncertain" in return_labels[2]
    assert "softmax 概率百分比〔未校准〕" in return_labels[2]
    assert "错误列显示原因" in return_labels[2]


def test_business_apis_are_named_and_page_events_are_not_published() -> None:
    named_endpoints = demo.get_api_info()["named_endpoints"]

    assert {
        "/activate_dataset_version",
        "/annotation_status",
        "/claim_next_unlabeled",
        "/dataset_version_status",
        "/import_labeled_dataset",
        "/list_models",
        "/navigate_annotation",
        "/predict_images",
        "/save_annotation",
        "/start_annotation",
        "/start_training",
        "/training_status",
        "/update_annotation_roi",
        "/upload_images",
    } == set(named_endpoints)

    prediction = named_endpoints["/predict_images"]
    assert len(prediction["parameters"]) == 1
    assert len(prediction["returns"]) == 3
    assert all(
        parameter["python_type"]["description"]
        for parameter in prediction["parameters"]
    )
    assert all(result["label"] for result in prediction["returns"])
    assert all(
        result["type"]
        for endpoint in named_endpoints.values()
        for result in endpoint["returns"]
    )
    annotation_response = named_endpoints["/start_annotation"]["returns"][0]["type"]
    assert "session_id" in annotation_response["properties"]
    assert "current" in annotation_response["properties"]
    import_parameters = named_endpoints["/import_labeled_dataset"]["parameters"]
    assert [
        item["parameter_name"] for item in import_parameters
    ] == [
        "dataset_root",
        "sharp_files",
        "borderline_files",
        "defocused_files",
    ]


def test_all_named_apis_document_parameters_and_responses() -> None:
    named_endpoints = create_ui(
        enable_annotation=True, enable_training=True
    ).get_api_info()["named_endpoints"]

    for endpoint in named_endpoints.values():
        assert endpoint["description"].strip()
        assert endpoint["returns"]
        assert all(
            parameter["python_type"]["description"].strip()
            for parameter in endpoint["parameters"]
        )
        assert all(result["label"].strip() and result["type"] for result in endpoint["returns"])

    training_parameters = {
        parameter["parameter_name"]: parameter["python_type"]["description"]
        for parameter in named_endpoints["/start_training"]["parameters"]
    }
    assert "默认 30" in training_parameters["epochs"]
    assert "mobilenet_v3_small" in training_parameters["architecture"]
    assert "auto、cpu、cuda" in training_parameters["device"]

    annotation_response = named_endpoints["/annotation_status"]["returns"][0][
        "python_type"
    ]["description"]
    assert "claimed_by_other" in annotation_response
    assert "image_size" in annotation_response
    roi_parameters = {
        parameter["parameter_name"]: parameter["python_type"]["description"]
        for parameter in named_endpoints["/update_annotation_roi"]["parameters"]
    }
    assert "只预览" in roi_parameters["save"]
    assert "重裁" in roi_parameters["save"]

    field_descriptions = {
        "/list_models": (
            "runs_dir", "models", "path", "dataset", "workspace", "modified",
            "architecture", "input_size", "dataset_version",
        ),
        "/dataset_version_status": (
            "active_version", "latest_version", "versions", "version", "parent",
            "created_at", "sample_count", "source_hashes", "training",
            "job_id", "status", "return_code",
        ),
        "/import_labeled_dataset": (
            "active_version", "latest_version", "versions", "training_job",
            "job_id", "status", "output",
        ),
        "/activate_dataset_version": (
            "active_version", "latest_version", "versions",
        ),
        "/training_status": (
            "job_id", "status", "return_code", "output", "log",
        ),
        "/start_annotation": (
            "session_id", "total", "index", "workspace_name", "input_dir",
            "output_dir", "roi_database_path", "current",
        ),
        "/annotation_status": (
            "session_id", "index", "total", "filename", "image_path", "label",
            "claimed_by_other", "labeled_count", "roi", "image_size",
        ),
        "/claim_next_unlabeled": ("claimed", "current"),
        "/save_annotation": ("saved", "skipped", "current"),
        "/update_annotation_roi": ("updated_labels", "current"),
    }
    for api_name, field_names in field_descriptions.items():
        description = named_endpoints[api_name]["returns"][0]["python_type"]["description"]
        assert all(field_name in description for field_name in field_names), api_name


def test_import_api_documents_parameters_validation_and_training() -> None:
    endpoint = create_ui().get_api_info()["named_endpoints"]["/import_labeled_dataset"]
    assert endpoint["description"] == "按类别导入图片，创建数据集版本并自动排队训练。"
    parameters = {
        p["parameter_name"]: p["python_type"]["description"]
        for p in endpoint["parameters"]
    }
    assert "服务端" in parameters["dataset_root"]
    assert "待提交的标注" in parameters["dataset_root"]
    for key, label in (
        ("sharp_files", "清晰"),
        ("borderline_files", "临界"),
        ("defocused_files", "模糊"),
    ):
        assert label in parameters[key]
        assert "空列表" in parameters[key]
        assert "40 MiB" in parameters[key]
        assert "SHA-256" in parameters[key]
    return_info = endpoint["returns"][0]
    for text in (
        "active_version", "latest_version", "versions", "version", "parent",
        "created_at", "sample_count", "source_hashes", "training_job",
        "job_id", "status", "output", "/training_status",
    ):
        assert text in return_info["python_type"]["description"]
    result_schema = return_info["type"]
    assert "training_job" in result_schema["required"]
    assert {"job_id", "status", "output"} <= set(
        result_schema["properties"]["training_job"]["properties"]
    )


def test_upload_images_api_validates_files_and_exposes_documentation(tmp_path, monkeypatch) -> None:
    import cv2
    import numpy as np
    import pytest

    ui = create_ui()
    endpoint = ui.get_api_info()["named_endpoints"]["/upload_images"]
    assert [param["parameter_name"] for param in endpoint["parameters"]] == ["files"]
    assert endpoint["description"] == "校验上传图片，不会保存到数据集或触发推理、训练。"
    parameter_description = endpoint["parameters"][0]["python_type"]["description"]
    assert all(text in parameter_description for text in
               ("handle_file", "40 MiB", "可解码"))
    output_description = endpoint["returns"][0]["python_type"]["description"]
    assert all(text in output_description for text in
               ("server_path", "filename", "size_bytes", "可能被清理"))
    callback = next(item.fn for item in ui.fns.values()
                    if getattr(item.fn, "__name__", "") == "upload_images")
    image = tmp_path / "frame.png"
    assert cv2.imwrite(str(image), np.zeros((10, 10, 3), dtype=np.uint8))
    result = callback([{"path": str(image)}])
    assert result == [{"server_path": str(image), "filename": "frame.png",
                       "size_bytes": image.stat().st_size}]
    with pytest.raises(ValueError, match="at least one"):
        callback([])
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        callback([{"path": str(bad)}])
    other = tmp_path / "bad.txt"
    other.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="extension"):
        callback([{"path": str(other)}])
    monkeypatch.setattr("webui.public_api.MAX_UPLOAD_BYTES", 1)
    with pytest.raises(ValueError, match="exceeds"):
        callback([{"path": str(image)}])
    monkeypatch.setattr("webui.public_api.MAX_UPLOAD_BYTES", 40 * 1024 * 1024)
    monkeypatch.setattr("webui.public_api.load_config",
                        lambda _path: {"image": {"max_dimension": 5}})
    with pytest.raises(ValueError, match="exceed"):
        callback([{"path": str(image)}])


def test_webui_uses_one_fixed_data_directory_without_workspace_picker() -> None:
    labels = {
        item["props"].get("label")
        for item in demo.config["components"]
    }

    assert "工作区根目录" not in labels
    assert "工作区子文件夹" not in labels
    assert "数据集目录（当前基础数据集）" not in labels
    root = next(
        component for component in demo.blocks.values()
        if isinstance(component, gr.State)
        and component.value == str(webui_service.ROOT / "data")
    )
    for callback in ("begin_annotation", "refresh_versions", "import_dataset", "train"):
        assert any(
            item.fn.__name__ == callback and root in item.inputs
            for item in demo.fns.values()
        )
    assert any(
        item.fn.__name__ == "begin_annotation"
        for item in demo.fns.values()
    )


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


def test_roi_input_validation_callbacks_are_not_public_endpoints() -> None:
    endpoints = demo.get_api_info()["named_endpoints"]

    assert "/predict_images" in endpoints
    assert not any("constrain_roi_inputs" in endpoint for endpoint in endpoints)


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
    gallery = next(
        item["id"]
        for item in components.values()
        if item["type"] == "gallery"
        and item.get("props", {}).get("elem_id") == "annotation-gallery"
    )
    annotation_actions = [
        dependency
        for dependency in demo.config["dependencies"]
        if gallery in dependency["outputs"]
    ]
    assert annotation_actions
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
