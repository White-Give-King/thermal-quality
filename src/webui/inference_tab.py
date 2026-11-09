"""「图片推理」标签页：模型选择与本地图片批量推理。"""

from __future__ import annotations

from typing import Any

import gradio as gr
import numpy as np

from thermal_focus.webui_service import WebUIService


def list_models(service: WebUIService) -> list[dict[str, Any]]:
    try:
        return service.list_models()["models"]
    except (OSError, ValueError, RuntimeError):
        return []


def model_choices(service: WebUIService) -> list[tuple[str, str]]:
    return [
        (
            f'{item["workspace"]} · {item["architecture"]} · {item["modified"]}',
            item["path"],
        )
        for item in list_models(service)
    ]


def default_model_path(service: WebUIService) -> str | None:
    models = list_models(service)
    return models[0]["path"] if models else None


def describe_model(service: WebUIService, model_path: str | None) -> str:
    if not model_path:
        return "未发现已训练模型；请先在「模型训练」页训练，或直接粘贴权重路径。"
    for item in list_models(service):
        if item["path"] == model_path:
            details = [
                f"**工作区**：`{item['workspace']}`",
                f"**架构**：`{item['architecture']}`",
            ]
            if item.get("modified"):
                details.append(f"**训练时间**：{item['modified']}")
            if item.get("input_size"):
                details.append(f"**输入尺寸**：`{item['input_size']}`")
            return "　·　".join(details)
    return f"自定义模型路径：`{model_path}`"


def build_inference_tab(
    service: WebUIService,
) -> tuple[gr.Dropdown, gr.Markdown]:
    gr.Markdown(
        "选择模型与本地图片，支持多选批量推理。模型分数 = 最高类别 softmax 概率 × 100，"
        "为未校准的原始分数，不代表准确率；低于配置阈值时显示「无法确定」。",
        elem_classes=["tab-hint"],
    )
    initial_model = default_model_path(service)
    with gr.Row():
        model_path = gr.Dropdown(
            label="选择模型（runs 目录下已训练权重，可输入自定义路径）",
            info=(
                "推荐使用训练完成的 best.pt。默认推理参数：居中热区 809 × 1080，"
                "图片最大边长 12000 px，置信度阈值 0.60；模型输入尺寸优先读取权重元数据，"
                "未记录时默认 224。修改 configs/default.yaml 或模型后，以实际配置和权重元数据为准。"
            ),
            choices=model_choices(service),
            value=initial_model,
            allow_custom_value=True,
            scale=4,
        )
        refresh_models_button = gr.Button("刷新模型列表", scale=0, min_width=130)
        prediction_device = gr.Dropdown(
            label="设备",
            choices=["auto", "cuda", "cpu"],
            value="auto",
            info="推荐 auto：CUDA 可用时使用 CUDA，否则自动使用 CPU。",
            scale=1,
            min_width=110,
        )
    model_info = gr.Markdown(describe_model(service, initial_model))
    prediction_files = gr.File(
        label="选择图片（可多选）",
        file_count="multiple",
        file_types=["image"],
        type="filepath",
        height=168,
    )
    with gr.Row(equal_height=True):
        predict_button = gr.Button(
            "开始推理",
            variant="primary",
            scale=0,
            min_width=150,
            size="lg",
        )
        prediction_message = gr.Markdown(elem_classes=["status-text"])
    prediction_gallery = gr.Gallery(
        label="推理预览",
        columns=4,
        object_fit="cover",
        type="numpy",
        allow_preview=True,
    )
    prediction_results = gr.Dataframe(
        label="分类结果",
        headers=["文件名", "预测", "模型分数", "错误"],
        value=[],
        interactive=False,
        wrap=True,
    )

    def refresh_model_dropdown() -> tuple[Any, str]:
        choices = model_choices(service)
        value = choices[0][1] if choices else None
        return gr.Dropdown(choices=choices, value=value), describe_model(service, value)

    def describe_selected_model(model: str) -> str:
        return describe_model(service, model)

    refresh_models_button.click(
        refresh_model_dropdown,
        inputs=[],
        outputs=[model_path, model_info],
    )
    model_path.change(
        describe_selected_model,
        inputs=[model_path],
        outputs=[model_info],
    )

    def predict_images(
        files: list[str] | None, model: str, device: str
    ) -> tuple[str, list[tuple[np.ndarray, str]], list[list[Any]]]:
        if not files:
            raise gr.Error("请先选择至少一张图片。")
        result = service.predict(files, model=model, device=device)
        previews: list[tuple[np.ndarray, str]] = []
        public_results = []
        for item in result["results"]:
            public_results.append(
                {
                    "filename": item["filename"],
                    "prediction": item.get("prediction"),
                    "error": item.get("error"),
                }
            )
            if item.get("annotated_image") is not None:
                label = item["prediction"].get("label_zh", item["prediction"]["status"])
                previews.append(
                    (
                        item["annotated_image"],
                        f'{item["filename"]}: 原图热区框 — {label}',
                    )
                )
                previews.append(
                    (
                        item["thermal_image"],
                        f'{item["filename"]}: 热区裁剪',
                    )
                )
        success_count = sum("prediction" in item for item in public_results)
        rows = [
            [
                item["filename"],
                (item.get("prediction") or {}).get("label_zh", ""),
                (item.get("prediction") or {}).get("model_score", ""),
                item.get("error") or "",
            ]
            for item in public_results
        ]
        return (
            f"完成：{success_count}/{len(public_results)} 张图片获得推理结果。",
            previews,
            rows,
        )

    predict_button.click(
        predict_images,
        inputs=[prediction_files, model_path, prediction_device],
        outputs=[prediction_message, prediction_gallery, prediction_results],
    )

    return model_path, model_info
