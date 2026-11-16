"""「图片推理」标签页：自动选择模型并批量推理。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np

from decision import LABELS
from image_io import load_image_for
from inference import load_config
from webui.api_docs import describe_api_component
from webui_service import MAX_UPLOAD_BYTES, WebUIService


def list_models(service: WebUIService) -> list[dict[str, Any]]:
    return service.list_models()["models"]


def default_model_path(models: list[dict[str, Any]]) -> str | None:
    for model in models:
        if Path(model["path"]).name == "best.pt":
            return model["path"]
    return None


def describe_model(
    models: list[dict[str, Any]], model_path: str | None
) -> str:
    if not model_path:
        return "未发现已训练的 best.pt 模型；请先训练模型。"
    for item in models:
        if item["path"] == model_path:
            details = [
                f"**数据集**：`{item['dataset']}`",
                f"**架构**：`{item['architecture']}`",
                f"**权重**：`{item['path']}`",
            ]
            if item.get("dataset_version"):
                details.insert(1, f"**数据集版本**：`{item['dataset_version']}`")
            if item.get("modified"):
                details.append(f"**训练时间**：{item['modified']}")
            if item.get("input_size"):
                details.append(f"**输入尺寸**：`{item['input_size']}`")
            return "　·　".join(details)
    return f"自定义模型路径：`{model_path}`"


def build_inference_tab(
    service: WebUIService,
    blocks: gr.Blocks,
    tab: gr.Tab | None = None,
) -> gr.Markdown:
    gr.Markdown(
        "每次推理自动使用最新的 best.pt 模型；CUDA 可用时使用 CUDA，否则使用 CPU。"
        "支持多选批量推理，低于配置置信度阈值时显示「无法确定」。",
        elem_classes=["tab-hint"],
    )
    models = list_models(service)
    initial_model = default_model_path(models)
    model_info = gr.Markdown(describe_model(models, initial_model), label="自动选择的模型")
    api_button = gr.Button(visible=False)
    gr.Markdown(
        "上传一张或多张图片；单张图片最大 40 MiB。",
        elem_classes=["tab-hint"],
    )
    prediction_files = gr.File(
        label="选择图片（可多选）",
        file_count="multiple",
        file_types=["image"],
        type="filepath",
        height=168,
    )
    describe_api_component(
        prediction_files,
        "图片文件列表。支持 PNG、JPG/JPEG、BMP、TIF/TIFF；每张图片最大 40 MiB，"
        "宽或高不能超过配置上限（默认 12000 px）。",
    )
    with gr.Row(equal_height=True):
        predict_button = gr.Button(
            "开始推理",
            variant="primary",
            scale=0,
            min_width=150,
            size="lg",
            interactive=False,
        )
        prediction_message = gr.Markdown(
            label=(
                "处理状态统计（成功获得推理结果数 / 输入图片总数；例如："
                "完成：2/3 张图片获得推理结果）"
            ),
            elem_classes=["status-text"],
        )
        describe_api_component(
            prediction_message,
            "推理处理状态和成功获得推理结果的图片数 / 输入图片总数。",
        )
    gr.Markdown(
        "上传后预览原图；推理完成后，预览图标注推理类别与最高概率，"
        "图库标题显示中文类别及各类别概率。",
        elem_classes=["tab-hint"],
    )
    prediction_gallery = gr.Gallery(
        label=(
            "推理预览图库（上传后显示原图；推理后图片标注类别和最高概率；"
            "标题含中文类别及各类别概率；推理失败图片不展示）"
        ),
        columns=1,
        object_fit="cover",
        type="numpy",
        allow_preview=False,
        height=400,
        elem_id="prediction-gallery",
    )
    describe_api_component(
        prediction_gallery,
        "成功推理图片的标注预览，包含预测类别、最高概率和各类别概率；失败图片不展示。",
    )
    gr.Markdown(
        "每张图片一行，显示预测类别、各类别概率或错误。",
        elem_classes=["tab-hint"],
    )
    prediction_results = gr.Dataframe(
        label=(
            "分类结果表（每张输入图片一行；Prediction 为 sharp/borderline/"
            "defocused/uncertain；sharp、borderline、defocused 列为 softmax 概率百分比"
            "〔未校准〕；失败时概率和预测留空，错误列显示原因）"
        ),
        headers=["文件名", "Prediction", *LABELS, "错误"],
        value=[],
        interactive=False,
        wrap=True,
    )
    describe_api_component(
        prediction_results,
        "逐图片推理结果表；概率为未校准的 softmax 百分比，失败时显示错误原因。",
    )
    def refresh_model_info() -> str:
        models = list_models(service)
        return describe_model(models, default_model_path(models))

    blocks.load(
        refresh_model_info,
        outputs=model_info,
        api_name=False,
    )
    if tab is not None:
        tab.select(
            refresh_model_info,
            outputs=model_info,
            api_name=False,
        )

    def update_uploaded_images(
        files: list[str] | None,
    ) -> tuple[Any, list[str | np.ndarray], str, list[list[Any]]]:
        previews: list[str | np.ndarray] = []
        config = None
        for file in files or []:
            path = Path(file)
            if path.suffix.lower() not in {".tif", ".tiff"}:
                previews.append(file)
                continue
            try:
                if path.stat().st_size > MAX_UPLOAD_BYTES:
                    raise ValueError("图片超过 40 MiB")
                if config is None:
                    config = load_config(service.config_path)
                image = load_image_for(path, config).image
                previews.append(image[:, :, ::-1].copy())
            except (OSError, ValueError, RuntimeError) as exc:
                gr.Warning(f"{path.name} 无法预览：{exc}；仍可尝试推理。")
                previews.append(file)
        return gr.update(interactive=bool(files)), previews, "", []

    prediction_files.change(
        update_uploaded_images,
        inputs=[prediction_files],
        outputs=[predict_button, prediction_gallery, prediction_message, prediction_results],
        api_name=False,
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
                prediction = item["prediction"]
                label = prediction.get("label_zh", prediction["status"])
                probabilities = prediction.get("probabilities")
                probability_details = (
                    " · ".join(
                        f"{class_name} {probabilities[class_name] * 100:.1f}%"
                        for class_name in LABELS
                        if class_name in probabilities
                    )
                    if isinstance(probabilities, dict)
                    else ""
                )
                caption = (
                    f'{item["filename"]}: {label} ({prediction["status"]})'
                )
                if probability_details:
                    caption = f"{caption} · {probability_details}"
                previews.append(
                    (
                        item["annotated_image"],
                        caption,
                    )
                )
        success_count = sum("prediction" in item for item in public_results)
        rows = []
        for item in public_results:
            prediction = item.get("prediction") or {}
            probabilities = prediction.get("probabilities")
            probability_cells = (
                [f"{probabilities[label] * 100:.1f}%" for label in LABELS]
                if isinstance(probabilities, dict)
                else [""] * len(LABELS)
            )
            rows.append(
                [
                    item["filename"],
                    prediction.get("status", ""),
                    *probability_cells,
                    item.get("error") or "",
                ]
            )
        return (
            f"完成：{success_count}/{len(public_results)} 张图片获得推理结果。",
            previews,
            rows,
        )

    def predict_automatically(files: list[str] | None):
        if not files:
            raise gr.Error("请先选择至少一张图片。")
        models = list_models(service)
        model = default_model_path(models)
        if model is None:
            raise gr.Error("未发现已训练的 best.pt 模型，请先训练模型。")
        message, previews, rows = predict_images(files, model, "auto")
        return message, previews, rows, describe_model(models, model)

    def predict_api(
        files: list[str] | None,
    ) -> tuple[str, list[tuple[np.ndarray, str]], list[list[Any]]]:
        message, previews, rows, _ = predict_automatically(files)
        return message, previews, rows

    predict_button.click(
        predict_automatically,
        inputs=[prediction_files],
        outputs=[prediction_message, prediction_gallery, prediction_results, model_info],
        api_name=False,
    )
    api_button.click(
        predict_api,
        inputs=[prediction_files],
        outputs=[prediction_message, prediction_gallery, prediction_results],
        api_name="predict_images",
        api_description="自动使用最新 best.pt 和可用设备，对图片进行热区裁剪和清晰度三分类。",
    )

    return model_info
