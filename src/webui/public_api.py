"""Stable, documented business APIs exposed by the Gradio application."""

from __future__ import annotations

from typing import Any, Callable, TypedDict, get_type_hints

import gradio as gr
from gradio.data_classes import FileData

from webui_service import WebUIService


class DatasetVersionRecord(TypedDict):
    version: int
    parent: int | None
    created_at: str
    sample_count: int
    source_hashes: list[str]


class DatasetVersionStatus(TypedDict):
    active_version: int
    latest_version: int
    versions: list[DatasetVersionRecord]


class ModelRecord(TypedDict):
    path: str
    dataset: str
    workspace: str
    modified: str
    architecture: str
    input_size: int | None


class ModelList(TypedDict):
    runs_dir: str
    models: list[ModelRecord]


class AnnotationStatus(TypedDict):
    session_id: str
    index: int
    total: int
    filename: str
    image_path: str
    label: str | None
    claimed_by_other: bool
    labeled_count: int
    roi: dict[str, int]
    image_size: dict[str, int]


class AnnotationStartResponse(TypedDict):
    session_id: str
    total: int
    index: int
    workspace_name: str
    input_dir: str
    output_dir: str
    roi_database_path: str
    current: AnnotationStatus


class AnnotationClaimResponse(TypedDict):
    claimed: bool
    current: AnnotationStatus


class AnnotationSaveResponse(TypedDict):
    saved: dict[str, str] | None
    skipped: bool | None
    current: AnnotationStatus


class AnnotationRoiResponse(TypedDict):
    updated_labels: int
    current: AnnotationStatus


class TrainingJob(TypedDict):
    job_id: str
    status: str
    output: str


class TrainingStatus(TypedDict):
    job_id: str
    status: str
    return_code: int | None
    output: str
    log: str


def register_public_apis(
    service: WebUIService, *, include_management: bool = True
) -> None:
    def list_models() -> ModelList:
        """List locally available trained model checkpoints."""
        return service.list_models()

    _register_api(list_models, "list_models", "列出本地已有的训练模型权重。")

    if not include_management:
        return

    def dataset_version_status(dataset_root: str) -> DatasetVersionStatus:
        """Read dataset version history and the currently active version."""
        return service.dataset_version_status(dataset_root)

    def import_labeled_dataset(
        dataset_root: str,
        sharp_files: list[FileData],
        borderline_files: list[FileData],
        defocused_files: list[FileData],
    ) -> DatasetVersionStatus:
        """Import uploaded images by label and create a new dataset version."""
        files_by_label = {
            "sharp": _uploaded_paths(sharp_files),
            "borderline": _uploaded_paths(borderline_files),
            "defocused": _uploaded_paths(defocused_files),
        }
        return service.import_labeled_dataset(dataset_root, files_by_label)

    def activate_dataset_version(
        dataset_root: str, version: int
    ) -> DatasetVersionStatus:
        """Activate an existing dataset version without deleting history."""
        return service.activate_dataset_version(dataset_root, version)

    def start_annotation(
        workspace_root: str, workspace_name: str | None = None
    ) -> AnnotationStartResponse:
        """Start an annotation session and return its first image and session ID."""
        started = service.start_annotation(workspace_root, workspace_name)
        return {**started, "current": service.annotation_status(started["session_id"])}

    def annotation_status(session_id: str) -> AnnotationStatus:
        """Get the current image, label, thermal region, and session progress."""
        return service.annotation_status(session_id)

    def navigate_annotation(
        session_id: str, image_index: int
    ) -> AnnotationStatus:
        """Navigate to a one-based image index in an annotation session."""
        service.navigate(session_id, image_index - 1)
        return service.annotation_status(session_id)

    def claim_next_unlabeled(session_id: str) -> AnnotationClaimResponse:
        """Claim the next available unlabeled image for this session."""
        result = service.claim_next_annotation(session_id)
        return {
            "claimed": result["claimed"],
            "current": service.annotation_status(session_id),
        }

    def save_annotation(
        session_id: str, label: str, advance: bool = True
    ) -> AnnotationSaveResponse:
        """Save the current image label and optionally advance to another image."""
        result = service.save_label(session_id, label, advance)
        return {
            "saved": result["saved"],
            "skipped": result["skipped"],
            "current": service.annotation_status(session_id),
        }

    def update_annotation_roi(
        session_id: str,
        x: int,
        y: int,
        width: int,
        height: int,
        save: bool = False,
    ) -> AnnotationRoiResponse:
        """Preview or save the current image's thermal crop coordinates."""
        result = service.update_roi(session_id, x, y, width, height, save)
        return {
            "updated_labels": result.get("updated_labels", 0),
            "current": service.annotation_status(session_id),
        }

    def start_training(
        dataset_root: str,
        epochs: int = 30,
        architecture: str = "mobilenet_v3_large",
        batch_size: int = 16,
        input_size: int = 224,
        learning_rate: float = 0.0001,
        seed: int = 42,
        device: str = "auto",
        resume: str = "",
        pretrained: bool = True,
    ) -> TrainingJob:
        """Start a background training job for the active dataset version."""
        return service.start_training(
            {
                "dataset_root": dataset_root,
                "epochs": epochs,
                "architecture": architecture,
                "batch_size": batch_size,
                "input_size": input_size,
                "learning_rate": learning_rate,
                "seed": seed,
                "device": device,
                "resume": resume,
                "pretrained": pretrained,
            }
        )

    def training_status(job_id: str) -> TrainingStatus:
        """Get the status, return code, output directory, and recent log of a job."""
        return service.training_status(job_id)

    for fn, name, description in (
        (
            dataset_version_status,
            "dataset_version_status",
            "读取数据集版本历史和当前生效版本。",
        ),
        (
            import_labeled_dataset,
            "import_labeled_dataset",
            "按类别导入图片并创建新的数据集版本。",
        ),
        (
            activate_dataset_version,
            "activate_dataset_version",
            "切换当前生效的数据集版本，不删除历史版本。",
        ),
        (
            start_annotation,
            "start_annotation",
            "启动图片标注会话并返回首张图片信息。",
        ),
        (
            annotation_status,
            "annotation_status",
            "读取标注会话当前图片、标签和热区信息。",
        ),
        (
            navigate_annotation,
            "navigate_annotation",
            "跳转到标注会话中的指定图片序号（从 1 开始）。",
        ),
        (
            claim_next_unlabeled,
            "claim_next_unlabeled",
            "领取当前会话的下一张未标注图片。",
        ),
        (
            save_annotation,
            "save_annotation",
            "保存当前图片标签，并可选择自动前进。",
        ),
        (
            update_annotation_roi,
            "update_annotation_roi",
            "预览或保存当前图片的热区裁剪坐标。",
        ),
        (
            start_training,
            "start_training",
            "为当前生效数据集版本启动后台训练任务。",
        ),
        (
            training_status,
            "training_status",
            "读取训练任务状态和近期日志。",
        ),
    ):
        _register_api(fn, name, description)


def _register_api(fn: Callable[..., Any], name: str, description: str) -> None:
    fn.__annotations__.update(get_type_hints(fn))
    gr.api(fn, api_name=name, api_description=description, queue=False)


def _uploaded_paths(files: list[FileData]) -> list[str]:
    paths = []
    for item in files:
        path = item.get("path") if isinstance(item, dict) else item.path
        if not path:
            raise ValueError("uploaded image is missing its server-side file path")
        paths.append(path)
    return paths
