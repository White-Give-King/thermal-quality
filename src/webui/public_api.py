"""Stable, documented business APIs exposed by the Gradio application."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, NotRequired, TypedDict, get_type_hints

import gradio as gr
from gradio.context import get_blocks_context
from gradio.data_classes import FileData

from image_io import IMAGE_SUFFIXES, decode_image_bytes
from inference import load_config
from webui.api_docs import describe_api_component
from webui_service import MAX_UPLOAD_BYTES, WebUIService


class DatasetVersionRecord(TypedDict):
    version: int
    parent: int | None
    created_at: str
    sample_count: int
    source_hashes: list[str]
    training: NotRequired[VersionTraining | None]


class VersionTraining(TypedDict):
    job_id: str
    status: str
    return_code: int | None


class DatasetVersionStatus(TypedDict):
    active_version: int
    latest_version: int
    versions: list[DatasetVersionRecord]
    training_job: NotRequired[TrainingJob]


class DatasetImportStatus(DatasetVersionStatus):
    training_job: TrainingJob


class ModelRecord(TypedDict):
    path: str
    dataset: str
    workspace: str
    modified: str
    architecture: str
    input_size: int | None
    dataset_version: str | None


class ModelList(TypedDict):
    runs_dir: str
    models: list[ModelRecord]


class UploadedImage(TypedDict):
    server_path: str
    filename: str
    size_bytes: int


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
    service: WebUIService, *, include_training: bool = False,
    include_annotation: bool = False,
) -> None:
    def list_models() -> ModelList:
        """List locally available trained model checkpoints."""
        return service.list_models()

    def upload_images(files: list[FileData]) -> list[UploadedImage]:
        """Validate images uploaded to the Gradio temporary file cache."""
        paths = _uploaded_paths(files)
        if not paths:
            raise ValueError("select at least one image to upload")
        config = load_config(service.config_path)
        image_config = config.get("image", {})
        results: list[UploadedImage] = []
        for value in paths:
            path = Path(value).expanduser().resolve()
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"unsupported image extension: {path.suffix}")
            if not path.is_file():
                raise FileNotFoundError(f"uploaded image does not exist: {path}")
            size = path.stat().st_size
            if size > MAX_UPLOAD_BYTES:
                raise ValueError(f"{path.name} exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB")
            decode_image_bytes(
                path.read_bytes(),
                max_dimension=int(image_config.get("max_dimension", 12000)),
                float_range=image_config.get("float_range"),
            )
            results.append({
                "server_path": str(path),
                "filename": path.name,
                "size_bytes": size,
            })
        return results

    _register_api(
        upload_images,
        "upload_images",
        "校验上传图片，不会保存到数据集或触发推理、训练。",
        input_descriptions=(
            "必填图片列表，至少一张；每项用 handle_file(\"image.jpg\") 提供。"
            "支持 PNG、JPG/JPEG、BMP、TIF/TIFF；单张不超过 40 MiB，"
            "宽高不超过配置上限（默认 12000 像素），且图片必须可解码。",
        ),
        output_description=(
            "返回图片信息列表，每项字段：server_path：Gradio 临时缓存中的服务端路径；"
            "filename：原文件名；size_bytes：文件大小，单位为字节。临时文件可能被清理，"
            "不代表已保存到数据集。"
        ),
    )

    def dataset_version_status(dataset_root: str) -> DatasetVersionStatus:
        """Read dataset version history and the currently active version."""
        return service.dataset_version_status(dataset_root)

    def import_labeled_dataset(
        dataset_root: str,
        sharp_files: list[FileData],
        borderline_files: list[FileData],
        defocused_files: list[FileData],
    ) -> DatasetImportStatus:
        """Create a dataset version and queue its training with default options."""
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

    _register_api(
        list_models,
        "list_models",
        "列出服务可发现的本地模型权重，按文件修改时间从新到旧排序。",
        output_description=(
            "runs_dir：旧版模型目录路径；未指定扫描目录时，接口同时扫描该目录和项目 data 目录。"
            "models：模型列表。每项包含 path（权重文件绝对路径）、"
            "dataset 和 workspace（数据集/工作区名称）、modified（修改时间）、"
            "architecture（架构）、input_size（输入尺寸或 null）及 dataset_version（版本号或 null）。"
        ),
    )
    _register_api(
        dataset_version_status,
        "dataset_version_status",
        "读取数据集版本历史、当前生效版本及各版本最近的训练状态。",
        input_descriptions=(
            "dataset_root：服务端已存在的数据集根目录绝对路径；WebUI 默认使用项目的 data 目录。",
        ),
        output_description=(
            "active_version：当前生效版本号；latest_version：已创建的最大版本号；versions："
            "版本记录数组。每项字段：version：版本号；parent：父版本号（初始版本为 null）；"
            "created_at：创建时间（ISO 8601）；sample_count：该版本样本数；"
            "source_hashes：导入源图片的 SHA-256 列表；training：该版本最近训练任务，"
            "无任务时为 null，有任务时含 job_id（任务 ID）、status（任务状态）和 "
            "return_code（退出码，运行中为 null）。"
        ),
    )

    file_info = (
        "图片文件列表；无该类别图片时传空列表 []。Python 客户端使用 handle_file 包装文件。"
        "支持 PNG、JPG/JPEG、BMP、TIF/TIFF，每张不超过 40 MiB，宽高不超过配置上限"
        "（默认 12000 像素），且必须可解码。与当前版本或本次提交的图片 SHA-256 重复时会报错。"
    )
    _register_api(
        import_labeled_dataset,
        "import_labeled_dataset",
        "按类别导入图片，创建数据集版本并自动排队训练。",
        input_descriptions=(
            "dataset_root：服务端已存在的数据集根目录绝对路径；三个图片列表均为空时，"
            "目录中须已有待提交的标注。",
            "sharp_files：清晰、对焦正常的图片。" + file_info,
            "borderline_files：清晰度临界或难以判断的图片。" + file_info,
            "defocused_files：明显模糊、失焦的图片。" + file_info,
        ),
        output_description=(
            "active_version：新启用版本号；latest_version：最新版本号；versions：版本记录数组，"
            "每项含 version（版本号）、parent（父版本号，初始版本为 null）、created_at（创建时间）、"
            "sample_count（样本数）、source_hashes（源图片 SHA-256 列表）和 training（该版本最近任务状态，"
            "结构见 /dataset_version_status）；training_job：本次自动训练任务，含 job_id（任务 ID）、"
            "status（queued、running 或 failed）和 output（模型输出目录）。用 /training_status "
            "查询退出码和日志；版本创建成功不代表训练成功。"
        ),
    )
    _register_api(
        activate_dataset_version,
        "activate_dataset_version",
        "切换当前生效的数据集版本，不删除或改写其他版本。",
        input_descriptions=(
            "dataset_root：服务端已存在的数据集根目录绝对路径；WebUI 默认使用项目的 data 目录。",
            "version：要启用的已存在版本号，整数。",
        ),
        output_description=(
            "active_version：切换后的当前版本号；latest_version：最新版本号；"
            "versions：全部版本记录数组，字段定义与 /dataset_version_status 相同。"
        ),
    )
    _register_api(
        training_status,
        "training_status",
        "查询由数据集导入或手动训练启动的后台任务。",
        input_descriptions=(
            "job_id：训练任务 ID，由 /import_labeled_dataset 或 /start_training 返回。",
        ),
        output_description=(
            "job_id：任务 ID；status：queued（排队中）、running（运行中）、"
            "completed（成功）或 failed（失败）；return_code：进程退出码，运行中或排队中为 null；"
            "output：模型输出目录；log：截至本次查询的近期日志文本。"
        ),
    )

    if include_training:
        _register_api(
            start_training,
            "start_training",
            "使用当前生效的数据集版本启动后台训练；同一服务已有训练任务或队列非空时会失败。",
            input_descriptions=(
                "dataset_root：服务端已存在的数据集根目录；训练使用其当前生效版本。",
                "epochs：训练轮数，正整数，默认 30。",
                "architecture：网络架构；可选 mobilenet_v3_small、mobilenet_v3_large、"
                "efficientnet_b0、resnet18；默认 mobilenet_v3_large。",
                "batch_size：每批训练样本数，正整数，默认 16。",
                "input_size：模型输入图像边长（像素），正整数，默认 224。",
                "learning_rate：学习率，有限正数，默认 0.0001。",
                "seed：随机种子整数，默认 42。",
                "device：计算设备，可选 auto、cpu、cuda；默认 auto。",
                "resume：用于续训的 checkpoint 文件路径；文件必须存在；默认空字符串（从头训练）。",
                "pretrained：是否加载预训练权重，布尔值，默认 true。",
            ),
            output_description=(
                "job_id：后台任务 ID；status：任务当前状态（通常为 queued 或 running，"
                "启动失败时为 failed）；output：模型输出目录。使用 /training_status "
                "查询任务状态、退出码和日志。"
            ),
        )

    if include_annotation:
        _register_api(
            start_annotation,
            "start_annotation",
            "在指定图片工作区启动标注会话，并领取首张可标注图片。",
            input_descriptions=(
                "workspace_root：服务端已存在的图片目录绝对路径；直接标注 data 时传 data 目录路径。",
                "workspace_name：可选工作区文件夹名；省略或传 data 时扫描根目录本身，"
                "否则在根目录中查找该工作区。",
            ),
            output_description=(
                "session_id：后续标注接口使用的会话 ID；total：工作区图片总数；"
                "index：首张当前图片序号（从 1 开始）；workspace_name：实际工作区名；"
                "input_dir：原始图片目录；output_dir：标注结果保存目录；"
                "roi_database_path：热区坐标数据库路径；current：当前图片状态，字段定义与 "
                "/annotation_status 相同。"
            ),
        )
        _register_api(
            annotation_status,
            "annotation_status",
            "读取标注会话当前图片、标签、热区和进度。",
            input_descriptions=("session_id：由 /start_annotation 返回的会话 ID。",),
            output_description=(
                "session_id：会话 ID；index：当前图片序号（从 1 开始）；total：图片总数；"
                "filename：原图文件名；image_path：服务端原图路径；"
                "label：已有标签（sharp、borderline、defocused）或 null（未标注）；"
                "claimed_by_other：是否已被其他会话领取；labeled_count：工作区已标注图片数；"
                "roi：热区坐标对象，含左上角 x、y 和宽高 width、height（像素）；"
                "image_size：原图尺寸对象，含 width、height（像素）。"
            ),
        )
        _register_api(
            navigate_annotation,
            "navigate_annotation",
            "跳转到标注会话中的指定图片序号并返回其状态。",
            input_descriptions=(
                "session_id：由 /start_annotation 返回的会话 ID。",
                "image_index：目标图片序号，从 1 开始，最大为 total。",
            ),
            output_description=(
                "跳转后图片状态对象；字段定义与 /annotation_status 相同。"
            ),
        )
        _register_api(
            claim_next_unlabeled,
            "claim_next_unlabeled",
            "为当前会话领取下一张可用的未标注图片；领取会在不同会话间互斥。",
            input_descriptions=("session_id：由 /start_annotation 返回的会话 ID。",),
            output_description=(
                "claimed：是否成功领取下一张未标注图片；current：当前图片状态对象，"
                "字段定义与 /annotation_status 相同。没有可领取图片时 claimed 为 false。"
            ),
        )
        _register_api(
            save_annotation,
            "save_annotation",
            "保存当前图片的清晰度标签，可选择自动领取下一张图片。",
            input_descriptions=(
                "session_id：由 /start_annotation 返回的会话 ID。",
                "label：类别，只能是 sharp（清晰）、borderline（临界）或 defocused（失焦）。",
                "advance：保存后是否自动领取下一张未标注图片，布尔值，默认 true。",
            ),
            output_description=(
                "saved：成功写入的标注记录；非 null 时含 image_path（裁剪图相对路径）、"
                "label（保存的类别）和 source_path（原图路径）；已标注图片跳过时为 null。"
                "skipped：已标注图片被跳过时为 true，正常保存时为 null。"
                "current：当前图片状态对象，字段定义与 /annotation_status 相同。"
            ),
        )
        _register_api(
            update_annotation_roi,
            "update_annotation_roi",
            "预览当前图片热区裁剪坐标，或保存坐标并重裁该图片已有标注。",
            input_descriptions=(
                "session_id：由 /start_annotation 返回的会话 ID。",
                "x：热区左上角横坐标，非负像素值。",
                "y：热区左上角纵坐标，非负像素值。",
                "width：热区宽度，正整数像素值。",
                "height：热区高度，正整数像素值。",
                "save：是否保存坐标；false（默认）只预览，true 会持久化坐标并重裁当前图片已有标注。",
            ),
            output_description=(
                "updated_labels：本次因热区变更而重新裁剪的已有标注数；"
                "current：使用新热区的当前图片状态对象，字段定义与 /annotation_status 相同。"
            ),
        )


def _register_api(
    fn: Callable[..., Any],
    name: str,
    description: str,
    *,
    input_descriptions: tuple[str, ...] = (),
    output_description: str = "",
) -> None:
    fn.__annotations__.update(get_type_hints(fn))
    dependency = gr.api(fn, api_name=name, api_description=description, queue=False)
    if input_descriptions or output_description:
        blocks = get_blocks_context()
        if blocks is None:
            raise RuntimeError("API documentation requires a Gradio Blocks context")
        endpoint = blocks.fns[dependency["id"]]
        if len(endpoint.inputs) != len(input_descriptions) or len(endpoint.outputs) != 1:
            raise ValueError(f"API documentation does not match endpoint: {name}")
        for component, text in zip(endpoint.inputs, input_descriptions, strict=True):
            describe_api_component(component, text)
        if output_description:
            component = endpoint.outputs[0]
            api_info = component.api_info_as_output
            component.api_info_as_output = lambda: {
                **api_info(),
                "description": output_description,
            }


def _uploaded_paths(files: list[FileData]) -> list[str]:
    paths = []
    for item in files:
        path = item.get("path") if isinstance(item, dict) else item.path
        if not path:
            raise ValueError("uploaded image is missing its server-side file path")
        paths.append(path)
    return paths
