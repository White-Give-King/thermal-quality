"""「数据集版本」标签页：导入已标注图片并管理数据集快照。"""

from __future__ import annotations

from typing import Any

import gradio as gr

from webui.data_root import DataRootPanel
from webui_service import WebUIService


def _file_paths(files: Any) -> list[str]:
    if files is None:
        return []
    values = files if isinstance(files, (list, tuple)) else [files]
    return [str(getattr(item, "name", item)) for item in values]


def _version_display(status: dict[str, Any]) -> tuple[list[tuple[str, int]], str]:
    versions = status["versions"]
    assert isinstance(versions, list)
    if not versions:
        return [], "尚未创建版本；首次导入会以当前标注为 v0001 基线，并生成新的数据集版本。"
    active_version = int(status["active_version"])
    choices: list[tuple[str, int]] = []
    active_count = 0
    for record in versions:
        if not isinstance(record, dict):
            continue
        version = int(record["version"])
        count = int(record["sample_count"])
        label = f'v{version:04d} · {count} 张'
        training = record.get("training")
        if training:
            states = {"queued": "排队中", "running": "训练中", "completed": "训练完成", "failed": "训练失败"}
            label += f' · {states.get(training["status"], training["status"])}'
        if version == active_version:
            label += "（当前生效）"
            active_count = count
        choices.append((label, version))
    return (
        choices,
        f"当前生效版本：**v{active_version:04d}**，"
        f"共 **{active_count}** 张已标注图片。",
    )


def build_dataset_version_tab(
    service: WebUIService, data_root: DataRootPanel
) -> None:
    gr.Markdown(
        "`data/` 当前内容作为 v1 基础数据集。按类别分别选择图片后导入，"
        "每次提交都会基于当前版本创建新版本并自动排队训练；同时只运行一个训练任务。"
        "每个版本保留独立标注快照；回档只切换当前生效版本，不删除历史版本。",
        elem_classes=["tab-hint"],
    )
    refresh_button = gr.Button("刷新版本列表")
    version_select = gr.Dropdown(
        label="数据集版本",
        choices=[],
        value=None,
        interactive=True,
    )
    version_status = gr.Markdown("点击“刷新版本列表”初始化并查看 `data/` 的基础版本。")
    training_detail = gr.Code(label="所选版本训练状态和日志（刷新版本列表更新）", language="shell", lines=8)
    with gr.Row():
        sharp_files = gr.File(
            label="清晰（sharp）图片",
            file_count="multiple",
            file_types=["image"],
            type="filepath",
        )
        borderline_files = gr.File(
            label="临界（borderline）图片",
            file_count="multiple",
            file_types=["image"],
            type="filepath",
        )
        defocused_files = gr.File(
            label="模糊（defocused）图片",
            file_count="multiple",
            file_types=["image"],
            type="filepath",
        )
    with gr.Row():
        import_button = gr.Button("导入并创建新版本", variant="primary")
        rollback_button = gr.Button("回档到所选版本")

    def refresh_versions(root: str):
        try:
            status = service.dataset_version_status(root)
        except (OSError, ValueError, RuntimeError) as exc:
            raise gr.Error(f"读取数据集版本失败：{exc}") from exc
        choices, message = _version_display(status)
        return gr.Dropdown(choices=choices, value=status["active_version"]), message

    def import_dataset(root: str, sharp: Any, borderline: Any, defocused: Any):
        files_by_label = {
            "sharp": _file_paths(sharp),
            "borderline": _file_paths(borderline),
            "defocused": _file_paths(defocused),
        }
        try:
            status = service.import_labeled_dataset(root, files_by_label)
        except (OSError, ValueError, RuntimeError) as exc:
            raise gr.Error(f"导入数据集失败：{exc}") from exc
        choices, message = _version_display(status)
        job = status["training_job"]
        training_message = {
            "queued": "训练已排队。",
            "running": "训练已启动。",
            "failed": f'训练启动失败：{service.training_status(job["job_id"])["log"]}',
        }[job["status"]]
        return (
            gr.Dropdown(choices=choices, value=status["active_version"]),
            f"导入成功。{training_message} 任务 ID：`{job['job_id']}`。{message}",
            None,
            None,
            None,
        )

    def rollback(root: str, version: int | float | None) -> tuple[Any, str]:
        if version is None or isinstance(version, bool) or int(version) != version:
            raise gr.Error("请选择要回档的数据集版本。")
        try:
            status = service.activate_dataset_version(root, int(version))
        except (OSError, ValueError, RuntimeError) as exc:
            raise gr.Error(f"回档失败：{exc}") from exc
        choices, message = _version_display(status)
        return (
            gr.Dropdown(choices=choices, value=status["active_version"]),
            f"已回档。{message}",
        )

    def show_training(root: str, version: int | float | None) -> str:
        if version is None:
            return "请选择数据集版本。"
        try:
            status = service.dataset_version_status(root)
            record = next((item for item in status["versions"] if item["version"] == version), None)
            if record is None:
                return "数据集版本不存在。"
            job = record.get("training")
            if not job:
                return "该版本在当前服务进程中没有训练任务。"
            details = service.training_status(job["job_id"])
            return (
                f'任务 {job["job_id"]}：{details["status"]}，'
                f'返回码：{details["return_code"]}\n{details["log"]}'
            )
        except (OSError, ValueError, RuntimeError) as exc:
            raise gr.Error(f"读取训练状态失败：{exc}") from exc

    refresh_button.click(
        refresh_versions,
        inputs=data_root.root,
        outputs=[version_select, version_status],
        api_name=False,
    )
    import_button.click(
        import_dataset,
        inputs=[
            data_root.root,
            sharp_files,
            borderline_files,
            defocused_files,
        ],
        api_name=False,
        outputs=[
            version_select,
            version_status,
            sharp_files,
            borderline_files,
            defocused_files,
        ],
    )
    rollback_button.click(
        rollback,
        inputs=[data_root.root, version_select],
        outputs=[version_select, version_status],
        api_name=False,
    )
    version_select.change(
        show_training, inputs=[data_root.root, version_select], outputs=training_detail,
        api_name=False,
    )
    refresh_button.click(
        show_training, inputs=[data_root.root, version_select], outputs=training_detail,
        api_name=False,
    )
