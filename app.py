"""Gradio interface for local annotation, inference, and model training."""

from __future__ import annotations

import json
import re
import shutil
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np

from thermal_focus.webui_service import ROOT, WebUIService

service = WebUIService()


def _selected_gallery_index(
    indices: list[int], selected_position: int | tuple[int, ...] | list[int]
) -> int:
    """Map a visible gallery position to the absolute source-image index."""
    position = (
        selected_position[0]
        if isinstance(selected_position, (tuple, list))
        else selected_position
    )
    if not 0 <= position < len(indices):
        raise ValueError("thumbnail position is out of range; refresh the gallery")
    return indices[position]


def _annotation_outputs(session_id: str, *, original: bool = False) -> tuple[Any, ...]:
    session = service.get_annotation_session(session_id)
    result = service.render_annotation(session, original=original)
    preview_errors = result.get("preview_errors", [])
    preview_warning = (
        f' | 无法预览：{", ".join(preview_errors)}' if preview_errors else ""
    )
    roi = result["roi"]
    gallery_indices = result["gallery_indices"]
    return (
        gr.update(
            value=result["gallery"],
            selected_index=(
                gallery_indices.index(result["index"])
                if result["index"] in gallery_indices
                else None
            ),
            preview=True,
        ),
        (
            f'{result["index"] + 1} / {result["total"]} | 已标注 {result["labeled_count"]} 张'
            f' | 当前：{result["label"] or "未标注"} | {result["filename"]}{preview_warning}'
        ),
        result["index"] + 1,
        roi["x"],
        roi["y"],
        roi["width"],
        roi["height"],
        gallery_indices,
    )


def _list_models() -> list[dict[str, Any]]:
    try:
        return service.list_models()["models"]
    except (OSError, ValueError, RuntimeError):
        return []


def _model_choices() -> list[tuple[str, str]]:
    return [
        (
            f'{item["workspace"]} · {item["architecture"]} · {item["modified"]}',
            item["path"],
        )
        for item in _list_models()
    ]


def _default_model_path() -> str | None:
    models = _list_models()
    return models[0]["path"] if models else None


def _describe_model(model_path: str | None) -> str:
    if not model_path:
        return "未发现已训练模型；请先在「模型训练」页训练，或直接粘贴权重路径。"
    for item in _list_models():
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


def create_ui() -> gr.Blocks:
    """Create the Gradio interface backed by the in-process service layer."""
    with gr.Blocks(
        title="红外热图清晰度与对焦质量",
    ) as demo:
        gr.Markdown("# 红外热图清晰度与对焦质量", elem_classes=["app-title"])
        gr.Markdown(
            "本地红外图像对焦质量的标注、推理与训练工具",
            elem_classes=["app-subtitle"],
        )
        workspace_records = gr.State([])
        annotation_session = gr.State("")
        gallery_indices_state = gr.State([])

        with gr.Group(elem_classes=["workspace-panel"]):
            with gr.Row(equal_height=True):
                workspace_root = gr.Textbox(
                    label="工作区根目录",
                    value=str(ROOT / "data"),
                    scale=4,
                )
                workspace_select = gr.Dropdown(
                    label="工作区子文件夹",
                    choices=[],
                    allow_custom_value=False,
                    scale=2,
                )
                scan_button = gr.Button(
                    "扫描", variant="primary", scale=0, min_width=110, size="lg"
                )
            with gr.Row(equal_height=True):
                workspace_message = gr.Markdown(
                    "点击「扫描」发现工作区，选择子文件夹后自动开始标注。",
                    elem_classes=["workspace-status"],
                    scale=0,
                )
                workspace_summary = gr.Markdown(
                    "此处显示所选工作区的图片数量与输出目录。",
                    elem_classes=["workspace-summary"],
                )

        def describe_workspace(record: dict[str, Any]) -> str:
            if not record:
                return ""
            return (
                f'**图片（{record["image_count"]} 张）**：`{record["image_dir"]}`  \n'
                f'**标注**：`data/{record["name"]}/`　·　'
                f'**训练输出**：`runs/{record["name"]}/`'
            )

        def scan_workspaces(root: str) -> tuple[Any, ...]:
            try:
                result = service.list_workspaces(root)
                records = result["workspaces"]
                choices = [item["name"] for item in records]
                preferred = "A7-10-19"
                selected = preferred if preferred in choices else (choices[0] if choices else None)
                record = next((item for item in records if item["name"] == selected), {})
                message = f"发现 {len(records)} 个工作区。" if records else "未找到含图片的子文件夹。"
                return (
                    records,
                    gr.Dropdown(choices=choices, value=selected),
                    message,
                    describe_workspace(record),
                )
            except (OSError, ValueError, RuntimeError) as exc:
                return [], gr.Dropdown(choices=[], value=None), f"扫描失败：{exc}", ""

        def choose_workspace(name: str, records: list[dict[str, Any]]) -> str:
            record = next((item for item in records if item["name"] == name), {})
            return describe_workspace(record)

        scan_button.click(
            scan_workspaces,
            inputs=workspace_root,
            outputs=[
                workspace_records,
                workspace_select,
                workspace_message,
                workspace_summary,
            ],
        )
        workspace_select.change(
            choose_workspace,
            inputs=[workspace_select, workspace_records],
            outputs=workspace_summary,
        )

        with gr.Tabs():
            with gr.Tab("数据集"):
                gr.Markdown(
                    "上传 JSON 巡检数据文件，或使用工作区路径。每个任务归档为 "
                    "`data/<任务名>/`：`<任务名>.json` 与 `images_source/` 原图，支持断点续传。"
                    "多选上传时按选中顺序归档，多个任务并行下载，同时下载数可随时调整。",
                    elem_classes=["tab-hint"],
                )
                with gr.Row():
                    json_files = gr.File(
                        label="上传 JSON 文件（可多选）",
                        file_count="multiple",
                        file_types=[".json"],
                        type="filepath",
                        height=140,
                    )
                    json_path_input = gr.Textbox(
                        label="或直接输入 JSON 文件/目录路径",
                        placeholder="例如：data/A7-2026-10-19-08-44-50/A7-2026-10-19-08-44-50.json 或 data/",
                        value=str(ROOT / "data"),
                        scale=2,
                    )
                dataset_summary = gr.Markdown(
                    "选择 JSON 后自动展示统计信息。",
                    elem_classes=["workspace-summary"],
                )
                dataset_table = gr.Dataframe(
                    label="设备类型分布",
                    headers=["部门", "设备类型", "有图片的设备数", "空图设备数"],
                    value=[],
                    interactive=False,
                    wrap=True,
                )
                with gr.Row():
                    download_output_dir = gr.Textbox(
                        label="输出根目录（留空则使用 data/）",
                        placeholder="可选；保存为 <根目录>/<任务名>/（JSON 与 images_source/）",
                        scale=3,
                    )
                    download_concurrency = gr.Slider(
                        label="同时下载的任务数（下载中可拖动调整）",
                        minimum=1,
                        maximum=8,
                        step=1,
                        value=3,
                        scale=2,
                    )
                    download_button = gr.Button(
                        "归档 JSON 并开始下载", variant="primary", min_width=170, scale=0
                    )
                download_status_msg = gr.Markdown(
                    "就绪。",
                    elem_classes=["status-text"],
                )
                download_log = gr.Code(
                    label="实时日志",
                    language="shell",
                    lines=12,
                    max_lines=12,
                    wrap_lines=True,
                )

                # ---------- JSON 解析 ----------

                def _resolve_json_input(
                    files: list[str] | None, path_text: str
                ) -> list[str]:
                    """有序输入：多选上传保持选中顺序，否则使用路径文本。"""
                    if files:
                        return list(files)
                    path_value = str(path_text or "").strip()
                    return [path_value] if path_value else []

                def _collect_json_files(json_inputs: list[str]) -> list[Path]:
                    """按输入顺序收集 JSON 文件；目录按文件名排序展开。"""
                    collected: list[Path] = []
                    for item in json_inputs:
                        p = Path(item)
                        if p.is_file():
                            collected.append(p)
                        elif p.is_dir():
                            collected.extend(
                                sorted(f for f in p.rglob("*.json") if f.is_file())
                            )
                    return collected

                def _load_doc(path: Path) -> dict[str, Any] | None:
                    try:
                        with path.open("r", encoding="utf-8-sig") as f:
                            doc = json.load(f)
                    except (OSError, json.JSONDecodeError):
                        return None
                    return doc if isinstance(doc, dict) else None

                def _dept_prefix(dept: str) -> str:
                    """部门名开头的字母/数字部分，如 A7分部 -> A7。"""
                    match = re.match(r"[A-Za-z0-9]+", str(dept or ""))
                    return match.group(0) if match else "dataset"

                def _run_name(doc: dict[str, Any] | None) -> str | None:
                    """任务名：部门前缀-年-月-日-时-分-秒（开始时间）。"""
                    data = doc.get("data") if isinstance(doc, dict) else None
                    if not isinstance(data, dict):
                        return None
                    prefix = _dept_prefix(str(data.get("deptName") or ""))
                    start = data.get("startTime")
                    if isinstance(start, (int, float)):
                        # 毫秒时间戳
                        dt = datetime.fromtimestamp(float(start) / 1000)
                        return f"{prefix}-{dt.strftime('%Y-%m-%d-%H-%M-%S')}"
                    return prefix

                def _parse_one_json(path: Path) -> dict[str, Any] | None:
                    doc = _load_doc(path)
                    if doc is None:
                        return None
                    data = doc.get("data")
                    if not isinstance(data, dict):
                        return None
                    dept = str(data.get("deptName") or "")
                    types = data.get("inspectionDeviceTypes", [])
                    if not isinstance(types, list):
                        return None
                    rows: list[list[Any]] = []
                    with_picture = 0
                    empty_picture = 0
                    for device_type in types:
                        if not isinstance(device_type, dict):
                            continue
                        type_name = str(device_type.get("name") or "未分类")
                        devices = device_type.get("inspectionDevices", [])
                        if not isinstance(devices, list):
                            continue
                        has_pic = 0
                        no_pic = 0
                        for d in devices:
                            if not isinstance(d, dict):
                                continue
                            pic = d.get("picture")
                            if isinstance(pic, str) and pic.strip():
                                has_pic += 1
                            else:
                                no_pic += 1
                        with_picture += has_pic
                        empty_picture += no_pic
                        rows.append([dept or "未知", type_name, has_pic, no_pic])
                    return {
                        "path": path,
                        "dept": dept,
                        "run_name": _run_name(doc),
                        "rows": rows,
                        "with_picture": with_picture,
                        "empty_picture": empty_picture,
                    }

                def preview_json(
                    files: list[str] | None, path_text: str
                ) -> tuple[str, list[list[Any]]]:
                    json_input = _resolve_json_input(files, path_text)
                    if not json_input:
                        return (
                            "请上传 JSON 文件或填写 JSON 路径。",
                            [],
                        )
                    json_paths = _collect_json_files(json_input)
                    if not json_paths:
                        return (
                            "未找到 JSON 文件，请检查上传文件或路径是否正确。",
                            [],
                        )
                    summaries = []
                    all_rows: list[list[Any]] = []
                    grand_pic = grand_empty = 0
                    for p in json_paths:
                        info = _parse_one_json(p)
                        if info is None:
                            summaries.append(f"- `{p.name}`：解析失败")
                            continue
                        run_name = info["run_name"] or p.stem
                        summaries.append(
                            f"- **{info['dept'] or '未知'}**（`{p.name}`）："
                            f"{info['with_picture']} 张有图，{info['empty_picture']} 张无图  \n"
                            f"  JSON 保存至 `data/{run_name}/{run_name}.json`，"
                            f"图片下载至 `data/{run_name}/images_source/`"
                        )
                        all_rows.extend(info["rows"])
                        grand_pic += info["with_picture"]
                        grand_empty += info["empty_picture"]
                    if not all_rows:
                        return (
                            "无法解析所选 JSON 文件，请确认 `data.inspectionDeviceTypes` 结构存在。",
                            [],
                        )
                    header = (
                        f"**共 {len(json_paths)} 个 JSON 文件，合计 {grand_pic} 张待下载图片**"
                    )
                    body = "\n".join(summaries)
                    return f"{header}\n\n{body}", all_rows

                # ---------- JSON 归档与输出目录 ----------

                DATA_ROOT = ROOT / "data"

                def _task_dir(run_name: str, user_root: str) -> Path:
                    base = str(user_root or "").strip()
                    root = Path(base).expanduser() if base else DATA_ROOT
                    return root / run_name

                def _archive_json(src: Path, task_dir: Path, run_name: str) -> Path:
                    """把 JSON 归档到 <任务目录>/<任务名>.json。"""
                    task_dir.mkdir(parents=True, exist_ok=True)
                    target = task_dir / f"{run_name}.json"
                    try:
                        if src.resolve() == target.resolve():
                            return target
                        shutil.copy2(src, target)
                    except OSError as exc:
                        raise gr.Error(f"保存 JSON 失败：{exc}") from exc
                    return target

                # ---------- 下载 ----------

                # 并发上限：滑杆实时写入，调度循环每轮都会重新读取
                concurrency_state = {"value": int(download_concurrency.value)}

                def _update_concurrency(value: float) -> None:
                    concurrency_state["value"] = max(1, int(value))

                def _collect_prepared(
                    json_inputs: list[str], output_root: str
                ) -> list[tuple[Path, str, Path]]:
                    prepared: list[tuple[Path, str, Path]] = []
                    seen: set[str] = set()
                    for p in _collect_json_files(json_inputs):
                        info = _parse_one_json(p)
                        if info is None:
                            continue
                        run_name = info["run_name"] or p.stem
                        if run_name in seen:
                            continue
                        seen.add(run_name)
                        task_dir = _task_dir(run_name, output_root)
                        archived = _archive_json(p, task_dir, run_name)
                        prepared.append((archived, run_name, task_dir / "images_source"))
                    return prepared

                def run_download_job(
                    files: list[str] | None,
                    path_text: str,
                    output_dir: str,
                ):
                    json_input = _resolve_json_input(files, path_text)
                    if not json_input:
                        raise gr.Error("请选择 JSON 文件或填写路径。")
                    prepared = _collect_prepared(json_input, output_dir)
                    if not prepared:
                        raise gr.Error("未找到可解析的 JSON 文件，请检查所选文件或路径。")

                    total = len(prepared)
                    prefix = ""
                    queue: deque[tuple[int, tuple[Path, str, Path]]] = deque(
                        enumerate(prepared, 1)
                    )
                    running: dict[str, tuple[int, str]] = {}  # job_id -> (序号, 任务名)
                    finished = 0
                    state_texts = {
                        "running": "运行中",
                        "completed": "已完成",
                        "failed": "失败",
                    }

                    def _launch_ready() -> None:
                        """按当前并发上限补发新任务（上限每轮实时读取，可下载中调整）。"""
                        nonlocal prefix
                        while queue and len(running) < concurrency_state["value"]:
                            index, (json_path, run_name, images_dir) = queue.popleft()
                            prefix += (
                                f"\n===== [{index}/{total}] {run_name} =====\n"
                                f"JSON：{json_path}\n输出：{images_dir}\n"
                            )
                            try:
                                started = service.start_download(
                                    str(json_path),
                                    output_dir=str(images_dir),
                                    dry_run=False,
                                )
                            except (RuntimeError, OSError) as exc:
                                raise gr.Error(f"启动下载失败：{exc}") from exc
                            running[started["job_id"]] = (index, run_name)

                    _launch_ready()
                    while running:
                        live_logs = ""
                        for job_id, (index, run_name) in list(running.items()):
                            status = service.download_status(job_id)
                            state_text = state_texts.get(
                                status["status"], status["status"]
                            )
                            log_tail = status.get("log", "")
                            if status["status"] == "running":
                                live_logs += (
                                    f"\n--- [{index}/{total}] {run_name}：{state_text} ---\n"
                                    f"{log_tail}"
                                )
                                continue
                            prefix += (
                                f"{log_tail.rstrip()}\n"
                                f"--- [{index}/{total}] {run_name}：{state_text} ---\n"
                            )
                            del running[job_id]
                            finished += 1
                        _launch_ready()
                        if running:
                            names = "、".join(run_name for _, run_name in running.values())
                            yield (
                                f"进度 {finished}/{total}；正在下载 {len(running)} 个"
                                f"（当前上限 {concurrency_state['value']}）：{names}",
                                prefix + live_logs,
                            )
                            time.sleep(1.5)

                    yield (
                        f"全部完成：共 {total} 个 JSON，图片已保存到对应任务目录的 `images_source/`。",
                        prefix,
                    )

                def start_download(files: list[str] | None, path_text: str, out_dir: str):
                    yield from run_download_job(files, path_text, out_dir)

                # ---------- 事件绑定 ----------

                preview_inputs = [json_files, json_path_input]
                preview_outputs = [dataset_summary, dataset_table]
                demo.load(
                    preview_json,
                    inputs=preview_inputs,
                    outputs=preview_outputs,
                )
                json_files.change(
                    preview_json,
                    inputs=preview_inputs,
                    outputs=preview_outputs,
                )
                json_path_input.change(
                    preview_json,
                    inputs=preview_inputs,
                    outputs=preview_outputs,
                )
                download_concurrency.change(
                    _update_concurrency,
                    inputs=download_concurrency,
                    outputs=[],
                )
                download_button.click(
                    start_download,
                    inputs=[json_files, json_path_input, download_output_dir],
                    outputs=[download_status_msg, download_log],
                )

            with gr.Tab("图片标注"):
                gr.Markdown(
                    "选择上方工作区后自动载入原图；点击下方缩略图可放大预览，"
                    "点击「清晰 / 临界 / 模糊」保存标注并切换到下一张，"
                    "标签与热区裁剪图写入 `data/<工作区名>/`。",
                    elem_classes=["tab-hint"],
                )
                annotation_gallery = gr.Gallery(
                    label="邻近图片",
                    columns=8,
                    rows=1,
                    type="numpy",
                    show_label=False,
                    height=420,
                    object_fit="contain",
                    allow_preview=True,
                )
                annotation_message = gr.Markdown(elem_classes=["status-text"])
                with gr.Row(equal_height=True, elem_classes=["nav-row"]):
                    previous_button = gr.Button(
                        "← 上一张",
                        size="lg",
                        scale=0,
                        min_width=110,
                        interactive=False,
                    )
                    annotation_index = gr.Number(
                        label="跳转到图片序号",
                        value=1,
                        precision=0,
                        show_label=False,
                        placeholder="序号",
                        container=False,
                        scale=0,
                        min_width=100,
                    )
                    next_button = gr.Button(
                        "下一张 →",
                        size="lg",
                        scale=0,
                        min_width=110,
                        interactive=False,
                    )
                    quick_sharp = gr.Button(
                        "清晰",
                        variant="primary",
                        size="lg",
                        scale=0,
                        min_width=110,
                        interactive=False,
                        elem_id="btn-sharp",
                    )
                    quick_borderline = gr.Button(
                        "临界",
                        variant="secondary",
                        size="lg",
                        scale=0,
                        min_width=110,
                        interactive=False,
                        elem_id="btn-borderline",
                    )
                    quick_defocused = gr.Button(
                        "模糊",
                        variant="stop",
                        size="lg",
                        scale=0,
                        min_width=110,
                        interactive=False,
                        elem_id="btn-defocused",
                    )
                    annotation_counter = gr.Markdown(
                        "选择工作区后显示进度。",
                        elem_classes=["status-text"],
                    )
                    original_view = gr.Checkbox(
                        label="显示原图与热区框",
                        value=False,
                        scale=0,
                        min_width=170,
                        elem_classes=["plain-checkbox"],
                    )
                with gr.Accordion("调整热区裁剪坐标", open=False):
                    with gr.Row():
                        roi_x = gr.Number(label="x", value=0, precision=0)
                        roi_y = gr.Number(label="y", value=0, precision=0)
                        roi_width = gr.Number(label="width", value=1, precision=0)
                        roi_height = gr.Number(label="height", value=1, precision=0)
                    with gr.Row(equal_height=True):
                        apply_roi = gr.Button(
                            "应用坐标预览",
                            scale=0,
                            min_width=150,
                            interactive=False,
                        )
                        save_roi = gr.Button(
                            "保存配置并更新已标注样本",
                            variant="primary",
                            scale=0,
                            min_width=220,
                            interactive=False,
                        )
                        roi_message = gr.Markdown(elem_classes=["status-text"])

                annotation_view_outputs = [
                    annotation_gallery,
                    annotation_counter,
                    annotation_index,
                    roi_x,
                    roi_y,
                    roi_width,
                    roi_height,
                    gallery_indices_state,
                ]

                _no_session_annotation = ([], "请先在上方选择工作区。", 1, 0, 0, 1, 1, [])

                annotation_buttons = [
                    previous_button,
                    next_button,
                    quick_sharp,
                    quick_borderline,
                    quick_defocused,
                    apply_roi,
                    save_roi,
                ]

                def _annotation_buttons_update(enabled: bool) -> tuple[Any, ...]:
                    return tuple(gr.update(interactive=enabled) for _ in annotation_buttons)

                def begin_annotation(
                    root: str, name: str, original: bool
                ) -> tuple[Any, ...]:
                    try:
                        result = service.start_annotation(root, name)
                        return (
                            result["session_id"],
                            f'已载入 {result["total"]} 张图片（按需读取），输出目录：{result["output_dir"]}',
                            *_annotation_outputs(result["session_id"], original=original),
                            *_annotation_buttons_update(True),
                        )
                    except (OSError, ValueError, RuntimeError) as exc:
                        return (
                            "",
                            f"启动失败：{exc}",
                            *_no_session_annotation,
                            *_annotation_buttons_update(False),
                        )

                workspace_select.change(
                    begin_annotation,
                    inputs=[workspace_root, workspace_select, original_view],
                    outputs=[
                        annotation_session,
                        annotation_message,
                        *annotation_view_outputs,
                        *annotation_buttons,
                    ],
                )

                def navigate_annotation(
                    session_id: str, index: float, original: bool = False
                ) -> tuple[Any, ...]:
                    if not session_id:
                        return _no_session_annotation
                    try:
                        service.navigate(session_id, int(index) - 1)
                        return _annotation_outputs(session_id, original=original)
                    except (OSError, ValueError, RuntimeError) as exc:
                        return ([], f"跳转失败：{exc}", index, 0, 0, 1, 1, [])

                previous_button.click(
                    lambda session, index, original: navigate_annotation(
                        session, max(1, index - 1), original
                    ),
                    inputs=[annotation_session, annotation_index, original_view],
                    outputs=annotation_view_outputs,
                )
                next_button.click(
                    lambda session, index, original: navigate_annotation(
                        session, index + 1, original
                    ),
                    inputs=[annotation_session, annotation_index, original_view],
                    outputs=annotation_view_outputs,
                )

                def _select_neighbour(
                    session_id: str,
                    indices: list[int],
                    original: bool,
                    event: gr.SelectData,
                ) -> tuple[Any, ...]:
                    if not session_id:
                        raise gr.Error("请先在上方选择工作区。")
                    try:
                        absolute_index = _selected_gallery_index(indices, event.index)
                    except ValueError as exc:
                        raise gr.Error(str(exc)) from exc
                    outputs = navigate_annotation(
                        session_id, absolute_index + 1, original
                    )
                    # Keep the strip and its index mapping untouched so the
                    # gallery preview opened by the click is not dismissed.
                    return (gr.update(), *outputs[1:-1], indices)

                annotation_gallery.select(
                    _select_neighbour,
                    inputs=[annotation_session, gallery_indices_state, original_view],
                    outputs=annotation_view_outputs,
                )
                annotation_index.submit(
                    navigate_annotation,
                    inputs=[annotation_session, annotation_index, original_view],
                    outputs=annotation_view_outputs,
                )
                original_view.change(
                    lambda session, original: _annotation_outputs(session, original=original)
                    if session
                    else _no_session_annotation,
                    inputs=[annotation_session, original_view],
                    outputs=annotation_view_outputs,
                )

                def label_image(
                    session_id: str, label: str, original: bool = False
                ) -> tuple[Any, ...]:
                    if not session_id:
                        raise gr.Error("请先在上方选择工作区。")
                    service.save_label(session_id, label, advance=True)
                    return _annotation_outputs(session_id, original=original)

                quick_sharp.click(
                    lambda session, original: label_image(session, "sharp", original),
                    inputs=[annotation_session, original_view],
                    outputs=annotation_view_outputs,
                )
                quick_borderline.click(
                    lambda session, original: label_image(
                        session, "borderline", original
                    ),
                    inputs=[annotation_session, original_view],
                    outputs=annotation_view_outputs,
                )
                quick_defocused.click(
                    lambda session, original: label_image(
                        session, "defocused", original
                    ),
                    inputs=[annotation_session, original_view],
                    outputs=annotation_view_outputs,
                )

                def update_roi(
                    session_id: str,
                    x: float,
                    y: float,
                    width: float,
                    height: float,
                    save: bool,
                    original: bool,
                ) -> tuple[Any, ...]:
                    if not session_id:
                        raise gr.Error("请先在上方选择工作区。")
                    result = service.update_roi(
                        session_id,
                        int(x),
                        int(y),
                        int(width),
                        int(height),
                        save=save,
                    )
                    message = (
                        f'热区已保存；更新了 {result["updated_labels"]} 个已标注样本。'
                        if save
                        else "热区预览已应用，尚未写入配置。"
                    )
                    return (
                        *_annotation_outputs(session_id, original=original),
                        message,
                    )

                roi_inputs = [
                    annotation_session,
                    roi_x,
                    roi_y,
                    roi_width,
                    roi_height,
                    original_view,
                ]
                apply_roi.click(
                    lambda session, x, y, width, height, original: update_roi(
                        session, x, y, width, height, False, original
                    ),
                    inputs=roi_inputs,
                    outputs=[*annotation_view_outputs, roi_message],
                )
                save_roi.click(
                    lambda session, x, y, width, height, original: update_roi(
                        session, x, y, width, height, True, original
                    ),
                    inputs=roi_inputs,
                    outputs=[*annotation_view_outputs, roi_message],
                )

            with gr.Tab("图片推理"):
                gr.Markdown(
                    "选择模型与本地图片，支持多选批量推理。模型分数 = 最高类别 softmax 概率 × 100，"
                    "为未校准的原始分数，不代表准确率；低于配置阈值时显示「无法确定」。",
                    elem_classes=["tab-hint"],
                )
                initial_model = _default_model_path()
                with gr.Row():
                    model_path = gr.Dropdown(
                        label="选择模型（runs 目录下已训练权重，可输入自定义路径）",
                        choices=_model_choices(),
                        value=initial_model,
                        allow_custom_value=True,
                        scale=4,
                    )
                    refresh_models_button = gr.Button(
                        "刷新模型列表", scale=0, min_width=130
                    )
                    prediction_device = gr.Dropdown(
                        label="设备",
                        choices=["auto", "cuda", "cpu"],
                        value="auto",
                        scale=1,
                        min_width=110,
                    )
                model_info = gr.Markdown(_describe_model(initial_model))
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
                    choices = _model_choices()
                    value = choices[0][1] if choices else None
                    return gr.Dropdown(choices=choices, value=value), _describe_model(value)

                refresh_models_button.click(
                    refresh_model_dropdown,
                    inputs=[],
                    outputs=[model_path, model_info],
                )
                model_path.change(
                    _describe_model,
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

            with gr.Tab("模型训练"):
                gr.Markdown(
                    "自动读取上方所选工作区的 `labels.csv` 进行训练，模型输出到该工作区的 `runs/` 目录。",
                    elem_classes=["tab-hint"],
                )
                with gr.Row():
                    epochs = gr.Number(label="Epoch", value=20, precision=0)
                    architecture = gr.Dropdown(
                        label="模型架构",
                        choices=[
                            "mobilenet_v3_small",
                            "mobilenet_v3_large",
                            "efficientnet_b0",
                            "resnet18",
                        ],
                        value="mobilenet_v3_small",
                        scale=2,
                    )
                    batch_size = gr.Number(label="Batch size", value=16, precision=0)
                    input_size = gr.Number(label="输入尺寸", value=224, precision=0)
                with gr.Row():
                    learning_rate = gr.Number(label="学习率", value=0.0001)
                    seed = gr.Number(label="随机种子", value=42, precision=0)
                    training_device = gr.Dropdown(
                        label="训练设备", choices=["auto", "cuda", "cpu"], value="auto"
                    )
                with gr.Row(equal_height=True):
                    resume_checkpoint = gr.Textbox(
                        label="继续训练权重（可选）",
                        placeholder="留空则从头训练",
                        scale=3,
                    )
                    use_pretrained = gr.Checkbox(
                        label="显式下载 torchvision 预训练权重（需要网络）",
                        value=False,
                        scale=1,
                        min_width=300,
                    )
                with gr.Row(equal_height=True):
                    train_button = gr.Button(
                        "开始训练",
                        variant="primary",
                        scale=0,
                        min_width=150,
                        size="lg",
                    )
                    training_message = gr.Markdown(
                        "尚未启动训练。", elem_classes=["status-text"]
                    )
                training_log = gr.Code(
                    label="训练日志",
                    language="shell",
                    lines=12,
                    max_lines=12,
                    wrap_lines=True,
                )

                def train(
                    name: str,
                    epoch_count: float,
                    model_architecture: str,
                    batch: float,
                    image_size: float,
                    lr: float,
                    random_seed: float,
                    device: str,
                    resume: str,
                    pretrained: bool,
                ):
                    if not name:
                        raise gr.Error("请先扫描并选择工作区。")
                    try:
                        started = service.start_training(
                            {
                                "workspace_name": name,
                                "epochs": int(epoch_count),
                                "architecture": model_architecture,
                                "batch_size": int(batch),
                                "input_size": int(image_size),
                                "learning_rate": lr,
                                "seed": int(random_seed),
                                "device": device,
                                "resume": resume,
                                "pretrained": pretrained,
                            }
                        )
                    except RuntimeError as exc:
                        raise gr.Error(str(exc)) from exc
                    job_id = started["job_id"]
                    while True:
                        status = service.training_status(job_id)
                        yield (
                            f'训练状态：{status["status"]} | 返回码：{status["return_code"]}',
                            status.get("log", ""),
                            gr.update(),
                            gr.update(),
                        )
                        if status["status"] != "running":
                            break
                        time.sleep(2)
                    if status["return_code"] == 0:
                        choices = _model_choices()
                        if choices:
                            value = choices[0][1]
                            yield (
                                "训练完成（返回码 0），已刷新模型列表并选中最新模型。",
                                status.get("log", ""),
                                gr.Dropdown(choices=choices, value=value),
                                _describe_model(value),
                            )

                train_button.click(
                    train,
                    inputs=[
                        workspace_select,
                        epochs,
                        architecture,
                        batch_size,
                        input_size,
                        learning_rate,
                        seed,
                        training_device,
                        resume_checkpoint,
                        use_pretrained,
                    ],
                    outputs=[training_message, training_log, model_path, model_info],
                )

    return demo


demo = create_ui()
# Keep demo.css/theme/head as non-None strings so App.create_app(demo) test path
# can render without jinja2 undefined errors. launch() treats them the same.
demo.css = """
.app-title h1 {
    margin: 0 0 2px 0;
    font-size: 24px;
}
.app-subtitle,
.app-subtitle p {
    margin: 0 0 12px 0;
    font-size: 14px;
    opacity: 0.75;
}
.tab-hint p {
    margin: 0 0 12px 0;
    font-size: 12.5px;
    line-height: 1.6;
    opacity: 0.7;
}
.workspace-status,
.workspace-summary {
    padding: 3px 6px;
}
.workspace-status p,
.workspace-summary p {
    margin: 0;
    font-size: 13px;
    opacity: 0.75;
}
.workspace-summary code {
    font-size: 12px;
}
.status-text,
.status-text p {
    margin: 0;
    font-size: 13px;
    opacity: 0.85;
    align-self: center;
}
.nav-row {
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
}
.plain-checkbox {
    align-self: center;
}
.plain-checkbox label {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
"""
demo.theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
demo.head = """
<script>
(function () {
    const targets = {
        'q': 'btn-sharp',
        'w': 'btn-borderline',
        'e': 'btn-defocused',
    };
    function clickIfReady(id) {
        const btn = document.getElementById(id);
        if (!btn) return;
        const disabled = btn.hasAttribute('disabled')
            || btn.getAttribute('aria-disabled') === 'true';
        if (!disabled) btn.click();
    }
    document.addEventListener('keydown', function (e) {
        const tag = (e.target || {}).tagName || '';
        const editable = tag === 'INPUT' || tag === 'TEXTAREA'
            || (e.target && e.target.isContentEditable);
        if (editable) return;
        const id = targets[e.key.toLowerCase()];
        if (id) {
            e.preventDefault();
            clickIfReady(id);
        }
    });
})();
</script>
"""

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        css=demo.css,
        head=demo.head,
    )
