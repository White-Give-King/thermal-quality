"""「数据集」标签页：JSON 巡检数据归档与多任务并行下载。"""

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

from webui_service import ROOT, WebUIService


def build_dataset_tab(service: WebUIService, blocks: gr.Blocks) -> None:
    gr.Markdown(
        "上传 JSON 巡检数据文件，或使用工作区路径。每个任务归档为 "
        "`data/<分部>/<任务名>/`：`<任务名>.json` 与 `images_source/` 原图，支持断点续传。"
        "没有有效图片的 JSON 会直接跳过，不创建工作区。"
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

    def _resolve_json_input(files: list[str] | None, path_text: str) -> list[str]:
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
            prefix = _dept_prefix(info["dept"])
            if info["with_picture"] <= 0:
                summaries.append(
                    f"- **{info['dept'] or '未知'}**（`{p.name}`）："
                    f"{info['with_picture']} 张有图，{info['empty_picture']} 张无图  \n"
                    f"  无有效图片，将跳过归档与下载，不创建工作区"
                )
                continue
            summaries.append(
                f"- **{info['dept'] or '未知'}**（`{p.name}`）："
                f"{info['with_picture']} 张有图，{info['empty_picture']} 张无图  \n"
                f"  JSON 保存至 `data/{prefix}/{run_name}/{run_name}.json`，"
                f"图片下载至 `data/{prefix}/{run_name}/images_source/`"
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

    def _task_dir(run_name: str, user_root: str, dept: str) -> Path:
        base = str(user_root or "").strip()
        root = Path(base).expanduser() if base else DATA_ROOT
        # 与 data/ 的分部归组结构保持一致：data/<分部>/<任务名>/
        return root / _dept_prefix(dept) / run_name

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
    ) -> tuple[list[tuple[Path, str, Path]], list[str]]:
        """返回 (待下载任务, 无有效图片被跳过的 JSON 文件名列表)。"""
        prepared: list[tuple[Path, str, Path]] = []
        skipped_empty: list[str] = []
        seen: set[str] = set()
        for p in _collect_json_files(json_inputs):
            info = _parse_one_json(p)
            if info is None:
                continue
            run_name = info["run_name"] or p.stem
            if run_name in seen:
                continue
            seen.add(run_name)
            if info["with_picture"] <= 0:
                # JSON 里没有任何有效图片：不建工作区、不归档、不下载。
                skipped_empty.append(p.name)
                continue
            task_dir = _task_dir(run_name, output_root, info["dept"])
            archived = _archive_json(p, task_dir, run_name)
            prepared.append((archived, run_name, task_dir / "images_source"))
        return prepared, skipped_empty

    def run_download_job(
        files: list[str] | None,
        path_text: str,
        output_dir: str,
    ):
        json_input = _resolve_json_input(files, path_text)
        if not json_input:
            raise gr.Error("请选择 JSON 文件或填写路径。")
        prepared, skipped_empty = _collect_prepared(json_input, output_dir)
        if not prepared:
            message = "未找到可下载的 JSON 文件，请检查所选文件或路径。"
            if skipped_empty:
                message += (
                    f"另有 {len(skipped_empty)} 个 JSON 无有效图片已跳过："
                    + "、".join(skipped_empty)
                )
            raise gr.Error(message)
        skipped_note = (
            f"（另有 {len(skipped_empty)} 个 JSON 无有效图片已跳过："
            + "、".join(skipped_empty)
            + "）"
            if skipped_empty
            else ""
        )

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
                    (
                        f"进度 {finished}/{total}；正在下载 {len(running)} 个"
                        f"（当前上限 {concurrency_state['value']}）：{names}"
                    ),
                    prefix + live_logs,
                )
                time.sleep(1.5)

        yield (
            f"全部完成：共 {total} 个 JSON，图片已保存到对应任务目录的 "
            f"`images_source/`。{skipped_note}",
            prefix,
        )

    def start_download(files: list[str] | None, path_text: str, out_dir: str):
        yield from run_download_job(files, path_text, out_dir)

    # ---------- 事件绑定 ----------

    preview_inputs = [json_files, json_path_input]
    preview_outputs = [dataset_summary, dataset_table]
    blocks.load(
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
