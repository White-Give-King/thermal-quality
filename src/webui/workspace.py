"""工作区面板：扫描根目录并选择待标注/训练的工作区。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gradio as gr

from thermal_focus.webui_service import ROOT, WebUIService


@dataclass
class WorkspacePanel:
    """工作区面板组件引用，供标注与训练标签页联动。"""

    root: gr.Textbox
    select: gr.Dropdown


def build_workspace_panel(service: WebUIService) -> WorkspacePanel:
    workspace_records = gr.State([])

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
                elem_id="workspace-message",
                elem_classes=["workspace-status"],
                scale=0,
            )
            workspace_summary = gr.Markdown(
                "此处显示所选工作区的图片数量与输出目录。",
                elem_id="workspace-summary",
                elem_classes=["workspace-summary"],
            )

    def describe_workspace(record: dict[str, Any]) -> str:
        if not record:
            return ""
        output_dir = record.get("output_dir") or record["data_dir"]
        return (
            f'**图片（{record["image_count"]} 张）**：`{record["image_dir"]}`  \n'
            f'**标注数据库**：`{record["annotation_database"]}`  \n'
            f'**模型输出**：`{output_dir}`（`runs/`）'
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

    return WorkspacePanel(root=workspace_root, select=workspace_select)
