"""Shared fixed data-directory control for annotation, versioning, and training."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from webui_service import ROOT


@dataclass(frozen=True)
class DataRootPanel:
    root: gr.Textbox


def build_data_root_panel() -> DataRootPanel:
    data_root = gr.Textbox(
        label="数据集目录（当前基础数据集）",
        value=str(ROOT / "data"),
        interactive=False,
    )
    return DataRootPanel(root=data_root)
