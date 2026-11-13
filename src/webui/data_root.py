"""Shared fixed data-directory state for annotation, versioning, and training."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from webui_service import ROOT


@dataclass(frozen=True)
class DataRootPanel:
    root: gr.State


def build_data_root_panel() -> DataRootPanel:
    data_root = gr.State(value=str(ROOT / "data"))
    return DataRootPanel(root=data_root)
