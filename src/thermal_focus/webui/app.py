"""Gradio WebUI 组装：工作区面板与各标签页在此拼装为完整界面。"""

from __future__ import annotations

import gradio as gr

from thermal_focus.webui.annotation_tab import build_annotation_tab
from thermal_focus.webui.dataset_tab import build_dataset_tab
from thermal_focus.webui.inference_tab import build_inference_tab
from thermal_focus.webui.training_tab import build_training_tab
from thermal_focus.webui.workspace import build_workspace_panel
from thermal_focus.webui_service import WebUIService


def create_ui(*, inference_only: bool = False) -> gr.Blocks:
    """Create the Gradio interface backed by the in-process service layer."""
    service = WebUIService()
    with gr.Blocks(
        title="红外热图清晰度与对焦质量",
    ) as demo:
        gr.Markdown("# 红外热图清晰度与对焦质量", elem_classes=["app-title"])
        gr.Markdown(
            "本地红外图像对焦质量的标注、推理与训练工具",
            elem_classes=["app-subtitle"],
        )
        if inference_only:
            gr.Markdown("仅推理模式：数据集、图片标注和模型训练功能已关闭。")
            build_inference_tab(service)
        else:
            workspace = build_workspace_panel(service)
            with gr.Tabs():
                with gr.Tab("数据集"):
                    build_dataset_tab(service, demo)
                with gr.Tab("图片标注"):
                    build_annotation_tab(service, workspace)
                with gr.Tab("图片推理"):
                    model_path, model_info = build_inference_tab(service)
                with gr.Tab("模型训练"):
                    build_training_tab(service, workspace, model_path, model_info)

    return demo


_CSS = """
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
/* Gallery thumbnails: remove Gradio's default square (1:1) aspect ratio
   so cells grow to the natural portrait ratio of thermal-panel images. */
.fixed-height {
    max-height: none !important;
}
.thumbnail-lg {
    aspect-ratio: unset !important;
}
.thumbnail-lg img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
}
/* Gallery click-preview overlay: keep the overlay full-screen so the dark
   mask covers everything behind it, and only cap the image inside so it
   doesn't feel overwhelming. */
.preview img {
    display: block !important;
    margin: auto !important;
    max-width: min(85vw, 1200px) !important;
    max-height: min(75vh, 900px) !important;
    object-fit: contain !important;
}
#annotation-gallery .thumbnails {
    height: 104px !important;
}
#annotation-gallery .preview .media-button {
    height: calc(100% - 104px) !important;
}
#annotation-gallery .thumbnail-small {
    width: 84px !important;
    height: 84px !important;
    transform: none !important;
}
.workspace-status,
.workspace-summary {
    padding: 3px 6px;
    overflow: visible !important;
}
.workspace-status p,
.workspace-summary p {
    margin: 0;
    font-size: 13px;
    opacity: 0.75;
}
#workspace-summary code {
    font-size: 12px;
    overflow-wrap: anywhere;
}
.status-text,
.status-text p {
    margin: 0;
    font-size: 13px;
    opacity: 0.85;
    align-self: center;
}
#annotation-counter p {
    overflow-wrap: anywhere;
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

_HEAD_JS = """
<script>
(function () {
    const targets = {
        'q': 'btn-sharp',
        'w': 'btn-borderline',
        'e': 'btn-defocused',
        'ArrowLeft': 'btn-sharp',
        'ArrowUp': 'btn-borderline',
        'ArrowDown': 'btn-previous',
        'ArrowRight': 'btn-defocused',
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
        const id = targets[e.key] || targets[e.key.toLowerCase()];
        if (id) {
            e.preventDefault();
            clickIfReady(id);
        }
    });
})();
</script>
"""


def create_demo(*, inference_only: bool = False) -> gr.Blocks:
    demo = create_ui(inference_only=inference_only)
    # Keep demo.css/theme/head as non-None strings so App.create_app(demo) test path
    # can render without jinja2 undefined errors. launch() treats them the same.
    demo.css = _CSS
    demo.theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
    demo.head = _HEAD_JS
    return demo


demo = create_demo()
