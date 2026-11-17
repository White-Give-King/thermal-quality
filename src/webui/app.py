"""Gradio WebUI 组装：固定数据目录面板与各功能标签页在此拼装。"""

from __future__ import annotations

import gradio as gr

from webui.annotation_tab import build_annotation_tab
from webui.data_root import build_data_root_panel
from webui.dataset_version_tab import build_dataset_version_tab
from webui.inference_tab import build_inference_tab
from webui.public_api import register_public_apis
from webui.training_tab import build_training_tab
from webui_service import WebUIService


def create_ui(*, enable_annotation: bool = False, enable_training: bool = False) -> gr.Blocks:
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
        data_root = build_data_root_panel()
        with gr.Tabs():
            if enable_annotation:
                with gr.Tab("图片标注"):
                    build_annotation_tab(service, data_root, demo)
            with gr.Tab("数据集版本") as dataset_version_tab:
                build_dataset_version_tab(service, data_root, demo, dataset_version_tab)
            with gr.Tab("图片推理") as inference_tab:
                model_info = build_inference_tab(service, demo, inference_tab)
            if enable_training:
                with gr.Tab("模型训练"):
                    build_training_tab(service, data_root, model_info)

        register_public_apis(
            service, include_training=enable_training,
            include_annotation=enable_annotation,
        )

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
.version-rollback-button:not(:disabled) {
    background-color: #2563eb !important;
    border-color: #2563eb !important;
    color: #fff !important;
}
.version-rollback-button:not(:disabled):hover {
    background-color: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
}
.version-rollback-button:focus-visible {
    outline: 2px solid #93c5fd;
    outline-offset: 2px;
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
#prediction-gallery .grid-wrap {
    height: 100% !important;
    padding-left: 0 !important;
    padding-top: 0 !important;
    padding-right: 0 !important;
    padding-bottom: 0 !important;
    overflow-y: auto !important;
    scroll-snap-type: y mandatory;
}
#prediction-gallery .grid-container {
    height: 100% !important;
    grid-auto-rows: 100% !important;
}
#prediction-gallery .gallery-item,
#prediction-gallery .thumbnail-lg {
    width: 100% !important;
    height: 100% !important;
    min-height: 0 !important;
    aspect-ratio: auto !important;
    box-sizing: border-box !important;
    scroll-snap-align: start;
    scroll-snap-stop: always;
}
#prediction-gallery .thumbnail-lg {
    border-top: 0 !important;
    border-bottom: 0 !important;
}
#prediction-gallery .thumbnail-lg::after {
    content: "";
    position: absolute;
    inset: 0;
    z-index: 2;
    box-sizing: border-box;
    border: 1px solid transparent;
    pointer-events: none;
}
#prediction-gallery .thumbnail-lg:hover::after,
#prediction-gallery .thumbnail-lg.selected::after {
    border-color: var(--color-accent);
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


def create_demo(*, enable_annotation: bool = False, enable_training: bool = False) -> gr.Blocks:
    demo = create_ui(enable_annotation=enable_annotation, enable_training=enable_training)
    # Keep demo.css/theme/head as non-None strings so App.create_app(demo) test path
    # can render without jinja2 undefined errors. launch() treats them the same.
    demo.css = _CSS
    demo.theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
    demo.head = _HEAD_JS
    return demo


demo = create_demo()
