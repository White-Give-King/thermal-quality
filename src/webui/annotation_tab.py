"""「图片标注」标签页：缩略图导航、快捷标注与热区调整。"""

from __future__ import annotations

from typing import Any

import gradio as gr

from webui.workspace import WorkspacePanel
from webui_service import WebUIService

# 未选择工作区时标注视图组件的占位输出。
_no_session_annotation = ([], "请先在上方选择工作区。", 1, 0, 0, 1, 1, [])


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


def _bounded_roi(
    x: float,
    y: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    roi_width = max(1, min(int(width), image_width))
    roi_height = max(1, min(int(height), image_height))
    roi_x = max(0, min(int(x), image_width - roi_width))
    roi_y = max(0, min(int(y), image_height - roi_height))
    return roi_x, roi_y, roi_width, roi_height


def _roi_input_updates(
    x: int, y: int, width: int, height: int
) -> tuple[dict[str, Any], ...]:
    return (
        gr.update(value=x),
        gr.update(value=y),
        gr.update(value=width),
        gr.update(value=height),
    )


def _annotation_outputs(
    service: WebUIService, session_id: str, *, original: bool = False
) -> tuple[Any, ...]:
    session = service.get_annotation_session(session_id)
    result = service.render_annotation(session, original=original)
    preview_errors = result.get("preview_errors", [])
    preview_warning = (
        f' | 无法预览：{", ".join(preview_errors)}' if preview_errors else ""
    )
    roi = result["roi"]
    gallery_indices = result["gallery_indices"]
    label_message = " | 已标注" if result["label"] else ""
    roi_inputs = _roi_input_updates(
        roi["x"],
        roi["y"],
        roi["width"],
        roi["height"],
    )
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
            f' | 当前：{result["label"] or "未标注"} | {result["filename"]}'
            f'{label_message}'
            f'{preview_warning}'
        ),
        result["index"] + 1,
        *roi_inputs,
        gallery_indices,
    )


def build_annotation_tab(service: WebUIService, workspace: WorkspacePanel) -> None:
    gr.Markdown(
        "选择上方工作区后即可开始标注。"
        "点击下方缩略图可放大预览，"
        "点击「清晰 / 临界 / 模糊」保存标注并切换到下一张，"
        "也可按 ← 清晰、↑ 临界、→ 模糊快速标注；"
        "每张图片独立识别黑边，未识别时使用居中热区；"
        "每图裁剪参数保存在工作区的 `annotations.sqlite3`，"
        "标签与热区裁剪图写入 `data/<工作区名>/`。",
        elem_classes=["tab-hint"],
    )
    annotation_session = gr.State("")
    gallery_indices_state = gr.State([])
    annotation_gallery = gr.Gallery(
        label="图片预览",
        show_label=False,
        elem_id="annotation-gallery",
        columns=2,
        rows=None,
        type="numpy",
        height="75vh",
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
            elem_id="btn-previous",
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
        next_unlabeled_button = gr.Button(
            "下一张未标注 →",
            size="lg",
            scale=0,
            min_width=160,
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
            elem_id="annotation-counter",
            elem_classes=["status-text"],
            scale=1,
            min_width=260,
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
                "保存当前图片裁剪并更新标注",
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

    annotation_buttons = [
        previous_button,
        next_button,
        next_unlabeled_button,
        quick_sharp,
        quick_borderline,
        quick_defocused,
        apply_roi,
        save_roi,
    ]

    def _annotation_buttons_update(enabled: bool) -> tuple[Any, ...]:
        return tuple(gr.update(interactive=enabled) for _ in annotation_buttons)

    def begin_annotation(root: str, name: str, original: bool) -> tuple[Any, ...]:
        try:
            result = service.start_annotation(root, name)
            return (
                result["session_id"],
                (
                    f'已载入 {result["total"]} 张图片（按需读取），输出目录：{result["output_dir"]}'
                    f'；标注与裁剪数据库：{result["roi_database_path"]}'
                ),
                *_annotation_outputs(service, result["session_id"], original=original),
                *_annotation_buttons_update(True),
            )
        except (OSError, ValueError, RuntimeError) as exc:
            return (
                "",
                f"启动失败：{exc}",
                *_no_session_annotation,
                *_annotation_buttons_update(False),
            )

    workspace.select.change(
        begin_annotation,
        inputs=[workspace.root, workspace.select, original_view],
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
            return _annotation_outputs(service, session_id, original=original)
        except (OSError, ValueError, RuntimeError) as exc:
            return ([], f"跳转失败：{exc}", index, 0, 0, 1, 1, [])

    previous_button.click(
        lambda session, index, original: navigate_annotation(
            session, max(1, index - 1), original
        ),
        inputs=[annotation_session, annotation_index, original_view],
        outputs=annotation_view_outputs,
    )

    def next_annotation(
        session_id: str, index: float, original: bool
    ) -> tuple[Any, ...]:
        if not session_id:
            return _no_session_annotation
        return navigate_annotation(session_id, index + 1, original)

    next_button.click(
        next_annotation,
        inputs=[annotation_session, annotation_index, original_view],
        outputs=annotation_view_outputs,
    )

    def next_unlabeled_annotation(
        session_id: str, original: bool
    ) -> tuple[Any, ...]:
        if not session_id:
            raise gr.Error("请先在上方选择工作区。")
        try:
            result = service.claim_next_annotation(session_id)
            outputs = list(_annotation_outputs(service, session_id, original=original))
            if not result["claimed"]:
                outputs[1] = f"没有可领取的未标注图片。{outputs[1]}"
            return tuple(outputs)
        except (OSError, ValueError, RuntimeError) as exc:
            return ([], f"切换未标注图片失败：{exc}", 1, 0, 0, 1, 1, [])

    next_unlabeled_button.click(
        next_unlabeled_annotation,
        inputs=[annotation_session, original_view],
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
        return navigate_annotation(session_id, absolute_index + 1, original)

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
        lambda session, original: _annotation_outputs(service, session, original=original)
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
        result = service.save_label(session_id, label, advance=True)
        outputs = list(_annotation_outputs(service, session_id, original=original))
        if result["skipped"]:
            outputs[1] = f"该图片已标注，已跳过。{outputs[1]}"
        return tuple(outputs)

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
            f'当前图片的热区参数已保存；更新了 {result["updated_labels"]} 个已标注样本。'
            if save
            else "当前图片的热区预览已应用，尚未保存参数。"
        )
        return (
            *_annotation_outputs(service, session_id, original=original),
            message,
        )

    def constrain_roi_inputs(
        session_id: str, x: float, y: float, width: float, height: float
    ) -> tuple[Any, ...]:
        if not session_id:
            return tuple(gr.update() for _ in range(4))
        image_width, image_height = service.annotation_image_size(session_id)
        bounded_roi = _bounded_roi(
            x, y, width, height, image_width, image_height
        )
        return _roi_input_updates(*bounded_roi)

    roi_inputs = [
        annotation_session,
        roi_x,
        roi_y,
        roi_width,
        roi_height,
        original_view,
    ]
    for roi_input in (roi_x, roi_y, roi_width, roi_height):
        roi_input.change(
            constrain_roi_inputs,
            inputs=roi_inputs[:5],
            outputs=[roi_x, roi_y, roi_width, roi_height],
        )
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
