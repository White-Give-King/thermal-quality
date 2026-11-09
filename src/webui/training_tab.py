"""「模型训练」标签页：基于所选工作区启动后台训练。"""

from __future__ import annotations

import time

import gradio as gr

from webui.inference_tab import describe_model, model_choices
from webui.workspace import WorkspacePanel
from webui_service import WebUIService


def build_training_tab(
    service: WebUIService,
    workspace: WorkspacePanel,
    model_path: gr.Dropdown,
    model_info: gr.Markdown,
) -> None:
    gr.Markdown(
        "自动读取上方所选工作区 `annotations.sqlite3` 中的标注进行训练，模型输出到该工作区的 `runs/` 子目录。"
        "默认使用预训练 MobileNetV3-Large、30 轮、batch 16、输入 224、学习率 0.0001；"
        "这是适合当前小数据集的推荐起点，不代表已通过超参数搜索验证为最优。",
        elem_classes=["tab-hint"],
    )
    with gr.Row():
        epochs = gr.Number(label="Epoch", value=30, precision=0)
        architecture = gr.Dropdown(
            label="模型架构",
            choices=[
                "mobilenet_v3_small",
                "mobilenet_v3_large",
                "efficientnet_b0",
                "resnet18",
            ],
            value="mobilenet_v3_large",
            info="推荐预训练 MobileNetV3-Large；如数据量很少或希望更轻量，可尝试 MobileNetV3-Small。",
            scale=2,
        )
        batch_size = gr.Number(
            label="Batch size",
            value=16,
            precision=0,
            info="推荐 16；显存不足时调小，显存充足也可比较 32。",
        )
        input_size = gr.Number(
            label="输入尺寸",
            value=224,
            precision=0,
            info="推荐 224，与 torchvision 预训练权重常用输入尺寸匹配。",
        )
    with gr.Row():
        learning_rate = gr.Number(
            label="学习率",
            value=0.0001,
            info="推荐 0.0001；不要设为 1.0，过大会导致训练不稳定。",
        )
        seed = gr.Number(label="随机种子", value=42, precision=0)
        training_device = gr.Dropdown(
            label="训练设备",
            choices=["auto", "cuda", "cpu"],
            value="auto",
            info="推荐 auto：CUDA 可用时自动使用 CUDA，否则使用 CPU。",
        )
    with gr.Row(equal_height=True):
        resume_checkpoint = gr.Textbox(
            label="继续训练权重（可选）",
            placeholder="留空则从头训练",
            scale=3,
        )
        use_pretrained = gr.Checkbox(
            label="显式下载 torchvision 预训练权重（需要网络）",
            value=True,
            info="推荐开启。首次使用需要网络下载权重；取消后从随机初始化开始训练。",
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
        root: str,
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
                    "workspace_root": root,
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
            choices = model_choices(service)
            if choices:
                value = choices[0][1]
                yield (
                    "训练完成（返回码 0），已刷新模型列表并选中最新模型。",
                    status.get("log", ""),
                    gr.Dropdown(choices=choices, value=value),
                    describe_model(service, value),
                )

    train_button.click(
        train,
        inputs=[
            workspace.root,
            workspace.select,
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
