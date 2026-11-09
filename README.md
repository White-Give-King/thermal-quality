---
title: 红外热图清晰度与对焦质量
sdk: gradio
app_file: app.py
---
# 红外热图清晰度与对焦质量识别

本项目判断伪彩色热图整体是否清晰、相机对焦是否正常。**有效热目标的轮廓只是人工观察清晰度时的参考，不是模型分类目标**；软件不标注目标有无，不使用 YOLO 或目标框。

模型输入为配置指定的整块热成像区域，MobileNetV3-Small 输出三类：

| 标签 | 含义 |
| --- | --- |
| `sharp` | 画面细节和可见轮廓清晰，对焦正常 |
| `borderline` | 清晰度处于临界状态或分类证据不足 |
| `defocused` | 画面明显模糊、轮廓扩散，对焦不正常 |
| `uncertain` | 模型分数未达到判定阈值、模型不可用或输出异常；不用于训练 |

不能仅凭伪彩色颜色或温度值判断清晰度。若有效目标不明显，应结合图中其他可见边缘和细节判断；仍无法判断时标为 `borderline`，不要因为“没有目标”就标成失焦。

## 环境

项目 Conda 环境使用 Python 3.14：

```powershell
conda env create -f environment.yml
conda activate thermal-quality
python -m pip install -e .
```

PyTorch is configured to use the CUDA 13.0 (`cu130`) wheels. The NVIDIA driver must support CUDA 13; a separate CUDA Toolkit installation is not required for PyTorch.

检查 CUDA：

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

`--pretrained` 会显式请求 torchvision 下载 MobileNetV3 权重；默认不假定有预训练模型或自动联网。

## 项目目录

```text
红外识别/
├── configs/       # 图像与热区配置
├── data/          # 任务工作区（含标注数据库、裁剪图和各工作区的 runs/ 模型输出）
├── runs/          # 旧版模型输出目录，仍会被模型列表扫描
├── scripts/       # 推理、训练、评估、阈值校准和数据下载工具
├── src/           # Python 源码模块直接位于此处，WebUI 界面代码在 webui/ 子目录
├── tests/         # 自动化测试
├── 数据集/        # 原始图片与设备元数据
└── archive/       # 已封存的旧标注和历史训练产物
```

下载工具位于 `scripts/download_inspection_pictures.py`，默认读取 `数据集/A2分部.json`；可用 `--help` 查看参数。`数据集/` 下的原图与 JSON 元数据保持原位，避免影响现有数据路径。

## 有效热区

[configs/default.yaml](./configs/default.yaml) 为命令行和图片推理提供默认热区配置，默认在原图居中裁剪 `809×1080`。命令行推理、图片推理和标注页都会先检测每张图片的近黑边界；未检测到有效边框时，推理使用配置中的热区裁剪，标注页使用配置中的宽高居中裁剪。每个工作区目录内的 `annotations.sqlite3` 保存各图片的裁剪坐标和标签记录。手动调整坐标时，位置和宽高的上下限随图片尺寸及当前裁剪框动态更新，且服务端会校验越界输入。点击“应用坐标预览”只临时预览；点击“保存当前图片裁剪并更新标注”会保存当前图片参数，并在该图已有标注时仅重裁该样本，不会修改全局 YAML。

## A7 标注

A7 本地图片位于 `数据集/A7-10-19/`。运行 `gradio app.py`（或双击 `启动WebUI.bat`）后，本机打开 `http://127.0.0.1:7860`，同一局域网内的协作者可通过标注主机的局域网 IP 访问 `http://<主机IP>:7860`。在“图片标注”页扫描 `数据集` 并选择工作区即可。WebUI 直接读主机上的原图，无需上传。把每个工作区的原图放在所选根目录的直接子文件夹中，例如：

```text
数据集/
├── A7-10-19/
│   ├── frame001.png
│   └── ...
└── CameraB/
    ├── frame001.png
    └── ...
```

在 WebUI“图片标注”页填写根目录（例如 `data` 或 `数据集`）并扫描，选中工作区后即自动进入标注（该工作区同时用于训练）。多台设备或多个浏览器页面连接同一个 WebUI 服务时，会共享未标注图片队列，同一张图不会同时分配给多个页面；领取任务闲置 30 分钟后会自动释放。每张图片的标签、源图路径和裁剪图路径都会保存在工作区 `annotations.sqlite3` 的 `image_annotations` 表中。图片只从主机原目录读取，不会上传或复制；标注数据与训练结果按工作区隔离保存。工作区位于 `data/` 下时（含分部归组结构 `data/A7|A8|A9|A10/<任务名>/`，扫描会自动进入下一级），裁剪图 `images/` 和模型 `runs/` 直接写入该工作区文件夹内；其他根目录（如 `数据集`）下的工作区则将裁剪图与模型输出到 `data/<工作区名>/`。例如选中 `data/A10/A10-2026-01-16-21-40-16` 时，标注与模型都保存在该任务文件夹内；选中 `数据集/A7-10-19` 时，标签数据库保存在原工作区，裁剪图与模型输出为 `data/A7-10-19/` 和 `data/A7-10-19/runs/`。工作区名称必须是单层文件夹名；不同根目录下的同名工作区会共用同一输出目录，请避免名称冲突。

局域网访问目前不带登录认证，请仅在可信网络中开放，并确保主机防火墙只允许可信设备访问 WebUI 端口。

点击三个类别按钮（清晰 / 临界 / 模糊）会保存当前图并自动领取下一张；“领取下一张”可领取其他网页尚未分配的待标注图片。缩略图栏显示当前图及邻近最多 8 张，点击可直接查看；序号框输入后回车也可跳转。标注时以有效目标的边缘是否清楚作为参考，不要判断“有没有目标”。裁剪图保存到 `data/A7-10-19/images/`。JSON 里的设备 `mark` 字段不是清晰度真值，不会自动导入。

此前 106 条 A7 “有/无有效目标”标注与当前任务不同，已移入 `archive/A7-target-presence-20261026/` 封存；原始 A7 图片没有改动。之前的训练结果亦不作为清晰度模型使用。请重新标注清晰度，不要把旧二分类标签转换或混入新训练集。

标注数据库中的 `image_annotations` 表包含裁剪图相对路径、类别标签、原图路径和标注时间。查看类别分布：

```powershell
python -c "from dataset import load_annotations; f=load_annotations(r'.\数据集\A7-10-19\annotations.sqlite3', image_root=r'.\data\A7-10-19'); print('samples=',len(f),'classes=',f.label.value_counts().to_dict())"
```

标注建议：覆盖不同设备、背景、距离、角度、目标大小和色标设置；特别收集正常对焦、明显失焦和两者之间的临界样本。用相机的对焦成功标志作参考记录即可，不能直接代替图像清晰度标签。没有代表性类别样本时不要训练或声称性能可靠。

## 训练

可选模型架构：`mobilenet_v3_small`、`mobilenet_v3_large`、`efficientnet_b0`、`resnet18`。训练脚本和 WebUI 默认采用预训练 `mobilenet_v3_large`、30 轮、batch 16、输入 224、学习率 `0.0001`；这是小数据集的推荐起点，不代表已通过超参数搜索验证为最优。预训练权重默认开启，首次训练需要网络；离线时可使用 `--no-pretrained`。权重会保存架构标识，推理和评估根据 checkpoint 自动加载对应网络。建议先用同一训练/验证划分比较，再以独立数据评估，不应仅凭模型名称推断准确率。

三类均需有足够样本，训练/验证/测试按标签分层拆分：

```powershell
python scripts/train_classifier.py `
  --database .\数据集\A7-10-19\annotations.sqlite3 `
  --image-root .\data\A7-10-19 `
  --output .\runs\A7-10-19 `
  --architecture mobilenet_v3_large `
  --device cuda `
  --epochs 30 `
  --batch-size 16 `
  --input-size 224 `
  --seed 42
```

以上参数与推荐默认值一致；如需离线训练，可添加 `--no-pretrained`。

训练输出 `best.pt`、`last.pt`、训练/验证/测试拆分数据库与训练配置。若类别不足或拆分无法分层，脚本会报错。正式性能评估应使用与训练样本独立的设备或采集时段，不能用测试集调阈值。

## 推理

### Gradio 网页界面（推理、标注与训练）

安装网页依赖后启动本地 Gradio 页面：

```powershell
python -m pip install -e ".[ui]"
gradio app.py
```

默认启动完整界面；若只需图片推理，使用 `gradio app.py -- --inference-only`。`--` 后的参数会传给应用脚本；此模式不显示数据集、标注和训练功能。

Windows 下也可以双击项目根目录的 `启动WebUI.bat`，它会调用 `thermal-quality` Conda 环境并启动本地 Gradio 服务。如果 Gradio 已在 7860 端口运行，脚本会复用现有服务并打开浏览器，不会重复启动；若端口被其他程序占用，会显示占用进程信息。关闭运行中的服务窗口即可停止服务。

打开 `http://127.0.0.1:7860`，或在局域网其他设备上打开 `http://<主机IP>:7860`：

- **图片推理**：选择本地训练得到的 `.pt` 权重、CPU/CUDA 和一张或多张图片；页面显示分类状态、未校准模型分数、概率 JSON、原图热区预览和热区裁剪图。推理会先检测热图黑边并裁掉，未检测到有效黑边时使用默认配置热区。模型分数是最高类别 softmax 概率乘以 100，不是准确率或清晰度百分比。坏图会单独显示错误，不会被当作模型结果。
- **图片标注**：填写工作区根目录，扫描并选择含图片的子文件夹（选中后立即进入标注，无需额外按钮）；多个连接到同一 WebUI 服务的页面共享未标注图片队列，可查看当前图和前后缩略图、跳转并标注三类。载入时只枚举图片路径，图像在显示或跳转时按需读取，不会预先解码整个工作区；图片不上传或复制，`data/` 下工作区的标注保存到工作区文件夹内，其他根目录的工作区保存到 `data/<工作区名>/`。
- **模型训练**：选择工作区后，页面自动读取该工作区的 `annotations.sqlite3` 数据库，模型输出到工作区 `runs/` 子目录；选择模型架构（MobileNetV3-Small/Large、EfficientNet-B0、ResNet18）、epoch、batch size、输入尺寸、学习率和设备。训练以本机后台任务运行，可查看状态和日志；完成后模型保存在工作区的 `best.pt`、`last.pt`，推理会从 checkpoint 自动识别架构。
- 标注数据库位于原工作区（如 `数据集/A7-10-19/annotations.sqlite3`）；裁剪图与默认权重位于 `data/A7-10-19/` 和 `data/A7-10-19/runs/best.pt`。数据库必须包含三类足够样本；否则训练脚本会给出失败日志。
- “下载预训练权重”默认关闭，勾选后需要网络。Gradio 界面单次只运行一个训练任务。

Gradio 界面只绑定在本机 `127.0.0.1`，不要直接暴露到局域网或公网；界面允许启动本机训练进程并读取用户指定路径的文件。命令行 `scripts/predict.py` 与训练脚本仍可独立使用。

### 命令行推理

```powershell
python scripts\predict.py `
  --input ".\数据集\A7-10-19" `
  --output ".\outputs\A7-10-19" `
  --model ".\runs\A7-10-19\best.pt" `
  --device cuda
```

每张图生成原图热区预览 `.annotated.jpg`、实际热区 `.thermal.png` 和结果 `.json`。JSON 的 `status` 为 `sharp`、`borderline`、`defocused` 或 `uncertain`，`probabilities` 是模型 softmax 输出，不等同于校准置信度；`model_score` 是最高类别 softmax 概率乘以 100 并保留一位小数，也未经校准。

## 评估

准备好独立测试集数据库后运行：

```powershell
python scripts/evaluate.py `
  --database .\data\independent-test\annotations.sqlite3 `
  --image-root .\data\independent-test `
  --model .\runs\A7-10-19\best.pt `
  --device cuda `
  --output .\runs\A7-10-19\test-report.json
```

评估包含三分类混淆矩阵、各类 Precision/Recall/F1、Macro-F1、失焦漏检率、清晰图误拒率、复核比例和推理延迟。小样本和训练集指标不能证明现场可靠性。

只用训练数据中的验证集校准置信度阈值，不要用测试集调整阈值：

```powershell
python scripts/calibrate_thresholds.py `
  --database .\runs\A7-10-19\validation_split.sqlite3 `
  --model .\runs\A7-10-19\best.pt `
  --device cuda `
  --output .\runs\A7-10-19\threshold-calibration.json
```

脚本从验证集的 Macro-F1 选择推荐阈值，并报告各候选阈值的失焦漏检率、清晰图误拒率和复核比例。将该阈值写入配置前，应结合现场风险人工复核；校准结果不代表独立测试性能。

## 测试

```powershell
python -m pytest
```
