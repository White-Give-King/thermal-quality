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

推荐 Python 3.11 和 Conda 环境：

```powershell
conda env create -f environment.yml
conda activate thermal-focus-quality
python -m pip install -e .
```

检查 CUDA：

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

`--pretrained` 会显式请求 torchvision 下载 MobileNetV3 权重；默认不假定有预训练模型或自动联网。

## 项目目录

```text
红外识别/
├── configs/       # 图像与热区配置
├── data/          # 按工作区保存的标注 CSV 和裁剪图
├── runs/          # 按工作区保存的模型、训练日志和数据拆分
├── scripts/       # 训练、评估、阈值校准和数据下载工具
├── src/           # 可复用的识别与数据处理代码
├── tests/         # 自动化测试
├── 数据集/        # 原始图片与设备元数据
└── archive/       # 已封存的旧标注和历史训练产物
```

下载工具位于 `scripts/download_inspection_pictures.py`，默认读取 `数据集/A2分部.json`；可用 `--help` 查看参数。`数据集/` 下的原图与 JSON 元数据保持原位，避免影响现有数据路径。

## 有效热区

[configs/default.yaml](./configs/default.yaml) 当前针对 A7 的 `1993×1080` 图片，热区坐标为原图左上角起点的半开 XYWH：`x=592, y=0, width=809, height=1080`。图片尺寸不符或坐标越界会报错。标注软件可调整热区坐标，并从原始图重新生成已有标注的裁剪图。

## A7 标注

A7 本地图片位于 `数据集/A7-10-19/`。运行 `python .\app.py`（或双击 `启动WebUI.bat`）后打开 `http://127.0.0.1:7860`，在“图片标注”页扫描 `数据集` 并选择工作区即可。WebUI 直接读本机原图，无需上传。把每个工作区的原图放在所选根目录的直接子文件夹中，例如：

```text
数据集/
├── A7-10-19/
│   ├── frame001.png
│   └── ...
└── CameraB/
    ├── frame001.png
    └── ...
```

在 WebUI“图片标注”页填写根目录（例如 `数据集`）并扫描，选中工作区后即自动进入标注（该工作区同时用于训练）。图片只从本机原目录读取，不会上传或复制；标注数据与训练结果按文件夹名隔离保存为 `data/<工作区名>/labels.csv`、`data/<工作区名>/images/` 和 `runs/<工作区名>/`。例如选中 `A7-10-19` 时，标注输出为 `data/A7-10-19/`，模型训练输出为 `runs/A7-10-19/`。工作区名称必须是单层文件夹名；不同根目录下的同名工作区会共用同一输出目录，请避免名称冲突。

点击三个类别按钮（清晰 / 临界 / 模糊）会保存当前图并自动进入下一张；缩略图栏显示当前图及邻近最多 8 张，点击可直接跳转；序号框输入后回车也可跳转。标注时以有效目标的边缘是否清楚作为参考，不要判断“有没有目标”。裁剪图保存到 `data/A7-10-19/images/`。JSON 里的设备 `mark` 字段不是清晰度真值，不会自动导入。

此前 106 条 A7 “有/无有效目标”标注与当前任务不同，已移入 `archive/A7-target-presence-20261026/` 封存；原始 A7 图片没有改动。之前的训练结果亦不作为清晰度模型使用。请重新标注清晰度，不要把旧二分类标签转换或混入新训练集。

`labels.csv` 格式：

```csv
image_path,label
images/example.png,sharp
images/uncertain-focus.png,borderline
images/blurred.png,defocused
```

标注建议：覆盖不同设备、背景、距离、角度、目标大小和色标设置；特别收集正常对焦、明显失焦和两者之间的临界样本。用相机的对焦成功标志作参考记录即可，不能直接代替图像清晰度标签。没有代表性类别样本时不要训练或声称性能可靠。

检查标注：

```powershell
python -c "from thermal_focus.dataset import load_labels; f=load_labels(r'.\data\A7-10-19\labels.csv'); print('samples=',len(f),'classes=',f.label.value_counts().to_dict())"
```

## 训练

可选模型架构：`mobilenet_v3_small`（默认轻量）、`mobilenet_v3_large`、`efficientnet_b0`、`resnet18`。它们使用相同的数据拆分与类别定义；权重会保存架构标识，推理和评估根据 checkpoint 自动加载对应网络。建议先用同一训练/验证划分比较，再以独立数据评估，不应仅凭模型名称推断准确率。

三类均需有足够样本，训练/验证/测试按标签分层拆分：

```powershell
python scripts/train_classifier.py `
  --labels .\data\A7-10-19\labels.csv `
  --output .\runs\A7-10-19 `
  --architecture mobilenet_v3_large `
  --device cuda `
  --pretrained `
  --epochs 30 `
  --batch-size 16 `
  --input-size 224 `
  --seed 42
```

训练输出 `best.pt`、`last.pt`、拆分 CSV 与训练配置。若类别不足或拆分无法分层，脚本会报错。正式性能评估应使用与训练样本独立的设备或采集时段，不能用测试集调阈值。

## 推理

### Gradio 网页界面（推理、标注与训练）

安装网页依赖后启动本地 Gradio 页面：

```powershell
python -m pip install -e ".[ui]"
python app.py
```

Windows 下也可以双击项目根目录的 `启动WebUI.bat`，它会调用 `thermal-focus-quality` Conda 环境并启动本地 Gradio 服务。如果 Gradio 已在 7860 端口运行，脚本会复用现有服务并打开浏览器，不会重复启动；若端口被其他程序占用，会显示占用进程信息。关闭运行中的服务窗口即可停止服务。

打开 `http://127.0.0.1:7860`：

- **图片推理**：选择本地训练得到的 `.pt` 权重、CPU/CUDA 和一张或多张图片；页面显示分类状态、未校准模型分数、概率 JSON、原图热区预览和热区裁剪图。模型分数是最高类别 softmax 概率乘以 100，不是准确率或清晰度百分比。坏图会单独显示错误，不会被当作模型结果。
- **图片标注**：填入工作区根目录，扫描并选择含图片的子文件夹（选中后立即进入标注，无需额外按钮）；在网页中查看当前图和前后缩略图、跳转并标注三类。载入时只枚举图片路径，图像在显示或跳转时按需读取，不会预先解码整个工作区；图片不上传或复制，标注保存至对应的 `data/<工作区名>/`。
- **模型训练**：选择工作区后，页面自动使用其 `data/<工作区名>/labels.csv` 和 `runs/<工作区名>/`；选择模型架构（MobileNetV3-Small/Large、EfficientNet-B0、ResNet18）、epoch、batch size、输入尺寸、学习率和设备。训练以本机后台任务运行，可查看状态和日志；完成后模型保存在工作区的 `best.pt`、`last.pt`，推理会从 checkpoint 自动识别架构。
- 默认标注 CSV 为 `data/A7-10-19/labels.csv`，默认权重输入为 `runs/A7-10-19/best.pt`。CSV 必须包含三类足够样本；否则训练脚本会给出失败日志。
- “下载预训练权重”默认关闭，勾选后需要网络。Gradio 界面单次只运行一个训练任务。

Gradio 界面只绑定在本机 `127.0.0.1`，不要直接暴露到局域网或公网；界面允许启动本机训练进程并读取用户指定路径的文件。命令行 `predict.py` 与训练脚本仍可独立使用。

### 命令行推理

```powershell
python .\predict.py `
  --input ".\数据集\A7-10-19" `
  --output ".\outputs\A7-10-19" `
  --model ".\runs\A7-10-19\best.pt" `
  --device cuda
```

每张图生成原图热区预览 `.annotated.jpg`、实际热区 `.thermal.png` 和结果 `.json`。JSON 的 `status` 为 `sharp`、`borderline`、`defocused` 或 `uncertain`，`probabilities` 是模型 softmax 输出，不等同于校准置信度；`model_score` 是最高类别 softmax 概率乘以 100 并保留一位小数，也未经校准。

## 评估

在独立测试 CSV 上运行：

```powershell
python scripts/evaluate.py `
  --labels .\data\A7-10-19\test.csv `
  --model .\runs\A7-10-19\best.pt `
  --device cuda `
  --output .\runs\A7-10-19\test-report.json
```

评估包含三分类混淆矩阵、各类 Precision/Recall/F1、Macro-F1、失焦漏检率、清晰图误拒率、复核比例和推理延迟。小样本和训练集指标不能证明现场可靠性。

只用训练脚本导出的验证拆分校准置信度阈值，不要用测试集调整阈值：

```powershell
python scripts/calibrate_thresholds.py `
  --labels .\runs\A7-10-19\validation_split.csv `
  --model .\runs\A7-10-19\best.pt `
  --device cuda `
  --output .\runs\A7-10-19\threshold-calibration.json
```

脚本从验证集的 Macro-F1 选择推荐阈值，并报告各候选阈值的失焦漏检率、清晰图误拒率和复核比例。将该阈值写入配置前，应结合现场风险人工复核；校准结果不代表独立测试性能。

## 测试

```powershell
python -m pytest
```
