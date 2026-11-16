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
| `uncertain` | 模型判定置信度不足、模型不可用或输出异常；不用于训练 |

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
├── data/          # 唯一数据集目录（v1 基础数据、标注、版本快照和模型输出）
├── runs/          # 旧版模型输出目录，仍会被模型列表扫描
├── scripts/       # 推理、训练、评估和阈值校准
├── src/           # Python 源码模块直接位于此处，WebUI 界面代码在 webui/ 子目录
├── tests/         # 自动化测试
├── 数据集/        # 原始图片与设备元数据
└── archive/       # 已封存的旧标注和历史训练产物
```

## 有效热区

[configs/default.yaml](./configs/default.yaml) 为命令行和图片推理提供默认热区配置，默认在原图居中裁剪 `809×1080`。命令行推理、图片推理和标注页都会先检测每张图片的近黑边界；未检测到有效边框时，推理使用配置中的热区裁剪，标注页使用配置中的宽高居中裁剪。`data/annotations.sqlite3` 保存标注与裁剪坐标，`data/images/` 保存标注裁剪图。手动调整坐标时，位置和宽高的上下限随图片尺寸及当前裁剪框动态更新，且服务端会校验越界输入。点击“应用坐标预览”只临时预览；点击“保存当前图片裁剪并更新标注”会保存当前图片参数，并在该图已有标注时仅重裁该样本，不会修改全局 YAML。

## A7 标注

A7 本地图片位于 `数据集/A7-10-19/`。运行 `gradio app.py -- --enable-annotation` 后，本机打开 `http://127.0.0.1:7860` 使用图片标注；默认启动不启用标注。WebUI 始终使用项目的 `data/` 作为唯一数据集目录；现有 `data/` 中的已标注数据会作为 v1 基础数据，之后通过“数据集版本”页导入的图片生成后续版本。标注页载入 `data/` 下识别到的原始图片目录。

```text
数据集/
├── A7-10-19/
│   ├── frame001.png
│   └── ...
└── CameraB/
    ├── frame001.png
    └── ...
```

打开图片标注页时会自动载入 `data/` 中的原始图。多个浏览器连接同一个服务时共享未标注队列，同一张图不会同时分配给多个页面；领取任务闲置 30 分钟后会自动释放。标注标签和裁剪图先保存在 `data/`。数据集版本页导入新图片时会将新增标注并入新版本；版本快照和图片资产位于 `data/dataset_versions/`。

提交新版本后会使用训练默认参数自动训练，模型分别保存在 `data/runs/dataset-vNNNN/`，模型列表显示对应的数据集版本。训练任务在当前 WebUI 服务内串行执行：已有任务时新版本按提交顺序排队，手动训练在队列非空时不能启动。版本页刷新可查看任务状态，导入结果包含任务 ID，可通过 `training_status` API 查询日志；训练失败不会阻止后续任务。队列和任务状态保存在服务进程内，重启服务会丢失待执行任务；已创建的数据集版本仍可在回档后手动训练。请勿同时启动多个 WebUI 服务或外部训练脚本，否则它们不受此队列约束。

局域网访问目前不带登录认证，请仅在可信网络中开放，并确保主机防火墙只允许可信设备访问 WebUI 端口。

点击三个类别按钮（清晰 / 临界 / 模糊）会保存当前图并自动领取下一张；“领取下一张”可领取其他网页尚未分配的待标注图片。缩略图栏显示当前图及邻近最多 8 张，点击可直接查看；序号框输入后回车也可跳转。标注时以有效目标的边缘是否清楚作为参考，不要判断“有没有目标”。JSON 里的设备 `mark` 字段不是清晰度真值，不会自动导入。

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

### Gradio 网页界面（推理、标注、数据集版本与训练）

安装网页依赖后启动本地 Gradio 页面：

```powershell
python -m pip install -e ".[ui]"
gradio app.py
```

默认启动数据集版本和图片推理界面，绑定本机 `127.0.0.1:7860`；图片标注和模型训练页面默认关闭。需要时使用 `gradio app.py -- --enable-annotation` 启用标注，或 `gradio app.py -- --enable-training` 启用手动模型训练；两个参数可同时使用。也可运行 `python app.py --enable-annotation --enable-training`。自定义端口时加上 `--port 8080`；使用 `gradio app.py` 时，`--` 后的参数会传给应用脚本。旧的 `--inference-only` 参数已移除。

Windows 下也可以双击项目根目录的 `启动WebUI.bat`，它会调用 `thermal-quality` Conda 环境并启动本地 Gradio 服务。如果 Gradio 已在 7860 端口运行，脚本会复用现有服务并打开浏览器，不会重复启动；若端口被其他程序占用，会显示占用进程信息。关闭运行中的服务窗口即可停止服务。

打开 `http://127.0.0.1:7860`：

- **图片推理**：选择一张或多张图片后，图库先显示原图，不显示推理详情；推理完成后显示标注和分类结果。重新选择图片会清除上一次的结果。页面每次推理都会扫描 `data/` 和旧版 `runs/`，自动使用修改时间最新的 `best.pt` 权重；没有 `best.pt` 时提示先训练。设备自动选择：CUDA 可用时使用 CUDA，否则使用 CPU。页面显示所用模型、分类状态、各类别概率和原图热区预览。推理会先检测热图黑边并裁掉，未检测到有效黑边时使用默认配置热区；多张图片会并发准备热区，并按最多 16 张一批运行模型推理。低于配置的置信度阈值时显示“无法确定”。坏图会单独显示错误，不会被当作模型结果。
- **图片标注**（需要 `--enable-annotation`）：载入固定的 `data/` 数据集目录，浏览原始图片并标注三类；标注结果写入 `data/annotations.sqlite3`。图片按需读取，多个页面共享未标注队列。
- **数据集版本**：`data/` 当前内容初始化为 v1；分别上传清晰、临界、模糊类别图片，一次提交按当前热区配置裁剪图片、合并新增标注并创建新版本。版本快照和共享图片保存在 `data/dataset_versions/`。回档仅切换当前生效版本，不删除历史版本；之后导入仍继续递增版本号。
- **模型训练**（需要 `--enable-training`）：始终读取 `data/` 当前生效版本，模型输出到 `data/runs/dataset-vNNNN/`。可选择模型架构（MobileNetV3-Small/Large、EfficientNet-B0、ResNet18）、epoch、batch size、输入尺寸、学习率和设备。训练以本机后台任务运行，可查看状态和日志；回档数据集不会自动切换推理模型。即使未启用此页面，提交新数据集版本仍会自动排队训练。
- WebUI 标注数据库和模型均保存在 `data/` 下；训练数据来自当前生效的数据集版本。数据库必须包含三类足够样本；否则训练脚本会给出失败日志。
- “下载预训练权重”默认开启，首次下载需要网络；取消勾选则从随机初始化开始。Gradio 界面单次只运行一个训练任务。

除页面内部同步事件外，Gradio API 文档还提供具名业务接口：`/upload_images`、`/predict_images`、`/list_models`；数据集版本的 `/dataset_version_status`、`/import_labeled_dataset`、`/activate_dataset_version`；以及查询自动训练任务的 `/training_status`。使用 `--enable-training` 时另外提供手动训练接口 `/start_training`。使用 `--enable-annotation` 时另外提供标注接口 `/start_annotation`、`/annotation_status`、`/navigate_annotation`、`/claim_next_unlabeled`、`/save_annotation`、`/update_annotation_roi`，标注接口通过会话 ID 关联操作，图片序号从 1 开始；导入接口按类别分别接收图片文件；训练启动接口返回任务 ID，再用状态接口轮询进度。`/predict_images` 只接收图片列表，和页面一样自动选用修改时间最新的 `best.pt` 及可用设备。**接口变更**：此前传入 `model` 和 `device` 的调用方需要移除这两个参数；返回的状态、图库和结果表格式不变。

通过 `gradio_client` 调用时，使用 `Client("http://127.0.0.1:7860")`，并按 API 文档中的示例传递参数和 `api_name`。服务只绑定本机回环地址，不向局域网或公网开放；界面和 API 均允许启动本机训练进程并读取用户指定路径的文件。命令行 `scripts/predict.py` 与训练脚本仍可独立使用。

`/upload_images` 接收非空图片文件列表并校验图片，返回服务端临时路径、文件名和大小；文件只保存在 Gradio 临时缓存，可能被清理，不会自动加入数据集或触发训练。示例：

```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7860")
uploaded = client.predict(
    files=[handle_file(r"C:\path\to\image.jpg")],
    api_name="/upload_images",
)
print(uploaded)
```

进行推理或创建数据集版本时，可直接向对应接口上传原始图片，无需先调用 `/upload_images`。

`/import_labeled_dataset` 的 `dataset_root` 是服务端已存在的数据集目录绝对路径；三个文件列表依次对应清晰、临界、模糊，没有图片的类别传 `[]`。例如：

```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7860")
result = client.predict(
    dataset_root=r"C:\path\to\thermal-quality\data",
    sharp_files=[handle_file(r"C:\path\to\sharp.jpg")],
    borderline_files=[],
    defocused_files=[],
    api_name="/import_labeled_dataset",
)
print(result["active_version"], result["training_job"])
print(client.predict(job_id=result["training_job"]["job_id"], api_name="/training_status"))
```

接口按当前生效版本创建并启用新版本，合并已有标注并以默认参数自动排队训练。图片格式、大小限制、去重规则、返回字段和失败处理详见 API 页面；版本创建成功不代表训练成功，须通过 `/training_status` 检查。

### 命令行推理

```powershell
python scripts\predict.py `
  --input ".\数据集\A7-10-19" `
  --output ".\outputs\A7-10-19" `
  --model ".\runs\A7-10-19\best.pt" `
  --device cuda
```

每张图生成原图热区预览 `.annotated.jpg`、实际热区 `.thermal.png` 和结果 `.json`。JSON 的 `status` 为 `sharp`、`borderline`、`defocused` 或 `uncertain`，`probabilities` 保存三个类别各自的模型概率；结果不返回汇总分数。

目录推理会以最多 16 张图片为一批进行模型前向计算，并并发准备每张图的热区；批次大小有上限以控制内存和显存占用。

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
