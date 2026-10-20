# UreterSAM demo

这是一个面向输尿管软镜激光碎石视频的 Endo-SemiS 简化实现，只分割两个前景类别：

- `stone`：结石
- `fiber`：激光光纤

背景为类别 0。Roboflow 数据中的 `tissue` 和 `laser` 类在转换时会被忽略。

## 1. 准备数据

从 [Roboflow Kidney Stones Fiber](https://universe.roboflow.com/ilk-awfpp/kidney-stones-fiber) 下载 **YOLOv8 Segmentation** 格式，或准备 COCO polygon 数据并解压，然后运行：

```powershell
python main.py prepare --source D:\path\to\export --format auto --output datasets\ureter_semis --labeled-fraction 0.2
```

独立转换器位于 [tools/convert_dataset.py](D:/ProjectProgram/UreterSAM/tools/convert_dataset.py)，会生成单通道类别掩膜以及 `manifest.json`。`auto` 会根据 YAML 或 COCO annotation JSON 判断格式。默认随机选取 20% 训练图像作为有标签集，其余图像作为半监督训练中的无标签集；验证集仍全部使用标签。若需要强制指定格式，可使用 `--format yolo` 或 `--format coco`。

## 2. 训练

```powershell
python main.py train --manifest datasets\ureter_semis\manifest.json --output runs\endosemis --epochs 60 --size 256 --batch-size 2
```

默认使用 `cuda:0`、256×256 输入、ResNet34 U-Net 双模型和混合精度。监督损失为带类别权重的 Cross Entropy + 多分类 Dice，默认权重为 `1,2,4`（背景、结石、光纤）。无标签伪标签同时使用 softmax 置信度阈值 `0.85` 和低熵筛选。输出包括 `best.pt`、`last.pt` 和 stone/fiber 的 Dice、IoU、Precision、Recall。确认 256×256 pipeline 后，可以将 `--size` 改为 `512`。

## 3. 视频演示

```powershell
python main.py demo --checkpoint runs\endosemis\best.pt --source D:\path\to\video.mp4 --output runs\demo.mp4 --show
```

也可以把 `--source` 设为 `0` 读取摄像头。画面以红色显示结石、青色显示光纤。

## 实现说明

模型骨架复用了官方 [Endo-SemiS](https://github.com/MedICL-VU/Endo-SemiS) 的定制 U-Net，并把原论文的二分类输出改为 `background/stone/fiber` 三分类输出。训练保留双网络、Monte Carlo dropout 不确定度筛选、交叉伪标签和联合伪标签机制。官方肾结石数据与权重未公开，因此此项目使用可下载的 Roboflow 标注数据建立演示流程。
