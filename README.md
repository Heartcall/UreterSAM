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

新转换的数据清单默认标为待审，训练入口会拒绝 `training_ready=false`；上面的训练示例只适用于已审核清单或既有基线数据。`laser` 不再被默认当作真实光纤，`stone-new` 与 `laser-fibre` 已支持。复核流程会单独保留激光光斑，并用 `255` 表示未知区域，避免将标注缺失误当作背景。

## 3. 视频演示

```powershell
python main.py demo --checkpoint runs\endosemis\best.pt --source D:\path\to\video.mp4 --output runs\demo.mp4 --show
```

也可以把 `--source` 设为 `0` 读取摄像头。画面以红色显示结石、青色显示光纤。

## 4. S0/S1 基线与数据复核

在项目根目录、已有 `uretersam` 环境中运行：

```powershell
python -m tools.evaluate_video --config runs/baselines/a0_20261027/config.json --output runs/baselines/a0_replay
python main.py prepare-review --config configs/dataset_sources_v2.json --output datasets/ureter_semis_v2_review_new
python -m unittest discover -s tests -v
```

输出目录必须是新的路径，工具不会覆盖旧结果。完整概率导出用于离线审计，压缩可能引入耗时尖峰；性能对照可加 `--no-probabilities`，其配置和结果应与完整导出分别记录。

新数据草稿包含激光光斑辅助掩膜、来源信息、分组线索、候选负样本联系表与 `review.csv`，不会自动把缺少标注的图像当作负样本。核实标注和原病例/视频来源后，使用 `tools.promote_review_dataset` 生成可训练清单。像素真值使用原视频分辨率的类别索引 PNG，支持调色板索引与 255 未知像素。

详见 [标注与复核协议](docs/evaluation/video_annotation_protocol.md)、[实施计划](docs/plans/segmentation_temporal_occlusion_plan.md)和[当前进展](docs/progress/2026-10-27-s0-s1.md)。当前事件区间来自用户反馈，事件标签不等于像素级真值；无真值视频只输出预测诊断，不计算分割准确率。

## 5. S2 训练准备与断点恢复

A1 的固定配置位于 `configs/train_a1_v1.json`，目前需要等待已审核的 `manifest.reviewed.json`。配置启动器 `python -m tools.train_experiment --config configs/train_a1_v1.json --output runs/a1_seed42` 会执行相同的数据审核检查。可用 `--stop-after-epoch 1` 在完整一轮后停止，随后使用相同配置和 `--resume runs/a1_seed42/last.pt` 继续；总轮数和调度不能在续训时更改。

`last.pt` 保存完整优化器、学习率、AMP、随机源与数据加载器状态；`best.pt` 供推理，不能代替完整训练断点。新训练拒绝覆盖非空目录，续训会核对配置、输入文件、源码与环境身份。原有不含续训状态的旧权重继续用于推理。

`tools.compare_checkpoints` 在同一已审核验证集上计算全局混淆矩阵和无结石图像误检统计；不使用测试像素，未定义指标保存为 null。真实 GPU 已完成 64/256 分辨率的合成数据续训检查，但正式 A1 重训尚未启动。详见 [S2 工程记录](docs/progress/2026-10-27-s2-infrastructure.md)。

## 6. 本地标注复核

运行 `python -m tools.review_server --port 8767`，浏览器打开 <http://127.0.0.1:8767>，可复核 29 帧关键帧及 2847 张训练草稿。支持画笔、多边形、实例身份与遮挡状态、版本保存和审核导出；来源待审的训练样本不能自动进入训练。详见 [工作台说明](docs/evaluation/review_workspace.md)。

## 7. S2 背景修正与纯监督检查

用户明确指定 **stone_new 停止用于训练，因为它只标注光纤尖端**。来源排除时的历史清单保留 kidney_stones_fiber_v1 的 190 张监督图像和 800 张备用图像；当前活动联合版本为521张监督训练、132张图像开发验证、144个原test关联保留组，见本节后续更新。**video_1 和 video_2 仅作验证，不参与任何训练或伪标签学习**；29 帧审核掩膜和完整视频的用途不变。原始标签与病例来源审核状态保持原样。

已有混合来源实验均已结束，仅保留历史结果，不续训、不用于新模型初始化。训练入口按名称、路径及已知图像哈希拒绝 stone_new，旧清单不能误用于后续训练。按[计划 v2.1](docs/plans/segmentation_temporal_occlusion_plan.md)，已从干净 ImageNet 初始化完成 kidney-only 单网 15 轮基线：选中第 14 轮的结石 Dice 0.8174、光纤 Dice 0.8308、负帧误检面积 0.1724%；末轮光纤 Dice 0.9027，仍未通过光纤门槛。该轮未验收，不延长或进入 S3；后续对照见下文。详见[基线结果](docs/progress/2026-10-27-kidney-only-baseline.md)；旧混合对照见[历史记录](docs/progress/2026-10-27-fiber-learning-ablation.md)。

原 A0 权重保留。新配置为 `configs/train_a1_sup_kidney_only_20261027.json`，新清单为 `datasets/ureter_semis_a1_sup_kidney_only_20261027/manifest.json`；后续导出默认使用 kidney-only 来源配置。详见[来源排除记录](docs/progress/2026-10-27-stone-new-training-exclusion.md)。

后续已完成逐帧光纤误检定位，以及共享类别权重 4→2 的固定 15 轮对照。同末轮光纤 FP 从 9913 减到 5998，但 Recall 从 0.9342 降到 0.8773，Dice 从 0.9027 降到 0.8955，因此不采用降权设置。详见[降权实验记录](docs/progress/2026-10-27-kidney-fiber-fp-weight-screen.md)。

2026-10-28 已完成 A1-DUAL 与相同总更新数的单网30轮对照。固定末轮双网结石/光纤 Dice 为 0.8081/0.9161，推理中位9.54 ms；单网15轮为0.8110/0.9027、5.07 ms，单网30轮为0.7908/0.9344。新候选均未通过共同门槛，暂不换成双网，不进入记忆阶段。该轮800张备用图像未读取，后续诊断见下文。详见[预算与成本记录](docs/progress/2026-10-28-kidney-dual-budget-screen.md)。

随后已完成190张监督图像与800张备用图像的只读伪标签诊断。三模型及增强一致筛选的光纤来源标签一致度94.12%，未达到预先固定95%条件，半监督训练暂缓；800张已用于诊断、仍未参与训练，验证视频未生成训练伪标签。5轮配对短程协议已冻结但不可执行。详见[伪标签准备记录](docs/progress/2026-10-28-kidney-ssl-readiness.md)。

A1-ALT已完成SMP0.4 SegFormer-B0与原U-Net的固定15轮同源监督对照。新模型结石/光纤Dice为0.7696/0.6404，低于U-Net的0.8110/0.9027，边界及光纤召回更差、实测推理未提速，因此不替换默认模型。以上为历史PIL/256网格指标。模型加载已支持显式架构标识，旧A0及原U-Net行为精确复现。详见[架构对照记录](docs/progress/2026-10-28-kidney-alt-segformer-screen.md)。

已完成U-Net256/512各15轮的统一原图评价对照：GT不缩放，预测概率映射回原图后取类别。固定末轮结石Dice为0.8100/0.8408，光纤Dice为0.9038/0.8731；512的光纤召回降至0.7884，中位原图输出延迟6.37→11.94ms。因此不采用全图512，S2仍未验收。训练仅190张kidney图像，视频仅验证，stone_new及800张备用图像均未参与本轮训练。详见[原图分辨率对照记录](docs/progress/2026-10-28-kidney-native-resolution-screen.md)，其指标不能与上面的历史模型网格结果混比。

此前完成六组自动局部精修ROI诊断：2窗口覆盖约90%结石，最差帧48.35%；增加窗口仍有1帧仅87.68%，均未同时满足覆盖与预算条件。29帧光纤均在自动区域内，但串行U-Net精修计算代理中位28–33ms，未显示速度优势；覆盖率不代表精修准确率。该轮无训练更新，未启动联合精修。详见[自动ROI诊断记录](docs/progress/2026-10-28-kidney-automatic-roi-feasibility.md)。

随后按用户授权完成v1/v2联合数据版本：2387张导出图像保守去重成797组，形成521张监督训练、132张图像开发验证、144个原test关联保留组，原29帧视频验证保持不变。新U-Net从ImageNet初始化，固定1425次更新/2850次图像使用，实际见519/521张。末轮视频原图结石/光纤Dice0.7695/0.8863，低于旧190基线0.8100/0.9038；图像开发验证为0.9057/0.8030，两个集合分别报告。新模型未采用，S2未通过；下一步先定位联合数据的结石漏检、光纤误检和标注分歧。详见[联合数据同预算基线记录](docs/progress/2026-10-28-kidney-joint-matched-budget-baseline.md)。

最新只读诊断已完成：结石FN净增6869且多数变为背景；光纤FP净增9181，主要增量靠近参考轮廓，但仍有远处腔壁误检。11组标签数量分歧并非11个已证实错标；5组保留test只看元数据，活动6组仅2组严格对齐，且未显示光纤与另一来源激光光斑重合。v1的裁剪/旋转/剪切增强、来源目标尺度与固定预算曝光密度均需区分，不据此停用v2或加训练。其后执行了旧190结石+A0光纤的固定类别组合及三网总成本对照，结果见下一段。详见[联合数据失败诊断](docs/progress/2026-10-28-kidney-joint-failure-diagnosis.md)。

最新已完成类别分工三网对照：29帧原图结石/光纤Dice为0.8106/0.9668，负帧误检面积0.0331%，三项预设初筛均通过。全部3网中位20.86ms、p9533.03ms，未含解码/显示；小目标仍漏1/3帧，132图像验证的光纤Dice仅0.7623，未验收S2或替换默认模型。新增3项测试与26项执行核验通过。下一步按固定协议完成两段视频2694帧回放、指定失败片段及总延迟检查，再决定底座/光纤精修；尚未加入时序记忆、未进入S3。详见[类别分工结果](docs/progress/2026-10-28-classwise-expert-control.md)。

最新完整回放已处理2694帧，导出三组8082张原图掩膜及5段用户区间；29帧RGB精确对齐，组合分割指标与上轮完全一致。视频1遮挡区间仍有显示帧367–368、403结石预测为空；三网核心p95为33.20/34.55ms，含诊断/四栏导出的离线吞吐23.59/23.23fps，尚未验证实时显示或遮挡记忆。3项新增测试及33项执行核验通过。下一步只做有限的A0光纤单成员两网成本对照，不训练、不自动替换默认或进入S3；若质量不达标则回到光纤轻量精修设计。详见[完整视频回放结果](docs/progress/2026-10-28-classwise-full-video-replay.md)。

## 实现说明

模型骨架复用了官方 [Endo-SemiS](https://github.com/MedICL-VU/Endo-SemiS) 的定制 U-Net，并把原论文的二分类输出改为 `background/stone/fiber` 三分类输出。训练保留双网络、Monte Carlo dropout 不确定度筛选、交叉伪标签和联合伪标签机制。官方肾结石数据与权重未公开，因此此项目使用可下载的 Roboflow 标注数据建立演示流程。
