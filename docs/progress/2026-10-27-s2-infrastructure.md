# S2 训练基础与可复现性检查

日期：2026-10-27。承接 [S0/S1 记录](2026-10-27-s0-s1.md)。本轮完成 S2 的工程准备和合成数据检查；**没有启动真实数据 A1 重训，没有生成新的临床分割效果结论，S3 尚未启动**。

## 已实现

### 固定配置和可恢复训练

- [A1 配置](../../configs/train_a1_v1.json)保留双 ResNet34 U-Net、256×256、batch 2、60 轮、AdamW 和原伪标签设置；预先记录复核种子集合 42、123、2026，主试验为 42。
- 固定 Python、NumPy、CPU/CUDA 随机源、cuDNN 与确定性算法；三个 DataLoader 分别持有随机生成器，并为子进程设置随机种子。
- 运行时记录训练配置、清单 SHA256、训练/验证实际文件哈希、核心源码哈希及依赖/设备信息。测试像素不被训练或验证工具读取。
- `last.pt` 保存两个网络、两个优化器、调度器、AMP scaler、所有随机状态、DataLoader 状态、训练历史与最佳轮次。恢复时严格核对配置、数据字节、源码与运行环境；改变总轮数会改变余弦调度，因此不允许直接续接。
- 支持轮次边界恢复；轮次中间中断时从上一完整轮次继续。`--stop-after-epoch` 可以提前停在完整轮次，仍使用原定的总轮数调度。
- 原子写入 `last.pt`；每次入选权重保存在 `selected/epoch_XXXX.pt`，保证更新中断不会丢失旧断点所引用的最佳模型。`best.pt` 是便于推理读取的当前最佳权重副本，`last.pt` 才包含完整续训状态。
- 新试验拒绝非空输出目录；在原目录续训时必须明确指定该目录的 `last.pt`。也支持从断点恢复到新目录，保留原最佳权重。
- 旧权重仍可用于推理，但缺少完整状态，不能声称可以精确恢复训练。

入口：[trainer.py](../../ureter_semis/trainer.py)、[experiment.py](../../ureter_semis/experiment.py)、[配置启动器](../../tools/train_experiment.py)。

### 损失与评价

- 保留真实无结石样本的监督，不删除背景样本；255 未知像素不参与监督或评价。
- 实际 GPU 检查发现，当前 PyTorch/CUDA 的默认二维交叉熵归约不支持严格确定性模式。已使用 log-softmax、类别索引和显式加权归约表达等价交叉熵，并检查数值与梯度；Dice/交叉熵归约使用 float32。
- 验证只累积一个全局混淆矩阵，不平均各批次 Dice。保存结石/光纤 TP、FP、FN、Dice、IoU、Precision 和 Recall。
- 分母为零的指标保存为 null，前景宏平均只纳入定义有效的类别，并记录类别数；不能与旧日志中把不存在类别记作 0 的均值直接比较。
- 单独记录无结石图像的误检图像数、比例、有效像素数和结石误检像素比例。结石是否存在在原图掩膜上判断，避免小结石缩小后消失而被归成负样本；带未知区域的无目标掩膜不会自动确认为无结石。
- 新增 [统一权重对比工具](../../tools/compare_checkpoints.py)，在同一已审核验证集、相同模型网格和预处理下重新计算 A0/A1 指标，并输出相对首个模型的差值。它不读取测试集像素，也不修改新目标接受阈值。

图像验证沿用训练器的 PIL 双线性预处理；原始视频 A0 保留 OpenCV 预处理以维持回归结果。两个口径分别记录，视频效果继续使用独立的视频标注评价工具。当前负样本像素统计范围是有效标签区域，若黑边未被标为 255，也会计入；该统计不能单独代替腔壁分层评价。

## 已执行的检查

自动测试共 **31 项通过**。新增检查覆盖：完整训练与分段续训逐项一致、在原目录续训、恢复到新目录、保留较早最佳轮次、拒绝数据/种子变化、子进程随机源重启、交叉熵数值/梯度、全局指标、未知区域、原图小目标、待审清单拒绝、配置种子范围和顺序解码。

在 RTX 4070 Ti 上，另使用**真实双 U-Net、随机初始化、合成图像及合成标签**执行完整两轮与一轮后恢复的对照。每轮包含一次实际优化更新，使用 CUDA 混合精度。结果如下：

| 检查尺寸 | 完整断点逐位相同张量 | 最佳权重逐位相同张量 | 新权重推理兼容 | PyTorch 峰值已分配显存 |
|---|---:|---:|---|---:|
| 64×64 | 1456 | 574 | 通过 | 约 0.89 GiB |
| 256×256 | 1456 | 574 | 通过 | 约 1.48 GiB |

这只是当前硬件/依赖、短合成数据试验的工程结果，不证明跨平台逐位复现或长训练显存上界，也不证明分割准确率提升。

结果：[64×64 检查](../../runs/engineering/training_resume_20261027_v2/summary.json)、[256×256 检查](../../runs/engineering/training_resume_256_20261027/summary.json)。两个试验均通过统一权重对比入口，完整训练/续训权重的指标差值为零：[256×256 对比记录](../../runs/engineering/training_resume_256_20261027/comparison.json)。最初因 CUDA 确定性交叉熵失败的试验目录保留为诊断记录，不能作为通过结果。

### 修复复核抽帧对齐

检查原始 A0 时发现 `video_2` 随机跳转到编号 100 后得到的帧，与从头顺序解码的编号 100 图像不同（图像通道数值最大差 53）。随机跳转图像的预测与原基线有 436 个不同标签像素，顺序解码图像的预测则完全一致。这是抽帧解码对齐问题，不能作为模型回归失败或改进证据。

已将用户复核集及后续自动候选短片导出改成顺序解码，从头跳过非目标区间，禁止倒序或重叠读取。当前复核集为 [user_events_20261027_v3](../../runs/review_sets/user_events_20261027_v3/manifest.json)，保留 6 段、29 帧及待审状态，附原视频 SHA256 与原始 BGR 解码帧 SHA256。旧复核集和旧自动候选短片不再作为标注依据；原基线的顺序推理掩膜与完整对照视频不变。

最终重新顺序读取两段视频，核对全部 29 帧原图、解码哈希和模型预标注对齐，确认其中 3 帧相对 `_v2` 得到更正；抽取 10 个 A0 帧重新推理，掩膜与冻结基线逐位一致，原权重 SHA256 未变化。29 个项目 Python 文件通过语法检查。[最终检查记录](../../runs/engineering/s2_checks_20261027.json)。

## 后续执行入口

以下命令供审核清单形成后使用；目前 `manifest.reviewed.json` 尚未生成，因此不能把这些命令的准备视为正式训练已完成。

```powershell
python -m tools.train_experiment --config configs/train_a1_v1.json --output runs/a1_seed42

# 总调度仍是配置中的 60 轮，先完整运行一轮
python -m tools.train_experiment --config configs/train_a1_v1.json --output runs/a1_trial --stop-after-epoch 1
python -m tools.train_experiment --config configs/train_a1_v1.json --output runs/a1_trial --resume runs/a1_trial/last.pt

python -m tools.compare_checkpoints --manifest datasets/ureter_semis_v2_review_20261027/manifest.reviewed.json --checkpoints runs/endosemis/best.pt runs/a1_seed42/best.pt --output runs/a1_seed42/a0_a1_validation.json
```

需要重新做工程检查时，使用新的输出目录：

```powershell
python -m tools.check_training_resume --output runs/engineering/training_resume_new --size 256
python -m unittest discover -s tests -v
```

后续仍遵循原计划依赖：完成 S1 的标注完整性/负样本审核与来源分组，形成正式训练清单；启动 A1 后校准接受条件并进行单帧对照。通过后再进入 S3 运动补偿与时间融合。来源无法确认的数据保持待审，不把文件名前缀或模型预标注升级为真值。
