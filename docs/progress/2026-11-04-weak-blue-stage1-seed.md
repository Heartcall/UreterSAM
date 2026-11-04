# S1：逐块验证后再判唯一主体

2026-11-04，研究分支，未改变默认、权重、训练数据或历史结果。执行协议与参数先冻结于 `docs/plans/weak_blue_staged_execution_20261104.md` 和 `configs/weak_blue_staged_control_20261104.json`；本阶段源码、测试及依赖哈希/快照见输出 preflight.json/source_snapshot.zip。

使用原P≥0.9、4连通面积≥16及全部原长度、形状、颜色、背景、外边界资格规则，逐块判断后仅允许一个合格主体；原组件label/core保留。最多32块是计算预算，超出未知，不选择最大块。原实现及阈值不改。6项回归测试通过。

两段分别在独立双网络进程完成722/1972帧，12帧预热后重置；所有原始RGB、raw及两类概率与历史哈希一致。70帧静态、70帧因果视频和132张图像评测全部完成，结石和union保护损失均为0。

| video_2一基帧 | 旧入口 | 新入口 | 新weak相对修复后union像素 | 原因/帽 |
|---|---|---|---:|---|
|317|3块提前拒绝|唯一合格|1275|外终止证据不足，弱候选|
|770|3块提前拒绝|唯一合格|379|RGB确认，椭圆残差拒绝|
|1204|3块提前拒绝|唯一合格|803|接受帽模型|
|1716|3块提前拒绝|唯一合格|374|外终止证据不足，弱候选|

全片video_1零帧变化，video_2新增52个可处理帧、52帧实际变化；其余旧非数量拒绝帧逐像素不变。不能将1275等数字视为新增TP：这些帧没有独立像素参考。video_2处理资格开关44→10次，面积步长p95 1019→878px，属于连续性代理指标，不能证明真实运动或精度。

旧数量拒绝的53帧中还有第769帧未解除：逐块验证得到2个qualified和2个短小/非细长组件，仍按歧义保留回退。这是全片实际证据，说明新入口不是把组件数量上限提高后强选最大块。

参考结果与旧weak-tip完全相同：70因果光纤Dice/Precision/Recall=0.958472/0.928881/0.990011，TP/FP/FN=448891/34369/4529，BoundaryF=0.925159；132图像为0.755957/0.628515/0.948225，TP/FP/FN=105583/62405/5765，BoundaryF=0.487950。低Lab参考FN没有变化。来源分组及逐帧指标保存在各scope summary及analysis目录。FP/Precision/Dice仅诊断，不作本轮门槛。

P50/P95双网加处理延迟：video_1=244.97/329.78ms，video_2=170.65/319.75ms；包括存储编码的实际吞吐3.86/4.12fps。相对历史P95满足≤1.5倍的继续条件，但仍不满足实时要求。全片共332个帽模型接受、209帧有实际帽像素变化；侧边仍未应用。

共同坐标下原722/1915个可处理帧的侧HF、RGB支持均逐像素不变；新启用52帧不混入这组配对平均。新增帧的分割、资格、像素及尖端状态均有独立完整日志，不能用共同子集掩盖新增风险。重点4帧和最大新增组317/422/426/393/418/1107/1101/1106以及1203/1205进行了RGB对比审查，未见新增主体明显扩张到腔壁/结石；光晕、细小毛刺和原始游离误检仍存在。此审查不是专家尖端标注。

全部冻结非退化检查通过，允许进入S2修复真实坐标梯度；这不是采用或准确率通过结论。原网络游离误检受保护仍在，修复身份入口不能解决它们。无独立尖端真值，不报告物理尖端精度。

输出根目录：`runs/a1_weak_blue_staged_20261104/s1_seed_qualified/`。`video_1_comparison.mp4`、`video_2_comparison.mp4`为原图/双网/旧weak-tip/S1四格全片；`focus1233-1291.mp4`为59帧重点区间。媒体按真实sourcePTS同步，保留video_2重复初始PTS/VFR。独立mask、图层压缩包、逐帧状态、定量评审和继续决定全部保留。

![第317帧](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/s1_seed_qualified/focus/video_2_frame0317.png>)

![第770帧](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/s1_seed_qualified/focus/video_2_frame0770.png>)

![第1204帧](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/s1_seed_qualified/focus/video_2_frame1204.png>)

![第1716帧](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/s1_seed_qualified/focus/video_2_frame1716.png>)
