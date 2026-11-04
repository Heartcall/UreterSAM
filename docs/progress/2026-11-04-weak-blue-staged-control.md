# 弱蓝扩展与尖端修整：分阶段执行与停止报告

2026-11-04。阅读当前链路和[前轮诊断](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/docs/progress/2026-11-04-weak-blue-seed-tip-diagnosis.md>)后，先冻结[分阶段协议](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/docs/plans/weak_blue_staged_execution_20261104.md>)、参数和依赖，再逐阶段实施。**完成S1/S2/S3，每阶段均完成全2694帧与70/132参考；S3功能评审暂停，S4实际轮廓编辑未实施。没有改默认分割、正式输出、模型权重、训练数据或历史结果。**

## 分阶段结果

| 阶段 | 实际实现 | 结论 |
|---|---|---|
|S1|逐块原资格验证后判断唯一主体，原label/core保留|317/770/1204/1716恢复处理；video_2新增52个可处理帧，处理开关44→10；旧非数量拒绝帧完全相同；继续条件通过|
|S2|原图Sobel一次计算，端点/帽按真实坐标采样|全片弱蓝主体与S1逐像素相同；1250内部支持4/6→6/6、帽增5px，1291从3/6→6/6但帽接缝仍拒绝；继续条件通过|
|S3|低支持新增weak叶节点剪除，连续带前沿和截面宽度自适应取样|量化分割门槛通过，但确认覆盖1720→1353；418个原确认丢失，326个仍满足当前RGB端部数值标准。不能确认无功能退化，不采用、不进入S4|
|S4|受限实际左右轮廓/端帽像素编辑|**未实施、未运行**；没有把画出的拟合线声称为实际修整|

详细变更、逐帧失败原因、焦点图和评审决定：[S1报告](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/docs/progress/2026-11-04-weak-blue-stage1-seed.md>)、[S2报告](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/docs/progress/2026-11-04-weak-blue-stage2-gradient.md>)、[S3报告](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/docs/progress/2026-11-04-weak-blue-stage3-coherent-tip.md>)。

S1保留第769帧的两个合格候选歧义，未强选最大块。S2两视频全片只发生帽差异，新增2901px、删8个旧实验帽像素，原union/结石损失0。S3剪枝步骤177px，后续帽又加入5px，最终剪枝移除172px；70参考中含4个已标光纤像素损失，不能只报132图像的11个参考FP减少。

## 为什么在S3停止

1250的连通末端团79px、前沿14px，内部6/6支持、外部0/9蓝色，RGB对比0.1423、score下降0.5725，却因1个保留像素超出支持前沿0.5084px而被降级。770对应仅0.1110px。末端bin密度和“任意更远像素即降级”的组合，把充分的当前外终止证据提前覆盖了。

这是**确认可用性下降**和逻辑归因，不是宣称物理端点准确率已下降；没有独立尖端真值，也不能声称减少了错误端点。冻结的TP/低亮度Recall/BoundaryF/RGB边界/HF/延迟等数字条件全部通过，仍不足以满足用户要求的阶段整体无明显退化审核。因此不偷偷调整密度阈值、farther距离或继续叠加轮廓处理。S3 `review.json`区分quantitative_passed=True、progression_passed=False，避免把功能暂停冒充数字门槛未过。

## 70帧视频参考（完整因果回放抽取）

| 方案 | 光纤Dice | Precision | Recall | TP | FP | FN | BoundaryF | 低Lab FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|旧weak-tip / S1|0.958472|0.928881|0.990011|448891|34369|4529|0.925159|2343|
|S2|0.958472|0.928843|0.990053|448910|34390|4510|0.925161|2340|
|S3|0.958472|0.928831|0.990067|448916|34397|4504|0.925082|2339|

S2相对S1恢复参考TP19、增加FP21、无TP损失。S3相对S2恢复TP19、损失TP13、增加FP13、减少FP6；净数不能隐藏被删除的正确像素。结石全部阶段保持Dice/Precision/Recall=0.740216/0.899778/0.628722。

## 132张图像参考

| 方案 | 光纤Dice | Precision | Recall | TP | FP | FN | BoundaryF | 低Lab FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|旧weak-tip / S1 / S2|0.755957|0.628515|0.948225|105583|62405|5765|0.487950|507|
|S3|0.755987|0.628556|0.948225|105583|62394|5765|0.488151|507|

结石全部保持Dice/Precision/Recall=0.906751/0.880212/0.934940。参考FP/Precision/Dice均仅诊断，不以可能漏标简单否决新增区域。70静态对照、逐帧指标和现有来源分组全部另存各阶段的 `video_native/summary.json`、`video_causal/summary.json`、`image_native/summary.json`，未挑选成功帧。

70/132已经参与开发，不能称独立泛化测试。其他视频帧无像素真值，只有回放与图像/几何代理诊断；没有全2694帧分割准确率。无独立tip标注，没有物理尖端误差、端点准确率或尖端Recall结论。

## 实际几何与帧间稳定性

S1把video_2处理开关44→10次、面积变化p95 1019→878px；不能解释为真实相机运动估计。S2/S3继续保持10次，S3面积步长p95 877px。S2在共同确认帧对上端点坐标与S1完全相同；S3在共同确认对上位移p95 video_1 2.00→1.27px，video_2 4.01→4.14px，未证明全面改善。全部确认集合的video_2位移p95反而4.00→4.47px，不能只报变好的一个视频。

侧HF全片约0.3415/0.4053px，三个阶段没有有意义的整体改善；侧边均尚未实际编辑。S2实际帽转角代理改善约2.6%/1.7%，S3再改善约0.26%/0.22%，均未达到原10%目标；接受模型的栅格接缝仍约40–48°量级，不等于模型≤15°接缝已在像素中实现。共同坐标代理也不能代替新模型自身接缝或真实边界标注。**本轮没有解决两侧平滑、圆钝帽整体目标。**

原始逐帧面积、坐标与资格状态未作时间平滑：[video_1曲线](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/temporal_plots/video_1_temporal.png>)、[video_2曲线](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/temporal_plots/video_2_temporal.png>)。

## 延迟与成本

双网、FP32、batch1、相同冻结权重；每视频/阶段独立进程，预热12帧后重置。每个已实现方案全部722/1972帧，不因效果不好跳过。

| 阶段 | video_1 P50/P95 ms | video_2 P50/P95 ms | 实际吞吐 v1/v2 fps |
|---|---:|---:|---:|
|S1|244.97/329.78|170.65/319.75|3.86/4.12|
|S2|231.41/293.86|237.06/292.56|4.23/4.04|
|S3|216.55/284.03|186.71/289.76|4.41/4.32|

P50/P95是双网及候选处理的逐帧延迟；吞吐还包含解码、原生mask/阶段压缩图层存储和编码。S3部分加速来自少确认、少运行帽，不能称同等功能加速。全部仍不满足实时要求。无新网络/训练；三个阶段共8082正式配对前向，72预热，202静态缓存配对，共8356对/16712次网络分支前向；参考202缓存仅计算一次，各阶段静态后处理独立复用以保证同一像素概率。

## 所有对照视频与截图

最终8格：原RGB、原始双网、旧weak-tip、S1、S2、S3、S4未执行、几何提案分析。每格原生400×400，保留真实sourcePTS、重复初始PTS及video_2 VFR，逐帧来源与掩膜身份另存。青蓝光纤、红色结石；实际增删见各阶段四格，分析中橙帽/青侧线都是拟合提案、白线为实际像素边界，当前RGB确认不是已验证物理尖端。所有输出仅研究。

- [video_1完整722帧，跨阶段8格](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/final_comparison/video_1_all_stages.mp4>)
- [video_2完整1972帧，跨阶段8格](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/final_comparison/video_2_all_stages.mp4>)
- [video_2 1233–1291重点59帧](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/final_comparison/focus1233-1291.mp4>)
- [video_2第1209帧短片](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/final_comparison/frame1209.mp4>)

| 阶段四格：RGB/双网/上阶段/本阶段 | video_1完整 | video_2完整 | 1233–1291完整 |
|---|---|---|---|
|S1|[v1](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/s1_seed_qualified/video_1_comparison.mp4>)|[v2](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/s1_seed_qualified/video_2_comparison.mp4>)|[59帧](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/s1_seed_qualified/focus1233-1291.mp4>)|
|S2|[v1](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/s2_native_gradient/video_1_comparison.mp4>)|[v2](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/s2_native_gradient/video_2_comparison.mp4>)|[59帧](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/s2_native_gradient/focus1233-1291.mp4>)|
|S3，功能评审暂停|[v1](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/s3_coherent_tip/video_1_comparison.mp4>)|[v2](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/s3_coherent_tip/video_2_comparison.mp4>)|[59帧](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/s3_coherent_tip/focus1233-1291.mp4>)|

固定异常帧：[317](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/final_comparison/focus/video_2_frame0317.png>)、[770](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/final_comparison/focus/video_2_frame0770.png>)、[1204](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/final_comparison/focus/video_2_frame1204.png>)、[1716](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/final_comparison/focus/video_2_frame1716.png>)。

重点高清图：[1209](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/final_comparison/focus/video_2_frame1209.png>)、[1233](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/final_comparison/focus/video_2_frame1233.png>)、[1250](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/final_comparison/focus/video_2_frame1250.png>)、[1266](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/final_comparison/focus/video_2_frame1266.png>)、[1291](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/final_comparison/focus/video_2_frame1291.png>)。全部59帧和额外异常帧均保留，没有只导出成功帧。

![1250跨阶段对照，S4未执行](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_staged_20261104/final_comparison/focus/video_2_frame1250.png>)

## 复现、回滚和下一步

新模块为 `tools/fiber_weak_stage1.py`、`fiber_weak_stage2.py`、`fiber_weak_stage3.py`，阶段运行入口 `tools/run_weak_staged_control.py --stage N`。阶段N>1必须先读取上一阶段review通过，S3的False会阻止S4。每阶段源码/测试/依赖与输入权重哈希、源码快照、原生mask、逐帧分支图层和状态都保留在独立目录。130测试日志为输出根 `tests.log`；末次工程核验及输出身份为 `engineering_validation.json`、`seal.json`。

全部冻结原始classwise网络输出哈希与历史逐帧相同；每阶段视频独立初始化并清空T1。未读训练521或144保留test组像素；未用stone_new、验证视频或衍生预测训练/标定模板。颜色仍为当前帧可信主体自适应色度/背景，未学习共享验证颜色模型。没有真实设备标定或激光落点验证。

回滚只需不选择这些独立实验模块；默认从未切换，无需覆盖旧权重/结果。S1/S2可作为后续研究基点，不能称已部署。最低复杂度下一步是保留S2确认，先把连续区域前沿作为只读风险观测，分离剪枝、内部取样和前沿否决，修正亚像素末端被误否决的机制；不直接降低阈值。**待新协议确认端点功能无明显退化后，再实施受限实际左右边界/端帽编辑。** 不自动重训、部署或进入结石时序记忆。
