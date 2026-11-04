# 收敛直边＋轴上蓝色顶点＋相切圆钝端帽：纠正实现与全片对照

2026-11-04。上一轮把两条实际收敛边改成平行恒宽杆、用平均宽度/2构造半圆，不符合用户本意。本轮独立修正：**保留两条原直线的收敛方向，按当前RGB最远蓝色证据确定轴上顶点，直接生成与原两边相切的圆弧和渐窄主体的最终掩膜。** 不是只画提案，也不再要求旧union每个像素保留。允许形状假设的误检与漏检，全部记录。默认推理、训练数据、权重及历史结果均未更改。

[执行前冻结协议](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/docs/plans/tapered_tangent_cap_execution_20261104.md>)、[固定参数](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/configs/tapered_tangent_cap_control_20261104.json>)、[实际实现](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/tools/fiber_tapered_tangent_cap.py>)。

## 几何与像素输出的实际定义

1. 双网络高置信主体只确认唯一身份。冻结S2掩膜中接触该主体的实例提供两侧观测，各用Huber正交直线独立拟合；没有样条、平行投影或逐像素弱蓝生长。沿用10–80%轴向观测，以减少入口裁切和端帽对直线拟合的影响。
2. 两边的远端单位方向之和给角平分方向u，两条原拟合线交点V确定精确中心轴。T取当前RGB搜索域内、深浅蓝均可、有局部蓝色支持且与前方背景有明暗/色度差异的**最远蓝色像素轴向投影**，再投回该轴；不是理论交点。蓝色像素本身可偏离轴，投影后的T必须在轴上。无P_fiber≥0.1门槛。搜索域随真实锥形收窄，加2px边界余量，非恒宽旧条带。
3. 设两侧半夹角alpha，d=(V−T)·u，r=d*sin(alpha)/(1−sin(alpha))，C=T−r*u。C在角平分轴上，到两条原直线的距离均为r。C到原两线的垂足为两个切点Q；圆弧从Q_left经T到Q_right，弧张角180°−2alpha，**不预设半圆，r也不再等于平均宽度/2**。
4. 真正栅格化两原直线之间、切点接缝后方的渐窄主体，加接缝前方圆内区域，直接替换旧fiber。没有与旧union或RGB eligible求交，没有强制保留旧锯齿，允许内部无逐像素证据的填充和旧正确像素的删除。结石保持原值；几何与stone相交时整帧回退，避免裁断帽形。
5. 删除“必须近似平行”、宽度CV/变化范围及种子保留率门槛。保留主体/线残差/基本尺度/跨度、尖端证据、面积和stone保护。若T已经到达或越过V，在保留原两边及T的前提下，该正半径相切圆无解，明确回退，不暗移T或改斜率。

这是形状重建，不代表每个填入像素都真实可见。蓝色判断和前方2/4/6px对比参数沿用旧值，没有根据本轮结果调阈值、学习验证共享颜色或重新训练。完整原始RGB、网络raw及概率逐帧与历史哈希一致。

## 完整执行和实际应用

|视频|完整帧数|有相切帽提案|实际应用/修改帧|回退|新增像素|删除旧fiber像素|
|---|---:|---:|---:|---:|---:|---:|
|video_1|722|722|654|68|29791|156145|
|video_2|1972|1932|1895|77|193357|645905|

共2549/2694帧实际应用，145帧回退。video_1的68次回退均为stone冲突；video_2主要原因按首原因计：35帧T到达/越过交点、25帧stone冲突、10帧主体细长性不足、1帧宽度范围、1帧跨度不足、2帧无可信主体、3帧主体歧义。并发拒绝原因另保存在逐帧日志，不能相加当帧总数。所有被拒绝的可解几何提案仍保存为PROPOSAL，正式实验输出不会偷用它们。

两个视频分别使用独立进程，预热12次后重置，完成2694对正式双网络前向；加24对预热为2718对/5436次网络分支前向。202张静态参考用冻结原始概率缓存，70因果参考从全视频逐帧输出抽取。12项新增几何/像素替换测试和相关回归合计228项通过；旧S2测试有未关闭日志文件ResourceWarning，无失败。合成正例以mock断言旧弱蓝扩展一旦调用就失败，验证新路径未执行扩展。

## 参考指标，仅作诊断

70因果与70静态输出指标相同；70中69帧应用、1帧stone冲突回退。132图像仅2帧应用，121无可信主体、5向远端发散、4主体细长性不足，因此不能据其整体指标声称普适几何改善。逐帧与所有已有来源分组见各scope summary。

|参考范围/方案|Fiber Dice|Precision|Recall|Boundary F|TP/FP/FN|
|---|---:|---:|---:|---:|---|
|70视频 旧S2|0.958472|0.928843|0.990053|0.925161|448910/34390/4510|
|70视频 新实际输出|0.969649|0.951795|0.988185|0.956552|448063/22693/5357|
|132图像 旧S2|0.755957|0.628515|0.948225|0.487950|105583/62405/5765|
|132图像 新实际输出|0.754693|0.627265|0.947094|0.486122|105457/62665/5891|

70视频：删除原TP 2930、恢复TP 2083，新增参考FP 2919、删除参考FP 14616；低Lab FN 2340→1925。

132图像：删除原TP 181、恢复TP 55，新增参考FP 1451、删除参考FP 1191；低Lab FN 507→505。

结石全部像素保持不变。70视频结石Dice/Precision/Recall=0.740216/0.899778/0.628722；132图像=0.906751/0.880212/0.934940。参考FP、Precision、Dice不作为本轮主要验收门槛，也不因可能漏标而隐藏已标TP损失。70视频几何与边界指标改善，但Recall略降；132图像没有改善。两套开发参考不能视为独立泛化测试。2694未逐帧标注，不能报告全视频分割准确率；无独立尖端真值，不能宣称物理尖端或激光落点定位更准。

## 几何、RGB证据及稳定性

video_1：可解提案圆弧张角中位155.97°、半径中位13.50px，圆心到原两线的距离与半径最大误差6.6e-14px。实际应用帧中，逐帧侧边距离P95的中位数1.296→1.162px；P50的中位数0.508→0.587px，不能只报改善项。原图Sobel边界梯度逐帧中位数的中位1.820→1.846，梯度高仅是RGB边界支持代理，不是真值。

video_1：旧mask中的自适应蓝色像素保留率96.52%；删去蓝色151623、恢复蓝色22667，新增非蓝色几何填充7124。这是颜色证据计数，不是真实TP/FP。相邻连续已应用帧T位移P50/P95=0.180/1.323px，超过10px次数0。没有时间状态强行平滑。

video_2：可解提案圆弧张角中位155.92°、半径中位6.57px，圆心到原两线的距离与半径最大误差1.1e-13px。实际应用帧中，逐帧侧边距离P95的中位数1.361→1.170px；P50的中位数0.520→0.596px，不能只报改善项。原图Sobel边界梯度逐帧中位数的中位3.547→4.462，梯度高仅是RGB边界支持代理，不是真值。

video_2：旧mask中的自适应蓝色像素保留率96.16%；删去蓝色628865、恢复蓝色174069，新增非蓝色几何填充19288。这是颜色证据计数，不是真实TP/FP。相邻连续已应用帧T位移P50/P95=0.296/4.348px，超过10px次数29。没有时间状态强行平滑。

video_2仍有尖端跳变，说明“最远合格蓝色像素的投影”会受局部蓝色杂点、色散/反光及搜索终止变化影响；本轮按要求实现它，没有事后换成较保守端点或隐藏失败。解析相切改善不等于像素边界/物理尖端准确。二值栅格仍存在原生分辨率台阶；放大图使用最近邻显示真实像素，橙色解析圆弧单独标出，未把平滑绘制的曲线冒充最终像素。

## 固定重点帧

|video_2帧|应用|原两侧夹角|T(x,y)|r(px)|圆弧张角|
|---|---|---:|---|---:|---:|
|317|是|29.73°|(170.88,249.19)|4.18|150.27°|
|770|是|23.25°|(151.79,203.84)|7.81|156.75°|
|1204|是|30.18°|(170.41,244.39)|3.89|149.82°|
|1209|是|30.19°|(170.32,244.24)|3.90|149.81°|
|1233|是|30.10°|(167.99,239.70)|4.70|149.90°|
|1250|是|29.77°|(168.31,240.25)|5.12|150.23°|
|1266|是|29.52°|(168.55,240.67)|5.47|150.48°|
|1291|是|29.41°|(169.43,240.18)|4.98|150.59°|
|1716|是|24.53°|(154.67,201.34)|5.65|155.47°|

1209/1250/1291半径约3.90/5.12/4.98px，而旧平行方案约29px；新的帽形按原收敛边实际构造，避免旧恒宽膨大的端部。固定重点均实际替换掩膜，317/770/1204/1716不再因两边收敛被判“不平行”跳过。1250仍删除部分原上侧蓝色边缘、恢复前端少量像素，不能说所有蓝色都被保留。

## 延迟与实时性

|视频|双网P50/P95 ms|几何P50/P95 ms|合计P50/P95 ms|实际离线吞吐fps|
|---|---:|---:|---:|---:|
|video_1|32.29/41.06|37.08/48.20|69.57/89.70|10.97|
|video_2|33.13/41.38|38.06/47.55|71.38/87.17|10.75|

FP32、batch1、两个冻结网络；吞吐包含解码、读取旧mask、双网、几何、图层保存和两画面编码。8格报告视频另行导出。**以上不含在线重新计算旧S2输入的成本**，因为本轮边界/回退输入来自缓存；当前即使不算该额外成本也未达到30fps。未训练、未部署，不能把离线吞吐当最终设备性能。

## 全部视频、图像和复现产物

8格为无标记原图、旧S2、保留收敛方向的两原直线、中心轴、真实蓝色源点投影至T、与原边相切的圆弧、纯几何提案、实际APPLIED/FALLBACK结果。青蓝fiber、红stone；端帽应用状态逐帧标明。全片及重点区间严格逐帧同步，原始面板native RGB哈希一致；导出复制真实sourcePTS/VFR和重复初始PTS，包数据未重排。所有拒绝帧保留。

- [video_1完整722帧](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_tapered_tangent_cap_20261104/comparison/video_1_comparison.mp4>)
- [video_2完整1972帧](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_tapered_tangent_cap_20261104/comparison/video_2_comparison.mp4>)
- [video_2第1233–1291帧](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_tapered_tangent_cap_20261104/comparison/focus1233-1291_comparison.mp4>)
- [video_2第1209帧](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_tapered_tangent_cap_20261104/comparison/frame1209_comparison.mp4>)

固定高清全图：[317](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_tapered_tangent_cap_20261104/comparison/focus/video_2_frame0317.png>)、[770](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_tapered_tangent_cap_20261104/comparison/focus/video_2_frame0770.png>)、[1204](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_tapered_tangent_cap_20261104/comparison/focus/video_2_frame1204.png>)、[1716](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_tapered_tangent_cap_20261104/comparison/focus/video_2_frame1716.png>)、[1209](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_tapered_tangent_cap_20261104/comparison/focus/video_2_frame1209.png>)、[1233](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_tapered_tangent_cap_20261104/comparison/focus/video_2_frame1233.png>)、[1250](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_tapered_tangent_cap_20261104/comparison/focus/video_2_frame1250.png>)、[1266](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_tapered_tangent_cap_20261104/comparison/focus/video_2_frame1266.png>)、[1291](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_tapered_tangent_cap_20261104/comparison/focus/video_2_frame1291.png>)。

尖端4倍原生像素放大图：[317](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_tapered_tangent_cap_20261104/comparison/detail/video_2_frame0317_tip_detail.png>)、[770](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_tapered_tangent_cap_20261104/comparison/detail/video_2_frame0770_tip_detail.png>)、[1204](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_tapered_tangent_cap_20261104/comparison/detail/video_2_frame1204_tip_detail.png>)、[1716](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_tapered_tangent_cap_20261104/comparison/detail/video_2_frame1716_tip_detail.png>)、[1209](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_tapered_tangent_cap_20261104/comparison/detail/video_2_frame1209_tip_detail.png>)、[1233](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_tapered_tangent_cap_20261104/comparison/detail/video_2_frame1233_tip_detail.png>)、[1250](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_tapered_tangent_cap_20261104/comparison/detail/video_2_frame1250_tip_detail.png>)、[1266](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_tapered_tangent_cap_20261104/comparison/detail/video_2_frame1266_tip_detail.png>)、[1291](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_tapered_tangent_cap_20261104/comparison/detail/video_2_frame1291_tip_detail.png>)。第一失败案例按每种首原因自动选取，见detail_figures.json，不只展示成功帧。

![1250：原图、旧S2、旧平行提案、新实际掩膜、相切圆弧和编辑像素](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_tapered_tangent_cap_20261104/comparison/detail/video_2_frame1250_tip_detail.png>)

独立masks、proposals、逐帧layers、frames.jsonl保存实际最终掩膜、二值几何提案、subject、blue_identity、search_domain、tip_eligible、rod、cap、edit_add/remove及各哈希；逐帧轴、源像素、T/C/r/切点、所有拒绝原因在日志和geometry_coordinates。summary保存逐帧与来源分组指标，reference_changes保存TP/FP/FN变化。shape_RGB_diagnostics保存全片逐帧几何/RGB/蓝色保留与稳定性诊断。

复现：在已配置uretersam环境运行 `python -m tools.run_tapered_tangent_cap_control`；运行目录存在时拒绝覆盖。源码/配置/权重/输入身份在preflight和source_snapshot.zip。报告/媒体审计/工程验证和seal另存，不更改历史或默认。未读训练521及144个保留test关联组像素；未使用stone_new或视频派生训练/模板标定。

本轮已完成用户纠正后的实际轮廓重建，值得作为新的几何研究对照保留。它解决了旧平行杆/半圆定义错误，在当前视频开发参考上边界质量改善，但仍有蓝色像素损失、图像域不改善和视频尖端跳变；因此维持实验身份，不自动替换正式分割，也不声称尖端定位准确率提高。
