# 几何优先：直杆＋半圆端帽实现及完整对照

2026-11-04。按本轮新指令停止新增逐像素弱蓝补头，独立实现直线＋中心轴＋轴上RGB尖端＋半圆掩膜。未改旧弱蓝代码、旧评审/结果、默认模型或训练数据。[执行前冻结协议](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/docs/plans/straight_semicircle_execution_20261104.md>)、[固定参数](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/configs/straight_semicircle_control_20261104.json>)、[实现](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/tools/fiber_straight_semicircle.py>)。

**实现和全部实验已完成，但本轮真实数据最终全部回退旧S2。2689个可计算几何提案全部保留，没有把被拒绝的圆滑形状冒充已采用结果。恒宽直杆＋严格半圆在当前画面中不宜直接采用。**

## 1. 实际实现

1. 当前双网络raw及P_fiber确认唯一可信高置信主体；采用S1逐块资格后判唯一规则。旧S2只作边界观测、对照和回退，从缓存读取；不执行S2扩展、弱蓝图生长或S3前沿否决。
2. 在旧掩膜中接触该主体的8连通实例上，提取10–80%轴向且距画面边缘≥3px的左右边界。分别进行Huber IRLS正交**直线**拟合，固定15轮重加权，没有样条。
3. 两条朝远端的单位切向之和归一化为最终方向u；在轴法向的24个等距截面上求两拟合线间距的**算术平均**w。宽度是原生像素坐标下几何距离，不是毫米标定。
4. 在该轴向有界RGB搜索带内，判断当前归一化色度蓝色、相对背景差异和3×3局部支持，比较前方2/4/6px颜色/明暗；不要求网络P≥0.1，也不要求所有前段像素相连或逐像素通过。取全部合格真实像素的最远轴向位置t，投影T=O+t*u。T严格在轴线上，不取理论直线交点。
5. r=w/2，C=T-r*u。掩膜为法向距轴≤r、轴向≤C的直条，加轴向≥C、到C距离≤r的半圆；按画面截断。**不与旧union或RGB eligible求交**，不做高斯模糊或逐像素补头。接受时清除所有旧fiber、直接用该几何掩膜替换，允许旧像素删除与内部无逐像素证据的填补。
6. 两侧夹角>15°、宽度robust CV>0.25或P90−P10/平均>0.6、边界线残差/跨度/细长性不合、主体或端部证据不足、面积/种子严重不符时回退。与stone有任何冲突也整帧回退，避免裁断圆帽或覆盖结石。所有其他拒绝原因同时记录，不只报告第一条。

这个输出是**形状假设重建**，不代表填入的每个像素都确实可见。没有新网络、训练、共享验证颜色模板、T1像素填补或Godot相机标定。

## 2. 半圆相切的必要数学区别

严格半圆的两个接缝切线平行，因此不能同时与两条不平行的原独立拟合线精确相切。本实现先用独立拟合线判断近似平行，通过后将最终直边规范化为O+t*u±(w/2)*n两条平行线，半圆与**这两条最终规范直边**严格相切。没有声称它同时与原始非平行拟合线相切。

图中黄/紫为原独立直线，白为最终平行规范直边。解析接缝角为0，栅格仍有分辨率造成的台阶，不能据此宣称真实边界或尖端准确。

## 3. 测试和完整执行

12项新测试覆盖鲁棒直线、精确C/T/r关系、轴上顶点、相切、真实直接重建、允许删除旧不规则像素、填补内部黑色缺口、低P暗蓝远端、RGB源点投影、非平行识别、完整回退、结石保护和输入/重置；共12项方法组织测试（部分项包含多个断言）。联合相邻历史测试共142项通过。合成正例还以mock断言旧weak/S2 expand一旦调用即报错，确认此路径不运行弱蓝补头。

两个独立双网络进程分别完成video_1全部722帧和video_2全部1972帧，每进程预热12帧后重置。2694个正式配对前向＋24个预热，共2718对/5436次网络分支前向。原始RGB、raw及两类概率与旧实验逐帧哈希一致。202静态参考使用已冻结的同一原始概率缓存，70因果参考从完整视频输出抽取；没有读训练521或144个保留test组像素。没有训练更新或stone_new使用。

## 4. 全片结果与真实失败原因

| 范围 | 帧数 | 可保存完整几何提案 | 实际应用 | 回退 |
|---|---:|---:|---:|---:|
|video_1|722|722|0|722|
|video_2|1972|1967|0|1972|
|合计|2694|2689|0|2694|

video_1原独立拟合边夹角22.27–27.28°，中位24.03°；video_2可拟合帧18.82–50.51°，中位24.08°。**全部超过本轮冻结的15°容差**。15°是本轮实验设定，不是设备标定或普适物理标准；没有根据结果提高门槛。

video_1全部722帧也触发宽度不稳定，306个几何提案触碰旧stone。video_2有1967帧不平行、1956帧宽度不稳定、220个提案触碰stone、30帧最远RGB证据到达搜索上限、1帧拟合跨度不足；另2帧无可信主体、3帧主体歧义。这些原因可同时发生，不能相加当总帧数。

重点1209/1250/1291：

| 帧 | 拟合夹角 | 平均w(px) | r(px) | 轴上T(x,y) | 实际应用 |
|---|---:|---:|---:|---|---|
|1209|30.19°|57.97|28.98|(170.32,244.24)|否：不平行、宽度不稳|
|1250|29.77°|57.48|28.74|(168.31,240.25)|否：不平行、宽度不稳|
|1291|29.41°|57.08|28.54|(169.43,240.18)|否：不平行、宽度不稳|

这些提案是完整直杆＋圆头，但平均宽度把逐渐变细的光纤变成了恒宽杆：入口侧容易截掉原主体，前端容易鼓胀。可见蓝色轮廓并不因圆滑就更准确。317/770/1204/1716亦完成独立直线、轴和帽提案，分别约29.73/23.25/30.18/24.53°，均按同一规则回退。

## 5. 最终掩膜与未采用提案必须分开评价

最终实际输出2694帧、70因果/静态和132图像均与旧S2逐像素相同，新增/删除均0，结石覆盖0。因此**本轮最终分割没有改善**；不能把回退保住旧指标当成新几何有效。

最终70参考光纤Dice/Precision/Recall=0.958472/0.928843/0.990053，TP/FP/FN=448910/34390/4510，BoundaryF=0.925161，低Lab FN2340；132图像为0.755957/0.628515/0.948225，TP/FP/FN=105583/62405/5765，BoundaryF=0.487950，低Lab FN507。结石70参考D/P/R=0.740216/0.899778/0.628722，132为0.906751/0.880212/0.934940，均不变。逐帧及已有分组保存在各scope summary。

为直接看清重建形状的代价，还对**未采用的纯几何提案**作反事实诊断，忽略全部回退条件，只在能构造提案的同一子集与旧mask比较：

| 提案可用子集 | 方案 | Fiber Dice | Precision | Recall | TP/FP/FN | BoundaryF |
|---|---|---:|---:|---:|---|---:|
|70视频帧|旧S2|0.958472|0.928843|0.990053|448910/34390/4510|0.925161|
|同70帧|纯几何，未采用|0.853406|0.820701|0.888825|403011/88046/50409|0.300343|
|132中的9张|旧S2同子集|0.812407|0.687801|0.992150|27175/12335/215|0.453790|
|同9张|纯几何，未采用|0.623043|0.497838|0.832384|22799/22997/4591|0.359994|

70帧提案删除参考TP49631、恢复3732，增加参考FP76004、减少22348；低Lab FN2340→5275。9张图像对应删除TP4417、恢复41，增加FP15901、减少5239，低Lab FN3→16。这是**拒绝提案的诊断**，不是正式采用结果或132张整体纯几何指标。未采用提案也会覆盖stone，相关指标仅反事实，正式输出没有覆盖。参考FP/Precision/Dice只诊断，可能漏标；但显著的已标TP损失同样不能隐去。

70/132已参与开发，不是独立最终泛化测试。未标注视频帧不报分割准确率；没有独立尖端真值，不宣称物理尖端位置、落点或端点准确率提高。

## 6. 性能与测量边界

| 范围 | 双网P50/P95 ms | 几何P50/P95 ms | 合计P50/P95 ms | 实际离线吞吐fps |
|---|---:|---:|---:|---:|
|video_1|30.32/37.15|31.39/38.86|61.56/74.18|12.09|
|video_2|30.87/38.38|32.45/40.09|63.44/76.26|11.58|

双网FP32、batch1，同一冻结模型。实际吞吐包括视频解码、旧mask读取、双网、几何、mask/图层保存及两画面编码；8格报告视频另行导出。即使最终回退也计算可用提案，方便完整审查。合计延迟**不含旧S2算法重新计算**，因为本轮边界/回退输入来自冻结缓存；因此不能把此数照搬为在线同时运行S2的新实时吞吐。新模块可接受调用者提供的当前旧mask，但本轮仅验证S2来源，不自动宣称以raw直接替代输入也有相同表现。即使暂不考虑旧输入生成成本，当前也不满足30fps。

## 7. 对比图、完整视频及独立输出

8格依次为：无标记原图、旧S2、独立拟合直线＋平行规范边、中心轴、蓝色源点投影至T、C/r/半圆、**纯几何提案**、**实际新结果APPLIED或FALLBACK**。每格原生400×400，所有面板同一源帧，按真实sourcePTS、VFR和重复初始PTS同步。未采用曲线和掩膜始终标PROPOSAL。

- [video_1完整722帧](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_straight_semicircle_20261104/comparison/video_1_comparison.mp4>)
- [video_2完整1972帧](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_straight_semicircle_20261104/comparison/video_2_comparison.mp4>)
- [video_2第1233–1291帧](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_straight_semicircle_20261104/comparison/focus1233-1291_comparison.mp4>)
- [video_2第1209帧短片](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_straight_semicircle_20261104/comparison/frame1209_comparison.mp4>)

固定图：[317](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_straight_semicircle_20261104/comparison/focus/video_2_frame0317.png>)、[770](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_straight_semicircle_20261104/comparison/focus/video_2_frame0770.png>)、[1204](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_straight_semicircle_20261104/comparison/focus/video_2_frame1204.png>)、[1716](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_straight_semicircle_20261104/comparison/focus/video_2_frame1716.png>)。

重点：[1209](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_straight_semicircle_20261104/comparison/focus/video_2_frame1209.png>)、[1233](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_straight_semicircle_20261104/comparison/focus/video_2_frame1233.png>)、[1250](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_straight_semicircle_20261104/comparison/focus/video_2_frame1250.png>)、[1266](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_straight_semicircle_20261104/comparison/focus/video_2_frame1266.png>)、[1291](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_straight_semicircle_20261104/comparison/focus/video_2_frame1291.png>)；所有59帧重点区间均保存，没有只挑成功帧。

![1250：直杆半圆提案与实际回退](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_straight_semicircle_20261104/comparison/focus/video_2_frame1250.png>)

下面是**合成测试**，不是手术样本：近似平行直杆满足假设时，确实替换旧mask、去除游离块并填入内部缺口，验证实现并非永远不应用。

![合成测试，实际应用几何重建](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_straight_semicircle_20261104/comparison/synthetic_applied.png>)

独立文件在 `runs/a1_straight_semicircle_20261104/`：

- `full_video/video_X/masks/`：实际最终类掩膜；`proposals/`：完整二值几何提案，拒绝也保留；`layers/`：种子关联主体、RGB蓝、搜索/尖端证据、直杆/帽、实际增删压缩图层。
- `full_video/video_X/frames.jsonl`：每帧原/新身份、两条直线、中心轴、w、蓝色源点、T/C/r、129点半圆曲线、是否应用、全部拒绝原因和处理耗时。
- `video_X_geometry_coordinates.json`：便于后续分析的逐帧几何坐标/状态及对应mask路径。
- 三个参考scope的 `summary.json`、`reference_changes.json`：完整/逐帧/来源分组指标；`proposal_reference_metrics.json`、`proposal_delta_diagnostics.json`：未采用提案同子集诊断。
- `preflight.json`、`source_snapshot.zip`、`media_identity.json`、`engineering_validation.json`、`seal.json`、`tests.log`：冻结源码、输入/权重/输出身份及工程核验。

## 8. 调用与后续判断

核心接口：`StraightSemicircle(config).run(frame_bgr, raw_mask, fiber_probability, old_mask)`。输入原生尺寸一致，mask类别0/1/2，probability为3×H×W；输出类掩膜，完整提案和几何元数据在layers/info。研究批量入口 [run_straight_semicircle_control.py](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/tools/run_straight_semicircle_control.py>)，已有输出目录拒绝覆盖。

本轮实现了用户指定的几何构造和回退，没有继续弱蓝修补；没有证明实际分割或端点准确率改善，也没有把全部回退包装成优化成功。当前观测更符合两条收敛直边的锥形主体。若下一轮希望保留真实收敛边并直接重建，可研究**渐窄直边＋相切圆弧**：该圆弧通常不是严格半圆，半径应与真实末端局部宽度/两侧切线一致，而不是整段平均宽度的一半。这需要另定协议，不在本轮偷偷替换用户要求，也不自动放宽15°或其他门槛。
