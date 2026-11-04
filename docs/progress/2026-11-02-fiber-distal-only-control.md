# 光纤可靠远端单向搜索：单因素完整对照

2026-11-02。按v2.28计划冻结协议后实现，三臂全部完成两片视频。**单向搜索恢复了一部分原被另一端回退清空的有效候选，但新候选仍未通过质量/成本门槛，不采用。** 当前默认A0、实际语义输出、模型权重、旧源码/结果和标签不变，无训练或结石记忆。

## 1. 唯一变化与复现

三臂为raw2原两网、blue_extent冻结上一轮双端、新distal_only。仅把搜索范围从两端改为既有规则已定向的远端：种子端到图边缘距离差≥0.02D的原入口提示，或原兼容前帧T1解符号；无方向则不生长。这里的方向资格仍是未由物理真值验证的启发式，不能保证远端身份正确。

种子资格、归一化RGB蓝色族、背景对比、背景至少6点、颜色/宽度/曲率/单侧弱边/T1门槛、端帽条件、面积预算、原结石屏障与4连通全部继承。非搜索端不产生多边形、异常或端点；搜索端异常仍按原规则整帧拒绝新增，不保留异常前的片段规避拒绝。没有同时降低颜色、几何或质量阈值，也未修改第1209帧已知的5/6背景不足问题。

底座仍是old190结石+A0 model1光纤，2网49,202,790参数、FP32/batch1、RGB PIL256/ImageNet/native概率和原语义竞争。202静态真实两网推理共享三臂；6独立进程各完成video_1的722帧、video_2的1972帧，共8082正式帧，另72预热，合计8356配对/16712网络分支调用。70因果参考从全视频抽取，不增加网络推理。

核验原网络概率/原掩膜及旧双端候选逐帧精确复现；新旧种子记录、T1原始prior/age/更新状态相同，新方案实际搜索的分支与旧方案对应分支精确相同。差异只来自另一端未搜索及其派生的少计算、少候选/冲突和不再受其异常清空。不存在将缓存当成正式前向的替代。

## 2. 主质量：70因果与132静态

以下候选方案指标均为独立评估的**虚拟合并**（只把候选原背景改为fiber），没有写回实际掩膜。真实输出所有类别及概率变化0；原正确结石/光纤删除0。70主口径仍为全视频因果，70独立静态另列机制诊断；132独立无状态，不借用T1。

|范围|输出|光纤Dice|Precision|Recall|边界F(3px)|TP|FP|FN|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|video_causal|raw2|0.945537|0.970967|0.921404|0.862027|417783|12492|35637|
|video_causal|blue_extent|0.945537|0.970967|0.921404|0.862027|417783|12492|35637|
|video_causal|distal_only|0.950679|0.970644|0.931518|0.881988|422369|12774|31051|
|image_native|raw2|0.765758|0.643433|0.945513|0.524869|105281|58343|6067|
|image_native|blue_extent|0.765572|0.643149|0.945558|0.524013|105286|58418|6062|
|image_native|distal_only|0.765758|0.643433|0.945513|0.524869|105281|58343|6067|

|范围|方案|新增TP / FP|新增Precision|原P<0.1 FN覆盖|低Lab FN覆盖|有新增帧|
|---|---|---:|---:|---:|---:|---:|
|video_causal|blue_extent|0 / 0|不可计算|0|0|0|
|video_causal|distal_only|4586 / 282|0.942071|3556|1961|19|
|image_native|blue_extent|5 / 75|0.062500|0|0|1|
|image_native|distal_only|0 / 0|不可计算|0|0|0|
|video_native|blue_extent|0 / 0|不可计算|0|0|0|
|video_native|distal_only|4586 / 282|0.942071|3556|1961|19|

|主范围|方案|原低Lab FN|虚拟低Lab FN|下降比例|
|---|---|---:|---:|---:|
|video_causal|blue_extent|17465|17465|0.0000%|
|video_causal|distal_only|17465|15504|11.2282%|
|image_native|blue_extent|507|507|0.0000%|
|image_native|distal_only|507|507|0.0000%|

低亮分组来自原RGB的OpenCV Lab L[0,85)，不是推理中的固定蓝色亮度门控。无新增分母不称100%Precision或通过。132没有新增不意味着已解决其光纤漏分；只是保持原输出。所有FP按冻结参考计，不因蓝色结构可能缺标而排除样本或改真值。70/132是开发材料，不是独立最终泛化测试。

两类别完整原图指标保存在各summary；结石始终不变：70因果Dice/Precision/Recall=0.740216/0.899778/0.628722、TP152516/FP16988/FN90065、边界F0.475440；132图像=0.906751/0.880212/0.934940、TP6120805/FP832978/FN425928、边界F0.393852。冲突机会和参考真实光纤被石头屏障挡住的数量保存在candidate计数，未覆盖原石头。

## 3. 单因素配对归因

|范围|候选发生变化帧|相对旧虚拟结果新增TP|损失TP|新增FP|移除FP|因取消另一端异常而接纳的帧|
|---|---:|---:|---:|---:|---:|---:|
|video_native|19|4586|0|282|0|19|
|video_causal|19|4586|0|282|0|19|
|image_native|1|0|5|0|75|0|

这里“损失TP”是相对上一轮候选虚拟合并的变化，不能和原网络正确像素删除混淆；后者始终为0。该对照支持“另一端异常确实曾清空部分可用当前证据”，但接纳像素仍有FP，不等于远端方向、蓝色身份和圆钝端帽全部正确。

方向歧义时只保留原输出，损失的候选机会显式计数；不强制选一个方向。新旧种子记录完全相同，所以固定video_2第1209帧依然为insufficient_seed_background，原采样10点仅5个在图内/要求6个，不进入生长，原FN4002/低Lab FN2626保持。这是预先保留的独立瓶颈，本轮没有试图把它调通；不能把该帧失败误归因于“暗蓝没有RGB证据”。

## 4. 全视频实际搜索与成本

|视频|实际远端walk次数|零搜索帧|方向歧义帧|种子记录与旧方案相同帧|候选掩膜改变帧|
|---|---:|---:|---:|---:|---:|
|video_1|722|0|0|722|50|
|video_2|1793|179|0|1972|992|

|方案|视频/帧|总P50 ms|总P95 ms|额外P95 ms|含审计导出fps|有新增候选帧|
|---|---|---:|---:|---:|---:|---:|
|raw2|video_1/722|15.09|19.61|0.00|41.59|0|
|raw2|video_2/1972|14.90|18.69|0.00|42.59|0|
|blue_extent|video_1/722|30.60|42.23|16.17|25.03|2|
|blue_extent|video_2/1972|30.89|41.96|17.92|25.27|31|
|distal_only|video_1/722|29.84|40.64|15.61|25.60|50|
|distal_only|video_2/1972|30.65|41.16|17.57|25.28|989|

RTX4070Ti，torch2.11+cu128。每臂独立进程实际推理，直接总时间含网络、原图映射、种子/外边界/连通/端帽/T1及必要元数据；吞吐再包含解码/详细审计/PNG/视频导出。报告每阶段p50/p95/max、CPU秒、Windows工作集和CUDA峰值，不将分位数相加，也不把不同轮次负载下的时间相减当加速结论。

不能因为取消一个方向就假定总耗时减半：种子和原T1仍每帧运行。没有语义真值的全片帧只报告机制和成本，不给全片像素精度。完整回放均保留，质量失败不免测。

## 5. 末端、分组和质量判定

端帽规则未改变，只有被搜索远端可输出候选xy；非搜索端和方向歧义均无确定尖端。输出数量不是准确率，没有独立尖端标注，尖端Recall/Precision/定位误差、遮挡恢复和激光落点准确性仍为null。完整分支状态及待审核尖端图均保留，语义骨架3native px覆盖为事后诊断，未参与选择且不是物理轴。

|主范围分组|帧数|原光纤Dice|单向虚拟Dice|单向新增TP / FP|
|---|---:|---:|---:|---:|
|video_1|43|0.976570|0.976563|1 / 5|
|video_2|27|0.906306|0.918361|4585 / 277|
|original29|29|0.969235|0.972782|1409 / 206|
|new_user_approved|41|0.929877|0.936095|3177 / 76|
|kidney_stones_fiber_v1|68|0.798775|0.798775|0 / 0|
|kidney_stones_fiber_v2|64|0.748768|0.748768|0 / 0|
|old190_exposed25|25|0.643663|0.643663|0 / 0|
|not_matched107|107|0.797731|0.797731|0 / 0|
|latest5|5|0.844614|0.860013|1001 / 28|

主门槛拒绝项（原样保留）：

- video_causal_new_precision
- video_causal:raw2:fiber_precision
- video_causal:classwise3:stone_dice
- video_causal:classwise3:stone_precision
- video_causal:classwise3:fiber_precision
- video_causal:classwise3:stone_boundary_f
- fixed1209_dark_FN_10percent
- image_native_new_precision
- image_native_dark_FN_10percent
- image_native:classwise3:stone_precision
- image_native:classwise3:fiber_recall
- image_native:classwise3:stone_boundary_f

部分classwise比较不足已存在于研究底座，不能全部归因于此次方向改变。新增Precision98%、三个范围暗FN下降10%、原29/raw2/classwise非退化、额外5ms与源帧预算均未放宽。已知1209问题意味着本单因素本就不能独立完成全项验收；其价值是可证伪地隔离一项失败原因，不能绕开门槛采用。

55项测试通过（10新方向机制+45回归），独立审计覆盖源码/协议/权重/输入身份、源码包、旧双端精确复现、全部种子与T1一致、实际选择分支一致、每帧≤1次walk/歧义不生像素、当前色度/连通/原类别保护、独立混淆与暗FN、全8082帧PNG/日志/直接计时和状态清空。工程核验通过不代表分割或物理尖端验收通过。

离线核验首轮误指上一轮协议身份，导致新源码包中找不到旧协议文档；已更正离线核验路径并全量重核72项通过，没有改源码包、算法、参数或任何正式推理结果。

原29/新增41/最近5、43/27病例、68/64来源、25/107曝光及全部争议样本保留；132来源标签仍待审核，未把模型预测变成真值。未读521训练或144保留test像素，无stone_new训练、视频学习/模板标定/伪标签、Godot真实标定、默认替换或结石记忆。272板为70静态+132+同70因果，未增加训练或推理计数。

## 6. 停止及下一步

按原门槛拒绝此固定候选，不放宽阈值或修改实际掩膜。下一步只设计**可见侧背景采样**的独立对照：保持本轮单向、颜色/形态/T1/端帽及质量门槛，先登记同帧可见侧采样布局和覆盖条件，避免简单把6点改成5点或跨视频学习背景。该因素与本轮方向改变分开，不将两个改动混称单因素；没有新协议就不启动另一候选实验。

- [冻结单因素协议](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/docs/plans/fiber_distal_only_execution_20261102.md>)
- [冻结配置](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/configs/fiber_distal_only_control_20261102.json>)
- [结果及全部门槛](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_distal_only_control_20261102/summary.json>)
- [独立审计](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_distal_only_control_20261102/engineering_checks.json>)
- [单因素配对归因](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_distal_only_control_20261102/analysis/direction_single_factor_pairs.json>)
- [70因果逐帧/分组](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_distal_only_control_20261102/video_causal/summary.json>)
- [70独立静态诊断](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_distal_only_control_20261102/video_native/summary.json>)
- [132静态逐帧/分组](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_distal_only_control_20261102/image_native/summary.json>)
- [逐帧两类别指标](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_distal_only_control_20261102/analysis/per_frame_actual_virtual.csv>)
- [亮度分层](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_distal_only_control_20261102/analysis/brightness_strata.json>)
- [骨架诊断](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_distal_only_control_20261102/analysis/centerline_diagnostic.json>)
- [video_1全722帧对比](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_distal_only_control_20261102/video_1_comparison.mp4>)
- [video_2全1972帧对比](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_distal_only_control_20261102/video_2_comparison.mp4>)

第1136帧包含689个新增参考TP及83个FP：能看到暗蓝漏分被候选覆盖，也能看到边界外扩，不能只展示黄色正确像素。

![1136原图/参考/概率/新增候选及TPFP](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_distal_only_control_20261102/analysis/all_reference_boards/video_causal_0061.jpg>)

![固定1209未改背景失败](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_distal_only_control_20261102/analysis/video_causal_video2_frame1209.jpg>)
