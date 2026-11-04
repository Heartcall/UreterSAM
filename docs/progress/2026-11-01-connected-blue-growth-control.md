# 可信杆体自适应蓝色连通生长：只读候选完整对照

2026-11-01。按用户“按照plan继续执行下一步”完成冻结参数登记、独立实现、202静态参考对照及三臂两片完整回放。**两候选均未通过，停止这一固定配置，不进入像素修改协议、训练或部署。**

## 1. 比较对象和数据边界

三臂为raw2原输入、connected_rgb当前RGB连通生长、connected_rgb_t1相同生长加冻结T1因果提示。底座是old190结石网络+A0 model1光纤网络，两网49,202,790参数，FP32/batch1，RGB原输入PIL256/ImageNet归一化，恢复原图连续概率后沿用原语义竞争。它是研究对照底座，项目默认A0未替换。

当前实现从P_fiber≥0.9且原类别为光纤的4连通核心建立本帧色度中位数/MAD及侧背景对照；只有唯一通过资格的种子才搜索。沿两端当前双侧RGB边缘逐段更新中心与切向，允许原P_fiber<0.1的像素成为候选。色度锚点不随生长漂移，没有固定HSV/V蓝色亮度门控、共享验证视频颜色模板、形态学闭合或历史掩膜填充。走廊只限定RGB计算范围；原结石在连通运算前形成屏障，最终新增候选只能位于原背景。缺边缘/颜色则停止该端，宽度/转角/预算异常拒绝整帧新增。

T1真实逐帧执行原evidence_state，仅给方向和异常截短提示，不更改候选或可见掩膜；两种候选的像素及去除时间/T1提示后的几何元数据逐帧完全相同。静态无视频状态，每段与预热之后独立清空。本轮没有检验“用T1修改候选接受决策”的额外算法。

70视频参考=原29+新增41（video_1 43、video_2 27）；132图像来源v1/v2为68/64，历史old190曝光25/未匹配107。原29及全部来源组都保留。没有读取521训练或144保留test像素，没有stone_new训练、视频训练/伪标签或Godot真实标定。两片全部2694帧只具70帧语义参考，不能宣称全片像素精度；70/132均是已参与开发的材料，不是独立最终测试。

三臂静态共享202次实际原始两网推理；6个独立视频进程各完成722/1972帧，共8082正式帧，另72预热。合计8356次两网配对推理、16712次实际网络分支调用，未以缓存替代正式前向。所有原分割/概率逐帧精确复现，**实际像素变化0**。

## 2. 实际分割与候选虚拟合并

虚拟合并仅在独立评估中将新增候选的原背景变为光纤；没有写回实际分割。下面B1/B2结果相同，不能写成两份独立改善证据。结石两类输出始终保留，原正确结石/光纤被删除数均为0。

|参考集/语义类别|输出|Dice|Precision|Recall|边界F(3px)|TP|FP|FN|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|70视频/stone|实际原输出|0.740216|0.899778|0.628722|0.475440|152516|16988|90065|
|70视频/stone|候选虚拟合并|0.740216|0.899778|0.628722|0.475440|152516|16988|90065|
|70视频/fiber|实际原输出|0.945537|0.970967|0.921404|0.862027|417783|12492|35637|
|70视频/fiber|候选虚拟合并|0.949197|0.971012|0.928340|0.877443|420928|12566|32492|
|132图像/stone|实际原输出|0.906751|0.880212|0.934940|0.393852|6120805|832978|425928|
|132图像/stone|候选虚拟合并|0.906751|0.880212|0.934940|0.393852|6120805|832978|425928|
|132图像/fiber|实际原输出|0.765758|0.643433|0.945513|0.524869|105281|58343|6067|
|132图像/fiber|候选虚拟合并|0.764413|0.640818|0.947076|0.521866|105455|59108|5893|

|新增候选指标（每个候选臂相同）|70视频|132图像|
|---|---:|---:|
|候选新像素|3219|939|
|参考TP / FP|3145 / 74|174 / 765|
|候选参考Precision|97.7011%|18.5304%|
|原P<0.1的FN候选覆盖|2094|91|
|原低Lab FN → 虚拟FN|17465 → 16138|507 → 497|
|低Lab FN虚拟减少|7.5981%|1.9724%|
|有新增候选的参考帧|19/70|15/132|
|参考无光纤却产生新增候选的帧|0|4|
|原结石屏障冲突像素|0|4|
|冲突中参考真实光纤像素|0|0|
|几何异常整帧拒绝|0|2|

低Lab指原RGB转换到OpenCV Lab的L通道[0,85)，不用于推理色度硬门控。候选覆盖不是已恢复像素。视频候选Precision低于冻结98%，图像误检更明显；三个暗段FN下降门槛也未全部满足。原门槛下每候选15项拒绝完整保留。其中对classwise的结石质量不足已存在于raw2，不能归因于生长新增：本轮原/虚拟结石完全相同。

图像显示序号13/75/88/129的现有参考无光纤，仍产生新增；按原参考均计FP。预先登记的45/63/67/85/101/129六张争议图也全部保留，无按结果排除。真实蓝色结构是否属于光纤、参考是否漏标尚需最小局部确认，不能自动改真值，也不能把参考FP率直接称为物理身份误认率。相对蓝色和双边缘未足以保证可靠身份。

## 3. 原29、病例和来源分组

|组|帧数|原光纤Dice|虚拟光纤Dice|新增参考TP|新增参考FP|
|---|---:|---:|---:|---:|---:|
|video_1|43|0.976570|0.976570|0|0|
|video_2|27|0.906306|0.914876|3145|74|
|original29|29|0.969235|0.971755|921|63|
|new_user_approved|41|0.929877|0.934312|2224|11|
|kidney_stones_fiber_v1|68|0.798775|0.798353|3|54|
|kidney_stones_fiber_v2|64|0.748768|0.747022|171|711|
|old190_exposed25|25|0.643663|0.643089|0|51|
|not_matched107|107|0.797731|0.796084|174|714|
|latest5|5|0.844614|0.862173|1124|6|

两类别的分组Dice/Precision/Recall、边界计数及逐帧混淆矩阵均保存在各scope/summary.json；逐帧表保留三臂606条候选记录，以及1212条两语义类别实际/虚拟混淆记录。旧CSV中actual_fiber_Dice/virtual_fiber_Dice两列是集合聚合指标；真正逐帧值位于fiber_virtual_*列。新增per_frame_actual_virtual_confusions.csv明确提供逐帧TP/FP/FN，避免将集合Dice误作逐帧精度。

补充计算语义参考掩膜的形态学骨架3原图像素容差覆盖，属于**结果后登记的诊断，未参与门槛或选择**；不是物理中心轴、无遮挡完整性或尖端真值。

|参考集|原骨架Recall / Precision|虚拟骨架Recall / Precision|
|---|---:|---:|
|video_native|0.760009 / 0.823350|0.777013 / 0.816147|
|image_native|0.754837 / 0.590521|0.758377 / 0.577744|

## 4. 暗段和独立末端诊断

固定video_2第1209帧仍没有候选，原光纤FN4002、低Lab FN2626，虚拟不变。原因是种子6882像素、PCA长度119.05/宽70.30，但五站只有后两站通过双侧边缘（要求≥3/5）。五站两侧RGB对比依次为0.03818/0.18536、0.02465/0.23026、0.02324/0.21783、0.14322/0.23954、0.18421/0.25682；前三站一侧低于0.04，内部三个色度点则均支持。没有靠降低边缘门槛补救。

只读诊断中，按这一**未通过资格**的种子色度查询，4002个FN中3031个兼容色度，2626个低Lab FN中1675个兼容；它们不满足完整候选准入，绝不是恢复TP。此例支持“先检查裁切/偏轴/单侧弱边导致的种子资格失败”，而不能证明扩大色度范围有效，也不能证明当前PCA是真实轴。

末端检查在已接受正新增后运行，使用原RGB三条轴向剖面和前方边缘连续性；内部颜色变化、疑似遮挡、未知与可见末端候选分开。静态视频有28个unknown、10个internal_transition分支、没有visible_end_candidate；静态图像为unknown4、internal_transition17、visible_end_candidate8、occlusion_suspected1。两端是候选假设，计数可一帧多项，不能当作已确定的唯一远端。

全video_1没有新增候选，末端检查不输出。全video_2共1935个unknown、684个internal_transition、140个visible_end_candidate、1个occlusion_suspected分支；这些仍是算法状态，不是人工验证的端点类别。没有独立尖端真值，尖端Recall/Precision、像素TP/FP/FN、定位误差、遮挡恢复及激光落点准确性均不可计算。不能以140个候选或视觉变长宣称改善端点准确率。

T1在video_1给722帧正端建议；video_2给1704帧正端、266帧未知、2帧当前/历史方向冲突。共49帧含截短或方向冲突提示。B1/B2全2694帧候选、提案、冲突与末端状态精确相同；本轮T1增加解释信息与成本，没有像素收益。静态和完整视频一一匹配。

## 5. 完整视频和真实成本

RTX4070Ti，torch2.11+cu128，6个独立进程，2网、FP32。总管线直接同步计时包括神经网络、原图映射、资格、生长、端部检查和B2完整T1；吞吐另包含解码、审计、PNG/视频导出。不是将各分位数相加。

|臂|视频/帧|总P50 ms|总P95 ms|额外P95 ms|含导出fps|有新增候选帧|
|---|---|---:|---:|---:|---:|---:|
|raw2|video_1/722|19.08|27.45|0.00|33.24|0|
|raw2|video_2/1972|19.40|28.14|0.00|33.12|0|
|connected_rgb|video_1/722|30.22|38.90|12.19|24.54|0|
|connected_rgb|video_2/1972|33.45|43.77|16.56|22.71|1380|
|connected_rgb_t1|video_1/722|33.82|46.53|17.73|22.24|0|
|connected_rgb_t1|video_2/1972|37.73|52.26|22.49|20.50|1380|

两候选均超额外p95≤5ms和约33.3ms源帧间隔预算；raw2同轮基线通过p95预算但存在单帧尖峰，不等同于严格最坏时间保证。无额外神经网络或训练；各进程GPU峰值分配223.92MiB，Windows峰值工作集约1.95–2.02GiB，CPU耗时/各阶段p50/p95/max详见汇总。候选与T1都是CPU处理，完整帧native图映射/连通/剖面是现阶段实时性风险，不能仅报网络forward时间。

全video_1所有722帧都以no_continuous_new_candidate回退；全video_2有1380/1972帧接受新增候选、592/1972帧没有新增。拒绝原因：384无连续新增、161无可信种子、24方向跳变、20色度根缺失/歧义、2宽度异常、1多可信种子。26帧几何异常拒绝、过预算未发生；所有回退均保留原分割。主指标不排除失败帧。

推理代码以4连通、颜色资格、当前逐段边缘及原结石屏障限制候选，合成测试与逐帧审计未发现修改保护类别或跨原结石生长。没有逐帧遮挡/物理身份真值，不能据此声称真实视频不存在所有未知区误连接或无证据尖端。观察到的图像FP及可能边界过扩均作为未通过证据保留。

## 6. 复现、产物及停止决定

27项针对性/回归测试通过（12生长+9时序证据+3classwise+3member）；66项独立离线核验通过。核验包含旧源码/协议/权重/输入身份、源码包逐字节比对、202静态原概率及原掩膜、独立重算混淆/候选TPFP/低P和低Lab覆盖、现有像素保护、四连通根、当前色度必要条件、全部8082正式帧日志/候选PNG/概率身份、六独立PID、T1不改候选、直接计时分位数及回退清空。

工程核验通过仅表示运行与记录一致，不意味着候选质量通过。复核时一条测试命令引用了不存在的test_two_network_experts模块，保留失败tests.log；更正为实际test_member_experts后27项全部通过，记录tests_verified.log。离线审计初次字段名引用错误更正后全量重核；未改算法、阈值或正式实验结果。

固定候选在视频少量暗段上有当前RGB价值，但没有解决关键1209，图像身份与边界/Precision不足，且总成本失败。**按冻结停止条件拒绝B1/B2，不修改实际掩膜，不准备部署。** 不扫颜色、边缘或T1参数，不重训、不改默认、不进入结石记忆。

下一步最小只读工作是分开检查：①1209等裁切杆体的单侧弱边、种子轴与站位；②新增负例13/75/88/129及预登记争议样本的身份/局部标签完整性；③连通与剖面的CPU成本。只提出可审查的部分可见边缘/身份确认协议，不直接降低资格或增加像素。若需新候选或训练，另行冻结协议。旧主计划的局部暗段归因未消失，合并到这一更具体的失败归因中。

全部202静态板、固定失败帧、原RGB/参考/原概率/候选/虚拟/冲突、亮度分层、逐帧与来源结果均保留。视频绿色仅表示候选，紫色是被拒提案或未评估搜索范围，黄色是未经验证的末端候选；第二列是真实原分割。

- [冻结提案](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/docs/plans/fiber_connected_blue_growth_proposal_20261101.md>)
- [冻结执行协议](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/docs/plans/fiber_connected_blue_growth_execution_20261101.md>)
- [配置](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/configs/fiber_connected_blue_growth_20261101.json>)
- [结果与全部拒绝项](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_connected_blue_growth_20261101/summary.json>)
- [66项独立核验](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_connected_blue_growth_20261101/engineering_checks.json>)
- [70帧逐帧及分组](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_connected_blue_growth_20261101/video_native/summary.json>)
- [132图像逐帧及分组](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_connected_blue_growth_20261101/image_native/summary.json>)
- [逐帧实际与虚拟混淆](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_connected_blue_growth_20261101/analysis/per_frame_actual_virtual_confusions.csv>)
- [亮度分层](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_connected_blue_growth_20261101/analysis/brightness_strata.json>)
- [骨架诊断](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_connected_blue_growth_20261101/analysis/centerline_diagnostic.json>)
- [1209种子资格诊断](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_connected_blue_growth_20261101/analysis/fixed1209_seed_failure_diagnosis.json>)
- [video_1全722帧对比](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_connected_blue_growth_20261101/video_1_comparison.mp4>)
- [video_2全1972帧对比](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_connected_blue_growth_20261101/video_2_comparison.mp4>)

![固定1209原图/参考/概率/无候选](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_connected_blue_growth_20261101/analysis/video_2_frame1209.jpg>)

补充：[最近新增5帧的完整分组结果](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_connected_blue_growth_20261101/analysis/latest5_groups.json>)。图像来源标签仍待审，原静态板通用Reviewed标题不代表132图像已人工审核；另存132张已更正标题的板且保留旧展示文件，主指标完全不变。[图像76更正展示](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_connected_blue_growth_20261101/analysis/image_reference_boards/0075.jpg>)。
