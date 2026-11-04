# 可信主体蓝色连通分割与轮廓后处理：完整对照

2026-11-02。**直接连通蓝色提取恢复了此前被几何回退阻断的真实光纤，包括固定第1209帧；代价是明显增加误检。轮廓修整减少少量FP，也删除正确新增像素，未满足冻结质量/成本条件，当前不采用。尖端定位准确性未验证。** 原默认A0、正式语义输出、权重、标签与历史结果保持不变。

## 1. 冻结设计及实际边界

先冻结参数与协议，再实现及观察结果。raw2是原两网研究基线；aggressive完整精确复现上一轮；connected_blue_union替换为当前RGB4连通组件；connected_blue_regularized只在同一组件后增加轮廓修剪。最终regularized预先指定。aggressive→union同时替换了提取机制和有界搜索/绝对面积保护，是机制对照，不能把所有差值只归因于曲率一项；union→regularized可独立衡量后处理影响。

主体身份、P≥0.9、同帧归一化RGB蓝色色度/可见背景及唯一合格种子资格完全继承aggressive。入口→中心的轴向和45°宽锥限制前向范围，不使用初始宽度设定延伸距离，也不再逐截面检查宽比、转角或曲率。颜色门槛沿用上一轮，不因P<0.1或亮度下降排除；黑隙/不明像素、原石头先屏障后4连通，不能用历史或原网络非蓝区域补桥。组件>25%或新增>10%原图面积仍整体回退，作为灾难性扩张保护；初始种子外边界资格仍可能拒绝真实光纤，并非完全无几何依赖。

轮廓后处理为1s高斯占据平滑、0.5阈值、与原蓝色组件求交并保护蓝色种子核心及其4连通，s=min(H,W)/400。它只能修剪union，不能新增候选外像素；全部原fiber/stone仍保留，因此原掩膜已有锯齿也不会被本轮强制理想化。可修剪外侧阶梯与可见端帽，但不强造直边、单调锥形、圆帽或不可见延长，不填洞/闭运算。

端部观察依赖当前组件前沿、局部宽度收拢、当前RGB端边界和端外已知非蓝背景。蓝色/原fiber继续为internal_transition，石头接触为occlusion_suspected，边界弱、范围切断、黑背景不可辨或安全拒绝则unknown/out_of_view。候选坐标只能来自可见前沿，不能因掩膜变长就断言找到真实头端。T1只用原概率因果更新，分视频/预热清空，不由新像素反馈；静态无状态。

四臂各完整video_1/722、video_2/1972，共10776正式配对帧，8独立顺序进程；202静态共享原两网前向，96预热，总11074配对/22148网络分支调用。保持old190结石+A0 model1光纤、2网49,202,790参数、FP32/batch1、RGB/PIL256/ImageNet及原图映射/类别竞争。全部失败也跑完。

## 2. 70因果、132图像及静态诊断

所有新分割指标来自影子候选与原mask的**虚拟合并**，正式输出完全不变。70主指标抽自完整视频因果运行，70独立静态另列；132逐图独立。70/132已用于开发，不能称独立最终泛化测试，也不能外推无像素参考的全片准确率。

|范围|方案|Dice|Precision|Recall|边界F(3px)|TP|FP|FN|
|---|---|---|---|---|---|---|---|---|
|video_causal|raw2|0.945537|0.970967|0.921404|0.862027|417783|12492|35637|
|video_causal|aggressive|0.959758|0.970400|0.949347|0.916320|430453|13130|22967|
|video_causal|connected_blue_union|0.965362|0.961827|0.968923|0.943348|439329|17436|14091|
|video_causal|connected_blue_regularized|0.965411|0.962151|0.968693|0.943246|439225|17278|14195|
|image_native|raw2|0.765758|0.643433|0.945513|0.524869|105281|58343|6067|
|image_native|aggressive|0.765758|0.643433|0.945513|0.524869|105281|58343|6067|
|image_native|connected_blue_union|0.760378|0.634772|0.947956|0.513183|105553|60732|5795|
|image_native|connected_blue_regularized|0.760487|0.634924|0.947956|0.513684|105553|60692|5795|
|video_native|raw2|0.945537|0.970967|0.921404|0.862027|417783|12492|35637|
|video_native|aggressive|0.959758|0.970400|0.949347|0.916320|430453|13130|22967|
|video_native|connected_blue_union|0.965362|0.961827|0.968923|0.943348|439329|17436|14091|
|video_native|connected_blue_regularized|0.965411|0.962151|0.968693|0.943246|439225|17278|14195|



|范围|方案|新增TP|新增FP|新增Precision|P<0.1 FN恢复|暗FN原|暗FN后|亮FN原|亮FN后|新增帧|
|---|---|---|---|---|---|---|---|---|---|---|
|video_causal|aggressive|12670|638|0.952059|10328|17465|10188|78|78|38|
|video_causal|connected_blue_union|21546|4944|0.813364|15530|17465|6046|78|75|70|
|video_causal|connected_blue_regularized|21442|4786|0.817523|15464|17465|6118|78|75|70|
|image_native|aggressive|0|0|不可计算|0|507|507|3656|3656|0|
|image_native|connected_blue_union|272|2389|0.102217|179|507|507|3656|3510|8|
|image_native|connected_blue_regularized|272|2349|0.103777|179|507|507|3656|3510|8|
|video_native|aggressive|12670|638|0.952059|10328|17465|10188|78|78|38|
|video_native|connected_blue_union|21546|4944|0.813364|15530|17465|6046|78|75|70|
|video_native|connected_blue_regularized|21442|4786|0.817523|15464|17465|6118|78|75|70|



暗/亮分组沿用当前原RGB的OpenCV Lab L[0,85)/[170,256)，HSV V三档另存；不是蓝色判别的亮度硬阈值，也不是尖端标注。颜色较深的可见本体恢复不能直接称尖端Recall提高。所有原正确fiber/stone删除0，所有正式概率/掩膜变化0。

视频新增TP主要来自低/中Lab亮度，union分别恢复11419/10124个FN，高Lab组仅3个；132图像低Lab FN一个也未恢复。视觉上的“亮蓝”不必落入高Lab档，因此颜色外观描述与亮度分层分开报告，不能称所有亮度条件都改善。

70因果中union新增像素另含76个参考未知像素，regularized保留65个；这些均不作为TP或FP计分，分母按冻结有效参考计算。修整另删11个未知参考像素，不能把它们判成正确或误检。

|结石(各臂相同)|Dice|Precision|Recall|TP|FP|FN|边界F|
|---|---|---|---|---|---|---|---|
|video_causal|0.740216|0.899778|0.628722|152516|16988|90065|0.475440|
|image_native|0.906751|0.880212|0.934940|6120805|832978|425928|0.393852|



## 3. 提取与轮廓处理的独立影响

|范围|配对|获得TP|丢失TP|增加FP|减少FP|候选变化帧|
|---|---|---|---|---|---|---|
|video_native|raw2->aggressive|12670|0|638|0|38|
|video_native|aggressive->connected_blue_union|8876|0|4306|0|70|
|video_native|connected_blue_union->connected_blue_regularized|0|104|0|158|53|
|image_native|raw2->aggressive|0|0|0|0|0|
|image_native|aggressive->connected_blue_union|272|0|2389|0|8|
|image_native|connected_blue_union->connected_blue_regularized|0|0|0|40|6|
|video_causal|raw2->aggressive|12670|0|638|0|38|
|video_causal|aggressive->connected_blue_union|8876|0|4306|0|70|
|video_causal|connected_blue_union->connected_blue_regularized|0|104|0|158|53|



这里丢失TP专指实验候选之间变化，原网络正确像素始终未删。轮廓更平滑不代表更准确：如果降低Recall或边界F，必须按冻结规则拒绝。所有原始RGB资格图、完整连通蓝色组件、处理后组件、最终新增、未采纳提案和石头冲突均逐帧保存，不能只保留平滑后的成功画面。

132来源参考仍待审核，蓝色结构和来源标签可能存在分歧，但全部按冻结标签计算，不自动将蓝色预测改成真值、不去除争议图像或只汇报视频改善。当前蓝色身份/原网络高P种子不保证一定是真光纤，连通也不能排除与其他蓝色结构接触的误扩张。

错误板检查：修整后视频FP最多的是显示1898，新增174TP/492FP，新增区域主要沿两侧轮廓外缘，存在“原掩膜已较完整却继续扩边”的问题；并非所有新增都来自暗前端。图像开发集显示69为5TP/1042FP，显示129为0TP/507FP且来源参考无光纤，RGB中仍有蓝色结构。后两者需区分边界标注/身份分歧，当前观察不足以定错标，全部冻结参考计数照常保留。

## 4. 固定1209与1233–1291

|video2显示1209|方案|新增TP|新增FP|剩余FN|Dice|Precision|Recall|边界F|暗FN恢复|状态|尖端判断|
|---|---|---|---|---|---|---|---|---|---|---|---|
|video_native|raw2|0|0|4002|0.782742|0.984477|0.649624|0.306034|0|raw_control|无观察|
|video_native|aggressive|0|0|4002|0.782742|0.984477|0.649624|0.306034|0|direction_or_curvature_jump|proximal_not_searched;direction_or_curvature_jump|
|video_native|connected_blue_union|3482|249|520|0.960952|0.967519|0.954474|0.821290|2278|candidate_accepted|weak_terminal_RGB_boundary|
|video_native|connected_blue_regularized|3481|243|521|0.961160|0.968031|0.954386|0.821593|2277|candidate_accepted|weak_terminal_RGB_boundary|
|video_causal|raw2|0|0|4002|0.782742|0.984477|0.649624|0.306034|0|raw_control|无观察|
|video_causal|aggressive|0|0|4002|0.782742|0.984477|0.649624|0.306034|0|direction_or_curvature_jump|proximal_not_searched;direction_or_curvature_jump|
|video_causal|connected_blue_union|3482|249|520|0.960952|0.967519|0.954474|0.821290|2278|candidate_accepted|weak_terminal_RGB_boundary|
|video_causal|connected_blue_regularized|3481|243|521|0.961160|0.968031|0.954386|0.821593|2277|candidate_accepted|weak_terminal_RGB_boundary|



旧aggressive在1209被局部曲率1.244>1.0清空。新连通提取不再使用该检查，恢复了真实可见亮/暗蓝像素，也增加边界外误检。当前末端RGB梯度仍不足，不输出确定尖端；不能把候选前沿坐标或更长mask当作准确物理头端。

用户指定1233–1291全部59帧已按四臂完整导出，加1209共240条焦点逐帧诊断。**该59帧区间在冻结70参考中没有任何人工像素掩膜**，所以区间TP/FP/Dice/Recall/边界F和尖端准确率无法计算；只能逐帧检查蓝色遗漏、绿色候选、紫色未采纳/修剪提案和端部状态。未把模型预测用作真值或虚构区间精度。

|1233–1291机制(非精度)|帧数|有新增帧|异常回退帧|新增像素总数|轮廓删除候选|原端点候选|可见性审计后|
|---|---|---|---|---|---|---|---|
|raw2|59|0|0|0|0|0|0|
|aggressive|59|0|59|0|0|0|0|
|connected_blue_union|59|59|0|219760|0|34|34|
|connected_blue_regularized|59|59|0|219310|450|32|32|



该区间aggressive的59帧全部因direction_or_curvature_jump拒绝新增；union与regularized的59帧均获准，说明不再受原截面异常整段清空。union有34帧给端点候选、regularized有32帧，其余主要为端边界弱或三剖面不一致；数量不是准确率。预先固定显示1233/1250/1266/1291截图中，绿色覆盖了原掩膜截短的蓝色前段，紫色小提案对应旧几何拒绝；平滑前后差异主要为细边缘修剪。当前颜色边界仍可能偏离真实轮廓，不据此断言整段分割或端点正确。

完整59帧短片、每一帧RGB及全部四臂逐帧记录均保留；紫色是拒绝/修剪部分，绿色是实验新增，原正式语义结果没有被这些颜色改变。后续人工补标可单列新参考版本，本轮不以可视化自行裁判像素真值。

## 5. 性能、端部与分组

|方案|视频/帧|总P50ms|总P95ms|额外P95ms|含导出fps|有新增帧|异常帧|端点候选|可见审计后|
|---|---|---|---|---|---|---|---|---|---|
|raw2|video_1/722|15.30|19.84|0.01|38.72|0|0|0|0|
|raw2|video_2/1972|15.81|20.69|0.01|37.79|0|0|0|0|
|aggressive|video_1/722|31.80|44.25|17.38|22.66|138|49|114|112|
|aggressive|video_2/1972|51.24|67.86|32.65|15.64|1640|201|9|9|
|connected_blue_union|video_1/722|57.88|82.16|43.77|14.01|721|0|428|428|
|connected_blue_union|video_2/1972|57.87|71.99|41.11|14.56|1891|0|848|848|
|connected_blue_regularized|video_1/722|72.85|82.73|45.84|11.96|721|0|409|409|
|connected_blue_regularized|video_2/1972|62.47|82.50|46.85|13.32|1889|0|777|777|



|方案|视频|CPU秒|墙钟秒含导出|工作集MiB|CUDA峰值MiB|
|---|---|---|---|---|---|
|raw2|video_1|291.56|18.65|1989.0|223.9|
|raw2|video_2|819.58|52.18|2093.4|223.9|
|aggressive|video_1|503.67|31.86|2089.2|223.9|
|aggressive|video_2|1992.72|126.13|2102.3|223.9|
|connected_blue_union|video_1|812.27|51.54|2082.9|223.9|
|connected_blue_union|video_2|2140.92|135.45|2132.3|223.9|
|connected_blue_regularized|video_1|960.11|60.38|2054.7|223.9|
|connected_blue_regularized|video_2|2342.08|148.07|2051.7|223.9|



RTX4070Ti/torch2.11+cu128，仍两网，无额外网络。独立进程直接总计时包含网络、原图映射、种子、RGB组件、可选平滑、端部和T1；吞吐另包含解码、六类PNG/细审计/视频导出。详细阶段P50/P95/max在worker summary，不相加分位数，不把离线导出fps当实时显示。顺序运行可能有负载/时钟变化，不把两臂时间差单独当因果加速。

任何端点数量增加都不等于定位更准。12帧独立尖端专项尚无确认真值，因此尖端Precision/Recall/误差、物理落点、遮挡恢复、标定与尺度仍未验证。可见性审计仅检查坐标附近有根主体相连的当前mask，不能确认解剖身份。

|范围|分组|帧数|原Dice|修整后Dice|新增TP|新增FP|
|---|---|---|---|---|---|---|
|video_causal|video_1|43|0.976570|0.976288|1032|1225|
|video_causal|video_2|27|0.906306|0.952399|20410|3561|
|video_causal|original29|29|0.969235|0.975720|4941|2850|
|video_causal|new_user_approved|41|0.929877|0.958681|16501|1936|
|image_native|kidney_stones_fiber_v1|68|0.798775|0.795836|267|749|
|image_native|kidney_stones_fiber_v2|64|0.748768|0.742261|5|1600|
|image_native|old190_exposed25|25|0.643663|0.643089|0|51|
|image_native|not_matched107|107|0.797731|0.790900|272|2298|
|video_causal|latest5|5|0.844614|0.958041|8466|619|



原29/新增41/最近5、病例43/27、来源68/64、旧训练曝光25/107、全部争议样本均保留；语义骨架3px覆盖另存，非物理中心线/尖端真值。

## 6. 验收与下一步

- aggressive：质量通过=False，质量及成本共同通过=False。

- connected_blue_union：质量通过=False，质量及成本共同通过=False。

- connected_blue_regularized：质量通过=False，质量及成本共同通过=False。


最终regularized拒绝项完整保留：


- video_causal_new_precision

- video_causal:raw2:fiber_precision

- video_causal:classwise3:stone_dice

- video_causal:classwise3:stone_precision

- video_causal:classwise3:fiber_precision

- video_causal:classwise3:stone_boundary_f

- video_causal_regularizer_Recall_nondecrease

- video_causal_regularizer_boundaryF_nondecrease

- image_native_new_precision

- image_native_no_growth_on_reference_absent

- image_native_dark_FN_10percent

- image_native:raw2:fiber_dice

- image_native:raw2:fiber_precision

- image_native:raw2:fiber_boundary_f

- image_native:classwise3:stone_precision

- image_native:classwise3:fiber_dice

- image_native:classwise3:fiber_precision

- image_native:classwise3:stone_boundary_f


新增Precision70/132各98%、暗FN70/132/1209各下降10%、原29及raw2/classwise非退化、原TP保护、额外P95≤5ms与源帧间隔预算均未放宽。部分底座相对classwise不足已存在，不能都归因于本轮。**结论：直接RGB连通路线证明了旧局部几何检查会阻断一部分真实可见光纤；当前提取仍引入过多FP，简单轮廓修剪不能弥补且有TP损失，不宜采用。**

97项机制/回归测试通过（21新+76旧），独立工程审计92/92通过：源码/协议/归档/模型/输入身份，旧aggressive逐帧精确复现，参考独立混淆/颜色/连通，全部正式帧阶段哈希、共享union输入、后处理子集及原类保护、直接计时和重置。工程通过不代表质量验收。未读取521训练或144保留test像素；stone_new和验证视频继续排除所有训练/伪标签/共享模板标定。

本轮停止，不继续调颜色/平滑阈值或部署。后续优先只读区分新增FP是轮廓边缘偏移、主体身份误认还是连通到其他蓝色结构，并以当前图像边界证据设计局部候选排序；同时单列59帧区间及可见尖端最小标注方案。新算法/训练另行冻结，不能把理想轮廓裁剪或更强平滑作为已证实方案。

## 7. 文件与完整视频

- [冻结执行协议](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/docs/plans/connected_blue_union_execution_20261102.md>)

- [冻结配置](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/configs/connected_blue_union_control_20261102.json>)

- [总结果及所有门槛](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_connected_blue_union_20261102/summary.json>)

- [独立核验](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_connected_blue_union_20261102/engineering_checks.json>)

- [逐帧两类指标](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_connected_blue_union_20261102/analysis/per_frame_actual_virtual.csv>)

- [提取/后处理配对归因](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_connected_blue_union_20261102/analysis/stage_pairs.json>)

- [暗/亮完整分层](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_connected_blue_union_20261102/analysis/full_brightness_strata.json>)

- [固定1209](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_connected_blue_union_20261102/analysis/fixed1209_evaluation.json>)

- [240条焦点逐帧记录](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_connected_blue_union_20261102/analysis/all_requested_focus_frames.csv>)

- [全部焦点元数据](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_connected_blue_union_20261102/analysis/focus_all_frames.json>)

- [骨架完整性](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_connected_blue_union_20261102/analysis/centerline_diagnostic.json>)

- [端部候选及机制](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_connected_blue_union_20261102/analysis/tip_and_mechanism_counts.json>)

- [272参考板](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_connected_blue_union_20261102/analysis/all_reference_boards>)

- [video_1完整722帧六格](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_connected_blue_union_20261102/video_1_comparison.mp4>)

- [video_2完整1972帧六格](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_connected_blue_union_20261102/video_2_comparison.mp4>)

- [指定1233–1291全59帧](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_connected_blue_union_20261102/video_2_frames1233-1291_focus.mp4>)


六格：上排原图/原正式mask/aggressive，下排union/regularized/仅后处理删除部分。绿色实验新增、紫色拒绝或修剪、黄色未验证端部候选，均不改变正式结果。

![固定1209修整后参考误差板](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_connected_blue_union_20261102/analysis/video_causal_video2_frame1209.jpg>)

![指定区间1233起点完整对照](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_connected_blue_union_20261102/analysis/video_2_frame1233_full.jpg>)

![最大视频新增FP例1898，黄色TP与粉色FP同时保留](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_connected_blue_union_20261102/analysis/all_reference_boards/video_causal_0067.jpg>)
