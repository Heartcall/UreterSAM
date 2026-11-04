# 固定方向先验与激进蓝色区域：七臂完整对照

2026-11-02。**本轮显著增加了部分暗蓝区域的可见像素召回，但固定第1209帧未恢复，新增误检与成本未通过冻结门槛，暂不采用。尖端定位是否更准仍未验证。** 默认A0、正式语义掩膜、原权重/标签/训练数据和历史结果均未改变。

## 1. 冻结方案与复现

先冻结协议与参数，再新增实现、合成测试及真实结果。研究底座仍为old190结石+A0 model1光纤，2网49,202,790参数，FP32/batch1、RGB/PIL256/ImageNet输入，概率映射回原图后沿用类别竞争。raw2与历史distal_only完整逐帧精确复现；不是将缓存当正式前向。

|方案|相对上一臂增加的因素族|
|---|---|
|raw2|冻结原始两网|
|distal_only|上一轮远端单向方案原样|
|visible_bg|11轴向站双侧仅采图内背景，仍需≥6点及≥0.4轴向覆盖|
|fixed_axis|近边缘主体点投影锚点→画面中心的方向提案，只搜正向|
|color_edges|当前颜色外边界可替代强RGB边，允许弱亮度差|
|broad_blue|扩大相对当前种子的蓝色色度族及暗色可靠性范围|
|aggressive|仅在最后放宽普通截面的宽比/转角/曲率|



相邻臂隔离一个**因素族**，不是单一参数或完整析因；组内相互关联的参数一次性预登记，不能声称排除顺序交互。最终aggressive预先指定，不按结果挑选最有利分支。固定方向是未标定启发式，局部轴仍由当前外边界拟合，并非假定物理光纤轴必穿图像中心。

颜色只用当前帧可信主体建立归一化RGB色度和当前背景对比；不设固定V阈值，不因P<0.1否决，不从两视频学习共享模板。两可见颜色外边界可低RGB对比，真正单侧出视野仍需更严颜色/宽度、兼容T1和段数上限。种子外生长每站必须有当前RGB见证；不跨黑隙/遮挡，不生成隐藏轮廓。原石头先屏障再4连通，只向原背景加入候选。

202静态共享真实两网推理；14个独立顺序进程完成七臂各722+1972帧，共18858正式配对帧，另168预热，总19228配对/38456网络分支调用。T1只由原概率因果更新、分视频及预热后清空；132图像无状态。完整失败结果也保留。

## 2. 分割质量：70因果与132图像

以下为候选与原语义结果的**虚拟合并**评估，没有写回正式掩膜。70来自完整视频因果运行；70独立静态仅为诊断。无参考的全片帧不报告准确率，70/132均已参与开发，不能称独立最终测试。

|范围|方案|Dice|Precision|Recall|边界F(3px)|TP|FP|FN|
|---|---|---|---|---|---|---|---|---|
|video_causal|raw2|0.945537|0.970967|0.921404|0.862027|417783|12492|35637|
|video_causal|distal_only|0.950679|0.970644|0.931518|0.881988|422369|12774|31051|
|video_causal|visible_bg|0.951014|0.970646|0.932160|0.884966|422660|12782|30760|
|video_causal|fixed_axis|0.951731|0.970722|0.933470|0.888508|423254|12766|30166|
|video_causal|color_edges|0.951099|0.970839|0.932145|0.885198|422653|12695|30767|
|video_causal|broad_blue|0.951407|0.970599|0.932958|0.887541|423022|12814|30398|
|video_causal|aggressive|0.959758|0.970400|0.949347|0.916320|430453|13130|22967|
|image_native|raw2|0.765758|0.643433|0.945513|0.524869|105281|58343|6067|
|image_native|distal_only|0.765758|0.643433|0.945513|0.524869|105281|58343|6067|
|image_native|visible_bg|0.765758|0.643433|0.945513|0.524869|105281|58343|6067|
|image_native|fixed_axis|0.765758|0.643433|0.945513|0.524869|105281|58343|6067|
|image_native|color_edges|0.765758|0.643433|0.945513|0.524869|105281|58343|6067|
|image_native|broad_blue|0.765758|0.643433|0.945513|0.524869|105281|58343|6067|
|image_native|aggressive|0.765758|0.643433|0.945513|0.524869|105281|58343|6067|
|video_native|raw2|0.945537|0.970967|0.921404|0.862027|417783|12492|35637|
|video_native|distal_only|0.950679|0.970644|0.931518|0.881988|422369|12774|31051|
|video_native|visible_bg|0.951014|0.970646|0.932160|0.884966|422660|12782|30760|
|video_native|fixed_axis|0.951731|0.970722|0.933470|0.888508|423254|12766|30166|
|video_native|color_edges|0.951099|0.970839|0.932145|0.885198|422653|12695|30767|
|video_native|broad_blue|0.951407|0.970599|0.932958|0.887541|423022|12814|30398|
|video_native|aggressive|0.959758|0.970400|0.949347|0.916320|430453|13130|22967|



|范围|方案|新增TP|新增FP|新增Precision|P<0.1 FN覆盖|原暗FN|虚拟暗FN|暗FN下降比例|新增帧|
|---|---|---|---|---|---|---|---|---|---|
|video_causal|distal_only|4586|282|0.942071|3556|17465|15504|0.112282|19|
|video_causal|visible_bg|4877|290|0.943875|3757|17465|15438|0.116061|20|
|video_causal|fixed_axis|5471|274|0.952306|4207|17465|15348|0.121214|21|
|video_causal|color_edges|4870|203|0.959984|3958|17465|15972|0.085485|15|
|video_causal|broad_blue|5239|322|0.942097|4185|17465|15237|0.127569|16|
|video_causal|aggressive|12670|638|0.952059|10328|17465|10188|0.416662|38|
|image_native|distal_only|0|0|不可计算|0|507|507|0.000000|0|
|image_native|visible_bg|0|0|不可计算|0|507|507|0.000000|0|
|image_native|fixed_axis|0|0|不可计算|0|507|507|0.000000|0|
|image_native|color_edges|0|0|不可计算|0|507|507|0.000000|0|
|image_native|broad_blue|0|0|不可计算|0|507|507|0.000000|0|
|image_native|aggressive|0|0|不可计算|0|507|507|0.000000|0|
|video_native|distal_only|4586|282|0.942071|3556|17465|15504|0.112282|19|
|video_native|visible_bg|4877|290|0.943875|3757|17465|15438|0.116061|20|
|video_native|fixed_axis|5471|274|0.952306|4207|17465|15348|0.121214|21|
|video_native|color_edges|4870|203|0.959984|3958|17465|15972|0.085485|15|
|video_native|broad_blue|5239|322|0.942097|4185|17465|15237|0.127569|16|
|video_native|aggressive|12670|638|0.952059|10328|17465|10188|0.416662|38|



暗段定义沿用原RGB OpenCV Lab L[0,85)，仅用于评价分层，不是推理蓝色亮度硬门控；HSV V三档另存。原正确光纤/结石删除均为0，正式所有像素/概率变化0。无新增时Precision不可计算，不能称100%或通过。全部参考FP保留，132来源标签仍待审核，不因可能缺标而更改主指标。

|原始范围|类别|Dice|Precision|Recall|边界F|TP|FP|FN|
|---|---|---|---|---|---|---|---|---|
|video_causal|stone|0.740216|0.899778|0.628722|0.475440|152516|16988|90065|
|video_causal|fiber|0.945537|0.970967|0.921404|0.862027|417783|12492|35637|
|image_native|stone|0.906751|0.880212|0.934940|0.393852|6120805|832978|425928|
|image_native|fiber|0.765758|0.643433|0.945513|0.524869|105281|58343|6067|



所有臂的结石上述指标完全相同。逐帧两类TP/FP/FN和指标、结石屏障冲突及参考光纤冲突计数均保留；既有原石头误占真实光纤的像素也不越权覆盖。

## 3. 累加因素归因与固定1209

|范围|相邻对照|获得TP|丢失TP|增加FP|移除FP|候选变化帧|
|---|---|---|---|---|---|---|
|video_causal|raw2->distal_only|4586|0|282|0|19|
|video_causal|distal_only->visible_bg|291|0|8|0|8|
|video_causal|visible_bg->fixed_axis|635|41|43|59|23|
|video_causal|fixed_axis->color_edges|544|1145|1|72|10|
|video_causal|color_edges->broad_blue|1028|659|122|3|15|
|video_causal|broad_blue->aggressive|7431|0|316|0|22|
|image_native|raw2->distal_only|0|0|0|0|0|
|image_native|distal_only->visible_bg|0|0|0|0|0|
|image_native|visible_bg->fixed_axis|0|0|0|0|0|
|image_native|fixed_axis->color_edges|0|0|0|0|0|
|image_native|color_edges->broad_blue|0|0|0|0|0|
|image_native|broad_blue->aggressive|0|0|0|0|0|
|video_native|raw2->distal_only|4586|0|282|0|19|
|video_native|distal_only->visible_bg|291|0|8|0|8|
|video_native|visible_bg->fixed_axis|635|41|43|59|23|
|video_native|fixed_axis->color_edges|544|1145|1|72|10|
|video_native|color_edges->broad_blue|1028|659|122|3|15|
|video_native|broad_blue->aggressive|7431|0|316|0|22|



此处丢失TP指相邻两个实验虚拟候选之间的变化，不能与删除原网络正确像素混淆；后者为0。颜色/边缘放宽可能改变种子拟合与路径，故结果不必单调；不能只凭总Dice把每项变化都说成有效。

70因果参考中，可见背景布局净增291TP/8FP；固定轴再净增594TP、净减16FP，但也丢失41个此前候选TP。弱RGB边缘资格单独放宽后反而净少601TP，说明路径/资格交互明显；扩大蓝色族净增369TP/119FP。最终几何容差因素带来最大的7431TP/316FP增量。因此收益主要来自更多原被形态整帧回退的当前颜色候选获准，不能单独归功于蓝色角度放宽或认为边缘越弱越好。

|video2显示1209|方案|新增TP|新增FP|暗FN覆盖|停止原因|有效背景点|原端部状态|
|---|---|---|---|---|---|---|---|
|video_native|raw2|0|0|0|raw_control||未进入|
|video_native|distal_only|0|0|0|no_trusted_seed|legacy10-site|未进入|
|video_native|visible_bg|0|0|0|no_trusted_seed|10|未进入|
|video_native|fixed_axis|0|0|0|direction_or_curvature_jump|13|unknown;unknown|
|video_native|color_edges|0|0|0|direction_or_curvature_jump|13|unknown;unknown|
|video_native|broad_blue|0|0|0|direction_or_curvature_jump|13|unknown;unknown|
|video_native|aggressive|0|0|0|direction_or_curvature_jump|13|unknown;unknown|
|video_causal|raw2|0|0|0|raw_control||未进入|
|video_causal|distal_only|0|0|0|no_trusted_seed|legacy10-site|未进入|
|video_causal|visible_bg|0|0|0|no_trusted_seed|10|未进入|
|video_causal|fixed_axis|0|0|0|direction_or_curvature_jump|13|unknown;unknown|
|video_causal|color_edges|0|0|0|direction_or_curvature_jump|13|unknown;unknown|
|video_causal|broad_blue|0|0|0|direction_or_curvature_jump|13|unknown;unknown|
|video_causal|aggressive|0|0|0|direction_or_curvature_jump|13|unknown;unknown|



第1209帧原光纤FN4002、低Lab FN2626。修正采样解决了旧5/6图内背景不足，但这不等于完成暗前端恢复。固定轴后种子可建立，当前版本仍被后续几何检查拒绝，原FN保持；没有通过调参把这个固定帧调通。

|1209同一拒绝截面|方案|单步角/限度|总角/限度|曲率量/限度|触发项|
|---|---|---|---|---|---|
|video_native|fixed_axis|4.567 / 10.000|4.567 / 20.000|0.566 / 0.350|curvature_width|
|video_native|color_edges|4.567 / 10.000|4.567 / 20.000|0.566 / 0.350|curvature_width|
|video_native|broad_blue|4.602 / 10.000|4.602 / 20.000|0.572 / 0.350|curvature_width|
|video_native|aggressive|10.872 / 15.000|15.474 / 30.000|1.244 / 1.000|curvature_width|
|video_causal|fixed_axis|4.567 / 10.000|4.567 / 20.000|0.566 / 0.350|curvature_width|
|video_causal|color_edges|4.567 / 10.000|4.567 / 20.000|0.566 / 0.350|curvature_width|
|video_causal|broad_blue|4.602 / 10.000|4.602 / 20.000|0.572 / 0.350|curvature_width|
|video_causal|aggressive|10.872 / 15.000|15.474 / 30.000|1.244 / 1.000|curvature_width|



最终aggressive在该帧仍有两侧可见强RGB边界；失败检查为宽度归一化的局部曲率量，而非亮度或P<0.1排除。截面重心移动、局部轴与宽度都会影响该量，不能仅凭超过门槛就证明图中没有真实光纤。数值为只读复算同一冻结拒绝截面的归因，不产生新变体或替代结果；没有把被拒前的提案片段挑出来当成功。

## 4. 完整视频性能与异常

|方案|视频/帧|总P50ms|总P95ms|额外P95ms|含导出fps|新增帧|异常拒绝帧|原端点候选|可见性审计后|
|---|---|---|---|---|---|---|---|---|---|
|raw2|video_1/722|19.22|27.89|0.01|33.59|0|0|0|0|
|raw2|video_2/1972|19.50|28.35|0.01|33.17|0|0|0|0|
|distal_only|video_1/722|38.44|52.96|22.43|20.14|50|327|25|24|
|distal_only|video_2/1972|40.24|56.66|25.71|19.50|989|724|0|0|
|visible_bg|video_1/722|38.85|54.33|22.69|19.81|51|329|25|24|
|visible_bg|video_2/1972|40.92|56.92|26.42|19.06|992|722|0|0|
|fixed_axis|video_1/722|30.56|42.42|16.19|25.06|44|305|21|20|
|fixed_axis|video_2/1972|31.90|44.19|18.26|24.19|1025|815|0|0|
|color_edges|video_1/722|30.50|41.61|15.83|25.18|44|419|21|20|
|color_edges|video_2/1972|31.39|44.53|18.34|24.36|737|1132|0|0|
|broad_blue|video_1/722|30.91|42.95|15.95|24.98|52|427|22|20|
|broad_blue|video_2/1972|31.72|42.91|18.42|24.47|805|1064|0|0|
|aggressive|video_1/722|30.49|42.31|16.39|25.09|138|49|114|112|
|aggressive|video_2/1972|33.34|46.69|20.17|23.14|1640|201|9|9|



RTX4070Ti/torch2.11+cu128，所有臂仍2网，无额外GPU网络。直接总延迟含两网及原图映射、颜色/几何/端部/T1和必要审计；实际离线吞吐再含解码、细日志、PNG和视频导出。每阶段P50/P95/max、CPU秒、工作集/CUDA峰值均在各worker summary；不相加分位数，不把离线导出吞吐等同实时显示性能。额外5ms、总P95≤源帧间隔的门槛不变。

进程按登记顺序串行运行，机器时钟/负载可随时间变化；这轮耗时用于直接报告和预算判断，不把不同臂的时间差直接解释成某个因素的因果加速。

|方案|视频|CPU进程秒|墙钟秒(含导出)|工作集峰值MiB|CUDA分配峰值MiB|
|---|---|---|---|---|---|
|raw2|video_1|331.97|21.50|2005.7|223.9|
|raw2|video_2|924.38|59.45|2049.4|223.9|
|distal_only|video_1|558.55|35.84|2054.9|223.9|
|distal_only|video_2|1565.47|101.13|2090.8|223.9|
|visible_bg|video_1|561.25|36.45|1961.8|223.9|
|visible_bg|video_2|1586.02|103.49|2090.6|223.9|
|fixed_axis|video_1|458.73|28.81|2055.8|223.9|
|fixed_axis|video_2|1303.97|81.51|2126.0|223.9|
|color_edges|video_1|457.11|28.67|2070.1|223.9|
|color_edges|video_2|1300.72|80.96|2126.8|223.9|
|broad_blue|video_1|462.64|28.91|2103.4|223.9|
|broad_blue|video_2|1288.45|80.58|2114.5|223.9|
|aggressive|video_1|459.95|28.78|2110.1|223.9|
|aggressive|video_2|1361.45|85.23|2116.9|223.9|



异常形态或面积超限清空全部新增并保留原分割；无唯一可信种子、颜色不可靠、真正轴向空隙或不可见状态不强补。全部18858帧候选/提案/石头冲突及14段回放已保留，可见性审计只防止端点落在根主体不可达区域，不证明没有身份误认或虚构解剖尖端。无语义真值帧的异常状态数量也不是误检率。

## 5. 尖端、分组与采用判定

端点原始检测、内部颜色过渡、疑似遮挡、出视野/未知状态和统一可见性审计分开保存。只有当前边界与端外背景证据充分才给候选xy；不能因生长变长就确定最终物理尖端。没有独立尖端真值，尖端Precision/Recall/定位误差仍null，无法声称定位改善，也未验证遮挡恢复、尺度、激光落点或Godot真实标定。

|范围|分组|帧数|原Dice|激进Dice|新增TP|新增FP|
|---|---|---|---|---|---|---|
|video_causal|video_1|43|0.976570|0.976615|51|31|
|video_causal|video_2|27|0.906306|0.939144|12619|607|
|video_causal|original29|29|0.969235|0.978605|3648|441|
|video_causal|new_user_approved|41|0.929877|0.947374|9022|197|
|image_native|kidney_stones_fiber_v1|68|0.798775|0.798775|0|0|
|image_native|kidney_stones_fiber_v2|64|0.748768|0.748768|0|0|
|image_native|old190_exposed25|25|0.643663|0.643663|0|0|
|image_native|not_matched107|107|0.797731|0.797731|0|0|
|video_causal|latest5|5|0.844614|0.904645|4069|113|



原29/新增41/最近5、43/27病例、68/64来源、25/107旧训练曝光及全部争议样本未排除。中心线3native px语义骨架完整性另存，为事后诊断，不是实际轴线或尖端真值。

|范围|方案|参考骨架覆盖|预测骨架Precision|原端点候选数|可见审计后|
|---|---|---|---|---|---|
|video_causal|raw2|0.760009|0.823350|0|0|
|video_causal|aggressive|0.820220|0.822693|17|17|
|image_native|raw2|0.754837|0.590521|0|0|
|image_native|aggressive|0.754837|0.590521|0|0|



最终aggressive拒绝项（全部门槛及各中间臂结果另存，不隐藏失败）：


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


另须同时通过独立性能门槛。部分相对classwise的不足已存在于raw2研究底座，不全部归因于本次改动。候选Precision98%、暗FN三个范围下降10%、原29及raw2/classwise非退化、原TP保护0损失均未降低。**可认定这轮在视频开发参考中提供了更强暗蓝像素召回证据，但不满足整体采用条件，不替换默认、不自动进入训练或结石S3。**

工程验证：76项机制/回归测试通过（21新+55旧），独立审计137/137通过，覆盖冻结源码/权重/参考/视频/归档、逐帧原概率/掩膜、旧方案精确复现、参考帧独立混淆矩阵及当前颜色/连通、全部帧原类别保护、固定方向/T1无反馈、14进程直接计时、全部PNG/回放及状态清空。工程通过不等于质量验收。未读521训练/144保留test像素，无stone_new/视频训练/伪标签或共享模板学习。

下一步建议仅做独立的低成本局部截面/端部归因设计：区分固定轴启发式、截面重心偏移和“局部拒绝导致整帧候选清空”各自影响，同时补最小可见尖端审核参考。当前轮参数与拒绝结果冻结，不继续扫阈值；新的像素修改或训练须另列协议，不因本轮Recall收益直接部署。

## 6. 结果入口

- [冻结实施协议](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/docs/plans/fiber_aggressive_blue_execution_20261102.md>)

- [冻结参数](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/configs/fiber_aggressive_blue_control_20261102.json>)

- [总结果/所有门槛](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_aggressive_blue_control_20261102/summary.json>)

- [独立工程审计](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_aggressive_blue_control_20261102/engineering_checks.json>)

- [逐帧两类指标](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_aggressive_blue_control_20261102/analysis/per_frame_actual_virtual.csv>)

- [相邻因素逐帧归因](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_aggressive_blue_control_20261102/analysis/cumulative_factor_pairs.json>)

- [固定1209拒绝数值](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_aggressive_blue_control_20261102/analysis/fixed1209_guard_diagnosis.json>)

- [暗段分层](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_aggressive_blue_control_20261102/analysis/brightness_strata.json>)

- [骨架完整性](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_aggressive_blue_control_20261102/analysis/centerline_diagnostic.json>)

- [端部状态/未验证候选](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_aggressive_blue_control_20261102/analysis/tip_and_mechanism_counts.json>)

- [全部272参考板](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_aggressive_blue_control_20261102/analysis/all_reference_boards>)

- [video_1完整722帧八格对比](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_aggressive_blue_control_20261102/video_1_comparison.mp4>)

- [video_2完整1972帧八格对比](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_aggressive_blue_control_20261102/video_2_comparison.mp4>)


对比视频：上排原图/raw2/distal_only/visible_bg，下排fixed_axis/color_edges/broad_blue/aggressive；绿色仅影子候选，紫色为拒绝提案/搜索范围，黄色为尚无尖端真值的候选位置。紫色不是分割输出，也不保证每个提案像素为真实光纤。

![固定1209原图/参考/概率/候选及TPFP](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_aggressive_blue_control_20261102/analysis/video_causal_video2_frame1209.jpg>)

显示987是新增FP最多的参考帧：967TP/100FP。黄色可见恢复与粉色边界外扩同时保留，仍有末端红色FN；这不是端点已准确定位的证据。此图为事后错误展示，不用于选择方案。

![显示987新增正确与误检并存](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_aggressive_blue_control_20261102/analysis/all_reference_boards/video_causal_0058.jpg>)
