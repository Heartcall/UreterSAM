# 蓝色本体外边界与单侧弱边：固定候选完整对照

2026-11-01。落实用户五项原则，完成数值冻结、独立实现及三臂两片完整测试。**新blue_extent未通过，停止这一固定配置，不采用、不改默认或实际掩膜，不训练。** 本轮证明了若干机制可以运行，也暴露出种子背景采样与双端全帧回退的结构限制，不能把失败简单解释成蓝色信息无效。

## 1. 方案与实施边界

三臂：raw2（原两网）、connected_rgb（冻结旧候选）、blue_extent（本帧蓝色族外边界+受约束单侧弱边+独立端帽）。底座仍为old190结石+A0 model1光纤，两网49,202,790参数、FP32/batch1、原RGB PIL256/ImageNet归一化/native连续概率与原类别竞争；这是研究底座，默认A0未替换。

新候选用归一化RGB离中性方向、相对蓝优势和侧背景对比，允许不同明度与原P<0.1像素。65点横截面定位连续蓝色区间的外边界，色度余量零交叉插值避免内部明暗边竞争。当前证据逐像素4连通到种子，原结石在连通前屏障，无膨胀/闭合/隐藏桥接，形状仅限制计算范围。单侧弱边必须更严颜色、宽度、曲率及有效前帧T1；最多2个连续弱截面。强当前证据不受旧历史否决，生长不反馈T1。

原数值协议在实现前冻结；离散定位和轴向RGB见证细节在真实数据实验前另行记录。合成开发中修正反向侧边顺序、出图停止和边界中点抖动，不改冻结阈值。严格曲率仍会拒绝部分锥形栅格轮廓，该局限单列测试保留。

202静态实际推理共享三臂；6独立视频进程各跑完722/1972帧，共8082正式帧、72预热。合计8356次两网配对推理/16712次实际网络分支调用；未用缓存冒充正式前向。70因果指标从完整视频结果抽取，无额外网络。272张参考板=70静态+132图像+同70因果，不是272张独立图像或272次新增推理。

## 2. 主质量结果：70因果与132静态

在结果前固定70视频主门槛采用完整因果输出，132图像无状态；70独立静态另列，不能择优混用。下面为原背景纳入候选后的**虚拟合并**，实际分割/概率逐帧精确保持，实际变化0；原正确光纤及结石删除0。

|范围|方案/输出|光纤Dice|Precision|Recall|边界F(3px)|TP|FP|FN|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|video_causal|raw2|0.945537|0.970967|0.921404|0.862027|417783|12492|35637|
|video_causal|connected_rgb|0.949197|0.971012|0.928340|0.877443|420928|12566|32492|
|video_causal|blue_extent|0.945537|0.970967|0.921404|0.862027|417783|12492|35637|
|image_native|raw2|0.765758|0.643433|0.945513|0.524869|105281|58343|6067|
|image_native|connected_rgb|0.764413|0.640818|0.947076|0.521866|105455|59108|5893|
|image_native|blue_extent|0.765572|0.643149|0.945558|0.524013|105286|58418|6062|

|范围|方案|新增候选TP / FP|新增Precision|原P<0.1 FN覆盖|低Lab FN覆盖|
|---|---|---:|---:|---:|---:|
|video_causal|connected_rgb|3145 / 74|97.7011%|2094|1327|
|video_causal|blue_extent|0 / 0|不可计算（无新增）|0|0|
|image_native|connected_rgb|174 / 765|18.5304%|91|10|
|image_native|blue_extent|5 / 75|6.2500%|0|0|
|video_native|connected_rgb|3145 / 74|97.7011%|2094|1327|
|video_native|blue_extent|0 / 0|不可计算（无新增）|0|0|

新候选70因果和70静态均无接纳新增：光纤TP417783/FP12492/FN35637不变，低Lab FN17465不变。132图像仅1帧接纳80像素（5TP/75FP），光纤FN6067→6062、FP58343→58418；低Lab FN507不变。新增参考Precision6.25%，显著低于98%；暗段10%下降与固定1209门槛失败，无新增视频分母不可计算，不能当100%或通过。主门槛15项拒绝完整保存，包括继承raw2对classwise的部分不足，不能把所有拒绝都归因于新生长。

结石完全不变：70因果Dice/Precision/Recall=0.740216/0.899778/0.628722，TP152516/FP16988/FN90065，边界F0.475440；132图像=0.906751/0.880212/0.934940，TP6120805/FP832978/FN425928，边界F0.393852。原stone屏障冲突和参考纤维冲突计数保存在各candidate条目；没有把冲突改为光纤。

## 3. 关键失败归因

第一，颜色/外边界提案与最终接纳必须分开。70因果几何回退前的当前RGB可达提案包含**5406参考TP和435FP**，最终全部清空；132图像提案84TP/517FP，最终只保留5TP/75FP。提案不是恢复的像素，且自身也未达98%Precision，不能直接并入。

第二，70视频有69帧种子合格，69帧全部触发整帧宽度/曲率回退；33帧主拒绝是宽度，36帧是方向/曲率。32帧的已定向远端分支没有形态异常，但另一端异常导致整帧新增拒绝。这是双端搜索与全帧拒绝耦合的明确证据，并不证明未异常远端一定正确。不能事后删掉另一端失败就宣布成功。132图像22帧至少一个种子合格、21帧几何异常；负例/身份问题仍存在，不能仅以收紧过滤获得“高质量”结论。

第三，固定video_2第1209帧仍没有进入生长。10个侧背景点只有**5个在图内，而配置要求至少6个**，原因insufficient_seed_background；裁切侧的丢失使这一步早于色度判别就失败。该种子没有得到可用背景，因此此例不能归因为暗蓝不相似。当前实现对只可见一侧背景的种子兼容不足，未通过这项场景需求；本轮不临时放宽采样数或挪动点位。原FN4002/低Lab FN2626均不变。

上述结果否定这一固定配置的工程可行性；尚未否定“连续不同明度蓝色属于同一管体”的场景假设。后续应拆开可见侧背景采样和可靠远端搜索这两个结构因素，而不是继续扫蓝色或Gamma阈值。

## 4. 单侧分支、末端与完整视频状态

新候选全video_1在722/722帧执行末端检查，修复旧候选全722帧因没有新增而不检查的问题；video_2有179帧没有唯一合格主体因而未检查。T1只来自原概率且每段/预热清空，静态不使用历史。

video_2显示帧27、30、1521通过单侧弱边局部/历史约束并接纳36、41、567个新增候选像素；三帧均无当前人工语义参考，准确率未知。不能用这一分支被触发、与历史一致或视觉连续来声称真实恢复。其有效prior_age均为0，原种子色度不因接纳而更新。

全video_1新候选的分支状态：unknown1212、visible_end_candidate25、internal_transition11、out_of_view5、occlusion_suspected191；全video_2为unknown3534、out_of_view39、internal_transition4、occlusion_suspected9、无visible_end_candidate。一帧可有两端状态，不是帧数/准确率。

70参考帧中输出3个可见末端候选，均处于整体新增被拒绝的帧；末端单独尝试不代表整帧形态或真实尖端通过。缺少独立尖端像素/位置标注，尖端Recall/Precision、定位误差、圆钝端帽准确性、遮挡恢复与激光落点准确性均保持不可计算。停止生长不等于尖端，黑隙/未知/范围终止不填帽。

## 5. 同轮独立进程完整成本

RTX4070Ti、torch2.11+cu128，原两网FP32/batch1。直接同步总时间包含网络、原图映射、种子/外边界/连通/末端及新臂T1；fps额外包含解码/详细审计/PNG和视频导出。以下不可与历史轮环境负载不同的延迟直接相减当算法加速。

|方案|视频/帧|总P50 ms|总P95 ms|额外P95 ms|含导出fps|有新增候选帧|
|---|---|---:|---:|---:|---:|---:|
|raw2|video_1/722|15.26|18.07|0.00|41.57|0|
|raw2|video_2/1972|15.21|18.09|0.00|42.03|0|
|connected_rgb|video_1/722|23.96|27.15|8.62|31.12|0|
|connected_rgb|video_2/1972|25.50|32.74|11.85|29.06|1380|
|blue_extent|video_1/722|30.70|42.41|16.25|25.02|2|
|blue_extent|video_2/1972|31.69|43.30|18.42|24.64|31|

新候选两片总p95超约33ms源帧预算，额外p95超5ms；即使没有接纳像素，资格/拟合/T1等也消耗时间。旧候选同轮总p95通过源帧预算，但额外阶段与质量仍失败。所有进程的每阶段p50/p95/max、CPU进程时间、Windows工作集及CUDA峰值分配均在summary中；没有额外网络/裁剪推理。

两片没有语义真值的帧只作机制和成本诊断；完整回放不等于完整像素精度测量。新候选只在video_1的2/722、video_2的31/1972帧接纳新增，其余回退，所有实际输出仍为原分割。全片不能宣称提高准确率。

## 6. 分组、参考与复现

|组|帧数|原光纤Dice|新候选虚拟Dice|新候选TP / FP|
|---|---:|---:|---:|---:|
|video_1|43|0.976570|0.976570|0 / 0|
|video_2|27|0.906306|0.906306|0 / 0|
|original29|29|0.969235|0.969235|0 / 0|
|new_user_approved|41|0.929877|0.929877|0 / 0|
|kidney_stones_fiber_v1|68|0.798775|0.798199|5 / 75|
|kidney_stones_fiber_v2|64|0.748768|0.748768|0 / 0|
|old190_exposed25|25|0.643663|0.643663|0 / 0|
|not_matched107|107|0.797731|0.797484|5 / 75|
|latest5|5|0.844614|0.844614|0 / 0|

70=原29+新增41（43/27病例），132=68/64来源及25/107历史曝光；所有原负例/争议图保留，不自动改标签或以缺标排除FP。图像来源标注仍待审核，板中reference不表示均已人工审核。保留的骨架覆盖是结果后诊断、未用于选方案，采用语义形态学骨架/3native px容差，不是物理轴或真实尖端。

45项测试通过（18新机制+27回归），70项独立离线核验通过，包括代码/协议/权重/输入/源码快照身份、原概率和掩膜复现、独立混淆与候选TPFP/低P低Lab、4连通和当前色度、原类别保护、单侧弱边的前帧兼容、全8082帧PNG/日志/直接计时、旧候选逐帧精确复现、六独立进程与状态清空。工程核验仅证明实现与记录一致，不表示分割或尖端通过。

没有读521训练或144保留test像素；stone_new继续退出训练，视频及派生物仅开发验证、不作共享模板标定/伪标签学习，旧混合A0仅原推理，无Godot真实标定。原数据划分、标签、默认、旧源码/结果和未实施T2-P均保留。70/132已参与开发，不是独立最终泛化测试。

## 7. 停止与下一步

本轮按冻结门槛拒绝新候选，不修改实际掩膜、放宽阈值或进入训练/默认部署。下一步优先设计“仅沿已确认远端搜索”的单因素对照，隔离另一端异常导致整帧清空的问题；视野裁切下可见侧背景采样另列独立因素，不把两者同时改后归因于颜色。先登记对照及保护规则，不直接重跑或增加新像素。未解决身份与成本前，不进入结石记忆。

- [数值执行协议](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/docs/plans/fiber_blue_extent_execution_20261101.md>)
- [实现澄清与合成局限](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/docs/plans/fiber_blue_extent_implementation_notes_20261101.md>)
- [冻结配置](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/configs/fiber_blue_extent_control_20261101.json>)
- [主结果与全部拒绝](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_blue_extent_control_20261101/summary.json>)
- [70因果逐帧与分组](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_blue_extent_control_20261101/video_causal/summary.json>)
- [70独立静态诊断](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_blue_extent_control_20261101/video_native/summary.json>)
- [132静态逐帧与分组](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_blue_extent_control_20261101/image_native/summary.json>)
- [逐帧两类别实际/虚拟指标](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_blue_extent_control_20261101/analysis/per_frame_actual_virtual.csv>)
- [亮度分层](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_blue_extent_control_20261101/analysis/brightness_strata.json>)
- [几何回退归因](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_blue_extent_control_20261101/analysis/rejection_attribution.json>)
- [原图骨架诊断](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_blue_extent_control_20261101/analysis/centerline_diagnostic.json>)
- [固定1209背景采样](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_blue_extent_control_20261101/analysis/fixed1209_background_diagnosis.json>)
- [70项独立核验](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_blue_extent_control_20261101/engineering_checks.json>)
- [video_1全部722帧对比](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_blue_extent_control_20261101/video_1_comparison.mp4>)
- [video_2全部1972帧对比](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_blue_extent_control_20261101/video_2_comparison.mp4>)

![1209固定失败：原图/参考/原概率/提案/虚拟/冲突](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_fiber_blue_extent_control_20261101/analysis/video_causal_video2_frame1209.jpg>)
