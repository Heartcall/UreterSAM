# 指定帧未执行与尖端不规则：只读归因

2026-11-04。按video_2六宫格字幕的一基帧号核对317、770、1204、1716（对应frame_index316、769、1203、1715）。同时核对video_1第317帧以排除编号歧义。读取现有完整日志/原生掩膜，复现5次原始双网前向，raw/结石概率/光纤概率哈希全部与历史相同；无训练、新候选分割或正式输出修改。只新增本诊断文件，历史结果与算法源码保持不变。

**四个video_2帧的共同原因是入口处提前拒绝：3个高置信连通块超过2的上限；尚未检查每块的形状和颜色，就整帧停止。因此不是弱蓝延伸不够激进，也不是端帽拟合失败。尖端不规则另有未应用侧边修整、只加不删、端点证据与梯度计算问题。**

## 1. 四帧实际执行路径

| video_2显示帧 | 主体高P块面积 | 另外两块面积 | 弱蓝新增像素 | 尖端/端帽 |
| --- | --- | --- | --- | --- |
|317|8391|106、39|0|未启动|
|770|5375|1237、625|0|未启动|
|1204|7110|50、16|0|未启动|
|1716|5772|53、69|0|未启动|

四组研究掩膜各自逐像素相同，且研究输出回退原始raw掩膜；blue_body、reachable_blue、cap_fill均为0，eligible=False，recovery与tip_observation均缺失。日志seed原因均为excessive_seed_hypotheses，components=3；外层仅显示no_trusted_seed/control，容易让观察者误以为颜色或端帽已经计算过。

![四帧高置信连通块](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/diagnostics/weak_blue_frame_review_20261104/four_frame_seed_gate.png>)

关键源代码：[AggressiveVariant.seeds](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/tools/fiber_aggressive_blue.py:42>)：以(raw==fiber)且P_fiber≥0.9建立4连通组件，先剔除面积<16，再执行len(ids)>max_components=2的整帧返回；长度、长宽比、颜色、背景及边缘检查全部在这次返回之后。

后续[ConnectedBlue.grow](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/tools/fiber_connected_blue_union.py:62>)接不到唯一seed，直接返回零候选；[外层资格判断](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/tools/fiber_blue_outline.py:128>)令eligible=False；[新扩展入口](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/tools/fiber_weak_blue_evidence.py:91>)跳过expand，自然也不会进入fit_cap。回退时保留raw而非抹掉光纤，因此画面仍有初始分割，但没有暗蓝恢复。

## 2. 已核对：这些块并不构成真正的多个合格主体

使用相同已冻结seed规则，在只读副本中分别检查各块，关闭历史，不做任何区域生长：四帧各自只有主块通过现有资格检查，另外两块全部为short_or_nonelongated_seed。

- 317：两杂块投影长度约9.59、6.85px，均小于20px。
- 770：右侧两块投影长/宽约26.35/46.28、22.06/34.56，不满足长宽比≥1。图像中位于右侧高光/蓝色反光区域。
- 1204：两杂块投影长度约6.33、2.64px。16px小块刚好达到面积计数门槛。
- 1716：两杂块投影长度约7.23、7.73px；其中53px块色度中位数B约0.265，并无主体那样明显的蓝色外观，但代码甚至未运行到颜色验证。

这证明是“计数在有效性判断之前”的逻辑顺序问题；不是没有明显杆体。孤立检查只证明主体资格，不证明解除入口后分割/端帽一定通过，其结果不能冒充新方案准确率。原始component label与隔离后的重编号分开保存。见[逐块资格记录](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/diagnostics/weak_blue_frame_review_20261104/per_component_qualification.json>)。

“唯一物理光纤”“高P像素连通块数量”并不等价：同一光纤可能被阈值打断，也可能有独立误检块。当前入口把两者过早绑定。

video_2共57帧未进入扩展，53帧有该数量拒绝，约占未进入帧的93%。相邻帧表现：316可处理→317跳过→318可处理；769/770跳过→771恢复。这个全有或全无开关会引入额外时序不连续。

317、770、1716的T1 prior_shaft实际可用，但入口仍提前退出；1204的T1先验不可用。T1在本轮新扩展仅作诊断，未用来解除该入口或生成像素。

## 3. 尖端不规则：没有真正把完整轮廓替换为平滑轮廓

**侧边拟合结果没有应用。** [新run函数](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/tools/fiber_weak_blue_evidence.py:92>)把side.run返回的处理掩膜、边缘提案丢弃，只保留info来求方向，并显式sides_applied=False。两段视频新臂side_fill发生像素变化的帧数均为0。分析图画出的侧线是几何观测，不能当成实际侧边已修整。

**最后只能增补，不能消除毛刺。** [扩展](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/tools/fiber_weak_blue_evidence.py:102>)和[端帽应用](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/tools/fiber_weak_blue_evidence.py:143>)都只把背景改成fiber，保留全部union和恢复体。相当于final = union ∪ reachable_blue ∪ cap_fill。端帽也只在当前颜色/梯度证据允许的椭圆内部少量填补，没有把椭圆外的突起裁掉。因此圆滑拟合曲线不会自动成为最终栅格外轮廓。

**末端本身仍由逐像素判别决定。** score≥0.4等条件最终仍生成二值路径，模糊、压缩、亮度/色度扰动会造成锯齿、缺口和细小突出；8邻接增加召回，但没有宽度/轮廓正则。frontier按最大轴向位置选远端带，孤立极值会影响前沿；小团/尖刺检测可拒绝确认尖端，却不会修剪已保留的突出像素。端帽顶又被固定在该前沿，可能导致椭圆RMS/接缝不合格。此处是代码机制，不表示所有不规则像素都已证实为误检。

**端帽有多重前置条件，且接受也可能无修改。** 只有current_RGB_confirmed才调用fit_cap；之后还要求RMS≤1.5px、模型接缝≤15°、RGB弧支持≥80%、弧匹配≥80%。全2694帧326帧接受帽模型，但仅205帧实际改像素，总共498像素；另外121个接受模型为零编辑。1209仅改1个帽像素，不可能重塑整段轮廓。

## 4. 新定位的实现缺陷：在样本列表上计算梯度

[terminal第66行](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/tools/fiber_weak_blue_evidence.py:66>)把2×3个采样RGB重排成6×1图像，再调用evidence。evidence内部执行二维Sobel，计算的是重排列表的邻接梯度，不是原图这些坐标的二维梯度；一列图像还使横向梯度消失，首末样本梯度受边界处理影响。

[fit_cap](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/tools/fiber_weak_blue_evidence.py:134>)对弧线RGB数组也调用evidence(inside[:,None,:])，同样把沿弧排列当成图像空间。暗蓝点需要RGB边缘支撑时，可能被错误判成不支持。该问题与四帧入口拒绝相互独立：那四帧根本未进入此处。

保持RGB、色度score、位置、所有阈值不变，只对样本点检查原图梯度，得到：

| 帧 | 原内部支持 | 使用原图梯度的逐点检查 | 原结果 |
| --- | --- | --- | --- |
|video_2 1209|6/6|6/6|RGB确认、帽编辑1px|
|video_2 1250|4/6|6/6|弱候选、未调用帽|
|video_2 1291|3/6|6/6|弱候选、未调用帽|
|video_1 317|6/6|6/6|RGB确认、帽拟合拒绝|

1250、1291的差异已经具体定位到独立RGB边缘支持项；不是简单把蓝色score门槛调低。原0.67门槛在6个样本中需要至少5个通过，4/6=0.6667仍失败，也放大了这个问题。固定±1px横向取样不随末端宽度调整，亦可能采到轮廓之外，但不能再将1250、1291全部归因于取样位置或颜色变暗。

这只是逐点诊断，没有输出修复后的分割或尖端，也不能证明修复后端帽、准确率或全视频表现通过。[采样数值](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/diagnostics/weak_blue_frame_review_20261104/terminal_sample_probe.json>)、[独立重算检查](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/diagnostics/weak_blue_frame_review_20261104/gradient_probe_checks.json>)均保存。

## 5. video_1第317帧的不同情况

若317指video_1，它确实做了弱蓝扩展：相对union新增311px、轴向前沿约+1.53px；只是两个新方案最终掩膜相同。帽的16个提案均未通过。被记录的所选提案接缝角21.63°>15°，RGB弧支持73.85%<80%，故cap_fill=0，并非入口跳过。见[同帧六宫格](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/diagnostics/weak_blue_frame_review_20261104/video_1_frame0317_six_panel.png>)、[分析图](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/diagnostics/weak_blue_frame_review_20261104/video_1_frame0317_analysis.png>)。

## 6. 建议的修复顺序（尚未实施）

1. 先对组件做现有长度/形状/颜色/入口证据资格检查，再对合格主体判唯一；原始候选计算预算与物理实例数量分开，保留原label。不要仅因出现第三块杂点就拒绝整帧。多合格主体时继续保留歧义回退。
2. 在原生RGB上一次计算梯度图，按真实采样坐标取梯度；颜色评分与空间梯度计算拆开，端点、端帽共用。先补“原图取样与重排样本梯度不同”的回归检查。
3. 核对宽度自适应的内部/外部取样和离散样本判据，明确内部颜色变化、外终止、遮挡、未知；仍不得靠历史或理论交点填补像素。
4. 另定实际轮廓修改协议：区分高可信可见主体与低可信外凸/光晕；明确哪些微小凸出允许修整。若仍保护全部union/恢复像素，则必须接受无法真正消除外凸毛刺的数学限制。

前两项优先，因为是可定位的入口/坐标使用问题。任何后续实际候选仍应先冻结新协议，再完成两段2694帧及70/132参考对照；本轮没有修改算法、门槛、默认、训练或历史输出。

完整只读数据：[diagnosis.json](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/diagnostics/weak_blue_frame_review_20261104/diagnosis.json>)；原图、高P组件、六宫格、尖端取样放大图均位于[诊断目录](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/diagnostics/weak_blue_frame_review_20261104>)。无独立尖端真值，本文不宣称物理定位准确率。
