# 多强度蓝色证据、远端恢复与圆钝帽：完整研究对照

2026-11-03。四组分别完整运行video_1的722帧和video_2的1972帧，共10776正式配对帧；另202静态和96预热，共11074配对前向、22148网络分支调用。**本轮未部署，默认/训练/标签/历史结果均未改变。**

弱蓝恢复在70帧因果参考中相对原union找回9562个参考TP，丢失原TP 0，参考新增FP 16933。低亮度参考FN从6046降到2343。这支持存在可恢复漏分；参考标注可能不完整，新增FP保留为诊断及人工审查，未用Precision/Dice作为本轮否决门槛。

几何组合验收：**未通过**。恢复更多蓝色像素与端帽轮廓改善分别判断；RGB终止确认和投影长度都不是独立物理尖端准确率。当前无独立尖端GT，不能宣称物理端点定位更准。

## 冻结与四组机制

执行前冻结[协议](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/docs/plans/weak_blue_evidence_execution_20261103.md>)、[参数](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/configs/weak_blue_evidence_control_20261103.json>)、[公式与边界修整细化](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/docs/plans/weak_blue_evidence_implementation_20261103.md>)。身份见[identity.json](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/protocols/weak_blue_evidence_control_20261103_v1/identity.json>)；源码/输入/权重与输出均另有快照和哈希。

对照：原union；原bisector；weak_blue_extension；weak_blue_extension_tip。原两组全帧与上一轮逐像素相同，计时为新进程真实双网推理。两新组使用相同的恢复体和尖端观察，只有最后端帽处理不同。两网仍是old190结石+A0 model1光纤、256输入/原图映射、FP32/batch1，49,202,790参数。

新证据把色向、相对蓝优势及局部杆体/背景色差分别归一化后连续加权，暗色按量化噪声可靠性减权。强/弱/不确定独立保存；不是用更低神经概率阈值。主体路径只能经过有当前强/弱蓝证据的8邻接像素，不确定区域不作桥。8邻接可连接对角接触的真实采样像素，没有插入缺失像素。远端有界搜索越过旧B最末6像素；旧B/union/原石头全保护，原始双网输出不变。

当前鲁棒侧拟合的角平分/局部中心线确定方向；80px或对角0.2（较小者）的固定前向预算、宽走廊及面积上限保护，不恢复逐截面转角/宽度整段否决。T1仅提供上一帧原概率状态诊断，未生成像素，视频各自清空；静态无历史。理论交点不输出。

新尖端结合恢复体连续远端团、内外证据下降、RGB或色度外终止对比；分已确认RGB终止、弱候选、未知。黑暗背景、石头接触、触及搜索边界保持未知。小团保留弱候选，前方散在弱蓝不直接等同真实光纤继续。

帽臂用恢复后实际前沿约束16固定椭圆，鲁棒末端侧切线、RMS、模型接缝、RGB弧支持及实际弧匹配共同验收；不退回较短尖端。不删除任何union/恢复体。帽凹口仅允许1.5px边界带内仍有蓝身份及独立RGB梯度的不确定像素，不能用它向前生长。该保护也意味着外凸毛刺不能靠删除消除；拟合成功但像素未变不能算圆钝改善。

76项机制/关联测试通过。工程核验30/30、媒体核验39/39通过。未读取521训练或144保留test组像素，stone_new持续排除，两个视频及派生不参与训练/伪标签/共享模板标定。

## 完整视频与同步对照

六宫格固定为：原始不标记RGB、初始双网、原union、原bisector、新弱蓝扩展、新弱蓝+帽。每格保留原生400×400；标题在图外。纤维青蓝/石头红色半透明，新增轮廓绿色。另保存六格证据分析视频，显示旧eligible/B、证据等级、弱候选、可达体、侧拟合/角平分线/远端团/尖端/帽。洋红实线是接受的帽模型，灰虚线是提案，字幕独立标实际帽编辑数。


| 范围 | 分割六宫格 | 证据/方向/帽分析 | 帧数 |
| --- | --- | --- | --- |
| video_1 | [完整对照](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/video_1_source_timed_six_panel.mp4>) | [完整分析](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/video_1_source_timed_analysis.mp4>) | 722 |
| video_2 | [完整对照](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/video_2_source_timed_six_panel.mp4>) | [完整分析](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/video_2_source_timed_analysis.mp4>) | 1972 |
| video_2 frame1209 | [区间对照](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/video_2_frame1209_six_panel_source_timed.mp4>) | [区间分析](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/video_2_frame1209_analysis_source_timed.mp4>) | 1 |
| video_2 frames1233-1291 | [区间对照](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/video_2_frames1233-1291_six_panel_source_timed.mp4>) | [区间分析](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/video_2_frames1233-1291_analysis_source_timed.mp4>) | 59 |


全60帧无损对照PNG：[截图目录](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/focus_screenshots>)；独立分析图：[分析目录](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/focus_analysis>)。坐标/状态：[video_1 CSV](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/video_1_tip_coordinates.csv>)、[video_2 CSV](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/video_2_tip_coordinates.csv>)；同时保存JSON。所有独立掩膜在full_video/<arm>/<video>/experimental，初始掩膜在full_video/raw2/<video>/masks。每帧所有层/掩膜/RGB/概率哈希、拒绝原因在frames.jsonl。

源video_2为VFR且初始有重复PTS；最终8段媒体逐样本复制源时序，不丢帧、不按平均帧率重采样。保持原冻结T1算法回退时钟另列algorithm_timestamp；导出采用source_media_timestamp。原图面板合成前全帧同源、60张PNG原图区域逐像素核对，编码MP4有损不声称逐字节等同RGB。

## 可见恢复、置信状态和阶段归因


| 范围 | 方案 | 有恢复帧 | 杆体新增px | 远端新增px | 平均延伸px | P95延伸px | 新增P<0.1 | 新增低Lab | RGB确认 | 弱尖端 | 帽接受/改像素 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| video_1 | weak_blue_extension | 722/722 | 223655 | 9266 | 2.397 | 3.687 | 76525 | 28865 | 426 | 39 | 0/0 |
| video_1 | weak_blue_extension_tip | 722/722 | 223655 | 9266 | 2.397 | 3.687 | 76525 | 28865 | 426 | 39 | 53/32 |
| video_2 | weak_blue_extension | 1915/1972 | 962117 | 58595 | 4.241 | 8.243 | 743078 | 519509 | 824 | 1066 | 0/0 |
| video_2 | weak_blue_extension_tip | 1915/1972 | 962117 | 58595 | 4.241 | 8.243 | 743078 | 519509 | 824 | 1066 | 273/173 |
| focus60 | weak_blue_extension | 60/60 | 55080 | 1501 | 3.946 | 6.274 | 44840 | 16335 | 35 | 25 | 0/0 |
| focus60 | weak_blue_extension_tip | 60/60 | 55080 | 1501 | 3.946 | 6.274 | 44840 | 16335 | 35 | 25 | 8/8 |


“杆体/远端”按旧B前沿投影分界，长度是当前掩膜观测，不是真实物理长度；低Lab沿用冻结L<85诊断。蓝色保留率100%是对既有B的保留，不是独立真值Recall。

全部阶段计数/每帧原因：[all_stage_diagnostics.csv](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/analysis/all_stage_diagnostics.csv>)、[all_stage_diagnostics.json](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/analysis/all_stage_diagnostics.json>)，未连接候选、走廊外、原石头阻挡、预算与搜索边界分别记录。色度score图量化0–255，图层sum是量化质量，不能误当像素数量。

video_1尖端原因分布见阶段记录；帽拒绝计数：`{"accepted_current_RGB_cap": 53, "small_cluster_with_terminal_evidence": 21, "ellipse_fit_residual": 41, "insufficient_current_RGB_arc": 114, "model_join_tangent_mismatch": 218, "coherent_front_but_weak_external_boundary": 18, "current_stone_contact": 257}`。

video_2尖端原因分布见阶段记录；帽拒绝计数：`{"model_join_tangent_mismatch": 348, "accepted_current_RGB_cap": 273, "insufficient_current_RGB_arc": 162, "small_cluster_with_terminal_evidence": 45, "coherent_front_but_weak_external_boundary": 1018, "ellipse_fit_residual": 41, "control": 57, "current_stone_contact": 25, "forward_blue_evidence_unresolved": 3}`。

focus60尖端原因分布见阶段记录；帽拒绝计数：`{"accepted_current_RGB_cap": 8, "coherent_front_but_weak_external_boundary": 25, "model_join_tangent_mismatch": 10, "insufficient_current_RGB_arc": 17}`。


| 范围 | 新扩展方向可用帧 | T1先验可用帧（仅诊断） |
| --- | --- | --- |
| video_1 | 722 | 721 |
| video_2 | 1915 | 1718 |
| focus60 | 60 | 0 |


焦点段上一帧T1 shaft不可用，不能把RGB恢复归因于时序先验；本轮T1没有参与像素生成或端点确认，所有恢复来自当前帧。


| 范围 | 新增像素独立RGB梯度支持 | 新增像素平均颜色score | 强蓝新增 | 弱蓝新增 |
| --- | --- | --- | --- | --- |
| video_1 | 92.06% | 0.546 | 35308 | 197613 |
| video_2 | 93.96% | 0.614 | 282133 | 738579 |
| focus60 | 92.18% | 0.781 | 35835 | 20746 |


## 几何测量与冻结门槛

共同侧HF/帽区间沿旧B固定坐标测实际栅格，包含回退帧；各方案接缝都按自身实际帽接点及最终主体范围重测，因此与上一轮按B范围的接缝数值略有不同，不能跨报告直接相减。控制掩膜仍逐像素相同。拟合模型指标与栅格指标分开；8邻接主体允许4连通显示为多个触角接触片段，不能把这个诊断直接视为无证据补线。


| 范围 | 方案 | 侧HF px | RGB边界支持 | 帽转角° | 接缝p95° | B最小保留 | 帽模型/实际改变 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| video_causal | connected_blue_union | 0.315 | 95.58% | 50.605 | 未定义 | 100.00% | 0/0 |
| video_causal | bisector_guided_blue_tip | 0.312 | 95.64% | 49.316 | 34.644 | 100.00% | 40/17 |
| video_causal | weak_blue_extension | 0.358 | 77.56% | 65.257 | 未定义 | 100.00% | 0/0 |
| video_causal | weak_blue_extension_tip | 0.358 | 77.56% | 64.249 | 36.668 | 100.00% | 13/10 |
| image_native | connected_blue_union | 0.626 | 63.83% | 43.114 | 未定义 | 100.00% | 0/0 |
| image_native | bisector_guided_blue_tip | 0.626 | 63.83% | 43.114 | 未定义 | 100.00% | 0/0 |
| image_native | weak_blue_extension | 0.617 | 69.36% | 43.515 | 未定义 | 100.00% | 0/0 |
| image_native | weak_blue_extension_tip | 0.617 | 69.36% | 43.515 | 55.251 | 100.00% | 1/0 |
| full_video_1 | connected_blue_union | 0.271 | 93.38% | 36.068 | 未定义 | 100.00% | 0/0 |
| full_video_1 | bisector_guided_blue_tip | 0.271 | 93.41% | 35.539 | 28.268 | 100.00% | 360/76 |
| full_video_1 | weak_blue_extension | 0.342 | 57.02% | 54.144 | 未定义 | 100.00% | 0/0 |
| full_video_1 | weak_blue_extension_tip | 0.342 | 57.02% | 53.902 | 38.731 | 100.00% | 53/32 |
| full_video_2 | connected_blue_union | 0.378 | 92.04% | 69.556 | 未定义 | 100.00% | 0/0 |
| full_video_2 | bisector_guided_blue_tip | 0.374 | 92.23% | 67.764 | 43.178 | 100.00% | 825/680 |
| full_video_2 | weak_blue_extension | 0.405 | 82.27% | 73.390 | 未定义 | 100.00% | 0/0 |
| full_video_2 | weak_blue_extension_tip | 0.405 | 82.27% | 73.043 | 42.301 | 100.00% | 273/173 |
| focus60 | connected_blue_union | 0.473 | 75.47% | 59.837 | 未定义 | 100.00% | 0/0 |
| focus60 | bisector_guided_blue_tip | 0.472 | 75.53% | 59.407 | 41.948 | 100.00% | 6/6 |
| focus60 | weak_blue_extension | 0.502 | 91.66% | 69.048 | 未定义 | 100.00% | 0/0 |
| focus60 | weak_blue_extension_tip | 0.502 | 91.66% | 68.538 | 44.623 | 100.00% | 8/8 |


门槛未因结果修改：侧HF相对union改善≥15%、实际帽转角改善≥10%、接缝p95≤15°、RGB边界支持下降≤2百分点、实际弧匹配≥80%、可观测帽模型覆盖≥25%（焦点≥50%），union/B保留100%、旧前沿无损失。


| 范围 | 未通过项 |
| --- | --- |
| image_native | side_HF_reduction, actual_rounding, actual_join_p95 |
| video_causal | side_HF_reduction, RGB_nondegrade, cap_model_coverage, actual_rounding, actual_join_p95 |
| full_video_1 | side_HF_reduction, RGB_nondegrade, cap_model_coverage, actual_rounding, actual_join_p95 |
| full_video_2 | side_HF_reduction, RGB_nondegrade, cap_model_coverage, actual_rounding, actual_join_p95 |
| focus60 | focus_cap_model_coverage, actual_rounding, actual_join_p95 |


全部几何逐帧数据：[full_per_frame_geometry.csv](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/analysis/full_per_frame_geometry.csv>)；[冻结验收明细](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/analysis/geometry_quality.json>)。

## 70帧与132图像参考：仅诊断

尖端确认并未随蓝色扩展自动改善：全视频旧角平分线RGB确认1741帧，新方案1250帧；焦点仍为35/60。新前沿变远且更窄，固定中心/两翼取样可落在真正末端外部，导致内部支持不足。第1250帧是4/6支持=0.6667，冻结0.67门槛实际要求至少5/6；本轮不根据结果调低。这是尚未解决的观测设计保守性。候选和完整可见体保留，弱候选没有被冒称确定尖端。


| 帽的独立增量（相对仅扩展） | video_1 | video_2 |
| --- | --- | --- |
| 实际帽转角下降 | 0.45% | 0.47% |
| 端帽像素实际改变帧 | 32 | 173 |


运行时弧匹配在恢复RGB主体加帽上检查；独立验收在包含受保护原网络主体的最终栅格上重新测量。两者分列，保留原凸出像素可使最终接缝仍不理想，不能把求解器验收当成最终轮廓改善。


| 参考 | 方案 | Fiber Dice | Precision | Recall | 边界F | TP | 参考FP | FN | 低Lab FN | 比union +TP/+FP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 70 causal | connected_blue_union | 0.96536 | 96.18% | 96.89% | 0.94335 | 439329 | 17436 | 14091 | 6046 | 0/0 |
| 70 causal | bisector_guided_blue_tip | 0.96543 | 96.17% | 96.92% | 0.94326 | 439433 | 17482 | 13987 | 6015 | 104/46 |
| 70 causal | weak_blue_extension | 0.95847 | 92.89% | 99.00% | 0.92516 | 448880 | 34360 | 4540 | 2345 | 9551/16924 |
| 70 causal | weak_blue_extension_tip | 0.95847 | 92.89% | 99.00% | 0.92516 | 448891 | 34369 | 4529 | 2343 | 9562/16933 |
| 132 image | connected_blue_union | 0.76038 | 63.48% | 94.80% | 0.51318 | 105553 | 60732 | 5795 | 507 | 0/0 |
| 132 image | bisector_guided_blue_tip | 0.76038 | 63.48% | 94.80% | 0.51318 | 105553 | 60732 | 5795 | 507 | 0/0 |
| 132 image | weak_blue_extension | 0.75596 | 62.85% | 94.82% | 0.48795 | 105583 | 62405 | 5765 | 507 | 30/1673 |
| 132 image | weak_blue_extension_tip | 0.75596 | 62.85% | 94.82% | 0.48795 | 105583 | 62405 | 5765 | 507 | 30/1673 |



| 参考 | 方案 | Stone Dice | Precision | Recall | 边界F |
| --- | --- | --- | --- | --- | --- |
| 70 causal | connected_blue_union | 0.74022 | 89.98% | 62.87% | 0.47544 |
| 70 causal | weak_blue_extension_tip | 0.74022 | 89.98% | 62.87% | 0.47544 |
| 132 image | connected_blue_union | 0.90675 | 88.02% | 93.49% | 0.39385 |
| 132 image | weak_blue_extension_tip | 0.90675 | 88.02% | 93.49% | 0.39385 |



| 132图像来源/暴露分组 | 张数 | 方案 | Fiber Dice | Recall | TP | 参考FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| kidney_stones_fiber_v1 | 68 | connected_blue_union | 0.79581 | 93.89% | 37579 | 16838 | 2446 |
| kidney_stones_fiber_v1 | 68 | bisector_guided_blue_tip | 0.79581 | 93.89% | 37579 | 16838 | 2446 |
| kidney_stones_fiber_v1 | 68 | weak_blue_extension | 0.79149 | 93.89% | 37581 | 17357 | 2444 |
| kidney_stones_fiber_v1 | 68 | weak_blue_extension_tip | 0.79149 | 93.89% | 37581 | 17357 | 2444 |
| kidney_stones_fiber_v2 | 64 | connected_blue_union | 0.74211 | 95.30% | 67974 | 43894 | 3349 |
| kidney_stones_fiber_v2 | 64 | bisector_guided_blue_tip | 0.74211 | 95.30% | 67974 | 43894 | 3349 |
| kidney_stones_fiber_v2 | 64 | weak_blue_extension | 0.73766 | 95.34% | 68002 | 45048 | 3321 |
| kidney_stones_fiber_v2 | 64 | weak_blue_extension_tip | 0.73766 | 95.34% | 68002 | 45048 | 3321 |
| old190_exposed25 | 25 | connected_blue_union | 0.64302 | 91.57% | 18365 | 18700 | 1691 |
| old190_exposed25 | 25 | bisector_guided_blue_tip | 0.64302 | 91.57% | 18365 | 18700 | 1691 |
| old190_exposed25 | 25 | weak_blue_extension | 0.63803 | 91.67% | 18386 | 19192 | 1670 |
| old190_exposed25 | 25 | weak_blue_extension_tip | 0.63803 | 91.67% | 18386 | 19192 | 1670 |
| not_matched107 | 107 | connected_blue_union | 0.79078 | 95.50% | 87188 | 42032 | 4104 |
| not_matched107 | 107 | bisector_guided_blue_tip | 0.79078 | 95.50% | 87188 | 42032 | 4104 |
| not_matched107 | 107 | weak_blue_extension | 0.78661 | 95.51% | 87197 | 43213 | 4095 |
| not_matched107 | 107 | weak_blue_extension_tip | 0.78661 | 95.51% | 87197 | 43213 | 4095 |


来源分组与old190暴露分组是两套相交视角，不累加为额外独立图像，也不证明病例独立。132张中只有10张进入可信主体扩展；低覆盖本身限制泛化结论。


| 剩余参考FN归因（新帽臂） | 阶段 | FN | 低Lab FN |
| --- | --- | --- | --- |
| video_causal | original_stone_protection | 85 | 7 |
| video_causal | outside_bounded_search | 281 | 13 |
| video_causal | uncertain_color_not_path | 3212 | 1978 |
| video_causal | RGB_identity_or_local_support_failed | 923 | 317 |
| video_causal | no_evidence_path_to_subject | 28 | 28 |
| video_causal | area_budget_fallback | 0 | 0 |
| video_causal | other_or_original_probabilistic_failure | 0 | 0 |
| image_native | original_stone_protection | 523 | 2 |
| image_native | no_qualified_unique_subject | 4482 | 505 |
| image_native | other_or_original_probabilistic_failure | 0 | 0 |
| image_native | outside_bounded_search | 741 | 0 |
| image_native | uncertain_color_not_path | 0 | 0 |
| image_native | RGB_identity_or_local_support_failed | 5 | 0 |
| image_native | no_evidence_path_to_subject | 0 | 0 |
| image_native | area_budget_fallback | 0 | 0 |
| image_native | side_direction_unavailable | 14 | 0 |


这些是整帧参考剩余漏分的互斥阶段归因，不是尖端专项真值；石头保护、无可信主体、方向缺失、搜索界外、不确定颜色、RGB身份/对比失败、路径断开及预算分别列出。[全部逐帧归因](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/analysis/reference_FN_stage_attribution.json>)。

70逐帧/132逐图语义与低亮度统计：[per_frame_semantic_diagnostics.csv](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/analysis/per_frame_semantic_diagnostics.csv>)。来源分组、old190暴露/非匹配分组、最近新增5帧以及固定1209参考：[reference_subset_diagnostics.json](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/analysis/reference_subset_diagnostics.json>)；所有静态/因果完整原记录在各scope/summary.json，272张参考对照图保留。

70/132均参与开发，不是独立最终泛化测试。1233–1291没有独立语义掩膜，仅做当前RGB/几何观察；其余无参考视频帧也不报告分割准确率。所有参考TP/FP均尊重原标签，未用预测补真值或重标。

## 固定五帧及异常扩张复核


| 显示帧 | 方案 | 状态 | 原因 | RGB坐标 | 延伸px | 新增px | 帽接受 | 帽编辑px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1209 | bisector_guided_blue_tip | current_RGB_confirmed | current_visible_blue_terminal | [164.0, 248.0] | 未定义 | 0 | True | 2 |
| 1209 | weak_blue_extension_tip | current_RGB_confirmed | recovered_visible_external_terminal | [165.0, 245.0] | 2.083 | 840 | True | 1 |
| 1233 | bisector_guided_blue_tip | internal_transition | blue_continues | None | 未定义 | 0 | False | 0 |
| 1233 | weak_blue_extension_tip | weak_candidate | coherent_front_but_weak_external_boundary | None | 4.760 | 920 | False | 0 |
| 1250 | bisector_guided_blue_tip | current_RGB_confirmed | current_visible_blue_terminal | [162.0, 242.0] | 未定义 | 0 | True | 11 |
| 1250 | weak_blue_extension_tip | weak_candidate | coherent_front_but_weak_external_boundary | None | 3.756 | 801 | False | 0 |
| 1266 | bisector_guided_blue_tip | current_RGB_confirmed | current_visible_blue_terminal | [162.0, 244.0] | 未定义 | 0 | True | 15 |
| 1266 | weak_blue_extension_tip | weak_candidate | coherent_front_but_weak_external_boundary | None | 5.476 | 959 | False | 0 |
| 1291 | bisector_guided_blue_tip | internal_transition | blue_continues | None | 未定义 | 0 | False | 0 |
| 1291 | weak_blue_extension_tip | weak_candidate | coherent_front_but_weak_external_boundary | None | 8.932 | 1061 | False | 0 |



| 1209整帧参考 | TP | 参考FP | FN | Dice |
| --- | --- | --- | --- | --- |
| connected_blue_union | 10902 | 366 | 520 | 0.96095 |
| bisector_guided_blue_tip | 10913 | 371 | 509 | 0.96124 |
| weak_blue_extension | 11331 | 777 | 91 | 0.96311 |
| weak_blue_extension_tip | 11332 | 777 | 90 | 0.96315 |


第1209帧：分割对照（左上原图无标记）。

![video2 frame1209](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/focus_screenshots/video2_frame1209.png>)

[同帧证据/角平分/尖端/端帽分析图](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/focus_analysis/video2_frame1209.png>)。

第1233帧：分割对照（左上原图无标记）。

![video2 frame1233](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/focus_screenshots/video2_frame1233.png>)

[同帧证据/角平分/尖端/端帽分析图](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/focus_analysis/video2_frame1233.png>)。

第1250帧：分割对照（左上原图无标记）。

![video2 frame1250](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/focus_screenshots/video2_frame1250.png>)

[同帧证据/角平分/尖端/端帽分析图](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/focus_analysis/video2_frame1250.png>)。

第1266帧：分割对照（左上原图无标记）。

![video2 frame1266](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/focus_screenshots/video2_frame1266.png>)

[同帧证据/角平分/尖端/端帽分析图](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/focus_analysis/video2_frame1266.png>)。

第1291帧：分割对照（左上原图无标记）。

![video2 frame1291](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/focus_screenshots/video2_frame1291.png>)

[同帧证据/角平分/尖端/端帽分析图](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/focus_analysis/video2_frame1291.png>)。

最大增长8帧及全部新增计数：[expansion_review.json](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/analysis/expansion_review.json>)。这些是风险优先复核样本，不自动称为正确光纤或误检。缺少独立真值时，RGB证据通过并不能排除蓝色背景/反光身份混淆；人工可视复核结论另列在下方，不隐藏失败。

可视复核已查看五张固定原生六宫格、3张证据图以及最大增长8帧，未发现这些样本出现明显大块腔壁连入；仍不能认证所有新增像素。第1500帧新增2142像素，扩展进入失焦/过曝的蓝色过渡边缘，有光晕风险；其RGB确认不等于物理端点可靠。第1188帧新增2177像素，前沿不规则且仍为弱候选。其余最大增长多集中319–328相邻帧，是相关场景而不是8个独立病例。[复核范围、图像身份与限制](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/analysis/visual_review.json>)。这不是新增专家标注或用户审核真值。

## 延迟、实际吞吐与回退

独立进程各12次预热后复位、全帧真实两网前向计时。P50/P95包含模型与对应后处理；重型逐层PNG/审计/编码不计核心时延，但全部计入实际研发管线吞吐。线程/GPU环境与峰值内存保存在每个summary。


| 视频 | 方案 | 核心P50 ms | 核心P95 ms | 新增弱蓝P95 ms | 后处理P95 ms | 含导出fps | 源帧预算通过 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| video_1 | connected_blue_union | 68.77 | 87.88 | 未定义 | 41.56 | 10.10 | False |
| video_2 | connected_blue_union | 69.07 | 85.72 | 未定义 | 40.77 | 10.17 | False |
| video_1 | bisector_guided_blue_tip | 165.35 | 231.32 | 未定义 | 154.92 | 5.40 | False |
| video_2 | bisector_guided_blue_tip | 119.72 | 203.56 | 未定义 | 144.83 | 6.31 | False |
| video_1 | weak_blue_extension | 118.46 | 163.10 | 58.08 | 97.54 | 6.54 | False |
| video_2 | weak_blue_extension | 120.02 | 165.50 | 58.99 | 97.52 | 6.48 | False |
| video_1 | weak_blue_extension_tip | 223.54 | 293.78 | 180.86 | 216.34 | 4.38 | False |
| video_2 | weak_blue_extension_tip | 155.68 | 283.16 | 179.86 | 213.21 | 4.82 | False |


颜色/身份/方向不可靠、面积预算失败时回退原union；已恢复体的尖端证据不够则保留分割、输出弱候选或未知；帽失败只保留恢复体。任何历史状态都不填像素。完整结果、拒绝、提案及回退均保留。

## 采用判断与下一步边界

本轮主要结论：连续RGB证据扩展带来可核对的参考漏分恢复；**端帽与总体几何是否达标按上表判断，组合尚未通过几何门槛**。不因参考FP增加自动否决恢复，也不因颜色看起来更完整就宣称真实末端准确率提高。

当前只保留研究候选和完整回放，不修改正式默认。最有价值的后续是对新增弱蓝的身份与外终止边界做重点人工复核，区分合理漏标恢复、背景泄漏、色度光晕和真正遮挡；若需要解除“全部蓝色不可删”才能消除外凸毛刺，必须另订像素保护协议。没有自动启动下一轮、重新训练、参数搜索或结石记忆。

全部原始结果目录：[a1_weak_blue_evidence_20261103](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103>)；工程审计[engineering_checks.json](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/engineering_checks.json>)；媒体核验[media_checks.json](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_weak_blue_evidence_20261103/media_checks.json>)。

回滚：默认分割从未切换；研究调用停用即可继续冻结union/原正式管线，权重和训练清单无需恢复。新核心与运行器分别为[fiber_weak_blue_evidence.py](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/tools/fiber_weak_blue_evidence.py>)、[run_weak_blue_evidence_control.py](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/tools/run_weak_blue_evidence_control.py>)；24项新增机制测试为[test_weak_blue_evidence.py](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/tests/test_weak_blue_evidence.py>)。历史结果只读保留，重复运行入口拒绝覆盖同一目录。
