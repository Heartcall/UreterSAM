# 角平分线引导的最远端蓝色团与圆钝端帽：完整对照


2026-11-03。三臂各完整测试video_1的722帧、video_2的1972帧，共8082正式配对帧；202静态共享前向、70同参考因果抽取均完成。**几何组合验收未通过，本轮不部署、不改默认正式分割或训练数据。** 参考FP/Precision/Dice仅诊断，没有用它们否决新方案。

实际改进幅度很小：两片侧HF相对union仅下降0.33%/0.91%，实际帽转角指标下降1.47%/2.58%；实际接缝角p95为28.10°/44.45°。虽然1185帧接受端帽模型，只有756帧发生端帽像素增加；共有960帧研究掩膜有增量、每帧最多37像素，原union无删除。焦点60帧为35个RGB支持观察、6个接受帽且6帧改像素，不能把模型覆盖称为稳定圆头。

## 冻结、机制与边界


执行前冻结[协议](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/docs/plans/bisector_blue_tip_execution_20261103.md>)、[参数](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/configs/bisector_blue_tip_control_20261103.json>)及[身份](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/protocols/bisector_blue_tip_control_20261103_v1/identity.json>)。对照为原union、修复版旧contour_sides_cap、新bisector_guided_blue_tip；旧两臂完整帧输出与上一轮逐像素相同，重新测真实推理时间，未用旧缓存代替计时。

新臂继承完整RGB蓝色4连通主体，不恢复旧局部曲率/宽度整段否决。复用当前B的鲁棒双侧直线/低曲率样条；两侧单位切线和为角平分方向，近似平行或弯曲时改局部中心线，朝入口至画面中心。理论交点没有被当作尖端。最远6像素带内须有唯一、面积/轴向跨度/最末带密度合格的连续蓝色团；单点长刺、无法确认的更远碎片不回退成较短的“好尖端”，而是未知。

候选为远端团边界带的稳健实际蓝色像素；当前RGB检查内部连续蓝色、前方非蓝背景、局部外终止对比，石头接触/出视野/黑暗未知/蓝色继续均拒绝确定点。该确认只代表**当前RGB蓝色外观的可见终止证据**，仍可能把外观转换误当物理末端；白色反光或不同材质不在独立尖端真值下验证，不能宣称端点更准。

以合格蓝色团可见前沿为帽顶轴向约束，不后退、不超过前沿；左右观测末端宽度/切向约束16个固定接点×椭圆形状解，按可行RGB拟合残差选择。原union光纤及B均100%保护；只在当前RGB证据带内增加侧边凹口/端帽像素，不删除原主体或真实候选尖端，不补不可见间隙。代价是外凸毛刺无法通过删除来消除，几何目标可能因此被否定。

全部静态/视频分段初始化与原T1状态按冻结路径运行；历史不生成像素。没有训练、验证视频模板标定或伪标签学习；stone_new继续排除，521训练与144保留test关联组像素未读取。两网49,202,790参数、FP32/batch1、256输入映射原图，权重和输入身份均保留。

134项测试（25项新增、109项既有回归）通过；独立工程审计29/29、媒体审计11/11通过。共8356次配对前向＝8082正式＋202静态＋72预热，16712网络分支调用；离线绘图、审核导出及统计新前向0。

## 六宫格与掩膜、坐标


最终复核视频统一3列×2行：纯原图、初始双网、union、旧cap、新方案、尖端分析。每个面板保持源400×400，不缩放错帧；文字在图外，原图面板合成前逐像素未标记。青蓝/红半透明掩膜保持原图可见；分析显示两侧黄/橙线、白虚线角平分方向、紫色远端团、RGB支持坐标和端帽。洋红实线是被接受模型，灰虚线是未采纳提案。

**模型被接受与真正改变像素单列。** 最终复核字幕标实际cap edits数量；0 edits表示模型吻合但没有端帽像素修改，不能算作轮廓改善。该字幕澄清见[补充](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/docs/plans/bisector_blue_tip_export_caption_20261103.md>)。原运行器导出保留，最终复核另存，不覆盖任何历史掩膜或结果。

| 最终复核视频 | 帧数 | 说明 |
| --- | --- | --- |
| [video_1六宫格完整视频](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/video_1_source_timed_six_panel.mp4>) | 722 | 原生30fps，精确源PTS |
| [video_2六宫格完整视频](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/video_2_source_timed_six_panel.mp4>) | 1972 | 源VFR，平均30.082069744fps |
| [第1209帧单源帧视频](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/video_2_frame1209_source_timed_six_panel.mp4>) | 1 | 原样本时长；另附无损PNG |
| [1233–1291连续区间视频](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/video_2_frames1233-1291_source_timed_six_panel.mp4>) | 59 | 不拼接1209造成假连续 |

坐标和状态：[video_1 CSV](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/video_1_final_detailed_tip_coordinates.csv>)、[video_2 CSV](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/video_2_final_detailed_tip_coordinates.csv>)及对应JSON；同帧初始raw2掩膜在full_video/raw2/<video>/masks，各方案独立研究掩膜在full_video/<arm>/<video>/experimental。每帧保存源RGB/概率/掩膜哈希、PTS、候选/最终/模型帽顶不同坐标及未知原因。新方案观察状态在outline.tip_observation；meta.tips是继承union的旧观察，不混作新尖端。

完整视频、区间和PNG的独立解码/同步检查见[媒体核验](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/media_checks.json>)；全部60张焦点六宫格原生无损PNG见[焦点截图目录](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/focus_screenshots_final_review>)。MP4采用有损编码，不能要求编码后RGB逐字节相同；合成前所有帧和无损PNG原图区域已核对同源哈希。

video_2源媒体存在可变帧时长及初始重复PTS。最终source_timed视频逐样本复制源时间表，不按平均帧率均匀排布，保留重复PTS和1972帧；未重新编码任何画面，样本payload逐个核对。坐标同时记录源媒体timestamp和冻结算法inference_timestamp：算法本轮仍用已冻结的OpenCV回退时钟，未更换或重算T1。详见[源时序工程修复](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/docs/plans/bisector_blue_tip_source_timing_20261103.md>)、[逐样本时序/码流核验](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/source_timed_media_identity.json>)。源异常不被伪装成严格CFR。

## 实际栅格几何、保留与应用覆盖


以下侧HF、RGB支持及帽转角在同一B轴/旧帽区间测量，包含回退帧，不能拿模型曲线光滑冒充实际掩膜改善。接缝另按各模型实际接点与轴计算，避免把新接点错用成旧接点。B保留不是独立真实像素Recall，仍存在颜色误认可能。帽转角是逐帧实际轮廓局部转角p95的均值；接缝列为被接受模型帧的实际栅格接缝角p95，含像素量化影响。

| 范围 | 方案 | 合格/总 | 侧HF px | RGB支持 | 实际帽转角° | 实际接缝p95° | 最小B保留 | 最大前沿损失px | 接受帽模型 | 帽像素改变帧 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| video_causal | connected_blue_union | 70/70 | 0.315 | 95.58% | 50.605 | 未定义 | 100.00% | 0.000 | 0 | 0 |
| video_causal | contour_sides_cap | 70/70 | 0.364 | 92.47% | 65.574 | 26.448 | 97.69% | 1.806 | 18 | 0 |
| video_causal | bisector_guided_blue_tip | 70/70 | 0.312 | 95.64% | 49.316 | 34.712 | 100.00% | 0.000 | 40 | 17 |
| image_native | connected_blue_union | 10/132 | 0.626 | 63.83% | 43.114 | 未定义 | 100.00% | 0.000 | 0 | 0 |
| image_native | contour_sides_cap | 10/132 | 0.797 | 79.45% | 51.537 | 未定义 | 97.74% | 0.000 | 0 | 0 |
| image_native | bisector_guided_blue_tip | 10/132 | 0.626 | 63.83% | 43.114 | 未定义 | 100.00% | 0.000 | 0 | 0 |
| full_video_1 | connected_blue_union | 722/722 | 0.271 | 93.38% | 36.068 | 未定义 | 100.00% | 0.000 | 0 | 0 |
| full_video_1 | contour_sides_cap | 722/722 | 0.372 | 92.45% | 56.380 | 26.355 | 97.85% | 1.399 | 254 | 0 |
| full_video_1 | bisector_guided_blue_tip | 722/722 | 0.271 | 93.41% | 35.539 | 28.101 | 100.00% | 0.000 | 360 | 76 |
| full_video_2 | connected_blue_union | 1915/1972 | 0.378 | 92.04% | 69.556 | 未定义 | 100.00% | 0.000 | 0 | 0 |
| full_video_2 | contour_sides_cap | 1915/1972 | 0.356 | 88.83% | 71.407 | 40.065 | 97.26% | 2.000 | 331 | 0 |
| full_video_2 | bisector_guided_blue_tip | 1915/1972 | 0.374 | 92.23% | 67.764 | 44.449 | 100.00% | 0.000 | 825 | 680 |
| focus60 | connected_blue_union | 60/60 | 0.473 | 75.47% | 59.837 | 未定义 | 100.00% | 0.000 | 0 | 0 |
| focus60 | contour_sides_cap | 60/60 | 0.385 | 65.77% | 56.379 | 26.596 | 98.18% | 1.603 | 19 | 0 |
| focus60 | bisector_guided_blue_tip | 60/60 | 0.472 | 75.53% | 59.407 | 40.677 | 100.00% | 0.000 | 6 | 6 |

旧cap分支没有新增cap_fill图层含义，旧分支该“帽像素改变帧”列0不表示旧后处理没有修改：它的完整差异由保存的原/研究掩膜和语义审计体现。该列用于新臂把接受模型与真正修改端帽分开；不据此比较旧分支修改频率。新臂完整视频实际改动数见工程审计counts。

冻结检查：

| 范围 | 未通过项 |
| --- | --- |
| image_native | side_HF_reduction、cap_model_coverage、actual_rounding、actual_join_p95、actual_arc_match |
| video_causal | side_HF_reduction、actual_rounding、actual_join_p95 |
| full_video_1 | side_HF_reduction、actual_rounding、actual_join_p95 |
| full_video_2 | side_HF_reduction、actual_rounding、actual_join_p95 |
| focus60 | focus_cap_model_coverage、actual_rounding、actual_join_p95 |

阈值：相对union侧HF至少改善15%、帽转角至少10%，RGB支持下降不超过2个百分点，实际接缝p95≤15°、弧匹配≥80%，可观测帽模型覆盖≥25%、焦点60帧≥50%；union/B保留100%，可见蓝色前沿无损失。未因结果调整门槛；空分母未验证。全部逐帧数值见[全视频几何CSV](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/analysis/full_per_frame_geometry.csv>)、[全部几何记录](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/analysis/all_geometry_records.json>)、[验收明细](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/analysis/geometry_quality.json>)。

## 固定五帧：高清原生截图及状态


| 显示帧 | RGB观察状态 | 当前支持坐标 | 帽模型接受 | 帽修改像素 | 原因 | 侧HF | RGB支持 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1209 | current_RGB_confirmed | [164.0, 248.0] | True | 2 | current_cluster_anchored_ellipse | 0.457 | 0.807 |
| 1233 | internal_transition | None | False | 0 | blue_continues | 0.452 | 0.772 |
| 1250 | current_RGB_confirmed | [162.0, 242.0] | True | 11 | current_cluster_anchored_ellipse | 0.465 | 0.815 |
| 1266 | current_RGB_confirmed | [162.0, 244.0] | True | 15 | current_cluster_anchored_ellipse | 0.479 | 0.750 |
| 1291 | internal_transition | None | False | 0 | blue_continues | 0.470 | 0.869 |

显示第1209帧：

![第1209帧原生六宫格](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/focus_screenshots_final_review/video2_frame1209.png>)

显示第1233帧：

![第1233帧原生六宫格](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/focus_screenshots_final_review/video2_frame1233.png>)

显示第1250帧：

![第1250帧原生六宫格](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/focus_screenshots_final_review/video2_frame1250.png>)

显示第1266帧：

![第1266帧原生六宫格](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/focus_screenshots_final_review/video2_frame1266.png>)

显示第1291帧：

![第1291帧原生六宫格](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/focus_screenshots_final_review/video2_frame1291.png>)

人工目视核对：1209/1250/1266显示被接受的椭圆弧，但实际帽只分别增加2/11/15像素，原侧边局部毛刺仍存在；1233/1291显示灰色候选并标内部蓝色继续，未强制确定尖端。最大扩张video_2第377帧共37像素（其中帽14像素），发生在边界带；原图有高光，不能凭颜色断定所有新增像素都正确，已保留原图和掩膜待审，没有大面积填充。

1209有整帧语义参考；1233–1291全59帧没有像素参考，均只作形态/RGB诊断。五图1200×960由六张400×400原生画面构成，不做虚构超分。坐标为原400×400画面，不是六宫格坐标。

## 尖端位移与未知


| 视频 | 方案 | 相邻有效对数 | 位移P50 px | 位移P95 px | 最大位移px |
| --- | --- | --- | --- | --- | --- |
| video_1 | connected_blue_union | 0 | 未定义 | 未定义 | 未定义 |
| video_1 | contour_sides_cap | 165 | 0.107 | 0.867 | 1.553 |
| video_1 | bisector_guided_blue_tip | 476 | 0.000 | 1.414 | 3.606 |
| video_2 | connected_blue_union | 0 | 未定义 | 未定义 | 未定义 |
| video_2 | contour_sides_cap | 135 | 0.502 | 2.372 | 16.984 |
| video_2 | bisector_guided_blue_tip | 929 | 1.000 | 4.472 | 40.447 |

只统计相邻帧双方均有候选坐标的位移，不跨未知区插值；旧cap坐标仅接受模型帧提供，新臂是RGB支持观察。覆盖率/定义不同，不能据更小位移宣称定位更准或时序跟踪更好。遮挡、出视野及未知继续不发布最终点，所有状态完整保留。

## 参考语义诊断与误扩张


| 范围 | 方案 | 光纤Dice | 参考Precision | 参考Recall | 边界F | TP | 参考FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| video_causal | raw2 | 0.945537 | 0.970967 | 0.921404 | 0.862027 | 417783 | 12492 | 35637 |
| video_causal | connected_blue_union | 0.965362 | 0.961827 | 0.968923 | 0.943348 | 439329 | 17436 | 14091 |
| video_causal | contour_sides_cap | 0.954313 | 0.977018 | 0.932639 | 0.917548 | 422877 | 9947 | 30543 |
| video_causal | bisector_guided_blue_tip | 0.965431 | 0.961739 | 0.969152 | 0.943261 | 439433 | 17482 | 13987 |
| image_native | raw2 | 0.765758 | 0.643433 | 0.945513 | 0.524869 | 105281 | 58343 | 6067 |
| image_native | connected_blue_union | 0.760378 | 0.634772 | 0.947956 | 0.513183 | 105553 | 60732 | 5795 |
| image_native | contour_sides_cap | 0.771069 | 0.651946 | 0.943457 | 0.505224 | 105052 | 56084 | 6296 |
| image_native | bisector_guided_blue_tip | 0.760378 | 0.634772 | 0.947956 | 0.513183 | 105553 | 60732 | 5795 |
| video_native | raw2 | 0.945537 | 0.970967 | 0.921404 | 0.862027 | 417783 | 12492 | 35637 |
| video_native | connected_blue_union | 0.965362 | 0.961827 | 0.968923 | 0.943348 | 439329 | 17436 | 14091 |
| video_native | contour_sides_cap | 0.954313 | 0.977018 | 0.932639 | 0.917548 | 422877 | 9947 | 30543 |
| video_native | bisector_guided_blue_tip | 0.965431 | 0.961739 | 0.969152 | 0.943261 | 439433 | 17482 | 13987 |

三臂结石结果完全不变：

| 范围 | 结石Dice | Precision | Recall | 边界F | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| video_causal | 0.740216 | 0.899778 | 0.628722 | 0.475440 | 152516 | 16988 | 90065 |
| image_native | 0.906751 | 0.880212 | 0.934940 | 0.393852 | 6120805 | 832978 | 425928 |
| video_native | 0.740216 | 0.899778 | 0.628722 | 0.475440 | 152516 | 16988 | 90065 |

TP/FP/FN逐帧、原正确像素删除和两类指标见[语义CSV](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/analysis/per_frame_semantic_diagnostics.csv>)；原29/新增41、两病例、图像v1/v2与暴露来源分组保存在各scope/summary.json/groups，低亮HSV_V及Lab_L分层保存在brightness_*.json。所有272份参考图板保留于[参考对照目录](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/analysis/all_reference_boards>)。这些均是已参与开发的参考，不是独立最终泛化测试，不据此推断2694帧准确率。

原union可能已存在腔壁或游离光纤误认，新臂保护策略不会自动修正它们；新增RGB凹口/帽像素同样可能误扩张。参考缺标争议不能自动当作成功，也不能自动当作真实误检。所有分歧和明显扩张供人工原图审查，不隐藏，不重标，也不按参考FP否决几何。

本轮每帧最大新增37像素，原union无删减；新增最多8帧的原图六宫格已保留在[扩张审查图板](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/analysis/expansion_review>)，全部2694帧增量见[扩张记录](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/analysis/expansion_review.json>)。没有独立真值的新增区域不自动视为正确；未见大面积增长不代表局部扩张已排除。

## 真实性能


| 方案 | 视频 | 管线P50ms | 管线P95ms | 新阶段P50ms | 新阶段P95ms | 含独立审计导出fps | CPU进程秒 | 峰CUDA MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| connected_blue_union | video_1 | 63.81 | 78.13 | 0.00 | 0.00 | 11.74 | 933.8 | 223.9 |
| connected_blue_union | video_2 | 64.89 | 78.75 | 0.00 | 0.00 | 11.64 | 2576.8 | 223.9 |
| contour_sides_cap | video_1 | 95.85 | 123.72 | 0.00 | 0.00 | 8.45 | 1304.1 | 223.9 |
| contour_sides_cap | video_2 | 93.26 | 117.81 | 0.00 | 0.00 | 8.72 | 3476.9 | 223.9 |
| bisector_guided_blue_tip | video_1 | 165.77 | 226.61 | 95.16 | 124.41 | 5.71 | 1859.8 | 223.9 |
| bisector_guided_blue_tip | video_2 | 114.10 | 197.87 | 36.80 | 113.57 | 6.81 | 4366.2 | 223.9 |

六独立进程顺序运行，RTX4070Ti，两网真实前向，预热后清空T1；统计/最终六宫格在全部计时结束后运行。5ms新阶段目标与源帧间隔整管线目标（video_1平均33.33ms、video_2平均33.24ms）分列；网络数不增加，新增拟合/RGB检验在CPU，不能将独立进程调度差当算法加速。实际吞吐含独立掩膜/逐帧日志和私有回放导出，最终六宫格离线导出另计，不冒充实时。

## 结论与复现


**本轮尚不值得替换union。** 保留完整蓝色主体的约束得到验证，但接受方向/端点或椭圆模型不等于实际轮廓已平滑、圆钝。上表不通过项是停止依据；不会为了好看裁短蓝色尖端或放宽门槛。 缺独立尖端真值，端点准确率=null；全视频无稠密真值，全片准确率=null。也没有验证物理尖端、毫米尺寸、遮挡恢复或激光落点。

下一步建议仅只读分解：远端团为什么未知、椭圆未通过何种RGB/接缝约束，以及100%保护的原边界对可修整程度的限制；如需允许修剪或其他端帽族，须另冻结单因素协议。当前没有自动继续新候选、训练、默认替换或结石时序记忆。

主要实现：[新候选](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/tools/fiber_bisector_blue_tip.py>)、[独立运行器](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/tools/run_bisector_blue_tip_control.py>)、[机制测试](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/tests/test_bisector_blue_tip.py>)。运行源码/权重/输入/配置/输出身份保存在[源码快照](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/source_snapshot.zip>)、[预检](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/preflight.json>)和[结果哈希清单](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/result_identity.json>)；已有输出目录禁止直接覆盖。回滚仅停用研究入口，正式默认未接入。

最初运行器导出亦完整保留（复核字幕之前的存档版本）：[video1存档](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/video_1_six_panel.mp4>)、[video2存档](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/video_2_six_panel.mp4>)、[1209存档](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/video_2_frame1209_six_panel.mp4>)、[59帧存档](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/video_2_frames1233-1291_six_panel.mp4>)。用于最终阅读的是上文review版本。

首次复核版本亦保留（重复坐标文字修复及源时序修正前）：[video1初稿](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/video_1_review_six_panel.mp4>)、[video2初稿](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/video_2_review_six_panel.mp4>)、[1209初稿](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/video_2_frame1209_review_six_panel.mp4>)、[59帧初稿](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/video_2_frames1233-1291_review_six_panel.mp4>)。render_failures保留字幕与CFR假设的工程诊断；未改变任何算法输出。

坐标文字清理后、精确源时序修正前的编码中间稿保存在[中间编码视频路径清单](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_bisector_blue_tip_20261103/final_review_export_summary.json>)；最终交付只使用上文source_timed四个视频。
