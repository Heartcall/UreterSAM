# 完整蓝色体的左右拟合与圆钝端帽：完整对照结果


2026-11-02。按用户本轮目标，**参考FP、Precision、Dice只作诊断，不参与采用门槛**；旧实验结论原样保存。主要评价实际栅格轮廓、当前RGB边界证据、可见长度和计算成本。

几何组合验收：**未全部通过**；左右拟合和圆钝端帽分项结果见下表。没有修改默认模型或正式输出，没有训练。全部四臂均完成video_1的722帧、video_2的1972帧，共10776正式配对帧；70帧因果参考和132张静态参考完整保留。

**当前不建议采用组合方案。** 两片实际侧边HF相对未修整蓝色体分别下降19.51%/23.36%，但video_1比原union更粗糙，video_2只有有限改善。焦点60帧端帽仅19/60应用；两片采纳端帽的实际接缝角p95为26.35°/40.06°，超出15°，图像开发集也未通过轮廓/RGB检查。失败归因是实际几何和成本，不是参考Precision。

## 1. 冻结、实现与工程修复


先冻结[原执行协议](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/docs/plans/blue_outline_execution_20261102.md>)及[原参数](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/configs/blue_outline_control_20261102.json>)，再实现四臂。原union逐像素复现；过渡臂blue_body_only单独隔离“释放原网络轮廓约束”的影响，避免把它与平滑效果混在一起。

| 方案 | 研究掩膜行为 |
| --- | --- |
| 原 connected_blue_union | 原正式fiber与完整RGB蓝色体求并，原样对照 |
| 仅蓝色体 blue_body_only | 只将关联可信种子的原fiber组件替换为当前RGB蓝色体B |
| 左右拟合 contour_sides | B的左右边界独立鲁棒直线/低曲率自然样条拟合 |
| 左右拟合＋椭圆帽 contour_sides_cap | 上一臂仅增加显式倾斜椭圆弧端帽及可见性检验 |

共同底座为old190结石＋A0 model1光纤，两网49,202,790参数；256输入、FP32/batch1，概率映射原图后进行研究后处理。每个计时进程都真实执行两网，不复用缓存代替计时。T1只用原概率作前帧位姿提示，每视频及预热后重置，不从新轮廓更新，不填历史像素。

蓝色4连通提取、当前帧颜色/背景模型和宽前向搜索域继承union，不恢复局部截面宽比/转角/曲率清空。初始网络确认身份和原组件关联；几何轴、长度和左右边界取完整B。每侧2像素分箱、12步Huber拟合；样条仅在有显著低频弯曲证据时替代直线。大偏离局部保留B，不整段清空；图框裁切段不被当成真实侧边。

端帽由两侧接点、半宽和中心线切向形成显式椭圆弧，固定16个当前帧形状解，65点采样。要求可见外终止、当前RGB/蓝色外边界支持、接缝切向与保留率。前方仍蓝、石头遮挡、视野裁切或黑暗未知时不强加端帽。几何新增仅允许B附近1.5像素内、满足当前色度及梯度的证据带；不外推超过B的可见远端，不跨无证据间隙。全体B保留至少97%、可见长度最多损失2像素，端帽区保留至少95%。像素随分辨率归一化。

第一轮同样完成了全部10776帧，但独立审计发现video_1显示第666帧的证据域根锚定错误：神经高置信核心中的对角孤立蓝色边缘点被误认为第二个根，使证据域错误置空并回退保留B。没有向未知区域填像素，但该证据记录不合格。第一轮源码、协议、全部输出和失败状态均保留在[v1失败记录](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_blue_outline_20261102/validation_status.json>)。

随后先冻结[工程修复协议](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/docs/plans/blue_outline_evidence_repair_20261102.md>)、[v2配置](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/configs/blue_outline_control_20261102_v2.json>)和[身份](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/protocols/blue_outline_control_20261102_v2/identity.json>)，只将证据根限制为“当前合格B∩可信核心”，**零数值参数、几何判据及评价门槛改变**，重新完成全部四臂双视频。下文均为v2结果，不拼接两轮。

v2完成124项机制/回归测试及45/45项独立审计；核验原概率/掩膜、固定union、同帧输入、T1、证据包含、石头保护、实际保存掩膜重建、真实前向及直接计时。v2共11074次配对前向（10776正式＋202静态＋96预热）/22148网络分支调用；两轮实际合计22148次配对前向/44296网络分支调用。离线几何/绘图/审计无新前向。

## 2. 实际轮廓：两侧平直度、连续性与RGB一致性


下列指标从最终二值栅格体测量，使用各臂共同B轴及杆体/帽区间，包含合格但回退的全部帧。HF是左右边界扣除3像素低频趋势后的平均绝对高频残差，越低越平滑；它不是拟合器自身残差。可见长度是完整B轴投影长度，不是物理长度。RGB支持为实际边界与当前颜色过渡/梯度的位置一致率，是工程代理，不能代替独立真值。

| 范围 | 方案 | 合格/总帧 | 双侧应用 | 左HF px | 右HF px | 平均HF px | RGB支持 | 平均长度px | 最小B保留 | 最大前沿损失px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| video_causal | 原 connected_blue_union | 70/70 | 0 | 0.375 | 0.254 | 0.315 | 95.58% | 142.694 | 100.00% | 0.000 |
| video_causal | 仅蓝色体 blue_body_only | 70/70 | 0 | 0.474 | 0.461 | 0.467 | 92.35% | 140.531 | 100.00% | 0.000 |
| video_causal | 左右拟合 contour_sides | 70/70 | 70 | 0.300 | 0.429 | 0.365 | 92.47% | 140.531 | 97.69% | 0.000 |
| video_causal | 左右拟合＋椭圆帽 contour_sides_cap | 70/70 | 70 | 0.300 | 0.428 | 0.364 | 92.47% | 140.396 | 97.69% | 1.806 |
| image_native | 原 connected_blue_union | 10/132 | 0 | 0.613 | 0.638 | 0.626 | 63.83% | 78.651 | 100.00% | 0.000 |
| image_native | 仅蓝色体 blue_body_only | 10/132 | 0 | 0.839 | 0.727 | 0.783 | 82.78% | 76.542 | 100.00% | 0.000 |
| image_native | 左右拟合 contour_sides | 10/132 | 5 | 0.868 | 0.727 | 0.797 | 79.45% | 76.542 | 97.74% | 0.000 |
| image_native | 左右拟合＋椭圆帽 contour_sides_cap | 10/132 | 5 | 0.868 | 0.727 | 0.797 | 79.45% | 76.542 | 97.74% | 0.000 |
| full_video_1 | 原 connected_blue_union | 722/722 | 0 | 0.295 | 0.248 | 0.271 | 93.38% | 126.371 | 100.00% | 0.000 |
| full_video_1 | 仅蓝色体 blue_body_only | 722/722 | 0 | 0.458 | 0.466 | 0.462 | 93.41% | 123.961 | 100.00% | 0.000 |
| full_video_1 | 左右拟合 contour_sides | 722/722 | 722 | 0.285 | 0.459 | 0.372 | 92.45% | 123.961 | 98.12% | 0.000 |
| full_video_1 | 左右拟合＋椭圆帽 contour_sides_cap | 722/722 | 722 | 0.284 | 0.459 | 0.372 | 92.45% | 123.824 | 97.85% | 1.399 |
| full_video_2 | 原 connected_blue_union | 1915/1972 | 0 | 0.444 | 0.311 | 0.378 | 92.04% | 170.176 | 100.00% | 0.000 |
| full_video_2 | 仅蓝色体 blue_body_only | 1915/1972 | 0 | 0.495 | 0.434 | 0.464 | 88.98% | 168.325 | 100.00% | 0.000 |
| full_video_2 | 左右拟合 contour_sides | 1915/1972 | 1913 | 0.335 | 0.377 | 0.356 | 88.83% | 168.325 | 97.26% | 0.000 |
| full_video_2 | 左右拟合＋椭圆帽 contour_sides_cap | 1915/1972 | 1913 | 0.335 | 0.377 | 0.356 | 88.83% | 168.195 | 97.26% | 2.000 |
| focus60 | 原 connected_blue_union | 60/60 | 0 | 0.541 | 0.405 | 0.473 | 75.47% | 220.005 | 100.00% | 0.000 |
| focus60 | 仅蓝色体 blue_body_only | 60/60 | 0 | 0.547 | 0.407 | 0.477 | 69.08% | 218.663 | 100.00% | 0.000 |
| focus60 | 左右拟合 contour_sides | 60/60 | 60 | 0.397 | 0.373 | 0.385 | 65.77% | 218.663 | 98.18% | 0.000 |
| focus60 | 左右拟合＋椭圆帽 contour_sides_cap | 60/60 | 60 | 0.397 | 0.373 | 0.385 | 65.77% | 218.384 | 98.18% | 1.603 |

| 范围 | 拟合相对B的HF下降 | 相对原union的HF下降 | B中心线HF | 拟合中心线HF | B渐窄违规率 | 最终渐窄违规率 | 可见主体4连通失败 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| video_causal | 22.01% | -15.78% | 0.421 | 0.341 | 12.93% | 13.07% | 0 |
| image_native | -1.81% | -27.38% | 0.560 | 0.565 | 21.15% | 20.87% | 0 |
| full_video_1 | 19.51% | -37.01% | 0.414 | 0.342 | 12.80% | 12.81% | 0 |
| full_video_2 | 23.36% | 5.72% | 0.390 | 0.309 | 14.56% | 14.75% | 0 |
| focus60 | 19.25% | 18.59% | 0.332 | 0.283 | 26.77% | 28.48% | 0 |

渐窄违规率为共同轴向剖面宽度增加超过1像素的比例，天然受可见弯曲和遮挡影响。没有强制单调变窄或无条件拉直；低频弯曲幅度、到直线残差、左右转角p95及模型选择均保存在逐帧结果中。相对B平滑不等于优于原union，两个对照必须同时看。

左右分别采用直线、样条或无有效拟合的帧数见[模型选择计数](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_blue_outline_20261102_v2/analysis/side_model_modes.json>)；自然样条用于保留有证据的低频弯曲，但没有独立外边界真值，不能据此称真实曲率更准确。

## 3. 圆钝端帽：模型与实际像素分开


| 范围 | 端帽应用/独立可观测 | 可观测采用率 | 全部合格采用率 | 侧拟合帽转角p95°* | 最终帽转角p95°* | 接受弧实际匹配 | 实际接缝角p95° |
| --- | --- | --- | --- | --- | --- | --- | --- |
| video_causal | 18/58 | 31.03% | 25.71% | 71.450 | 65.574 | 98.03% | 26.448 |
| image_native | 0/2 | 0.00% | 0.00% | 51.537 | 51.537 | 未定义 | 未定义 |
| full_video_1 | 254/507 | 50.10% | 35.18% | 64.901 | 56.380 | 98.78% | 26.355 |
| full_video_2 | 331/1527 | 21.68% | 17.28% | 74.639 | 71.407 | 95.69% | 40.065 |
| focus60 | 19/59 | 32.20% | 31.67% | 60.476 | 56.379 | 93.28% | 26.596 |

*帽转角指标为每帧实际帽轮廓按固定弧长采样的局部转角p95，再对全部合格帧求均值；不与连续椭圆模型的导数平滑度混用。实际接缝角p95独立测量采纳帧的栅格轮廓，可能显著大于拟合模型的≤15°。观测代理只表示当前可见外终止证据，不代表已有尖端真值。

未采用端帽的原因完整保留：[各范围原因计数](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_blue_outline_20261102_v2/analysis/cap_reasons.json>)。拒绝拟合时保留当前可见蓝色末端；灰色虚线是未应用提案，不能作为已实现的圆头报告。

## 4. 冻结几何门槛的逐项结论


| 范围 | 预先固定检查 | 结果 |
| --- | --- | --- |
| image_native | sides_application | 通过 |
| image_native | raster_side_roughness | 未通过 |
| image_native | RGB_boundary_nondegrade | 未通过 |
| image_native | final_RGB_boundary_nondegrade | 未通过 |
| image_native | cap_application | 未通过 |
| image_native | blue_body_only_visible_retention | 通过 |
| image_native | contour_sides_visible_retention | 通过 |
| image_native | contour_sides_cap_visible_retention | 通过 |
| image_native | actual_cap_rounding | 未通过 |
| image_native | actual_arc_match | 未通过 |
| image_native | actual_join_tangent | 未通过 |
| video_causal | sides_application | 通过 |
| video_causal | raster_side_roughness | 通过 |
| video_causal | RGB_boundary_nondegrade | 通过 |
| video_causal | final_RGB_boundary_nondegrade | 通过 |
| video_causal | cap_application | 通过 |
| video_causal | blue_body_only_visible_retention | 通过 |
| video_causal | contour_sides_visible_retention | 通过 |
| video_causal | contour_sides_cap_visible_retention | 通过 |
| video_causal | actual_cap_rounding | 未通过 |
| video_causal | actual_arc_match | 通过 |
| video_causal | actual_join_tangent | 未通过 |
| full_video_1 | sides_application | 通过 |
| full_video_1 | raster_side_roughness | 通过 |
| full_video_1 | RGB_boundary_nondegrade | 通过 |
| full_video_1 | final_RGB_boundary_nondegrade | 通过 |
| full_video_1 | cap_application | 通过 |
| full_video_1 | blue_body_only_visible_retention | 通过 |
| full_video_1 | contour_sides_visible_retention | 通过 |
| full_video_1 | contour_sides_cap_visible_retention | 通过 |
| full_video_1 | actual_cap_rounding | 通过 |
| full_video_1 | actual_arc_match | 通过 |
| full_video_1 | actual_join_tangent | 未通过 |
| full_video_2 | sides_application | 通过 |
| full_video_2 | raster_side_roughness | 通过 |
| full_video_2 | RGB_boundary_nondegrade | 通过 |
| full_video_2 | final_RGB_boundary_nondegrade | 通过 |
| full_video_2 | cap_application | 未通过 |
| full_video_2 | blue_body_only_visible_retention | 通过 |
| full_video_2 | contour_sides_visible_retention | 通过 |
| full_video_2 | contour_sides_cap_visible_retention | 通过 |
| full_video_2 | actual_cap_rounding | 未通过 |
| full_video_2 | actual_arc_match | 通过 |
| full_video_2 | actual_join_tangent | 未通过 |
| focus60 | focus_cap_application | 未通过 |
| focus60 | blue_body_only_visible_retention | 通过 |
| focus60 | contour_sides_visible_retention | 通过 |
| focus60 | contour_sides_cap_visible_retention | 通过 |
| focus60 | actual_cap_rounding | 未通过 |
| focus60 | actual_arc_match | 通过 |
| focus60 | actual_join_tangent | 未通过 |

这些门槛只涉及实际轮廓、RGB证据、可见性和覆盖率，没有使用参考FP、Precision或Dice否决候选；空分母记未验证。逐帧数据及计算定义：[检查与数值](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_blue_outline_20261102_v2/analysis/geometry_quality.json>)、[全部逐帧几何](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_blue_outline_20261102_v2/analysis/all_geometry_records.json>)、[完整视频几何CSV](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_blue_outline_20261102_v2/analysis/full_per_frame_geometry.csv>)。

## 5. 固定1209及1233–1291：全60帧检查


| 1209方案 | 实际侧HF | RGB支持 | 可见长度px | B保留 | 端帽应用 | 端帽原因 |
| --- | --- | --- | --- | --- | --- | --- |
| 原 connected_blue_union | 0.469 | 79.15% | 217.641 | 100.00% | False | control |
| 仅蓝色体 blue_body_only | 0.468 | 73.04% | 216.243 | 100.00% | False | control |
| 左右拟合 contour_sides | 0.397 | 72.46% | 216.243 | 98.48% | False | cap_not_requested |
| 左右拟合＋椭圆帽 contour_sides_cap | 0.397 | 72.46% | 216.243 | 98.48% | False | cap_visible_pixels_lost |

1209整帧参考诊断（并非尖端精度）：

| 方案 | 参考TP | 参考FP | FN | 低Lab FN | Dice | 边界F |
| --- | --- | --- | --- | --- | --- | --- |
| 原两网 | 7420 | 117 | 4002 | 2626 | 0.782742 | 0.306034 |
| 原 connected_blue_union | 10902 | 366 | 520 | 348 | 0.960952 | 0.821290 |
| 仅蓝色体 blue_body_only | 10726 | 366 | 696 | 377 | 0.952829 | 0.699818 |
| 左右拟合 contour_sides | 10709 | 240 | 713 | 393 | 0.957400 | 0.709431 |
| 左右拟合＋椭圆帽 contour_sides_cap | 10709 | 240 | 713 | 393 | 0.957400 | 0.709431 |

1233–1291全59帧没有像素参考，不能报告该区间分割或尖端准确率。1209有整帧语义参考，缺独立尖端参考。60帧均有原RGB、完整蓝色体、左右拟合线、提案/采纳椭圆帽及最终研究掩膜；按显示帧号1起算，日志frame_index从0起算。

![1209逐阶段对照](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_blue_outline_20261102_v2/analysis/focus8_all60_boards/video2_frame1209.jpg>)

![1250逐阶段对照](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_blue_outline_20261102_v2/analysis/focus8_all60_boards/video2_frame1250.jpg>)

[59帧几何短片](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_blue_outline_20261102_v2/video_2_frames1233-1291_all_stages.mp4>)；[全部60帧逐帧板](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_blue_outline_20261102_v2/analysis/focus8_all60_boards>)；[逐帧曲线/原因/测量](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_blue_outline_20261102_v2/analysis/focus60_geometry_and_curves.json>)。

原图人工目视核对：1209暗蓝前端保留，端帽因可见蓝色像素损失而未采用；1250有显式椭圆弧采纳，但下侧仍有局部毛刺，不能把理想弧等同于整个轮廓都已平直。1209画面上方仍有原网络游离光纤预测，本轮按冻结策略保留未关联原碎片，不能声称消除了全部游离误检。红色结石区域为原预测，本轮未修正其既有错误。

## 6. 参考指标：诊断而非本轮门槛


真值可能漏标；“参考FP”只表示与当前冻结参考不一致，不自动等于组织误检，也不自动等于漏标。全部保存供人工看原图，不改标签、不排除不利样本。70与132都已参与开发，不是独立最终泛化测试。

人工抽看了图像开发集索引68和128（从0起）：前者的蓝色体边缘与参考存在范围分歧，后者原图右侧有蓝色结构而参考未给出光纤区域，不能将其516个新增参考FP直接认定为组织误扩张或已确认的漏标。两例的原图和参考均保留：[索引68](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_blue_outline_20261102_v2/analysis/all_reference_boards/image_native_0068.jpg>)、[索引128](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_blue_outline_20261102_v2/analysis/all_reference_boards/image_native_0128.jpg>)；完整争议排序见[人工检查清单](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_blue_outline_20261102_v2/analysis/manual_reference_disagreement_review.csv>)。图像板上的frame表示图序，只有视频板对应源视频帧号。

| 范围 | 方案 | 光纤Dice | 参考Precision | 参考Recall | 边界F | TP | 参考FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| video_causal | 原两网 | 0.945537 | 0.970967 | 0.921404 | 0.862027 | 417783 | 12492 | 35637 |
| video_causal | 原 connected_blue_union | 0.965362 | 0.961827 | 0.968923 | 0.943348 | 439329 | 17436 | 14091 |
| video_causal | 仅蓝色体 blue_body_only | 0.953883 | 0.969624 | 0.938644 | 0.919573 | 425600 | 13333 | 27820 |
| video_causal | 左右拟合 contour_sides | 0.954458 | 0.976994 | 0.932939 | 0.918605 | 423013 | 9961 | 30407 |
| video_causal | 左右拟合＋椭圆帽 contour_sides_cap | 0.954313 | 0.977018 | 0.932639 | 0.917548 | 422877 | 9947 | 30543 |
| image_native | 原两网 | 0.765758 | 0.643433 | 0.945513 | 0.524869 | 105281 | 58343 | 6067 |
| image_native | 原 connected_blue_union | 0.760378 | 0.634772 | 0.947956 | 0.513183 | 105553 | 60732 | 5795 |
| image_native | 仅蓝色体 blue_body_only | 0.770727 | 0.650953 | 0.944516 | 0.504775 | 105170 | 56393 | 6178 |
| image_native | 左右拟合 contour_sides | 0.771069 | 0.651946 | 0.943457 | 0.505224 | 105052 | 56084 | 6296 |
| image_native | 左右拟合＋椭圆帽 contour_sides_cap | 0.771069 | 0.651946 | 0.943457 | 0.505224 | 105052 | 56084 | 6296 |
| video_native | 原两网 | 0.945537 | 0.970967 | 0.921404 | 0.862027 | 417783 | 12492 | 35637 |
| video_native | 原 connected_blue_union | 0.965362 | 0.961827 | 0.968923 | 0.943348 | 439329 | 17436 | 14091 |
| video_native | 仅蓝色体 blue_body_only | 0.953883 | 0.969624 | 0.938644 | 0.919573 | 425600 | 13333 | 27820 |
| video_native | 左右拟合 contour_sides | 0.954458 | 0.976994 | 0.932939 | 0.918605 | 423013 | 9961 | 30407 |
| video_native | 左右拟合＋椭圆帽 contour_sides_cap | 0.954313 | 0.977018 | 0.932639 | 0.917548 | 422877 | 9947 | 30543 |

结石所有臂逐像素相同：

| 范围 | 结石Dice | Precision | Recall | 边界F | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| video_causal | 0.740216 | 0.899778 | 0.628722 | 0.475440 | 152516 | 16988 | 90065 |
| image_native | 0.906751 | 0.880212 | 0.934940 | 0.393852 | 6120805 | 832978 | 425928 |
| video_native | 0.740216 | 0.899778 | 0.628722 | 0.475440 | 152516 | 16988 | 90065 |

| 范围 | 方案 | 新增参考TP | 删除原参考TP | 新增参考FP | 删除原参考FP | 新增命中低Lab原FN | 删除原fiber总像素 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| video_causal | 原 connected_blue_union | 21546 | 0 | 4944 | 0 | 11419 | 0 |
| video_causal | 仅蓝色体 blue_body_only | 21546 | 13729 | 4944 | 4103 | 11419 | 17832 |
| video_causal | 左右拟合 contour_sides | 20304 | 15074 | 2284 | 4815 | 11038 | 19889 |
| video_causal | 左右拟合＋椭圆帽 contour_sides_cap | 20277 | 15183 | 2269 | 4814 | 11036 | 19997 |
| image_native | 原 connected_blue_union | 272 | 0 | 2389 | 0 | 0 | 0 |
| image_native | 仅蓝色体 blue_body_only | 272 | 383 | 2389 | 4339 | 0 | 4722 |
| image_native | 左右拟合 contour_sides | 269 | 498 | 2153 | 4412 | 0 | 4910 |
| image_native | 左右拟合＋椭圆帽 contour_sides_cap | 269 | 498 | 2153 | 4412 | 0 | 4910 |
| video_native | 原 connected_blue_union | 21546 | 0 | 4944 | 0 | 11419 | 0 |
| video_native | 仅蓝色体 blue_body_only | 21546 | 13729 | 4944 | 4103 | 11419 | 17832 |
| video_native | 左右拟合 contour_sides | 20304 | 15074 | 2284 | 4815 | 11038 | 19889 |
| video_native | 左右拟合＋椭圆帽 contour_sides_cap | 20277 | 15183 | 2269 | 4814 | 11036 | 19997 |

这轮研究掩膜允许用RGB主体替换关联原网络组件，因此会删除原参考TP，不能沿用旧“原fiber全部保护”的说法；正式掩膜仍全部未变。增删分开，低Lab“候选命中”也不等于扣除删除后的净收益。精确相邻阶段影响见[阶段TP/FP增删](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_blue_outline_20261102_v2/analysis/stage_semantic_pairs.json>)；全部两类逐帧参考诊断见[逐帧语义CSV](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_blue_outline_20261102_v2/analysis/per_frame_semantic_diagnostics.csv>)。

| 范围 | 来源/暴露分组 | 张数 | 方案 | Dice | 参考Precision | 参考Recall |
| --- | --- | --- | --- | --- | --- | --- |
| video_causal | video_1 | 43 | 原 connected_blue_union | 0.976260 | 0.971825 | 0.980735 |
| video_causal | video_1 | 43 | 仅蓝色体 blue_body_only | 0.962027 | 0.983209 | 0.941739 |
| video_causal | video_1 | 43 | 左右拟合 contour_sides | 0.961352 | 0.989535 | 0.934731 |
| video_causal | video_1 | 43 | 左右拟合＋椭圆帽 contour_sides_cap | 0.961102 | 0.989559 | 0.934236 |
| video_causal | video_2 | 27 | 原 connected_blue_union | 0.952329 | 0.949847 | 0.954824 |
| video_causal | video_2 | 27 | 仅蓝色体 blue_body_only | 0.944272 | 0.953781 | 0.934950 |
| video_causal | video_2 | 27 | 左右拟合 contour_sides | 0.946324 | 0.962375 | 0.930799 |
| video_causal | video_2 | 27 | 左右拟合＋椭圆帽 contour_sides_cap | 0.946304 | 0.962406 | 0.930732 |
| video_causal | original29 | 29 | 原 connected_blue_union | 0.975586 | 0.969660 | 0.981584 |
| video_causal | original29 | 29 | 仅蓝色体 blue_body_only | 0.960990 | 0.976385 | 0.946074 |
| video_causal | original29 | 29 | 左右拟合 contour_sides | 0.961102 | 0.983361 | 0.939827 |
| video_causal | original29 | 29 | 左右拟合＋椭圆帽 contour_sides_cap | 0.960977 | 0.983431 | 0.939525 |
| video_causal | new_user_approved | 41 | 原 connected_blue_union | 0.958688 | 0.956693 | 0.960690 |
| video_causal | new_user_approved | 41 | 仅蓝色体 blue_body_only | 0.949258 | 0.965222 | 0.933813 |
| video_causal | new_user_approved | 41 | 左右拟合 contour_sides | 0.950136 | 0.972848 | 0.928459 |
| video_causal | new_user_approved | 41 | 左右拟合＋椭圆帽 contour_sides_cap | 0.949977 | 0.972843 | 0.928161 |
| image_native | kidney_stones_fiber_v1 | 68 | 原 connected_blue_union | 0.795811 | 0.690575 | 0.938888 |
| image_native | kidney_stones_fiber_v1 | 68 | 仅蓝色体 blue_body_only | 0.799884 | 0.700015 | 0.932992 |
| image_native | kidney_stones_fiber_v1 | 68 | 左右拟合 contour_sides | 0.800953 | 0.701837 | 0.932667 |
| image_native | kidney_stones_fiber_v1 | 68 | 左右拟合＋椭圆帽 contour_sides_cap | 0.800953 | 0.701837 | 0.932667 |
| image_native | kidney_stones_fiber_v2 | 64 | 原 connected_blue_union | 0.742111 | 0.607627 | 0.953045 |
| image_native | kidney_stones_fiber_v2 | 64 | 仅蓝色体 blue_body_only | 0.755564 | 0.626768 | 0.950984 |
| image_native | kidney_stones_fiber_v2 | 64 | 左右拟合 contour_sides | 0.755531 | 0.627363 | 0.949511 |
| image_native | kidney_stones_fiber_v2 | 64 | 左右拟合＋椭圆帽 contour_sides_cap | 0.755531 | 0.627363 | 0.949511 |
| image_native | old190_exposed25 | 25 | 原 connected_blue_union | 0.643021 | 0.495481 | 0.915686 |
| image_native | old190_exposed25 | 25 | 仅蓝色体 blue_body_only | 0.649142 | 0.504350 | 0.910550 |
| image_native | old190_exposed25 | 25 | 左右拟合 contour_sides | 0.647351 | 0.503674 | 0.905714 |
| image_native | old190_exposed25 | 25 | 左右拟合＋椭圆帽 contour_sides_cap | 0.647351 | 0.503674 | 0.905714 |
| image_native | not_matched107 | 107 | 原 connected_blue_union | 0.790778 | 0.674725 | 0.955045 |
| image_native | not_matched107 | 107 | 仅蓝色体 blue_body_only | 0.802304 | 0.693301 | 0.951978 |
| image_native | not_matched107 | 107 | 左右拟合 contour_sides | 0.803160 | 0.694701 | 0.951748 |
| image_native | not_matched107 | 107 | 左右拟合＋椭圆帽 contour_sides_cap | 0.803160 | 0.694701 | 0.951748 |

最近新增5帧仍单列：

| 方案 | Dice | 参考Precision | 参考Recall | 边界F | TP | 参考FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 原 connected_blue_union | 0.958002 | 0.973449 | 0.943037 | 0.850158 | 39120 | 1067 | 2363 |
| 仅蓝色体 blue_body_only | 0.950794 | 0.977201 | 0.925777 | 0.804861 | 38404 | 896 | 3079 |
| 左右拟合 contour_sides | 0.953208 | 0.986612 | 0.921992 | 0.803829 | 38247 | 519 | 3236 |
| 左右拟合＋椭圆帽 contour_sides_cap | 0.953208 | 0.986612 | 0.921992 | 0.803829 | 38247 | 519 | 3236 |

固定亮度分层FN（OpenCV的Lab_L与HSV_V分别使用0–84、85–169、170–255，不混同两者）：

| 范围 | 方案 | Lab暗FN | Lab中FN | Lab亮FN | V暗FN | V中FN | V亮FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| video_causal | 原两网 | 17465 | 18094 | 78 | 4594 | 22069 | 8974 |
| video_causal | 原 connected_blue_union | 6046 | 7970 | 75 | 2356 | 9094 | 2641 |
| video_causal | 仅蓝色体 blue_body_only | 7308 | 19123 | 1389 | 2447 | 15357 | 10016 |
| video_causal | 左右拟合 contour_sides | 7822 | 21208 | 1377 | 2516 | 16842 | 11049 |
| video_causal | 左右拟合＋椭圆帽 contour_sides_cap | 7830 | 21336 | 1377 | 2517 | 16914 | 11112 |
| image_native | 原两网 | 507 | 1904 | 3656 | 220 | 1557 | 4290 |
| image_native | 原 connected_blue_union | 507 | 1778 | 3510 | 220 | 1453 | 4122 |
| image_native | 仅蓝色体 blue_body_only | 507 | 1983 | 3688 | 220 | 1616 | 4342 |
| image_native | 左右拟合 contour_sides | 507 | 2087 | 3702 | 220 | 1715 | 4361 |
| image_native | 左右拟合＋椭圆帽 contour_sides_cap | 507 | 2087 | 3702 | 220 | 1715 | 4361 |

原29/新增41、最近5、低亮度分层另见[参考子集诊断](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_blue_outline_20261102_v2/analysis/reference_subset_diagnostics.json>)及各scope的brightness_*.json。所有272份静态/因果参考对照板完整保留于[参考可视化目录](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_blue_outline_20261102_v2/analysis/all_reference_boards>)；没有用预测生成尖端真值。

## 7. 八个独立进程的真实成本


| 方案 | 视频 | 帧 | 整管线P50ms | 整管线P95ms | 新增轮廓P50ms | 新增轮廓P95ms | 实际含导出fps | CPU进程秒 | 峰工作集MiB | 峰CUDA MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 原 connected_blue_union | video_1 | 722 | 60.35 | 71.71 | 1.14 | 1.43 | 12.92 | 869.6 | 2129.8 | 223.9 |
| 原 connected_blue_union | video_2 | 1972 | 60.55 | 70.26 | 1.14 | 1.38 | 13.07 | 2352.7 | 2165.9 | 223.9 |
| 仅蓝色体 blue_body_only | video_1 | 722 | 59.36 | 70.35 | 1.10 | 1.39 | 13.32 | 847.6 | 2121.8 | 223.9 |
| 仅蓝色体 blue_body_only | video_2 | 1972 | 61.49 | 72.54 | 1.13 | 1.40 | 12.85 | 2399.2 | 2135.3 | 223.9 |
| 左右拟合 contour_sides | video_1 | 722 | 82.34 | 94.74 | 16.36 | 19.78 | 10.04 | 1124.2 | 2117.6 | 223.9 |
| 左右拟合 contour_sides | video_2 | 1972 | 83.29 | 97.38 | 16.93 | 19.94 | 9.96 | 3099.3 | 2097.2 | 223.9 |
| 左右拟合＋椭圆帽 contour_sides_cap | video_1 | 722 | 91.72 | 105.79 | 27.04 | 32.50 | 9.23 | 1232.2 | 2121.5 | 223.9 |
| 左右拟合＋椭圆帽 contour_sides_cap | video_2 | 1972 | 89.11 | 108.09 | 20.26 | 32.78 | 9.37 | 3296.7 | 2146.7 | 223.9 |

同一4070Ti、两网一次前向、8个独立进程依次运行，各12帧预热后清空状态。整管线为实际测得的网络/原T1/蓝色提取/本轮轮廓等；详细保存层、编码和磁盘I/O另计入实际吞吐。新增轮廓目标P95≤5ms，整管线另比源30fps的33.33ms，二者不得混淆。进程依次运行存在调度/频率差异，不能把不同进程总耗时差直接视为算法加速。

## 8. 结论、边界与下一步


本轮已经实现独立左右直线/低曲率样条和显式椭圆弧，而不是高斯平滑；但它们仍受当前RGB证据和可见像素保留约束。拟合曲线可能平滑，实际掩膜却因证据交集、局部保留和栅格化继续有毛刺或接缝。是否值得采用须以第2–4节真实栅格结果判断；不能只展示理想曲线。

不部署、不改正式默认。建议下一步先只读分解“理想曲线→证据裁切→栅格接缝”的误差，并对固定焦点的真实可见外边界/端帽做最小人工确认，再制定单因素改进协议；不要直接放宽RGB/保留门槛或以几何填充不可见部分。这是待办，尚未执行新候选、训练或结石时序记忆。

全视频没有稠密真值，独立尖端真值仍缺失；不宣称全片分割准确率、尖端定位准确率、遮挡恢复、物理长度或激光落点已验证。训练521和保留test144关联组像素未读取；video_1/video_2及派生数据不训练、不伪标签学习、不共享颜色模板标定，stone_new继续排除。

## 9. 完整产物与复现


[video_1完整722帧对比](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_blue_outline_20261102_v2/video_1_comparison.mp4>)；[video_2完整1972帧对比](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_blue_outline_20261102_v2/video_2_comparison.mp4>)；[原始汇总](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_blue_outline_20261102_v2/summary.json>)；[独立核验](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_blue_outline_20261102_v2/engineering_checks.json>)；[运行源码快照](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_blue_outline_20261102_v2/source_snapshot.zip>)；[最终产物哈希清单](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/runs/a1_blue_outline_20261102_v2/result_identity.json>)。

每帧原输出、蓝色候选、RGB证据带、左右/端帽提案、实际处理体、增删与研究掩膜，以及RGB/概率/源码身份保存在full_video和静态scope中；重跑须新建输出目录，不能覆盖。新实现从实验入口独立调用，回滚只需停用研究入口，默认代码路径未接入。

实现入口：[原拟合器](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/tools/fiber_blue_outline.py>)、[证据锚定修复子类](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/tools/fiber_blue_outline_repaired.py>)、[独立运行器](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/tools/run_blue_outline_repaired_control.py>)；测试：[25项原轮廓机制测试](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/tests/test_fiber_blue_outline.py>)、[2项修复回归](<E:/Study/Wang Shuangyi/Wu Yuanzhao/UreterSAM/tests/test_fiber_blue_outline_repaired.py>)，连同97项既有回归共124项。源快照和协议身份用于重现同一轮次，不直接在已存在输出目录重跑。
