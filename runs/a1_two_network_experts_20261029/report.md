# 固定A0单光纤成员两网质量与成本对照（2026-10-29）

已完成两个预定义成员的有限对照。model1两网组合同时提高70帧和132图像的光纤Dice与边界F，降低计算成本，但有病例退化、视频Precision下降和图像Recall微降；保留为低成本候选，不直接采用。model2总体光纤质量较差。本轮结束成员搜索，不扫阈值、不加几何或颜色、不自动训练或进入S3。

## 冻结范围与复现

三网classwise为旧190结石网+A0两个光纤成员（73,804,185参数）；两网为同一旧190结石网+A0 model1或model2（49,202,790参数）。两网实际仅构建两张GPU网络，没有隐藏运行未选成员。所有方案PIL RGB256/ImageNet输入、FP32、batch1、原图双线性概率映射align_corners=False、各专家argmax和光纤覆盖顺序一致。没有阈值、组件、颜色、模板或时序修改。成员变化仍会通过光纤优先影响少量结石像素，因此结石并非严格不变。

202张静态×3组均重新推理，原图连续概率与历史各成员缓存逐值相等（最大误差0）；缓存仅作复现核验。三网所有静态汇总与A+原始结果完全一致；全片2694帧三网掩膜与D0完全一致。输入/标签/权重/源码保持身份，源码与冻结配置另存source_snapshot.zip。测试集仍70帧（原29+新增41，病例43/27），本轮未发现新确认标注。132图像按68/64来源、25/107旧190曝光另报。

TRAIN和144保留test像素读取0，训练更新0，stone_new排除不变。两个视频及派生物只作开发验证。42/120随机样本尚未完成，70便利/开发参考不能外推全片精度；无独立尖端真值，尖端Recall/定位及物理落点均不可计算。

## 整体质量

| 参考 | 方案 | 类别 | Dice | Precision | Recall | 边界F(3px) | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 视频70 | classwise3 | stone | 0.740232 | 0.899826 | 0.628722 | 0.475776 | 152516 | 16979 | 90065 |
| 视频70 | classwise3 | fiber | 0.939825 | 0.976050 | 0.906193 | 0.846259 | 410886 | 10082 | 42534 |
| 视频70 | member1_2net | stone | 0.740216 | 0.899778 | 0.628722 | 0.475440 | 152516 | 16988 | 90065 |
| 视频70 | member1_2net | fiber | 0.945537 | 0.970967 | 0.921404 | 0.862027 | 417783 | 12492 | 35637 |
| 视频70 | member2_2net | stone | 0.740256 | 0.899895 | 0.628722 | 0.476063 | 152516 | 16966 | 90065 |
| 视频70 | member2_2net | fiber | 0.931167 | 0.975106 | 0.891017 | 0.824694 | 404005 | 10314 | 49415 |
| 图像132 | classwise3 | stone | 0.906703 | 0.880229 | 0.934820 | 0.393873 | 6120016 | 832740 | 426717 |
| 图像132 | classwise3 | fiber | 0.762334 | 0.638232 | 0.946348 | 0.512720 | 105374 | 59729 | 5974 |
| 图像132 | member1_2net | stone | 0.906751 | 0.880212 | 0.934940 | 0.393852 | 6120805 | 832978 | 425928 |
| 图像132 | member1_2net | fiber | 0.765758 | 0.643433 | 0.945513 | 0.524869 | 105281 | 58343 | 6067 |
| 图像132 | member2_2net | stone | 0.906625 | 0.880265 | 0.934612 | 0.394220 | 6118652 | 832268 | 428081 |
| 图像132 | member2_2net | fiber | 0.745511 | 0.614525 | 0.947462 | 0.488909 | 105498 | 66176 | 5850 |

边界与混淆计数排除未知像素；所有70帧/132图像逐项指标见per_frame_metrics.csv及各组JSON。结石可见面积分组、小目标零TP帧也保留在JSON，不用平均值掩盖漏检。

## 配对像素变化

| 参考 | 方案 | 类别 | 原TP删除 | TP恢复 | FP新增 | FP移除 |
| --- | --- | --- | --- | --- | --- | --- |
| video_native | member1_2net | stone | 0 | 0 | 21 | 12 |
| video_native | member1_2net | fiber | 2205 | 9102 | 3306 | 896 |
| video_native | member2_2net | stone | 0 | 0 | 4 | 17 |
| video_native | member2_2net | fiber | 8800 | 1919 | 2214 | 1982 |
| image_native | member1_2net | stone | 515 | 1304 | 862 | 624 |
| image_native | member1_2net | fiber | 1025 | 932 | 3254 | 4640 |
| image_native | member2_2net | stone | 1855 | 491 | 632 | 1104 |
| image_native | member2_2net | fiber | 573 | 697 | 9164 | 2717 |

model1视频光纤净增6897 TP、2410 FP；图像净少93 TP、少1386 FP。不能把净变化当作“没有删正确像素”：分别删除2205和1025个原TP。图像结石TP变化还受光纤覆盖顺序影响。

## 来源、病例和历史参考分组

| 参考 | 分组 | 数量 | 方案 | 结石Dice | 光纤Dice | 光纤Precision | 光纤Recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| video_native | video_1 | 43 | classwise3 | 0.838336 | 0.978289 | 0.979006 | 0.977573 |
| video_native | video_2 | 27 | classwise3 | 0.681516 | 0.890088 | 0.971881 | 0.820994 |
| video_native | original29 | 29 | classwise3 | 0.810617 | 0.966802 | 0.989034 | 0.945548 |
| video_native | new_user_approved | 41 | classwise3 | 0.694195 | 0.921867 | 0.967186 | 0.880604 |
| video_native | video_1 | 43 | member1_2net | 0.838287 | 0.976570 | 0.976653 | 0.976487 |
| video_native | video_2 | 27 | member1_2net | 0.681516 | 0.906306 | 0.963328 | 0.855658 |
| video_native | original29 | 29 | member1_2net | 0.810572 | 0.969235 | 0.985210 | 0.953771 |
| video_native | new_user_approved | 41 | member1_2net | 0.694195 | 0.929877 | 0.961395 | 0.900359 |
| video_native | video_1 | 43 | member2_2net | 0.838406 | 0.976983 | 0.978761 | 0.975210 |
| video_native | video_2 | 27 | member2_2net | 0.681516 | 0.871023 | 0.969774 | 0.790526 |
| video_native | original29 | 29 | member2_2net | 0.810681 | 0.959110 | 0.988389 | 0.931515 |
| video_native | new_user_approved | 41 | member2_2net | 0.694195 | 0.912545 | 0.966013 | 0.864685 |
| image_native | kidney_stones_fiber_v1 | 68 | classwise3 | 0.892406 | 0.796967 | 0.690174 | 0.942861 |
| image_native | kidney_stones_fiber_v2 | 64 | classwise3 | 0.910363 | 0.744287 | 0.612512 | 0.948306 |
| image_native | old190_exposed25 | 25 | classwise3 | 0.906794 | 0.640724 | 0.490156 | 0.924811 |
| image_native | not_matched107 | 107 | classwise3 | 0.906685 | 0.794550 | 0.682262 | 0.951080 |
| image_native | kidney_stones_fiber_v1 | 68 | member1_2net | 0.892450 | 0.798775 | 0.698753 | 0.932217 |
| image_native | kidney_stones_fiber_v2 | 64 | member1_2net | 0.910413 | 0.748768 | 0.616633 | 0.952974 |
| image_native | old190_exposed25 | 25 | member1_2net | 0.906874 | 0.643663 | 0.496244 | 0.915686 |
| image_native | not_matched107 | 107 | member1_2net | 0.906726 | 0.797731 | 0.686454 | 0.952066 |
| image_native | kidney_stones_fiber_v1 | 68 | member2_2net | 0.892128 | 0.781765 | 0.665275 | 0.947708 |
| image_native | kidney_stones_fiber_v2 | 64 | member2_2net | 0.910334 | 0.726594 | 0.589288 | 0.947324 |
| image_native | old190_exposed25 | 25 | member2_2net | 0.906757 | 0.623323 | 0.468686 | 0.930245 |
| image_native | not_matched107 | 107 | member2_2net | 0.906597 | 0.778288 | 0.658550 | 0.951244 |

model1的主要收益在video_2（0.890088→0.906306），video_1从0.978289降到0.976570。旧29帧光纤Dice为0.966802/0.969235/0.959110（三网/model1/model2）。两候选均通过原29三项初筛；原门槛未改变。另行预冻结的70/132逐类Dice/P/R/边界均不退化检查，两候选均未通过。该检查比旧初筛更严格，不能混称“未通过原29门槛”，也不表示model1没有研究价值。无默认采用。

- member1_2net不退化检查退化项：video70_stone_dice, video70_stone_precision, video70_fiber_precision, video70_stone_boundary_f, image132_stone_precision, image132_fiber_recall, image132_stone_boundary_f。

- member2_2net不退化检查退化项：video70_fiber_dice, video70_fiber_precision, video70_fiber_recall, video70_fiber_boundary_f, image132_stone_dice, image132_stone_recall, image132_fiber_dice, image132_fiber_precision, image132_fiber_boundary_f。


## 中心线补充代理

| 参考 | 方案 | 参考中心线像素 | 覆盖 | 覆盖率 |
| --- | --- | --- | --- | --- |
| video_native | classwise3 | 8551 | 7771 | 0.908783 |
| video_native | member1_2net | 8551 | 7969 | 0.931938 |
| video_native | member2_2net | 8551 | 7533 | 0.880950 |
| image_native | classwise3 | 2941 | 2655 | 0.902754 |
| image_native | member1_2net | 2941 | 2682 | 0.911935 |
| image_native | member2_2net | 2941 | 2648 | 0.900374 |

该指标从全部可见光纤参考组件细化得到，排除未知255的3px邻域和边缘，不连隐藏区域。它是事后补充覆盖代理，既不是解剖尖端真值，也不改变门槛。

## 完整视频性能

RTX4070Ti，torch2.11+cu128，12帧预热；每个方案/视频使用独立新进程，共6进程、8082帧。各组均传回融合掩膜和两个独立专家的原图三类概率，保持相同接口。该接口及运行环境时序与历史A+不同，仅作本轮匹配比较，不与历史延迟相减来归因加速。

| 方案 | 视频 | 实际网络 | 帧数 | 管线P50/P95 ms | 总周期P50/P95 ms | 含导出fps | 峰值CUDA MiB | 改变帧数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classwise3 | video_1 | 3 | 722 | 18.57/21.81 | 22.20/26.43 | 44.12 | 319.82 | 0 |
| classwise3 | video_2 | 3 | 1972 | 18.89/22.97 | 22.40/26.88 | 43.88 | 319.82 | 0 |
| member1_2net | video_1 | 2 | 722 | 14.53/17.70 | 18.16/24.54 | 53.31 | 223.92 | 722 |
| member1_2net | video_2 | 2 | 1972 | 14.59/18.79 | 18.14/23.36 | 53.60 | 223.92 | 1972 |
| member2_2net | video_1 | 2 | 722 | 14.26/17.62 | 17.76/23.45 | 54.67 | 223.92 | 722 |
| member2_2net | video_2 | 2 | 1972 | 14.51/17.72 | 17.89/22.33 | 54.80 | 223.92 | 1972 |

管线包含预处理、全部所需网络、原图映射/融合/CPU传输。总周期另含解码、渲染编码、PNG/日志/审计，离线fps还含编码释放。三组p95均在约33ms源帧预算内，但单帧最大值仍可超预算；未测试实际采集、显示队列、丢帧和持续运行，不宣称实时系统验收。两网CUDA分配峰值约223.92MiB，三网319.82MiB；此为allocated而非设备总占用。模型没有拟合/颜色分支，接受/回退率不适用；每帧直接推理，无隐藏轮廓填补。两个成员均改变2694/2694帧，此仅表示预测变化。

六份timings.json保留全部逐帧分解、最大值。六份frames.jsonl保留RGB/掩膜哈希、面积与有无参考。全片没有稠密真值；未知帧不得以预测变化、面积或视频观感冒充准确率。


### video_1已知片段及输出行为

- 显示帧33–37：classwise3: stone_empty=[]；member1_2net: stone_empty=[]；member2_2net: stone_empty=[]。无参考帧的空掩膜不自动判为真实漏检。

- 显示帧54–72：classwise3: stone_empty=[56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72]；member1_2net: stone_empty=[56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72]；member2_2net: stone_empty=[56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72]。无参考帧的空掩膜不自动判为真实漏检。

- 显示帧77–78：classwise3: stone_empty=[77, 78]；member1_2net: stone_empty=[77, 78]；member2_2net: stone_empty=[77, 78]。无参考帧的空掩膜不自动判为真实漏检。

- 显示帧80–113：classwise3: stone_empty=[80, 81, 82, 83, 84, 85, 86, 87, 88, 92, 93, 94, 95, 96, 101, 102, 104, 105, 106, 107]；member1_2net: stone_empty=[80, 81, 82, 83, 84, 85, 86, 87, 88, 92, 93, 94, 95, 96, 101, 102, 104, 105, 106, 107]；member2_2net: stone_empty=[80, 81, 82, 83, 84, 85, 86, 87, 88, 92, 93, 94, 95, 96, 101, 102, 104, 105, 106, 107]。无参考帧的空掩膜不自动判为真实漏检。

- 显示帧330–722：classwise3: stone_empty=[367, 368, 403]；member1_2net: stone_empty=[367, 368, 403]；member2_2net: stone_empty=[367, 368, 403]。无参考帧的空掩膜不自动判为真实漏检。


[完整四栏视频](video_1_comparison.mp4)，[诊断板](video_1_diagnostic_board.jpg)。诊断板按model1最大TP删除、最大FP新增、最大净TP恢复及最晚已审帧选择；包含失败和改善，不作为选型子集。索引见diagnostic_board_selection.json。


### video_2已知片段及输出行为


[完整四栏视频](video_2_comparison.mp4)，[诊断板](video_2_diagnostic_board.jpg)。诊断板按model1最大TP删除、最大FP新增、最大净TP恢复及最晚已审帧选择；包含失败和改善，不作为选型子集。索引见diagnostic_board_selection.json。


## 结论、停止与下一步

固定两个成员实验到此结束。model2不作为优先底座；model1保留低成本候选，三网保留冻结对照，默认仍原A0。两者都通过旧初筛，但不能按本轮不退化规则直接推荐替换，不继续搜索更多成员或调阈值。S2尚未验收，不自动进入结石时序记忆。

下一步按已有计划开展光纤专用轻量精修可行性，先做只读的可修正区域/错误分布和时间预算分析，重点是model1新增FP、丢失TP及132图像弱项，固定三网作为对照；先判断局部ROI能否同时覆盖错误与保持正确像素。原联合ROI失败仍保留，真值只作上界评价，不能用来生成推理ROI。冻结ROI/融合方案后再决定是否实验；任何训练、专门尖端头或光度增强须另提数据/初始化/预算/验证/回滚计划并批准。12帧尖端标注和120分层样本持续可补，不用预测生成真值。

工程：3项新增测试+3项classwise回归通过；76项独立后验核验通过，重新检查606静态及8082全片掩膜、六个进程、全部视频解码/哈希、源码归档、冻结输入/权重、概率和时分位数。中心线另含3项辅助自检。训练/默认/划分/标签/历史输出均未修改。


复现：`python -m unittest tests.test_member_experts tests.test_classwise_experts`；实验`python -m tools.run_two_network_experts`（已存在结果目录时拒绝覆盖）。检查`preflight.json`、`model_identities.json`、`static_quality.json`、`engineering_checks.json`和`summary.json`。
