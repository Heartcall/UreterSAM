# T0光纤时序几何状态对照（2026-10-30）

已完成冻结四臂、202张静态复现和8个独立进程的10776帧完整回放。新门控机制可运行，但存在当前观测被延迟接收的可视化失败，完整几何管线也超源帧预算，不推荐直接采用固定三帧确认作为更优方案。所有分割像素逐帧保持原model1，不把状态变化称为分割准确率改善。默认A0、模型权重、标签/划分与历史结果不变；没有训练、T1搜索或结石S3。

## 固定设计与实际实现

raw2仅两网；current_observation加入当前RGB/FOV/双侧边缘/端帽拟合；legacy_state_only使用原alpha-beta并立即接收越界观测重置；gated_state_only使用新异常暂存/确认规则。三几何臂的当前拟合逐帧完全相同。旧拟合器、旧跟踪器、alpha0.65/beta0.10、尖端0.08D/长度0.10D/方向20°/宽度比[0.5,2]门槛均保持。新入口与状态器独立，不调用visible_fiber/rasterize，不裁剪、填充、连通组件或额外运行网络。

几何状态为入口x/y、轴角、长度、两端宽度、端帽轴比；末端及两侧方向从同一状态推导，不独立平滑。新状态器只在三个连续且相互一致的越界观测后，于第三次当前时刻重定位，前两帧不回改；最多3帧仅预测，第4帧清空活动状态。待确认与活动过期分开，不借重新初始化绕过确认。静态图每张清空，视频在正式帧0前清空预热状态；源/分辨率/异常时间变化清空两分支。

所有状态只是二维图像几何，没有校准协方差/概率置信区间，尖端标记不是物理激光落点。第一观测可能本来就错，连续错误也能通过确认；此轮没有用历史重选组件或重新搜索图像。当前拟合器仍从最大合格入口邻近组件起拟合，其component_provenance是观测选择诊断，组件标签不是实例身份，原始片段没有被删除。

## 完整回放：状态、拟合与失效

| 臂 | 视频 | 帧数 | 当前有效拟合 | 有状态 | 仅预测 | 状态事件 |
| --- | --- | --- | --- | --- | --- | --- |
| raw2 | video_1 | 722 | 0 | 0 | 0 | {"raw_no_geometry": 722} |
| raw2 | video_2 | 1972 | 0 | 0 | 0 | {"raw_no_geometry": 1972} |
| current_observation | video_1 | 722 | 722 | 722 | 0 | {"current_observation": 722} |
| current_observation | video_2 | 1972 | 1276 | 1276 | 0 | {"current_observation": 1276, "no_observation": 696} |
| legacy_state_only | video_1 | 722 | 722 | 722 | 0 | {"initialized": 1, "updated": 721} |
| legacy_state_only | video_2 | 1972 | 1276 | 1377 | 101 | {"initialized": 23, "updated": 1246, "predicted_not_applied": 101, "expired_reset": 22, "no_state": 573, "measurement_jump_reset": 7} |
| gated_state_only | video_1 | 722 | 722 | 722 | 0 | {"initialized": 1, "updated": 721} |
| gated_state_only | video_2 | 1972 | 1276 | 1369 | 107 | {"initialized": 24, "updated": 1234, "predicted_missing": 95, "expired_reset": 21, "no_state": 579, "expired_reset_with_pending": 2, "rejected_jump_pending": 12, "confirmed_relocation": 4, "invalid_prediction_reset": 1} |


video_1拟合拒绝原因：{"image_supported_fit": 722}。

video_1全部门控事件（显示帧）：。


video_2拟合拒绝原因：{"image_supported_fit": 1276, "shaft_not_straight_enough": 261, "width_profile_out_of_family": 420, "nonpositive_or_unresolved_width": 1, "insufficient_two_sided_image_edges": 12, "shaft_too_short": 1, "tip_near_or_outside_view_boundary": 1}。

video_2全部门控事件（显示帧）：311:expired_reset, 349:expired_reset, 409:expired_reset_with_pending, 430:expired_reset, 439:rejected_jump_pending, 440:rejected_jump_pending, 441:confirmed_relocation, 558:expired_reset, 567:expired_reset, 579:expired_reset, 587:expired_reset, 599:expired_reset, 620:rejected_jump_pending, 621:rejected_jump_pending, 622:invalid_prediction_reset, 637:expired_reset, 642:rejected_jump_pending, 643:rejected_jump_pending, 644:confirmed_relocation, 970:expired_reset, 1021:expired_reset, 1099:expired_reset, 1114:expired_reset, 1191:expired_reset, 1373:expired_reset, 1396:expired_reset, 1406:expired_reset, 1501:rejected_jump_pending, 1502:rejected_jump_pending, 1503:confirmed_relocation, 1582:expired_reset, 1688:expired_reset, 1693:rejected_jump_pending, 1706:rejected_jump_pending, 1707:rejected_jump_pending, 1708:rejected_jump_pending, 1709:expired_reset_with_pending, 1710:confirmed_relocation, 1839:expired_reset, 1894:expired_reset。

拒绝不是观测必错，确认也不是运动真实；没有独立几何标注，不能把更少跳动、更多可用状态或更小观测残差当作尖端准确率。当前观测缺失时，预测仍单独标记，不变成可见掩膜。真实快速移动可能带来两帧确认延迟，过期也可能丢失真实轨迹；解析合成轨迹的14项测试证明机制，不能给真实视频事件自动贴“真/假”标签。

**定性失败检查：video_2显示帧439–440，原RGB中的可见蓝色光纤已处于当前拟合所支持的新位置，而新门控黄色预测仍落在旧位置、明显偏离当前光纤；441才确认重定位。** 这支持“固定等待三次会造成可见跟随延迟”的定性结论，不能仅以较小速度p95称更好。显示帧643的最大新旧端点分离也保留在诊断板。没有独立点真值，不把状态到当前拟合的距离称为尖端定位误差，不据此给全部12次暂存判错；这是可视化审核记录，不是新增GT标注。

## 几何运动诊断（非真值误差）

| 视频 | 臂 | 共同可用帧 | 共同有效相邻对 | 派生末端速度p50/p95 px/s | 轴角速度p95 deg/s | 长度速度p95 px/s |
| --- | --- | --- | --- | --- | --- | --- |
| video_1 | current_observation | 722 | 721 | 7.945280/59.543270 | 10.557627 | 59.442403 |
| video_1 | legacy_state_only | 722 | 721 | 5.943438/34.450195 | 6.087669 | 34.315724 |
| video_1 | gated_state_only | 722 | 721 | 5.943438/34.450195 | 6.087669 | 34.315724 |
| video_2 | current_observation | 1274 | 1230 | 37.586731/410.563454 | 43.006069 | 373.042815 |
| video_2 | legacy_state_only | 1274 | 1230 | 30.994408/289.799996 | 34.111897 | 249.822229 |
| video_2 | gated_state_only | 1274 | 1230 | 30.303340/277.852379 | 33.472789 | 247.277979 |

geometry_diagnostics.json同时保留各臂全部可用对、共同可用对、加速度及状态相对当前拟合的残差。条件集合的帧数明确列出，未将无拟合或无状态帧从全体结果剔除。轴角按角差wrap统计；图中角度跨±π的绘制折点不自动判为物理跳变。没有光流补偿或相机运动标定。

## 分割质量不变

| 参考 | 臂 | 类别 | Dice | Precision | Recall | 边界F | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| video_native | 全部四臂 | stone | 0.740216 | 0.899778 | 0.628722 | 0.475440 | 152516 | 16988 | 90065 |
| video_native | 全部四臂 | fiber | 0.945537 | 0.970967 | 0.921404 | 0.862027 | 417783 | 12492 | 35637 |
| image_native | 全部四臂 | stone | 0.906751 | 0.880212 | 0.934940 | 0.393852 | 6120805 | 832978 | 425928 |
| image_native | 全部四臂 | fiber | 0.765758 | 0.643433 | 0.945513 | 0.524869 | 105281 | 58343 | 6067 |

202张静态实际新推理一次，四臂共享这些相同掩膜，不虚报为808次静态前向；原图两专家三类FP32概率、掩膜和汇总均与既有model1逐值一致。所有10776帧逐帧核验旧原图掩膜/RGB，保存不可变原生掩膜引用和哈希。各类原TP删除、新增TP、FP移除/新增均0。原29/新增41、43/27病例、68/64来源、25/107曝光及小结石分组保留；三网classwise质量另存冻结参照。

原29初筛和此前model1相对三网各项不退化失败判断不变；不以相同掩膜通过本轮不变性检查就采用model1或验收S2。85/523个既有光纤-结石冲突未解决。70帧是重复开发验证；42/120随机参考未完成，不外推全片精度；独立尖端/方向真值仍无，tip_metrics为空。

## 独立进程完整性能

| 臂 | 视频 | 管线p50/p95/max ms | 拟合p50/p95 ms | 状态p50/p95 ms | 周期p50/p95 ms | 含视频审计fps | CUDA MiB | 源帧p95预算 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw2 | video_1 | 19.39/27.72/31.44 | 0.00/0.00 | 0.024/0.029 | 22.71/31.39 | 42.07 | 223.92 | True |
| raw2 | video_2 | 19.06/27.63/33.06 | 0.00/0.00 | 0.024/0.028 | 22.29/31.24 | 42.65 | 223.92 | True |
| current_observation | video_1 | 39.97/49.31/55.59 | 17.85/24.22 | 0.067/0.112 | 44.46/54.75 | 22.08 | 223.92 | False |
| current_observation | video_2 | 36.67/46.27/62.16 | 15.92/21.61 | 0.060/0.103 | 41.18/51.11 | 23.78 | 223.92 | False |
| legacy_state_only | video_1 | 37.99/47.89/53.31 | 17.27/23.61 | 0.103/0.183 | 42.41/52.89 | 23.08 | 223.92 | False |
| legacy_state_only | video_2 | 39.19/47.98/58.96 | 17.44/23.41 | 0.097/0.173 | 43.87/53.26 | 22.45 | 223.92 | False |
| gated_state_only | video_1 | 40.66/49.75/55.33 | 18.26/24.08 | 0.124/0.224 | 45.49/55.50 | 21.77 | 223.92 | False |
| gated_state_only | video_2 | 37.90/47.11/57.59 | 16.66/22.39 | 0.119/0.215 | 42.36/52.16 | 23.11 | 223.92 | False |

同轮匹配raw2基线、两张实际U-Net共49,202,790参数、RTX4070Ti、FP32、batch1、12帧临时状态预热后清空。管线包括RGB/PIL、两网、原图掩膜及两份三类概率传CPU、当前FOV/拟合和状态更新/派生。周期另含解码、视频编码、哈希/选中帧概率审计、组件来源报告；编码释放计入离线fps，最终JSON写盘与模型加载在计时之外。组件来源重复计算是诊断开销，位于管线计时外、周期内。只与本轮比较，不相减两个p95或与历史接口混用。峰值是CUDA allocated，不代表设备总占用或CPU内存。

每片时间基准：{"video_1": "decoded_monotonic_pts", "video_2": "frame_index_over_source_fps"}。可靠性预扫描验证单调PTS及RGB身份；不用推理墙钟速度代替视频时间，70稀疏帧也未串成连续序列。p95筛查不是实际采集/显示队列验收，所有失败臂仍完整回放。

## 判断与下一步

新状态器的因果、异常拒绝与三次确认机制已通过测试和完整日志重放；真实几何准确性仍未验证，分割质量没有变化。门控臂本轮管线源帧p95筛查：至少一片未通过。应同时看拟合与状态阶段耗时，不能因为状态向量很小就把整体管线称为实时。

按冻结协议结束T0，不继续改alpha/beta、跳变门槛、确认数或TTL。下一步先另行制定T1候选搜索协议：保留全图当前分割，历史只限定低成本当前图像观测范围，以当前概率/双侧边缘/对比度验证候选；**当前证据充分时允许覆盖历史，不能一律等待三帧；当前证据不足时才更多依赖连续性，必要时区分镜头运动与光纤自身变化。** 无证据保持拒绝，允许可见断片。若当前拟合已主导成本，先在方案中限定轻量观测和拟合预算，不能直接叠加旧重网裁剪或模板相交。T1必须有无历史的匹配对照和同样完整两片测试；未冻结前不实施。没有几何真值时仍只能报告诊断，任何可见像素融合、训练、默认采用或S3都不由本次自动授权。

11项新增状态测试+3项旧几何回归（共14）通过；63项独立核验通过。重新检查202静态与10776完整掩膜记录、8个进程、三臂相同当前观测、旧跟踪器独立重放、新状态器全片与半片前缀重放、阶段计时代数/分位数及10段视频编码/哈希。源/权重/标签/划分/历史结果均冻结。

- [video_1完整五栏对比](video_1_state_comparison.mp4)、[状态诊断板](video_1_state_diagnostic_board.jpg)、[完整参数曲线](video_1_parameter_traces.png)

- [video_2完整五栏对比](video_2_state_comparison.mp4)、[状态诊断板](video_2_state_diagnostic_board.jpg)、[完整参数曲线](video_2_parameter_traces.png)


叠加图中青色为当前观测支持的拟合/滤波状态，黄色为仅历史预测，洋红为待确认端部；所有均是未标定几何。视频状态条占用画面底部一小段显示空间，左侧Input始终保留完整原图，原生分割参考不受影响；诊断板在图外列状态，完整原图可另查引用。
