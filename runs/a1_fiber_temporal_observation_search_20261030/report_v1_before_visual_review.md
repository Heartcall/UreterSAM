# T1轻量当前观测与历史搜索完整对照

2026-10-30。**四臂、202张静态复现及8独立进程10776帧完整回放已完成。轻量观测和两种状态方案均通过本轮耗时门槛；历史搜索增加了有状态覆盖，但没有独立几何真值，不能据此认定更准。所有分割像素不变，没有分割准确率提升、训练或默认替换。**

## 设计、数据与身份

原model1两网49,202,790参数，RGB/PIL256、ImageNet归一化、FP32/batch1、原图双线性概率映射和光纤优先保持。raw2不观测；light_current每帧最多2轴候选；evidence_state加入强当前优先的部分状态；history_search只在没有独立强当前胜者时增加最多1个历史候选。当前提案160网格、概率0.2、组件最少4点；8截面×每侧17位置，3×25端部采样，两次鲁棒直线拟合，无旧全图Sobel/密集端帽、裁剪重网、光流、新颜色模板或额外网络。

候选仅元数据；没有最大组件删除、模板相交、隐藏像素填补或跨视频身份继承。RGB差只验证边界，不是蓝色身份硬门控；浅白端部可提供证据。可见前沿可能只是遮挡边界，不能当真实圆钝尖端。无端部时只输出杆体，端部组最多预测3帧；无向杆体不初始化入口组，用历史消歧只能算弱观测。

强当前证据先于历史门控，明显越界本帧重获、清零相关速度，没有T0固定三帧等待。兼容强/弱观测沿用alpha0.65/beta0.10；弱观测只更新已有兼容组。历史搜索用同一当前RGB/网络检验，永不升级独立强证据；没有支持则拒绝/过期，全图当前提案始终运行。六参数联合图像状态没有端帽半径、真实设备标定或校准置信区间。

70视频参考、132图像及原29/新增41、43/27病例、68/64来源、25/107曝光分组冻结；70是重复开发验证。42/120随机抽样未完成，不能外推全片分割精度；12帧尖端专项未补，无尖端Recall、物理尖端误差或落点准确率。视频与派生物均不训练，stone_new排除，521训练及144保留test像素读取0。冻结配置中的“unlabelled/zero-seed static445”措辞在preflight澄清：该样本有语义参考，只是网络无种子，不是没有真值。未改配置或任何阈值。

## 完整视频性能（同轮匹配）

| 臂 | 视频 | 总管线p50/p95/max ms | 几何p50/p95 ms | 含导出周期p50/p95 ms | 含导出fps | 进程CPU秒 | CUDA MiB | 源帧预算 | 几何≤5ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw2 | video_1 | 13.68/15.66/27.99 | 0.01/0.01 | 16.55/19.57 | 59.39 | 193.52 | 223.92 | True | True |
| raw2 | video_2 | 14.03/15.27/26.49 | 0.01/0.01 | 16.93/18.69 | 59.02 | 531.97 | 223.92 | True | True |
| light_current | video_1 | 17.20/18.59/23.89 | 3.54/3.83 | 20.07/22.26 | 49.47 | 234.25 | 223.92 | True | True |
| light_current | video_2 | 17.62/19.31/91.59 | 3.62/4.61 | 20.47/22.50 | 48.64 | 652.88 | 223.92 | True | True |
| evidence_state | video_1 | 17.44/19.06/28.73 | 3.68/4.02 | 20.31/23.00 | 48.77 | 238.19 | 223.92 | True | True |
| evidence_state | video_2 | 17.46/19.40/92.38 | 3.72/4.70 | 20.05/22.32 | 49.55 | 640.97 | 223.92 | True | True |
| history_search | video_1 | 17.21/19.48/22.83 | 3.67/4.13 | 19.86/22.77 | 49.76 | 231.94 | 223.92 | True | True |
| history_search | video_2 | 17.42/19.25/93.83 | 3.74/4.70 | 20.07/22.09 | 49.59 | 640.36 | 223.92 | True | True |

RTX4070Ti，两片各臂独立进程；各12帧临时状态预热后清空，共96次预热前向不计入10776正式帧。另202次静态前向，共11074次网络调用，不能将四臂共享静态掩膜冒充808次前向。无额外神经网络/裁剪，几何主要为CPU。进程CPU秒含解码/编码与审计，不是几何纯CPU占用；CUDA峰值是allocated，不是设备总内存。

总管线直接计时包含两网、原图掩膜及两份三类概率传CPU、全部观测/状态/公开元数据转换；几何内层计时含grid/FOV、采样/拟合、质量和状态，不含返回时的公开元数据转换，差额仍在总管线中。逐帧全部分阶段分布在timings.json；不相加或相减分位数，不用历史T0推算加速。周期/fps还含解码、编码、哈希、选中帧概率核验；最终大JSON写盘、模型加载在计时外，后续重新渲染的说明视频也不计入原性能。通过p95门槛不等于已验证实时相机采集/显示队列。

源时间均预扫描验证单调PTS，视频各自独立状态，静态逐张清空；完整70参考状态从真实视频帧索引取，不把稀疏参考当相邻帧。所有四臂即使失败仍测两片，没有跳过。

## 观测、状态与历史的实际增量

| 臂 | 视频 | 合格杆体帧 | 合格前沿帧 | 强当前胜者帧 | 歧义帧 | 有杆体状态 | 有端部状态 | 历史搜索帧 | 历史接受帧 | 强当前覆盖帧 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw2 | video_1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| raw2 | video_2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| light_current | video_1 | 722 | 722 | 722 | 0 | 722 | 722 | 0 | 0 | 0 |
| light_current | video_2 | 1698 | 1644 | 1697 | 1 | 1697 | 1643 | 0 | 0 | 0 |
| evidence_state | video_1 | 722 | 722 | 722 | 0 | 722 | 722 | 0 | 0 | 0 |
| evidence_state | video_2 | 1698 | 1644 | 1697 | 1 | 1764 | 1748 | 0 | 0 | 18 |
| history_search | video_1 | 722 | 722 | 722 | 0 | 722 | 722 | 0 | 0 | 0 |
| history_search | video_2 | 1698 | 1644 | 1697 | 1 | 1796 | 1766 | 105 | 72 | 20 |

杆体/前沿“合格”统计所有当前候选，包含歧义候选；状态统计实际选中/预测后的可用性，分母各722/1972。video_2历史搜索105帧、接受72帧，杆体状态由1764增至1796、端部由1748增至1766；video_1始终有强当前证据，没有触发历史搜索，不能在此片宣称历史增益。video_2的72次历史接受中只有49次同时更新端部组；历史杆体可见不代表头部已找到。更多覆盖也可能维持错误，不能称72帧恢复正确或尖端召回提高。

video_1/light_current 状态：`{"shaft": {"current_observation": 722}, "endpoint": {"current_observation": 722}}`；当前候选原因：`{"qualified_shaft": 722, "insufficient_edge_pairs": 6}`。

video_1/evidence_state 状态：`{"shaft": {"strong_initialized": 1, "strong_updated": 721}, "endpoint": {"strong_initialized": 1, "strong_updated": 721}}`；当前候选原因：`{"qualified_shaft": 722, "insufficient_edge_pairs": 6}`。

video_1/history_search 状态：`{"shaft": {"strong_initialized": 1, "strong_updated": 721}, "endpoint": {"strong_initialized": 1, "strong_updated": 721}}`；当前候选原因：`{"qualified_shaft": 722, "insufficient_edge_pairs": 6}`。

video_2/light_current 状态：`{"shaft": {"current_observation": 1697, "unavailable": 275}, "endpoint": {"current_observation": 1643, "unavailable": 329}}`；当前候选原因：`{"qualified_shaft": 1699, "insufficient_edge_pairs": 704, "width_profile": 2, "shaft_too_short": 10, "line_residual": 1}`。

video_2/evidence_state 状态：`{"shaft": {"strong_initialized": 13, "strong_updated": 1666, "strong_current_override": 18, "predicted": 66, "expired": 12, "unavailable": 196, "weak_updated": 1}, "endpoint": {"strong_initialized": 17, "strong_updated": 1609, "predicted": 104, "expired": 16, "unavailable": 208, "strong_current_override": 17, "weak_updated": 1}}`；当前候选原因：`{"qualified_shaft": 1699, "insufficient_edge_pairs": 704, "width_profile": 2, "shaft_too_short": 10, "line_residual": 1}`。

video_2/history_search 状态：`{"shaft": {"strong_initialized": 7, "strong_updated": 1670, "strong_current_override": 20, "weak_updated": 72, "predicted": 27, "expired": 6, "unavailable": 170}, "endpoint": {"strong_initialized": 13, "strong_updated": 1612, "predicted": 74, "expired": 12, "unavailable": 194, "strong_current_override": 18, "weak_updated": 49}}`；当前候选原因：`{"qualified_shaft": 1699, "insufficient_edge_pairs": 704, "width_profile": 2, "shaft_too_short": 10, "line_residual": 1}`。

全部接受/拒绝、歧义、重获、过期事件及全部候选均保留。geometry_diagnostics.json同时记录所有可用相邻对与三臂共同可用对的前沿速度/角速度/长度速度及分母，不将低抖动或拟合残差称真实准确率。没有光流或真实相机运动分离，过曝、组织双边缘、高置信误检、错误初始化、缺种子和长遮挡仍可能失败。

## 已知失败帧回归

video_2显示439帧：当前观测qN约1、双侧8/8支持，T1两状态臂本帧均strong_current_override，直接采用当前新位置；440、441走兼容更新，取消了T0在439–440仍保留旧位置的固定等待。622、643帧也逐项保留当前证据、原T0与T1状态；这验证了更新行为和定性跟随现象，不是有独立尖端GT的定位精度结论。

固定回归帧439/440/441/622/643、首个历史接受、无状态上下文和video_1均匀上下文均导出，选择规则在known_frame_diagnostics.json。全部帧都在日志与完整视频，没有只报告成功片段。需要结合RGB审核前沿是否落在真正可见端部；不能仅凭qN高就判定正确。

## 分割精度：所有四臂完全不变

| 参考 | 类别 | Dice | Precision | Recall | 边界F | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| video_native | stone | 0.740216 | 0.899778 | 0.628722 | 0.475440 | 152516 | 16988 | 90065 |
| video_native | fiber | 0.945537 | 0.970967 | 0.921404 | 0.862027 | 417783 | 12492 | 35637 |
| image_native | stone | 0.906751 | 0.880212 | 0.934940 | 0.393852 | 6120805 | 832978 | 425928 |
| image_native | fiber | 0.765758 | 0.643433 | 0.945513 | 0.524869 | 105281 | 58343 | 6067 |

所有10776帧原图掩膜与原model1逐帧一致，202张静态概率和掩膜精确复现。全部四臂TP删除/恢复、FP移除/新增均0，不能把不变性称分割改善。逐帧70/132的混淆矩阵、边界与各原始分组完整保存，classwise冻结参照另列；model1先前不退化检查未通过的结论不变，不自动采用底座。原85/523个结石保护冲突仍未解决。

| 参考 | 可见语义中心线Recall | 中心线Precision | clDice |
| --- | --- | --- | --- |
| video_native | 0.931075 | 0.968837 | 0.949580 |
| image_native | 0.911935 | 0.752349 | 0.824490 |

中心线为对现有可见语义掩膜进行Zhang-Suen细化后的精确像素覆盖，逐样本分母已保存；此项为只读描述性补充，不是新门槛、真实解剖中心线或尖端真值。所有臂因掩膜相同而相同。未知255不计入前景，未用预测生成尖端参考。

## 核验、可视化与停止

19项新增机制测试+11项T0回归，共30项通过；73项独立核验通过：源码/配置/权重/输入身份、202静态候选及指标重算、8独立进程完整10776帧、逐帧原掩膜和RGB重解码、相同当前观测、两状态全片及前缀重放、选中帧历史图像证据复算、阶段分位数与完整视频。审计没有额外网络前向，没有修改阈值或原结果。

原运行导出state_comparison/state_replay保留。其“仅杆体、无端部”面板用了固定20%画幅的示意线，不是已观测长度；最终evidence_comparison改成明确的短灰色方向箭头，不画未知端部，避免误读。该修正仅重新渲染已有元数据，不改变状态、掩膜、指标或性能记录。青色为当前支持状态、黄色为含预测组的几何；两侧直线及前沿圆点是未标定线模型示意，不代表真实端帽轮廓。状态条始终在原图之外。

- [video_1完整四方案对比](video_1_evidence_comparison.mp4)、[T0/T1回归诊断板](video_1_evidence_diagnostic_board.jpg)、[完整参数曲线](video_1_parameter_traces.png)

- [video_2完整四方案对比](video_2_evidence_comparison.mp4)、[T0/T1回归诊断板](video_2_evidence_diagnostic_board.jpg)、[完整参数曲线](video_2_parameter_traces.png)

T1有限四臂按协议收尾，运行成本和因果机制通过，真实几何准确性仍未验收。历史只证明能增加当前证据支持的状态更新，不能证明更准确；优先保留更简单的light_current/evidence_state作研究对照，不将history_search或model1接入默认。下一步先只读审核误接收/过期及观测-语义参考差异，形成单独的可见像素修正候选与协议；本轮不自动进入T2实现、训练、默认替换、结石S3或物理落点。任何训练仍须独立批准数据、初始化、实际增强、预算和回滚方案。
