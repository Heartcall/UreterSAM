# G0 前向stone候选与提案回退删除

2026-11-04。针对video_1末帧的尖端后缩，当前G0执行路径作两项明确修改；旧冻结实验及其报告不改写。

1. 尖端候选仍要求当前像素不是stone，并保留蓝色身份、局部3×3支持、前向蓝色比例、RGB/色度终止对比、有效采样和画面边距条件；但前向2/4/6px采样碰到stone不再否决该候选。日志改记`forward_stone_candidates`、`forward_stone_rejection_enabled=false`和`selected_forward_stone`。
2. 只要唯一主体、尖端T和正半径相切圆已经构造出来，拟合跨度/宽度/细长性/残差、搜索上限、面积预算和stone相交均只作为`nonblocking_diagnostics`，不再保留旧S2。G0几何照常替换旧fiber；与Raw2 stone相交的提案像素不改stone类，另存`stone_clipped`层和计数。

无唯一可信主体、可信主体没有唯一旧连通域、侧线交点/方向不可解、没有尖端T，或T到达/越过侧线交点时不存在可应用的G0几何。这些帧状态为`unavailable`，而不是把不存在的几何标为已应用。输入尺寸或旧stone身份不一致仍是数据错误并抛异常。

新配置为`configs/tapered_tangent_cap_no_fallback_20261104.json`，正式执行入口`tools/run_tapered_tangent_cap_control.py`已切换到该配置。原`configs/tapered_tangent_cap_control_20261104.json`保留旧回退语义，使历史mask、日志、RGB校正对照和seal仍可复现。新输出使用独立目录`runs/a1_tapered_tangent_cap_no_fallback_20261104`，不会覆盖旧结果。

新增合成测试覆盖：前向stone候选可被实际选中；stone相交时应用并裁切stone；质量和面积检查不阻断；不可构造状态为unavailable；旧冻结配置仍逐帧回退。完整视频尚未以新配置重跑，因此本文只声明实现和测试行为，不声明video_1/video_2的新全片指标或尖端准确率。
