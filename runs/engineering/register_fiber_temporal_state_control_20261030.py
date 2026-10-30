"""Record completed T0 without enabling pixel fusion, search, training or defaults."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.fiber_reference_workspace import read,sha256
from ureter_semis.experiment import atomic_json
OUT=ROOT/'runs/a1_fiber_temporal_state_gate_20261030'

def main():
    s=read(OUT/'summary.json');audit=read(OUT/'engineering_checks.json');cfg=read(OUT/'preflight.json')['config']
    assert read(OUT/'job.json')['phase']=='completed' and s['all_masks_unchanged'] and s['full_forward_frames']==10776
    report_path=ROOT/'docs/progress/2026-10-30-fiber-temporal-state-control.md'
    if report_path.exists():raise FileExistsError(report_path)
    lines=['# 光纤时序几何状态T0对照\n',
        '2026-10-30。**已完成四臂、8个独立进程、两片各722/1972帧，共10776帧。异常门控机制实现并通过核验，但可视化出现跟随延迟，完整几何管线也未满足源帧预算，不采用为默认。分割掩膜逐像素不变，本轮没有分割准确率改善。**\n',
        '按执行前冻结方案，raw2、逐帧当前几何观测、旧alpha-beta状态、新门控状态共用相同model1两网底座。三几何臂的当前RGB/FOV/边缘/端帽拟合逐帧相同；旧拟合器/跟踪器、alpha/beta和原跳变门槛未修改，新门控只改变异常怎样进入状态。入口、轴向、长度、宽度与端帽联合表示，末端和两侧方向由同一状态推导，不独立平滑。\n',
        '新状态器将孤立越界观测暂存，连续3次彼此一致后在当前时刻重定位，不回改前两帧；最多3帧仅预测，第4帧清空，待确认不会借活动状态过期绕过确认。每视频/每静态图独立状态，预热后清空；源/分辨率/时间异常清空。预测只为独立元数据，不裁剪或填补可见光纤，不强连断片。\n',
        '## 状态行为\n']
    lines+=['| 臂 | 视频 | 当前有效拟合 | 有状态帧 | 仅预测帧 | 状态事件 |','|---|---|---:|---:|---:|---|']
    for arm,vs in s['full_video'].items():
        for vid,v in vs.items():lines.append('| '+' | '.join(map(str,[arm,vid,v['fit_accepted_frames'],v['state_available_frames'],v['prediction_only_frames'],json.dumps(v['states'],ensure_ascii=False)]))+' |')
    gated=s['full_video']['gated_state_only'];old=s['full_video']['legacy_state_only']
    lines.append('\nvideo_1没有触发异常，旧/新均初始化1次、更新721次，不能声称在该片验证了异常拒绝收益。video_2旧版有7次立即跳变重置，新版有12帧异常暂存（另2帧活动状态到期仍待确认）、4次确认重定位；另有1次非法几何外推清空（显示帧622），前一状态末端宽度已很小，继续外推将不再满足有效几何。该失败保留，未夹紧成理想轮廓或填像素。')
    lines.append(f"\nvideo_2旧/新有状态帧分别{old['video_2']['state_available_frames']}/{gated['video_2']['state_available_frames']}，仅预测{old['video_2']['prediction_only_frames']}/{gated['video_2']['prediction_only_frames']}。异常拒绝可能减少错误跳变，也可能延迟真实移动、降低可用覆盖；没有独立尖端/方向真值，不能给这些真实事件自动判真伪或报告尖端误差。完整运动曲线、全可用/共同可用帧对分开报告，低抖动不等于高准确率。\n")
    lines.append('定性审核发现：video_2显示帧439–440中，当前RGB可见光纤及当前拟合已支持新位置，新门控仍预测旧位置、明显偏离当前光纤，441才确认。该例是固定三帧等待造成跟随延迟的可视化证据；不是新增尖端GT，也不用于回调本轮阈值或给全部暂存事件判错。故不能推荐新固定门控优于旧状态器，下一版需要充分当前图像证据覆盖历史先验，避免盲目等待。\n')
    lines+=['## 实际完整管线性能\n','| 臂 | video_1 p50/p95 ms | video_2 p50/p95 ms |','|---|---:|---:|']
    for arm,vs in s['full_video'].items():
        vals=[f"{vs[v]['timings']['pipeline_total_ms']['p50']:.2f}/{vs[v]['timings']['pipeline_total_ms']['p95']:.2f}" for v in ['video_1','video_2']]
        lines.append('| '+' | '.join([arm,*vals])+' |')
    lines.append('\n新状态更新中位约0.12ms、p95约0.22ms；当前FOV/边缘/端帽拟合中位约17–18ms，主导新增开销。新门控完整管线p95为49.75/47.11ms，超过约33ms源帧预算。实际仍两网49.20M参数，FP32、batch1、RTX4070Ti，12帧临时状态预热后清空；源时间采用逐帧核验的单调解码PTS。\n')
    lines.append('计时含全部所需网络与两份原图三类概率/融合掩膜传CPU，另外分解拟合与状态；完整周期还含解码、视频编码、哈希、组件来源报告，最终JSON写盘不计入离线fps。只用本轮匹配参照，不相加/相减分位数，不与历史接口混比。组件来源额外计算在管线外、周期内。所有臂即使成本失败也测完整两片；不宣称实时采集/显示验收。\n')
    lines.append('## 质量、验证与下一步\n')
    lines.append('202张静态原图重新推理一次，各臂共享相同掩膜；70帧两专家概率与全部分割指标精确复现model1。视频/图像结石Dice为0.740216/0.906751，光纤为0.945537/0.765758，TP/FP/FN、边界及各原29/新41、病例、来源/曝光分组不变。10776帧均与已有model1原图掩膜一致，TP删除或恢复、FP增减全部0。先前model1相对三网严格不退化失败仍有效，未默认采用。\n')
    lines.append(f"11项新增状态测试、3项旧几何回归及{audit['passed']}项独立核验通过，包括全部原掩膜/来源身份、相同当前观测、旧状态独立重放、新状态全片与前缀因果重放、计时和10段视频导出。首次审计对归一化角度用了严格浮点相等而误报；已改为1e-12数值容差，初次日志保留，没有改实验状态器、门控阈值、源码或结果。\n")
    lines.append('T0按固定四臂收尾，不再调门槛、确认数或TTL。下一步先制定T1低成本当前图像观测/历史辅助候选搜索的独立协议：历史只限定搜索位置，当前网络/双侧边缘/对比度提供证据，强当前证据应能覆盖历史，证据不足才更多依赖连续性；区分镜头运动与光纤变化的需要单列。先限定观测与拟合预算、全图保底、真实快速移动/无种子/遮挡回退，再设计有无历史对照。不能直接叠加旧重网裁剪、把预测轮廓填成可见光纤，或自动进入像素修正、训练、默认替换、结石S3。\n')
    lines.append('视频及派生物不训练，stone_new全部训练排除，原521/132/144/29与活动70标签不变，本轮TRAIN/144保留test像素读取0。70帧是重复开发参考，42/120随机标注未完成，不外推全片精度；12帧尖端专项仍待补，Godot参数不作真实标定。\n')
    lines.append('- [完整报告](../../runs/a1_fiber_temporal_state_gate_20261030/report.md)、[工程核验](../../runs/a1_fiber_temporal_state_gate_20261030/engineering_checks.json)、[几何诊断](../../runs/a1_fiber_temporal_state_gate_20261030/geometry_diagnostics.json)\n- [video_1完整五栏对比](../../runs/a1_fiber_temporal_state_gate_20261030/video_1_state_comparison.mp4)、[video_2完整五栏对比](../../runs/a1_fiber_temporal_state_gate_20261030/video_2_state_comparison.mp4)\n- [video_1参数曲线](../../runs/a1_fiber_temporal_state_gate_20261030/video_1_parameter_traces.png)、[video_2参数曲线](../../runs/a1_fiber_temporal_state_gate_20261030/video_2_parameter_traces.png)\n- [执行前冻结方案](../plans/fiber_temporal_state_gate_20261030.md)、[执行前冻结配置](../../configs/fiber_temporal_state_gate_20261030.json)（保留当时“未执行”状态；当前执行状态见job/本报告）\n')
    report_path.write_text('\n'.join(lines),encoding='utf-8')
    p=ROOT/'docs/plans/segmentation_temporal_occlusion_plan.md';text=p.read_text(encoding='utf-8');assert '版本：v2.15' in text
    text=text.replace('版本：v2.15','版本：v2.16',1);new=[]
    for line in text.splitlines():
        if line.startswith('| S2 |'):line='| S2 | T0四臂状态对照完成202静态和各2694帧；掩膜不变。新异常暂存/三次确认机制通过测试；真实几何准确性未验证，拟合成本使新管线p95约49.75/47.11ms未达预算 | 下一步制定T1低成本当前图像观测/历史辅助搜索协议，先限制拟合预算和失效回退。T0不继续调参；不自动像素修正、训练、默认采用或进入结石S3 |'
        elif line.startswith('- [ ] **当前下一步：T0时序几何状态四臂实现与完整回放**'):
            line=f'- [x] **T0时序几何状态四臂实现与完整回放**：完成8独立进程10776帧和202静态复现，掩膜逐像素不变。video_2新门控异常暂存12帧、确认重定位4、非法外推清空1；439–440定性可见跟随延迟，441才确认，不能称新门控更准。video_1无异常触发。新管线p95为49.75/47.11ms，当前拟合成本仍高。14项测试、{audit["passed"]}项独立核验通过，旧源码/配置/权重保留，无训练或S3。详见[结果](../progress/2026-10-30-fiber-temporal-state-control.md)。\n- [ ] **当前下一步：T1低成本观测与历史辅助搜索协议**：先限定当前图像观测/拟合的预算与证据要求，历史只提供搜索位置，充分当前证据可覆盖历史，不能无条件等待3帧；必要时区分镜头运动与光纤变化。全图保底，允许可见断片，快速移动/初始无种子/遮挡/过期要明确拒绝或重获。单列有无历史匹配对照、固定候选数和两片全量性能；不直接叠加旧裁剪重网或模板裁掩膜。未经新协议不实施T1、像素融合或训练；无独立几何参考不宣称尖端/方向更准。'
        new.append(line)
    addition='2026-10-30 T0最新执行更新（覆盖以上历史下一步）：四臂202静态与各2694帧完整状态对照已完成，共8进程10776帧，所有掩膜不变；新门控在video_2记录12帧暂存、4次确认、1次非法外推清空，video_1未触发异常。状态器中位约0.12ms，但当前拟合主导成本，新管线p95为49.75/47.11ms未达预算。不把低抖动称准确率或修改默认，下一步另行制定T1低成本当前图像观测/历史辅助搜索协议，不自动执行像素修正、训练或S3。'
    p.write_text('\n'.join(new)+'\n\n'+addition+'[本轮结果](../progress/2026-10-30-fiber-temporal-state-control.md)。\n',encoding='utf-8')
    p=ROOT/'README.md';text=p.read_text(encoding='utf-8');assert addition not in text
    p.write_text(text.replace('\n## 实现说明','\n'+addition+'[结果与完整视频](docs/progress/2026-10-30-fiber-temporal-state-control.md)。\n\n## 实现说明',1),encoding='utf-8')
    p=ROOT/'annotations/data_review_status_20261027.json';d=read(p)
    d.update(fiber_temporal_state_control_implemented=True,fiber_temporal_state_control_executed=True,fiber_temporal_state_control_adopted=False,
        fiber_temporal_state_control_report='docs/progress/2026-10-30-fiber-temporal-state-control.md',fiber_temporal_state_control_summary=str(OUT/'summary.json'),
        fiber_temporal_state_control_engineering_checks=audit['passed'],fiber_temporal_state_control_new_tests=11,fiber_temporal_state_control_regression_tests=3,
        fiber_temporal_state_control_full_frames=10776,fiber_temporal_state_control_static_fresh_frames=202,fiber_temporal_state_control_all_masks_unchanged=True,
        fiber_temporal_state_control_gated_p95_ms={v:x['timings']['pipeline_total_ms']['p95'] for v,x in gated.items()},
        current_execution_checks=str(OUT/'engineering_checks.json'),last_executed_diagnostic_config='configs/fiber_temporal_state_gate_20261030.json',
        latest_instruction='Continue frozen T0 implementation and complete both videos for all4arms',latest_instruction_effect='CompletedT0:8fresh processes10776frames+202static, state-only exact masks,14tests, gated outlier/confirmation behavior; geometry pipe latency fails. No T1/training/default/S3.',
        execution_order_confirmation='T0 finite comparison completed; next prepare separate T1 lightweight current-image observation/history-search protocol, no automatic execution or pixel changes',
        remaining_prerequisites=[
            'Next: separate T1 plan with lightweight current-image observation/fit budget, history-only search location and current evidence; full-frame fallback and fast-motion/no-seed/occlusion/reset behavior',
            'T0 complete: all4arms2694each, masks unchanged,14tests; current fit dominates cost,49.75/47.11ms gated p95 fails source budget; no more threshold/TTL/confirmation tuning',
            'True geometry not validated; motion/jitter/residual diagnostics not accuracy; static/multiple-case state isolation retained, no future smoothing',
            'No automatic T1 pixel fusion/training/default adoption or stoneS3; model1 still candidate, frozen classwise reference and default unchanged',
            '70development frames and42of120 partial sample,12tip references pending; no whole-video accuracy extrapolation or physical tip/laser truth',
            'All video derivatives validation only; stone_new excluded;144test pixels excluded; training needs separately approved compliant initialization/data/augmentation/budget/validation/rollback plan'
        ])
    atomic_json(p,d)
    p=ROOT/'annotations/video_evaluation_registry.json';d=read(p)
    d.update(latest_plan_executed=True,latest_state_control_config='configs/fiber_temporal_state_gate_20261030.json',latest_state_control_result=str(OUT/'summary.json'),latest_state_control_result_sha256=sha256(OUT/'summary.json'),latest_state_control_changes_visible_masks=False)
    atomic_json(p,d)
    print(json.dumps({'plan':'v2.16','checks':audit['passed'],'full_frames':10776,'status':'T0_completed_not_adopted','next':'T1_protocol_only'}))

if __name__=='__main__':main()
