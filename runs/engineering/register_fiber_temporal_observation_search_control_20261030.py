"""Register completed finite T1 control, retaining old defaults and negative evidence."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.fiber_reference_workspace import read,sha256
from ureter_semis.experiment import atomic_json
OUT=ROOT/'runs/a1_fiber_temporal_observation_search_20261030'


def main():
    s=read(OUT/'summary.json');a=read(OUT/'engineering_checks.json');static=read(OUT/'static_geometry_diagnostics.json')
    assert read(OUT/'job.json')['phase']=='completed' and s['all_masks_unchanged'] and s['full_forward_frames']==10776 and a['passed']==73
    assert all(v['geometry_p95_within_design_budget'] and v['pipeline_p95_within_source_interval'] for values in s['full_video'].values() for v in values.values())
    artifacts=read(OUT/'report_artifacts.json')
    assert all(sha256(v['path'])==v['sha256'] for v in artifacts['artifacts'].values())
    doc=ROOT/'docs/progress/2026-10-30-fiber-temporal-observation-search-control.md'
    if doc.exists():raise FileExistsError(doc)
    text=['# T1轻量观测、当前证据优先与历史搜索对照\n',
          '2026-10-30。**四臂202静态复现与各2694帧完整视频测试完成，8独立进程共10776帧。耗时与因果机制通过，但几何适用范围和准确性仍未通过；不默认采用。所有分割掩膜不变，本轮没有分割准确率改善。**\n',
          '按执行前冻结协议，raw2仅原model1两网；light_current增加160网格最多2个当前轴线及有界RGB/概率双边缘与前沿观测；evidence_state增加强当前优先的部分状态；history_search仅在没有强当前胜者时增加最多1个历史候选。每轴8横截面、每侧17采样和3×25端部采样，取消旧全原图密集边缘/端帽枚举。没有额外网络、裁剪重网、蓝色硬门控或几何掩膜修正。\n',
          '强当前证据越界本帧重获，不无条件等待3次；弱观测须兼容历史，杆体/端部独立有效与过期，端部缺失不直接缩短。历史只搜索当前RGB和网络证据，不能补齐遮挡或升级独立强证据；静态图无状态传递，各视频独立初始化。\n',
          '| 方案 | video_1 总管线p50/p95 ms | video_2 总管线p50/p95 ms | 两片几何p95 ms |\n|---|---:|---:|---:|']
    for arm,vs in s['full_video'].items():
        vals=[f"{vs[v]['timings']['pipeline_total_ms']['p50']:.2f}/{vs[v]['timings']['pipeline_total_ms']['p95']:.2f}" for v in ['video_1','video_2']]
        geo='/'.join(f"{vs[v]['timings']['geometry_total_ms']['p95']:.2f}" for v in ['video_1','video_2'])
        text.append('| '+' | '.join([arm,*vals,geo])+' |')
    text+=['\n三几何方案在两片均通过几何p95≤5ms及直接测得的总管线源帧预算。FP32/batch1、两网49.20M、RTX4070Ti，每进程12帧临时状态预热后清空。管线含全部网络、两份原图三类概率与掩膜传CPU和元数据处理；含解码/编码/审计的实际离线吞吐约48.6–49.8fps。没有实时摄像头/显示队列验收；不把分位数相减，也不与历史T0接口混比。\n',
             'video_2历史搜索105帧、接受72帧，杆体状态覆盖由1764增至1796，端部由1748增至1766；只有49次历史接受同时更新端部。video_1无历史触发，不能称其验证了历史增益。覆盖增加不证明位置正确，72次接受不叫72帧正确恢复。\n',
             'video_2显示439帧当前qN约1、8/8双边缘支持，两T1状态臂本帧直接重获；440/441正常更新，定性上避免T0的固定等待延迟。622/643、首个历史接受311及缺失1266帧保留。1266帧RGB仍有光纤，但只有3/8双边缘支持而拒绝，历史已过期，仍是失败；311当前种子4/8失败、历史候选5/8并以当前高概率通过，只证明候选搜索路径有增量。\n',
             '**静态适用范围仍受限：70视频参考有68张可用几何；132图像中50张参考含光纤，仅11张有状态，另1张参考未标光纤却被接受，总12/132。** 失败板的该接受候选落在局部亮斑附近，与右侧蓝色细条分离，提示高网络概率与双边缘不能独立保证身份；这是图像观察，原标注完整性仍需区分，不新增GT或直接判物理尖端。另445像素无种子样本和宽度族拒绝的真实语义参考保留。不能因视频成本通过就称通用几何定位达标。\n',
             '70/132所有分割概率、掩膜和指标精确复现：视频结石/光纤Dice0.740216/0.945537，图像0.906751/0.765758。各类TP删除/恢复、FP新增/减少均0；逐帧与原29/新41、病例、来源及曝光分组保留，原classwise质量参照与model1未采用结论不变。补充可见语义掩膜细化中心线覆盖/clDice仅作描述，非解剖中心线/尖端真值或新门槛。\n',
             '19项新增机制测试、11项T0回归与73项独立核验通过，包括202静态候选重算、10776原图掩膜/RGB重解码、相同当前观测、完整/前缀状态重放、选中帧历史图像证据和完整视频。源码、配置、权重、标签与输入身份归档，无训练，521训练/144保留test像素读取0。视频及派生物仅验证，stone_new全部训练排除，Godot参数不作真实标定。\n',
             '70帧已参与开发；42/120随机标注未齐，不外推全片精度。独立尖端/方向真值仍没有，不报告尖端Recall、定位误差、遮挡恢复或物理落点准确率。没有默认替换、T2像素融合、训练或结石S3。\n',
             '按有限四臂停止，不调阈值/权重/TTL或扩候选。下一步只读梳理T1拒绝、亮斑误接收、几何方向/端部与语义参考差异，分开观测定位、身份判断及网络漏检，再形成独立的可见像素修正候选协议；不自动实施T2，也不预设颜色/几何或训练必有效。\n',
             '- [完整报告](../../runs/a1_fiber_temporal_observation_search_20261030/report.md)、[静态失败补充](../../runs/a1_fiber_temporal_observation_search_20261030/static_geometry_limitations.md)、[73项核验](../../runs/a1_fiber_temporal_observation_search_20261030/engineering_checks.json)\n- [video_1完整四方案对比](../../runs/a1_fiber_temporal_observation_search_20261030/video_1_evidence_comparison.mp4)、[video_2完整四方案对比](../../runs/a1_fiber_temporal_observation_search_20261030/video_2_evidence_comparison.mp4)\n- [439帧T0/T1对照](../../runs/a1_fiber_temporal_observation_search_20261030/video_2_frame439_comparison.jpg)、[图像失败板](../../runs/a1_fiber_temporal_observation_search_20261030/image_development_failure_board.jpg)\n- [执行前冻结方案](../plans/fiber_temporal_observation_search_20261030.md)及[配置](../../configs/fiber_temporal_observation_search_20261030.json)保留当时未执行状态；当前状态以本记录/job为准。\n',
             '最终evidence_comparison视频把端部未知时的图示改为短灰色方向箭头，避免把原state_comparison中的固定长度示意线误读为真实杆体长度；旧导出保留。重绘仅处理已有元数据，不改推理结果、性能或掩膜。状态条在图外，所有几何仍为未标定示意。\n']
    doc.write_text('\n'.join(text),encoding='utf-8')
    main=ROOT/'docs/plans/segmentation_temporal_occlusion_plan.md';t=main.read_text(encoding='utf-8');assert '版本：v2.17' in t
    lines=[]
    for line in t.replace('版本：v2.17','版本：v2.18',1).splitlines():
        if line.startswith('| S2 |'):line='| S2 | T1四臂202静态及8进程10776全片完成，30测试/73核验；几何p95≤4.70ms、总p95≤19.49ms通过成本，但掩膜不变，图像几何只12/132可用，准确性未验收 | 下一步只读定位拒绝/亮斑误接收/端部与语义参考差异，再冻结独立像素修正候选协议；不调T1阈值，不自动T2实现、训练、默认采用或结石S3 |'
        elif line.startswith('- [ ] **当前下一步：T1实现与完整四臂对照**'):
            line='- [x] **T1实现与完整四臂对照**：完成19新增+11T0回归测试、202静态和8独立进程10776帧、73项独立核验；掩膜不变。成本通过，439帧强当前立即重获；历史105搜索/72接受只增加状态覆盖，不证明准确。图像50张参考含光纤仅11张有状态，另1张无参考光纤却接受；不默认采用。见[结果](../progress/2026-10-30-fiber-temporal-observation-search-control.md)。\n- [ ] **当前下一步：T1失效归因及独立像素修正协议**：先只读检查双边缘拒绝、亮斑误接收、FOV/方向/端部与语义参考差异，分开身份、观测定位和网络漏检；在无尖端真值条件下明确不可计算指标，再提出最小可见像素修正候选和独立对照。不调整本轮阈值，不自动实施T2、训练、默认替换或结石S3。'
        lines.append(line)
    addition='2026-10-30 T1最新执行更新（覆盖以上历史下一步）：四臂202静态及8独立进程10776帧全部完成，30测试/73独立核验通过。总管线p95为raw2 15.66/15.27、light18.59/19.31、evidence19.06/19.40、history19.48/19.25ms；三几何方案均通过5ms设计预算。video_2历史105搜索/72接受仅证明覆盖增加，439帧强当前立即重获；图像50张含光纤参考仅11张有状态、另1张参考无光纤却接受，通用几何准确性未验收。掩膜全部不变、无准确率提升。下一步只读失效归因并制定独立像素修正候选协议，不自动T2实现、训练、默认采用或S3。'
    main.write_text('\n'.join(lines)+'\n\n'+addition+'[本轮结果](../progress/2026-10-30-fiber-temporal-observation-search-control.md)。\n',encoding='utf-8')
    rp=ROOT/'README.md';rt=rp.read_text(encoding='utf-8');assert rt.count('\n## 实现说明')==1
    rp.write_text(rt.replace('\n## 实现说明','\n'+addition+'[完整报告与视频](docs/progress/2026-10-30-fiber-temporal-observation-search-control.md)。\n\n## 实现说明',1),encoding='utf-8')
    path=ROOT/'annotations/data_review_status_20261027.json';d=read(path);prefix='fiber_temporal_observation_search'
    d.update({prefix+'_implemented':True,prefix+'_executed':True,prefix+'_adopted':False,prefix+'_report':doc.relative_to(ROOT).as_posix(),
              prefix+'_summary':str(OUT/'summary.json'),prefix+'_engineering_checks':73,prefix+'_new_tests':19,prefix+'_regression_tests':11,
              prefix+'_full_frames':10776,prefix+'_static_fresh_frames':202,prefix+'_all_masks_unchanged':True,
              prefix+'_all_runtime_gates_passed':True,prefix+'_geometry_accuracy_validated':False,
              prefix+'_history_search_attempted_video2':105,prefix+'_history_used_video2':72,
              prefix+'_static_geometry_available_image132':12,prefix+'_static_geometry_available_in_gt_positive50':11})
    d.update(current_execution_checks=str(OUT/'engineering_checks.json'),last_executed_diagnostic_config='configs/fiber_temporal_observation_search_20261030.json',
             latest_instruction='Continue frozen T1 implementation and all four full-video arms',
             latest_instruction_effect='Completed T1:202static+10776full frames,30tests/73audits, same masks, cost passes, geometry accuracy and static applicability unvalidated; no adoption/training/S3',
             execution_order_confirmation='T1 finite experiment complete; next read-only T1 failure diagnosis and separate visible-pixel-correction protocol, no automatic T2 implementation/training',
             remaining_prerequisites=[
                 'Next read-only: why visible fiber loses bilateral edges, why bright-spot candidate accepted, FOV/direction/front/semantic-reference differences; separate identity/localization/network/label factors',
                 'Then formulate separate minimal visible-pixel correction protocol, no automatic T2 implementation or threshold tuning',
                 'T1 cost and causal mechanisms passed; geometry accuracy not validated:12of132 image poses,11of50 reference-positive; more history coverage is not accuracy',
                 '70development refs and42of120 incomplete random sample cannot establish whole-video accuracy;12dedicated tip references remain pending',
                 'No new training/default/stoneS3, model1 still unadopted; video derivatives validation only, stone_new excluded,521train and144test pixels excluded'
             ])
    atomic_json(path,d)
    path=ROOT/'annotations/video_evaluation_registry.json';reg=read(path)
    reg['previous_state_control_before_T1']={k:v for k,v in reg.items() if k.startswith('latest_state_control_')}
    reg.update(latest_plan_executed=True,latest_state_control_config='configs/fiber_temporal_observation_search_20261030.json',
               latest_state_control_result=str(OUT/'summary.json'),latest_state_control_result_sha256=sha256(OUT/'summary.json'),latest_state_control_changes_visible_masks=False,
               latest_state_control_geometry_accuracy_validated=False,latest_state_control_report=doc.relative_to(ROOT).as_posix())
    atomic_json(path,reg)
    atomic_json(OUT/'registration.json',{'registrar_sha256':sha256(Path(__file__)),'progress_document_sha256':sha256(doc),'main_plan_version':'v2.18',
                                        'completed':True,'adopted':False,'training_updates':0,'default_changed':False,'state_only':True,
                                        'report_artifacts_sha256':sha256(OUT/'report_artifacts.json')})
    print(json.dumps({'main_plan':'v2.18','T1':'completed_not_adopted','full_frames':10776,'tests':30,'independent_checks':73}))


if __name__=='__main__':main()
