"""Freeze dark-input response protocol only; no images decoded or transforms run."""
import json,re,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.fiber_reference_workspace import read,sha256
from ureter_semis.experiment import atomic_json
CFG='configs/fiber_dark_input_control_20261030.json'
DOC='docs/plans/fiber_dark_input_control_20261030.md'
PROGRESS='docs/progress/2026-10-30-fiber-dark-input-protocol.md'
OUT=ROOT/'runs/protocols/fiber_dark_input_control_20261030_v1'


def main():
    config=read(ROOT/CFG);pre=read(ROOT/'runs/a1_fiber_temporal_observation_search_20261030/preflight.json')
    assert not OUT.exists() and not(ROOT/PROGRESS).exists()
    assert not(ROOT/config['planned_output']).exists() and all(not(ROOT/p).exists() for p in config['planned_files'])
    registry_path=ROOT/'annotations/video_evaluation_registry.json';registry=read(registry_path)
    actual={k:v for k,v in registry.items() if k.startswith('latest_state_control_') or k.startswith('latest_inference_control')}
    manifest_path=ROOT/'runs/fiber_reference_d1_20261029/manifest.json';session_path=ROOT/'runs/fiber_reference_d1_20261029/session/session.json'
    manifest=read(manifest_path);session=read(session_path);active=read(ROOT/config['data']['video_manifest']);images=read(ROOT/config['data']['image_manifest'])
    tiprows=[r for r in manifest['records'] if r['kind']=='tip'];worklist=[]
    for r in tiprows:
        worklist.append({'id':r['id'],'role':'existing_pending_tip_pilot','video_id':r['video_id'],'frame_index':r['frame_index'],
            'display_frame':r['frame_index']+1,'image':r['image'],'image_sha256':r['image_sha256'],'annotation_state_at_planning':session['decisions'].get(r['id'],{}).get('status','pending'),
            'context_semantic_mask':r.get('context_mask'),'tip_truth_created':False,'training_allowed':False})
    for vid,index,role in [('video_1',77,'complete_end_comparator'),('video_2',1208,'dark_visible_front_failure')]:
        r=next(r for r in active['val'] if r['video_id']==vid and r['frame_index']==index)
        worklist.append({'id':f'planned_tip_extension:{vid}:{index}','role':role,'video_id':vid,'frame_index':index,'display_frame':index+1,
            'image':r['image'],'image_sha256':sha256(r['image']),'annotation_state_at_planning':'planned_not_annotated',
            'context_semantic_mask':r['mask'],'tip_truth_created':False,'training_allowed':False})
    assert len(worklist)==14 and len({(r['video_id'],r['frame_index']) for r in worklist})==14
    checks={
        'finite_three_arms':config['arms']==['raw2','fiber_vlift','fiber_channel_gamma'],
        'single_fixed_gamma':config['input']['gamma']==.7,
        'two_network_instances':config['base']['networks']==2 and config['base']['parameters']==49202790,
        'six_fresh_processes_full_even_failure':config['execution']['expected_processes']==6 and config['execution']['expected_full_forward_frames']==3*(722+1972)==8082 and config['execution']['all_arms_full_even_static_or_runtime_failure'],
        'actual_distinct_static_inputs':config['execution']['expected_static_paired_frame_inferences']==3*(70+132)==606,
        'reference_counts_unchanged':len(active['val'])==70 and len(images['val'])==132 and len(images['train']['labeled'])==521 and len(images['reserved_test'])==144,
        'same_frozen_source_code':all(sha256(ROOT/k)==v for k,v in pre['sources'].items()),
        'same_weight_files':all(sha256(v['path'])==v['sha256'] for v in pre['checkpoints'].values()),
        'same_semantic_manifest':sha256(ROOT/config['data']['video_manifest'])==registry['active_video_test_manifest_sha256'] and sha256(ROOT/config['data']['image_manifest'])==registry['original_image132_and_train521_manifest_sha256'],
        'original12pending_not_truth':len(tiprows)==12 and all(session['decisions'].get(r['id'],{}).get('status','pending')=='pending' for r in tiprows),
        'planning_worklist14unique_no_labels':len(worklist)==14 and all(not r['tip_truth_created'] and not r['training_allowed'] for r in worklist),
        'worklist_images_identity':all(sha256(r['image'])==r['image_sha256'] for r in worklist),
        'not_implemented_or_executed':all(not(ROOT/p).exists() for p in config['planned_files']) and not(ROOT/config['planned_output']).exists(),
        'existing_T2P_protocol_unchanged':sha256(ROOT/'configs/fiber_local_evidence_control_20261030.json')==read(ROOT/'runs/a1_fiber_dark_appearance_diagnosis_20261030/identity.json')['frozen_T2P_config_sha256'],
        'unchanged_quality_gate_reference':sha256(ROOT/config['quality']['unchanged_gate_reference'])==read(ROOT/'runs/a1_two_network_experts_20261029/preflight.json')['config_sha256']
    }
    assert all(checks.values()),checks
    OUT.mkdir(parents=True)
    atomic_json(OUT/'tip_reference_worklist.json',{'role':'planned_tasks_NOT_reference_labels','entries':worklist,
        'existing_tip_records':12,'additional_planned_records':2,'approved_tip_truth':0,
        'source_manifest':str(manifest_path),'source_manifest_sha256':sha256(manifest_path),
        'source_session_metadata_sha256_at_planning':sha256(session_path),
        'policy':'Original D1 manifest/session/labels untouched. This snapshot does not prevent later legitimate annotation; never treat planned/pending as GT.'})
    report='''# 暗色前端固定输入响应协议已制定

2026-10-30。**本步仅核对实现路径、参考状态并冻结协议；没有执行输入变换、模型前向、分割修正或训练。** 默认、权重、原标签/划分和全部历史实验保留。

## 最小对照

三臂均为原old190结石+A0 model1光纤两网，49.20M参数：

| 臂 | 光纤网络输入 | 结石网络输入 |
|---|---|---|
| raw2 | 原256 RGB | 原256 RGB |
| fiber_vlift | 最大通道V的gamma0.7提亮，RGB共同缩放 | 原256 RGB |
| fiber_channel_gamma | RGB逐通道gamma0.7 | 原256 RGB |

变换在PIL双线性256缩放后、ImageNet归一化前，以float32运行，不转uint8或改变原图/GT。共同缩放保持归一化RGB比例；两变换最大通道值一致，但Lab亮度/局部对比度不完全匹配，不能把差异简单归于色相或照明因果。固定一个强度，不扫gamma或按视频适配。

每帧仅结石一次、对应光纤输入一次，不叠加原光纤前向、TTA、概率取max、缓存保护或时序/几何填充。原结石概率必须精确一致，但新光纤优先级可能覆盖旧结石，最终结石/高置信原正确光纤不保证逐像素不变，须审计损失。

## 为什么先检验输入响应

1209帧的可见暗段在原模型中已低P，原局部T2-P作用域不能接触大部分错误。该三臂先判断同一网络能否从固定光度表达中恢复暗段证据，以及亮斑/腔壁FP、结石和正确像素的代价。若只增Recall却损害其他质量则拒绝全局变换；若固定变换无效，只否定这两个变换，不否定全部亮度假设。

当前有标签Dataset实际路径仅做同步水平/竖直翻转；无标签strong路径才调用亮度/对比度/色彩与模糊；验证不做这些增强。supervised_trainer和trainer的调用已核对，不能根据函数名假定有标签也有光度增强，也不从当前代码反推A0全部历史。此线索不等于已证明错误由训练增强不足造成，当前不训练。

杆体与暗头不同外观、内部明暗过渡与真正末端、几何搜索范围均作为后续独立问题。本轮不把新概率送入T1改状态，不自动延长头部或估计物理长度。

## 参考与验收

70视频+132图像、原29及现有分组保持。原图Lab/HSV固定亮度分组，不按变换后输入重新分组。低/中Lab组召回不能退化，原P<0.1的FN恢复与新增FP共同报告；原29/70/132/classwise和同轮raw2质量门槛不放宽，另有限制粗删除TP和逐帧损失。暗目标通过还要求70、132和固定1209帧低亮组FN严格下降，不仅看成功示例。

现有12帧尖端均pending，14帧计划工作单保留12帧并追加78/1209，没有生成标签或改工作台。沿用D1可见尖端/杆体/非光纤/未知及独立ROI和末端点规则。没有确认参考时可先开展整帧/亮度组对照，尖端Recall、物理端点误差保持null，不要求重新审核全部训练数据。

未来实际运行每臂202张，606次配对帧推理；每臂分别完整测试722/1972帧，6新进程8082正式帧，另报12帧预热。即使静态或成本失败也测完整两片。变换p95≤5ms是设计目标，总管线仍直接检验源帧预算，同时记录真实含导出吞吐。不用缓存评估冒充速度测试，不相减p95。

完整视频只有70帧参考处能报精确指标；42/120抽样未齐，不能外推全片准确率。默认A0、神经权重、原521/132/144/29和70标签不动，视频/派生物与stone_new不训练，训练/144保留test像素读取0。任何训练仍需单独计划与批准。

下一步实施固定三臂、机制测试和完整对照，不增加gamma搜索、局部融合或额外网络。T2-P原协议保留未执行，结石记忆未启动。

- [完整实施协议](../plans/fiber_dark_input_control_20261030.md)、[机器配置](../../configs/fiber_dark_input_control_20261030.json)
- [14帧计划工作单（不是标签）](../../runs/protocols/fiber_dark_input_control_20261030_v1/tip_reference_worklist.json)
- [前期暗段证据](2026-10-30-fiber-dark-appearance-observation.md)
'''
    (ROOT/PROGRESS).write_text(report,encoding='utf-8')
    paths=[CFG,DOC,PROGRESS,'ureter_semis/data.py','ureter_semis/trainer.py','ureter_semis/supervised_trainer.py',
        'tools/member_experts.py','tools/classwise_experts.py','tools/replay_classwise_experts.py','configs/compare_two_network_experts_20261029_v2.json',
        'runs/a1_fiber_dark_appearance_diagnosis_20261030/summary.json','runs/a1_fiber_dark_appearance_diagnosis_20261030/identity.json',
        'runs/protocols/fiber_dark_input_control_20261030_v1/tip_reference_worklist.json']
    files={p:sha256(ROOT/p) for p in paths}
    with zipfile.ZipFile(OUT/'protocol_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(ROOT/p,p)
    with zipfile.ZipFile(OUT/'protocol_sources.zip') as z:checks['archive_roundtrip_exact']=all(z.read(p)==(ROOT/p).read_bytes() for p in files)
    for p in [ROOT/DOC,ROOT/PROGRESS]:checks[p.name+'_relative_links']=all((p.parent/link).resolve().exists() for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')))
    assert all(checks.values())
    atomic_json(OUT/'identity.json',{'status':'protocol_only_not_implemented_not_executed','files':files,'weights':pre['checkpoints'],
        'manifests':{config['data'][k]:sha256(ROOT/config['data'][k]) for k in ['video_manifest','image_manifest']},
        'frozen_base_preflight':str(ROOT/'runs/a1_fiber_temporal_observation_search_20261030/preflight.json'),
        'frozen_base_preflight_sha256':sha256(ROOT/'runs/a1_fiber_temporal_observation_search_20261030/preflight.json'),
        'archive_sha256':sha256(OUT/'protocol_sources.zip'),'planned_full_processes':6,'planned_full_paired_frame_inferences':8082,
        'planned_static_paired_frame_inferences':606,'actual_new_model_forwards':0,'actual_transformed_inputs':0,'new_reference_masks':0,
        'training_updates':0,'training_pixels_read':0,'reserved_test_pixels_read':0})
    atomic_json(OUT/'engineering_checks.json',{'passed':sum(checks.values()),'checks':checks,'scope':'protocol consistency and existing identities, NOT transform unit tests or experimental results'})
    p=ROOT/'docs/plans/segmentation_temporal_occlusion_plan.md';text=p.read_text(encoding='utf-8');assert '版本：v2.20' in text;lines=[]
    for line in text.replace('版本：v2.20','版本：v2.21',1).splitlines():
        if line.startswith('| S2 |'):line='| S2 | 暗段固定光度输入三臂协议与14帧待确认工作单已制定；当前监督/无监督增强路径核对完成，尚无新输入变换/前向/修正 | 下一步实现raw2/共同RGB缩放/逐通道gamma0.7，606静态配对帧+6进程8082完整帧；暗组/两类质量/TP/成本共同验收，不扫参数、不训练或默认采用/S3 |'
        elif line.startswith('- [ ] **当前下一步：暗端低概率与杆头外观专项协议**'):
            line='- [x] **暗端专项第一阶段协议（固定输入响应）**：三臂固定gamma0.7、光纤输入变换而结石输入不变；明确同V不等于同Lab亮度，不自动融合/几何填充。14帧计划工作单仅待确认、无真值生成；当前有标签/无标签实际增强路径已核对。方案已冻结，未实现或执行。详见[协议](fiber_dark_input_control_20261030.md)。\n- [ ] **当前下一步：暗端固定输入三臂实现与完整对照**：新增独立变换/运行入口/机制测试，raw2精确复现，结石概率保持，最终结石/原正确光纤损失需审计；606静态配对帧、三臂各完整2694帧、6新进程8082帧，失败不免测。低/中亮组保护、原质量门槛及TP损失/总耗时门槛不放宽；无尖端真值不报尖端准确率。只运行一个gamma，不自动局部杆头融合、训练、默认替换或S3。'
        lines.append(line)
    addition='2026-10-30 暗段固定输入协议更新（覆盖以上历史下一步）：已冻结raw2、光纤RGB共同缩放V提亮、光纤逐通道gamma三臂，固定gamma0.7，均两网一次前向，结石输入/概率不变但最终掩膜可因光纤覆盖改变；同V不等于同Lab亮度，必须同时看暗组恢复、组织FP和结石损失。现12帧尖端均pending，另78/1209仅列14帧计划工作单，不改标签。当前有标签只走几何翻转、无标签strong才走光度路径已核对，未据此自动训练。下一步606静态与6进程8082完整帧固定对照；本步变换/新前向/训练0，默认与T2-P原协议不变。'
    p.write_text('\n'.join(lines)+'\n\n'+addition+'[本步记录](../progress/2026-10-30-fiber-dark-input-protocol.md)。\n',encoding='utf-8')
    p=ROOT/'README.md';text=p.read_text(encoding='utf-8');assert text.count('\n## 实现说明')==1
    p.write_text(text.replace('\n## 实现说明','\n'+addition+'[协议与状态](docs/progress/2026-10-30-fiber-dark-input-protocol.md)。\n\n## 实现说明',1),encoding='utf-8')
    p=ROOT/'annotations/data_review_status_20261027.json';status=read(p)
    status.update(fiber_dark_input_plan_prepared=True,fiber_dark_input_implemented=False,fiber_dark_input_executed=False,fiber_dark_input_adopted=False,
        fiber_dark_input_plan=DOC,fiber_dark_input_config=CFG,fiber_dark_input_plan_identity=str(OUT/'identity.json'),
        fiber_dark_input_tip_worklist=str(OUT/'tip_reference_worklist.json'),fiber_dark_input_tip_worklist_is_truth=False,
        fiber_dark_input_plan_report=PROGRESS,current_plan_checks=str(OUT/'engineering_checks.json'),
        latest_instruction='Continue next plan step: freeze dark-distal current-input response protocol',
        latest_instruction_effect='Fixed3arm input-response protocol and14pending tip-worklist only; no transform/new network forward/mask/training/default/S3. Current actual labeled/unlabeled augmentation paths checked.',
        execution_order_confirmation='Next implement frozen three-arm dark-input experiment,606static paired frames and6full-process8082frames. No parameter sweep or automatic training/fusion.',
        remaining_prerequisites=[
            'Next implement/test frozen dark-input3arms;606static paired inferences then each arm both complete videos,6fresh processes8082frames even quality/runtime failure',
            'Raw2 exact native reproduction, original stone probabilities identical; final stone/high-confidence fiber pixels can change and losses must be reported/gated',
            'Dark Lab groups defined from original RGB, no TP regression; overall/classwise/original29/TP/runtime gates unchanged. No gamma selection or image-dependent best-arm.',
            '12tip pilot pending+2planned extensions are not labels;70/132 semantic study can proceed, tip metrics remain null without independent review',
            'T2P retained unimplemented local control; no automatic local shaft/front model, temporal fill, training/default/stoneS3; videos validation only, stone_new excluded,521train/144test pixels excluded'
        ])
    atomic_json(p,status)
    registry['previous_plan_before_dark_input']={k:v for k,v in registry.items() if k.startswith('latest_plan_')}
    registry.update(latest_plan_config=CFG,latest_plan_document=DOC,latest_plan_identity=str(OUT/'identity.json'),latest_plan_executed=False,
        current_next_step='implement_fixed_dark_input_three_arm_control',tip_reference_planning_worklist=str(OUT/'tip_reference_worklist.json'),tip_reference_planning_worklist_is_truth=False)
    assert {k:v for k,v in registry.items() if k.startswith('latest_state_control_') or k.startswith('latest_inference_control')}==actual
    atomic_json(registry_path,registry)
    atomic_json(OUT/'registration.json',{'registrar_sha256':sha256(Path(__file__)),'main_plan_version':'v2.21','protocol_checks':sum(checks.values()),
        'actual_new_model_forwards':0,'actual_transformed_inputs':0,'training_updates':0,'default_changed':False,'worklist_is_truth':False})
    print(json.dumps({'protocol':'ready_not_executed','main_plan':'v2.21','checks':sum(checks.values()),'planned_tip_tasks':14,'new_forwards':0}))


if __name__=='__main__':main()
