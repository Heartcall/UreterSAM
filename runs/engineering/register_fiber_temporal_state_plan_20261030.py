"""Validate/register a protocol only; no model construction or pixel reads."""
import hashlib,json,re,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.fiber_reference_workspace import read,sha256
from ureter_semis.experiment import atomic_json

def main():
    cfg_path=ROOT/'configs/fiber_temporal_state_gate_20261030.json';doc_path=ROOT/'docs/plans/fiber_temporal_state_gate_20261030.md';cfg=read(cfg_path)
    legacy=read(ROOT/'configs/fiber_geometry_prior_20261028.json');prior=read(ROOT/'runs/a1_two_network_experts_20261029/preflight.json')
    registry=read(ROOT/'annotations/video_evaluation_registry.json');frozen_manifest=registry['active_video_test_manifest'];manifest_hash=sha256(frozen_manifest)
    status_path=ROOT/'annotations/data_review_status_20261027.json';status=read(status_path)
    before={str(p):sha256(p) for p in [ROOT/k for k in prior['sources']]}
    checks={}
    def check(name,value):
        checks[name]=bool(value)
        if not value:raise AssertionError(name)
    check('protocol_not_executed',cfg['status']=='READY_FOR_IMPLEMENTATION_NOT_EXECUTED')
    check('output_absent',not (ROOT/cfg['planned_output']).exists())
    check('future_implementation_absent',all(not (ROOT/p).exists() for p in ['tools/fiber_temporal_state.py','tools/run_fiber_temporal_state_control.py','tests/test_fiber_temporal_state.py']))
    check('four_fixed_arms',cfg['arms']==['raw2','current_observation','legacy_state_only','gated_state_only'])
    check('all_arms_complete_videos',cfg['execution']['all_arms_full_replay_even_failure'] and cfg['execution']['frames']==[722,1972])
    check('process_frame_counts',cfg['execution']['expected_processes']==len(cfg['arms'])*2 and cfg['execution']['expected_total_full_frame_forwards']==len(cfg['arms'])*(722+1972))
    check('base_budget',cfg['base']['networks']==2 and cfg['base']['parameters']==49202790)
    check('alpha_beta_unchanged',all(cfg['state'][k]==legacy['tracking'][k] for k in ['alpha','beta','maximum_prediction_only_frames']))
    check('tip_limit_unchanged',cfg['innovation_limits']['tip_jump_diagonal_fraction']==legacy['tracking']['reset_tip_jump_diagonal_fraction'])
    check('length_limit_unchanged',cfg['innovation_limits']['length_jump_diagonal_fraction']==legacy['tracking']['reset_length_jump_diagonal_fraction'])
    check('angle_limit_unchanged',cfg['innovation_limits']['angle_jump_degrees']==legacy['tracking']['reset_angle_jump_degrees'])
    check('width_limit_unchanged',cfg['innovation_limits']['width_ratio']==legacy['tracking']['reset_width_ratio'])
    check('current_fitter_frozen',cfg['observation']['frozen_fit_config']=='configs/fiber_geometry_prior_20261028.json:fit')
    check('confirmation_separate_from_old_reset',cfg['state']['confirmation_observations']==3)
    check('static_reference_counts',len(read(ROOT/cfg['data']['video_manifest'])['val'])==70 and len(read(ROOT/cfg['data']['image_manifest'])['val'])==132)
    check('active_reference_unchanged',manifest_hash==registry['active_video_test_manifest_sha256'])
    check('sample_truth_limits',cfg['data']['whole_video_sample_approved']==42 and cfg['data']['whole_video_sample_planned']==120 and cfg['data']['tip_truth_available'] is False)
    check('no_training_or_test_reads_planned',cfg['data']['training_pixels_read']==cfg['data']['reserved_test_pixels_read']==0)
    check('historical_frozen_sources_current',all(sha256(ROOT/p)==h for p,h in prior['sources'].items()))
    check('frozen_weights',all(sha256(v['path'])==v['sha256'] for v in prior['checkpoints'].values()))
    for doc in [doc_path,ROOT/'docs/plans/segmentation_temporal_occlusion_plan.md']:
        # Current main plan has historical links; only new design links are validated here.
        if doc==doc_path:
            check('design_document_links',all((doc.parent/target).exists() for target in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')) if not target.startswith(('http','#'))))
    check('plan_v2_15', 'v2.15' in (ROOT/'docs/plans/segmentation_temporal_occlusion_plan.md').read_text(encoding='utf-8'))
    out=ROOT/'runs/protocols/fiber_temporal_state_gate_20261030_v1'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    paths=[cfg_path,doc_path,ROOT/'configs/fiber_geometry_prior_20261028.json',ROOT/'tools/fiber_shape_prior.py',ROOT/'tools/run_fiber_shape_control.py',ROOT/'tools/member_experts.py',ROOT/'ureter_semis/video_metrics.py']
    identity={str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in paths}
    with zipfile.ZipFile(out/'protocol_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,p.relative_to(ROOT).as_posix())
    atomic_json(out/'identity.json',{'status':'protocol_only_not_executed','files':identity,'weights':prior['checkpoints'],'manifests':{str(ROOT/cfg['data'][k]):sha256(ROOT/cfg['data'][k]) for k in ['video_manifest','image_manifest']},'archive_sha256':sha256(out/'protocol_snapshot.zip'),'planned_processes':8,'planned_frames':10776,'actual_new_video_forwards':0,'training_updates':0,'training_pixels_read':0,'reserved_test_pixels_read':0})
    status.update({
        'latest_instruction':'Continue plan after user emphasized adjacent-frame fiber geometry continuity',
        'latest_instruction_effect':'Prepared T0 four-arm state-only protocol; current fitter and original masks frozen, old immediate-reset vs isolated-jump gate/3-observation relocation. Implementation and experiments not executed, no training/default adoption.',
        'execution_order_confirmation':'Next implement frozen T0 state-only control and8independent complete-video processes; T1 history-assisted candidate search is separate conditional protocol, training separate approval',
        'fiber_temporal_state_plan_prepared':True,'fiber_temporal_state_control_implemented':False,'fiber_temporal_state_control_executed':False,'fiber_temporal_state_control_adopted':False,
        'fiber_temporal_state_plan':'docs/plans/fiber_temporal_state_gate_20261030.md','fiber_temporal_state_config':'configs/fiber_temporal_state_gate_20261030.json','fiber_temporal_state_plan_identity':str(out/'identity.json'),
        'current_plan_checks':str(out/'engineering_checks.json'),
        'remaining_prerequisites':[
            'Next: implement T0 current_observation/legacy_state_only/gated_state_only plus raw2, without changing current raw masks or old fitter/tracker files',
            'Test isolated jump rejection, causal3-observation relocation, expiry/pending, angle/dt, source/shape/time reset, static-order independence and raw-mask protection before full replay',
            'All4arms complete both videos in8fresh processes10776frames even failure, matched cost and per-frame state/provenance; no jitter-as-accuracy claims',
            'T1 current-image candidate search from predicted pose requires separately frozen evidence/budget/ablation; not implemented in T0',
            'Current local-only ROI failed, model1 remains candidate; no threshold/cap/member search, automatic training/default/S3 or invented visible tip',
            '70development references usable,42of120 sample incomplete,12tip references pending; videos/derivatives validation only, stone_new excluded,144test pixels excluded',
            'Any training requires separately approved data/init/supervision/augmentation/budget/quality/full-video/rollback plan; no mixed A0 training initialization'
        ]})
    atomic_json(status_path,status)
    registry.update(latest_plan_config='configs/fiber_temporal_state_gate_20261030.json',latest_plan_document='docs/plans/fiber_temporal_state_gate_20261030.md',latest_plan_identity=str(out/'identity.json'),latest_plan_executed=False)
    atomic_json(ROOT/'annotations/video_evaluation_registry.json',registry)
    addition='2026-10-30 相邻帧几何连续性方案更新（覆盖以上历史下一步）：已完成T0状态对照设计和固定配置，尚未实现/实验。保留当前拟合器及旧alpha-beta/跳变门槛，仅比较立即重置与孤立异常暂存、连续3观测确认重定位；所有分割像素不变，历史只输出带年龄的几何状态。下一步实现四臂并完整测试两片（8独立进程10776帧），无真值不称尖端更准。历史辅助搜索T1单独冻结，不自动进入像素修正、训练、默认替换或结石S3。'
    p=ROOT/'docs/plans/segmentation_temporal_occlusion_plan.md';s=p.read_text(encoding='utf-8');assert addition not in s
    p.write_text(s+'\n\n'+addition+'[T0方案](fiber_temporal_state_gate_20261030.md)。\n',encoding='utf-8')
    p=ROOT/'README.md';s=p.read_text(encoding='utf-8');assert addition not in s and '\n## 实现说明' in s
    p.write_text(s.replace('\n## 实现说明','\n'+addition+'[设计与实施顺序](docs/plans/fiber_temporal_state_gate_20261030.md)。\n\n## 实现说明',1),encoding='utf-8')
    check('legacy_sources_unmodified_by_registration',all(sha256(p)==h for p,h in before.items()))
    check('labels_manifest_unmodified_by_registration',sha256(frozen_manifest)==manifest_hash)
    check('last_real_control_unchanged',registry['latest_inference_control']=='configs/compare_two_network_experts_20261029_v2.json' and status['last_executed_diagnostic_config']=='configs/fiber_refinement_feasibility_20261030.json')
    atomic_json(out/'engineering_checks.json',{'role':'protocol consistency checks, not unit tests or experiment validation','passed':sum(checks.values()),'checks':checks,'registrar_sha256':sha256(Path(__file__)),'new_model_forwards':0,'new_geometry_experiments':0})
    print(json.dumps({'status':'protocol_prepared_not_executed','checks':sum(checks.values()),'planned_full_frames':10776,'new_forward_frames':0}))

if __name__=='__main__':main()
