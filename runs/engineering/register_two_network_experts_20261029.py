"""Update current status only; original labels, splits and run outputs stay frozen."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.fiber_reference_workspace import read,sha256
from ureter_semis.experiment import atomic_json

out=ROOT/'runs/a1_two_network_experts_20261029'
result=read(out/'summary.json'); audit=read(out/'engineering_checks.json')
assert read(out/'job.json')['phase']=='completed' and audit['passed']==76
status_path=ROOT/'annotations/data_review_status_20261027.json';status=read(status_path)
status.update({
    'latest_instruction':'Continue next step of existing plan; complete both videos per actual comparison scheme',
    'latest_instruction_effect':'Fixed A0 model1/model2 two-network ablation completed:70/132 fresh static plus full2694frames per scheme, matched3net baseline; no new approved labels. Both legacy29 screening passed, strict current nonregression failed. model1 retained lower-cost candidate, no adoption.',
    'execution_order_confirmation':'Finite member ablation completed; next read-only fiber-specific lightweight refinement feasibility, no automatic training/default replacement/S3',
    'remaining_prerequisites':[
        'Next: read-only fiber-specific refinement feasibility using model1 error distribution and frozen3net reference; automatic ROI coverage, outside errors, stone protection and cost before candidate implementation',
        'Member search finished: both legacy29 gates passed but strict70/132 nonregression failed; retain model1 cost-quality candidate, no automatic adoption or threshold/member search',
        'Blue A+ and uniform boost rejected; preserve all results and frozen thresholds',
        'Dedicated12 tip references pending; no anatomical tip metrics from semantic/axis proxies',
        'Whole-video probability sample42of120 incomplete; active70 development references usable, no full-video accuracy extrapolation',
        'Every actual comparison arm requires both complete videos even after static rejection; no144 reserved-test pixels, stone_new excluded, all video derivatives validation only',
        'Training, photometric augmentation or new tip head requires separate approved plan; no automatic S3 or physical tip/laser acceptance'
    ],
    'current_execution_checks':str(out/'engineering_checks.json'),
    'last_executed_diagnostic_config':'configs/compare_two_network_experts_20261029_v2.json',
    'two_network_expert_control_executed':True,
    'two_network_expert_control_adopted':False,
    'two_network_expert_control_deferred':False,
    'two_network_expert_control_deferred_reason':None,
    'two_network_expert_control_report':'docs/progress/2026-10-29-two-network-expert-control.md',
    'two_network_expert_control_summary':str(out/'summary.json'),
    'two_network_expert_control_engineering_checks':76,
    'two_network_expert_control_new_tests':3,
    'two_network_expert_control_regression_tests':3,
    'two_network_expert_control_full_frames_per_scheme':2694,
    'two_network_expert_control_full_frames_all_schemes':8082,
    'two_network_expert_control_legacy29_passed':{'model1':True,'model2':True},
    'two_network_expert_control_current_nonregression_passed':{'model1':False,'model2':False},
    'two_network_expert_control_next':'read_only_fiber_specific_refinement_feasibility',
    'two_network_expert_control_low_cost_candidate':'member1_2net_not_adopted',
    'two_network_expert_control_training_updates':0,
    's3_started':False,'stone_memory_started':False
})
atomic_json(status_path,status)
registry_path=ROOT/'annotations/video_evaluation_registry.json';registry=read(registry_path)
registry.update(latest_inference_control='configs/compare_two_network_experts_20261029_v2.json',latest_inference_control_result=str(out/'summary.json'),latest_inference_control_result_sha256=sha256(out/'summary.json'),latest_control_adopted=False)
atomic_json(registry_path,registry)
addition='2026-10-29 固定两网成员最新更新（覆盖以上历史“下一步”）：三网及model1/model2两网均完成70/132新推理和各2694帧完整测试，6进程8082帧，76项核验通过。model1光纤Dice为视频0.945537、图像0.765758，高于三网0.939825/0.762334；但视频Precision、图像Recall和部分结石指标退化，保留低成本候选。两者均通过原29初筛，均未通过另行预冻结的各项不退化检查，未替换默认。同轮三网P95为21.81/22.97ms，model1为17.70/18.79ms；仅作本轮匹配比较。当前结束成员搜索，下一步只读分析光纤专用轻量精修的自动ROI覆盖、错误分布、结石保护及成本，再冻结候选；不自动训练或进入S3。70帧不外推全片精度，尖端待补。'
plan=ROOT/'docs/plans/segmentation_temporal_occlusion_plan.md'; text=plan.read_text(encoding='utf-8')
assert addition not in text
plan.write_text(text+'\n\n'+addition+'[本轮结果](../progress/2026-10-29-two-network-expert-control.md)。\n',encoding='utf-8')
readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8');marker='\n## 实现说明'
assert marker in text and addition not in text
readme.write_text(text.replace(marker,'\n'+addition+'[结果与完整对比视频](docs/progress/2026-10-29-two-network-expert-control.md)。\n'+marker,1),encoding='utf-8')
assert sha256(ROOT/registry['active_video_test_manifest'])==registry['active_video_test_manifest_sha256']
assert result['training_updates']==0 and result['adopted'] is False
print(json.dumps({'updated':['README','plan_v2.13','status','registry'],'labels_changed':False,'default_changed':False}))
