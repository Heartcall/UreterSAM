"""Record completion, without promoting a compute proxy to a segmentation method."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.fiber_reference_workspace import read,sha256
from ureter_semis.experiment import atomic_json
OUT=ROOT/'runs/a1_fiber_refinement_feasibility_20261030'
result=read(OUT/'summary.json');audit=read(OUT/'engineering_checks.json')
assert read(OUT/'job.json')['phase']=='completed' and audit['passed']==58
assert not any(result['coverage_and_proxy_cost_feasible'].values())
p=ROOT/'annotations/data_review_status_20261027.json';d=read(p)
d.update({
    'latest_instruction':'Continue next planned step: fiber-specific refinement feasibility',
    'latest_instruction_effect':'Completed70/132 fiber error/ROI/stone-protected oracle diagnosis and matched coarse/hard/soft compute proxies on each2694frames; both profiles fail image coverage and proxy latency. No refined masks, training/default change or S3.',
    'execution_order_confirmation':'Fiber-only bounded ROI feasibility completed and failed; next prepare global-fallback/empty-prediction/stone-protection study plan before any training request',
    'last_executed_diagnostic_config':'configs/fiber_refinement_feasibility_20261030.json',
    'current_execution_checks':str(OUT/'engineering_checks.json'),
    'fiber_specific_refinement_feasibility_executed':True,
    'fiber_specific_refinement_feasibility_passed':False,
    'fiber_specific_refinement_feasibility_report':'docs/progress/2026-10-30-fiber-refinement-feasibility.md',
    'fiber_specific_refinement_feasibility_summary':str(OUT/'summary.json'),
    'fiber_specific_refinement_feasibility_engineering_checks':58,
    'fiber_specific_refinement_feasibility_new_tests':4,
    'fiber_specific_refinement_feasibility_regression_tests':6,
    'fiber_specific_refinement_feasibility_full_frames_per_arm':2694,
    'fiber_specific_refinement_feasibility_full_frames_all_arms':8082,
    'fiber_specific_refinement_feasibility_training_updates':0,
    'fiber_specific_refinement_feasibility_new_segmentation_accuracy_available':False,
    'fiber_specific_refinement_feasibility_image_outside_fn_pixels':445,
    'fiber_specific_refinement_feasibility_image_gt_fiber_locked_by_stone':523,
    'fiber_specific_refinement_feasibility_video_gt_fiber_locked_by_stone':85,
    'fiber_specific_refinement_feasibility_hard_proxy_p95_ms':{vid:v['timings']['proxy_pipeline_ms']['p95'] for vid,v in result['full_video']['hard2'].items()},
    'fiber_specific_refinement_feasibility_soft_proxy_p95_ms':{vid:v['timings']['proxy_pipeline_ms']['p95'] for vid,v in result['full_video']['soft02_2'].items()},
    'remaining_prerequisites':[
        'Next: prepare fiber-specific full-frame fallback/empty-prediction/stone-protection comparison plan; distinguish FOV/scale, reference completeness and network errors before choosing training changes',
        'Current hard/soft local-only ROI training route stopped:445image FN outside even uncapped, zero-coverage frame; no cap/margin/threshold expansion to pass',
        'Heavy same-network compute proxies failed frame budget; future lightweight architecture cost/memory and actual crop accuracy not yet measured',
        'Any training needs separate approved data/init/augmentation/supervision/budget/quality/full-video/rollback plan; mixed A0 never new training initialization',
        'Keep model1 lower-cost candidate only, frozen3net reference and existing default; no extra member search or automatic S3',
        '70 approved development frames usable;42of120 probability sample incomplete, no whole-video accuracy extrapolation;12tip references pending',
        'All videos/derivatives validation only; stone_new excluded;144 reserved-test pixels excluded; no automatic label changes or renewed all-data manual review prerequisite'
    ]})
atomic_json(p,d)
p=ROOT/'annotations/video_evaluation_registry.json';d=read(p)
# Keep latest_inference_control pointing to the actual member segmentation comparison.
d.update(latest_feasibility_diagnostic='configs/fiber_refinement_feasibility_20261030.json',latest_feasibility_result=str(OUT/'summary.json'),latest_feasibility_result_sha256=sha256(OUT/'summary.json'),latest_feasibility_is_segmentation_accuracy=False)
atomic_json(p,d)
addition='2026-10-30 光纤专用ROI可行性最新更新（覆盖以上历史下一步）：hard/soft有界窗口已完成70/132覆盖/误差/结石保护上界，以及3臂各2694帧完整计算代理，58项核验通过。视频FN/FP全覆盖，但图像445个FN无种子落窗外，最低覆盖0，不限窗口也不改善；两组覆盖未通过。保护原结石还锁住85个视频、523个图像GT光纤像素。复用已有光纤网的裁剪计算代理P95为hard39.76/42.21ms、soft40.74/42.67ms，均超源帧预算；没有精修分割结果。停止本轮局部训练路线，不扩窗口/调阈值；下一步先制定含全图保底、空预测、结石保护冲突的光纤专项对照计划，区分视野/尺度/标签/网络，训练须另行批准。没有新训练、默认替换或S3。'
p=ROOT/'docs/plans/segmentation_temporal_occlusion_plan.md';s=p.read_text(encoding='utf-8');assert addition not in s
p.write_text(s+'\n\n'+addition+'[本轮结果](../progress/2026-10-30-fiber-refinement-feasibility.md)。\n',encoding='utf-8')
p=ROOT/'README.md';s=p.read_text(encoding='utf-8');assert '\n## 实现说明' in s and addition not in s
p.write_text(s.replace('\n## 实现说明','\n'+addition+'[报告与完整ROI诊断视频](docs/progress/2026-10-30-fiber-refinement-feasibility.md)。\n\n## 实现说明',1),encoding='utf-8')
assert sha256(d['active_video_test_manifest'])==d['active_video_test_manifest_sha256']
print(json.dumps({'plan':'v2.14','status':'feasibility_completed_failed_not_trained','label_changes':0,'default_changes':0}))
