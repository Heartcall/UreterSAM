"""Check and register the immutable 36-frame supplement, preserving original evaluation gates."""
import csv
import json
from pathlib import Path

import numpy as np

from tools.fiber_reference_workspace import ROOT, read, sha256
from ureter_semis.experiment import atomic_json

folder = ROOT / 'runs/review_snapshots/video_development_supplement_20261029_v1'
manifest = read(folder / 'manifest.json'); rows = manifest['val']
summary = read(folder / 'comparison/summary.json')
new = [r for r in rows if r['evaluation_cohort'] == 'new_user_approved']
old = [r for r in rows if r['evaluation_cohort'] == 'original29']
checks = []
def check(name, passed):
    checks.append({'name': name, 'passed': bool(passed)})
    if not passed: raise AssertionError(name)

check('36_new_plus_original29_equals65_unique_frames', len(new) == 36 and len(old) == 29 and len(rows) == 65 and len({r['id'] for r in rows}) == 65)
check('new23_13_and_combined43_22_by_video', [sum(r['video_id'] == v for r in new) for v in ('video_1', 'video_2')] == [23, 13] and [sum(r['video_id'] == v for r in rows) for v in ('video_1', 'video_2')] == [43, 22])
check('snapshot_and_original_source_image_mask_hashes_match', all(sha256(r[f]) == r[f + '_sha256'] and sha256(r['source_' + f]) == r[f + '_sha256'] for r in rows for f in ('image', 'mask')))
check('frozen_session_is_exact_atomic_capture', sha256(folder / 'session_at_freeze.json') == manifest['source_session_sha256'])
check('live_session_was_not_changed_by_import', sha256(manifest['source_session']) == manifest['source_session_sha256'])
check('sources_unchanged_after_evaluation', all(sha256(ROOT / p) == h for p, h in manifest['sources'].items()))
check('original521_132_144_split_manifest_unchanged', sha256(manifest['original_training_split_manifest']) == manifest['original_training_split_manifest_sha256'])
check('reference_manifest_is_validation_only_and_not_training', manifest['training_ready'] is False and not manifest['train']['labeled'] and not manifest['train']['unlabeled'] and all(r['training_allowed'] is False for r in rows))
check('only_approved_visible_pixels_added', all(r['review_status'] == 'approved' and r['reviewer'] and r['annotation_basis'] == 'human_visible_pixels' for r in new))
check('37_of120_sampled_frames_no_completed_design', sum(c['approved_random_sample'] for c in manifest['coverage'].values()) == 37)
check('no_semantic_to_anatomical_tip_promotion', manifest['tip_reference_promoted_from_semantic'] is False and manifest['tip_metrics'] is None)
check('11_schemes_complete_all2694_and_original29_exact', len(summary['models']) == 11 and all(m['all_full_video_cached_outputs_reverified'] == 2694 and m['changed_output_files'] == 0 for m in summary['models'].values()) and all(c['historical29_exact'] for c in summary['checks']))
for name, model in summary['models'].items():
    original = np.array(model['original29']['metrics']['confusion_matrix'])
    added = np.array(model['new_approved']['metrics']['confusion_matrix'])
    total = np.array(model['combined_approved']['metrics']['confusion_matrix'])
    cases = sum(np.array(c['combined_approved']['metrics']['confusion_matrix']) for c in model['cases'].values())
    check(name + '_confusion_conserved_across_cohorts_and_cases', np.array_equal(original + added, total) and np.array_equal(cases, total))
check('no_full_video_or_tip_accuracy_from_incomplete_reference', all(m['dense_exact_accuracy'] is None and m['tip_metrics'] is None and m['stratified_pooled_estimate']['metrics'] is None and all(c['stratified_whole_video_estimate']['metrics'] is None for c in m['cases'].values()) for m in summary['models'].values()))

registry_path = ROOT / 'annotations/video_evaluation_registry.json'
if registry_path.exists(): raise FileExistsError(registry_path)
registry = {'schema_version': 1, 'version': 'video_development65_20261029_v1',
    'role': 'active_video_test_is_repeated_development_validation_not_independent_final_test',
    'user_authorization': 'User selected and annotated frames, requested adding them to the test set on2026-10-29',
    'active_video_test_manifest': str(folder / 'manifest.json'), 'active_video_test_manifest_sha256': sha256(folder / 'manifest.json'),
    'active_video_test_frames': 65, 'additional_frames': 36, 'additional_manifest': str(folder / 'new_frames_manifest.json'),
    'additional_manifest_sha256': sha256(folder / 'new_frames_manifest.json'),
    'original29_regression_manifest': str(ROOT / 'datasets/ureter_semis_kidney_joint_20261028/video_validation_manifest.json'),
    'original29_regression_manifest_sha256': sha256(ROOT / 'datasets/ureter_semis_kidney_joint_20261028/video_validation_manifest.json'),
    'original_image132_and_train521_manifest': manifest['original_training_split_manifest'],
    'original_image132_and_train521_manifest_sha256': manifest['original_training_split_manifest_sha256'],
    'evaluation_result': str(folder / 'comparison/summary.json'), 'evaluation_result_sha256': sha256(folder / 'comparison/summary.json'),
    'whole_video_probability_sample_completed': 37, 'whole_video_probability_sample_planned': 120,
    'whole_video_estimate_available': False, 'tip_metrics_available': False,
    'policy': 'Use active65 for additional static development tests; report original29/new36/combined65 separately. Original29/132 adoption gates unchanged. Do not require complete120 to add approved frames, but no whole-video estimate from this partial selection. Videos and derivatives never enter training;144 reserved-test groups remain untouched.'}
atomic_json(registry_path, registry)
check('active65_and_new36_registry_hashes_match', sha256(registry['active_video_test_manifest']) == registry['active_video_test_manifest_sha256'] and sha256(registry['additional_manifest']) == registry['additional_manifest_sha256'])
atomic_json(folder / 'engineering_checks.json', {'passed': True, 'checks': checks, 'count': len(checks),
    'new_frames': 36, 'combined_frames': 65, 'all_prediction_identities_verified': 29634,
    'live_session_changed': False, 'training_updates': 0, 'reserved_test_pixels_read': 0})
with (folder / 'frame_inventory.csv').open('w', encoding='utf-8-sig', newline='') as handle:
    fields = ['video_id', 'frame_index', 'one_based_frame', 'evaluation_cohort', 'reviewer', 'source_revision', 'unknown_pixels', 'image', 'mask']
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
lines = ['# 新增视频标注测试结果（2026-10-29）', '',
    '36帧新增人工标注已加入当前视频测试（开发验证）集，合计65帧；原29帧单独保留。该版本仅作验证，不加入训练。', '',
    '| 视频 | 原参考 | 新增参考 | 当前合计 | 固定60帧样本已完成 |', '|---|---:|---:|---:|---:|']
for vid, c in manifest['coverage'].items():
    lines.append(f'| {vid} | {c["original29"]} | {c["new_approved"]} | {c["combined"]} | {c["approved_random_sample"]}/60 |')
lines += ['', '新增36帧含未知像素'+str(sum(r['unknown_pixels'] for r in new))+'个，统计时排除；合计65帧未知'+str(sum(r['unknown_pixels'] for r in rows))+'个。', '',
    '以下结果仅针对已审核帧，不能视为全片精度。37/120的部分样本不用于全片加权估计；两个既有开发病例也不能改称独立最终测试。', '',
    '| 方案 | 新36帧结石Dice | 新36帧光纤Dice | 合计65帧结石Dice | 合计65帧光纤Dice |', '|---|---:|---:|---:|---:|']
for name, m in summary['models'].items():
    a, b = m['new_approved']['metrics'], m['combined_approved']['metrics']
    lines.append(f'| {name} | {a["stone"]["dice"]:.6f} | {a["fiber"]["dice"]:.6f} | {b["stone"]["dice"]:.6f} | {b["fiber"]["dice"]:.6f} |')
lines += ['', '## Classwise逐类指标', '', '| 参考范围 | 类别 | Dice | Precision | Recall | TP | FP | FN |', '|---|---|---:|---:|---:|---:|---:|---:|']
for scope, label in [('original29', '原29帧'), ('new_approved', '新增36帧'), ('combined_approved', '合计65帧')]:
    for c in ('stone', 'fiber'):
        r = summary['models']['classwise'][scope]['metrics'][c]
        lines.append(f'| {label} | {c} | {r["dice"]:.6f} | {r["precision"]:.6f} | {r["recall"]:.6f} | {int(r["tp"])} | {int(r["fp"])} | {int(r["fn"])} |')
lines += ['', '原29帧所有11方案的指标均与前轮精确一致。11方案各2694帧缓存文件身份重新核验通过，共29634条；本轮未重新推理或计时，不能作为新的速度比较。逐帧混淆矩阵、3像素边界F、正确像素删除/恢复及误检变化见comparison目录。', '',
    '已冻结session_at_freeze.json与对应审核版本，复制65帧原图/掩膜并记录哈希，不修改在线会话、原标签、旧输出或521/132/144/29划分。新增3项针对性测试及4项既有参考测试通过；'+str(len(checks))+'项导入/评测核验通过。', '',
    '新标注只有整帧语义，尚未单独确认尖端/杆体身份；12帧尖端专项仍待确认，不能从整根光纤掩膜自动生成尖端真值。没有训练、改变默认模型或启动结石记忆。', '',
    '当前测试入口登记在annotations/video_evaluation_registry.json；manifest.json为65帧总清单，new_frames_manifest.json为36帧新增清单。以此冻结版本继续静态开发对照，并分别报告原29/新增36/合计65，不放宽既定29/132验收门槛。', '',
    '如继续补标，以新快照目录导入，不覆盖本轮参考或指标。完整120帧可用后，再按既定分层抽样协议估计全片精度。']
(folder / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({'registered': str(registry_path), 'counts': manifest['counts'], 'checks': len(checks), 'whole_video_estimates': None}))
