"""Register the five additional approved frames without replacing the65-frame history."""
from collections import Counter
from pathlib import Path
import json

import numpy as np

from tools.fiber_reference_workspace import ROOT, read, sha256
from ureter_semis.experiment import atomic_json

folder = ROOT / 'runs/review_snapshots/video_development_supplement_20261029_v2'
previous = ROOT / 'runs/review_snapshots/video_development_supplement_20261029_v1'
manifest = read(folder / 'manifest.json'); prior_manifest = read(previous / 'manifest.json')
summary = read(folder / 'comparison/summary.json'); previous_summary = read(previous / 'comparison/summary.json')
prior_rows = {r['id']: r for r in prior_manifest['val']}; rows = manifest['val']
delta = [r for r in rows if r['id'] not in prior_rows]
checks = []
def check(name, condition):
    checks.append({'name': name, 'passed': bool(condition)})
    if not condition: raise AssertionError(name)

check('70_unique_frames_include_previous65_and_five_new', len(rows) == 70 and len(delta) == 5 and len({r['id'] for r in rows}) == 70 and set(prior_rows) <= {r['id'] for r in rows})
check('previous65_labels_unchanged', all(r['mask_sha256'] == prior_rows[r['id']]['mask_sha256'] and r['image_sha256'] == prior_rows[r['id']]['image_sha256'] for r in rows if r['id'] in prior_rows))
check('five_new_frames_are_video2_and_approved', all(r['video_id'] == 'video_2' and r['review_status'] == 'approved' for r in delta))
check('all_snapshot_and_source_hashes_match', all(sha256(r[k]) == r[k + '_sha256'] and sha256(r['source_' + k]) == r[k + '_sha256'] for r in rows for k in ('image', 'mask')))
check('online_session_unchanged_by_capture', sha256(manifest['source_session']) == manifest['source_session_sha256'])
check('training_split_unchanged_no_train_rows', sha256(manifest['original_training_split_manifest']) == manifest['original_training_split_manifest_sha256'] and not manifest['train']['labeled'] and not manifest['train']['unlabeled'])
check('42of120_sample_references_incomplete', sum(c['approved_random_sample'] for c in manifest['coverage'].values()) == 42)
for name, model in summary['models'].items():
    frames = read(folder / f'comparison/{name}.json')['frames']
    old_matrix = sum(np.array(r['confusion_matrix']) for r in frames if r['id'] in prior_rows)
    check(name + '_previous65_accuracy_exact', np.array_equal(old_matrix, previous_summary['models'][name]['combined_approved']['metrics']['confusion_matrix']))
    check(name + '_full2694_cached_outputs_unchanged', model['all_full_video_cached_outputs_reverified'] == 2694 and model['changed_output_files'] == 0)
    check(name + '_no_unlabelled_full_or_tip_accuracy', model['stratified_pooled_estimate']['metrics'] is None and model['dense_exact_accuracy'] is None and model['tip_metrics'] is None)

registry_path = ROOT / 'annotations/video_evaluation_registry.json'
registry_bytes = registry_path.read_bytes(); registry = json.loads(registry_bytes)
if registry['active_video_test_frames'] != 65: raise ValueError('Unexpected active evaluation version; inspect before updating')
(folder / 'previous_active_registry.json').write_bytes(registry_bytes)
registry.update(version='video_development70_20261029_v2', previous_active_manifest=registry['active_video_test_manifest'],
    active_video_test_manifest=str(folder / 'manifest.json'), active_video_test_manifest_sha256=sha256(folder / 'manifest.json'),
    active_video_test_frames=70, additional_frames=41, latest_incremental_frames=5,
    additional_manifest=str(folder / 'new_frames_manifest.json'), additional_manifest_sha256=sha256(folder / 'new_frames_manifest.json'),
    evaluation_result=str(folder / 'comparison/summary.json'), evaluation_result_sha256=sha256(folder / 'comparison/summary.json'),
    whole_video_probability_sample_completed=42,
    policy='Active70 video development reference; report original29/cumulative41/latest5/combined70 separately. Previous65 frozen version retained. Original29/132 adoption gates unchanged.42of120 partial labels do not establish full-video accuracy. Video derivatives never train;144 reserved-test pixels excluded.')
atomic_json(registry_path, registry)
atomic_json(folder / 'latest_five_manifest.json', {**manifest, 'role': 'latest_five_video_development_reference_only', 'val': delta,
    'counts': {'latest_incremental_frames': len(delta)}, 'previous_manifest_sha256': sha256(previous / 'manifest.json')})
atomic_json(folder / 'engineering_checks.json', {'passed': True, 'count': len(checks), 'checks': checks,
    'new_this_turn': 5, 'cumulative_additional_frames': 41, 'combined_frames': 70, 'full_output_identity_records': 29634,
    'previous65_unchanged': True, 'live_session_changed': False, 'training_updates': 0, 'reserved_test_pixels_read': 0})
lines = ['# 第二批新增视频标注入集（2026-10-29）', '',
    '本次新增5帧均来自video_2，当前视频测试开发集70帧：video_1为43帧，video_2为27帧。累计新增41帧，原29帧另列，上一版65帧完整保留。', '',
    '本次新增帧号（从1开始）：'+ '、'.join(str(r['one_based_frame']) for r in delta)+'。', '',
    '| 方案 | 合计70帧结石Dice | 合计70帧光纤Dice |', '|---|---:|---:|']
for name, m in summary['models'].items():
    q = m['combined_approved']['metrics']; lines.append(f'| {name} | {q["stone"]["dice"]:.6f} | {q["fiber"]["dice"]:.6f} |')
lines += ['', '11方案的上一版65帧结果完全一致，原29帧结果也精确复现；各2694帧缓存输出身份不变，共29634条。'+str(len(checks))+'项核验通过。没有训练、默认改变或旧标注覆盖。', '',
    '固定分层样本完成42/120（video_1 24/60，video_2 18/60）。这里只报告已审帧，尚不发布全片精度估计；12帧尖端专项未标，不能从整根光纤语义生成尖端真值。用户已明确允许用现有整帧参考继续A+受控对照，尖端指标待补。']
(folder / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({'active': 70, 'latest_added': 5, 'sampled': 42, 'checks': len(checks)}))
