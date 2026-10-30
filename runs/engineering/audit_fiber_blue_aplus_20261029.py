"""Independent pixel conservation, cache identity, gating and complete-video timing audit."""
from collections import Counter
import hashlib
from pathlib import Path
import zipfile

import cv2
import numpy as np

from tools.fiber_reference_workspace import ROOT, read, load_mask, sha256
from ureter_semis.experiment import atomic_json

out = ROOT / 'runs/a1_fiber_blue_aplus_20261029'
config = read(ROOT / 'configs/fiber_blue_aplus_20261029.json'); summary = read(out / 'summary.json')
pre = read(out / 'preflight.json'); names = config['schemes']; checks = []
def check(name, condition, detail=None):
    checks.append({'name': name, 'passed': bool(condition), 'detail': detail})
    if not condition: raise AssertionError(name)

check('job_completed_no_training_default_or_adoption', read(out / 'job.json')['phase'] == 'completed' and summary['training_updates'] == 0 and summary['default_changed'] is False and summary['adopted'] is False)
check('frozen_configuration_and_sources_unchanged', sha256(ROOT / 'configs/fiber_blue_aplus_20261029.json') == pre['config_sha256'] and all(sha256(ROOT / k) == v for k, v in pre['sources'].items()))
check('all_frozen_inputs_and_caches_unchanged', all(sha256(k) == v for k, v in pre['related'].items()))
check('weights_and_videos_match_frozen_identity', all(sha256(x['path']) == x['sha256'] for x in [*pre['checkpoints'].values(), *pre['videos']]))
with zipfile.ZipFile(out / 'source_snapshot.zip') as archive:
    check('source_archive_unique_and_exact', len(archive.namelist()) == len(set(archive.namelist())) and all(hashlib.sha256(archive.read(k)).hexdigest() == v for k, v in pre['sources'].items()))

totals = {}; sample_count = 0
for scope, manifest_path in [('image_native', config['image_manifest']), ('video_native', config['video_manifest'])]:
    rows = read(ROOT / manifest_path)['val']; result = read(out / scope / 'summary.json')
    matrices = {n: np.zeros((3, 3), np.int64) for n in names}
    delta = {n: Counter() for n in names[1:]}; applied = {n: Counter() for n in names[1:]}
    for i, row in enumerate(rows):
        with __import__('PIL.Image', fromlist=['Image']).open(row['mask']) as image: target = np.asarray(image).copy()
        raw = load_mask(out / f'{scope}/raw/{i:04d}.png', target.shape, {0, 1, 2}); valid = target != 255
        for name in names:
            pred = load_mask(out / f'{scope}/{name}/{i:04d}.png', target.shape, {0, 1, 2})
            matrices[name] += np.bincount((target[valid].astype(int) * 3 + pred[valid]).ravel(), minlength=9).reshape(3, 3)
            if name == 'raw': continue
            if np.any(pred[raw == 2] != 2) or np.any((pred == 1) != (raw == 1)):
                raise AssertionError('Protected static labels changed')
            extra = (pred == 2) & (raw != 2)
            current = {'lost_tp': 0, 'removed_fp': 0, 'gained_tp': int((extra & (target == 2)).sum()),
                       'added_fp': int((extra & (target != 2) & valid).sum())}
            if current != result['records'][i]['changes'][name]: raise AssertionError('Static paired counts differ')
            application = result['records'][i]['application'][name]
            if int(extra.sum()) != application['added_pixels']: raise AssertionError('Actual additions differ from log')
            if application['fallback'] == 'addition_budget' and application['proposed_pixels'] <= application['budget_pixels']:
                raise AssertionError('Budget fallback not justified')
            delta[name].update(current); applied[name][application['fallback'] or 'applied'] += 1
        sample_count += 1
    for name in names:
        m = result['models'][name]['metrics']
        check(scope + '_' + name + '_pixel_metrics_exact', np.array_equal(matrices[name], m['confusion_matrix']))
    for name in names[1:]:
        check(scope + '_' + name + '_zero_TP_deletion_and_stone_change', delta[name]['lost_tp'] == 0 and all(result['models'][name]['metrics'][k] == result['models']['raw']['metrics'][k] for k in ('stone_tp', 'stone_fp', 'stone_fn')))
    totals[scope] = {'paired': {k: dict(v) for k, v in delta.items()}, 'fallback_counts': {k: dict(v) for k, v in applied.items()}}
check('all202_static_samples_used', sample_count == 202)
image = read(out / 'image_native/summary.json'); video = read(out / 'video_native/summary.json')
check('image_cohorts25_and107_retained', image['groups']['old190_exposed25']['frames'] == 25 and image['groups']['not_matched107']['frames'] == 107)
check('original29_new41_cases43_and27_retained', [video['groups'][k]['frames'] for k in ('original29', 'new41', 'video_1', 'video_2')] == [29, 41, 43, 27])
check('new_probability_inferences41_only', image['new_probability_inferences'] == 0 and video['new_probability_inferences'] == 41)
for path in sorted((out / 'video_native/new_probabilities').glob('*.npz')):
    with np.load(path) as arrays:
        if len(arrays.files) != 8 or any(a.dtype != np.float32 or not np.isfinite(a).all() or not np.allclose(a.sum(0), 1, atol=2e-5) for a in arrays.values()):
            raise AssertionError('New41 FP32 probabilities invalid')
check('all41_new_FP32_member_and_grid_native_caches_valid', len(list((out / 'video_native/new_probabilities').glob('*.npz'))) == 41)
check('low_probability_FN_bins_conserve_all_reference_FN', sum(sum(r['network']['raw_fn_probability_bins_diagnostic'].values()) for r in video['records']) == video['models']['raw']['metrics']['fiber_fn'])
check('tip_metrics_not_generated_from_axis_proxy', video['tip_metrics'] is None and video['reference_axis_is_anatomical_tip'] is False and all(r['tip_metrics'] is None for r in video['records']))
gate = read(out / 'static_gates.json')
check('both_fixed_arms_rejected_without_relaxing_gates', all(not g['passed'] and 'image132_fiber_dice' in g['rejected_checks'] for g in gate.values()))

pids = []; frames = 0; full_totals = {}; all_video_raw_equal = 0
for vid, expected in [('video_1', 722), ('video_2', 1972)]:
    logs = {}; results = {}
    for name in names:
        folder = out / f'full_video/{name}/{vid}'; r = read(folder / 'summary.json')
        logs[name] = [__import__('json').loads(s) for s in (folder / 'frames.jsonl').read_text().splitlines()]
        timing = read(folder / 'timings.json'); pids.append(r['pid']); frames += r['frames']; results[name] = r
        check(name + '_' + vid + '_complete722_or1972_and_static_match', r['frames'] == expected == r['encoded_frames'] == len(logs[name]) == len(timing) and r['approved_frame_static_matches'] == (43 if vid == 'video_1' else 27) and r['networks'] == 3)
        check(name + '_' + vid + '_full_timing_percentiles_exact', all(np.isclose(r['timings'][k][q], func([t[k] for t in timing]), rtol=0, atol=1e-10) for k in timing[0] for q, func in [('mean', np.mean), ('p50', np.median), ('p95', lambda x: np.percentile(x, 95)), ('max', np.max)]))
        check(name + '_' + vid + '_H264_file_identity', sha256(folder / 'replay.mp4') == r['replay_sha256'])
    additions = {name: 0 for name in names[1:]}; changed = Counter(); fallbacks = {name: Counter() for name in names}
    for i in range(expected):
        raw = load_mask(out / f'full_video/raw/{vid}/masks/{i:06d}.png', (400, 400), {0, 1, 2})
        for name in names:
            pred = load_mask(out / f'full_video/{name}/{vid}/masks/{i:06d}.png', raw.shape, {0, 1, 2})
            log = logs[name][i]
            if log['frame_index'] != i or hashlib.sha256(pred.tobytes()).hexdigest() != log['mask_raw_bytes_sha256'] or hashlib.sha256(raw.tobytes()).hexdigest() != log['raw_mask_sha256']:
                raise AssertionError('Native full-video file identity differs')
            if log['rgb_raw_bytes_sha256'] != logs['raw'][i]['rgb_raw_bytes_sha256']: raise AssertionError('Cross-scheme source frame differs')
            if np.any((raw == 2) & (pred != 2)) or np.any((raw == 1) != (pred == 1)):
                raise AssertionError('Full-video protected pixel changed')
            added = int(((pred == 2) & (raw != 2)).sum())
            if added != log['application']['added_pixels']: raise AssertionError('Full-video additions differ from log')
            changed[name] += int(added > 0); fallbacks[name][log['application'].get('fallback') or 'applied'] += 1
            if name != 'raw': additions[name] += added
    check(vid + '_all3_native_outputs_and_protections_verified', all(results[n]['changed_frames'] == changed[n] and results[n]['fallback_counts'] == dict(fallbacks[n]) and results[n]['protection_violations'] == 0 for n in names))
    full_totals[vid] = {'added_pixels': additions, 'changed_frames': dict(changed), 'fallback_counts': {k: dict(v) for k, v in fallbacks.items()}}
check('six_independent_fresh_processes_all8082_frames', len(set(pids)) == 6 and frames == 8082 and read(out / 'job.json')['pid'] not in pids)
for vid, export in summary['comparison_videos'].items():
    check(vid + '_complete_four_column_comparison_identity', export['frames'] == (722 if vid == 'video_1' else 1972) == export['decoded_frames'] and sha256(export['path']) == export['sha256'])
check('no_full_video_accuracy_tip_or_memory_claim', summary['whole_video_accuracy_estimates'] is None and summary['tip_metrics'] is None and summary['stone_memory_started'] is False)
atomic_json(out / 'engineering_checks.json', {'passed': True, 'count': len(checks), 'checks': checks,
    'static_changes': totals, 'full_video_changes': full_totals, 'workers': pids, 'full_frames_all_schemes': frames,
    'zero_original_fiber_and_stone_deletion': True, 'tip_metrics': None, 'training_updates': 0,
    'train_pixels_read': 0, 'reserved_test_pixels_read': 0, 'source_sha256': sha256(Path(__file__))})
print(__import__('json').dumps({'passed': True, 'checks': len(checks), 'full_frames': frames}))
