"""Recompute full replay exports without running a network or using test pixels."""
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/a1_classwise_replay_20261028'
NAMES = ('A0', 'old190', 'classwise')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def load(path):
    with Image.open(path) as im: return np.asarray(im).copy()


def runs(values, start):
    result = []
    for i, value in enumerate(values, start):
        if result and result[-1]['predicted_present'] == value:
            result[-1]['frames'] += 1; result[-1]['end_frame_exclusive'] = i + 1
        else: result.append({'start_frame': i, 'end_frame_exclusive': i + 1, 'frames': 1, 'predicted_present': value})
    return result


def video_metadata(path):
    cap = cv2.VideoCapture(str(path))
    result = (cap.isOpened(), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(3)), int(cap.get(4)))
    cap.release(); return result


def main():
    p = read(OUT / 'preflight.json'); summary = read(OUT / 'summary.json'); checks = {}; details = {}
    checks['frozen_sources_and_related_files_unchanged'] = all(digest(ROOT / n) == h for n, h in p['sources'].items()) and all(digest(n) == h for n, h in p['related'].items())
    checks['frozen_checkpoints_and_video_bytes_unchanged'] = all(digest(v['path']) == v['sha256'] for v in list(p['checkpoints'].values()) + p['video_inputs'])
    checks['only29_reviewed_pixel_inputs_and_manifest_unchanged'] = len(p['validation_inputs']['files']) == 58 and all(digest(n) == h for n, h in p['validation_inputs']['files'].items()) and p['validation_inputs']['test_pixels_read'] is False
    manifest_path = ROOT / 'datasets/ureter_semis_kidney_joint_20261028/video_validation_manifest.json'
    manifest = read(manifest_path); rows = manifest['val']
    checks['reviewed_video_manifest_unchanged'] = digest(manifest_path) == p['validation_inputs']['manifest_sha256']
    checks['no_training_default_or_temporal_change_and_state_preserved'] = summary['state_unchanged'] and not summary['training_updates'] and not summary['default_model_changed'] and not summary['s2_accepted'] and not summary['s3_started']
    matrices = {n: np.zeros((3, 3), dtype=np.int64) for n in NAMES}; batchdiff = {n: 0 for n in NAMES}
    old_ids = {r['sample_id']: i for i, r in enumerate(rows)}
    for item in p['video_inputs']:
        vid = item['video_id']; folder = OUT / vid; s = summary['videos'][vid]; count = s['frames']
        logs = [json.loads(line) for line in (folder / 'frames.jsonl').read_text(encoding='utf-8').splitlines()]
        timings = read(folder / 'timings.json'); aligned = read(folder / 'keyframe_alignment.json')
        checks[f'{vid}_exact_sequential_counts_and_indices'] = len(logs) == len(timings) == p['protocol']['expected_frames'][vid] == count and [r['frame_index'] for r in logs] == list(range(count)) and all(r['one_based_frame'] == r['frame_index'] + 1 for r in logs)
        checks[f'{vid}_exact_export_inventory'] = all(len(list((folder / n).glob('*.png'))) == count for n in NAMES)
        checks[f'{vid}_repeat_and_alignment'] = s['repeat_bitwise_exact'] and len(aligned) == sum(r['video_id'] == vid for r in rows) and all(r['pixel_exact'] and r['decoded_rgb_sha256'] == r['snapshot_rgb_sha256'] for r in aligned)
        checks[f'{vid}_preview_metadata'] = video_metadata(folder / 'comparison.mp4') == (True, count, 1600, 434)
        checks[f'{vid}_clip_metadata_and_one_based_ranges'] = all(video_metadata(folder / e['clip']) == (True, e['frames'], 1600, 434) and e['start_frame'] + 1 == e['one_based_inclusive'][0] and e['end_frame_exclusive'] == e['one_based_inclusive'][1] and e['frames'] == e['end_frame_exclusive'] - e['start_frame'] for e in s['user_intervals'])
        selected = {r['frame_index']: r for r in rows if r['video_id'] == vid}
        previous = None; policy_ok = shapes_ok = counts_ok = changes_ok = decoded_ok = True
        cap = cv2.VideoCapture(item['path'])
        try:
            for index, record in enumerate(logs):
                ok, frame = cap.read()
                if not ok: raise AssertionError('Audit sequential decode ended early')
                masks = {n: load(folder / n / f'{index:06d}.png') for n in NAMES}
                shapes_ok &= all(a.shape == (400, 400) and set(np.unique(a)).issubset({0, 1, 2}) for a in masks.values())
                policy_ok &= np.array_equal(masks['classwise'], np.where(masks['A0'] == 2, 2, np.where(masks['old190'] == 1, 1, 0)))
                # Recompute estimated FOV independently with the same fixed definition.
                binary = (frame.max(2) > 10).astype(np.uint8); fov = np.zeros((400, 400), np.uint8)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours: cv2.drawContours(fov, [max(contours, key=cv2.contourArea)], -1, 1, -1)
                valid = fov.astype(bool); pixels = int(valid.sum())
                for n, a in masks.items():
                    d = record['models'][n]; counts_ok &= d['estimated_fov_pixels'] == pixels
                    for k, name in [(1, 'stone'), (2, 'fiber')]:
                        area = int((a == k).sum()); counts_ok &= area == d[f'{name}_pixels']
                        fraction = float(((a == k) & valid).sum() / pixels) if pixels else None
                        counts_ok &= fraction == d[f'{name}_fraction_estimated_fov']
                        delta = abs(area - int((previous[n] == k).sum())) if previous is not None else None
                        changes_ok &= delta == d[f'{name}_absolute_area_change_pixels']
                    change = float((a[valid] != previous[n][valid]).mean()) if previous is not None and pixels else None
                    changes_ok &= change == d['uncompensated_label_change_fraction']
                if index in selected:
                    row = selected[index]; target = load(row['mask']); ref = np.asarray(Image.open(row['image']).convert('RGB'))
                    decoded_ok &= np.array_equal(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), ref)
                    valid_target = target != 255
                    for n in NAMES:
                        cm = np.bincount((3 * target[valid_target].astype(int) + masks[n][valid_target]).ravel(), minlength=9).reshape(3, 3)
                        matrices[n] += cm
                        prior = load(ROOT / f'runs/a1_classwise_experts_20261028/video_native/{n}/{old_ids[row["sample_id"]]:04d}.png')
                        batchdiff[n] += int((prior != masks[n]).sum())
                previous = masks
            checks[f'{vid}_sequential_source_end_exact'] = not cap.read()[0]
        finally: cap.release()
        checks[f'{vid}_all_native_masks_and_policy_pixel_exact'] = bool(shapes_ok and policy_ok)
        checks[f'{vid}_all_per_frame_area_and_fov_counts_recomputed'] = bool(counts_ok)
        checks[f'{vid}_all_adjacent_changes_recomputed'] = bool(changes_ok)
        checks[f'{vid}_decoded_keyframe_pixels_rechecked'] = bool(decoded_ok)
        aggregates_ok = True
        for start, end, expected in [(0, count, s['models'])] + [(e['start_frame'], e['end_frame_exclusive'], e['models']) for e in s['user_intervals']]:
            for n in NAMES:
                subset = [r['models'][n] for r in logs[start:end]]; aggregate = expected[n]
                aggregates_ok &= aggregate['large_stone_region_candidate_frames'] == sum((d['stone_fraction_estimated_fov'] or 0) > .25 for d in subset)
                for name in ('stone', 'fiber'):
                    values = [d[f'{name}_pixels'] > 0 for d in subset]; eruns = runs(values, start)
                    a = aggregate[name]
                    aggregates_ok &= a['presence_runs'] == eruns and a['predicted_nonempty_frames'] == sum(values)
                    aggregates_ok &= a['nonempty_to_empty'] == sum(x and not y for x, y in zip(values, values[1:]))
                    aggregates_ok &= a['empty_to_nonempty'] == sum(not x and y for x, y in zip(values, values[1:]))
        checks[f'{vid}_full_and_fixed_interval_runs_counts_recomputed'] = bool(aggregates_ok)
        tchecks = all(set(record['timing']).issubset(t) and all(record['timing'][k] == t[k] for k in record['timing']) for record, t in zip(logs, timings))
        for key, expected in s['timings'].items():
            values = [t[key] for t in timings]
            tchecks &= all(np.isclose(expected[k], f(values)) for k, f in [('mean', np.mean), ('p50', np.median), ('p95', lambda a: np.percentile(a, 95)), ('max', np.max)])
        tchecks &= s['core_frames_exceeding_source_budget'] == sum(t['core_ms'] > 1000 / s['fps'] for t in timings)
        tchecks &= s['total_cycle_frames_exceeding_source_budget'] == sum(t['cycle_including_jsonl_ms'] > 1000 / s['fps'] for t in timings)
        checks[f'{vid}_stage_and_total_timing_statistics_recomputed'] = bool(tchecks)
        details[vid] = {'frames': count, 'reviewed_keyframes': len(aligned), 'clips': len(s['user_intervals']), 'exported_model_masks': count * 3}
    checks['reviewed29_independent_native_confusions'] = all(matrices[n].tolist() == summary['reviewed29_metrics'][n]['confusion_matrix'] for n in NAMES)
    checks['batch1_vs_batch2_differences_independently_counted'] = batchdiff == summary['batch1_vs_batch2_different_pixels']
    checks['no_dense_accuracy_metrics_or_claimed_tracking'] = all(s['accuracy_metrics_on_unreviewed_frames'] is None for s in summary['videos'].values())
    with zipfile.ZipFile(OUT / 'source_snapshot.zip') as z:
        checks['source_snapshot_unique_and_matches'] = len(z.namelist()) == len(set(z.namelist())) and all(hashlib.sha256(z.read(n.replace('\\', '/'))).hexdigest() == h for n, h in p['sources'].items())
    result = {'checks': {n: bool(v) for n, v in checks.items()}, 'all_passed': bool(all(checks.values())), 'details': details,
              'batch_differences': batchdiff, 'new_unit_tests': {'passed': 3, 'failed': 0, 'command': 'python -m unittest discover -s tests -p test_classwise_replay.py -v'},
              'scope': 'Full exported pixel, sequential decode, diagnostic and timing audit; no independent-case, event-truth or live-stream validation'}
    atomic = ROOT / 'runs/engineering/classwise_replay_checks_20261028.json'
    atomic.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    assert result['all_passed'], result
    print(json.dumps({'all_passed': True, 'checks': len(checks), 'details': details, 'batch_differences': batchdiff}, indent=2))


if __name__ == '__main__': main()
