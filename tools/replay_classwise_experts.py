"""Frozen, stateless full-video expert replay; no dense-truth accuracy claims."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import zipfile

import cv2
import numpy as np
from PIL import Image
import torch

from ureter_semis.data import _tensor
from ureter_semis.experiment import atomic_json, input_identity, runtime_identity, seed_training, sha256, source_identity
from ureter_semis.video_metrics import estimated_fov
from .classwise_experts import ClasswiseExperts
from .run_classwise_expert_control import summarize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/a1_classwise_replay_20261028'
NAMES = ('A0', 'old190', 'classwise')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pil_input(frame, size, device):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return _tensor(Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR)).unsqueeze(0).to(device)


def intervals(protocol, video_id, frame_count):
    result = []
    for kind, ranges in protocol['user_events_one_based_inclusive'].get(video_id, {}).items():
        for first, last in ranges:
            if not 1 <= first <= last <= frame_count:
                raise ValueError('Invalid one-based inclusive user interval')
            result.append({'event': kind, 'one_based_inclusive': [first, last],
                           'start_frame': first - 1, 'end_frame_exclusive': last,
                           'frames': last - first + 1, 'reference': 'user_report_not_dense_ground_truth'})
    return result


def mask_diagnostics(mask, fov, previous):
    valid = int(fov.sum())
    result = {'estimated_fov_pixels': valid}
    for k, name in [(1, 'stone'), (2, 'fiber')]:
        area = int((mask == k).sum())
        result[f'{name}_pixels'] = area
        result[f'{name}_fraction_estimated_fov'] = float(((mask == k) & fov).sum() / valid) if valid else None
        result[f'{name}_absolute_area_change_pixels'] = abs(area - int((previous == k).sum())) if previous is not None else None
    result['uncompensated_label_change_fraction'] = float((mask[fov] != previous[fov]).mean()) if previous is not None and valid else None
    return result


def stats(values):
    values = [v for v in values if v is not None]
    return {'mean': float(np.mean(values)), 'p50': float(np.median(values)), 'p95': float(np.percentile(values, 95)),
            'max': float(np.max(values))} if values else None


def presence_runs(rows, name):
    if not rows:
        return []
    runs = []
    for row in rows:
        index = row['frame_index']; present = row[f'{name}_pixels'] > 0
        if runs and runs[-1]['end_frame_exclusive'] != index:
            raise ValueError('Runs require consecutive frames')
        if runs and runs[-1]['predicted_present'] == present:
            runs[-1]['end_frame_exclusive'] = index + 1
            runs[-1]['frames'] += 1
        else:
            runs.append({'start_frame': index, 'end_frame_exclusive': index + 1, 'frames': 1, 'predicted_present': present})
    return runs


def summarize_diagnostics(rows, fps):
    result = {'frames': len(rows), 'scope': 'Prediction-only proxies, no motion compensation or dense event truth',
              'uncompensated_label_change_fraction': stats([r['uncompensated_label_change_fraction'] for r in rows[1:]]),
              'large_stone_region_candidate_frames': sum((r['stone_fraction_estimated_fov'] or 0) > .25 for r in rows)}
    result['large_stone_candidate_seconds_not_fp_duration'] = result['large_stone_region_candidate_frames'] / fps
    for name in ('stone', 'fiber'):
        runs = presence_runs(rows, name)
        result[name] = {'area_pixels': stats([r[f'{name}_pixels'] for r in rows]),
                        'absolute_area_change_pixels': stats([r[f'{name}_absolute_area_change_pixels'] for r in rows[1:]]),
                        'predicted_nonempty_frames': sum(r[f'{name}_pixels'] > 0 for r in rows),
                        'nonempty_to_empty': sum(a['predicted_present'] and not b['predicted_present'] for a, b in zip(runs, runs[1:])),
                        'empty_to_nonempty': sum(not a['predicted_present'] and b['predicted_present'] for a, b in zip(runs, runs[1:])),
                        'presence_runs': runs}
    return result


def save_png(path, array):
    ok, data = cv2.imencode('.png', array)
    if not ok:
        raise RuntimeError(f'Cannot encode {path}')
    data.tofile(str(path))


def render(frame, masks, video_id, index):
    panels = []
    for name in ('Input', *NAMES):
        panel = frame.copy()
        if name != 'Input':
            for k, color in [(1, (40, 40, 235)), (2, (235, 210, 20))]:
                selected = masks[name] == k
                panel[selected] = (.48 * frame[selected] + .52 * np.asarray(color)).astype(np.uint8)
        title = np.zeros((34, frame.shape[1], 3), np.uint8)
        cv2.putText(title, f'{name} | {video_id} frame {index + 1}', (6, 21), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(np.concatenate([title, panel], axis=0))
    return np.concatenate(panels, axis=1)


@torch.inference_mode()
def infer(experts, frame, keep_probabilities=False):
    experts.sync(); started = time.perf_counter()
    tensor = pil_input(frame, experts.size, experts.device)
    experts.sync(); prepared = time.perf_counter()
    result = experts.predict_native(tensor, [frame.shape[:2]])[0]
    # Split forward/native timing without performing extra network calls.
    # predict_native covers all three networks plus native probability mapping.
    experts.sync(); computed = time.perf_counter()
    masks = {n: result[k].cpu().numpy().copy() for n, k in [('A0', 'fiber_mask'), ('old190', 'stone_mask'), ('classwise', 'mask')]}
    experts.sync(); finished = time.perf_counter()
    timing = {'preprocess_ms': (prepared - started) * 1000,
              'all3_forward_and_native_gpu_ms': (computed - prepared) * 1000,
              'native_masks_cpu_transfer_ms': (finished - computed) * 1000,
              'core_ms': (finished - started) * 1000}
    probabilities = tuple(result[k].cpu().numpy().copy() for k in ('stone_probability', 'fiber_probability')) if keep_probabilities else None
    return masks, timing, probabilities


def model_state_hash(experts):
    digest = hashlib.sha256()
    for i, model in enumerate(experts.models):
        for key, value in model.state_dict().items():
            digest.update(f'{i}:{key}'.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run_video(item, protocol, experts, validation_rows, progress):
    vid = item['video_id']; folder = OUT / vid; folder.mkdir()
    for name in NAMES:
        (folder / name).mkdir()
    (folder / 'clips').mkdir(); (folder / 'keyframes').mkdir()
    source = Path(item['path']); cap = cv2.VideoCapture(str(source))
    writers = []
    if not cap.isOpened():
        raise RuntimeError(f'Cannot decode {source}')
    fps = float(cap.get(cv2.CAP_PROP_FPS)); count = protocol['expected_frames'][vid]
    width, height = int(cap.get(3)), int(cap.get(4))
    if fps <= 0 or not np.isfinite(fps) or (width, height) != (item['width'], item['height']):
        raise ValueError('Video geometry or frame rate differs')
    events = intervals(protocol, vid, count)
    try:
        ok, first = cap.read()
        if not ok:
            raise ValueError('Empty video')
        for _ in range(12):
            infer(experts, first)
        a, _, ap = infer(experts, first, True); b, _, bp = infer(experts, first, True)
        if not all(np.array_equal(a[n], b[n]) for n in NAMES) or not all(np.array_equal(x, y) for x, y in zip(ap, bp)):
            raise ValueError('Identical-frame repeat failed')
        del a, b, ap, bp
        cap.release(); cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError('Cannot reopen video for sequential replay')
        torch.cuda.reset_peak_memory_stats()
        def writer(path):
            w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 4, height + 34))
            if not w.isOpened():
                raise RuntimeError(f'Cannot write {path}')
            writers.append(w); return w
        preview = writer(folder / 'comparison.mp4')
        event_writers = []
        for i, event in enumerate(events):
            event['clip'] = f'clips/{i:02d}_{event["event"]}_{event["one_based_inclusive"][0]}-{event["one_based_inclusive"][1]}.mp4'
            event_writers.append(writer(folder / event['clip']))
        selected = {r['frame_index']: r for r in validation_rows if r['video_id'] == vid}
        previous = {n: None for n in NAMES}; logs = []; alignment = []; sampled = []
        started = time.perf_counter()
        with (folder / 'frames.jsonl').open('w', encoding='utf-8') as log:
            while True:
                cycle = time.perf_counter(); ok, frame = cap.read(); decoded = time.perf_counter()
                if not ok:
                    break
                index = len(logs)
                if index >= count or frame.shape != (height, width, 3):
                    raise ValueError('Unexpected frame count or shape')
                masks, timing, _ = infer(experts, frame)
                diagnostics_started = time.perf_counter(); fov = estimated_fov(frame, frame.shape[:2])
                diagnostics = {n: mask_diagnostics(masks[n], fov, previous[n]) for n in NAMES}
                if index in selected:
                    row = selected[index]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    snapshot = np.asarray(Image.open(row['image']).convert('RGB'))
                    identical = rgb.shape == snapshot.shape and np.array_equal(rgb, snapshot)
                    check = {'sample_id': row['sample_id'], 'video_id': vid, 'frame_index': index, 'one_based_frame': index + 1,
                             'decoded_rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                             'snapshot_rgb_sha256': hashlib.sha256(snapshot.tobytes()).hexdigest(), 'pixel_exact': identical}
                    alignment.append(check)
                    if not identical:
                        atomic_json(folder / 'alignment_failure.json', check)
                        raise ValueError('Decoded keyframe differs from frozen snapshot')
                    sampled.append((row, {n: masks[n].copy() for n in NAMES}))
                diagnosed = time.perf_counter()
                for name in NAMES:
                    save_png(folder / name / f'{index:06d}.png', masks[name])
                masks_written = time.perf_counter()
                canvas = render(frame, masks, vid, index); preview.write(canvas)
                for event, ew in zip(events, event_writers):
                    if event['start_frame'] <= index < event['end_frame_exclusive']:
                        ew.write(canvas)
                if index in selected:
                    save_png(folder / 'keyframes' / f'{index:06d}.jpg.png', canvas)
                rendered = time.perf_counter()
                record = {'frame_index': index, 'one_based_frame': index + 1, 'timestamp_s': index / fps,
                          'coordinate_system': 'native_probability_first', 'dense_ground_truth_available': False,
                          'models': diagnostics, 'timing': {**timing, 'decode_ms': (decoded - cycle) * 1000,
                              'diagnostic_and_keyframe_check_ms': (diagnosed - diagnostics_started) * 1000,
                              'mask_export_ms': (masks_written - diagnosed) * 1000,
                              'render_and_video_write_ms': (rendered - masks_written) * 1000}}
                # Includes serialization and flushed write, excluding only this timing sidecar append.
                log.write(json.dumps(record, allow_nan=False) + '\n'); log.flush()
                record['timing']['cycle_including_jsonl_ms'] = (time.perf_counter() - cycle) * 1000
                logs.append(record); previous = masks
                if (index + 1) % 200 == 0:
                    progress('replaying', video_id=vid, frames_completed=index + 1, expected_frames=count)
                    print(f'{vid}: {index + 1}/{count}', flush=True)
        for w in writers:
            w.release()
        writers.clear(); elapsed = time.perf_counter() - started
        if len(logs) != count or len(alignment) != len(selected):
            raise ValueError('Frame/keyframe count differs from protocol')
        # Persist exact cycle times separately: frames.jsonl records stage times before JSONL serialization.
        atomic_json(folder / 'timings.json', [r['timing'] for r in logs])
        atomic_json(folder / 'keyframe_alignment.json', alignment)
        by_model = {n: summarize_diagnostics([{'frame_index': r['frame_index'], **r['models'][n]} for r in logs], fps) for n in NAMES}
        for event in events:
            subset = logs[event['start_frame']:event['end_frame_exclusive']]
            event['models'] = {n: summarize_diagnostics([{'frame_index': r['frame_index'], **r['models'][n]} for r in subset], fps) for n in NAMES}
        deadline = 1000 / fps
        result = {'video_id': vid, 'frames': len(logs), 'fps': fps, 'source_frame_budget_ms': deadline,
                  'decode': 'sequential from frame zero', 'warmup_frames_excluded': 12, 'repeat_bitwise_exact': True,
                  'native_geometry': [height, width], 'wall_seconds_including_writer_flush': elapsed,
                  'offline_frames_per_second_including_exports': len(logs) / elapsed,
                  'timings': {key: stats([r['timing'][key] for r in logs]) for key in logs[0]['timing']},
                  'core_frames_exceeding_source_budget': sum(r['timing']['core_ms'] > deadline for r in logs),
                  'total_cycle_frames_exceeding_source_budget': sum(r['timing']['cycle_including_jsonl_ms'] > deadline for r in logs),
                  'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                  'models': by_model, 'user_intervals': events,
                  'accuracy_metrics_on_unreviewed_frames': None,
                  'limitations': ['Prediction presence, jumps and large regions are not true disappearance, identity, occlusion recovery or false positives',
                      'Estimated FOV and uncompensated changes are image/motion dependent',
                      'Offline four-panel rendering/writing is measured; live display, input queue and dropped frames are not tested',
                      'Shared3network reference exports do not provide isolated A0/old190 total latency']}
        atomic_json(folder / 'summary.json', result)
        return result, sampled
    finally:
        cap.release()
        for w in writers:
            w.release()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(); job = {'pid': os.getpid(), 'training_updates': 0, 'default_model_changed': False, 'temporal': False, 's2_accepted': False}
    def progress(phase, **extra):
        job.update(phase=phase, **extra); atomic_json(OUT / 'job.json', job)
    progress('preflight')
    try:
        protocol_path = ROOT / 'configs/replay_classwise_experts_20261028.json'; protocol = read(protocol_path)
        original_config = ROOT / protocol['video_source_identity_from']; inputs = read(original_config)['inputs']
        predictor = read(ROOT / protocol['frozen_predictor_manifest'])
        checkpoint = {n: predictor['checkpoints'][n] for n in ('old190', 'A0')}
        for name, key in [('old190', 'stone_checkpoint_sha256'), ('A0', 'fiber_checkpoint_sha256')]:
            if sha256(Path(checkpoint[name]['path'])) != checkpoint[name]['sha256'] or checkpoint[name]['sha256'] != protocol[key]:
                raise ValueError('Frozen expert weight identity differs')
        for item in inputs:
            if sha256(Path(item['path'])) != item['sha256']:
                raise ValueError('Video source identity differs')
        manifest_path = ROOT / 'datasets/ureter_semis_kidney_joint_20261028/video_validation_manifest.json'
        manifest = read(manifest_path); rows = manifest['val']
        identity = input_identity(manifest_path, Path(manifest['root']), manifest)
        related = [protocol_path, original_config, ROOT / protocol['frozen_predictor_manifest'], ROOT / protocol['frozen_expert_protocol'],
                   manifest_path.parent / 'manifest.json', ROOT / 'runs/a1_classwise_experts_20261028/summary.json']
        for name in NAMES:
            related += sorted((ROOT / f'runs/a1_classwise_experts_20261028/video_native/{name}').glob('*.png'))
        extra_sources = ['tools/replay_classwise_experts.py', 'tools/classwise_experts.py', 'tools/run_classwise_expert_control.py',
                         'tools/diagnose_joint_failures.py', 'tools/evaluate_native_checkpoints.py', 'tools/evaluate_spatial_detail.py',
                         'tests/test_classwise_replay.py']
        sources = source_identity(); sources.update({n: sha256(ROOT / n) for n in extra_sources})
        preflight = {'protocol': protocol, 'related': {str(p): sha256(p) for p in related}, 'sources': sources,
                     'video_inputs': inputs, 'checkpoints': checkpoint, 'validation_inputs': identity,
                     'training_updates': 0, 'reserved_test_pixels_read': False}
        atomic_json(OUT / 'preflight.json', preflight)
        with zipfile.ZipFile(OUT / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            for name in sources:
                z.write(ROOT / name, name.replace('\\', '/'))
            z.write(protocol_path, str(protocol_path.relative_to(ROOT)))
        seed_training(protocol['seed']); experts = ClasswiseExperts(checkpoint['old190']['path'], checkpoint['A0']['path'])
        if (experts.model_count, experts.parameter_count, experts.size) != (3, 73804185, 256):
            raise ValueError('Frozen model budget differs')
        atomic_json(OUT / 'runtime.json', runtime_identity(experts.device))
        before = model_state_hash(experts); videos = {}; sampled = {}
        for item in inputs:
            progress('loading_video', video_id=item['video_id'], frames_completed=0)
            result, selected = run_video(item, protocol, experts, rows, progress)
            videos[item['video_id']] = result
            sampled.update({r['sample_id']: (r, masks) for r, masks in selected})
        progress('reviewed29_validation')
        ordered = [sampled[r['sample_id']] for r in rows]
        targets = [np.asarray(Image.open(r['mask'])).copy() for r in rows]
        metrics = {n: summarize(rows, targets, [masks[n] for _, masks in ordered]) for n in NAMES}
        previous = read(ROOT / 'runs/a1_classwise_experts_20261028/summary.json')['video']['models']
        changes = []
        for i, (row, masks) in enumerate(ordered):
            changes.append({'sample_id': row['sample_id'], 'different_pixels_from_batch2': {
                n: int((masks[n] != np.asarray(Image.open(ROOT / f'runs/a1_classwise_experts_20261028/video_native/{n}/{i:04d}.png'))).sum()) for n in NAMES}})
        native = {'models': metrics, 'batch1_vs_batch2': changes,
                  'previous_metrics': {n: previous[n]['metrics'] for n in NAMES},
                  'policy': 'Differences recorded without threshold reselection; GT unchanged and decoded RGB must match exactly'}
        atomic_json(OUT / 'reviewed29.json', native)
        from ureter_semis.supervised_trainer import assess
        gates = read(ROOT / 'runs/a1_classwise_experts_20261028/preflight.json')['native_video_gates']
        screening = assess(metrics['classwise']['metrics'], gates)
        if model_state_hash(experts) != before or any(model.training for model in experts.models):
            raise ValueError('Model state or eval mode changed')
        assert input_identity(manifest_path, Path(manifest['root']), manifest) == identity
        assert all(sha256(ROOT / name) == h for name, h in sources.items())
        assert all(sha256(Path(name)) == h for name, h in preflight['related'].items())
        assert all(sha256(Path(v['path'])) == v['sha256'] for v in checkpoint.values())
        assert all(sha256(Path(v['path'])) == v['sha256'] for v in inputs)
        result = {'videos': videos, 'reviewed29_metrics': {n: metrics[n]['metrics'] for n in NAMES},
                  'batch1_vs_batch2_different_pixels': {n: sum(r['different_pixels_from_batch2'][n] for r in changes) for n in NAMES},
                  'reviewed29_screening': screening, 'frozen_gates': gates,
                  'frames_total': sum(v['frames'] for v in videos.values()), 'model_count': 3, 'parameter_count': 73804185,
                  'model_state_before_and_after_sha256': before, 'state_unchanged': True,
                  'training_updates': 0, 'reserved_test_pixels_read': False, 'default_model_changed': False,
                  's2_accepted': False, 's3_started': False}
        atomic_json(OUT / 'summary.json', result)
        progress('completed', frames_completed=result['frames_total'], expected_frames=2694, screening_passed=screening['passed'])
        print(json.dumps({'phase': 'completed', 'frames': result['frames_total'], 'batch_differences': result['batch1_vs_batch2_different_pixels'],
                          'timings': {n: v['timings'] for n, v in videos.items()}}, indent=2), flush=True)
    except Exception as error:
        progress('failed', error=str(error)); (OUT / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8'); raise


if __name__ == '__main__':
    main()
