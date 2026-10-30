"""Frozen A+ experiment; static diagnostics plus every scheme on both complete videos."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

import cv2
import numpy as np
from PIL import Image
import torch

from ureter_semis.experiment import atomic_json, input_identity, runtime_identity, seed_training, source_identity
from .classwise_experts import ClasswiseExperts
from .fiber_reference_workspace import ROOT, read, sha256, load_mask
from .fiber_blue_soft_prior import refine, blue_evidence
from .fiber_tip_diagnostics import frozen_probabilities, reference_axis, appearance_fields, distributions, whole_fiber_diagnosis
from .replay_classwise_experts import pil_input, save_png, model_state_hash, stats
from .run_classwise_expert_control import summarize
from .diagnose_joint_failures import paired_changes

CONFIG = ROOT / 'configs/fiber_blue_aplus_20261029.json'


def frame_read(path):
    with Image.open(path) as image: return cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2BGR)


def margin_from_probability(p):
    return np.log(np.maximum(p[2], np.float32(1e-7)) / np.maximum(np.max(p[:2], axis=0), np.float32(1e-7)))


def cached_paths(row, index, video, original_indices):
    if not video:
        base = ROOT / 'runs/a1_fiber_tip_d0_20261029/image_native'
        return base / 'probabilities' / f'{index:04d}.npz', base / 'raw_classwise' / f'{index:04d}.png'
    original_index = original_indices.get((row['video_id'], row['frame_index']))
    cache = ROOT / f'runs/a1_fiber_tip_d0_20261029/video_native/probabilities/{original_index:04d}.npz' if original_index is not None else None
    raw = ROOT / f'runs/a1_fiber_tip_d0_20261029/full_video/{row["video_id"]}/raw_classwise/{row["frame_index"]:06d}.png'
    return cache, raw


def static_set(rows, video, experts, out, config, original_indices, diagnostic_settings, progress):
    folder = out / ('video_native' if video else 'image_native'); folder.mkdir()
    names = config['schemes']
    for name in names: (folder / name).mkdir()
    (folder / 'new_probabilities').mkdir()
    targets = []; predictions = {name: [] for name in names}; records = []
    color_pool = defaultdict(lambda: defaultdict(list)); new_forwards = 0
    for i, row in enumerate(rows):
        frame = frame_read(row['image'])
        target = load_mask(row['mask'], frame.shape[:2], {0, 1, 2, 255})
        cache, raw_path = cached_paths(row, i, video, original_indices)
        if cache:
            with np.load(cache) as data: arrays = {k: data[k].copy() for k in data.files}
            arrays['raw_classwise'] = load_mask(raw_path, frame.shape[:2], {0, 1, 2})
        else:
            arrays = frozen_probabilities(experts, frame); new_forwards += 1
            np.savez_compressed(folder / 'new_probabilities' / f'{i:04d}.npz', **{k: v for k, v in arrays.items() if k not in {'raw_classwise', 'old190_mask'}})
        raw = arrays['raw_classwise']; historical = load_mask(raw_path, target.shape, {0, 1, 2})
        p = arrays['a0_native']; reconstructed = np.zeros(raw.shape, np.uint8)
        reconstructed[arrays['old190_native'].argmax(0) == 1] = 1; reconstructed[p.argmax(0) == 2] = 2
        if not np.array_equal(raw, historical) or not np.array_equal(raw, reconstructed):
            raise ValueError('Frozen native baseline reproduction differs')
        margin = margin_from_probability(p); outputs = {'raw': raw}; application = {}
        for name in names[1:]: outputs[name], application[name] = refine(frame, raw, margin, config['refinement'], name)
        changes = {}
        for name in names:
            prediction = outputs[name]
            if np.any((raw == 2) & (prediction != 2)) or np.any((raw == 1) != (prediction == 1)):
                raise AssertionError('Protected raw labels changed')
            save_png(folder / name / f'{i:04d}.png', prediction); predictions[name].append(prediction)
            changes[name] = paired_changes(target, raw, prediction, 2)
        record = {'sample_id': row['sample_id'], 'video_id': row.get('video_id'), 'frame_index': row.get('frame_index'),
                  'application': application, 'changes': changes, 'tip_metrics': None}
        if video:
            axis, t = reference_axis(frame, target, diagnostic_settings)
            fields = appearance_fields(frame, arrays, axis)
            fields.update(blue_evidence(frame, config['refinement']))
            truth = target == 2
            selected = {'reference_TP': truth & (raw == 2), 'reference_FN': truth & (raw != 2),
                        'raw_FP': (target != 255) & ~truth & (raw == 2)}
            if axis['valid']: selected['distal_axis_proxy_NOT_tip'] = truth & (t >= .85)
            for label, region in selected.items():
                for key, field in fields.items():
                    if region.any(): color_pool[label][key].append(field[region].astype(np.float32))
            record.update(reference_axis=axis, network=whole_fiber_diagnosis(target, arrays, diagnostic_settings),
                          appearance={label: distributions(fields, region, [5, 25, 50, 75, 95]) for label, region in selected.items()})
        records.append(record); targets.append(target)
        if (i + 1) % 20 == 0: progress('static_video' if video else 'static_image', completed=i + 1, total=len(rows))
    result = {'models': {name: summarize(rows, targets, predictions[name]) for name in names},
              'records': records, 'new_probability_inferences': new_forwards, 'baseline_different_pixels': 0,
              'tip_metrics': None, 'reference_axis_is_anatomical_tip': False}
    groups = {}
    if video:
        selectors = {vid: [i for i, r in enumerate(rows) if r['video_id'] == vid] for vid in ('video_1', 'video_2')}
        selectors.update({label: [i for i, r in enumerate(rows) if r['evaluation_cohort'] == cohort]
                          for label, cohort in [('original29', 'original29'), ('new41', 'new_user_approved')]})
    else:
        selectors = {name: [i for i, r in enumerate(rows) if r['source_dataset'] == name] for name in sorted({r['source_dataset'] for r in rows})}
        exposed = set(read(ROOT / 'runs/a1_classwise_experts_20261028/summary.json')['image']['historical_exposure_cohorts']['historical_old190_exposed']['sample_ids'])
        selectors.update({name: [i for i, r in enumerate(rows) if (r['sample_id'] in exposed) == value]
                          for name, value in [('old190_exposed25', True), ('not_matched107', False)]})
    for label, indices in selectors.items():
        groups[label] = {'frames': len(indices), 'models': {name: summarize([rows[i] for i in indices], [targets[i] for i in indices], [predictions[name][i] for i in indices]) for name in names}}
    result['groups'] = groups
    pooled = {}
    for label, features in color_pool.items():
        merged = {k: np.concatenate(v) for k, v in features.items()}; normal = merged.pop('normal_color_gradient', None)
        pooled[label] = distributions(merged, np.ones(len(next(iter(merged.values()))), bool), [5, 25, 50, 75, 95])
        if normal is not None: pooled[label]['normal_color_gradient_valid_axes_only'] = {'pixels': len(normal), 'mean': float(normal.mean())}
    result['appearance_pooled'] = pooled
    atomic_json(folder / 'summary.json', result)
    return result


def gates(video, image, config):
    epsilon = config['gates']['nonregression_epsilon']; result = {}
    cohorts = {'video_combined70': video['models'], 'video_original29': video['groups']['original29']['models'],
               'video_new41': video['groups']['new41']['models'], 'image132': image['models']}
    for name in config['schemes'][1:]:
        checks = {}; improved = False
        for label, models in cohorts.items():
            before, after = models['raw'], models[name]
            for metric in ('fiber_dice', 'fiber_precision', 'fiber_recall'):
                a, b = before['metrics'][metric], after['metrics'][metric]
                checks[label + '_' + metric] = a is not None and b is not None and b >= a - epsilon
            checks[label + '_fiber_boundary_f'] = after['boundaries']['2']['f_score'] >= before['boundaries']['2']['f_score'] - epsilon
            checks[label + '_stone_identical_counts'] = all(before['metrics'][m] == after['metrics'][m] for m in ('stone_tp', 'stone_fp', 'stone_fn'))
            improved |= after['metrics']['fiber_dice'] > before['metrics']['fiber_dice'] + epsilon
        checks['at_least_one_dice_improvement'] = improved
        result[name] = {'checks': checks, 'passed': all(checks.values()), 'rejected_checks': [k for k, v in checks.items() if not v]}
    return result


@torch.inference_mode()
def video_pipeline(experts, frame, config, scheme):
    experts.sync(); start = time.perf_counter()
    result = experts.predict_native(pil_input(frame, experts.size, experts.device), [frame.shape[:2]])[0]
    raw = result['mask'].cpu().numpy().copy()
    # Preserve the D0 control's required/native transfer interface, identically in all arms.
    stone = result['stone_mask'].cpu().numpy().copy(); a0 = result['fiber_mask'].cpu().numpy().copy()
    probability = result['fiber_probability'][2].cpu().numpy().copy()
    if scheme != 'raw':
        full = result['fiber_probability'].cpu().numpy().copy(); margin = margin_from_probability(full)
    experts.sync(); core_end = time.perf_counter()
    if scheme == 'raw': prediction = raw; record = {'applied': False, 'added_pixels': 0, 'proposed_pixels': 0, 'fallback': 'raw_control'}
    else: prediction, record = refine(frame, raw, margin, config['refinement'], scheme)
    end = time.perf_counter()
    return raw, prediction, record, {'base_pipeline_ms': (core_end - start) * 1000,
        'refinement_ms': (end - core_end) * 1000, 'pipeline_ms': (end - start) * 1000}


def panel(frame, name, mask, index):
    image = frame.copy()
    if mask is not None:
        for value, color in [(1, (40, 40, 235)), (2, (235, 210, 20))]:
            region = mask == value; image[region] = (.45 * frame[region] + .55 * np.asarray(color)).astype(np.uint8)
    head = np.zeros((34, frame.shape[1], 3), np.uint8)
    cv2.putText(head, f'{name} | frame{index + 1}', (4, 22), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([head, image])


def video_worker(out, config, scheme, vid):
    pre = read(out / 'preflight.json'); info = next(v for v in pre['videos'] if v['video_id'] == vid)
    folder = out / 'full_video' / scheme / vid; folder.mkdir(parents=True); (folder / 'masks').mkdir()
    for item in [info, *pre['checkpoints'].values()]:
        if sha256(item['path']) != item['sha256']: raise ValueError('Worker identity changed')
    seed_training(42); experts = ClasswiseExperts(pre['checkpoints']['old190']['path'], pre['checkpoints']['A0']['path'])
    before = model_state_hash(experts); atomic_json(folder / 'runtime.json', {'pid': os.getpid(), **runtime_identity(experts.device)})
    manifest = read(ROOT / config['video_manifest']); selected = {r['frame_index']: r for r in manifest['val'] if r['video_id'] == vid}
    population = read(ROOT / 'runs/fiber_reference_d1_20261029/manifest.json')['records']
    pop = {r['frame_index']: r for r in population if r['kind'] == 'semantic' and r['video_id'] == vid}
    reference_log = [json.loads(s) for s in (ROOT / f'runs/a1_fiber_tip_d0_20261029/full_video/{vid}/frames.jsonl').read_text().splitlines()]
    cap = cv2.VideoCapture(info['path']); ok, first = cap.read()
    if not ok: raise ValueError('Empty video')
    for _ in range(config['full_video']['warmup_frames']): video_pipeline(experts, first, config, scheme)
    cap.release(); cap = cv2.VideoCapture(info['path']); fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(str(folder / 'replay.mp4'), cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*'H264'), fps, (800, 434))
    if not writer.isOpened(): raise RuntimeError('H264 encoder unavailable')
    timings = []; counts = Counter(); changed = 0; protections = 0; alignment = 0; static_equal = 0; i = 0
    indices = {(r['video_id'], r['frame_index']): j for j, r in enumerate(manifest['val'])}
    torch.cuda.reset_peak_memory_stats(); wall_start = time.perf_counter()
    try:
        with (folder / 'frames.jsonl').open('w', encoding='utf-8') as log:
            while True:
                start = time.perf_counter(); ok, frame = cap.read(); decoded = time.perf_counter()
                if not ok: break
                raw, prediction, application, timing = video_pipeline(experts, frame, config, scheme)
                after_pipeline = time.perf_counter()
                writer.write(np.hstack([panel(frame, 'Input', None, i), panel(frame, scheme, prediction, i)])); rendered = time.perf_counter()
                rgb_hash = hashlib.sha256(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
                if rgb_hash != reference_log[i]['rgb_raw_bytes_sha256'] or hashlib.sha256(raw.tobytes()).hexdigest() != reference_log[i]['mask_raw_bytes_sha256']:
                    raise ValueError('Full-video raw frame/baseline identity differs')
                if np.any((raw == 2) & (prediction != 2)) or np.any((raw == 1) != (prediction == 1)):
                    protections += 1; raise AssertionError('Protected raw labels changed')
                changed += int(not np.array_equal(raw, prediction)); counts[application.get('fallback') or 'applied'] += 1
                if i in selected:
                    row = selected[i]
                    if rgb_hash != row['rgb_raw_bytes_sha256']: raise ValueError('Approved RGB alignment differs')
                    alignment += 1
                    static = load_mask(out / f'video_native/{scheme}/{indices[(vid, i)]:04d}.png', raw.shape, {0, 1, 2})
                    if not np.array_equal(static, prediction): raise ValueError('Static/full-video refinement differs')
                    static_equal += 1
                save_png(folder / 'masks' / f'{i:06d}.png', prediction)
                entry = {'frame_index': i, 'rgb_raw_bytes_sha256': rgb_hash,
                    'raw_mask_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                    'mask_raw_bytes_sha256': hashlib.sha256(prediction.tobytes()).hexdigest(),
                    'approved_semantic_reference': i in selected, 'application': application,
                    'dense_accuracy': None, 'tip_metrics': None}
                log.write(json.dumps(entry) + '\n'); end = time.perf_counter()
                timing.update(decode_ms=(decoded - start) * 1000, render_encode_ms=(rendered - after_pipeline) * 1000,
                    audit_export_ms=(end - rendered) * 1000, cycle_ms=(end - start) * 1000)
                timings.append(timing); i += 1
                if i % 200 == 0: print(f'{scheme} {vid} {i}/{info["frames"]}', flush=True)
    finally: cap.release(); writer.release()
    wall = time.perf_counter() - wall_start
    if i != info['frames'] or alignment != len(selected) or static_equal != len(selected): raise ValueError('Incomplete full-video replay')
    if before != model_state_hash(experts): raise ValueError('Neural weights changed')
    check = cv2.VideoCapture(str(folder / 'replay.mp4')); encoded = 0
    while check.read()[0]: encoded += 1
    check.release()
    if encoded != i: raise ValueError('Video export frame count differs')
    result = {'scheme': scheme, 'video_id': vid, 'pid': os.getpid(), 'frames': i, 'encoded_frames': encoded,
        'fps': fps, 'warmup_frames': config['full_video']['warmup_frames'], 'networks': 3,
        'timings': {k: stats([t[k] for t in timings]) for k in timings[0]}, 'offline_fps_including_all_exports': i / wall,
        'wall_seconds': wall, 'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
        'changed_frames': changed, 'protection_violations': protections, 'fallback_counts': dict(counts),
        'raw_baseline_different_pixels': 0, 'approved_frame_static_matches': static_equal,
        'tip_metrics': None, 'dense_accuracy': None, 'model_state_sha256': before,
        'realtime_p95_passed': stats([t['pipeline_ms'] for t in timings])['p95'] <= 1000 / fps,
        'replay_sha256': sha256(folder / 'replay.mp4')}
    atomic_json(folder / 'timings.json', timings); atomic_json(folder / 'summary.json', result)
    print(json.dumps({'completed': scheme, 'video': vid, 'frames': i, 'p95': result['timings']['pipeline_ms']['p95']}), flush=True)


def comparison_video(out, config, info):
    vid = info['video_id']; cap = cv2.VideoCapture(info['path']); i = 0
    path = out / f'{vid}_comparison.mp4'
    writer = cv2.VideoWriter(str(path), cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*'H264'), info['fps'], (1600, 434))
    if not writer.isOpened(): raise RuntimeError('Comparison H264 encoder unavailable')
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            panels = [panel(frame, 'Input', None, i)]
            for name in config['schemes']:
                mask = load_mask(out / f'full_video/{name}/{vid}/masks/{i:06d}.png', frame.shape[:2], {0, 1, 2})
                panels.append(panel(frame, name, mask, i))
            writer.write(np.hstack(panels)); i += 1
    finally: cap.release(); writer.release()
    check = cv2.VideoCapture(str(path)); decoded = 0
    while check.read()[0]: decoded += 1
    check.release()
    if i != info['frames'] or decoded != i: raise ValueError('Comparison output incomplete')
    return {'path': str(path), 'frames': i, 'decoded_frames': decoded, 'sha256': sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--worker', nargs=2, metavar=('SCHEME', 'VIDEO'))
    args = parser.parse_args(); config = read(CONFIG); out = ROOT / config['output']
    if args.worker: return video_worker(out, config, *args.worker)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(); job = {'pid': os.getpid(), 'training_updates': 0, 'default_changed': False, 'tip_metrics': None}
    def progress(phase, **extra):
        job.update(phase=phase, **extra); atomic_json(out / 'job.json', job); print(phase + ' ' + json.dumps(extra), flush=True)
    progress('preflight')
    try:
        vm_path = ROOT / config['video_manifest']; vm = read(vm_path)
        im_path = ROOT / config['image_manifest']; im = read(im_path)
        old = read(ROOT / 'datasets/ureter_semis_kidney_joint_20261028/video_validation_manifest.json')
        original_indices = {(r['video_id'], r['frame_index']): i for i, r in enumerate(old['val'])}
        if len(vm['val']) != 70 or len(im['val']) != 132: raise ValueError('Frozen static sample scope changed')
        cp = {k: v for k, v in read(ROOT / config['experts'])['checkpoints'].items() if k in {'old190', 'A0'}}
        videos = read(ROOT / 'runs/baselines/a0_20261027/config.json')['inputs']
        for item in [*cp.values(), *videos]:
            if sha256(item['path']) != item['sha256']: raise ValueError('Model/video identity differs')
        input_hash = input_identity(im_path, Path(im['root']), {'train': {'labeled': [], 'unlabeled': []}, 'val': im['val'] + vm['val']})
        extra = [CONFIG, vm_path, ROOT / config['experts'], ROOT / 'configs/fiber_tip_d0_20261029.json', ROOT / 'runs/a1_classwise_experts_20261028/summary.json']
        for video, rows in [(True, vm['val']), (False, im['val'])]:
            for i, row in enumerate(rows):
                cache, raw = cached_paths(row, i, video, original_indices)
                extra.append(raw)
                if cache: extra.append(cache)
        related = {str(p): sha256(p) for p in extra}
        sources = {Path(p).as_posix(): digest for p, digest in source_identity().items()}
        for p in list((ROOT / 'tools').glob('*.py')) + list((ROOT / 'tests').glob('*.py')):
            sources[p.relative_to(ROOT).as_posix()] = sha256(p)
        pre = {'config': config, 'config_sha256': sha256(CONFIG), 'sources': sources, 'related': related,
            'input_identity': input_hash, 'checkpoints': cp, 'videos': videos, 'train_pixels_read': 0, 'reserved_test_pixels_read': 0}
        atomic_json(out / 'preflight.json', pre)
        with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in sources: archive.write(ROOT / name, name)
            archive.write(CONFIG, CONFIG.relative_to(ROOT).as_posix())
        seed_training(42); experts = ClasswiseExperts(cp['old190']['path'], cp['A0']['path']); before = model_state_hash(experts)
        atomic_json(out / 'runtime.json', runtime_identity(experts.device))
        settings = read(ROOT / 'configs/fiber_tip_d0_20261029.json')['diagnostic_only']
        progress('static_image132'); image = static_set(im['val'], False, experts, out, config, original_indices, settings, progress)
        progress('static_video70'); video = static_set(vm['val'], True, experts, out, config, original_indices, settings, progress)
        if before != model_state_hash(experts): raise ValueError('Weights changed during static work')
        del experts; torch.cuda.empty_cache()
        quality = gates(video, image, config); atomic_json(out / 'static_gates.json', quality)
        full = {}
        for scheme in config['schemes']:
            full[scheme] = {}
            for info in videos:
                vid = info['video_id']; progress('full_video_fresh_process', scheme=scheme, video_id=vid)
                with (out / f'{scheme}_{vid}.log').open('w', encoding='utf-8') as log:
                    subprocess.run([sys.executable, '-m', 'tools.run_fiber_blue_aplus', '--worker', scheme, vid], cwd=ROOT,
                        stdout=log, stderr=subprocess.STDOUT, check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                full[scheme][vid] = read(out / f'full_video/{scheme}/{vid}/summary.json')
        exports = {}
        for info in videos:
            progress('comparison_video_export', video_id=info['video_id']); exports[info['video_id']] = comparison_video(out, config, info)
        if any(sha256(ROOT / p) != h for p, h in sources.items()) or any(sha256(p) != h for p, h in related.items()):
            raise ValueError('Frozen sources/inputs changed during execution')
        if input_identity(im_path, Path(im['root']), {'train': {'labeled': [], 'unlabeled': []}, 'val': im['val'] + vm['val']}) != input_hash:
            raise ValueError('Validation labels changed during execution')
        result = {'static_quality_gates': quality, 'full_video': full, 'comparison_videos': exports,
            'static_video_frames': len(vm['val']), 'static_image_frames': len(im['val']), 'full_video_frames_per_scheme': 2694,
            'static_results': {'video': str(out / 'video_native/summary.json'), 'image': str(out / 'image_native/summary.json')},
            'adopted': False, 'tip_metrics': None, 'whole_video_accuracy_estimates': None, 'training_updates': 0,
            'train_pixels_read': 0, 'reserved_test_pixels_read': 0, 'default_changed': False, 'stone_memory_started': False}
        atomic_json(out / 'summary.json', result); progress('completed')
    except Exception as error:
        progress('failed', error=str(error)); (out / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8'); raise


if __name__ == '__main__': main()
