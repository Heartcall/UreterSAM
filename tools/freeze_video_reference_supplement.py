"""Freeze approved D1 frames as a video development test supplement and audit cached schemes."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image

from ureter_semis.experiment import atomic_json
from .fiber_reference_workspace import DEFAULT, ROOT, load_mask, read, sha256, validate_reference
from .fiber_reference_evaluation import confusion, paired, summarize_observed, stratified_estimate
from .evaluate_spatial_detail import boundary_counts


def within(parent, relative):
    parent = Path(parent).resolve()
    path = (parent / relative).resolve()
    path.relative_to(parent)
    return path


def approved_semantic_decisions(manifest, state):
    """A partial human selection is useful; pending and tip labels are not semantic truth."""
    records = {r['id']: r for r in manifest['records']}
    selected = []
    for key, decision in state['decisions'].items():
        if key not in records:
            raise ValueError('Unknown review record: ' + key)
        row = records[key]
        if row['kind'] != 'semantic' or decision['status'] != 'approved':
            continue
        if row.get('reference_mask'):
            raise ValueError('Original29 read-only reference cannot be replaced by a supplement')
        if not decision.get('reviewer', '').strip() or decision.get('independent_review') is not True or decision.get('whole_frame_reviewed') is not True:
            raise ValueError('Whole-frame approval metadata incomplete: ' + key)
        selected.append((row, decision))
    return sorted(selected, key=lambda pair: (pair[0]['video_id'], pair[0]['frame_index']))


def verify_rgb(row):
    image = Path(row['image'])
    if sha256(image) != row['image_sha256']:
        raise ValueError('Reference image file changed')
    with Image.open(image) as source:
        rgb = np.asarray(source.convert('RGB'))
    if rgb.shape != (row['height'], row['width'], 3) or hashlib.sha256(rgb.tobytes()).hexdigest() != row['rgb_raw_bytes_sha256']:
        raise ValueError('Reference frame does not match verified sequential video RGB')


def freeze(workspace, output):
    workspace, output = Path(workspace).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Never overwrite a frozen reference snapshot: ' + str(output))
    live_manifest = workspace / 'manifest.json'
    manifest = read(live_manifest)
    session_path = workspace / 'session/session.json'
    session_bytes = session_path.read_bytes()  # Atomic commit point; never reload mutable exports.
    state = json.loads(session_bytes)
    if state['inputs']['manifest_sha256'] != sha256(live_manifest):
        raise ValueError('Review session source manifest changed')
    for path, digest in manifest['inputs'].items():
        if sha256(path) != digest:
            raise ValueError('Frozen review input changed: ' + path)
    for name, digest in manifest['sources'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Frozen D1 source changed: ' + name)
    for video in manifest['videos']:
        if sha256(video['path']) != video['sha256']:
            raise ValueError('Original video changed')
    new = approved_semantic_decisions(manifest, state)
    if not new:
        raise ValueError('No new approved whole-frame reference to add')
    original = [r for r in manifest['records'] if r['kind'] == 'semantic' and r.get('reference_mask')]
    if len(original) != 29:
        raise ValueError('Original29 reference scope changed')
    prepared = []
    for row, decision in [(r, None) for r in original] + new:
        verify_rgb(row)
        mask_path = within(workspace / 'session', decision['mask']) if decision else Path(row['reference_mask'])
        digest = decision['mask_sha256'] if decision else row['reference_mask_sha256']
        if sha256(mask_path) != digest:
            raise ValueError('Reviewed reference mask changed')
        mask = load_mask(mask_path, (row['height'], row['width']), {0, 1, 2, 255})
        if decision:
            if decision['image_sha256'] != row['image_sha256']:
                raise ValueError('Decision/image identity mismatch')
            validated = validate_reference(row, decision, mask)
            if any(validated[k] != decision[k] for k in ('unknown_pixels', 'valid_pixels')):
                raise ValueError('Saved reference pixel counts disagree with the actual mask')
        elif not (mask != 255).any():
            raise ValueError('Original reference contains no known pixels')
        cohort = 'new_user_approved' if decision else 'original29'
        folder = output / 'frames' / row['video_id'] / f'{row["frame_index"]:06d}'
        record = {**row, 'sample_id': f'reviewed_{row["video_id"]}_{row["frame_index"]:06d}',
            'evaluation_cohort': cohort, 'image': str(folder / 'image.png'), 'mask': str(folder / 'mask.png'),
            'image_sha256': row['image_sha256'], 'mask_sha256': digest,
            'review_status': 'approved', 'reviewer': decision['reviewer'] if decision else row['reference_reviewer'],
            'source_revision': decision['revision'] if decision else 0,
            'annotation_basis': decision['annotation_basis'] if decision else 'inherited_frozen_original29_manual_review',
            'source_image': row['image'], 'source_mask': str(mask_path),
            'note': decision.get('note', '') if decision else '',
            'saved_utc': decision.get('saved_utc') if decision else None,
            'stone_presence_review': 'present' if (mask == 1).any() else 'absent' if (mask != 255).all() else 'unknown',
            'unknown_pixels': int((mask == 255).sum()), 'valid_pixels': int((mask != 255).sum()),
            'class_pixels': {str(c): int((mask == c).sum()) for c in (0, 1, 2, 255)},
            'training_allowed': False, 'group_id': row['patient_id']}
        prepared.append((record, Path(row['image']), mask_path))
    if len({r['id'] for r, _, _ in prepared}) != len(prepared):
        raise ValueError('Duplicate video frame in the union')
    output.mkdir(parents=True)
    rows = []
    for row, source_image, source_mask in sorted(prepared, key=lambda x: (x[0]['video_id'], x[0]['frame_index'])):
        Path(row['image']).parent.mkdir(parents=True)
        shutil.copyfile(source_image, row['image']); shutil.copyfile(source_mask, row['mask'])
        if sha256(row['image']) != row['image_sha256'] or sha256(row['mask']) != row['mask_sha256']:
            raise ValueError('Reference changed while copying; snapshot is incomplete')
        rows.append(row)
    (output / 'session_at_freeze.json').write_bytes(session_bytes)
    shutil.copyfile(live_manifest, output / 'workspace_manifest_at_freeze.json')
    source_paths = [Path(__file__), ROOT / 'tests/test_video_reference_supplement.py',
        ROOT / 'tools/fiber_reference_workspace.py', ROOT / 'tools/fiber_reference_evaluation.py',
        ROOT / 'tools/evaluate_spatial_detail.py', ROOT / 'ureter_semis/experiment.py']
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    with zipfile.ZipFile(output / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in source_hashes:
            archive.write(ROOT / name, name.replace('\\', '/'))
    dataset_manifest_path = ROOT / 'datasets/ureter_semis_kidney_joint_20261028/manifest.json'
    config_path = ROOT / 'configs/fiber_reference_d1_20261029.json'
    cohorts = Counter(r['evaluation_cohort'] for r in rows)
    coverage = {}
    for video in manifest['videos']:
        vid = video['video_id']; pop = [r for r in manifest['records'] if r['kind'] == 'semantic' and r['video_id'] == vid]
        selected = {r['id'] for r in rows if r['video_id'] == vid}
        coverage[vid] = {'video_frames': video['frames'], 'original29': sum(r['video_id'] == vid for r in original),
            'new_approved': sum(r['video_id'] == vid for r, _ in new), 'combined': len(selected),
            'planned_random_sample': sum(r['sampling_selected'] for r in pop),
            'approved_random_sample': sum(r['sampling_selected'] and r['id'] in selected for r in pop)}
    payload = {'schema_version': 3, 'root': str(output), 'role': 'video_development_test_only_never_training',
        'classes': ['background', 'stone', 'fiber'], 'ignore_index': 255, 'training_ready': False,
        'train': {'labeled': [], 'unlabeled': []}, 'val': rows, 'test': [], 'reserved_unlabeled': [],
        'counts': {'original': cohorts['original29'], 'new': cohorts['new_user_approved'], 'combined': len(rows)},
        'coverage': coverage, 'videos': manifest['videos'], 'sampling': manifest['sampling'],
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'source_session': str(session_path), 'source_session_sha256': hashlib.sha256(session_bytes).hexdigest(),
        'source_workspace_manifest_sha256': sha256(live_manifest),
        'live_session_advanced_during_snapshot': sha256(session_path) != hashlib.sha256(session_bytes).hexdigest(),
        'sources': source_hashes, 'original_training_split_manifest': str(dataset_manifest_path),
        'original_training_split_manifest_sha256': sha256(dataset_manifest_path),
        'prediction_config_sha256': sha256(config_path),
        'independent_test_available': False, 'tip_reference_promoted_from_semantic': False,
        'tip_metrics': None, 'instance_reference_available': False,
        'evaluation_limitations': ['Original29 and new approved frames are development video references, not independent final generalization tests',
            'User selected a partial subset of the fixed120 probability sample; no full-video ratio estimates until planned references are complete',
            'Unknown255 excluded and counted; visible semantic fiber masks do not provide separate anatomical tip labels',
            'Original521/132/144/29 split and all historical gates/results unchanged; all video derivatives excluded from training']}
    atomic_json(output / 'manifest.json', payload)
    atomic_json(output / 'new_frames_manifest.json', {**payload, 'val': [r for r in rows if r['evaluation_cohort'] == 'new_user_approved'],
        'role': 'new_user_selected_video_development_test_supplement_only', 'counts': {'new': cohorts['new_user_approved']}})
    return output / 'manifest.json'


def evaluate(snapshot):
    snapshot = Path(snapshot).resolve()
    manifest_path = snapshot / 'manifest.json'; manifest = read(manifest_path)
    output = snapshot / 'comparison'
    if output.exists(): raise FileExistsError(output)
    config_path = ROOT / 'configs/fiber_reference_d1_20261029.json'; config = read(config_path)
    if sha256(config_path) != manifest['prediction_config_sha256']:
        raise ValueError('Frozen prediction configuration changed')
    for name, digest in manifest['sources'].items():
        if sha256(ROOT / name) != digest: raise ValueError('Snapshot evaluation source changed')
    population = [r for r in read(snapshot / 'workspace_manifest_at_freeze.json')['records'] if r['kind'] == 'semantic']
    rows = manifest['val']; lookup = {(r['video_id'], r['frame_index']): r for r in rows}
    targets, baselines = {}, {}
    for row in rows:
        if sha256(row['image']) != row['image_sha256'] or sha256(row['mask']) != row['mask_sha256']:
            raise ValueError('Frozen evaluation reference changed')
        shape = (row['height'], row['width'])
        targets[row['id']] = load_mask(row['mask'], shape, {0, 1, 2, 255})
        baselines[row['id']] = load_mask(ROOT / config['schemes']['classwise'].format(video=row['video_id'], frame=row['frame_index']), shape, {0, 1, 2})
    old_audit = ROOT / 'runs/fiber_reference_d1_20261029/initial_accuracy_audit/summary.json'
    historical = read(old_audit)
    output.mkdir()
    models = {}; checks = []
    for name, pattern in config['schemes'].items():
        previous = historical['models'][name]
        if not previous['prediction_complete']: raise ValueError('Historical prediction coverage incomplete')
        identity_log = Path(previous['all_frame_output_identity_log'])
        if sha256(identity_log) != previous['identity_log_sha256']:
            raise ValueError('Historical all-frame identity log changed')
        identity_rows = [json.loads(line) for line in identity_log.read_text(encoding='utf-8').splitlines()]
        if len(identity_rows) != len(population): raise ValueError('Full-video prediction count differs')
        observations = {}; per_frame = []; changed = []
        for old, pop in zip(identity_rows, population):
            if old['id'] != pop['id']: raise ValueError('Frame identity order mismatch')
            path = ROOT / pattern.format(video=pop['video_id'], frame=pop['frame_index'])
            if str(path) != old['prediction'] or sha256(path) != old['prediction_sha256']:
                changed.append(pop['id']); continue
            row = lookup.get((pop['video_id'], pop['frame_index']))
            if row is None: continue
            pred = load_mask(path, (row['height'], row['width']), {0, 1, 2}); target = targets[row['id']]
            current = {'confusion_matrix': confusion(pred, target).tolist(), 'unknown_pixels': int((target == 255).sum()),
                'boundaries': {str(c): boundary_counts(pred, target, c, 3) for c in (1, 2)},
                'paired': {label: paired(target, baselines[row['id']], pred, c) for c, label in ((1, 'stone'), (2, 'fiber'))}}
            observations[row['id']] = current
            per_frame.append({'id': row['id'], 'video_id': row['video_id'], 'frame_index': row['frame_index'],
                              'cohort': row['evaluation_cohort'], 'reference_sha256': row['mask_sha256'],
                              'prediction_sha256': old['prediction_sha256'], **current})
        if changed or len(observations) != len(rows): raise ValueError(f'{name}: changed outputs or missing reference predictions: {changed}')
        old_summary = summarize_observed([r for r in rows if r['evaluation_cohort'] == 'original29'], observations)
        if old_summary != previous['sparse_video29']:
            raise ValueError('Original29 regression result changed: ' + name)
        complete = summarize_observed(rows, observations)
        new_summary = summarize_observed([r for r in rows if r['evaluation_cohort'] == 'new_user_approved'], observations)
        case_results = {}
        for vid in ('video_1', 'video_2'):
            case_rows = [r for r in rows if r['video_id'] == vid]
            pop = [r for r in population if r['video_id'] == vid]
            case_results[vid] = {'combined_approved': summarize_observed(case_rows, observations),
                'new_approved': summarize_observed([r for r in case_rows if r['evaluation_cohort'] == 'new_user_approved'], observations),
                'stratified_whole_video_estimate': stratified_estimate(pop, observations), 'dense_exact_accuracy': None}
        models[name] = {'all_full_video_cached_outputs_reverified': len(identity_rows), 'changed_output_files': len(changed),
            'original29': old_summary, 'new_approved': new_summary, 'combined_approved': complete,
            'cases': case_results, 'stratified_pooled_estimate': stratified_estimate(population, observations),
            'dense_exact_accuracy': None, 'tip_metrics': None, 'frames': per_frame}
        atomic_json(output / f'{name}.json', models[name])
        checks.append({'scheme': name, 'historical29_exact': True, 'full_video_output_files_unchanged': len(identity_rows),
                       'labelled_frames': len(observations)})
        print(f'{name}: full output identities {len(identity_rows)}, references {len(observations)}, new {new_summary["frames"]}', flush=True)
    # Check mask/image conservation and source-state identities without touching live exports.
    for row in rows:
        if sha256(row['source_image']) != row['image_sha256'] or sha256(row['source_mask']) != row['mask_sha256']:
            raise ValueError('Source reference changed while evaluating')
    if sha256(manifest['original_training_split_manifest']) != manifest['original_training_split_manifest_sha256']:
        raise ValueError('Original training/development/reserved split changed')
    result = {'scope': 'Frozen approved-frame development test, including partial user selection; not full-video accuracy or a new latency benchmark',
        'snapshot_manifest_sha256': sha256(manifest_path), 'snapshot_source_sha256': sha256(snapshot / 'source_snapshot.zip'),
        'old_all_video_audit_sha256': sha256(old_audit), 'counts': manifest['counts'], 'coverage': manifest['coverage'],
        'models': {name: {k: v for k, v in m.items() if k != 'frames'} for name, m in models.items()},
        'checks': checks, 'checks_passed': True, 'networks_loaded': 0, 'gradient_updates': 0,
        'train_pixels_read': 0, 'reserved_test_pixels_read': 0, 'default_changed': False,
        'live_session_modified': False, 'original29_reference_files_unchanged': True}
    atomic_json(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/review_snapshots/video_development_supplement_20261029_v1')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    path = freeze(args.workspace, args.output)
    print(json.dumps({'snapshot': str(path), 'counts': read(path)['counts'], 'coverage': read(path)['coverage']}), flush=True)
    if args.evaluate: evaluate(args.output)


if __name__ == '__main__': main()
