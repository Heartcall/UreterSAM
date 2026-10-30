"""Audit complete-video predictions; separate sparse, sampled-estimated and dense accuracy."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from ureter_semis.experiment import atomic_json
from .evaluate_spatial_detail import boundary_counts, boundary_summary
from .fiber_reference_workspace import DEFAULT, ROOT, ReferenceStore, load_mask, read, sha256


def confusion(prediction, target):
    valid = target != 255
    if prediction.shape != target.shape or not set(np.unique(prediction)) <= {0, 1, 2}:
        raise ValueError('Prediction must contain native semantic classes0/1/2')
    return np.bincount((target[valid].astype(int) * 3 + prediction[valid]).ravel(), minlength=9).reshape(3, 3)


def metrics(matrix):
    matrix = np.asarray(matrix)
    result = {'confusion_matrix': matrix.tolist(), 'valid_pixels': float(matrix.sum())}
    for c, name in ((1, 'stone'), (2, 'fiber')):
        tp = float(matrix[c, c]); fp = float(matrix[:, c].sum() - tp); fn = float(matrix[c, :].sum() - tp)
        result[name] = {'tp': tp, 'fp': fp, 'fn': fn,
                        'dice': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                        'iou': tp / (tp + fp + fn) if tp + fp + fn else None,
                        'precision': tp / (tp + fp) if tp + fp else None,
                        'recall': tp / (tp + fn) if tp + fn else None}
    return result


def paired(target, baseline, prediction, label):
    valid = target != 255; truth = target == label
    before = baseline == label; after = prediction == label
    return {'removed_correct_pixels': int((before & ~after & truth).sum()),
            'recovered_correct_pixels': int((~before & after & truth).sum()),
            'removed_fp': int((before & ~after & ~truth & valid).sum()),
            'added_fp': int((~before & after & ~truth & valid).sum())}


def tip_metrics(prediction, tip_reference, roi):
    """Confirmed shaft and unknown pixels do not count as tip negatives."""
    x0, y0, x1, y1 = roi
    region = np.zeros(tip_reference.shape, bool); region[y0:y1, x0:x1] = True
    valid = region & np.isin(tip_reference, [0, 1])
    truth = tip_reference == 1; predicted = prediction == 2
    tp = int((valid & truth & predicted).sum()); fp = int((valid & ~truth & predicted).sum())
    fn = int((valid & truth & ~predicted).sum())
    boundary_target = np.full(tip_reference.shape, 255, np.uint8)
    boundary_target[valid & ~truth] = 0; boundary_target[valid & truth] = 2
    return {'tp': tp, 'fp': fp, 'fn': fn, 'valid_pixels': int(valid.sum()),
            'recall': tp / (tp + fn) if tp + fn else None,
            'precision': tp / (tp + fp) if tp + fp else None,
            'dice': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            'boundary': boundary_summary([boundary_counts(prediction, boundary_target, 2, 3)]),
            'tip_localization_error': None,
            'tip_localization_reason': 'These semantic schemes do not output a validated visible-distal-point prediction',
            'scope': 'manual_tip_ROI; confirmed_shaft_and_unknown_excluded; no_physical_tip_localization'}


def ratio_interval(tp, fp, fn, covariance, metric):
    """First-order design variance, conditional on the fixed finite video population."""
    if metric == 'dice':
        d = 2 * tp + fp + fn
        if not d: return None
        value = 2 * tp / d; gradient = np.array([2 * (fp + fn), -2 * tp, -2 * tp]) / d ** 2
    elif metric == 'precision':
        d = tp + fp
        if not d: return None
        value = tp / d; gradient = np.array([fp, -tp, 0]) / d ** 2
    else:
        d = tp + fn
        if not d: return None
        value = tp / d; gradient = np.array([fn, 0, -tp]) / d ** 2
    se = float(np.sqrt(max(0, gradient @ covariance @ gradient)))
    return {'estimate': value, 'standard_error': se, 'approximate_95_interval': [max(0, value - 1.96 * se), min(1, value + 1.96 * se)]}


def stratified_estimate(population, observed):
    """No complete-population estimate until every preselected sample is approved."""
    selected = [r for r in population if r['sampling_selected']]
    missing = [r['id'] for r in selected if r['id'] not in observed]
    if missing:
        return {'status': 'awaiting_all_planned_sample_references', 'planned': len(selected),
                'approved_and_predicted': len(selected) - len(missing), 'missing_ids': missing, 'metrics': None}
    strata = defaultdict(list)
    for row in selected:
        strata[row['stratum_id']].append(row)
    estimated = np.zeros((3, 3), float); variances = {1: np.zeros((3, 3)), 2: np.zeros((3, 3))}
    estimated_unknown = 0.; boundary = {1: [], 2: []}
    for rows in strata.values():
        N, n = rows[0]['stratum_N'], rows[0]['stratum_n']
        if len(rows) != n or n < 2 or n > N:
            raise ValueError('Invalid fixed sampling stratum')
        weight = N / n
        matrices = [np.asarray(observed[r['id']]['confusion_matrix'], float) for r in rows]
        estimated += weight * sum(matrices)
        estimated_unknown += weight * sum(observed[r['id']]['unknown_pixels'] for r in rows)
        for c in (1, 2):
            vectors = np.array([[m[c, c], m[:, c].sum() - m[c, c], m[c, :].sum() - m[c, c]] for m in matrices])
            variances[c] += N ** 2 * (1 - n / N) * np.cov(vectors, rowvar=False, ddof=1) / n
            boundary[c].extend({k: v * weight for k, v in observed[r['id']]['boundaries'][str(c)].items()} for r in rows)
    result = metrics(estimated)
    result.update(estimated_unknown_pixels=estimated_unknown,
                  estimated_unknown_fraction=estimated_unknown / (estimated_unknown + estimated.sum()) if estimated_unknown + estimated.sum() else None)
    return {'status': 'complete_probability_sample_not_dense_truth', 'planned': len(selected), 'metrics': result,
            'boundaries': {str(c): boundary_summary(boundary[c]) for c in (1, 2)},
            'intervals': {name: {metric: ratio_interval(result[name]['tp'], result[name]['fp'], result[name]['fn'], variances[c], metric)
                               for metric in ('dice', 'precision', 'recall')} for c, name in ((1, 'stone'), (2, 'fiber'))},
            'scope': 'N_h/n_h weighted finite-video ratio estimates; approximate design intervals, not independent-patient inference'}


def summarize_observed(records, observed):
    available = [observed[r['id']] for r in records if r['id'] in observed]
    if not available:
        return {'frames': 0, 'metrics': None}
    return {'frames': len(available), 'metrics': metrics(sum(np.asarray(r['confusion_matrix']) for r in available)),
            'unknown_pixels': sum(r['unknown_pixels'] for r in available),
            'boundaries': {str(c): boundary_summary([r['boundaries'][str(c)] for r in available]) for c in (1, 2)},
            'paired_with_classwise': {name: {k: sum(r['paired'][name][k] for r in available) for k in available[0]['paired'][name]}
                                     for name in ('stone', 'fiber')}}


def evaluate(workspace=DEFAULT, output=None, scheme_names=None):
    workspace = Path(workspace).resolve()
    output = Path(output).resolve() if output else workspace / 'initial_accuracy_audit'
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    config_path = ROOT / 'configs/fiber_reference_d1_20261029.json'
    config = read(config_path)
    store = ReferenceStore(workspace / 'manifest.json', workspace / 'session')
    manifest = read(store.manifest_path)
    for path, digest in manifest['inputs'].items():
        if sha256(path) != digest:
            raise ValueError('Frozen reference preparation input changed')
    for video in manifest['videos']:
        if sha256(video['path']) != video['sha256']:
            raise ValueError('Video changed')
    for path, digest in manifest['sources'].items():
        if sha256(ROOT / path) != digest:
            raise ValueError('Frozen reference preparation source changed')
    export_path = store.directory / 'references.json'
    refs = read(export_path)['records']
    population = [r for r in refs if r['kind'] == 'semantic']
    tip_refs = [r for r in refs if r['kind'] == 'tip' and r['eligible'] and r.get('tip_metrics_eligible')]
    targets = {}
    for row in refs:
        if not row['eligible']: continue
        store.verify(row)
        if sha256(row['mask']) != row['mask_sha256']:
            raise ValueError('Reference hash changed')
        targets[row['id']] = load_mask(row['mask'], (row['height'], row['width']), {0, 1, 2, 255})
    schemes = config['schemes']
    names = scheme_names or list(schemes)
    models = {}; index = []
    baseline_cache = {}
    for row in population:
        if row['id'] in targets:
            p = ROOT / schemes[config['default_baseline']].format(video=row['video_id'], frame=row['frame_index'])
            baseline_cache[row['id']] = load_mask(p, (row['height'], row['width']), {0, 1, 2})
    for name in names:
        observed = {}; predicted = {}; failures = []; tip_results = []
        log_path = output / f'{name}_frames.jsonl'
        tip_by_frame = {(r['video_id'], r['frame_index']): r for r in tip_refs}
        with log_path.open('w', encoding='utf-8') as log:
            for row in population:
                path = ROOT / schemes[name].format(video=row['video_id'], frame=row['frame_index'])
                entry = {'id': row['id'], 'video_id': row['video_id'], 'frame_index': row['frame_index'],
                         'sampling_selected': row['sampling_selected'], 'reference_available': row['eligible'],
                         'prediction': str(path), 'metrics': None}
                try:
                    pred = load_mask(path, (row['height'], row['width']), {0, 1, 2})
                    entry['prediction_sha256'] = sha256(path)
                    predicted[row['id']] = True
                    if row['id'] in targets:
                        target = targets[row['id']]
                        current = {'confusion_matrix': confusion(pred, target).tolist(),
                                   'unknown_pixels': int((target == 255).sum()),
                                   'boundaries': {str(c): boundary_counts(pred, target, c, config['boundary_tolerance_native_pixels']) for c in (1, 2)},
                                   'paired': {label: paired(target, baseline_cache[row['id']], pred, c) for c, label in ((1, 'stone'), (2, 'fiber'))}}
                        observed[row['id']] = current
                        entry['metrics'] = {**metrics(current['confusion_matrix']), **current}
                    tip = tip_by_frame.get((row['video_id'], row['frame_index']))
                    if tip:
                        tip_results.append({'id': tip['id'], **tip_metrics(pred, targets[tip['id']], tip['tip_roi'])})
                except (OSError, ValueError) as error:
                    entry['error'] = str(error); failures.append({'id': row['id'], 'error': str(error)})
                log.write(json.dumps(entry) + '\n')
        cases = {}
        for vid in ('video_1', 'video_2'):
            rows = [r for r in population if r['video_id'] == vid]
            approved = sum(r['eligible'] for r in rows)
            complete = all(r['id'] in predicted for r in rows)
            cases[vid] = {'total_frames': len(rows), 'predicted_frames': sum(r['id'] in predicted for r in rows),
                          'approved_reference_frames': approved, 'reference_coverage_fraction': approved / len(rows),
                          'sampled_frames': sum(r['sampling_selected'] for r in rows),
                          'sampled_approved': sum(r['sampling_selected'] and r['eligible'] for r in rows),
                          'sparse_reviewed_diagnostic': summarize_observed([r for r in rows if r.get('reference_mask')], observed),
                          'stratified_whole_video_estimate': stratified_estimate(rows, observed) if complete else {'status': 'incomplete_prediction', 'metrics': None},
                          'dense_exact_accuracy': summarize_observed(rows, observed) if complete and approved == len(rows) else None}
        pooled_complete = len(predicted) == len(population)
        models[name] = {'prediction_complete': pooled_complete, 'prediction_frames': len(predicted), 'failures': failures,
                        'cases': cases, 'sparse_video29': summarize_observed([r for r in population if r.get('reference_mask')], observed),
                        'stratified_pooled_estimate': stratified_estimate(population, observed) if pooled_complete else {'status': 'incomplete_prediction', 'metrics': None},
                        'dense_exact_accuracy': summarize_observed(population, observed) if pooled_complete and len(observed) == len(population) else None,
                        'tip_metrics': tip_results or None, 'all_frame_output_identity_log': str(log_path), 'identity_log_sha256': sha256(log_path)}
        print(f'{name}: outputs {len(predicted)}/{len(population)}, approved references {len(observed)}, dense accuracy unavailable={models[name]["dense_exact_accuracy"] is None}', flush=True)
    result = {'schema_version': 1, 'models': models, 'config_sha256': sha256(config_path),
              'manifest_sha256': sha256(store.manifest_path), 'reference_export_sha256': sha256(export_path),
              'source_snapshot_sha256': sha256(workspace / 'source_snapshot.zip'),
              'scope': 'Cached native-output accuracy and coverage audit, not inference or a latency measurement. All references are repeated development validation.',
              'sampling_policy': config['sampling'], 'networks_loaded': 0, 'training_updates': 0,
              'train_pixels_read': 0, 'reserved_test_pixels_read': 0, 'default_model_changed': False}
    artifacts = ['runs/a1_classwise_experts_20261028/predictor_manifest.json',
                 'runs/a1_fiber_tip_d0_20261029/preflight.json',
                 'runs/a1_classwise_replay_20261028/preflight.json',
                 'runs/a1_fiber_geometry_20261028/preflight.json',
                 'runs/a1_fiber_component_gate_20261029/preflight.json',
                 'runs/a1_direct_fiber_template_20261029/preflight.json',
                 'runs/a1_direct_neural_observation_20261029/preflight.json',
                 'runs/prior_method_comparison_20261029/preflight.json']
    result['historical_model_config_source_identity_artifacts'] = {p: sha256(ROOT / p) for p in artifacts}
    atomic_json(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--schemes', nargs='+')
    args = parser.parse_args()
    evaluate(args.workspace, args.output, args.schemes)


if __name__ == '__main__':
    main()
