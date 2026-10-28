"""Independent exported-mask audit; reads validation pixels only."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/a1_classwise_experts_20261028'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    with Image.open(path) as im:
        return np.asarray(im).copy()


def main():
    s = read(OUT / 'summary.json'); p = read(OUT / 'preflight.json')
    mpath = ROOT / 'datasets/ureter_semis_kidney_joint_20261028/manifest.json'
    m = read(mpath); vm = read(mpath.parent / 'video_validation_manifest.json')
    checks = {}
    checks['frozen_config_sources_related_and_checkpoints_unchanged'] = (
        digest(ROOT / 'configs/compare_classwise_experts_20261028.json') == p['protocol_sha256']
        and all(digest(ROOT / name) == h for name, h in p['sources'].items())
        and all(digest(name) == h for name, h in p['related'].items())
        and all(digest(v['path']) == v['sha256'] for v in p['checkpoints'].values()))
    checks['validation_inputs_unchanged'] = digest(mpath) == p['inputs']['manifest_sha256'] and all(
        digest(path) == h for path, h in p['inputs']['files'].items())
    expected_inputs = {str(Path(r[k]).resolve()) for r in m['val'] + vm['val'] for k in ('image', 'mask')}
    checks['only161_validation_images_and_masks_in_pixel_input_inventory'] = (
        set(p['inputs']['files']) == expected_inputs and len(expected_inputs) == 322
        and p['inputs']['test_pixels_read'] is False)
    checks['dataset_counts_unchanged'] = (len(m['train']['labeled']), len(m['val']), len(m['reserved_test']),
        len(vm['val']), len(m['train']['unlabeled'])) == (521, 132, 144, 29, 0)
    checks['no_training_or_default_change'] = s['training_updates'] == 0 and s['default_model_changed'] is False
    checks['initial_screening_passed_but_no_adoption'] = s['screening']['passed'] and not s['adopted'] and not s['s2_accepted']
    details = {}
    all_masks = {}
    for scope, rows in [('video', vm['val']), ('image', m['val'])]:
        data = s[scope]; folder = OUT / f'{scope}_native'
        predictions = {name: [load(folder / name / f'{i:04d}.png') for i in range(len(rows))]
                       for name in data['models']}
        targets = [load(r['mask']) for r in rows]
        all_masks[scope] = (rows, targets, predictions)
        checks[f'{scope}_native_shapes_and_class_values'] = all(
            a.shape == target.shape and set(np.unique(a)).issubset({0, 1, 2})
            for name, arrays in predictions.items() for a, target in zip(arrays, targets))
        checks[f'{scope}_hard_policy_pixel_exact'] = all(np.array_equal(c, np.where(a == 2, 2, np.where(o == 1, 1, 0)))
            for a, o, c in zip(predictions['A0'], predictions['old190'], predictions['classwise']))
        checks[f'{scope}_fiber_identical_to_a0_stone_subset_of_old190'] = all(
            np.array_equal(c == 2, a == 2) and not ((c == 1) & (o != 1)).any()
            for a, o, c in zip(predictions['A0'], predictions['old190'], predictions['classwise']))
        matrix_ok = True; negatives = True
        for name, arrays in predictions.items():
            matrices = []
            negative_pixels = negative_fp = negative_frames = negative_fp_frames = 0
            for row, target, a, frame in zip(rows, targets, arrays, data['models'][name]['frames']):
                valid = target != 255
                cm = np.bincount((3 * target[valid].astype(int) + a[valid]).ravel(), minlength=9).reshape(3, 3)
                matrices.append(cm); matrix_ok &= cm.tolist() == frame['confusion_matrix']
                if row.get('stone_presence_review') == 'absent':
                    fp = int(((a == 1) & valid & (target != 1)).sum())
                    negative_frames += 1; negative_fp_frames += int(fp > 0)
                    negative_pixels += int(valid.sum()); negative_fp += fp
            metrics = data['models'][name]['metrics']
            matrix_ok &= np.sum(matrices, axis=0).tolist() == metrics['confusion_matrix']
            negatives &= (negative_frames, negative_pixels, negative_fp, negative_fp_frames) == (
                metrics['stone_negative_images'], metrics['stone_negative_valid_pixels'],
                metrics['stone_negative_fp_pixels'], metrics['stone_negative_images_with_fp'])
        checks[f'{scope}_independent_numpy_confusions_all_four_models'] = bool(matrix_ok)
        checks[f'{scope}_reviewed_negative_counts_reproduced'] = bool(negatives)
        conflicts = {str(k): 0 for k in (0, 1, 2, 255)}
        for target, a, o in zip(targets, predictions['A0'], predictions['old190']):
            overlap = (o == 1) & (a == 2)
            for k in conflicts: conflicts[k] += int((overlap & (target == int(k))).sum())
        checks[f'{scope}_conflict_partitions_recomputed'] = conflicts == data['conflicts']['by_reference']
        details[scope] = {'conflicts_by_reference': conflicts}
    cohorts = s['image']['historical_exposure_cohorts']
    exposed = set(cohorts['historical_old190_exposed']['sample_ids'])
    unmatched = set(cohorts['not_matched_to_old190_exposure']['sample_ids'])
    checks['image25_107_cohorts_disjoint_complete'] = (len(exposed), len(unmatched), len(exposed & unmatched)) == (25, 107, 0) and exposed | unmatched == {r['sample_id'] for r in m['val']}
    raw = read(mpath.parent / 'raw_inventory.json')['records']; groups = read(mpath.parent / 'groups.json')
    old = read(ROOT / 'datasets/ureter_semis_a1_sup_kidney_only_20261027/manifest.json')
    old_hashes = {r['image_sha256'] for r in old['train']['labeled']}
    exposed_groups = {g['group_id'] for g in groups if any(raw[i]['image_sha256'] in old_hashes for i in g['member_indices'])}
    checks['historical_exposure_recomputed_from_raw_hashes'] = exposed == {r['sample_id'] for r in m['val'] if r['group_id'] in exposed_groups}
    history = read(ROOT / 'runs/a1_sup_kidney_resolution_20261028/native_comparison/comparison.json')['models']
    checks['a0_old190_native_video_references_reproduced'] = all(s['video']['models'][n]['metrics'] == history[h]['metrics'] for n, h in [('A0', 'A0'), ('old190', 'unet256')])
    checks['joint_cached_references_exact'] = all(s[scope]['models']['joint521']['metrics'] == read(ROOT / f'runs/a1_sup_kidney_joint_20261028/{scope}_native/comparison.json')['models']['joint_final']['metrics'] for scope in ('video', 'image'))
    gates = p['native_video_gates']; a0 = s['video']['models']['A0']['metrics']; c = s['video']['models']['classwise']['metrics']
    checks['fresh_gates_and_pass_recomputed'] = (gates == {'stone_dice_min': a0['stone_dice'], 'fiber_dice_min': a0['fiber_dice'] - .01, 'negative_fp_fraction_max': a0['stone_negative_fp_pixel_fraction']}
        and c['stone_dice'] >= gates['stone_dice_min'] and c['fiber_dice'] >= gates['fiber_dice_min'] and c['stone_negative_fp_pixel_fraction'] <= gates['negative_fp_fraction_max'])
    counts = {'A0': (2, 49202790), 'old190': (1, 24601395), 'joint521': (1, 24601395), 'classwise': (3, 73804185)}
    checks['all_four_costs_full_model_counts12warm87samples'] = all(
        (v['model_count'], v['parameter_count']) == counts[n] and v['warmup_frames'] == 12 and v['samples'] == len(v['raw_samples']) == 87
        for n, v in s['performance'].items())
    checks['latency_percentiles_recomputed'] = all(np.isclose(v['timings']['pipeline_ms'][k], fn([r['pipeline_ms'] for r in v['raw_samples']]))
        for v in s['performance'].values() for k, fn in [('mean', np.mean), ('p50', np.median), ('p95', lambda x: np.percentile(x, 95))])
    with zipfile.ZipFile(OUT / 'source_snapshot.zip') as archive:
        checks['snapshot_unique_entries_and_source_hashes'] = len(archive.namelist()) == len(set(archive.namelist())) and all(
            hashlib.sha256(archive.read(name.replace('\\', '/'))).hexdigest() == h for name, h in p['sources'].items())
    result = {'checks': {k: bool(v) for k, v in checks.items()}, 'all_passed': all(checks.values()),
        'new_unit_tests': {'passed': 3, 'failed': 0, 'command': 'python -m unittest discover -s tests -p test_classwise_experts.py -v',
            'evidence': 'All three passed before control execution; no full-suite rerun claimed'},
        'details': details, 'scope': 'Export correctness and frozen controls only; no temporal, clinical or independent-case validation'}
    path = ROOT / 'runs/engineering/classwise_experts_checks_20261028.json'
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    assert result['all_passed'], result
    render_examples(all_masks)
    print(json.dumps({'checks_passed': len(checks), 'all_passed': result['all_passed'], 'output': str(path)}, indent=2))


def render_examples(all_masks):
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    examples = [('video', 'reviewed_video_1_000072'), ('video', 'reviewed_video_1_000721'),
                ('video', 'reviewed_video_2_001822'), ('image', 'kidney_stones_fiber_v2__joint__e4a3e164d6d744a489817e05')]
    columns = ['Reference', 'A0', 'old190', 'classwise']
    canvas = Image.new('RGB', (1280, 4 * 358 + 48), '#20252b'); draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), 'Validation overlays: red=stone, cyan=fiber. Video labels reviewed; image label pending.', font=font, fill='white')
    index_rows = []
    for j, (scope, sample_id) in enumerate(examples):
        rows, targets, predictions = all_masks[scope]
        i = next(i for i, r in enumerate(rows) if r['sample_id'] == sample_id)
        row = rows[i]; target = targets[i]; image = Image.open(row['image']).convert('RGB')
        rgb = np.asarray(image).copy(); y = 48 + j * 358
        for col, name in enumerate(columns):
            mask = target if col == 0 else predictions[name][i]
            overlay = rgb.copy()
            for k, color in [(1, [235, 40, 40]), (2, [20, 210, 235])]:
                selected = mask == k
                overlay[selected] = (.48 * rgb[selected] + .52 * np.array(color)).astype(np.uint8)
            canvas.paste(Image.fromarray(overlay).resize((320, 320)), (col * 320, y + 22))
            draw.text((col * 320 + 5, y + 2), name, font=small, fill='white')
        label = f"{row.get('video_id') or 'image-dev'} | index {row.get('frame_index', i)} | {sample_id}"
        draw.text((5, y + 342), label, font=small, fill='white')
        index_rows.append({'scope': scope, 'sample_id': sample_id, 'export_index': i, 'frame_index_zero_based': row.get('frame_index'), 'image': row['image'], 'mask': row['mask']})
    canvas.save(OUT / 'validation_examples.jpg', quality=93)
    (OUT / 'validation_examples.json').write_text(json.dumps(index_rows, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
