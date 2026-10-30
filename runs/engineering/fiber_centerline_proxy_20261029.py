"""Reference-derived centerline coverage only; never creates tip or training labels."""
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from tools.fiber_reference_workspace import ROOT, read, load_mask, sha256
from ureter_semis.experiment import atomic_json


def thin(region):
    """Two-subiteration binary thinning; all visible components retained."""
    a = np.pad(region.astype(bool), 1)
    for iteration in range(max(region.shape) + 1):
        removed = 0
        for phase in (0, 1):
            p = [a[:-2, 1:-1], a[:-2, 2:], a[1:-1, 2:], a[2:, 2:], a[2:, 1:-1], a[2:, :-2], a[1:-1, :-2], a[:-2, :-2]]
            count = sum(x.astype(np.uint8) for x in p)
            transitions = sum((~p[i] & p[(i + 1) % 8]).astype(np.uint8) for i in range(8))
            triplets = ((p[0] & p[2] & p[4]), (p[2] & p[4] & p[6])) if phase == 0 else ((p[0] & p[2] & p[6]), (p[0] & p[4] & p[6]))
            delete = a[1:-1, 1:-1] & (count >= 2) & (count <= 6) & (transitions == 1) & ~triplets[0] & ~triplets[1]
            removed += int(delete.sum()); a[1:-1, 1:-1][delete] = False
        if not removed: return a[1:-1, 1:-1]
    raise RuntimeError('Thinning did not converge')


def self_test():
    line = np.zeros((20, 30), bool); line[10, 2:10] = True; line[10, 15:25] = True
    assert np.array_equal(thin(line), line)
    box = np.zeros((25, 25), bool); box[7:18, 3:22] = True
    skeleton = thin(box)
    assert skeleton.any() and not (skeleton & ~box).any() and skeleton.sum() < box.sum()
    assert not thin(np.zeros((5, 5), bool)).any()
    return 3


def main():
    out = ROOT / 'runs/a1_fiber_blue_aplus_20261029'
    if read(out / 'job.json')['phase'] != 'completed': raise RuntimeError('Run after isolated video benchmarks')
    tests = self_test(); config = read(ROOT / 'configs/fiber_blue_aplus_20261029.json')
    result = {'role': 'reference-mask-derived visible centerline coverage proxy, not anatomical tip or physical axis truth',
        'method': 'two-subiteration thinning of all reference class2 components, no largest-component choice or hidden gap fill; unknown255 and3px neighborhood/image-border margin excluded',
        'adoption_gate_changed': False, 'posthoc_supplementary_metric': True, 'self_tests_passed': tests,
        'source_sha256': sha256(Path(__file__)), 'sets': {}}
    for scope, manifest in [('video_native', config['video_manifest']), ('image_native', config['image_manifest'])]:
        pooled = {name: defaultdict(int) for name in config['schemes']}; frames = []
        for i, row in enumerate(read(ROOT / manifest)['val']):
            with __import__('PIL.Image', fromlist=['Image']).open(row['mask']) as im: target = np.asarray(im).copy()
            valid = cv2.erode((target != 255).astype(np.uint8), np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
            center = thin(target == 2) & valid; total = int(center.sum()); record = {'sample_id': row['sample_id'], 'centerline_pixels': total, 'models': {}}
            for name in config['schemes']:
                prediction = load_mask(out / f'{scope}/{name}/{i:04d}.png', target.shape, {0, 1, 2})
                missed = center & (prediction != 2); count = int(missed.sum())
                n, labels, stats, _ = cv2.connectedComponentsWithStats(missed.astype(np.uint8), 8)
                longest = int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
                record['models'][name] = {'covered': total - count, 'missed': count, 'coverage': (total - count) / total if total else None,
                    'missed_fragments': n - 1, 'longest_missed_fragment_pixels': longest}
                pooled[name]['covered'] += total - count; pooled[name]['missed'] += count; pooled[name]['reference_centerline_pixels'] += total
                pooled[name]['missed_fragments'] += n - 1
                pooled[name]['longest_missed_fragment_pixels'] = max(pooled[name]['longest_missed_fragment_pixels'], longest)
            frames.append(record)
        summary = {}
        for name, values in pooled.items():
            summary[name] = {**dict(values), 'coverage': values['covered'] / values['reference_centerline_pixels'] if values['reference_centerline_pixels'] else None}
        result['sets'][scope] = {'models': summary, 'frames': frames}
    atomic_json(out / 'centerline_coverage_proxy.json', result)
    print({scope: data['models'] for scope, data in result['sets'].items()})


if __name__ == '__main__': main()
