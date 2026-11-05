"""Independently inspect saved masks, probabilities, statuses and geometry."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np


def close(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.shape == b.shape and np.allclose(a, b, atol=1e-9, rtol=0)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, (float, int)) and not isinstance(a, bool):
        return np.isclose(a, b, atol=1e-9, rtol=0)
    return a == b


def compare(reference, candidate):
    def records(path):
        return [json.loads(line) for line in (path/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
    before, after = records(reference), records(candidate)
    assert len(before) == len(after)
    counts, examples = Counter(), []
    for a, b in zip(before, after):
        index = a['frame_index']; assert index == b['frame_index']
        for key in ('source_rgb_sha256', 'raw_sha256', 'fiber_probability_sha256', 's2_sha256', 'final_sha256'):
            if a[key] != b[key]:
                counts[key+'_mismatch_frames'] += 1
                if len(examples) < 20:
                    examples.append(dict(frame_index=index, mismatch=key))
        if not close(a['g0'], b['g0']):
            counts['geometry_mismatch_frames'] += 1
            if len(examples) < 20:
                examples.append(dict(frame_index=index, mismatch='geometry'))
        for folder in ('masks', 's2_masks'):
            x = cv2.imread(str(reference/folder/f'{index:06d}.png'), cv2.IMREAD_UNCHANGED)
            y = cv2.imread(str(candidate/folder/f'{index:06d}.png'), cv2.IMREAD_UNCHANGED)
            assert x is not None and y is not None
            n = int(np.count_nonzero(x != y))
            counts[folder+'_changed_pixels'] += n
            counts[folder+'_changed_frames'] += n > 0
    return dict(reference=str(reference), candidate=str(candidate), frames=len(before),
                counts=dict(counts), examples=examples, passed=not any(counts.values()),
                geometry_absolute_tolerance=1e-9, mask_and_probability_policy='exact')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', required=True); ap.add_argument('--candidate', required=True)
    ap.add_argument('--output', required=True); args = ap.parse_args()
    result = compare(Path(args.reference), Path(args.candidate))
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    assert result['passed'], 'Candidate output changed'


if __name__ == '__main__':
    main()
