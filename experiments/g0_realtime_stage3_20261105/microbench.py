"""Bounded primitive screening; not an end-to-end performance result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from tools.fiber_connected_blue_growth import chroma
from tools.fiber_connected_blue_union import rooted
from tools.fiber_weak_stage2 import native_gradient, connect
from tests.test_tapered_tangent_cap import scene
from experiments.g0_realtime_stage2_20261105.runtime import Runtime as V2
from .runtime import Runtime
from . import pixels


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    args = parser.parse_args(); output = Path(args.output)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    with threadpool_limits(1):
        cv2.setNumThreads(1)
        old, new = V2(), Runtime()
        rng = np.random.default_rng(23)
        image, raw, probability, _ = scene()
        seeds, _ = old.observer.seeds(image, raw, probability, None, 99)
        seed = seeds[0]
        random = rng.integers(0, 256, image.shape, np.uint8)
        binary = rng.random(raw.shape) > .6; core = rng.random(raw.shape) > .98
        results = []
        def measure(name, reference, candidate):
            for f in (reference, candidate):
                for _ in range(5): f()
            timings = [[], []]
            for iteration in range(40):
                # Alternate order to limit drift in a small screening test.
                for i in ((0, 1) if iteration % 2 == 0 else (1, 0)):
                    start = time.perf_counter(); (reference, candidate)[i]()
                    timings[i].append((time.perf_counter()-start)*1000)
            rows = [dict(median_ms=float(np.median(x)), p95_ms=float(np.percentile(x, 95))) for x in timings]
            results.append(dict(name=name, reference=rows[0], candidate=rows[1],
                                median_speedup=rows[0]['median_ms']/rows[1]['median_ms']))
        measure('chroma_random_400', lambda:chroma(random), lambda:pixels.chroma(random))
        measure('gradient_random_400', lambda:native_gradient(random), lambda:pixels.native_gradient(random))
        measure('rooted_4', lambda:rooted(binary, core, 3), lambda:pixels.rooted(binary, core, 3))
        measure('connect_8', lambda:connect(binary, core), lambda:pixels.connect(binary, core))
        for name, frame in (('synthetic', image), ('random', random)):
            gradient = native_gradient(frame); cfg = old.stage.q['evidence']
            measure('evidence_'+name, lambda:old.colors.fast_evidence(frame, seed, cfg, gradient),
                    lambda:new.pixels.evidence(frame, seed, cfg, gradient))
            measure('color_'+name, lambda:old.observer.color(frame.reshape(-1, 3), seed),
                    lambda:new.observer.color(frame.reshape(-1, 3), seed))
        (output/'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(json.dumps(results, indent=2))


if __name__ == '__main__': main()
