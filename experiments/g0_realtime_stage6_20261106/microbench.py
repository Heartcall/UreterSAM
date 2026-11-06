"""Paired domain-only screening; not an end-to-end speed claim."""
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from tests.test_tapered_tangent_cap import scene
from experiments.g0_realtime_stage4_20261105.spatial import GridCache
from .test_runtime import reference_domain
from .spatial import domain_from_body


def main():
    output = Path('runs/g0_realtime_stage6_20261106/microbench.json')
    assert not output.exists()
    _, _, _, labels = scene()
    cfg = dict(maximum_forward_at400=60., maximum_forward_diagonal_fraction=.2,
               lateral_width_multiplier=1.4, lateral_margin_at400=8.)
    rng = np.random.default_rng(92); rows = []
    with threadpool_limits(1):
        for name, body in (('fixture', labels == 2), ('scattered', rng.random((400, 400)) < .05)):
            grid = GridCache(); grid.indices(body.shape)
            args = (body, np.array([20., 200.]), np.array([1., 0.]), cfg, 1.)
            old = lambda: reference_domain(*args, grid=grid)
            new = lambda: domain_from_body(*args)
            np.testing.assert_array_equal(old()[0], new()[0])
            for _ in range(5): old(); new()
            timings = [[], []]
            for index in range(40):
                for arm in ((0, 1) if index % 2 == 0 else (1, 0)):
                    start = time.perf_counter(); (old, new)[arm](); timings[arm].append((time.perf_counter()-start)*1000)
            medians = [float(np.median(a)) for a in timings]
            rows.append(dict(case=name, reference_ms=medians[0], candidate_ms=medians[1],
                             speedup=medians[0]/medians[1], counters=new()[3]))
    output.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__': main()
