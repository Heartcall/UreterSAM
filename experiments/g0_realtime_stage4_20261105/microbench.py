"""Isolate spatial mechanisms; synthetic screening, not full-video throughput."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from tests.test_tapered_tangent_cap import scene
from tools.fiber_tapered_tangent_cap import fit_geometry
from experiments.g0_realtime_stage3_20261105.runtime import Runtime as V3
from .runtime import Runtime


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True)
    path=Path(parser.parse_args().output)
    if path.exists(): raise FileExistsError(path)
    path.mkdir(parents=True)
    with threadpool_limits(1):
        cv2.setNumThreads(1)
        frame,raw,probability,body=scene()
        runtimes=[V3(),Runtime(roi=False),Runtime()]
        seed=runtimes[0].observer.seeds(frame,raw,probability,None,99)[0][0]
        cfg=runtimes[0].g0.cfg['tip']
        geometry,_=fit_geometry(body==2,seed['u'],runtimes[0].g0.cfg['fit'],1.)
        funcs=[]
        for runtime in runtimes:
            # As in actual G0 following S2, current-frame chroma is already hot.
            runtime.frame=frame; runtime.chroma(frame)
            funcs.append(runtime.g0.run.__func__.__globals__['locate_tip'])
        for f in funcs:
            for _ in range(5): f(frame,raw,geometry,seed,cfg,1.)
        timings=[[],[],[]]
        for iteration in range(40):
            order=(0,1,2) if iteration%2==0 else (2,1,0)
            for index in order:
                start=time.perf_counter(); funcs[index](frame,raw,geometry,seed,cfg,1.)
                timings[index].append((time.perf_counter()-start)*1000)
        rows={name:dict(median_ms=float(np.median(x)),p95_ms=float(np.percentile(x,95)))
              for name,x in zip(('v3','grid_only','grid_and_roi'),timings)}
        rows['spatial_counters']=runtimes[2].spatial_counters
        rows['roi_fraction']=rows['spatial_counters']['tip_roi_pixels']/rows['spatial_counters']['tip_full_pixels']
        (path/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(json.dumps(rows,indent=2))


if __name__=='__main__': main()
