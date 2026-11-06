"""Alternating primitive screening, not full pipeline throughput."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np
from threadpoolctl import threadpool_limits
from tests.test_tapered_tangent_cap import scene
from experiments.g0_realtime_stage4_20261105.runtime import Runtime as V4
from experiments.g0_realtime_stage3_20261105.pixels import native_gradient
from .runtime import Runtime

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True)
    output=Path(ap.parse_args().output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    with threadpool_limits(1):
        cv2.setNumThreads(1)
        old,new=V4(),Runtime();image,raw,p,_=scene()
        seed=old.observer.seeds(image,raw,p,None,99)[0][0]
        rng=np.random.default_rng(53);random=rng.integers(0,256,image.shape,np.uint8)
        rows=[]
        def measure(name,reference,candidate):
            for f in (reference,candidate):
                for _ in range(5):f()
            timings=[[],[]]
            for iteration in range(40):
                for index in ((0,1) if iteration%2==0 else (1,0)):
                    start=time.perf_counter();(reference,candidate)[index]()
                    timings[index].append((time.perf_counter()-start)*1000)
            med=[float(np.median(x)) for x in timings]
            rows.append(dict(name=name,reference_median_ms=med[0],candidate_median_ms=med[1],speedup=med[0]/med[1]))
        for name,frame in (('synthetic',image),('random',random)):
            gradient=native_gradient(frame);cfg=old.stage.q['evidence']
            measure('evidence_'+name,lambda:old.colors.fast_evidence(frame,seed,cfg,gradient),lambda:new.backend.evidence(frame,seed,cfg,gradient))
            measure('color_'+name,lambda:old.observer.color(frame.reshape(-1,3),seed),lambda:new.observer.color(frame.reshape(-1,3),seed))
        yy,xx=np.indices(raw.shape);o=np.array([193.2,188.7]);u=np.array([.61,.7924]);n=np.array([-u[1],u[0]])
        measure('dense_projection',lambda:((xx-o[0])*u[0]+(yy-o[1])*u[1],(xx-o[0])*n[0]+(yy-o[1])*n[1]),lambda:new.backend.project(raw.shape,o,u,n))
        (output/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
