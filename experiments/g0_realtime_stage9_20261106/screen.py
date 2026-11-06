"""Same-input CPU screening, not live-camera or full-video performance."""
from pathlib import Path
import json,time,zipfile
import cv2,numpy as np,torch
from threadpoolctl import threadpool_limits
from tools.fiber_reference_workspace import ROOT,read,sha256
from tools.run_fiber_temporal_state_control import construct
from tools.fiber_photometric_input import infer
from ureter_semis.experiment import seed_training
from experiments.g0_realtime_20261105.benchmark import array_hash,geometry_record,DIAGNOSTIC
from experiments.g0_realtime_20261105.compare import close
from experiments.g0_realtime_stage7_20261106.runtime import Runtime as V7
from experiments.g0_realtime_stage7_20261106.color import from_chroma
from .runtime import Runtime,indexed_color


def main():
    folder=ROOT/'runs/g0_realtime_stage9_20261106/cpu_screen';folder.mkdir(parents=True,exist_ok=False)
    frozen=read(ROOT/'runs/g0_realtime_stage8_20261106/full/v7-pipeline_video_1/identity.json')['sources']
    assert all(sha256(ROOT/name)==value for name,value in frozen.items())
    sources=dict(frozen)
    sources.update({p.relative_to(ROOT).as_posix():sha256(p) for p in Path(__file__).parent.glob('*.py')})
    with zipfile.ZipFile(folder/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in sources:z.write(ROOT/name,name)
    limit=threadpool_limits(1);torch.set_num_threads(1);cv2.setNumThreads(1)
    pre=read(ROOT/'runs/a1_weak_blue_evidence_20261103/preflight.json')
    video=next(v for v in pre['videos'] if v['video_id']=='video_2')
    assert sha256(video['path'])==video['sha256']
    source=read(ROOT/'runs/a1_weak_blue_evidence_20261103/source_video_2.json')['frames']
    expected={r['frame_index']:r for r in map(json.loads,(ROOT/'runs/g0_realtime_stage8_20261106/full/v7-serial_video_2/frames.jsonl').read_text().splitlines())}
    seed_training(42);experts=construct(pre);cap=cv2.VideoCapture(video['path']);i=0;inputs=[];warmup=None
    while True:
        ok,frame=cap.read()
        if not ok:break
        if i==0 or i in DIAGNOSTIC['video_2']:
            raw,_,fp,_=infer(experts,frame,'raw2')
            if i==0:warmup=(frame.copy(),raw.copy(),fp.copy())
            if i in DIAGNOSTIC['video_2']:
                hashes=dict(source_rgb_sha256=array_hash(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),
                            raw_sha256=array_hash(raw),fiber_probability_sha256=array_hash(fp))
                assert all(expected[i][k]==v for k,v in hashes.items())
                inputs.append((i,frame.copy(),raw.copy(),fp.copy(),hashes))
        i+=1
    cap.release();assert len(inputs)==23 and warmup is not None
    del experts;torch.cuda.empty_cache()
    arms=[V7(),Runtime()];fixtures=[];base_region=arms[0].region_color
    capture=[False]
    def region(frame,mask,seed):
        answer=base_region(frame,mask,seed)
        if capture[0] and frame is arms[0].frame and frame.dtype==np.uint8 and np.count_nonzero(mask)>=4096:
            c,n=arms[0].chroma(frame)
            fixtures.append((c.copy(),n.copy(),mask.copy(),dict(seed)))
        return answer
    arms[0].growth.grow.__func__.__globals__['region_color']=region
    rounds=[];records=[]
    for cycle in range(4):
        for r in arms:r.reset()
        for j in range(12):
            for k in ((0,1) if (cycle+j)%2==0 else (1,0)):arms[k].run(*warmup,j/30,'warmup')
        for r in arms:r.reset()
        times=[[],[]]
        capture[0]=cycle==0
        for offset,(i,frame,raw,fp,hashes) in enumerate(inputs):
            for k in ((0,1) if (cycle+offset)%2==0 else (1,0)):
                # Fixture copying is outside reported timings: disable capture
                # around the timed call, then gather in an untimed replay below.
                capture[0]=False
                start=time.perf_counter();arms[k].run(frame,raw,fp,source[i]['timestamp_seconds'],'video_2')
                elapsed=(time.perf_counter()-start)*1000;times[k].append(elapsed)
                records.append(dict(round=cycle,arm=('v7','indexed')[k],frame=i,post_ms=elapsed))
            np.testing.assert_array_equal(arms[0].stage.final_mask,arms[1].stage.final_mask)
            np.testing.assert_array_equal(arms[0].g0.final_mask,arms[1].g0.final_mask)
            assert close(geometry_record(arms[0].g0.info),geometry_record(arms[1].g0.info))
            assert close(arms[0].stage.outline.get('cap'),arms[1].stage.outline.get('cap'))
        means=[float(np.mean(x)) for x in times]
        rounds.append(dict(round=cycle,v7_ms=means[0],indexed_ms=means[1],reduction_percent=(1-means[1]/means[0])*100))
    arms[0].reset();capture[0]=True
    for i,frame,raw,fp,_ in inputs:arms[0].run(frame,raw,fp,source[i]['timestamp_seconds'],'video_2')
    capture[0]=False;assert fixtures
    micro=[]
    def reference(c,n,m,s):
        np.count_nonzero(m)
        return from_chroma(arms[0].observer,c[m],n[m],s)
    for c,n,m,s in fixtures:
        np.testing.assert_array_equal(reference(c,n,m,s),indexed_color(arms[0].observer,c,n,m,s))
    for cycle in range(4):
        totals=[0.,0.]
        for offset,(c,n,m,s) in enumerate(fixtures):
            for k in ((0,1) if (cycle+offset)%2==0 else (1,0)):
                fn=reference if k==0 else lambda c,n,m,s:indexed_color(arms[0].observer,c,n,m,s)
                start=time.perf_counter()
                for _ in range(10):fn(c,n,m,s)
                totals[k]+=(time.perf_counter()-start)*1000
        micro.append(dict(round=cycle,v7_ms_per_region=totals[0]/len(fixtures)/10,
                          indexed_ms_per_region=totals[1]/len(fixtures)/10,
                          reduction_percent=(1-totals[1]/totals[0])*100))
    assert all(sha256(ROOT/name)==value for name,value in sources.items())
    result=dict(scope='CPU fixed-input screening on 23 selected video2 frames, not full-video/live FPS; four alternating/reversed rounds, independent state; NN precomputed once; no dropped-frame correctness claim',
        frames=23,rounds=rounds,comparisons=92,output_mismatches=0,micro_rounds=micro,micro_fixtures=len(fixtures),
        pooled_v7_ms=float(np.mean([r['post_ms'] for r in records if r['arm']=='v7'])),
        pooled_indexed_ms=float(np.mean([r['post_ms'] for r in records if r['arm']=='indexed'])),
        inputs=[dict(frame=i,**h) for i,_,_,_,h in inputs],sources=sources,
        original_stage8_sources_unchanged=len(frozen),custom_g0_dll_enabled=False)
    (folder/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (folder/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','inputs')},indent=2))
    limit.restore_original_limits()


if __name__=='__main__':main()
