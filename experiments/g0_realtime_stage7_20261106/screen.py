"""Independent ablations on identical resident video2 diagnostic inputs."""
from pathlib import Path
import json,time,zipfile
import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from tools.fiber_reference_workspace import ROOT,read,sha256
from tools.run_fiber_temporal_state_control import construct
from tools.fiber_photometric_input import infer
from ureter_semis.experiment import seed_training
from experiments.g0_realtime_20261105.benchmark import array_hash,geometry_record,DIAGNOSTIC
from experiments.g0_realtime_20261105.compare import close
from experiments.g0_realtime_stage6_20261106.runtime import Runtime as V6
from .runtime import Runtime


def main():
    folder=ROOT/'runs/g0_realtime_stage7_20261106/ablations';folder.mkdir()
    previous=read(ROOT/'runs/g0_realtime_stage6_20261106/full_packed/v6-serial_video_2/identity.json')
    assert all(sha256(ROOT/name)==value for name,value in previous['sources'].items())
    sources=dict(previous['sources'])
    for pattern in ('*.py','*.ps1'):
        sources.update({p.relative_to(ROOT).as_posix():sha256(p) for p in (ROOT/'experiments/g0_realtime_stage7_20261106').glob(pattern)})
    with zipfile.ZipFile(folder/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in sources:archive.write(ROOT/name,name)
    limit=threadpool_limits(1);torch.set_num_threads(1);cv2.setNumThreads(1)
    pre=read(ROOT/'runs/a1_weak_blue_evidence_20261103/preflight.json')
    video=next(v for v in pre['videos'] if v['video_id']=='video_2')
    assert sha256(video['path'])==video['sha256']
    source=read(ROOT/'runs/a1_weak_blue_evidence_20261103/source_video_2.json')['frames']
    expected={r['frame_index']:r for r in map(json.loads,(ROOT/'runs/g0_realtime_stage6_20261106/full_packed/v6-serial_video_2/frames.jsonl').read_text().splitlines())}
    seed_training(42);experts=construct(pre)
    cap=cv2.VideoCapture(video['path']);index=0;inputs=[];warmup=None
    while True:
        ok,frame=cap.read()
        if not ok:break
        if index==0 or index in DIAGNOSTIC['video_2']:
            raw,_,fp,_=infer(experts,frame,'raw2')
            if index==0:warmup=(frame.copy(),raw.copy(),fp.copy())
            if index in DIAGNOSTIC['video_2']:
                hashes=dict(source_rgb_sha256=array_hash(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),raw_sha256=array_hash(raw),fiber_probability_sha256=array_hash(fp))
                assert all(expected[index][key]==value for key,value in hashes.items())
                inputs.append((index,frame.copy(),raw.copy(),fp.copy(),hashes))
        index+=1
    cap.release();assert len(inputs)==23 and warmup is not None
    del experts;torch.cuda.empty_cache()
    names=('v6','reuse_only','broadcast_only','combined')
    arms=(V6(),Runtime(True,False),Runtime(False,True),Runtime())
    rounds=[];records=[]
    for cycle in range(3):
        for r in arms:r.reset()
        for j in range(12):
            for k in np.roll(np.arange(4),(cycle+j)%4):arms[k].run(*warmup,j/30,'warmup')
        for r in arms:r.reset()
        totals={name:[] for name in names}
        for offset,(index,frame,raw,fp,hashes) in enumerate(inputs):
            order=np.roll(np.arange(4),(cycle+offset)%4)
            if cycle%2:order=order[::-1]
            for k in order:
                start=time.perf_counter();arms[k].run(frame,raw,fp,source[index]['timestamp_seconds'],'video_2');elapsed=(time.perf_counter()-start)*1000
                totals[names[k]].append(elapsed)
                records.append(dict(round=cycle,arm=names[k],frame=index,post_ms=elapsed))
            for r in arms[1:]:
                np.testing.assert_array_equal(arms[0].stage.final_mask,r.stage.final_mask)
                np.testing.assert_array_equal(arms[0].g0.final_mask,r.g0.final_mask)
                assert close(geometry_record(arms[0].g0.info),geometry_record(r.g0.info))
                assert close(arms[0].stage.outline.get('cap'),r.stage.outline.get('cap'))
        means={name:float(np.mean(values)) for name,values in totals.items()}
        rounds.append(dict(round=cycle,mean_ms=means,reduction_percent={name:(1-value/means['v6'])*100 for name,value in means.items() if name!='v6'}))
    assert all(sha256(ROOT/name)==value for name,value in sources.items())
    result=dict(rounds=rounds,diagnostic_frames=23,round_count=3,comparisons=207,output_mismatches=0,
        input_identities=[dict(frame=i,**h) for i,_,_,_,h in inputs],sources=sources,custom_g0_dll_enabled=False,
        scope='Only CPU postprocessing timed; resident identical neural outputs; rotating/reversing arm order, independent states, twelve warmups then reset per round; selected video2 frames only, not full-video FPS')
    (folder/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (folder/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    print(json.dumps(rounds,indent=2));limit.restore_original_limits()


if __name__=='__main__':main()
