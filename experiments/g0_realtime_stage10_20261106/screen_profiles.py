"""Three exact CPU arms on identical resident video2 diagnostic inputs."""
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
from experiments.g0_realtime_stage8_20261106.benchmark import host_memory
from .profile_runtime import Runtime,ProfileOnlyRuntime
from .identity import sources


def main():
    folder=ROOT/'runs/g0_realtime_stage10_20261106/profile_screen';folder.mkdir(parents=True,exist_ok=False)
    identity=sources()
    with zipfile.ZipFile(folder/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in identity:z.write(ROOT/name,name)
    limit=threadpool_limits(1);torch.set_num_threads(1);cv2.setNumThreads(1)
    memory=host_memory()
    if memory['available_physical_bytes']<512*1024**2+2*23*(400*400*(3+1+12)):
        raise MemoryError('Insufficient free host memory for resident diagnostic inputs')
    pre=read(ROOT/'runs/a1_weak_blue_evidence_20261103/preflight.json')
    assert all(sha256(x['path'])==x['sha256'] for x in pre['checkpoints'].values())
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
    names=('v7','profile_only','combined');arms=[V7(),ProfileOnlyRuntime(),Runtime()]
    rounds=[];records=[]
    for cycle in range(4):
        for r in arms:r.reset()
        for j in range(12):
            for k in np.roll(np.arange(3),(cycle+j)%3):arms[k].run(*warmup,j/30,'warmup')
        for r in arms:r.reset()
        times=[[],[],[]]
        for offset,(i,frame,raw,fp,hashes) in enumerate(inputs):
            order=np.roll(np.arange(3),(cycle+offset)%3)
            if cycle%2:order=order[::-1]
            for k in order:
                start=time.perf_counter();arms[k].run(frame,raw,fp,source[i]['timestamp_seconds'],'video_2')
                elapsed=(time.perf_counter()-start)*1000;times[k].append(elapsed)
                records.append(dict(round=cycle,arm=names[k],frame=i,post_ms=elapsed))
            for r in arms[1:]:
                np.testing.assert_array_equal(arms[0].stage.final_mask,r.stage.final_mask)
                np.testing.assert_array_equal(arms[0].g0.final_mask,r.g0.final_mask)
                assert close(geometry_record(arms[0].g0.info),geometry_record(r.g0.info))
                assert close(arms[0].stage.outline.get('cap'),r.stage.outline.get('cap'))
                assert r.counters==arms[0].counters and r.extra_counts==arms[0].extra_counts
        means={name:float(np.mean(values)) for name,values in zip(names,times)}
        rounds.append(dict(round=cycle,mean_ms=means,reduction_percent={n:(1-v/means['v7'])*100 for n,v in means.items() if n!='v7'}))
    counts={name:getattr(r,'seed_counts',{}).copy() for name,r in zip(names,arms)}
    micro=[]
    assert all(sha256(ROOT/name)==value for name,value in identity.items())
    result=dict(scope='CPU only, 23 selected video2 inputs, four rotating/reversing rounds; original source timestamps, independent state, twelve warmups/reset each round; no dropped-frame or camera/display validation',
        frames=23,rounds=rounds,comparisons=184,output_mismatches=0,micro_rounds=micro,
        pooled_mean_ms={n:float(np.mean([r['post_ms'] for r in records if r['arm']==n])) for n in names},
        seed_counts_last_post_round=counts,
        inputs=[dict(frame=i,**h) for i,_,_,_,h in inputs],sources=identity,
        host_memory=memory,custom_g0_dll_enabled=False)
    (folder/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (folder/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','inputs')},indent=2))
    limit.restore_original_limits()


if __name__=='__main__':main()
