"""Frozen post-G0 CPU controls: each arm/video in its own full-replay process."""
import argparse,json,os,sys,time,subprocess,zipfile,shutil
from pathlib import Path
from collections import Counter
import cv2,numpy as np
from .fiber_reference_workspace import ROOT,read,sha256,load_mask
from .fiber_rgb_corrected_taper import RGBCorrectedTaper
from .run_two_network_experts import array_hash
from .replay_classwise_experts import save_png,stats
from .run_classwise_expert_control import summarize
from .run_fiber_dark_input_control import select_groups
from .diagnose_joint_failures import paired_changes
from ureter_semis.experiment import atomic_json
CONFIG=ROOT/'configs/tapered_rgb_control_20261104.json'
def logs(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
def folder(p):
    p.mkdir(parents=True)
    for k in ['masks','layers']:(p/k).mkdir()
def packed(dest,i,p):
    save_png(dest/f'masks/{i:06d}.png',p.final_mask);np.savez_compressed(dest/f'layers/{i:06d}.npz',**p.layers)
    return dict(final_sha256=array_hash(p.final_mask),layer_hashes={k:array_hash(v) for k,v in p.layers.items()})
def report_scope(cfg,out,scope,rows,gt,new,base,records):
    groups={name:dict(frames=len(ix),summary=summarize([rows[i] for i in ix],[gt[i] for i in ix],[new[i] for i in ix]),baseline=summarize([rows[i] for i in ix],[gt[i] for i in ix],[base[i] for i in ix])) for name,ix in select_groups(rows,scope!='image_native').items()}
    atomic_json(out/scope/'summary.json',dict(summary=summarize(rows,gt,new),baseline=summarize(rows,gt,base),groups=groups,records=records,applied=sum(x['info']['applied'] for x in records),reasons=dict(Counter(x['info']['reason'] for x in records))))
def static(cfg,out):
    bc=read(ROOT/cfg['baseline_config']);base=ROOT/cfg['baseline'];cache=ROOT/bc['static_cache'];assert all(sha256(cache/k)==v for k,v in read(cache/'identity.json')['files'].items())
    for arm in cfg['arms']:
        g=RGBCorrectedTaper(cfg,arm);armout=out/arm
        for scope,key in [('video_native','reference_video'),('image_native','reference_image')]:
            dest=armout/scope;folder(dest);rows=read(ROOT/bc[key])['val'];oldrows=read(base/scope/'summary.json')['records'];gt=[];new=[];before=[];records=[]
            for i,r in enumerate(rows):
                with np.load(cache/scope/f'{i:04d}.npz') as z:frame=z['frame'];raw=z['raw'];target=z['gt']
                b=load_mask(base/scope/f'masks/{i:06d}.png',raw.shape,{0,1,2});old=load_mask(ROOT/bc['old_result']/scope/f'masks/{i:06d}.png',raw.shape,{0,1,2});assert array_hash(b)==oldrows[i]['final_sha256'];assert array_hash(raw)==oldrows[i]['raw_sha256']
                with np.load(base/scope/f'layers/{i:06d}.npz') as z:subject=z['subject']>0
                g.run(frame,raw,old,b,oldrows[i]['info'],subject);ids=packed(dest,i,g);records.append(dict(index=i,sample_id=r['sample_id'],video_id=r.get('video_id'),frame_index=r.get('frame_index'),baseline_sha256=array_hash(b),raw_sha256=array_hash(raw),info=g.info,**ids));gt.append(target.copy());new.append(g.final_mask.copy());before.append(b)
            report_scope(cfg,armout,scope,rows,gt,new,before,records)
def worker(cfg,out,arm,vid):
    pre=read(out/'preflight.json');assert all(sha256(ROOT/k)==v for k,v in pre['sources'].items());bc=read(ROOT/cfg['baseline_config']);base=ROOT/cfg['baseline'];data=logs(base/f'full_video/{vid}/frames.jsonl');dest=out/arm/f'full_video/{vid}';folder(dest);g=RGBCorrectedTaper(cfg,arm)
    def context(i):
        b=load_mask(base/f'full_video/{vid}/masks/{i:06d}.png',(400,400),{0,1,2});old=load_mask(ROOT/bc['old_result']/f'full_video/{vid}/masks/{i:06d}.png',(400,400),{0,1,2});raw=load_mask(ROOT/cfg['raw_masks']/f'{vid}/masks/{i:06d}.png',(400,400),{0,1,2});r=data[i];assert array_hash(b)==r['final_sha256'] and array_hash(raw)==r['raw_sha256']
        with np.load(base/f'full_video/{vid}/layers/{i:06d}.npz') as z:subject=z['subject']>0
        return raw,old,b,r['info'],subject
    cap=cv2.VideoCapture(str(ROOT/f'data/videos/{vid}.mp4'));ok,frame=cap.read();cap.release();assert ok;c=context(0)
    for _ in range(cfg['warmup']):g.run(frame,*c)
    g.reset();cap=cv2.VideoCapture(str(ROOT/f'data/videos/{vid}.mp4'));n=0;times=[];started=time.perf_counter()
    with (dest/'frames.jsonl').open('w',encoding='utf-8') as f:
        while True:
            cycle=time.perf_counter();ok,frame=cap.read()
            if not ok:break
            assert array_hash(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))==data[n]['source_rgb_sha256'];c=context(n);a=time.perf_counter();g.run(frame,*c);ms=(time.perf_counter()-a)*1000;ids=packed(dest,n,g);f.write(json.dumps(dict(frame_index=n,source_rgb_sha256=data[n]['source_rgb_sha256'],source_timestamp=data[n]['source_timestamp'],baseline_sha256=data[n]['final_sha256'],raw_sha256=data[n]['raw_sha256'],info=g.info,**ids),allow_nan=False,separators=(',',':'))+'\n');times.append(dict(correction_ms=ms,cycle_ms=(time.perf_counter()-cycle)*1000));n+=1
            if n%500==0:print(arm,vid,n,flush=True)
    cap.release();elapsed=time.perf_counter()-started;assert n==len(data)==(722 if vid=='video_1' else 1972);g.reset();assert not g.layers
    atomic_json(dest/'timings.json',times);atomic_json(dest/'summary.json',dict(arm=arm,video=vid,frames=n,pid=os.getpid(),warmup=cfg['warmup'],state_reset=True,network_forwards=0,timings={k:stats([t[k] for t in times]) for k in times[0]},offline_fps=n/elapsed,wall_seconds=elapsed,measurement_scope='CPU correction after cached G0; decode/input read/save included only in cycle and throughput; no upstream neural/S2/G0 cost'))
    print('COMPLETE',arm,vid,flush=True)
def causal(cfg,out):
    bc=read(ROOT/cfg['baseline_config']);base=ROOT/cfg['baseline'];rows=read(ROOT/bc['reference_video'])['val'];bs=read(base/'video_causal/summary.json')['records']
    for arm in cfg['arms']:
        armout=out/arm;dest=armout/'video_causal';folder(dest);data={v:logs(armout/f'full_video/{v}/frames.jsonl') for v in ['video_1','video_2']};targets=[];new=[];before=[];records=[]
        for i,r in enumerate(rows):
            v,j=r['video_id'],r['frame_index'];source=armout/f'full_video/{v}'
            for d,suffix in [('masks','.png'),('layers','.npz')]:shutil.copy2(source/d/f'{j:06d}{suffix}',dest/d/f'{i:06d}{suffix}')
            gt=load_mask(r['mask'],(400,400),{0,1,2,255});b=load_mask(base/f'video_causal/masks/{i:06d}.png',(400,400),{0,1,2});assert array_hash(b)==bs[i]['final_sha256'];p=load_mask(dest/f'masks/{i:06d}.png',(400,400),{0,1,2});row=data[v][j];records.append(dict(index=i,sample_id=r['sample_id'],video_id=v,frame_index=j,info=row['info'],baseline_sha256=row['baseline_sha256'],final_sha256=row['final_sha256'],layer_hashes=row['layer_hashes']));targets.append(gt);new.append(p);before.append(b)
        report_scope(cfg,armout,'video_causal',rows,targets,new,before,records)
def changes(cfg,out):
    bc=read(ROOT/cfg['baseline_config']);result={}
    for arm in cfg['arms']:
        result[arm]={}
        for scope,cache in [('video_native','video_native'),('image_native','image_native'),('video_causal','video_native')]:
            ds=read(out/arm/scope/'summary.json');total=Counter();old_dark=new_dark=0;per=[]
            for i,r in enumerate(ds['records']):
                with np.load(ROOT/bc['static_cache']/cache/f'{i:04d}.npz') as z:gt=z['gt'];im=z['frame']
                b=load_mask(ROOT/cfg['baseline']/scope/f'masks/{i:06d}.png',gt.shape,{0,1,2});n=load_mask(out/arm/scope/f'masks/{i:06d}.png',gt.shape,{0,1,2});p=paired_changes(gt,b,n,2);total.update(p);dark=(gt==2)&(cv2.cvtColor(im,cv2.COLOR_BGR2LAB)[...,0]<85);old_dark+=int((dark&(b!=2)).sum());new_dark+=int((dark&(n!=2)).sum());per.append(dict(sample_id=r['sample_id'],**p))
            result[arm][scope]=dict(paired=dict(total),lowLab_FN_baseline=old_dark,lowLab_FN_new=new_dark,per_frame=per)
    atomic_json(out/'reference_changes.json',result)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',nargs=2);args=ap.parse_args();cfg=read(CONFIG);out=ROOT/cfg['output']
    if args.worker:return worker(cfg,out,*args.worker)
    assert not out.exists();out.mkdir();bc=read(ROOT/cfg['baseline_config']);network=read(ROOT/bc['network_preflight']);sources=dict(network['sources'])
    for p in [CONFIG,Path(__file__),ROOT/cfg['baseline_config'],ROOT/'tools/fiber_rgb_corrected_taper.py',ROOT/'tools/fiber_rgb_side_observation.py',ROOT/'tools/fiber_tapered_tangent_cap.py',ROOT/'tools/fiber_straight_semicircle.py',ROOT/'tests/test_rgb_side_observation.py',ROOT/'tests/test_rgb_corrected_taper.py',ROOT/'docs/plans/tapered_rgb_control_20261104.md']:sources[p.relative_to(ROOT).as_posix()]=sha256(p)
    assert all(sha256(Path(z['path']))==z['sha256'] for z in network['checkpoints'].values());assert all(sha256(Path(z['path']))==z['sha256'] for z in network['videos'])
    atomic_json(out/'preflight.json',dict(config=cfg,sources=sources,baseline_seal_sha256=sha256(ROOT/cfg['baseline']/'seal.json'),diagnosis_findings_sha256=sha256(ROOT/cfg['diagnosis']/'diagnosis_findings.md'),diagnosis_cause_sha256=sha256(ROOT/cfg['diagnosis']/'cause_analysis.json'),checkpoints=network['checkpoints'],videos=network['videos'],upstream_model_state_sha256=read(ROOT/cfg['baseline']/'full_video/video_1/summary.json')['model_state_sha256'],network_forwards=0,training_updates=0,train_pixels_read=0,reserved_test_pixels_read=0,default_changed=False))
    with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(ROOT/p,p)
    static(cfg,out)
    for arm in cfg['arms']:
        for vid in ['video_1','video_2']:
            atomic_json(out/'job.json',dict(phase='full_video',arm=arm,video=vid))
            with (out/f'{arm}_{vid}.log').open('w',encoding='utf-8') as f:subprocess.run([sys.executable,'-m','tools.run_tapered_rgb_control','--worker',arm,vid],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    causal(cfg,out);changes(cfg,out);assert all(sha256(ROOT/k)==h for k,h in sources.items());atomic_json(out/'job.json',dict(phase='inference_complete',candidate_full_frames=3*2694,static_candidate_frames=3*202,network_forwards=0));print('ALL_CONTROLS_COMPLETE',flush=True)
if __name__=='__main__':main()
