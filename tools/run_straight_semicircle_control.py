"""Frozen geometry-first control; independent full-video neural workers."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,zipfile,shutil
from collections import Counter
import cv2,numpy as np,torch
from .fiber_reference_workspace import ROOT,read,sha256,load_mask
from .fiber_straight_semicircle import StraightSemicircle
from .fiber_photometric_input import infer
from .run_fiber_temporal_state_control import construct
from .run_fiber_dark_input_control import count_forwards,checked_writer,verify_video,select_groups
from .run_two_network_experts import array_hash
from .replay_classwise_experts import save_png,stats,model_state_hash
from .run_classwise_expert_control import summarize
from .diagnose_joint_failures import paired_changes
from ureter_semis.experiment import atomic_json,seed_training

CONFIG=ROOT/'configs/straight_semicircle_control_20261104.json'
def logs(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]
def folder(path):
    path.mkdir(parents=True)
    for k in ['masks','proposals','layers']:(path/k).mkdir()
def packed(path,i,p):
    save_png(path/'masks'/f'{i:06d}.png',p.final_mask);save_png(path/'proposals'/f'{i:06d}.png',p.layers['geometric_proposal']);np.savez_compressed(path/'layers'/f'{i:06d}.npz',**p.layers)
    return dict(final_sha256=array_hash(p.final_mask),proposal_sha256=array_hash(p.layers['geometric_proposal']),layer_hashes={k:array_hash(v) for k,v in p.layers.items()})
def report_scope(cfg,out,scope,rows,targets,pred,old,records):
    groups={name:dict(frames=len(ids),summary=summarize([rows[i] for i in ids],[targets[i] for i in ids],[pred[i] for i in ids]),baseline=summarize([rows[i] for i in ids],[targets[i] for i in ids],[old[i] for i in ids])) for name,ids in select_groups(rows,scope!='image_native').items()}
    atomic_json(out/scope/'summary.json',dict(summary=summarize(rows,targets,pred),baseline=summarize(rows,targets,old),groups=groups,records=records,applied=sum(r['info']['applied'] for r in records),reasons=dict(Counter(r['info']['reason'] for r in records))))
def static(cfg,out):
    cache=ROOT/cfg['static_cache'];identity=read(cache/'identity.json');assert all(sha256(cache/k)==v for k,v in identity['files'].items())
    for scope,key in [('video_native','reference_video'),('image_native','reference_image')]:
        dest=out/scope;folder(dest);rows=read(ROOT/cfg[key])['val'];oldrows=read(ROOT/cfg['old_result']/scope/'summary.json')['records'];targets=[];pred=[];before=[];records=[]
        for i,row in enumerate(rows):
            with np.load(cache/scope/f'{i:04d}.npz') as z:frame,raw,p,gt=[z[k].copy() for k in ['frame','raw','fp','gt']]
            old=load_mask(ROOT/cfg['old_result']/scope/'masks'/f'{i:06d}.png',raw.shape,{0,1,2});assert array_hash(old)==oldrows[i]['final_mask_sha256'];g=StraightSemicircle(cfg);g.run(frame,raw,p,old);ids=packed(dest,i,g);records.append(dict(index=i,sample_id=row['sample_id'],video_id=row.get('video_id'),frame_index=row.get('frame_index'),old_sha256=array_hash(old),raw_sha256=array_hash(raw),info=g.info,**ids));targets.append(gt);pred.append(g.final_mask.copy());before.append(old)
        report_scope(cfg,out,scope,rows,targets,pred,before,records)
def worker(cfg,out,vid):
    pre=read(out/'preflight.json');assert all(sha256(ROOT/k)==h for k,h in pre['sources'].items());network=read(ROOT/cfg['network_preflight']);video=next(x for x in network['videos'] if x['video_id']==vid);assert sha256(video['path'])==video['sha256'];assert all(sha256(x['path'])==x['sha256'] for x in network['checkpoints'].values());source=read(ROOT/cfg['source_identity']/f'source_{vid}.json')['frames'];oldrows=logs(ROOT/cfg['old_result']/f'full_video/{vid}/frames.jsonl');oldpath=ROOT/cfg['old_result']/f'full_video/{vid}/masks'
    seed_training(42);experts=construct(network);before=model_state_hash(experts);count,handles=count_forwards(experts);g=StraightSemicircle(cfg);dest=out/'full_video'/vid;folder(dest);cap=cv2.VideoCapture(video['path']);ok,first=cap.read();cap.release();assert ok;old0=load_mask(oldpath/'000000.png',(400,400),{0,1,2})
    def run(frame,old):
        experts.sync();start=time.perf_counter();raw,sp,fp,_=infer(experts,frame,'raw2');mid=time.perf_counter();g.run(frame,raw,fp,old);end=time.perf_counter();return raw,sp,fp,dict(network_pipeline_ms=(mid-start)*1000,geometry_ms=(end-mid)*1000,pipeline_total_ms=(end-start)*1000)
    for _ in range(cfg['warmup']):run(first,old0)
    count[:]=[0,0];g.reset();cap=cv2.VideoCapture(video['path']);writer=checked_writer(dest/'native_replay.mp4',video['fps'],(800,400));times=[];n=0;wall=time.perf_counter()
    try:
        with (dest/'frames.jsonl').open('w',encoding='utf-8') as f:
            while True:
                begin=time.perf_counter();ok,frame=cap.read()
                if not ok:break
                old=load_mask(oldpath/f'{n:06d}.png',(400,400),{0,1,2});a=oldrows[n];assert array_hash(old)==a['final_mask_sha256'];raw,sp,fp,t=run(frame,old);assert array_hash(raw)==a['raw_sha256'] and array_hash(sp)==a['stone_probability_sha256'] and array_hash(fp)==a['fiber_probability_sha256'];assert array_hash(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))==source[n]['rgb_raw_sha256'];ids=packed(dest,n,g)
                panels=[]
                for mask in [old,g.final_mask]:
                    im=frame.copy()
                    for k,c in [(1,(0,0,255)),(2,(255,210,0))]:
                        q=mask==k;im[q]=(.65*im[q]+.35*np.asarray(c)).astype(np.uint8)
                    panels.append(im)
                writer.write(np.hstack(panels));f.write(json.dumps(dict(frame_index=n,source_rgb_sha256=source[n]['rgb_raw_sha256'],source_timestamp=source[n]['timestamp_seconds'],raw_sha256=array_hash(raw),stone_probability_sha256=array_hash(sp),fiber_probability_sha256=array_hash(fp),old_sha256=array_hash(old),info=g.info,**ids),allow_nan=False,separators=(',',':'))+'\n');t['cycle_ms']=(time.perf_counter()-begin)*1000;times.append(t);n+=1
                if n%500==0:print(vid,n,flush=True)
    finally:cap.release();writer.release()
    elapsed=time.perf_counter()-wall;assert n==video['frames'] and count==[n,n] and model_state_hash(experts)==before;g.reset();assert g.layers=={}
    for h in handles:h.remove()
    atomic_json(dest/'timings.json',times);atomic_json(dest/'summary.json',dict(video_id=vid,frames=n,pid=os.getpid(),network_branches=count,warmup_branches=[cfg['warmup']]*2,model_state_sha256=before,timings={k:stats([x[k] for x in times]) for k in times[0]},offline_fps=n/elapsed,wall_seconds=elapsed,state_reset=True,weak_growth_calls=0,replay=verify_video(dest/'native_replay.mp4',n)));print('COMPLETE',vid,flush=True)
def causal(cfg,out):
    dest=out/'video_causal';folder(dest);rows=read(ROOT/cfg['reference_video'])['val'];full={v:logs(out/f'full_video/{v}/frames.jsonl') for v in ['video_1','video_2']};oldrows=read(ROOT/cfg['old_result']/'video_causal/summary.json')['records'];targets=[];pred=[];before=[];records=[]
    for i,row in enumerate(rows):
        v,j=row['video_id'],row['frame_index'];source=out/f'full_video/{v}'
        for k,suffix in [('masks','.png'),('proposals','.png'),('layers','.npz')]:shutil.copy2(source/k/f'{j:06d}{suffix}',dest/k/f'{i:06d}{suffix}')
        old=load_mask(ROOT/cfg['old_result']/'video_causal/masks'/f'{i:06d}.png',(400,400),{0,1,2});assert array_hash(old)==oldrows[i]['final_mask_sha256'];n=load_mask(dest/'masks'/f'{i:06d}.png',(400,400),{0,1,2});gt=load_mask(row['mask'],(400,400),{0,1,2,255});r=full[v][j];records.append(dict(index=i,sample_id=row['sample_id'],video_id=v,frame_index=j,**{k:r[k] for k in ['info','old_sha256','raw_sha256','final_sha256','proposal_sha256','layer_hashes']}));targets.append(gt);pred.append(n);before.append(old)
    report_scope(cfg,out,'video_causal',rows,targets,pred,before,records)
def references(cfg,out):
    result={}
    for scope,kind in [('video_native','video_native'),('video_causal','video_native'),('image_native','image_native')]:
        ds=read(out/scope/'summary.json');pairs=Counter();dark_old=dark_new=dark_total=0;per=[]
        for i,r in enumerate(ds['records']):
            with np.load(ROOT/cfg['static_cache']/kind/f'{i:04d}.npz') as z:gt=z['gt'];im=z['frame']
            old=load_mask(ROOT/cfg['old_result']/scope/'masks'/f'{i:06d}.png',gt.shape,{0,1,2});new=load_mask(out/scope/'masks'/f'{i:06d}.png',gt.shape,{0,1,2});d=paired_changes(gt,old,new,2);pairs.update(d);dark=(gt==2)&(cv2.cvtColor(im,cv2.COLOR_BGR2LAB)[...,0]<85);dark_old+=int((dark&(old!=2)).sum());dark_new+=int((dark&(new!=2)).sum());dark_total+=int(dark.sum());per.append(dict(index=i,sample_id=r['sample_id'],**d))
        result[scope]=dict(paired=dict(pairs),lowLab_FN_old=dark_old,lowLab_FN_new=dark_new,lowLab_reference=dark_total,per_frame=per)
    atomic_json(out/'reference_changes.json',result)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker');args=ap.parse_args();cfg=read(CONFIG);out=ROOT/cfg['output']
    if args.worker:return worker(cfg,out,args.worker)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);network=read(ROOT/cfg['network_preflight']);sources=dict(network['sources'])
    for p in [Path(__file__),CONFIG,ROOT/'tools/fiber_straight_semicircle.py',ROOT/'tools/fiber_weak_stage1.py',ROOT/'tests/test_straight_semicircle.py',ROOT/'docs/plans/straight_semicircle_execution_20261104.md']:sources[p.relative_to(ROOT).as_posix()]=sha256(p)
    atomic_json(out/'preflight.json',dict(config=cfg,sources=sources,old_seal_sha256=sha256(ROOT/cfg['old_seal']),old_result_preflight_sha256=sha256(ROOT/cfg['old_result']/'preflight.json'),static_cache_identity_sha256=sha256(ROOT/cfg['static_cache']/'identity.json'),checkpoints=network['checkpoints'],videos=network['videos'],training_updates=0,train_pixels_read=0,reserved_test_pixels_read=0,default_changed=False,weak_growth_calls=0))
    with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(ROOT/p,p)
    static(cfg,out)
    for vid in ['video_1','video_2']:
        atomic_json(out/'job.json',dict(phase='full_video',video=vid))
        with (out/f'{vid}.log').open('w',encoding='utf-8') as f:subprocess.run([sys.executable,'-m','tools.run_straight_semicircle_control','--worker',vid],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    causal(cfg,out);references(cfg,out);assert all(sha256(ROOT/k)==h for k,h in sources.items());atomic_json(out/'job.json',dict(phase='inference_complete',full_frames=2694,static_frames=202,new_static_network_forwards=0));print('INFERENCE_COMPLETE',flush=True)
if __name__=='__main__':main()
