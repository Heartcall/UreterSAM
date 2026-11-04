"""One-stage runner. Later stages are rejected unless prior review passes."""
from pathlib import Path
import argparse,importlib,json,os,subprocess,sys,time,zipfile,shutil
import cv2,numpy as np,torch
from ureter_semis.experiment import atomic_json,seed_training
from .fiber_reference_workspace import ROOT,read,sha256,load_mask
from .fiber_photometric_input import infer
from .run_fiber_temporal_state_control import construct
from .run_fiber_dark_input_control import count_forwards,checked_writer,verify_video,select_groups
from .run_two_network_experts import array_hash
from .run_fiber_blue_aplus import frame_read
from .replay_classwise_experts import save_png,stats,model_state_hash
from .run_classwise_expert_control import summarize
from .diagnose_joint_failures import paired_changes

CONFIG=ROOT/'configs/weak_blue_staged_control_20261104.json'
def pipeline(stage):return importlib.import_module('tools.fiber_weak_stage'+str(stage)).GrowthPipeline
def module(stage):return ROOT/f'tools/fiber_weak_stage{stage}.py'
def logs(path):return [json.loads(s) for s in Path(path).open(encoding='utf-8')]
def packed(folder,index,p):
    arrays={k:np.asarray(v,np.uint8) for k,v in p.layers.items()};np.savez_compressed(folder/'layers'/f'{index:06d}.npz',**arrays);save_png(folder/'masks'/f'{index:06d}.png',p.final_mask)
    return {k:dict(sha256=array_hash(v),pixels=int(v.sum())) for k,v in arrays.items()}
def folder(path):path.mkdir(parents=True);(path/'layers').mkdir();(path/'masks').mkdir()
def maincfg():return read(CONFIG)
def cache(cfg):
    out=ROOT/cfg['output'];base=ROOT/cfg['baseline'];pre=read(base/'preflight.json');target=out/'static_cache'
    if (target/'identity.json').exists():return
    target.mkdir(parents=True,exist_ok=False);seed_training(42);experts=construct(pre);counts,handles=count_forwards(experts);before=model_state_hash(experts);ids={}
    for scope,key in [('video_native','video'),('image_native','image')]:
        rows=read(ROOT/cfg['references'][key])['val'];old=read(base/scope/'summary.json')['records'];dest=target/scope;dest.mkdir()
        for i,(row,record) in enumerate(zip(rows,old)):
            frame=frame_read(row['image']);raw,sp,fp,_=infer(experts,frame,'raw2');gt=load_mask(row['mask'],raw.shape,{0,1,2,255});h=dict(mask_raw_sha256=array_hash(raw),stone_probability_raw_sha256=array_hash(sp),fiber_probability_raw_sha256=array_hash(fp));assert all(record[k]==v for k,v in h.items());np.savez_compressed(dest/f'{i:04d}.npz',frame=frame,raw=raw,sp=sp,fp=fp,gt=gt);ids[f'{scope}/{i:04d}.npz']=sha256(dest/f'{i:04d}.npz')
            if (i+1)%40==0:print('static_cache',scope,i+1,flush=True)
    assert counts==[202,202] and model_state_hash(experts)==before
    for h in handles:h.remove()
    atomic_json(target/'identity.json',dict(files=ids,network_branches=counts,paired_forwards=202,model_state_sha256=before,source_identity=str(base/'preflight.json'),training_updates=0));del experts;torch.cuda.empty_cache()
def static(cfg,stage,out):
    basecfg=read(ROOT/cfg['baseline_config']);cls=pipeline(stage);P=read(ROOT/cfg['baseline']/'preflight.json');t1=read(ROOT/basecfg['t1']['config'])
    for scope,key in [('video_native','video'),('image_native','image')]:
        dest=out/scope;folder(dest);rows=read(ROOT/cfg['references'][key])['val'];pred=[];targets=[];records=[];raws=[]
        for i,row in enumerate(rows):
            with np.load(ROOT/cfg['output']/f'static_cache/{scope}/{i:04d}.npz') as z:frame,raw,fp,gt=[z[k].copy() for k in ['frame','raw','fp','gt']]
            p=cls(basecfg,'weak_blue_extension_tip',t1,False);_,_,_,meta=p.run(frame,raw,fp);layers=packed(dest,i,p);pred.append(p.final_mask.copy());targets.append(gt);raws.append(raw);records.append(dict(index=i,sample_id=row['sample_id'],video_id=row.get('video_id'),frame_index=row.get('frame_index'),outline=p.outline,meta=meta,stage_layers=layers,final_mask_sha256=array_hash(p.final_mask),raw_sha256=array_hash(raw)))
        groups={name:dict(frames=len(ids),summary=summarize([rows[i] for i in ids],[targets[i] for i in ids],[pred[i] for i in ids])) for name,ids in select_groups(rows,scope=='video_native').items()};atomic_json(dest/'summary.json',dict(summary=summarize(rows,targets,pred),raw=summarize(rows,targets,raws),groups=groups,records=records,new_forwards=0))
def worker(cfg,stage,vid,out):
    pre=read(out/'preflight.json');base=ROOT/cfg['baseline'];oldpre=read(base/'preflight.json');video=next(v for v in oldpre['videos'] if v['video_id']==vid)
    assert all(sha256(ROOT/k)==v for k,v in pre['sources'].items());assert sha256(video['path'])==video['sha256'];assert all(sha256(x['path'])==x['sha256'] for x in oldpre['checkpoints'].values())
    seed_training(42);experts=construct(oldpre);before=model_state_hash(experts);counts,handles=count_forwards(experts);basecfg=read(ROOT/cfg['baseline_config']);p=pipeline(stage)(basecfg,'weak_blue_extension_tip',read(ROOT/basecfg['t1']['config']),True);source=read(base/f'source_{vid}.json')['frames'];reference=logs(base/f'full_video/weak_blue_extension_tip/{vid}/frames.jsonl');dest=out/'full_video'/vid;folder(dest);cap=cv2.VideoCapture(video['path']);ok,first=cap.read();cap.release();assert ok
    def run(frame,ts):
        experts.sync();start=time.perf_counter();raw,sp,fp,_=infer(experts,frame,'raw2');mid=time.perf_counter();_,_,_,meta=p.run(frame,raw,fp,ts,vid);end=time.perf_counter();return raw,sp,fp,meta,dict(network_pipeline_ms=(mid-start)*1000,postprocess_ms=(end-mid)*1000,pipeline_total_ms=(end-start)*1000)
    for j in range(cfg['execution']['warmup']):run(first,j/video['fps'])
    p.reset();counts[:]=[0,0];cap=cv2.VideoCapture(video['path']);times=[];n=0;wall=time.perf_counter();writer=checked_writer(dest/'native_replay.mp4',video['fps'],(800,400))
    try:
        with (dest/'frames.jsonl').open('w',encoding='utf-8') as f:
            while True:
                begin=time.perf_counter();ok,frame=cap.read()
                if not ok:break
                raw,sp,fp,meta,t=run(frame,source[n]['timestamp_seconds']);old=reference[n];assert array_hash(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))==source[n]['rgb_raw_sha256'];assert all(array_hash(v)==old[k] for v,k in [(raw,'mask_raw_sha256'),(sp,'stone_probability_raw_sha256'),(fp,'fiber_probability_raw_sha256')]);layers=packed(dest,n,p);assert np.all(p.final_mask[raw==1]==1)
                a=frame.copy();b=frame.copy()
                for k,c in [(1,(0,0,255)),(2,(255,210,0))]:
                    q=raw==k;a[q]=(.65*a[q]+.35*np.array(c)).astype(np.uint8);q=p.final_mask==k;b[q]=(.65*b[q]+.35*np.array(c)).astype(np.uint8)
                writer.write(np.hstack([a,b]));f.write(json.dumps(dict(frame_index=n,timestamp_seconds=source[n]['timestamp_seconds'],source_rgb_sha256=source[n]['rgb_raw_sha256'],raw_sha256=array_hash(raw),fiber_probability_sha256=array_hash(fp),stone_probability_sha256=array_hash(sp),final_mask_sha256=array_hash(p.final_mask),outline=p.outline,meta=meta,stage_layers=layers),allow_nan=False,separators=(',',':'))+'\n');t['cycle_ms']=(time.perf_counter()-begin)*1000;times.append(t);n+=1
                if n%500==0:print('stage',stage,vid,n,flush=True)
    finally:cap.release();writer.release()
    elapsed=time.perf_counter()-wall;assert n==video['frames'] and counts==[n,n] and model_state_hash(experts)==before;p.reset();assert p.tracker.source is None
    for h in handles:h.remove()
    atomic_json(dest/'timings.json',times);atomic_json(dest/'summary.json',dict(stage=stage,video_id=vid,pid=os.getpid(),frames=n,network_branches=counts,warmup_branches=[12,12],model_state_sha256=before,timings={k:stats([x[k] for x in times]) for k in times[0]},offline_fps=n/elapsed,wall_seconds=elapsed,state_reset=True,replay=verify_video(dest/'native_replay.mp4',n)));print('stage_complete',stage,vid,flush=True)
def causal(cfg,out):
    rows=read(ROOT/cfg['references']['video'])['val'];ds=out/'video_causal';folder(ds);full={vid:logs(out/f'full_video/{vid}/frames.jsonl') for vid in ['video_1','video_2']};targets=[];pred=[];records=[]
    for i,row in enumerate(rows):
        vid=row['video_id'];j=row['frame_index'];src=out/f'full_video/{vid}';shutil.copy2(src/'masks'/f'{j:06d}.png',ds/'masks'/f'{i:06d}.png');shutil.copy2(src/'layers'/f'{j:06d}.npz',ds/'layers'/f'{i:06d}.npz');r=full[vid][j];records.append(dict(index=i,sample_id=row['sample_id'],video_id=vid,frame_index=j,**{k:r[k] for k in ['outline','meta','stage_layers','final_mask_sha256']}));gt=load_mask(row['mask'],(row['height'],row['width']),{0,1,2,255});targets.append(gt);pred.append(load_mask(ds/'masks'/f'{i:06d}.png',gt.shape,{0,1,2}))
    groups={name:dict(frames=len(ids),summary=summarize([rows[i] for i in ids],[targets[i] for i in ids],[pred[i] for i in ids])) for name,ids in select_groups(rows,True).items()};atomic_json(ds/'summary.json',dict(summary=summarize(rows,targets,pred),groups=groups,records=records,new_forwards=0))
def main():
    a=argparse.ArgumentParser();a.add_argument('--stage',type=int,required=True);a.add_argument('--worker');args=a.parse_args();cfg=maincfg();stage=args.stage;assert 1<=stage<=4;out=ROOT/cfg['output']/cfg['stages'][stage-1]
    if args.worker:return worker(cfg,stage,args.worker,out)
    if stage>1:assert read(out.parent/cfg['stages'][stage-2]/'review.json')['progression_passed'],'Prior review failed; do not proceed'
    if out.exists():raise FileExistsError(out)
    assert module(stage).exists();out.mkdir(parents=True);cache(cfg);prior=read(ROOT/cfg['baseline']/'preflight.json');sources=dict(prior['sources']);extra=[Path(__file__),CONFIG,ROOT/'docs/plans/weak_blue_staged_execution_20261104.md']
    for k in range(1,stage+1):extra.extend([module(k),ROOT/f'tests/test_weak_stage{k}.py'])
    for p in extra:sources[p.relative_to(ROOT).as_posix()]=sha256(p)
    atomic_json(out/'preflight.json',dict(stage=stage,config=cfg,sources=sources,inputs=prior['inputs'],checkpoints=prior['checkpoints'],videos=prior['videos'],baseline_preflight_sha256=sha256(ROOT/cfg['baseline']/'preflight.json'),train_pixels_read=0,test_pixels_read=0,default_changed=False))
    with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(ROOT/p,p)
    static(cfg,stage,out)
    for vid in ['video_1','video_2']:
        atomic_json(out/'job.json',dict(phase='full_video',stage=stage,video=vid))
        with (out/f'{vid}.log').open('w',encoding='utf-8') as f:subprocess.run([sys.executable,'-m','tools.run_weak_staged_control','--stage',str(stage),'--worker',vid],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    causal(cfg,out);assert all(sha256(ROOT/k)==v for k,v in sources.items());atomic_json(out/'job.json',dict(phase='inference_completed_pending_review',stage=stage,full_frames=2694,static=202,new_stage_static_forwards=0));print('STAGE_INFERENCE_DONE',stage,flush=True)
if __name__=='__main__':main()
