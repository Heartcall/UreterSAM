"""Frozen train-calibrated direct shape localization with full causal replay."""
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

import cv2
import numpy as np
from PIL import Image
import torch

from ureter_semis.experiment import atomic_json,input_identity,seed_training,sha256,source_identity,runtime_identity
from ureter_semis.supervised_trainer import assess
from ureter_semis.video_metrics import estimated_fov
from .classwise_experts import ClasswiseExperts
from .direct_fiber_template import DirectTracker,locate
from .run_fiber_shape_control import core
from .run_classwise_expert_control import summarize
from .diagnose_joint_failures import paired_changes
from .replay_classwise_experts import intervals,mask_diagnostics,summarize_diagnostics,stats,save_png,model_state_hash

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/a1_direct_fiber_template_20261029'
NAMES=('raw_classwise','image_template','soft_template','causal_template')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def load(p):
    with Image.open(p) as im:return np.asarray(im).copy()


def compose(raw,stone,outputs):
    return {'raw_classwise':raw,**{n:np.where(v['fiber'],2,np.where(stone==1,1,0)).astype(np.uint8) for n,v in outputs.items()}}


def pair_summary(rows,targets,predictions):
    result={}
    for name in predictions:
        if name=='raw_classwise':continue
        result[name]={str(k):dict(sum((Counter(paired_changes(t,a,b,k)) for t,a,b in zip(targets,predictions['raw_classwise'],predictions[name])),Counter())) for k in (1,2)}
    return result


def image_eval(rows,experts,prior,config):
    names=config['image_arms'];folder=OUT/'image_native';folder.mkdir()
    for n in names:(folder/n).mkdir()
    targets=[];predictions={n:[] for n in names};records=[]
    for i,row in enumerate(rows):
        rgb=np.asarray(Image.open(row['image']).convert('RGB'));frame=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        raw,stone,_,p,core_ms=core(experts,frame);outputs,timing=locate(frame,prior,config,p,requested=names[1:]);masks=compose(raw,stone,outputs)
        records.append({'sample_id':row['sample_id'],'outputs':{n:v['record'] for n,v in outputs.items()},'timings':timing})
        targets.append(load(row['mask']))
        for n,a in masks.items():save_png(folder/n/f'{i:04d}.png',a);predictions[n].append(a)
    result={'models':{n:summarize(rows,targets,a) for n,a in predictions.items()},'records':records,'paired_changes':pair_summary(rows,targets,predictions),'cohorts':{}}
    previous=read(ROOT/'runs/a1_classwise_experts_20261028/summary.json')['image']['historical_exposure_cohorts'];exposed=set(previous['historical_old190_exposed']['sample_ids'])
    for label,seen in [('historical_old190_exposed',True),('not_matched_to_old190_exposure',False)]:
        ids=[i for i,r in enumerate(rows) if (r['sample_id'] in exposed)==seen]
        result['cohorts'][label]={'images':len(ids),'sample_ids':[rows[i]['sample_id'] for i in ids],'models':{n:summarize([rows[i] for i in ids],[targets[i] for i in ids],[a[i] for i in ids])['metrics'] for n,a in predictions.items()}}
    atomic_json(folder/'comparison.json',result);return result


def render(frame,masks,title):
    panels=[]
    for name in ('Input',*NAMES):
        panel=frame.copy()
        if name!='Input':
            for k,color in [(1,(40,40,235)),(2,(235,210,20))]:
                selected=masks[name]==k;panel[selected]=(.48*frame[selected]+.52*np.asarray(color)).astype(np.uint8)
        header=np.zeros((34,frame.shape[1],3),np.uint8);cv2.putText(header,f'{name} | {title}',(4,21),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1,cv2.LINE_AA)
        panels.append(np.vstack([header,panel]))
    return np.hstack(panels)


def run_video(item,experts,prior,config,reviewed,progress):
    vid=item['video_id'];folder=OUT/vid;folder.mkdir()
    for n in (*NAMES,'clips','keyframes'):(folder/n).mkdir()
    cap=cv2.VideoCapture(item['path']);writers=[]
    if not cap.isOpened():raise ValueError('Cannot open video')
    fps=float(cap.get(cv2.CAP_PROP_FPS));count=item['frames'];events=intervals(read(ROOT/'configs/replay_classwise_experts_20261028.json'),vid,count)
    tracker=DirectTracker(config['tracking']);selected={r['frame_index']:r for r in reviewed if r['video_id']==vid};samples={};alignment=[]
    def writer(path):
        w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(item['width']*5,item['height']+34))
        if not w.isOpened():raise RuntimeError('Cannot write preview')
        writers.append(w);return w
    try:
        ok,first=cap.read()
        if not ok:raise ValueError('Empty video')
        for _ in range(12):core(experts,first)
        raw,stone,_,p,_=core(experts,first)
        a,_=locate(first,prior,config,p,DirectTracker(config['tracking']));b,_=locate(first,prior,config,p,DirectTracker(config['tracking']))
        if any(a[n]['record']!=b[n]['record'] or not np.array_equal(a[n]['fiber'],b[n]['fiber']) for n in a):raise ValueError('Direct search repeat failure')
        cap.release();cap=cv2.VideoCapture(item['path']);preview=writer(folder/'comparison.mp4');clipwriters=[]
        for i,e in enumerate(events):e['clip']=f'clips/{i:02d}_{e["event"]}_{e["one_based_inclusive"][0]}-{e["one_based_inclusive"][1]}.mp4';clipwriters.append(writer(folder/e['clip']))
        rows=[];timings=[];previous={n:None for n in NAMES};different=0;started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        with (folder/'frames.jsonl').open('w',encoding='utf-8') as log:
            while True:
                t0=time.perf_counter();ok,frame=cap.read();decoded=time.perf_counter()
                if not ok:break
                i=len(rows)
                if i>=count or frame.shape[:2]!=(item['height'],item['width']):raise ValueError('Video frame geometry/count differs')
                raw,stone,_,p,core_ms=core(experts,frame);outputs,timing=locate(frame,prior,config,p,tracker);masks=compose(raw,stone,outputs)
                cached=load(ROOT/f'runs/a1_classwise_replay_20261028/{vid}/classwise/{i:06d}.png');different+=int((raw!=cached).sum())
                if i in selected:
                    row=selected[i];exact=np.array_equal(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),np.asarray(Image.open(row['image']).convert('RGB')))
                    alignment.append({'sample_id':row['sample_id'],'frame_index':i,'pixel_exact':exact})
                    if not exact:raise ValueError('Reviewed RGB alignment differs')
                    samples[row['sample_id']]={n:a.copy() for n,a in masks.items()}
                fov=estimated_fov(frame,frame.shape[:2]);diagnostics={n:mask_diagnostics(a,fov,previous[n]) for n,a in masks.items()}
                for n,a in masks.items():save_png(folder/n/f'{i:06d}.png',a)
                canvas=render(frame,masks,f'{vid} frame{i+1}');preview.write(canvas)
                for e,w in zip(events,clipwriters):
                    if e['start_frame']<=i<e['end_frame_exclusive']:w.write(canvas)
                if i in selected:save_png(folder/'keyframes'/f'{i:06d}.png',canvas)
                record={'frame_index':i,'one_based_frame':i+1,'outputs':{n:v['record'] for n,v in outputs.items()},'models':diagnostics,'dense_ground_truth_available':False}
                log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
                timings.append({'shared3model_core_ms':core_ms,**timing,'decode_ms':(decoded-t0)*1000,'all_branch_cycle_ms':(time.perf_counter()-t0)*1000});rows.append(record);previous=masks
                if (i+1)%100==0:progress('replaying',video_id=vid,frames_completed=i+1,expected_frames=count);print(f'{vid}: {i+1}/{count}',flush=True)
        for w in writers:w.release()
        writers.clear();elapsed=time.perf_counter()-started
        if len(rows)!=count or len(alignment)!=len(selected):raise ValueError('Final frame counts differ')
        statuses=Counter(r['outputs']['causal_template']['tracking']['status'] for r in rows);tracker.reset()
        result={'video_id':vid,'frames':count,'fps':fps,'repeat_exact':True,'tracker_reset':tracker.pose is None,'raw_historical_different_pixels':different,'reviewed_alignment':alignment,
            'accepted':{n:sum(r['outputs'][n]['accepted'] for r in rows) for n in NAMES[1:]},'tracking_statuses':dict(statuses),
            'models':{n:summarize_diagnostics([{'frame_index':r['frame_index'],**r['models'][n]} for r in rows],fps) for n in NAMES},
            'timings':{k:stats([r[k] for r in timings]) for k in timings[0]},'offline_all_branches_fps':count/elapsed,'wall_seconds_including_writer_flush':elapsed,
            'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'user_intervals':events,'dense_accuracy_metrics':None,'scope':'Shared3model research replay; image-only arm does not consume neural probability but isolated benchmark is needed for1model total cost'}
        for e in events:
            part=rows[e['start_frame']:e['end_frame_exclusive']];e['models']={n:summarize_diagnostics([{'frame_index':r['frame_index'],**r['models'][n]} for r in part],fps) for n in NAMES}
        atomic_json(folder/'timings.json',timings);atomic_json(folder/'summary.json',result);return result,samples
    finally:
        cap.release()
        for w in writers:w.release()


def main():
    if not OUT.exists() or (OUT/'job.json').exists():raise FileExistsError('Calibrate into a fresh folder first; never overwrite a run')
    config_path=ROOT/'configs/direct_fiber_template_20261029.json';config=read(config_path);calibration=read(OUT/'calibration.json');prior=read(OUT/'template.json')
    if sha256(config_path)!=calibration['config_sha256']:raise ValueError('Calibration config changed')
    job={'pid':os.getpid(),'gradient_updates':0,'template_calibration_train_images':521,'default_model_changed':False,'s2_accepted':False,'stone_memory_started':False}
    def progress(phase,**extra):job.update(phase=phase,**extra);atomic_json(OUT/'job.json',job)
    progress('preflight')
    try:
        mpath=ROOT/config['manifest'];m=read(mpath);vm_path=mpath.parent/'video_validation_manifest.json';vm=read(vm_path)
        if sha256(mpath)!=calibration['manifest_sha256'] or any(sha256(Path(n))!=h for n,h in calibration['files'].items()):raise ValueError('TRAIN calibration files changed')
        identity=input_identity(mpath,Path(m['root']),{'train':m['train'],'val':m['val']+vm['val']})
        checkpoints={n:v for n,v in read(ROOT/config['experts'])['checkpoints'].items() if n in ('old190','A0')};inputs=read(ROOT/'runs/baselines/a0_20261027/config.json')['inputs']
        for v in list(checkpoints.values())+inputs:
            if sha256(Path(v['path']))!=v['sha256']:raise ValueError('Expert/video identity differs')
        extra=['tools/direct_fiber_template.py','tools/calibrate_direct_fiber_template.py','tools/run_direct_fiber_template.py','tools/benchmark_direct_fiber_template.py','tests/test_direct_fiber_template.py','tools/classwise_experts.py','tools/fiber_shape_prior.py','tools/run_fiber_shape_control.py','tools/replay_classwise_experts.py','tools/run_classwise_expert_control.py','tools/diagnose_joint_failures.py','tools/evaluate_native_checkpoints.py','tools/evaluate_spatial_detail.py']
        sources=source_identity();sources.update({n:sha256(ROOT/n) for n in extra})
        related=[config_path,ROOT/'configs/benchmark_direct_fiber_template_20261029.json',OUT/'template.json',OUT/'calibration.json',vm_path,ROOT/config['experts'],ROOT/'runs/baselines/a0_20261027/config.json',ROOT/'configs/replay_classwise_experts_20261028.json']
        for v in inputs:related+=sorted((ROOT/f'runs/a1_classwise_replay_20261028/{v["video_id"]}/classwise').glob('*.png'))
        preflight={'config':config,'template':prior,'inputs':identity,'calibration':calibration,'checkpoints':checkpoints,'video_inputs':inputs,'sources':sources,'related':{str(p):sha256(p) for p in related},'reserved_test_pixels_read':False}
        atomic_json(OUT/'preflight.json',preflight)
        with zipfile.ZipFile(OUT/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for n in sources:z.write(ROOT/n,n.replace('\\','/'))
            for p in (config_path,OUT/'template.json',ROOT/'configs/benchmark_direct_fiber_template_20261029.json'):z.write(p,str(p.relative_to(ROOT)))
        seed_training(42);experts=ClasswiseExperts(checkpoints['old190']['path'],checkpoints['A0']['path']);before=model_state_hash(experts);atomic_json(OUT/'runtime.json',runtime_identity(experts.device))
        progress('image132');image=image_eval(m['val'],experts,prior,config);selected={};videos={}
        for item in inputs:
            progress('loading_video',video_id=item['video_id'],frames_completed=0,expected_frames=item['frames'])
            result,samples=run_video(item,experts,prior,config,vm['val'],progress);videos[item['video_id']]=result;selected.update(samples)
        targets=[load(r['mask']) for r in vm['val']];predictions={n:[selected[r['sample_id']][n] for r in vm['val']] for n in NAMES}
        video={n:summarize(vm['val'],targets,a) for n,a in predictions.items()};pairs=pair_summary(vm['val'],targets,predictions)
        assert model_state_hash(experts)==before and all(not model.training for model in experts.models)
        del experts;torch.cuda.empty_cache();performance={}
        for name in NAMES:
            progress('isolated_cost',arm=name)
            with (OUT/f'benchmark_{name}.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-m','tools.benchmark_direct_fiber_template','--arm',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            performance[name]=read(OUT/f'performance_{name}.json')
        assert input_identity(mpath,Path(m['root']),{'train':m['train'],'val':m['val']+vm['val']})==identity
        assert all(sha256(ROOT/n)==h for n,h in sources.items()) and all(sha256(Path(n))==h for n,h in preflight['related'].items())
        assert all(sha256(Path(v['path']))==v['sha256'] for v in list(checkpoints.values())+inputs)
        result={'image':image,'video29':video,'video29_paired_changes':pairs,'videos':videos,'performance':performance,
            'screening':{n:assess(v['metrics'],config['gates']) for n,v in video.items()},'shared_template':prior,'gradient_updates':0,'calibration_train_images':521,
            'default_model_changed':False,'s2_accepted':False,'stone_memory_started':False,'reserved_test_pixels_read':False,'tip_angle_reference_available':False,'model_state_sha256_unchanged':before}
        atomic_json(OUT/'summary.json',result);progress('completed',frames_completed=2694,expected_frames=2694,video_id=None)
        print(json.dumps({'phase':'completed','video':{n:v['metrics'] for n,v in video.items()},'screening':result['screening'],'cost':{n:v['timings']['pipeline_ms'] for n,v in performance.items()}},indent=2),flush=True)
    except Exception as error:
        progress('failed',error=str(error));(OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
