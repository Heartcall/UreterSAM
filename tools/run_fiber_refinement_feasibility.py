"""Frozen fiber-only coverage/oracle diagnosis and complete-video compute proxies."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from ureter_semis.experiment import atomic_json,input_identity,seed_training,runtime_identity
from .fiber_reference_workspace import ROOT,read,sha256,load_mask
from .fiber_refinement_feasibility import fiber_rois,stone_protected_upper_bound,region_counts
from .refinement_roi import roi_union
from .member_experts import MemberExperts
from .run_two_network_experts import infer,array_hash
from .run_fiber_blue_aplus import panel
from .replay_classwise_experts import pil_input,model_state_hash,stats
from .run_classwise_expert_control import summarize
CONFIG=ROOT/'configs/fiber_refinement_feasibility_20261030.json'

def aggregate(records):
    result={'frames':len(records),'roi_count_mean':float(np.mean([r['roi_count'] for r in records])),
        'roi_count_p95':float(np.percentile([r['roi_count'] for r in records],95)),
        'frames_without_roi':sum(r['roi_count']==0 for r in records),
        'roi_union_fraction':sum(r['roi_area_union'] for r in records)/sum(r['image_pixels'] for r in records),'regions':{}}
    for name in records[0]['regions']:
        values={k:sum(r['regions'][name][k] for r in records) for k in ['total','inside','outside']}
        values['coverage']=values['inside']/values['total'] if values['total'] else None
        present=[r['regions'][name] for r in records if r['regions'][name]['total']]
        values['present_frames']=len(present)
        values['minimum_frame_coverage']=min(v['inside']/v['total'] for v in present) if present else None
        values['zero_coverage_frames']=sum(v['inside']==0 for v in present)
        result['regions'][name]=values
    return result

def checks(totals,gates):
    r=totals['regions']
    return {'pooled_target_coverage':r['target']['coverage'] is not None and r['target']['coverage']>=gates['pooled_target_coverage_min'],
        'minimum_present_frame_target_coverage':r['target']['minimum_frame_coverage'] is not None and r['target']['minimum_frame_coverage']>=gates['minimum_present_frame_target_coverage_min'],
        'pooled_fn_coverage':r['fn']['total']==0 or r['fn']['coverage']>=gates['pooled_fn_coverage_min'],
        'pooled_fp_coverage':r['fp']['total']==0 or r['fp']['coverage']>=gates['pooled_fp_coverage_min'],
        'roi_count':totals['roi_count_p95']<=gates['roi_count_p95_max']}

def static_diagnosis(cfg,out,progress):
    previous=ROOT/cfg['reference_run'];results={}
    for scope,key in [('video_native','video_manifest'),('image_native','image_manifest')]:
        rows=read(ROOT/cfg[key])['val'];m1=read(previous/f'{scope}/member1_2net.json');cw=read(previous/f'{scope}/classwise3.json')
        targets=[];preds=[];refs=[];records={p:[] for p in cfg['profiles']};oracle={p:[] for p in cfg['profiles']};errors=[]
        pools={label:defaultdict(list) for label in ['fn','fp','lost_tp_vs_classwise','added_fp_vs_classwise']}
        for i,row in enumerate(rows):
            coarse=np.array(Image.open(previous/f'{scope}/member1_2net/{i:04d}.png'));reference=np.array(Image.open(previous/f'{scope}/classwise3/{i:04d}.png'))
            with np.load(m1['records'][i]['probability_reference']) as cache:p=cache['a0_member1_native'][2].copy()
            if array_hash(coarse)!=m1['records'][i]['mask_raw_sha256']:raise ValueError('Frozen coarse output differs')
            # Both bounded and uncapped diagnostic proposals are created before target is read.
            boxes={n:fiber_rois(coarse,p,cfg['proposal'],n,cap=2) for n in cfg['profiles']}
            uncapped={n:fiber_rois(coarse,p,cfg['proposal'],n,cap=None) for n in cfg['profiles']}
            target=load_mask(row['mask'],coarse.shape,{0,1,2,255});valid=target!=255;truth=target==2
            distance=cv2.distanceTransform((coarse!=2).astype(np.uint8),cv2.DIST_L2,5) if (coarse==2).any() else np.full(coarse.shape,np.inf)
            regions={'fn':truth&(coarse!=2),'fp':valid&~truth&(coarse==2),'lost_tp_vs_classwise':truth&(reference==2)&(coarse!=2),'added_fp_vs_classwise':valid&~truth&(reference!=2)&(coarse==2)}
            err={'sample_id':row['sample_id'],'video_id':row.get('video_id'),'frame_index':row.get('frame_index'),'regions':{}}
            for name,area in regions.items():
                probs=p[area];dist=distance[area]
                ph=np.histogram(probs,bins=[0,.1,.2,.5,.9,1.000001])[0].tolist()
                dh=[int((dist<=3).sum()),int(((dist>3)&(dist<=16)).sum()),int(((dist>16)&(dist<=48)).sum()),int((dist>48).sum())]
                by_gt={str(c):int((area&(target==c)).sum()) for c in [0,1,2]}
                err['regions'][name]={'pixels':int(area.sum()),'probability_bins_0_01_02_05_09_1':ph,'distance_bins_le3_le16_le48_gt48':dh,'by_target_class':by_gt}
                pools[name]['probability'].append(probs);pools[name]['distance'].append(dist)
            errors.append(err)
            for name in cfg['profiles']:
                counts=region_counts(target,coarse,reference,boxes[name]);counts.update(sample_id=row['sample_id'],video_id=row.get('video_id'),frame_index=row.get('frame_index'),boxes=boxes[name],
                    uncapped_boxes=uncapped[name],uncapped_counts=region_counts(target,coarse,reference,uncapped[name]))
                records[name].append(counts);oracle[name].append(stone_protected_upper_bound(coarse,target,boxes[name]))
            targets.append(target);preds.append(coarse);refs.append(reference)
            if (i+1)%30==0:progress('static_diagnosis',scope=scope,frames=i+1)
        if summarize(rows,targets,preds)!=m1['summary'] or summarize(rows,targets,refs)!=cw['summary']:raise ValueError('Frozen summary differs')
        indices={}
        if scope=='video_native':
            for v in ['video_1','video_2']:indices[v]=[i for i,r in enumerate(rows) if r['video_id']==v]
            for v in ['original29','new_user_approved']:indices[v]=[i for i,r in enumerate(rows) if r['evaluation_cohort']==v]
        else:
            for v in sorted({r['source_dataset'] for r in rows}):indices[v]=[i for i,r in enumerate(rows) if r['source_dataset']==v]
            exposed=set(read(ROOT/'runs/a1_classwise_experts_20261028/summary.json')['image']['historical_exposure_cohorts']['historical_old190_exposed']['sample_ids'])
            indices['old190_exposed25']=[i for i,r in enumerate(rows) if r['sample_id'] in exposed];indices['not_matched107']=[i for i,r in enumerate(rows) if r['sample_id'] not in exposed]
        summary={}
        for name in cfg['profiles']:
            a=aggregate(records[name]);g=checks(a,cfg['feasibility_gates']);o=summarize(rows,targets,oracle[name]);groups={}
            for label,ids in indices.items():
                groups[label]={'coverage':aggregate([records[name][i] for i in ids]),'diagnostic_oracle_not_model_metrics':summarize([rows[i] for i in ids],[targets[i] for i in ids],[oracle[name][i] for i in ids])['metrics']}
            summary[name]={'aggregate':a,'coverage_checks':g,'coverage_feasible':all(g.values()),'diagnostic_oracle_not_model':o,'groups':groups,
                'uncapped_capacity_diagnostic':aggregate([r['uncapped_counts'] for r in records[name]]),'frames':records[name]}
        pooled={}
        for name,fields in pools.items():
            pr=np.concatenate(fields['probability']);di=np.concatenate(fields['distance'])
            pooled[name]={'pixels':len(pr),'probability_bins_0_01_02_05_09_1':np.histogram(pr,bins=[0,.1,.2,.5,.9,1.000001])[0].tolist(),
                'probability_mean':float(pr.mean()) if len(pr) else None,'distance_bins_le3_le16_le48_gt48':[int((di<=3).sum()),int(((di>3)&(di<=16)).sum()),int(((di>16)&(di<=48)).sum()),int((di>48).sum())]}
        results[scope]={'profiles':summary,'errors':{'summary':pooled,'frames':errors},'coarse_actual_model':m1['summary'],'classwise_reference_actual_model':cw['summary'],'tip_metrics':None}
        atomic_json(out/f'{scope}.json',results[scope])
    return results

def draw_boxes(frame,mask,boxes,name,index):
    image=panel(frame,name,mask,index)
    for x0,y0,x1,y1 in boxes:cv2.rectangle(image,(x0,y0+34),(x1-1,y1-1+34),(0,255,255),1)
    return image

@torch.inference_mode()
def worker(cfg,out,name,vid):
    pre=read(out/'preflight.json');info=next(v for v in pre['videos'] if v['video_id']==vid);previous=ROOT/cfg['reference_run']
    seed_training(42);cp=pre['checkpoints'];experts=MemberExperts(cp['old190']['path'],cp['A0']['path'],'model1')
    if experts.model_count!=2 or experts.parameter_count!=49202790:raise ValueError('Incorrect coarse graph')
    before=model_state_hash(experts)
    folder=out/f'full_video/{name}/{vid}';folder.mkdir(parents=True)
    old=[json.loads(s) for s in (previous/f'full_video/member1_2net/{vid}/frames.jsonl').read_text().splitlines()]
    def run(frame):
        experts.sync();start=time.perf_counter();mask,sp,fp,t=infer(experts,frame);coarse_end=time.perf_counter()
        boxes=[] if name=='member1_raw' else fiber_rois(mask,fp[2],cfg['proposal'],name,cap=2)
        proposed=time.perf_counter()
        for x0,y0,x1,y1 in boxes:
            crop=frame[y0:y1,x0:x1];tensor=pil_input(crop,256,experts.device)
            probability=experts.fiber_model(tensor).softmax(1)
            native=F.interpolate(probability.float(),size=crop.shape[:2],mode='bilinear',align_corners=False)[0].cpu().numpy()
            if not np.isfinite(native).all():raise ValueError('Invalid compute proxy output')
        experts.sync();end=time.perf_counter()
        t.update(coarse_pipeline_ms=(coarse_end-start)*1000,proposal_ms=(proposed-coarse_end)*1000,crop_proxy_ms=(end-proposed)*1000,proxy_pipeline_ms=(end-start)*1000)
        return mask,boxes,t
    cap=cv2.VideoCapture(info['path']);ok,first=cap.read();cap.release()
    if not ok:raise ValueError('Empty video')
    for _ in range(12):run(first)
    cap=cv2.VideoCapture(info['path']);writer=cv2.VideoWriter(str(folder/'diagnostic.mp4'),cv2.CAP_MSMF,cv2.VideoWriter_fourcc(*'H264'),info['fps'],(800,434))
    if not writer.isOpened():raise RuntimeError('Encoder unavailable')
    timings=[];records=[];i=0;torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    try:
        while True:
            cycle=time.perf_counter();ok,frame=cap.read();decoded=time.perf_counter()
            if not ok:break
            mask,boxes,t=run(frame);inferred=time.perf_counter()
            rgb_hash=array_hash(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));mask_hash=array_hash(mask)
            if old[i]['source_rgb_raw_sha256']!=rgb_hash or old[i]['mask_raw_sha256']!=mask_hash:raise ValueError('Full coarse baseline changed')
            writer.write(np.hstack([panel(frame,'Input',None,i),draw_boxes(frame,mask,boxes,name+' compute only',i)]))
            record={'frame_index':i,'source_rgb_raw_sha256':rgb_hash,'coarse_mask_raw_sha256':mask_hash,'boxes':boxes,'crop_proxy_calls':len(boxes),'refined_mask':None,'segmentation_accuracy':None,'tip_metrics':None}
            records.append(record);end=time.perf_counter();t.update(decode_ms=(decoded-cycle)*1000,render_audit_ms=(end-inferred)*1000,cycle_ms=(end-cycle)*1000);timings.append(t);i+=1
            if i%300==0:print(f'{name} {vid} {i}/{info["frames"]}',flush=True)
    finally:cap.release();writer.release()
    wall=time.perf_counter()-start
    if i!=info['frames'] or model_state_hash(experts)!=before:raise ValueError('Incomplete video or changed parameters')
    cap=cv2.VideoCapture(str(folder/'diagnostic.mp4'));decoded=0
    while cap.read()[0]:decoded+=1
    cap.release()
    if decoded!=i:raise ValueError('Incomplete encoded video')
    (folder/'frames.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n',encoding='utf-8')
    atomic_json(folder/'timings.json',timings)
    result={'arm':name,'video_id':vid,'pid':os.getpid(),'runtime':runtime_identity(experts.device),'frames':i,'decoded_frames':decoded,'model_count':experts.model_count,'parameters':experts.parameter_count,
        'model_state_sha256':before,'timings':{k:stats([t[k] for t in timings]) for k in timings[0]},'roi_count':stats([r['crop_proxy_calls'] for r in records]),
        'total_crop_proxy_calls':sum(r['crop_proxy_calls'] for r in records),'source_fps':info['fps'],'proxy_within_source_frame_budget_p95':stats([t['proxy_pipeline_ms'] for t in timings])['p95']<=1000/info['fps'],
        'offline_diagnostic_fps':i/wall,'wall_seconds':wall,'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'video_sha256':sha256(folder/'diagnostic.mp4'),
        'all_coarse_masks_match_previous':True,'crop_accuracy_available':False,'trained_refiner':False,'tip_metrics':None,'whole_video_accuracy':None}
    atomic_json(folder/'summary.json',result);print(json.dumps({'completed':name,'video':vid,'p95':result['timings']['proxy_pipeline_ms']['p95']}),flush=True)

def comparison(cfg,out,info):
    vid=info['video_id'];previous=ROOT/cfg['reference_run'];logs={n:[json.loads(s) for s in (out/f'full_video/{n}/{vid}/frames.jsonl').read_text().splitlines()] for n in cfg['profiles']}
    cap=cv2.VideoCapture(info['path']);writer=cv2.VideoWriter(str(out/f'{vid}_roi_comparison.mp4'),cv2.CAP_MSMF,cv2.VideoWriter_fourcc(*'H264'),info['fps'],(1600,434));i=0
    if not writer.isOpened():raise RuntimeError('Encoder unavailable')
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            mask=load_mask(previous/f'full_video/member1_2net/{vid}/masks/{i:06d}.png',frame.shape[:2],{0,1,2})
            panels=[panel(frame,'Input',None,i),panel(frame,'coarse member1 (unchanged)',mask,i)]
            for name in cfg['profiles']:panels.append(draw_boxes(frame,mask,logs[name][i]['boxes'],name+' ROI only - NOT refined',i))
            writer.write(np.hstack(panels));i+=1
    finally:cap.release();writer.release()
    cap=cv2.VideoCapture(str(out/f'{vid}_roi_comparison.mp4'));n=0
    while cap.read()[0]:n+=1
    cap.release()
    if n!=i or n!=info['frames']:raise ValueError('Incomplete comparison')
    return {'path':str(out/f'{vid}_roi_comparison.mp4'),'frames':n,'sha256':sha256(out/f'{vid}_roi_comparison.mp4')}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',nargs=2);args=parser.parse_args();cfg=read(CONFIG);out=ROOT/cfg['output']
    if args.worker:return worker(cfg,out,*args.worker)
    if out.exists():raise FileExistsError(out)
    out.mkdir();job={'pid':os.getpid(),'training_updates':0,'default_changed':False}
    def progress(phase,**extra):job.update(phase=phase,**extra);atomic_json(out/'job.json',job);print(phase+' '+json.dumps(extra),flush=True)
    try:
        progress('preflight');previous=ROOT/cfg['reference_run'];prior=read(previous/'preflight.json');vm=read(ROOT/cfg['video_manifest']);im=read(ROOT/cfg['image_manifest'])
        if len(vm['val'])!=70 or len(im['val'])!=132:raise ValueError('Frozen reference scope changed')
        sources={**prior['sources']}
        for p in [Path(__file__),ROOT/'tools/fiber_refinement_feasibility.py',ROOT/'tests/test_fiber_refinement_feasibility.py']:sources[p.relative_to(ROOT).as_posix()]=sha256(p)
        if any(sha256(ROOT/p)!=v for p,v in sources.items()):raise ValueError('Existing source changed')
        inputs=input_identity(ROOT/cfg['image_manifest'],Path(im['root']),{'train':{'labeled':[],'unlabeled':[]},'val':vm['val']+im['val']})
        related={str(CONFIG):sha256(CONFIG),str(previous/'summary.json'):sha256(previous/'summary.json'),str(previous/'preflight.json'):sha256(previous/'preflight.json'),str(ROOT/cfg['video_manifest']):sha256(ROOT/cfg['video_manifest'])}
        related.update(prior['related'])
        for scope in ['video_native','image_native']:
            for name in ['member1_2net','classwise3']:
                p=previous/f'{scope}/{name}.json';related[str(p)]=sha256(p)
                for i in range(70 if scope=='video_native' else 132):
                    p=previous/f'{scope}/{name}/{i:04d}.png';related[str(p)]=sha256(p)
        for vid in ['video_1','video_2']:
            for file in ['frames.jsonl','summary.json']:
                p=previous/f'full_video/member1_2net/{vid}/{file}';related[str(p)]=sha256(p)
        pre={'config':cfg,'config_sha256':sha256(CONFIG),'sources':sources,'related':related,'inputs':inputs,'checkpoints':prior['checkpoints'],'videos':prior['videos'],'train_pixels_read':0,'reserved_test_pixels_read':0,'training_updates':0}
        atomic_json(out/'preflight.json',pre)
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for path in sources:archive.write(ROOT/path,path)
            archive.write(CONFIG,CONFIG.relative_to(ROOT).as_posix())
        static=static_diagnosis(cfg,out,progress);full={}
        for name in cfg['full_video']['arms']:
            full[name]={}
            for info in pre['videos']:
                vid=info['video_id'];progress('full_video_compute_proxy',arm=name,video_id=vid)
                with (out/f'{name}_{vid}.log').open('w',encoding='utf-8') as stream:subprocess.run([sys.executable,'-m','tools.run_fiber_refinement_feasibility','--worker',name,vid],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                full[name][vid]=read(out/f'full_video/{name}/{vid}/summary.json')
        exports={}
        for info in pre['videos']:progress('comparison_export',video_id=info['video_id']);exports[info['video_id']]=comparison(cfg,out,info)
        feasible={name:all(static[scope]['profiles'][name]['coverage_feasible'] for scope in static) and all(full[name][v]['proxy_within_source_frame_budget_p95'] for v in full[name]) for name in cfg['profiles']}
        if any(sha256(ROOT/p)!=v for p,v in sources.items()) or any(sha256(p)!=v for p,v in related.items()):raise ValueError('Frozen source/reference changed')
        if input_identity(ROOT/cfg['image_manifest'],Path(im['root']),{'train':{'labeled':[],'unlabeled':[]},'val':vm['val']+im['val']})!=inputs:raise ValueError('Labels changed')
        if any(sha256(x['path'])!=x['sha256'] for x in [*pre['checkpoints'].values(),*pre['videos']]):raise ValueError('Weights/video changed')
        result={'role':'feasibility and compute proxy, NOT new segmentation models','coverage_feasible':{scope:{n:d['coverage_feasible'] for n,d in v['profiles'].items()} for scope,v in static.items()},
            'coverage_and_proxy_cost_feasible':feasible,'full_video':full,'comparison_videos':exports,'full_frames_per_arm':2694,'full_frames_all_arms':8082,
            'default_changed':False,'adopted':False,'training_updates':0,'train_pixels_read':0,'reserved_test_pixels_read':0,'tip_metrics':None,'whole_video_accuracy':None,'trained_refiner':False,'s3_started':False}
        atomic_json(out/'summary.json',result);progress('completed',full_frames=8082)
    except Exception as e:progress('failed',error=str(e));raise

if __name__=='__main__':main()
