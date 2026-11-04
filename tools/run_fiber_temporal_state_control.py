"""T0 four-arm state-only control; all visible masks remain exact raw member1."""
from __future__ import annotations
import argparse
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
from ureter_semis.experiment import atomic_json,input_identity,seed_training,runtime_identity
from ureter_semis.video_metrics import estimated_fov
from .fiber_reference_workspace import ROOT,read,sha256,load_mask
from .fiber_shape_prior import fit_shape
from .fiber_temporal_state import GeometryStateControl,valid_geometry
from .member_experts import MemberExperts
from .run_two_network_experts import infer,array_hash,model_hash
from .run_fiber_blue_aplus import panel,frame_read
from .replay_classwise_experts import model_state_hash,save_png,stats
from .run_classwise_expert_control import summarize
CONFIG=ROOT/'configs/fiber_temporal_state_gate_20261030.json'


def observation(frame,mask,fiber,legacy):
    fov=estimated_fov(frame,mask.shape)
    fit=fit_shape(frame,fiber[2],mask==2,fov,legacy['fit'])
    return fit,fov


def component_provenance(mask,fov,settings):
    """Reporting outside timed inference; labels identify components, not instances."""
    dist=cv2.distanceTransform(np.pad(fov.astype(np.uint8),1),cv2.DIST_L2,5)[1:-1,1:-1]
    n,labels,counts,_=cv2.connectedComponentsWithStats((mask==2).astype(np.uint8),8);eligible=[]
    near=dist<=settings['entry_distance_fraction']*min(mask.shape)
    for k in range(1,n):
        if counts[k,cv2.CC_STAT_AREA]>=settings['minimum_candidate_pixels'] and int(((labels==k)&near).sum())>=3:
            eligible.append((int(counts[k,cv2.CC_STAT_AREA]),k))
    chosen=max(eligible) if eligible else None
    return {'components':n-1,'eligible_entry_components':len(eligible),'chosen_label':chosen[1] if chosen else None,
        'chosen_area':chosen[0] if chosen else None,'label_is_instance_identity':False}


def state_panel(frame,arm,mask,result,index):
    out=panel(frame,arm,mask,index)
    if result is None:return out
    g=result['geometry'];color=(0,230,255) if result['prediction_only'] else (255,255,0)
    if g:
        for segment in g['side_segments']:
            a,b=[tuple(np.rint(p).astype(int)+[0,34]) for p in segment];cv2.line(out,a,b,color,1,cv2.LINE_AA)
        tip=tuple(np.rint(g['tip_xy']).astype(int)+[0,34]);cv2.circle(out,tip,4,color,1,cv2.LINE_AA)
    if result['pending_state'] is not None:
        z=result['pending_state'];p=np.array(z[:2])+z[3]*np.array([np.cos(z[2]),np.sin(z[2])])
        cv2.drawMarker(out,tuple(np.rint(p).astype(int)+[0,34]),(255,0,255),cv2.MARKER_CROSS,8,1)
    text=f"{result['status']} age={result['age']} pending={result['pending_count']}"
    cv2.rectangle(out,(0,411),(399,433),(0,0,0),-1)
    cv2.putText(out,text,(3,427),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1,cv2.LINE_AA)
    return out


def source_scan(out,info,previous):
    old=[json.loads(s) for s in (previous/f'full_video/member1_2net/{info["video_id"]}/frames.jsonl').read_text().splitlines()]
    cap=cv2.VideoCapture(info['path']);records=[]
    while True:
        ok,frame=cap.read()
        if not ok:break
        pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000;h=array_hash(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));i=len(records)
        if h!=old[i]['source_rgb_raw_sha256']:raise ValueError('Decoded source RGB changed')
        records.append({'frame_index':i,'rgb_raw_sha256':h,'decoded_pts_seconds':float(pts) if np.isfinite(pts) else None})
    cap.release()
    if len(records)!=info['frames']:raise ValueError('Source count changed')
    pts=[r['decoded_pts_seconds'] for r in records]
    reliable=all(v is not None for v in pts) and all(b>a for a,b in zip(pts,pts[1:]))
    mode='decoded_monotonic_pts' if reliable else 'frame_index_over_source_fps'
    for r in records:r['timestamp_seconds']=r['decoded_pts_seconds'] if reliable else r['frame_index']/info['fps']
    atomic_json(out/f'source_{info["video_id"]}.json',{'mode':mode,'frames':records})
    return mode


def construct(pre):
    cp=pre['checkpoints'];experts=MemberExperts(cp['old190']['path'],cp['A0']['path'],'model1')
    if experts.model_count!=2 or experts.parameter_count!=49202790:raise ValueError('Graph budget changed')
    return experts


def static_reproduction(cfg,pre,out,progress):
    previous=ROOT/cfg['base']['reference_run'];legacy=read(ROOT/'configs/fiber_geometry_prior_20261028.json');seed_training(42);experts=construct(pre);before=model_state_hash(experts)
    result={}
    for scope,key in [('video_native','video_manifest'),('image_native','image_manifest')]:
        rows=read(ROOT/cfg['data'][key])['val'];old=read(previous/f'{scope}/member1_2net.json');targets=[];predictions=[];records=[]
        (out/f'{scope}/shared_raw2').mkdir(parents=True)
        for i,row in enumerate(rows):
            frame=frame_read(row['image']);mask,sp,fp,_=infer(experts,frame)
            if array_hash(mask)!=old['records'][i]['mask_raw_sha256'] or array_hash(sp)!=old['records'][i]['stone_probability_raw_sha256'] or array_hash(fp)!=old['records'][i]['fiber_probability_raw_sha256']:raise ValueError('Fresh static native probabilities/mask differ')
            fit,fov=observation(frame,mask,fp,legacy);measured=fit['state'] if fit['accepted'] else None
            states={arm:GeometryStateControl(arm,cfg,legacy).step(measured if arm!='raw2' else None,mask.shape,0,row['sample_id']) for arm in cfg['arms']}
            if not all(not r['apply_to_mask'] for r in states.values()):raise ValueError('State applied to visible mask')
            save_png(out/f'{scope}/shared_raw2/{i:04d}.png',mask)
            target=load_mask(row['mask'],mask.shape,{0,1,2,255});targets.append(target);predictions.append(mask)
            records.append({'sample_id':row['sample_id'],'video_id':row.get('video_id'),'frame_index':row.get('frame_index'),
                'mask_raw_sha256':array_hash(mask),'stone_probability_raw_sha256':array_hash(sp),'fiber_probability_raw_sha256':array_hash(fp),
                'fit':fit,'component_provenance':component_provenance(mask,fov,legacy['fit']),'states':states,'tip_metrics':None})
            if (i+1)%30==0:progress('static_reproduction',scope=scope,frames=i+1)
        current=summarize(rows,targets,predictions)
        if current!=old['summary']:raise ValueError('Static metrics changed')
        result[scope]={'actual_raw2_fresh_inferences':len(rows),'summary':current,'groups':old['groups'],'records':records,
            'all_arm_masks':'identical shared_raw2 native outputs, no repeated static forwards per arm',
            'paired_changes_all_arms':{cls:{k:0 for k in ['lost_tp','gained_tp','added_fp','removed_fp']} for cls in ['stone','fiber']},
            'classwise_frozen_reference':read(previous/f'{scope}/classwise3.json')['summary'],'tip_metrics':None}
        atomic_json(out/f'{scope}.json',result[scope])
    if model_state_hash(experts)!=before:raise ValueError('Static weights changed')
    atomic_json(out/'static_model_identity.json',{'model_state_sha256':before,'individual_model_sha256':[model_hash(m) for m in experts.models],'runtime':runtime_identity(experts.device),'models':experts.model_count,'parameters':experts.parameter_count})
    del experts;torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def worker(cfg,out,arm,vid):
    pre=read(out/'preflight.json');legacy=read(ROOT/'configs/fiber_geometry_prior_20261028.json');previous=ROOT/cfg['base']['reference_run'];info=next(v for v in pre['videos'] if v['video_id']==vid)
    for x in [info,*pre['checkpoints'].values()]:
        if sha256(x['path'])!=x['sha256']:raise ValueError('Worker input changed')
    source=read(out/f'source_{vid}.json');old=[json.loads(s) for s in (previous/f'full_video/member1_2net/{vid}/frames.jsonl').read_text().splitlines()]
    rows=read(ROOT/cfg['data']['video_manifest'])['val'];selected={r['frame_index']:i for i,r in enumerate(rows) if r['video_id']==vid};static=read(out/'video_native.json')
    seed_training(42);experts=construct(pre);before=model_state_hash(experts);control=GeometryStateControl(arm,cfg,legacy,1/info['fps'])
    folder=out/f'full_video/{arm}/{vid}';folder.mkdir(parents=True)
    def run(frame,timestamp,source_id):
        experts.sync();start=time.perf_counter();mask,sp,fp,t=infer(experts,frame);coarse=time.perf_counter()
        fit=fov=None
        if arm!='raw2':fit,fov=observation(frame,mask,fp,legacy)
        fitted=time.perf_counter();measured=fit['state'] if fit is not None and fit['accepted'] else None
        state=control.step(measured,mask.shape,timestamp,source_id);ended=time.perf_counter()
        t.update(coarse_pipeline_ms=(coarse-start)*1000,fit_fov_ms=(fitted-coarse)*1000,state_update_ms=(ended-fitted)*1000,pipeline_total_ms=(ended-start)*1000)
        return mask,sp,fp,fit,fov,state,t
    cap=cv2.VideoCapture(info['path']);ok,first=cap.read();cap.release()
    if not ok:raise ValueError('Empty source')
    for i in range(cfg['execution']['warmup_frames']):run(first,i/info['fps'],'warmup_only')
    control.reset();cap=cv2.VideoCapture(info['path']);writer=cv2.VideoWriter(str(folder/'state_replay.mp4'),cv2.CAP_MSMF,cv2.VideoWriter_fourcc(*'H264'),info['fps'],(800,434))
    if not writer.isOpened():raise RuntimeError('Video encoder unavailable')
    records=[];times=[];i=0;matches=0;torch.cuda.reset_peak_memory_stats();wall=time.perf_counter()
    try:
        while True:
            start=time.perf_counter();ok,frame=cap.read();decoded=time.perf_counter()
            if not ok:break
            pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000;src=source['frames'][i]
            if src['decoded_pts_seconds'] is not None and abs(pts-src['decoded_pts_seconds'])>1e-9:raise ValueError('Decoded PTS changed')
            mask,sp,fp,fit,fov,state,t=run(frame,src['timestamp_seconds'],vid);inferred=time.perf_counter()
            rgb_hash=array_hash(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));mask_hash=array_hash(mask)
            if rgb_hash!=src['rgb_raw_sha256'] or mask_hash!=old[i]['mask_raw_sha256']:raise ValueError('Full raw masks/RGB differ')
            if i in selected:
                r=static['records'][selected[i]]
                if mask_hash!=r['mask_raw_sha256'] or array_hash(sp)!=r['stone_probability_raw_sha256'] or array_hash(fp)!=r['fiber_probability_raw_sha256']:raise ValueError('Selected static/native probabilities differ')
                matches+=1
            if state['apply_to_mask']:raise ValueError('Geometry changed segmentation')
            writer.write(np.hstack([panel(frame,'Input',None,i),state_panel(frame,arm,mask,state,i)]));rendered=time.perf_counter()
            provenance=component_provenance(mask,fov,legacy['fit']) if fov is not None else None
            records.append({'frame_index':i,'timestamp_seconds':src['timestamp_seconds'],'source_rgb_raw_sha256':rgb_hash,'mask_raw_sha256':mask_hash,
                'mask_reference':str(previous/f'full_video/member1_2net/{vid}/masks/{i:06d}.png'),'fit':fit,'fit_valid_geometry':bool(fit and fit['accepted'] and valid_geometry(fit['state'])),
                'component_provenance':provenance,'tracked':state,'approved_semantic_reference':i in selected,'tip_metrics':None,'unlabelled_accuracy':None})
            end=time.perf_counter();t.update(decode_ms=(decoded-start)*1000,render_encode_ms=(rendered-inferred)*1000,audit_provenance_ms=(end-rendered)*1000,cycle_ms=(end-start)*1000);times.append(t);i+=1
            if i%300==0:print(f'{arm} {vid} {i}/{info["frames"]}',flush=True)
    finally:cap.release();writer.release()
    duration=time.perf_counter()-wall
    if i!=info['frames'] or matches!=len(selected) or model_state_hash(experts)!=before:raise ValueError('Incomplete replay or weights changed')
    control.reset();check=cv2.VideoCapture(str(folder/'state_replay.mp4'));encoded=0
    while check.read()[0]:encoded+=1
    check.release()
    if encoded!=i:raise ValueError('Incomplete encoded replay')
    (folder/'frames.jsonl').write_text('\n'.join(json.dumps(r,allow_nan=False) for r in records)+'\n',encoding='utf-8')
    atomic_json(folder/'timings.json',times)
    result={'arm':arm,'video_id':vid,'pid':os.getpid(),'frames':i,'encoded_frames':encoded,'source_fps':info['fps'],'timestamp_mode':source['mode'],'networks':experts.model_count,'parameters':experts.parameter_count,
        'individual_model_sha256':[model_hash(m) for m in experts.models],'model_state_sha256':before,'runtime':runtime_identity(experts.device),'approved_static_matches':matches,
        'timings':{k:stats([t[k] for t in times]) for k in times[0]},'offline_fps_including_render_audit_encode':i/duration,'wall_seconds':duration,'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'pipeline_p95_within_source_interval':stats([t['pipeline_total_ms'] for t in times])['p95']<=1000/info['fps'],'states':dict(Counter(r['tracked']['status'] for r in records)),
        'fit_reasons':dict(Counter(r['fit']['reason'] if r['fit'] else 'not_fitted' for r in records)),
        'fit_accepted_frames':sum(r['fit_valid_geometry'] for r in records),'prediction_only_frames':sum(r['tracked']['prediction_only'] for r in records),
        'state_available_frames':sum(r['tracked']['state'] is not None for r in records),'video_sha256':sha256(folder/'state_replay.mp4'),
        'all_raw_masks_unchanged':True,'state_cleared_after_video':control.source is None and control.timestamp is None,'tip_metrics':None,'whole_video_accuracy':None}
    atomic_json(folder/'summary.json',result);print(json.dumps({'completed':arm,'video':vid,'p95':result['timings']['pipeline_total_ms']['p95']}),flush=True)


def comparison(out,cfg,info):
    vid=info['video_id'];logs={arm:[json.loads(s) for s in (out/f'full_video/{arm}/{vid}/frames.jsonl').read_text().splitlines()] for arm in cfg['arms']}
    cap=cv2.VideoCapture(info['path']);writer=cv2.VideoWriter(str(out/f'{vid}_state_comparison.mp4'),cv2.CAP_MSMF,cv2.VideoWriter_fourcc(*'H264'),info['fps'],(2000,434));i=0
    if not writer.isOpened():raise RuntimeError('Encoder unavailable')
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            mask=load_mask(logs['raw2'][i]['mask_reference'],frame.shape[:2],{0,1,2});panels=[panel(frame,'Input',None,i)]
            for arm in cfg['arms']:panels.append(state_panel(frame,arm,mask,logs[arm][i]['tracked'],i))
            writer.write(np.hstack(panels));i+=1
    finally:cap.release();writer.release()
    cap=cv2.VideoCapture(str(out/f'{vid}_state_comparison.mp4'));n=0
    while cap.read()[0]:n+=1
    cap.release()
    if n!=info['frames'] or i!=n:raise ValueError('Incomplete comparison video')
    return {'path':str(out/f'{vid}_state_comparison.mp4'),'frames':n,'sha256':sha256(out/f'{vid}_state_comparison.mp4')}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',nargs=2);args=parser.parse_args();cfg=read(CONFIG);out=ROOT/cfg['planned_output']
    if args.worker:return worker(cfg,out,*args.worker)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);job={'pid':os.getpid(),'training_updates':0,'default_changed':False}
    def progress(phase,**extra):job.update(phase=phase,**extra);atomic_json(out/'job.json',job);print(phase+' '+json.dumps(extra),flush=True)
    try:
        progress('preflight');previous=ROOT/cfg['base']['reference_run'];prior=read(previous/'preflight.json');legacy=read(ROOT/'configs/fiber_geometry_prior_20261028.json')
        identity=read(ROOT/'runs/protocols/fiber_temporal_state_gate_20261030_v1/identity.json')
        if any(sha256(ROOT/k)!=v for k,v in identity['files'].items()):raise ValueError('Prepared protocol/source changed')
        vm=read(ROOT/cfg['data']['video_manifest']);im=read(ROOT/cfg['data']['image_manifest'])
        if len(vm['val'])!=70 or len(im['val'])!=132:raise ValueError('Reference count changed')
        inputs=input_identity(ROOT/cfg['data']['image_manifest'],Path(im['root']),{'train':{'labeled':[],'unlabeled':[]},'val':vm['val']+im['val']})
        sources={**prior['sources']}
        for p in list((ROOT/'tools').glob('*.py'))+list((ROOT/'tests').glob('*.py')):sources[p.relative_to(ROOT).as_posix()]=sha256(p)
        if any(sha256(ROOT/k)!=v for k,v in prior['sources'].items()):raise ValueError('Legacy sources changed')
        related={**prior['related'],str(CONFIG):sha256(CONFIG),str(ROOT/cfg['design_document']):sha256(ROOT/cfg['design_document'])}
        for scope in ['video_native','image_native']:
            for arm in ['member1_2net','classwise3']:
                p=previous/f'{scope}/{arm}.json';related[str(p)]=sha256(p)
        for v in prior['videos']:
            if sha256(v['path'])!=v['sha256']:raise ValueError('Source video changed')
            progress('source_timestamp_audit',video_id=v['video_id']);source_scan(out,v,previous)
            source_path=out/f'source_{v["video_id"]}.json';related[str(source_path)]=sha256(source_path)
            for name in ['frames.jsonl','summary.json']:
                p=previous/f'full_video/member1_2net/{v["video_id"]}/{name}';related[str(p)]=sha256(p)
            for p in (previous/f'full_video/member1_2net/{v["video_id"]}/masks').glob('*.png'):related[str(p)]=sha256(p)
        for cp in prior['checkpoints'].values():
            if sha256(cp['path'])!=cp['sha256']:raise ValueError('Checkpoint changed')
        pre={'config':cfg,'config_sha256':sha256(CONFIG),'sources':sources,'related':related,'inputs':inputs,'checkpoints':prior['checkpoints'],'videos':prior['videos'],
            'training_updates':0,'train_pixels_read':0,'reserved_test_pixels_read':0,'unit_tests_before_run':{'new':11,'legacy_regression':3,'passed':14}}
        atomic_json(out/'preflight.json',pre)
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for path in sources:archive.write(ROOT/path,path)
            archive.write(CONFIG,CONFIG.relative_to(ROOT).as_posix());archive.write(ROOT/cfg['design_document'],cfg['design_document'])
        static=static_reproduction(cfg,pre,out,progress);full={}
        for arm in cfg['arms']:
            full[arm]={}
            for info in pre['videos']:
                vid=info['video_id'];progress('full_video_fresh_process',arm=arm,video_id=vid)
                with (out/f'{arm}_{vid}.log').open('w',encoding='utf-8') as log:subprocess.run([sys.executable,'-m','tools.run_fiber_temporal_state_control','--worker',arm,vid],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                full[arm][vid]=read(out/f'full_video/{arm}/{vid}/summary.json')
        exports={}
        for info in pre['videos']:progress('comparison_export',video_id=info['video_id']);exports[info['video_id']]=comparison(out,cfg,info)
        if any(sha256(ROOT/k)!=v for k,v in sources.items()) or any(sha256(k)!=v for k,v in related.items()):raise ValueError('Frozen source/reference changed')
        if inputs!=input_identity(ROOT/cfg['data']['image_manifest'],Path(im['root']),{'train':{'labeled':[],'unlabeled':[]},'val':vm['val']+im['val']}):raise ValueError('Reference changed')
        if any(sha256(x['path'])!=x['sha256'] for x in [*pre['checkpoints'].values(),*pre['videos']]):raise ValueError('Weights or source videos changed')
        atomic_json(out/'summary.json',{'full_video':full,'comparison_videos':exports,'static_fresh_forwards':202,'full_forward_frames':10776,'state_only':True,'all_masks_unchanged':True,
            'default_changed':False,'adopted':False,'training_updates':0,'train_pixels_read':0,'reserved_test_pixels_read':0,'tip_metrics':None,'whole_video_accuracy':None,'t1_search_started':False,'s3_started':False})
        progress('completed',full_frames=10776)
    except Exception as e:
        progress('failed',error=str(e));(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':main()
