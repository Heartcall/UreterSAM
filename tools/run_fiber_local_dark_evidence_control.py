"""Frozen local dark-front feasibility: shared static inference, six full-video workers."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
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
import torch

from ureter_semis.experiment import atomic_json,input_identity,runtime_identity,seed_training
from .fiber_reference_workspace import ROOT,read,sha256,load_mask
from .fiber_local_dark_evidence import ARMS,LocalDarkEvidence
from .fiber_photometric_input import infer
from .run_fiber_dark_input_control import count_forwards,process_memory,checked_writer,verify_video,select_groups,centerline_counts,centerline_pool,brightness,pool_brightness,json_lines,FIXED
from .run_fiber_temporal_state_control import construct,source_scan
from .run_two_network_experts import array_hash,model_hash
from .run_fiber_blue_aplus import frame_read,panel
from .replay_classwise_experts import model_state_hash,save_png,stats
from .run_classwise_expert_control import summarize

CONFIG=ROOT/'configs/fiber_local_dark_evidence_20261031.json'


def observation_hash(obs):
    return hashlib.sha256(json.dumps({k:v for k,v in obs.items() if k!='timings'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def profiles(obs):
    return [p for c in obs['candidates'] for p in c['profiles']]


def corridor(shape,obs):
    result=np.zeros(shape,np.uint8)
    for p in profiles(obs):
        cv2.fillConvexPoly(result,np.rint(p['corridor_corners']).astype(np.int32),1)
    return result.astype(bool)


def probe_ids(obs,shape):
    h,w=shape;all_ids=set();supported=set();conflict=set()
    for p in profiles(obs):
        xy=np.asarray(p['primary_xy']);valid=np.asarray(p['primary_valid'],bool);q=np.floor(xy+.5).astype(int)
        valid &= (q[...,0]>=0)&(q[...,0]<w)&(q[...,1]>=0)&(q[...,1]<h)
        ids=q[...,1]*w+q[...,0];all_ids.update(ids[valid].tolist())
        good=valid&np.asarray(p['supported_probes'],bool);supported.update(ids[good].tolist())
        clash=valid&np.asarray(p['protected_conflict'],bool);conflict.update(ids[clash].tolist())
    return np.asarray(sorted(all_ids),int),np.asarray(sorted(supported),int),np.asarray(sorted(conflict),int)


def evidence_metrics(target,mask,probability,frame,obs):
    all_ids,supported,conflicts=probe_ids(obs,target.shape);gt=target.ravel();raw=mask.ravel();prob=probability.ravel()
    valid=all_ids[gt[all_ids]!=255];hits=supported[gt[supported]!=255]
    truth=gt[valid]==2;support_truth=gt[hits]==2
    region=corridor(target.shape,obs);fn=(target==2)&(mask!=2);lab=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)[:,:,0]
    inside=cv2.distanceTransform((target==2).astype(np.uint8),cv2.DIST_L2,5)>3*min(target.shape)/400
    d={'probe_points':len(all_ids),'probe_valid_known':len(valid),'probe_unknown':len(all_ids)-len(valid),
        'probe_fiber':int(truth.sum()),'probe_nonfiber':int((~truth).sum()),'supported_points':len(supported),'supported_known':len(hits),
        'supported_fiber':int(support_truth.sum()),'supported_nonfiber':int((~support_truth).sum()),'supported_unknown':len(supported)-len(hits),
        'supported_stone_reference':int((gt[hits]==1).sum()),'supported_raw_stone_conflicts':len(conflicts),
        'FN_probe_points':int(((gt[valid]==2)&(raw[valid]!=2)).sum()),
        'FN_lowP_probe_points':int(((gt[valid]==2)&(raw[valid]!=2)&(prob[valid]<.1)).sum()),
        'supported_FN_points':int(((gt[hits]==2)&(raw[hits]!=2)).sum()),
        'supported_FN_lowP_points':int(((gt[hits]==2)&(raw[hits]!=2)&(prob[hits]<.1)).sum()),
        'supported_lowLab_FN_lowP_points':int(((gt[hits]==2)&(raw[hits]!=2)&(prob[hits]<.1)&(lab.ravel()[hits]<85)).sum()),
        'supported_FP_originalPge09':int(((gt[hits]!=2)&(prob[hits]>=.9)).sum()),
        'corridor_area':int(region.sum()),'corridor_FN_opportunity_NOT_recovery':int((fn&region).sum()),
        'corridor_FN_lowP_opportunity':int((fn&region&(probability<.1)).sum()),'corridor_lowLab_FN_opportunity':int((fn&region&(lab<85)).sum()),
        'corridor_interior_FN_opportunity':int((fn&region&inside).sum()),'corridor_nonfiber_known':int((region&(target!=2)&(target!=255)).sum()),
        'corridor_stone_reference':int((region&(target==1)).sum()),'corridor_unknown':int((region&(target==255)).sum()),
        'all_FN':int(fn.sum()),'FN_outside_corridor':int((fn&~region).sum()),'reference_fiber_absent':not bool((target==2).any()),
        'unambiguous_on_fiber_absent':not bool((target==2).any()) and obs['hypothesis']=='unambiguous_supported_hypothesis'}
    d['support_precision']=d['supported_fiber']/d['supported_known'] if d['supported_known'] else None
    d['support_recall_on_fixed_probe_population']=d['supported_fiber']/d['probe_fiber'] if d['probe_fiber'] else None
    return d


def pool_evidence(records,arm):
    rows=[r['evidence'][arm] for r in records];keys=[k for k,v in rows[0].items() if isinstance(v,(int,bool))]
    result={k:sum(int(r[k]) for r in rows) for k in keys}
    result['support_precision']=result['supported_fiber']/result['supported_known'] if result['supported_known'] else None
    result['support_recall_on_fixed_probe_population']=result['supported_fiber']/result['probe_fiber'] if result['probe_fiber'] else None
    result['frames']=len(rows)
    return result


def draw_observation(frame,obs,arm,index):
    image=frame.copy()
    for c in obs['candidates']:
        for p in c['profiles']:
            corners=np.rint(p['corridor_corners']).astype(np.int32)
            cv2.polylines(image,[corners],True,(0,170,230),1,cv2.LINE_AA)
            xy=np.array(p['primary_xy']);valid=np.array(p['primary_valid'],bool);supported=np.array(p['supported_probes'],bool);conflict=np.array(p['protected_conflict'],bool)
            for q,good,hit,clash in zip(xy.reshape(-1,2),valid.ravel(),supported.ravel(),conflict.ravel()):
                if not good:continue
                color=(220,0,230) if clash else (30,255,30) if hit else (150,150,150)
                cv2.circle(image,tuple(np.rint(q).astype(int)),1,color,-1)
            if 'side_edge_xy' in p:
                for q in np.array(p['side_edge_xy'])[np.array(p['bilateral_supported'],bool)].reshape(-1,2):cv2.circle(image,tuple(np.rint(q).astype(int)),1,(255,200,0),-1)
    image=panel(image,arm,None,index);strip=np.zeros((32,frame.shape[1],3),np.uint8)
    cv2.putText(strip,obs['hypothesis']+'; no mask edit',(3,19),cv2.FONT_HERSHEY_SIMPLEX,.34,(255,255,255),1,cv2.LINE_AA)
    return np.vstack([image,strip])


def raw_panel(frame,mask,index):
    return np.vstack([panel(frame,'Raw2 semantic mask (unchanged)',mask,index),np.zeros((32,frame.shape[1],3),np.uint8)])


def static(cfg,pre,out,progress):
    seed_training(42);experts=construct(pre);before=model_state_hash(experts);counts,handles=count_forwards(experts)
    observer=LocalDarkEvidence(cfg,read(ROOT/cfg['proposal']['frozen_seed_config']))
    for scope,key in [('video_native','video_manifest'),('image_native','image_manifest')]:
        rows=read(ROOT/cfg['evaluation']['data'][key])['val'];old=read(ROOT/cfg['base']['reference_run']/f'{scope}/raw2.json')
        folder=out/scope;folder.mkdir();(folder/'masks').mkdir();records=[];targets=[];predictions=[]
        for i,row in enumerate(rows):
            frame=frame_read(row['image']);mask,sp,fp,t=infer(experts,frame,'raw2');target=load_mask(row['mask'],mask.shape,{0,1,2,255})
            hashes={'mask_raw_sha256':array_hash(mask),'stone_probability_raw_sha256':array_hash(sp),'fiber_probability_raw_sha256':array_hash(fp)}
            if any(v!=old['records'][i][k] for k,v in hashes.items()):raise ValueError('Frozen static raw output differs')
            observations={a:observer.observe(frame,mask,fp,a) for a in ARMS}
            for a in ARMS:
                if observations[a]['apply_to_mask'] or array_hash(mask)!=hashes['mask_raw_sha256']:raise ValueError('Observer changed mask')
            for x,y in zip(profiles(observations['geometry_window']),profiles(observations['local_current_evidence'])):
                if x['primary_xy']!=y['primary_xy'] or x['primary_valid']!=y['primary_valid']:raise ValueError('Paired probe populations differ')
            save_png(folder/f'masks/{i:04d}.png',mask)
            records.append({'sample_id':row['sample_id'],'video_id':row.get('video_id'),'frame_index':row.get('frame_index'),**hashes,
                'observations':observations,'observation_hashes':{a:observation_hash(observations[a]) for a in ARMS},
                'evidence':{a:evidence_metrics(target,mask,fp[2],frame,observations[a]) for a in ARMS},
                'brightness':brightness(frame,target,mask,fp[2]),'semantic_centerline':centerline_counts(target,mask),'tip_metrics':None})
            targets.append(target);predictions.append(mask)
            if (i+1)%30==0:progress('static',scope=scope,completed=i+1)
        semantic=summarize(rows,targets,predictions)
        if semantic!=old['summary']:raise ValueError('Semantic metrics differ')
        groups={}
        for name,ids in select_groups(rows,scope=='video_native').items():
            rr=[records[i] for i in ids]
            groups[name]={'frames':len(ids),'summary':summarize([rows[i] for i in ids],[targets[i] for i in ids],[predictions[i] for i in ids]),
                'evidence':{a:pool_evidence(rr,a) for a in ARMS},'brightness':pool_brightness(rr),'semantic_centerline':centerline_pool(rr)}
        atomic_json(folder/'summary.json',{'actual_shared_paired_frame_inferences':len(rows),'summary':semantic,'groups':groups,
            'evidence':{a:pool_evidence(records,a) for a in ARMS},'brightness':pool_brightness(records),'semantic_centerline':centerline_pool(records),
            'records':records,'all_arm_semantic_pixels_unchanged':True,'paired_pixel_changes':{c:{k:0 for k in ['lost_tp','gained_tp','added_fp','removed_fp']} for c in ['stone','fiber']},
            'tip_metrics':None,'classwise_frozen_reference':read(ROOT/cfg['base']['historical_reference']/f'{scope}/classwise3.json')['summary']})
    if counts!=[202,202] or model_state_hash(experts)!=before:raise ValueError('Static graph/weights differ')
    for h in handles:h.remove()
    atomic_json(out/'static_model_identity.json',{'model_state_sha256':before,'individual_model_sha256':[model_hash(m) for m in experts.models],
        'actual_branch_forwards':counts,'networks':2,'parameters':experts.parameter_count,'runtime':runtime_identity(experts.device)})
    del experts;torch.cuda.empty_cache()


def static_gates(out):
    video=read(out/'video_native/summary.json');image=read(out/'image_native/summary.json');a='local_current_evidence'
    fixed=next(r for r in video['records'] if r['video_id']=='video_2' and r['frame_index']==1208)
    checks={}
    for name,data in [('video70',video),('image132',image)]:
        e=data['evidence'][a];checks[name+'_lowP_FN_support']=e['supported_FN_lowP_points']>0
        checks[name+'_support_precision_ge095']=e['support_precision'] is not None and e['support_precision']>=.95
        checks[name+'_no_unambiguous_hypothesis_on_fiber_absent']=e['unambiguous_on_fiber_absent']==0
        checks[name+'_masks_exact']=data['all_arm_semantic_pixels_unchanged']
    checks['fixed_video2_display1209_lowP_FN_support']=fixed['evidence'][a]['supported_FN_lowP_points']>0
    return {'checks':checks,'passed':all(checks.values()),'rejected_checks':[k for k,v in checks.items() if not v],
        'decision_scope':'Only whether to justify later pixel candidate DESIGN; never mask modification/adoption/tip validation.'}


def worker(cfg,out,arm,vid):
    pre=read(out/'preflight.json');info=next(v for v in pre['videos'] if v['video_id']==vid)
    for item in [info,*pre['checkpoints'].values()]:
        if sha256(item['path'])!=item['sha256']:raise ValueError('Worker input differs')
    if any(sha256(ROOT/k)!=v for k,v in pre['sources'].items()):raise ValueError('Worker source differs')
    source=read(out/f'source_{vid}.json');old=json_lines(ROOT/cfg['base']['reference_run']/f'full_video/raw2/{vid}/frames.jsonl')
    rows=read(ROOT/cfg['evaluation']['data']['video_manifest'])['val'];selected={r['frame_index']:i for i,r in enumerate(rows) if r['video_id']==vid}
    selected_static=read(out/'video_native/summary.json');seed_training(42);experts=construct(pre);before=model_state_hash(experts);counts,handles=count_forwards(experts)
    observer=LocalDarkEvidence(cfg,read(ROOT/cfg['proposal']['frozen_seed_config']));folder=out/f'full_video/{arm}/{vid}';folder.mkdir(parents=True)
    cap=cv2.VideoCapture(info['path']);ok,first=cap.read();cap.release()
    if not ok:raise ValueError('Empty video')
    def run(frame):
        experts.sync();start=time.perf_counter();mask,sp,fp,t=infer(experts,frame,'raw2');coarse=time.perf_counter()
        obs=observer.observe(frame,mask,fp,arm);end=time.perf_counter()
        t.update(obs['timings']);t.update(base_pipeline_ms=(coarse-start)*1000,pipeline_total_ms=(end-start)*1000)
        return mask,sp,fp,obs,t
    warm=cfg['execution']['warmup_frames_per_process']
    for _ in range(warm):run(first)
    counts[:]=[0,0];cap=cv2.VideoCapture(info['path']);h,w=first.shape[:2];writer=checked_writer(folder/'replay.mp4',info['fps'],(w*2,h+66))
    times=[];hypotheses=Counter();reasons=Counter();matches=0;i=0;cpu_peak=0;wall=time.perf_counter();cpu=time.process_time();torch.cuda.reset_peak_memory_stats()
    try:
        with (folder/'frames.jsonl').open('w',encoding='utf-8') as log:
            while True:
                start=time.perf_counter();ok,frame=cap.read();decoded=time.perf_counter()
                if not ok:break
                mask,sp,fp,obs,t=run(frame);inferred=time.perf_counter();src=source['frames'][i]
                hashes={'mask_raw_sha256':array_hash(mask),'stone_probability_raw_sha256':array_hash(sp),'fiber_probability_raw_sha256':array_hash(fp)}
                rgb_hash=array_hash(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if rgb_hash!=src['rgb_raw_sha256'] or any(v!=old[i][k] for k,v in hashes.items()):raise ValueError('Full source/mask/probability differs')
                if src['decoded_pts_seconds'] is not None and abs(pts-src['decoded_pts_seconds'])>1e-9:raise ValueError('PTS differs')
                obs_hash=observation_hash(obs)
                if i in selected:
                    r=selected_static['records'][selected[i]]
                    if any(v!=r[k] for k,v in hashes.items()) or obs_hash!=r['observation_hashes'][arm]:raise ValueError('Static/full observer differs')
                    matches+=1
                if obs['apply_to_mask']:raise ValueError('Mask modification requested')
                writer.write(np.hstack([raw_panel(frame,mask,i),draw_observation(frame,obs,arm,i)]));rendered=time.perf_counter()
                record={'frame_index':i,'timestamp_seconds':src['timestamp_seconds'],'source_rgb_raw_sha256':rgb_hash,**hashes,
                    'mask_reference':str(ROOT/cfg['base']['reference_run']/f'full_video/raw2/{vid}/masks/{i:06d}.png'),
                    'observation':obs,'observation_sha256':obs_hash,'approved_semantic_reference':i in selected,'tip_metrics':None,'unlabelled_accuracy':None}
                log.write(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n')
                hypotheses[obs['hypothesis']]+=1;reasons.update(c['reason'] for c in obs['candidates']);cpu_peak=max(cpu_peak,process_memory()[0])
                end=time.perf_counter();t.update(decode_ms=(decoded-start)*1000,render_audit_encode_ms=(rendered-inferred)*1000,record_export_ms=(end-rendered)*1000,cycle_ms=(end-start)*1000)
                times.append(t);i+=1
                if i%300==0:print(f'{arm} {vid} {i}/{info["frames"]}',flush=True)
    finally:cap.release();writer.release()
    wall=time.perf_counter()-wall;cpu=time.process_time()-cpu
    if i!=info['frames'] or counts!=[i,i] or matches!=len(selected) or model_state_hash(experts)!=before:raise ValueError('Coverage/graph/weights differ')
    for handle in handles:handle.remove()
    movie=verify_video(folder/'replay.mp4',i);keys=sorted(set().union(*(t.keys() for t in times)))
    timing={k:stats([t.get(k,0.) for t in times]) for k in keys}
    result={'arm':arm,'video_id':vid,'pid':os.getpid(),'frames':i,'replay':movie,'approved_static_matches':matches,'all_raw_masks_and_both_probabilities_exact':True,
        'actual_branch_forwards':counts,'warmup_branch_forwards':[warm,warm],'networks':2,'parameters':experts.parameter_count,
        'model_state_sha256':before,'individual_model_sha256':[model_hash(m) for m in experts.models],'runtime':runtime_identity(experts.device),
        'source_fps':info['fps'],'timestamp_mode':source['mode'],'timings':timing,'pipeline_p95_within_source_interval':timing['pipeline_total_ms']['p95']<=1000/info['fps'],
        'observer_p95_within_design_target':timing['observer_total_ms']['p95']<=5,'hypotheses':dict(hypotheses),'candidate_reasons':dict(reasons),
        'offline_fps_including_decode_audit_export':i/wall,'wall_seconds':wall,'cpu_process_seconds':cpu,'cpu_sampled_peak_rss_mib':cpu_peak/2**20,
        'windows_peak_working_set_mib':process_memory()[1]/2**20,'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'cuda_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'tip_metrics':None,'whole_video_accuracy':None}
    atomic_json(folder/'timings.json',times);atomic_json(folder/'summary.json',result)
    print(json.dumps({'completed':arm,'video':vid,'p95':timing['pipeline_total_ms']['p95']}),flush=True)


def comparison(cfg,out,info):
    vid=info['video_id'];cap=cv2.VideoCapture(info['path']);w,h=info['width'],info['height'];i=0
    logs={a:(out/f'full_video/{a}/{vid}/frames.jsonl').open(encoding='utf-8') for a in ARMS}
    path=out/f'{vid}_comparison.mp4';writer=checked_writer(path,info['fps'],(4*w,h+66))
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            records={a:json.loads(next(logs[a])) for a in ARMS};mask=load_mask(records['raw2']['mask_reference'],frame.shape[:2],{0,1,2})
            original=np.vstack([panel(frame,'Original source',None,i),np.zeros((32,w,3),np.uint8)])
            writer.write(np.hstack([original,raw_panel(frame,mask,i),draw_observation(frame,records['geometry_window']['observation'],'Geometry: opportunity only',i),
                draw_observation(frame,records['local_current_evidence']['observation'],'Local current evidence',i)]));i+=1
    finally:
        cap.release();writer.release()
        for f in logs.values():f.close()
    if i!=info['frames']:raise ValueError('Comparison coverage differs')
    return verify_video(path,i)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',nargs=2);args=parser.parse_args();cfg=read(CONFIG);out=ROOT/cfg['planned_output']
    if tuple(cfg['arms'])!=ARMS:raise ValueError('Frozen arms differ')
    if args.worker:
        if args.worker[0] not in ARMS or args.worker[1] not in ['video_1','video_2']:raise ValueError('Invalid worker')
        return worker(cfg,out,*args.worker)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);job={'pid':os.getpid(),'training_updates':0,'default_changed':False}
    def progress(phase,**extra):job.update(phase=phase,**extra);atomic_json(out/'job.json',job);print(phase+' '+json.dumps(extra),flush=True)
    try:
        progress('preflight');protocol=read(ROOT/'runs/protocols/fiber_local_dark_evidence_20261031_v1/identity.json')
        if any(sha256(ROOT/k)!=v for k,v in protocol['files'].items()):raise ValueError('Protocol dependency changed')
        prior=read(ROOT/protocol['frozen_prior_preflight'])
        if sha256(ROOT/protocol['frozen_prior_preflight'])!=protocol['frozen_prior_preflight_sha256']:raise ValueError('Base identity changed')
        if any(sha256(ROOT/k)!=v for k,v in prior['sources'].items()):raise ValueError('Old source changed')
        data=cfg['evaluation']['data'];vm=read(ROOT/data['video_manifest']);im=read(ROOT/data['image_manifest'])
        inputs=input_identity(ROOT/data['image_manifest'],Path(im['root']),{'train':{'labeled':[],'unlabeled':[]},'val':vm['val']+im['val']})
        if inputs!=prior['inputs'] or len(vm['val'])!=70 or len(im['val'])!=132:raise ValueError('Reference differs')
        sources={**prior['sources']}
        for p in list((ROOT/'tools').glob('*.py'))+list((ROOT/'tests').glob('*.py')):sources[p.relative_to(ROOT).as_posix()]=sha256(p)
        related={str(CONFIG):sha256(CONFIG),str(ROOT/cfg['design_document']):sha256(ROOT/cfg['design_document'])}
        for scope in ['video_native','image_native']:
            p=ROOT/cfg['base']['reference_run']/f'{scope}/raw2.json';related[str(p)]=sha256(p)
        for info in prior['videos']:
            if sha256(info['path'])!=info['sha256']:raise ValueError('Video changed')
            progress('source_timestamp_audit',video_id=info['video_id']);source_scan(out,info,ROOT/cfg['base']['historical_reference'])
            for p in [out/f'source_{info["video_id"]}.json',ROOT/cfg['base']['reference_run']/f'full_video/raw2/{info["video_id"]}/frames.jsonl']:related[str(p)]=sha256(p)
        for cp in prior['checkpoints'].values():
            if sha256(cp['path'])!=cp['sha256']:raise ValueError('Weights changed')
        pre={'config':cfg,'config_sha256':sha256(CONFIG),'sources':sources,'related':related,'inputs':inputs,'checkpoints':prior['checkpoints'],'videos':prior['videos'],
            'train_pixels_read':0,'reserved_test_pixels_read':0,'mechanism_tests_passed':15,'implementation_clarification':
            'Internal-transition metadata retains every consecutive neural decrease (no new drop cutoff) or RGB contrast>=frozen0.04 plus2later bilateral stations. It has no effect on support acceptance or masks.'}
        atomic_json(out/'preflight.json',pre)
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in sources:z.write(ROOT/name,name)
            z.write(CONFIG,CONFIG.relative_to(ROOT).as_posix());z.write(ROOT/cfg['design_document'],cfg['design_document'])
        static(cfg,pre,out,progress);gates=static_gates(out);atomic_json(out/'static_quality.json',gates);full={}
        for arm in ARMS:
            full[arm]={}
            for info in pre['videos']:
                vid=info['video_id'];progress('full_video_fresh_process',arm=arm,video_id=vid)
                with (out/f'{arm}_{vid}.log').open('w',encoding='utf-8') as log:
                    subprocess.run([sys.executable,'-m','tools.run_fiber_local_dark_evidence_control','--worker',arm,vid],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                        check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                full[arm][vid]=read(out/f'full_video/{arm}/{vid}/summary.json')
        exports={}
        for info in pre['videos']:progress('comparison_export',video_id=info['video_id']);exports[info['video_id']]=comparison(cfg,out,info)
        if any(sha256(ROOT/k)!=v for k,v in sources.items()) or any(sha256(k)!=v for k,v in related.items()):raise ValueError('Frozen sources/related inputs changed')
        if inputs!=input_identity(ROOT/data['image_manifest'],Path(im['root']),{'train':{'labeled':[],'unlabeled':[]},'val':vm['val']+im['val']}):raise ValueError('Reference changed')
        if any(sha256(x['path'])!=x['sha256'] for x in [*pre['checkpoints'].values(),*pre['videos']]):raise ValueError('Weights/video changed')
        runtime=all(v['pipeline_p95_within_source_interval'] and v['observer_p95_within_design_target'] for v in full['local_current_evidence'].values())
        atomic_json(out/'summary.json',{'static_quality':gates,'full_video':full,'comparison_videos':exports,'evidence_feasibility_passed':gates['passed'] and runtime,
            'actual_static_paired_frame_inferences':202,'actual_full_paired_frame_inferences':8082,'warmup_paired_frames':72,'full_processes':6,
            'all_masks_unchanged':True,'training_updates':0,'train_pixels_read':0,'reserved_test_pixels_read':0,'adopted':False,'default_changed':False,
            'tip_metrics':None,'whole_video_accuracy':None,'s3_started':False})
        progress('completed',static_frames=202,full_frames=8082)
    except Exception as error:
        progress('failed',error=str(error));(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
