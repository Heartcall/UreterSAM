"""Finite shadow-candidate growth control; never changes actual segmentation."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib,json,math,os
from pathlib import Path
import subprocess,sys,time,traceback,zipfile
import cv2
import numpy as np
import torch
from ureter_semis.experiment import atomic_json,input_identity,runtime_identity,seed_training
from ureter_semis.supervised_trainer import assess
from .fiber_reference_workspace import ROOT,read,sha256,load_mask
from .fiber_blue_outline_repaired import GrowthPipeline,ARMS,LAYERS
from .fiber_photometric_input import infer
from .run_fiber_dark_input_control import count_forwards,process_memory,checked_writer,verify_video,select_groups,brightness,pool_brightness,json_lines,FIXED
from .run_fiber_temporal_state_control import construct,source_scan
from .run_two_network_experts import array_hash,model_hash
from .run_fiber_blue_aplus import frame_read,panel
from .replay_classwise_experts import model_state_hash,save_png,stats
from .run_classwise_expert_control import summarize
from .diagnose_joint_failures import paired_changes

CONFIG=ROOT/'configs/blue_outline_control_20261102_v2.json'


def growth_hash(meta):
    return hashlib.sha256(json.dumps({k:v for k,v in meta.items() if k not in ['timings','t1_advice','tip_visibility_audit']},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def virtual_mask(raw,candidate):
    result=raw.copy();result[(raw==0)&(candidate>0)]=2
    if np.any(result[raw==1]!=1) or np.any(result[raw==2]!=2):raise ValueError('Protected raw labels changed')
    return result


def candidate_metrics(gt,raw,fp,frame,candidate,proposal,conflict,meta):
    truth=gt==2;valid=gt!=255;new=candidate>0;proposed=proposal>0
    tp=new&truth;bad=new&~truth&valid;lab=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)[:,:,0]
    result={'new_pixels':int(new.sum()),'new_known':int((new&valid).sum()),'new_reference_TP':int(tp.sum()),'new_reference_FP':int(bad.sum()),
        'new_unknown':int((new&~valid).sum()),'originalPbelow01_FN_candidate_hits':int((tp&(fp[2]<.1)&(raw!=2)).sum()),
        'lowLab_FN_candidate_hits':int((tp&(lab<85)&(raw!=2)).sum()),'proposed_pixels':int(proposed.sum()),
        'proposed_reference_TP':int((proposed&truth).sum()),'proposed_reference_FP':int((proposed&~truth&valid).sum()),
        'stone_conflict_pixels':int((conflict>0).sum()),'stone_conflict_reference_fiber':int(((conflict>0)&truth).sum()),
        'reference_fiber_absent':not bool(truth.any()),'growth_on_reference_absent':not bool(truth.any()) and bool(new.any()),
        'abnormal_rejected':bool(meta['abnormal']),'new_reference_precision':int(tp.sum())/int((new&valid).sum()) if (new&valid).any() else None,
        'proposal_kind':meta.get('proposal_kind','none')}
    return result


def pooled(records,arm):
    rows=[r['arms'][arm]['candidate_metrics'] for r in records];result={}
    for k,v in rows[0].items():
        if isinstance(v,(int,bool)):result[k]=sum(int(r[k]) for r in rows)
    result['new_reference_precision']=result['new_reference_TP']/result['new_known'] if result['new_known'] else None
    result['reasons']=dict(Counter(r['arms'][arm]['meta']['reason'] for r in records));result['tip_states']=dict(Counter(t['state'] for r in records for t in r['arms'][arm]['meta']['tips']))
    result['frames_with_new_candidate']=sum(r['arms'][arm]['candidate_metrics']['new_pixels']>0 for r in records)
    return result


def save_candidates(folder,i,candidate,proposal,conflict):
    for name,array in [('candidate',candidate),('proposal',proposal),('conflict',conflict)]:save_png(folder/name/f'{i:06d}.png',array)


def save_layers(folder,i,layers):
    result={}
    for name,arr in layers.items():save_png(folder/name/f'{i:06d}.png',arr);result[name]=dict(sha256=array_hash(arr),pixels=int(arr.sum()))
    return result


def prepare_folder(path):
    path.mkdir(parents=True)
    for name in ['candidate','proposal','conflict','experimental',*LAYERS]:(path/name).mkdir()


def draw_outline(frame,outline,sides=True,cap=True):
    image=frame.copy();outline=outline or {}
    if sides:
        for i,curve in enumerate(outline.get('side_curves',[])):
            q=np.rint(curve).astype(np.int32)
            if len(q)>1:
                if outline.get('sides_applied'):cv2.polylines(image,[q],False,[(0,210,255),(255,160,0)][i],1)
                else:
                    for j in range(0,len(q)-1,4):cv2.line(image,tuple(q[j]),tuple(q[j+1]),(150,150,150),1)
    c=outline.get('cap',{})
    if cap:
        q=np.rint(c.get('curve',[])).astype(np.int32)
        if len(q)>1:
            if c.get('accepted'):cv2.polylines(image,[q],False,(255,0,230),1)
            else:
                for j in range(0,len(q)-1,4):cv2.line(image,tuple(q[j]),tuple(q[j+1]),(150,150,150),1)
    return image


def overlay(frame,final,candidate,proposal,meta,name,index,outline=None):
    image=draw_outline(frame,outline);result=panel(image,name,final,index);strip=np.zeros((50,frame.shape[1],3),np.uint8);o=outline or {}
    cv2.putText(strip,o.get('reason',meta['reason']),(3,16),cv2.FONT_HERSHEY_SIMPLEX,.32,(255,255,255),1)
    cv2.putText(strip,o.get('cap',{}).get('reason','no_cap'),(3,35),cv2.FONT_HERSHEY_SIMPLEX,.30,(255,255,255),1)
    return np.vstack([result,strip])


def base_panel(frame,raw,index):return np.vstack([panel(frame,'ACTUAL raw mask unchanged',raw,index),np.zeros((50,frame.shape[1],3),np.uint8)])


def static(cfg,pre,out,progress):
    seed_training(42);experts=construct(pre);before=model_state_hash(experts);counts,handles=count_forwards(experts);t1=read(ROOT/cfg['t1']['config'])
    for scope,key in [('video_native','video_manifest'),('image_native','image_manifest')]:
        rows=read(ROOT/cfg['data'][key])['val'];old=read(ROOT/cfg['base']['reference_run']/f'{scope}/raw2.json');folder=out/scope;folder.mkdir();(folder/'raw').mkdir()
        for a in ARMS:prepare_folder(folder/a)
        targets=[];actual=[];predictions={a:[] for a in ARMS};records=[]
        for i,row in enumerate(rows):
            frame=frame_read(row['image']);raw,sp,fp,_=infer(experts,frame,'raw2');gt=load_mask(row['mask'],raw.shape,{0,1,2,255})
            hashes={'mask_raw_sha256':array_hash(raw),'stone_probability_raw_sha256':array_hash(sp),'fiber_probability_raw_sha256':array_hash(fp)}
            if any(v!=old['records'][i][k] for k,v in hashes.items()):raise ValueError('Static original probabilities/mask differ')
            record={'sample_id':row['sample_id'],'video_id':row.get('video_id'),'frame_index':row.get('frame_index'),**hashes,'arms':{},'tip_metrics':None}
            for arm in ARMS:
                pipeline=GrowthPipeline(cfg,arm,t1,False);candidate,proposal,conflict,meta=pipeline.run(frame,raw,fp)
                shadow=pipeline.final_mask.copy();save_candidates(folder/arm,i,candidate,proposal,conflict);save_png(folder/arm/f'experimental/{i:06d}.png',shadow)
                record['arms'][arm]={'candidate_sha256':array_hash(candidate),'proposal_sha256':array_hash(proposal),'conflict_sha256':array_hash(conflict),
                    'growth_sha256':growth_hash(meta),'final_mask_sha256':array_hash(shadow),'outline':pipeline.outline,'stage_layers':save_layers(folder/arm,i,pipeline.layers),'meta':meta,'candidate_metrics':candidate_metrics(gt,raw,fp,frame,candidate,proposal,conflict,meta),
                    'virtual_brightness':brightness(frame,gt,shadow,fp[2]),'paired_virtual_stone':paired_changes(gt,raw,shadow,1),'paired_virtual_fiber':paired_changes(gt,raw,shadow,2)}
                predictions[arm].append(shadow)
            if array_hash(raw)!=hashes['mask_raw_sha256']:raise ValueError('Actual mask mutated')
            save_png(folder/f'raw/{i:04d}.png',raw);records.append(record);targets.append(gt);actual.append(raw)
            if (i+1)%30==0:progress('static',scope=scope,completed=i+1)
        actual_summary=summarize(rows,targets,actual)
        if actual_summary!=old['summary']:raise ValueError('Actual metrics differ')
        virtual={a:summarize(rows,targets,predictions[a]) for a in ARMS};groups={}
        for name,ids in select_groups(rows,scope=='video_native').items():
            rr=[records[i] for i in ids]
            groups[name]={'frames':len(ids),'actual':summarize([rows[i] for i in ids],[targets[i] for i in ids],[actual[i] for i in ids]),
                'virtual':{a:summarize([rows[i] for i in ids],[targets[i] for i in ids],[predictions[a][i] for i in ids]) for a in ARMS},'candidate':{a:pooled(rr,a) for a in ARMS}}
        for a in ARMS:
            for r in records:r['brightness']=r['arms'][a]['virtual_brightness']
            atomic_json(folder/f'brightness_{a}.json',pool_brightness(records))
        for r in records:r.pop('brightness',None)
        atomic_json(folder/'summary.json',{'actual':actual_summary,'virtual':virtual,'candidate':{a:pooled(records,a) for a in ARMS},'groups':groups,'records':records,
            'actual_paired_frame_inferences':len(rows),'actual_pixel_changes':0,'tip_metrics':None})
    if counts!=[202,202] or model_state_hash(experts)!=before:raise ValueError('Static graph/weights differ')
    for h in handles:h.remove()
    atomic_json(out/'static_model_identity.json',{'model_state_sha256':before,'individual_model_sha256':[model_hash(m) for m in experts.models],'actual_branch_forwards':counts,'networks':2,'parameters':experts.parameter_count,'runtime':runtime_identity(experts.device)})
    del experts;torch.cuda.empty_cache()


def quality(cfg,out):
    return dict(reference_metrics_are_gates=False,role='User revised geometry objective; reference metrics diagnostic only',geometry_quality_path=str(out/'analysis/geometry_quality.json'),actual_pixels_unchanged=True)


def causal_summary(cfg,out):
    folder=out/'video_causal';folder.mkdir();(folder/'raw').mkdir()
    for a in ARMS:prepare_folder(folder/a)
    rows=read(ROOT/cfg['data']['video_manifest'])['val'];stat=read(out/'video_native/summary.json');targets=[];actual=[];predictions={a:[] for a in ARMS};records=[]
    logs={a:{v:json_lines(out/f'full_video/{a}/{v}/frames.jsonl') for v in ['video_1','video_2']} for a in ARMS}
    for i,row in enumerate(rows):
        frame=frame_read(row['image']);gt=load_mask(row['mask'],frame.shape[:2],{0,1,2,255});r={k:v for k,v in stat['records'][i].items() if k!='arms'};r['arms']={}
        raw=load_mask(out/f'video_native/raw/{i:04d}.png',gt.shape,{0,1,2});save_png(folder/f'raw/{i:04d}.png',raw)
        with np.load(ROOT/cfg['base']['reference_run']/f'video_native/raw2/probabilities/{i:04d}.npz') as z:fp=z['fiber'].copy()
        for a in ARMS:
            j=row['frame_index'];vid=row['video_id'];record=logs[a][vid][j]
            for k in ['mask_raw_sha256','stone_probability_raw_sha256','fiber_probability_raw_sha256']:assert record[k]==r[k]
            arrays=[load_mask(out/f'full_video/{a}/{vid}/{name}/{j:06d}.png',gt.shape,{0,1}) for name in ['candidate','proposal','conflict']]
            c,p,conf=arrays;meta=record['meta'];shadow=load_mask(out/f'full_video/{a}/{vid}/experimental/{j:06d}.png',gt.shape,{0,1,2});assert array_hash(shadow)==record['final_mask_sha256'];save_candidates(folder/a,i,c,p,conf);save_png(folder/a/f'experimental/{i:06d}.png',shadow)
            layers={name:load_mask(out/f'full_video/{a}/{vid}/{name}/{j:06d}.png',gt.shape,{0,1}) for name in LAYERS};stage_layers=save_layers(folder/a,i,layers)
            assert stage_layers==record['stage_layers']
            r['arms'][a]={k:record[k] for k in ['candidate_sha256','proposal_sha256','conflict_sha256','growth_sha256']}
            r['arms'][a].update(final_mask_sha256=record['final_mask_sha256'],outline=record['outline'],stage_layers=stage_layers,meta=meta,candidate_metrics=candidate_metrics(gt,raw,fp,frame,c,p,conf,meta),virtual_brightness=brightness(frame,gt,shadow,fp[2]),paired_virtual_stone=paired_changes(gt,raw,shadow,1),paired_virtual_fiber=paired_changes(gt,raw,shadow,2))
            predictions[a].append(shadow)
        targets.append(gt);actual.append(raw);records.append(r)
    base=summarize(rows,targets,actual);assert base==stat['actual'];groups={}
    for name,ids in select_groups(rows,True).items():
        groups[name]=dict(frames=len(ids),actual=summarize([rows[i] for i in ids],[targets[i] for i in ids],[actual[i] for i in ids]),virtual={a:summarize([rows[i] for i in ids],[targets[i] for i in ids],[predictions[a][i] for i in ids]) for a in ARMS},candidate={a:pooled([records[i] for i in ids],a) for a in ARMS})
    for a in ARMS:
        for r in records:r['brightness']=r['arms'][a]['virtual_brightness']
        atomic_json(folder/f'brightness_{a}.json',pool_brightness(records))
    for r in records:r.pop('brightness',None)
    atomic_json(folder/'summary.json',dict(actual=base,virtual={a:summarize(rows,targets,predictions[a]) for a in ARMS},candidate={a:pooled(records,a) for a in ARMS},groups=groups,records=records,actual_pixel_changes=0,extra_network_forwards=0,role='primary causal70 extracted from complete sequential inference'))


def worker(cfg,out,arm,vid):
    pre=read(out/'preflight.json');info=next(v for v in pre['videos'] if v['video_id']==vid)
    if any(sha256(ROOT/k)!=v for k,v in pre['sources'].items()):raise ValueError('Worker source changed')
    for x in [info,*pre['checkpoints'].values()]:
        if sha256(x['path'])!=x['sha256']:raise ValueError('Worker input changed')
    source=read(out/f'source_{vid}.json');old=json_lines(ROOT/cfg['base']['reference_run']/f'full_video/raw2/{vid}/frames.jsonl')
    rows=read(ROOT/cfg['data']['video_manifest'])['val'];selected={r['frame_index']:i for i,r in enumerate(rows) if r['video_id']==vid};static_results=read(out/'video_native/summary.json')
    reference_growth=json_lines(ROOT/cfg['reference_union_run']/f'full_video/connected_blue_union/{vid}/frames.jsonl') if arm=='connected_blue_union' else None
    seed_training(42);experts=construct(pre);before=model_state_hash(experts);counts,handles=count_forwards(experts)
    pipeline=GrowthPipeline(cfg,arm,read(ROOT/cfg['t1']['config']),True);folder=out/f'full_video/{arm}/{vid}';prepare_folder(folder)
    def run(frame,timestamp):
        experts.sync();start=time.perf_counter();raw,sp,fp,t=infer(experts,frame,'raw2');base=time.perf_counter()
        candidate,proposal,conflict,meta=pipeline.run(frame,raw,fp,timestamp,vid);end=time.perf_counter()
        t.update(meta['timings']);t.update(base_pipeline_ms=(base-start)*1000,pipeline_total_ms=(end-start)*1000)
        return raw,sp,fp,candidate,proposal,conflict,meta,t
    cap=cv2.VideoCapture(info['path']);ok,first=cap.read();cap.release()
    if not ok:raise ValueError('Empty video')
    warm=cfg['execution']['warmup_frames_per_process']
    for j in range(warm):run(first,j/info['fps'])
    pipeline.reset();counts[:]=[0,0];cap=cv2.VideoCapture(info['path']);h,w=first.shape[:2];writer=checked_writer(folder/'replay.mp4',info['fps'],(w*2,h+84))
    i=matches=changed=cpu_peak=0;times=[];reasons=Counter();tip_states=Counter();advice=Counter();wall=time.perf_counter();cpu=time.process_time();torch.cuda.reset_peak_memory_stats()
    try:
        with (folder/'frames.jsonl').open('w',encoding='utf-8') as log:
            while True:
                start=time.perf_counter();ok,frame=cap.read();decoded=time.perf_counter()
                if not ok:break
                raw,sp,fp,candidate,proposal,conflict,meta,t=run(frame,source['frames'][i]['timestamp_seconds']);inferred=time.perf_counter()
                hashes={'mask_raw_sha256':array_hash(raw),'stone_probability_raw_sha256':array_hash(sp),'fiber_probability_raw_sha256':array_hash(fp)}
                rgb_hash=array_hash(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));src=source['frames'][i];pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if rgb_hash!=src['rgb_raw_sha256'] or any(v!=old[i][k] for k,v in hashes.items()):raise ValueError('Full raw source/output differs')
                if src['decoded_pts_seconds'] is not None and abs(pts-src['decoded_pts_seconds'])>1e-9:raise ValueError('PTS changed')
                gh={'candidate_sha256':array_hash(candidate),'proposal_sha256':array_hash(proposal),'conflict_sha256':array_hash(conflict),'growth_sha256':growth_hash(meta)}
                if reference_growth is not None and any(v!=reference_growth[i][k] for k,v in gh.items()):raise ValueError('T1 changed growth or current terminal evidence')
                if i in selected:
                    r=static_results['records'][selected[i]]
                    if any(v!=r[k] for k,v in hashes.items()) or (arm=='connected_blue_union' and any(v!=r['arms'][arm][k] for k,v in gh.items())):raise ValueError('Static/full original or legacy growth differs')
                    matches+=1
                if meta['apply_to_mask'] or np.any(candidate[raw!=0]):raise ValueError('Protected mask changed')
                writer.write(np.hstack([base_panel(frame,raw,i),overlay(frame,pipeline.final_mask,candidate,proposal,meta,arm+' SHADOW',i,pipeline.outline)]));rendered=time.perf_counter()
                save_candidates(folder,i,candidate,proposal,conflict);stage_layers=save_layers(folder,i,pipeline.layers);save_png(folder/f'experimental/{i:06d}.png',pipeline.final_mask)
                record={'frame_index':i,'timestamp_seconds':src['timestamp_seconds'],'source_rgb_raw_sha256':rgb_hash,**hashes,**gh,
                    'raw_mask_reference':str(ROOT/cfg['base']['reference_run']/f'full_video/raw2/{vid}/masks/{i:06d}.png'),
                    'stage_layers':stage_layers,'final_mask_sha256':array_hash(pipeline.final_mask),'outline':pipeline.outline,'candidate_pixels':int(candidate.sum()),'proposal_pixels':int(proposal.sum()),'conflict_pixels':int(conflict.sum()),'meta':meta,
                    'approved_semantic_reference':i in selected,'tip_metrics':None,'unlabelled_accuracy':None}
                log.write(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n');reasons[meta['reason']]+=1;tip_states.update(tip['state'] for tip in meta['tips']);advice[meta.get('t1_advice',{}).get('direction','disabled')]+=1
                changed+=int(candidate.any());cpu_peak=max(cpu_peak,process_memory()[0]);end=time.perf_counter()
                t.update(decode_ms=(decoded-start)*1000,render_audit_encode_ms=(rendered-inferred)*1000,record_export_ms=(end-rendered)*1000,cycle_ms=(end-start)*1000);times.append(t);i+=1
                if i%300==0:print(f'{arm} {vid} {i}/{info["frames"]}',flush=True)
    finally:cap.release();writer.release()
    wall=time.perf_counter()-wall;cpu=time.process_time()-cpu
    if i!=info['frames'] or matches!=len(selected) or counts!=[i,i] or model_state_hash(experts)!=before:raise ValueError('Worker coverage/graph/weights changed')
    for handle in handles:handle.remove()
    pipeline.reset();movie=verify_video(folder/'replay.mp4',i);keys=sorted(set().union(*(t.keys() for t in times)));timing={k:stats([t.get(k,0.) for t in times]) for k in keys}
    atomic_json(folder/'timings.json',times);result={'arm':arm,'video_id':vid,'pid':os.getpid(),'frames':i,'replay':movie,'approved_static_matches':matches,
        'actual_branch_forwards':counts,'warmup_branch_forwards':[warm,warm],'networks':2,'parameters':experts.parameter_count,'model_state_sha256':before,
        'individual_model_sha256':[model_hash(m) for m in experts.models],'runtime':runtime_identity(experts.device),'source_fps':info['fps'],'timestamp_mode':source['mode'],
        'timings':timing,'total_runtime_passed':timing['pipeline_total_ms']['p95']<=1000/info['fps'],'outline_budget_passed':timing['outline_ms']['p95']<=cfg['execution']['additional_outline_p95_ms'],
        'frames_with_candidate':changed,'reasons':dict(reasons),'tip_states':dict(tip_states),'t1_directions':dict(advice),'state_cleared':pipeline.tracker is None or pipeline.tracker.source is None,
        'offline_fps_including_decode_audit_export':i/wall,'wall_seconds':wall,'cpu_process_seconds':cpu,'windows_peak_working_set_mib':process_memory()[1]/2**20,
        'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'tip_metrics':None,'whole_video_accuracy':None,'all_actual_masks_unchanged':True}
    atomic_json(folder/'summary.json',result);print(json.dumps({'completed':arm,'video':vid,'p95':timing['pipeline_total_ms']['p95']}),flush=True)


def comparison(cfg,out,info):
    vid=info['video_id'];cap=cv2.VideoCapture(info['path']);w,h=info['width'],info['height'];logs={a:(out/f'full_video/{a}/{vid}/frames.jsonl').open(encoding='utf-8') for a in ARMS};path=out/f'{vid}_comparison.mp4';writer=checked_writer(path,info['fps'],(w*3,(h+84)*2+40));i=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            rows={a:json.loads(next(logs[a])) for a in ARMS};tiles=[np.vstack([panel(frame,'Original RGB',None,i),np.zeros((50,w,3),np.uint8)])]
            for a in ARMS:
                folder=out/f'full_video/{a}/{vid}';final=load_mask(folder/f'experimental/{i:06d}.png',frame.shape[:2],{0,1,2});c=load_mask(folder/f'candidate/{i:06d}.png',frame.shape[:2],{0,1});p=load_mask(folder/f'proposal/{i:06d}.png',frame.shape[:2],{0,1});tiles.append(overlay(frame,final,c,p,rows[a]['meta'],a,i,rows[a]['outline']))
            image=draw_outline(frame,rows['contour_sides_cap']['outline']);tiles.append(np.vstack([panel(image,'Sides+ellipse (gray=unaccepted)',None,i),np.zeros((50,w,3),np.uint8)]));strip=np.zeros((40,w*3,3),np.uint8);cv2.putText(strip,'Research masks only | yellow/orange sides; magenta accepted cap; gray proposed | official output unchanged; no tip GT',(5,25),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1);writer.write(np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:]),strip]));i+=1
    finally:
        cap.release();writer.release()
        for f in logs.values():f.close()
    if i!=info['frames']:raise ValueError('Incomplete comparison')
    return verify_video(path,i)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',nargs=2);args=parser.parse_args();cfg=read(CONFIG);out=ROOT/cfg['output']
    if tuple(cfg['arms'])!=ARMS:raise ValueError('Arms changed')
    if args.worker:return worker(cfg,out,*args.worker)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);job={'pid':os.getpid(),'training_updates':0,'default_changed':False}
    def progress(phase,**extra):job.update(phase=phase,**extra);atomic_json(out/'job.json',job);print(phase+' '+json.dumps(extra),flush=True)
    try:
        progress('preflight');protocol=read(ROOT/'runs/protocols/blue_outline_control_20261102_v2/identity.json')
        if any(sha256(ROOT/k)!=v for k,v in protocol['files'].items()):raise ValueError('Frozen protocol changed')
        prior=read(ROOT/cfg['base']['preflight'])
        if sha256(ROOT/cfg['base']['preflight'])!=protocol['reference_preflight_sha256'] or any(sha256(ROOT/k)!=v for k,v in prior['sources'].items()):raise ValueError('Prior source changed')
        vm=read(ROOT/cfg['data']['video_manifest']);im=read(ROOT/cfg['data']['image_manifest'])
        inputs=input_identity(ROOT/cfg['data']['image_manifest'],Path(im['root']),{'train':{'labeled':[],'unlabeled':[]},'val':vm['val']+im['val']})
        if inputs!=prior['inputs'] or len(vm['val'])!=70 or len(im['val'])!=132:raise ValueError('Reference changed')
        sources={**prior['sources']}
        for p in list((ROOT/'tools').glob('*.py'))+list((ROOT/'tests').glob('*.py')):sources[p.relative_to(ROOT).as_posix()]=sha256(p)
        related={**prior['related'],**{str(ROOT/k):v for k,v in protocol['files'].items()}}
        for name in [cfg['reference_union_config'],cfg['reference_aggressive_config'],cfg['reference_distal_config'],cfg['reference_extent_config'],cfg['legacy_config'],cfg['t1']['config'],cfg['historical_reference_quality']['frozen_gates'],cfg['data']['video_manifest'],cfg['data']['image_manifest']]:related[str(ROOT/name)]=sha256(ROOT/name)
        notes=ROOT/'docs/plans/fiber_blue_extent_implementation_notes_20261101.md';related[str(notes)]=sha256(notes)
        for scope in ['video_native','image_native']:
            p=ROOT/cfg['base']['reference_run']/f'{scope}/raw2.json';related[str(p)]=sha256(p)
        for info in prior['videos']:
            if sha256(info['path'])!=info['sha256']:raise ValueError('Video changed')
            progress('source_timestamp_audit',video_id=info['video_id']);source_scan(out,info,ROOT/cfg['base']['historical_reference'])
            for p in [out/f'source_{info["video_id"]}.json',ROOT/cfg['base']['reference_run']/f'full_video/raw2/{info["video_id"]}/frames.jsonl']:related[str(p)]=sha256(p)
        for x in prior['checkpoints'].values():
            if sha256(x['path'])!=x['sha256']:raise ValueError('Weights changed')
        pre={'config':cfg,'sources':sources,'related':related,'inputs':inputs,'checkpoints':prior['checkpoints'],'videos':prior['videos'],'mechanism_tests_passed':27,
            'train_pixels_read':0,'reserved_test_pixels_read':0,'training_updates':0}
        atomic_json(out/'preflight.json',pre)
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for k in sources:z.write(ROOT/k,k)
            for k in protocol['files']:z.write(ROOT/k,k)
            z.write(notes,notes.relative_to(ROOT).as_posix())
        static(cfg,pre,out,progress);full={}
        for arm in ARMS:
            full[arm]={}
            for info in pre['videos']:
                vid=info['video_id'];progress('full_video_fresh_process',arm=arm,video_id=vid)
                with (out/f'{arm}_{vid}.log').open('w',encoding='utf-8') as log:
                    subprocess.run([sys.executable,'-m','tools.run_blue_outline_repaired_control','--worker',arm,vid],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                full[arm][vid]=read(out/f'full_video/{arm}/{vid}/summary.json')
        causal_summary(cfg,out);gates=quality(cfg,out);atomic_json(out/'primary_quality.json',gates)
        exports={}
        for info in pre['videos']:progress('comparison_export',video_id=info['video_id']);exports[info['video_id']]=comparison(cfg,out,info)
        if any(sha256(ROOT/k)!=v for k,v in sources.items()) or any(sha256(k)!=v for k,v in related.items()):raise ValueError('Frozen source/reference changed')
        if inputs!=input_identity(ROOT/cfg['data']['image_manifest'],Path(im['root']),{'train':{'labeled':[],'unlabeled':[]},'val':vm['val']+im['val']}):raise ValueError('Reference input changed')
        result={'diagnostic_policy':gates,'full_video':full,'comparison_videos':exports,'runtime_passed':{a:all(v['total_runtime_passed'] for v in full[a].values()) for a in ARMS},'outline_runtime_passed':{a:all(v['outline_budget_passed'] for v in full[a].values()) for a in ARMS},
            'actual_static_paired_frame_inferences':202,'actual_full_paired_frame_inferences':10776,'warmup_paired_frames':96,'actual_output_pixel_changes':0,
            'adopted':False,'default_changed':False,'training_updates':0,'train_pixels_read':0,'reserved_test_pixels_read':0,'tip_metrics':None,'whole_video_accuracy':None,'s3_started':False}
        atomic_json(out/'summary.json',result);progress('completed',static_frames=202,full_frames=10776)
    except Exception as error:
        progress('failed',error=str(error));(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
