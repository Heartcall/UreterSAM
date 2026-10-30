"""Frozen T1 finite four-arm experiment; never changes a semantic mask."""
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
import torch
from ureter_semis.experiment import atomic_json, input_identity, seed_training, runtime_identity
from .fiber_reference_workspace import ROOT, read, sha256, load_mask
from .fiber_fast_observation import FastFiberObserver, public_observation
from .fiber_evidence_tracker import EvidenceTracker, empty_current, stateless_result
from .run_fiber_temporal_state_control import construct, source_scan
from .run_two_network_experts import infer, array_hash, model_hash
from .run_fiber_blue_aplus import panel, frame_read
from .replay_classwise_experts import model_state_hash, save_png, stats
from .run_classwise_expert_control import summarize

CONFIG = ROOT/'configs/fiber_temporal_observation_search_20261030.json'


class Pipeline:
    def __init__(self, cfg, arm, dt=1/30):
        self.arm = arm; self.cfg = cfg; self.observer = FastFiberObserver(cfg)
        self.tracker = EvidenceTracker(cfg, dt) if arm in ('evidence_state', 'history_search') else None

    def reset(self):
        if self.tracker: self.tracker.reset()

    def observe(self, frame, probability, timestamp, source):
        start = time.perf_counter(); prepared = self.tracker.prepare(frame.shape[:2], timestamp, source) if self.tracker else None
        prepared_at = time.perf_counter()
        current = self.observer.current(frame, probability) if self.arm != 'raw2' else empty_current()
        current_at = time.perf_counter(); history = None; history_attempted = False
        if self.arm == 'history_search' and current['strong_label'] is None and prepared['prior']['shaft'] is not None:
            history_attempted = True; history = self.observer.history(frame, probability, prepared['prior'])
        history_at = time.perf_counter()
        tracked = self.tracker.update(current, history, prepared) if self.tracker else stateless_result(current)
        ended = time.perf_counter()
        timings = {'state_prepare_ms': (prepared_at-start)*1000, 'history_sampling_fit_ms': (history_at-current_at)*1000,
                   'state_update_ms': (ended-history_at)*1000, 'geometry_total_ms': (ended-start)*1000,
                   **current.get('timings', {'grid_fov_proposals_ms':0.,'current_sampling_fit_ms':0.,'quality_select_ms':0.})}
        return public_observation({k:v for k,v in current.items() if k != 'timings'}), public_observation(history), tracked, history_attempted, timings


def state_panel(frame, arm, mask, tracked, index):
    image = panel(frame, arm, mask, index); shape = frame.shape[:2]
    shaft = tracked['groups']['shaft']; endpoint = tracked['groups']['endpoint']
    color = (0,220,255) if shaft['predicted'] else (255,255,0)
    if shaft['value'] is not None:
        ex,ey,theta,width = shaft['value']; e = np.array([ex,ey]); u = np.array([np.cos(theta),np.sin(theta)]); n = np.array([-u[1],u[0]])
        length, distal = endpoint['value'] if endpoint['value'] is not None else (min(shape)*.20, width)
        for sign in (-1,1):
            a = tuple(np.rint(e+sign*n*width/2+[0,34]).astype(int)); b = tuple(np.rint(e+u*length+sign*n*distal/2+[0,34]).astype(int))
            cv2.line(image,a,b,color,1,cv2.LINE_AA)
        if endpoint['value'] is not None:
            pc = (0,220,255) if tracked['front_is_prediction'] else color
            cv2.circle(image,tuple(np.rint(e+u*length+[0,34]).astype(int)),4,pc,1,cv2.LINE_AA)
    # Status goes below native pixels; the source image and mask are not obscured.
    strip = np.zeros((28,frame.shape[1],3),np.uint8)
    text = f"S:{shaft['status']} E:{endpoint['status']}"
    cv2.putText(strip,text,(3,18),cv2.FONT_HERSHEY_SIMPLEX,.31,(255,255,255),1,cv2.LINE_AA)
    return np.vstack([image,strip])


def static_reproduction(cfg, pre, out, progress):
    previous = ROOT/cfg['base']['reference_run']; seed_training(42); experts = construct(pre); before = model_state_hash(experts)
    for scope,key in [('video_native','video_manifest'),('image_native','image_manifest')]:
        rows = read(ROOT/cfg['data'][key])['val']; old = read(previous/f'{scope}/member1_2net.json'); targets=[]; predictions=[]; records=[]
        (out/f'{scope}/shared_raw2').mkdir(parents=True)
        for i,row in enumerate(rows):
            frame = frame_read(row['image']); mask,sp,fp,_ = infer(experts,frame)
            hashes = {'mask_raw_sha256':array_hash(mask),'stone_probability_raw_sha256':array_hash(sp),'fiber_probability_raw_sha256':array_hash(fp)}
            if any(h != old['records'][i][k] for k,h in hashes.items()): raise ValueError('Static raw output differs')
            observations = {}
            for arm in cfg['arms']:
                pipeline = Pipeline(cfg,arm)
                current,history,tracked,attempted,_ = pipeline.observe(frame,fp,0,row['sample_id'])
                if history is not None or attempted: raise ValueError('History used on independent static image')
                observations[arm] = {'current':current,'history':history,'tracked':tracked}
            save_png(out/f'{scope}/shared_raw2/{i:04d}.png',mask)
            targets.append(load_mask(row['mask'],mask.shape,{0,1,2,255})); predictions.append(mask)
            records.append({'sample_id':row['sample_id'],'video_id':row.get('video_id'),'frame_index':row.get('frame_index'),**hashes,'observations':observations,'tip_metrics':None})
            if (i+1)%30 == 0: progress('static_reproduction',scope=scope,frames=i+1)
        summary = summarize(rows,targets,predictions)
        if summary != old['summary']: raise ValueError('Static metrics differ')
        atomic_json(out/f'{scope}.json',{'actual_raw2_fresh_inferences':len(rows),'summary':summary,'groups':old['groups'],'records':records,
            'all_arm_masks':'shared exact raw2 output; no multiple static forwards claimed','classwise_frozen_reference':read(previous/f'{scope}/classwise3.json')['summary'],
            'paired_changes_all_arms':{c:{k:0 for k in ('lost_tp','gained_tp','added_fp','removed_fp')} for c in ('stone','fiber')},'tip_metrics':None})
    if model_state_hash(experts) != before: raise ValueError('Static weights changed')
    atomic_json(out/'static_model_identity.json',{'model_state_sha256':before,'individual_model_sha256':[model_hash(m) for m in experts.models],'runtime':runtime_identity(experts.device),'models':2,'parameters':experts.parameter_count})
    del experts; torch.cuda.empty_cache()


@torch.inference_mode()
def worker(cfg,out,arm,vid):
    pre = read(out/'preflight.json'); previous = ROOT/cfg['base']['reference_run']; info = next(x for x in pre['videos'] if x['video_id']==vid)
    for x in [info,*pre['checkpoints'].values()]:
        if sha256(x['path']) != x['sha256']: raise ValueError('Worker input changed')
    if any(sha256(ROOT/k)!=v for k,v in pre['sources'].items()): raise ValueError('Worker source changed')
    source = read(out/f'source_{vid}.json'); old = [json.loads(s) for s in (previous/f'full_video/member1_2net/{vid}/frames.jsonl').read_text().splitlines()]
    rows = read(ROOT/cfg['data']['video_manifest'])['val']; selected = {r['frame_index']:i for i,r in enumerate(rows) if r['video_id']==vid}; static = read(out/'video_native.json')
    seed_training(42); experts = construct(pre); before = model_state_hash(experts); pipeline = Pipeline(cfg,arm,1/info['fps'])
    folder = out/f'full_video/{arm}/{vid}'; folder.mkdir(parents=True)
    def run(frame,timestamp,source_id):
        experts.sync(); start = time.perf_counter(); mask,sp,fp,t = infer(experts,frame); coarse = time.perf_counter()
        current,history,tracked,attempted,gt = pipeline.observe(frame,fp,timestamp,source_id); end = time.perf_counter()
        t.update(gt); t.update(coarse_pipeline_ms=(coarse-start)*1000,pipeline_total_ms=(end-start)*1000)
        return mask,sp,fp,current,history,tracked,attempted,t
    cap = cv2.VideoCapture(info['path']); ok,first = cap.read(); cap.release()
    if not ok: raise ValueError('Empty source')
    for i in range(cfg['execution']['warmup_frames']): run(first,i/info['fps'],'scratch_warmup')
    pipeline.reset(); cap = cv2.VideoCapture(info['path']); height,width = first.shape[:2]
    writer = cv2.VideoWriter(str(folder/'state_replay.mp4'),cv2.CAP_MSMF,cv2.VideoWriter_fourcc(*'H264'),info['fps'],(2*width,height+62))
    if not writer.isOpened(): raise RuntimeError('Encoder unavailable')
    times=[]; records=[]; matches=0; i=0; wall=time.perf_counter(); cpu_start=time.process_time(); torch.cuda.reset_peak_memory_stats()
    try:
        while True:
            start = time.perf_counter(); ok,frame = cap.read(); decoded = time.perf_counter()
            if not ok: break
            src = source['frames'][i]; pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if src['decoded_pts_seconds'] is not None and abs(pts-src['decoded_pts_seconds'])>1e-9: raise ValueError('PTS changed')
            mask,sp,fp,current,history,tracked,attempted,t = run(frame,src['timestamp_seconds'],vid); inferred=time.perf_counter()
            rgb_hash = array_hash(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); mask_hash=array_hash(mask)
            if rgb_hash != src['rgb_raw_sha256'] or mask_hash != old[i]['mask_raw_sha256']: raise ValueError('Actual current source/mask changed')
            if i in selected:
                r=static['records'][selected[i]]
                if mask_hash!=r['mask_raw_sha256'] or array_hash(sp)!=r['stone_probability_raw_sha256'] or array_hash(fp)!=r['fiber_probability_raw_sha256']: raise ValueError('Selected probability mismatch')
                matches+=1
            if tracked['apply_to_mask']: raise ValueError('State applied to mask')
            input_panel=np.vstack([panel(frame,'Input',None,i),np.zeros((28,width,3),np.uint8)])
            writer.write(np.hstack([input_panel,state_panel(frame,arm,mask,tracked,i)])); rendered=time.perf_counter()
            records.append({'frame_index':i,'timestamp_seconds':src['timestamp_seconds'],'source_rgb_raw_sha256':rgb_hash,'mask_raw_sha256':mask_hash,
                            'mask_reference':str(previous/f'full_video/member1_2net/{vid}/masks/{i:06d}.png'),
                            'current':current,'history':history,'history_search_attempted':attempted,'tracked':tracked,
                            'approved_semantic_reference':i in selected,'tip_metrics':None,'unlabelled_accuracy':None})
            end=time.perf_counter();t.update(decode_ms=(decoded-start)*1000,render_encode_ms=(rendered-inferred)*1000,audit_record_ms=(end-rendered)*1000,cycle_ms=(end-start)*1000)
            times.append(t);i+=1
            if i%300==0: print(f'{arm} {vid} {i}/{info["frames"]}',flush=True)
    finally: cap.release();writer.release()
    wall=time.perf_counter()-wall; cpu_seconds=time.process_time()-cpu_start
    if i!=info['frames'] or matches!=len(selected) or model_state_hash(experts)!=before: raise ValueError('Coverage or weight mismatch')
    pipeline.reset();check=cv2.VideoCapture(str(folder/'state_replay.mp4')); encoded=0
    while check.read()[0]: encoded+=1
    check.release()
    if encoded!=i: raise ValueError('Incomplete video')
    (folder/'frames.jsonl').write_text('\n'.join(json.dumps(r,allow_nan=False) for r in records)+'\n',encoding='utf-8');atomic_json(folder/'timings.json',times)
    statuses={g:dict(Counter(r['tracked']['groups'][g]['status'] for r in records)) for g in ('shaft','endpoint')}
    result={'arm':arm,'video_id':vid,'pid':os.getpid(),'frames':i,'encoded_frames':encoded,'source_fps':info['fps'],'timestamp_mode':source['mode'],
            'networks':2,'parameters':experts.parameter_count,'model_state_sha256':before,'individual_model_sha256':[model_hash(m) for m in experts.models],
            'runtime':{**runtime_identity(experts.device),'opencv':cv2.__version__,'opencv_threads':cv2.getNumThreads()},'approved_static_matches':matches,
            'timings':{k:stats([t[k] for t in times]) for k in times[0]},'wall_seconds':wall,'cpu_process_seconds_including_render':cpu_seconds,
            'offline_fps_including_render_audit_encode':i/wall,'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'pipeline_p95_within_source_interval':stats([t['pipeline_total_ms'] for t in times])['p95']<=1000/info['fps'],
            'geometry_p95_within_design_budget':stats([t['geometry_total_ms'] for t in times])['p95']<=cfg['budget']['additional_geometry_stage_p95_target_ms'],
            'statuses':statuses,'ambiguous_frames':sum(r['current']['ambiguous'] for r in records),
            'current_strong_frames':sum(r['current']['strong_label'] is not None for r in records),
            'qualified_shaft_frames':sum(any(c['shaft_observed'] for c in r['current']['candidates']) for r in records),
            'qualified_front_frames':sum(any(c['front_observed'] for c in r['current']['candidates']) for r in records),
            'history_attempted_frames':sum(r['history_search_attempted'] for r in records),
            'history_used_frames':sum(r['tracked'].get('used_provenance')=='history_conditioned' for r in records),
            'strong_override_frames':sum(bool(r['tracked'].get('strong_override_groups')) for r in records),
            'candidate_reasons':dict(Counter(c['reason'] for r in records for c in r['current']['candidates'])),
            'all_raw_masks_unchanged':True,'state_cleared_after_video':pipeline.tracker is None or pipeline.tracker.source is None,
            'video_sha256':sha256(folder/'state_replay.mp4'),'tip_metrics':None,'whole_video_accuracy':None}
    atomic_json(folder/'summary.json',result);print(json.dumps({'completed':arm,'video':vid,'p95':result['timings']['pipeline_total_ms']['p95']}),flush=True)


def comparison(out,cfg,info):
    vid=info['video_id']; logs={a:[json.loads(s) for s in (out/f'full_video/{a}/{vid}/frames.jsonl').read_text().splitlines()] for a in cfg['arms']}
    cap=cv2.VideoCapture(info['path']);w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    path=out/f'{vid}_state_comparison.mp4';writer=cv2.VideoWriter(str(path),cv2.CAP_MSMF,cv2.VideoWriter_fourcc(*'H264'),info['fps'],(w*5,h+62));i=0
    if not writer.isOpened(): raise RuntimeError('Encoder unavailable')
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            mask=load_mask(logs['raw2'][i]['mask_reference'],frame.shape[:2],{0,1,2})
            panels=[np.vstack([panel(frame,'Input',None,i),np.zeros((28,w,3),np.uint8)])]
            panels.extend(state_panel(frame,a,mask,logs[a][i]['tracked'],i) for a in cfg['arms'])
            writer.write(np.hstack(panels));i+=1
    finally: cap.release();writer.release()
    cap=cv2.VideoCapture(str(path));n=0
    while cap.read()[0]:n+=1
    cap.release()
    if n!=i or n!=info['frames']:raise ValueError('Incomplete comparison video')
    return {'path':str(path),'frames':n,'sha256':sha256(path)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',nargs=2);args=parser.parse_args();cfg=read(CONFIG);out=ROOT/cfg['planned_output']
    if args.worker: return worker(cfg,out,*args.worker)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);job={'pid':os.getpid(),'training_updates':0,'default_changed':False}
    def progress(phase,**extra):job.update(phase=phase,**extra);atomic_json(out/'job.json',job);print(phase+' '+json.dumps(extra),flush=True)
    try:
        progress('preflight');prior=read(ROOT/cfg['reference_t0']/'preflight.json');identity=read(ROOT/'runs/protocols/fiber_temporal_observation_search_20261030_v1/identity.json')
        if any(sha256(ROOT/k)!=v for k,v in identity['files'].items()):raise ValueError('T1 frozen protocol or old source changed')
        if any(sha256(ROOT/k)!=v for k,v in prior['sources'].items()):raise ValueError('Old source changed')
        vm=read(ROOT/cfg['data']['video_manifest']);im=read(ROOT/cfg['data']['image_manifest'])
        if len(vm['val'])!=70 or len(im['val'])!=132:raise ValueError('Reference count changed')
        inputs=input_identity(ROOT/cfg['data']['image_manifest'],Path(im['root']),{'train':{'labeled':[],'unlabeled':[]},'val':vm['val']+im['val']})
        if inputs!=prior['inputs']:raise ValueError('Approved reference identity differs')
        sources={**prior['sources']}
        for p in list((ROOT/'tools').glob('*.py'))+list((ROOT/'tests').glob('*.py')):sources[p.relative_to(ROOT).as_posix()]=sha256(p)
        related={str(CONFIG):sha256(CONFIG),str(ROOT/cfg['design_document']):sha256(ROOT/cfg['design_document'])}
        previous=ROOT/cfg['base']['reference_run']
        for scope in ['video_native','image_native']:
            for arm in ['member1_2net','classwise3']:
                p=previous/f'{scope}/{arm}.json';related[str(p)]=sha256(p)
        for info in prior['videos']:
            if sha256(info['path'])!=info['sha256']:raise ValueError('Video changed')
            progress('source_timestamp_audit',video_id=info['video_id']);source_scan(out,info,previous)
            p=out/f'source_{info["video_id"]}.json';related[str(p)]=sha256(p)
            for p in (previous/f'full_video/member1_2net/{info["video_id"]}/masks').glob('*.png'):related[str(p)]=sha256(p)
            p=previous/f'full_video/member1_2net/{info["video_id"]}/frames.jsonl';related[str(p)]=sha256(p)
        for cp in prior['checkpoints'].values():
            if sha256(cp['path'])!=cp['sha256']:raise ValueError('Weights changed')
        pre={'config':cfg,'config_sha256':sha256(CONFIG),'sources':sources,'related':related,'inputs':inputs,'checkpoints':prior['checkpoints'],'videos':prior['videos'],
             'training_updates':0,'train_pixels_read':0,'reserved_test_pixels_read':0,'unit_tests_before_run':{'new':19,'T0_regression':11,'passed':30},
             'protocol_wording_clarification':'static445-pixel case has semantic GT; unlabelled/zero-seed in frozen config means no neural seed, NOT absence of GT. No thresholds or execution rules changed.'}
        atomic_json(out/'preflight.json',pre)
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in sources:z.write(ROOT/p,p)
            z.write(CONFIG,CONFIG.relative_to(ROOT).as_posix());z.write(ROOT/cfg['design_document'],cfg['design_document'])
        static_reproduction(cfg,pre,out,progress);full={}
        for arm in cfg['arms']:
            full[arm]={}
            for info in pre['videos']:
                vid=info['video_id'];progress('full_video_fresh_process',arm=arm,video_id=vid)
                with (out/f'{arm}_{vid}.log').open('w',encoding='utf-8') as log:
                    subprocess.run([sys.executable,'-m','tools.run_fiber_temporal_observation_search','--worker',arm,vid],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                full[arm][vid]=read(out/f'full_video/{arm}/{vid}/summary.json')
        exports={}
        for info in pre['videos']:progress('comparison_export',video_id=info['video_id']);exports[info['video_id']]=comparison(out,cfg,info)
        if any(sha256(ROOT/k)!=v for k,v in sources.items()) or any(sha256(k)!=v for k,v in related.items()):raise ValueError('Source/reference changed')
        if inputs!=input_identity(ROOT/cfg['data']['image_manifest'],Path(im['root']),{'train':{'labeled':[],'unlabeled':[]},'val':vm['val']+im['val']}):raise ValueError('Reference changed')
        if any(sha256(x['path'])!=x['sha256'] for x in [*pre['checkpoints'].values(),*pre['videos']]):raise ValueError('Input weights or video changed')
        atomic_json(out/'summary.json',{'full_video':full,'comparison_videos':exports,'static_fresh_forwards':202,'full_forward_frames':10776,
            'state_only':True,'all_masks_unchanged':True,'default_changed':False,'adopted':False,'training_updates':0,'train_pixels_read':0,'reserved_test_pixels_read':0,
            'tip_metrics':None,'whole_video_accuracy':None,'s3_started':False})
        progress('completed',full_frames=10776)
    except Exception as e:
        progress('failed',error=str(e));(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__': main()
