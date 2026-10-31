"""Finite frozen three-arm input response experiment, including six full-video processes."""
from __future__ import annotations

import argparse
import ctypes
import json
import math
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

from ureter_semis.experiment import atomic_json, input_identity, runtime_identity, seed_training
from ureter_semis.supervised_trainer import assess
from .fiber_reference_workspace import ROOT, read, sha256, load_mask
from .fiber_photometric_input import ARMS, GAMMA, infer, rgb_grid, transform_rgb
from .run_fiber_temporal_state_control import construct, source_scan
from .run_two_network_experts import array_hash, model_hash
from .run_fiber_blue_aplus import frame_read, panel
from .replay_classwise_experts import save_png, stats, model_state_hash
from .run_classwise_expert_control import summarize
from .diagnose_joint_failures import paired_changes

CONFIG = ROOT/'configs/fiber_dark_input_control_20261030.json'
FIXED = {'video_1':{77}, 'video_2':{310,438,621,642,1208,1265}}
COUNT_KEYS = ('gt_pixels','tp','fp','fn','fn_Pbelow01','gt_probability_sum')


def process_memory():
    """Read native Windows working set without adding an environment dependency."""
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
    counters=Counters();counters.cb=ctypes.sizeof(counters)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.WorkingSetSize,counters.PeakWorkingSetSize


def json_lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]


def count_forwards(experts):
    counts = [0, 0]
    def hook(index):
        def called(*args): counts[index] += 1
        return called
    handles = [m.register_forward_hook(hook(i)) for i,m in enumerate(experts.models)]
    return counts, handles


def finish_counts(d):
    return {**d, 'recall':d['tp']/d['gt_pixels'] if d['gt_pixels'] else None,
            'mean_probability':d['gt_probability_sum']/d['gt_pixels'] if d['gt_pixels'] else None}


def brightness(frame, target, mask, probability):
    truth=target==2; valid=target!=255; pred=mask==2
    fields={'HSV_V':cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)[:,:,2],
            'Lab_Lstar':cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)[:,:,0]}
    interior=cv2.distanceTransform(truth.astype(np.uint8),cv2.DIST_L2,5)>3*min(target.shape)/400
    result={}
    for feature,value in fields.items():
        bins=[]
        for lower,upper in [(0,85),(85,170),(170,256)]:
            region=(value>=lower)&(value<upper)
            def counts(r):
                t=truth&r; tp=t&pred; fn=t&~pred; fp=pred&~truth&valid&r
                return finish_counts({'gt_pixels':int(t.sum()),'tp':int(tp.sum()),'fp':int(fp.sum()),'fn':int(fn.sum()),
                    'fn_Pbelow01':int((fn&(probability<.1)).sum()),'gt_probability_sum':float(probability[t].sum(dtype=np.float64))})
            bins.append({**counts(region),'interior':counts(region&interior)})
        result[feature]=bins
    return result


def pool_brightness(records):
    result={}
    for feature in ('HSV_V','Lab_Lstar'):
        bins=[]
        for k in range(3):
            selected=[r['brightness'][feature][k] for r in records]
            d=finish_counts({n:sum(s[n] for s in selected) for n in COUNT_KEYS})
            d['interior']=finish_counts({n:sum(s['interior'][n] for s in selected) for n in COUNT_KEYS})
            bins.append(d)
        result[feature]=bins
    return result


def skeleton(binary):
    """Morphological semantic skeleton; neither a physical axis nor anatomical tip."""
    remaining=binary.astype(np.uint8); result=np.zeros_like(remaining)
    kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    while remaining.any():
        eroded=cv2.erode(remaining,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
        opened=cv2.dilate(eroded,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
        result |= remaining & ~opened
        remaining=eroded
    return result.astype(bool)


def centerline_counts(target, mask):
    truth=target==2; pred=(mask==2)&(target!=255); s=skeleton(truth); p=skeleton(pred)
    return {'reference_skeleton_pixels':int(s.sum()),'reference_skeleton_covered':int((s&pred).sum()),
            'predicted_skeleton_pixels':int(p.sum()),'predicted_skeleton_in_reference':int((p&truth).sum())}


def centerline_pool(records):
    keys=('reference_skeleton_pixels','reference_skeleton_covered','predicted_skeleton_pixels','predicted_skeleton_in_reference')
    d={k:sum(r['semantic_centerline'][k] for r in records) for k in keys}
    d['reference_coverage']=d['reference_skeleton_covered']/d['reference_skeleton_pixels'] if d['reference_skeleton_pixels'] else None
    d['predicted_precision']=d['predicted_skeleton_in_reference']/d['predicted_skeleton_pixels'] if d['predicted_skeleton_pixels'] else None
    return d


def color_summary(original, transformed):
    def distribution(x):
        return {'mean':float(x.mean()),'p05_p25_p50_p75_p95':np.percentile(x,[5,25,50,75,95]).tolist()}
    def fields(x):
        hsv=cv2.cvtColor(x,cv2.COLOR_RGB2HSV); lab=cv2.cvtColor(x,cv2.COLOR_RGB2LAB)
        return {'V':distribution(hsv[:,:,2]),'saturation':distribution(hsv[:,:,1]),'Lab_Lstar_0_100':distribution(lab[:,:,0]),
                'RGB':{str(c):distribution(x[:,:,c]) for c in range(3)}}
    a=original.sum(2,keepdims=True); b=transformed.sum(2,keepdims=True)
    ca=np.divide(original,a,out=np.zeros_like(original),where=a>0)
    cb=np.divide(transformed,b,out=np.zeros_like(transformed),where=b>0)
    return {'original':fields(original),'transformed':fields(transformed),
            'max_chromaticity_absolute_change':float(np.abs(ca-cb).max()),
            'original_grid_raw_sha256':array_hash(original),'transformed_grid_raw_sha256':array_hash(transformed)}


def select_groups(rows, video):
    if video:
        groups={v:[i for i,r in enumerate(rows) if r['video_id']==v] for v in ('video_1','video_2')}
        groups.update({c:[i for i,r in enumerate(rows) if r['evaluation_cohort']==c] for c in ('original29','new_user_approved')})
    else:
        exposed=set(read(ROOT/'runs/a1_classwise_experts_20261028/summary.json')['image']['historical_exposure_cohorts']['historical_old190_exposed']['sample_ids'])
        groups={s:[i for i,r in enumerate(rows) if r['source_dataset']==s] for s in sorted({r['source_dataset'] for r in rows})}
        groups.update({name:[i for i,r in enumerate(rows) if (r['sample_id'] in exposed)==flag] for name,flag in [('old190_exposed25',True),('not_matched107',False)]})
    return groups


def static_arm(cfg, pre, out, arm, progress):
    seed_training(42); experts=construct(pre); before=model_state_hash(experts); counts,handles=count_forwards(experts)
    previous=ROOT/cfg['base']['reference_run']
    for scope,key in [('video_native','video_manifest'),('image_native','image_manifest')]:
        rows=read(ROOT/cfg['data'][key])['val']; old=read(previous/f'{scope}/member1_2net.json')
        folder=out/f'{scope}/{arm}';folder.mkdir(parents=True);(folder/'probabilities').mkdir()
        targets=[];predictions=[];records=[]
        for i,row in enumerate(rows):
            frame=frame_read(row['image']); target=load_mask(row['mask'],frame.shape[:2],{0,1,2,255})
            mask,sp,fp,t=infer(experts,frame,arm)
            hashes={'mask_raw_sha256':array_hash(mask),'stone_probability_raw_sha256':array_hash(sp),'fiber_probability_raw_sha256':array_hash(fp)}
            if hashes['stone_probability_raw_sha256']!=old['records'][i]['stone_probability_raw_sha256']:raise ValueError('Stone probability changed')
            if arm=='raw2' and any(v!=old['records'][i][k] for k,v in hashes.items()):raise ValueError('Raw2 native reproduction differs')
            if any(p.dtype!=np.float32 or not np.isfinite(p).all() or not np.allclose(p.sum(0),1,atol=2e-5) for p in (sp,fp)):raise ValueError('Invalid native probability')
            base=mask if arm=='raw2' else load_mask(out/f'{scope}/raw2/{i:04d}.png',mask.shape,{0,1,2})
            bp=fp[2] if arm=='raw2' else np.load(out/f'{scope}/raw2/probabilities/{i:04d}.npz')['fiber'][2]
            truth=target==2; valid=target!=255; recovered=truth&(base!=2)&(mask==2); created=(mask==2)&(base!=2)&~truth&valid
            original=rgb_grid(frame); transformed=transform_rgb(original,arm)
            if arm=='fiber_vlift' and color_summary(original,transformed)['max_chromaticity_absolute_change']>3e-7:raise ValueError('Chroma preservation differs')
            np.savez_compressed(folder/f'probabilities/{i:04d}.npz',stone=sp,fiber=fp)
            save_png(folder/f'{i:04d}.png',mask)
            record={'sample_id':row['sample_id'],'video_id':row.get('video_id'),'frame_index':row.get('frame_index'),**hashes,
                'image_sha256':sha256(row['image']),'reference_sha256':sha256(row['mask']),
                'paired_stone':paired_changes(target,base,mask,1),'paired_fiber':paired_changes(target,base,mask,2),
                'brightness':brightness(frame,target,mask,fp[2]),'semantic_centerline':centerline_counts(target,mask),
                'input_color':color_summary(original,transformed),'input_V_target_max_absolute_error':float(np.abs(transformed.max(2)-(original.max(2) if arm=='raw2' else original.max(2)**np.float32(GAMMA))).max()),
                'baseline_FN_Pbelow01_recovered_TP':int((recovered&(bp<.1)).sum()),
                'baseline_FN_Pbelow01_remaining':int((truth&(base!=2)&(bp<.1)&(mask!=2)).sum()),
                'new_FP_baselinePbelow01':int((created&(bp<.1)).sum()),
                'baseline_highconfidence_TP_lost':int((truth&(base==2)&(bp>=.9)&(mask!=2)).sum()),
                'changed_pixels':int((mask!=base).sum()),'timings_not_latency_acceptance':t,'tip_metrics':None}
            records.append(record); targets.append(target);predictions.append(mask)
            if (i+1)%30==0:progress('static_arm',arm=arm,scope=scope,completed=i+1)
        summary=summarize(rows,targets,predictions)
        if arm=='raw2' and summary!=old['summary']:raise ValueError('Raw2 metrics differ')
        groups={}
        for name,ids in select_groups(rows,scope=='video_native').items():
            selected=[records[j] for j in ids]
            groups[name]={'frames':len(ids),'summary':summarize([rows[j] for j in ids],[targets[j] for j in ids],[predictions[j] for j in ids]),
                          'brightness':pool_brightness(selected),'semantic_centerline':centerline_pool(selected)}
        atomic_json(out/f'{scope}/{arm}.json',{'summary':summary,'groups':groups,'records':records,
            'brightness':pool_brightness(records),'semantic_centerline':centerline_pool(records),'actual_paired_frame_inferences':len(rows),
            'paired_changes':{c:{k:sum(r['paired_'+c][k] for r in records) for k in ('lost_tp','gained_tp','added_fp','removed_fp')} for c in ('stone','fiber')},'tip_metrics':None})
    if counts!=[202,202] or model_state_hash(experts)!=before:raise ValueError('Static graph/weights differ')
    for h in handles:h.remove()
    atomic_json(out/f'static_identity_{arm}.json',{'model_state_sha256':before,'individual_model_sha256':[model_hash(m) for m in experts.models],
        'networks':2,'parameters':experts.parameter_count,'actual_branch_forwards':counts,'runtime':runtime_identity(experts.device)})
    del experts;torch.cuda.empty_cache()


def quality(cfg,out):
    frozen=read(ROOT/cfg['quality']['unchanged_gate_reference']);eps=frozen['recommendation_checks']['epsilon'];results={}
    for arm in ARMS:
        checks={};violations=[]; legacy=None
        for scope in ('video_native','image_native'):
            candidate=read(out/f'{scope}/{arm}.json');base=read(out/f'{scope}/raw2.json')
            classwise=read(ROOT/cfg['base']['reference_run']/f'{scope}/classwise3.json')
            for refname,ref in [('raw2',base),('classwise3',classwise)]:
                for metric in frozen['recommendation_checks']['metrics']:
                    if metric.endswith('_boundary_f'):
                        c='1' if metric.startswith('stone') else '2'
                        a=ref['summary']['boundaries'][c]['f_score'];b=candidate['summary']['boundaries'][c]['f_score']
                    else:a=ref['summary']['metrics'][metric];b=candidate['summary']['metrics'][metric]
                    checks[f'{scope}:{refname}:{metric}']=a is not None and b is not None and b>=a-eps
            subsets=[('all',candidate,base)]
            if scope=='video_native':
                subsets.extend((v,candidate['groups'][v],base['groups'][v]) for v in ('video_1','video_2'))
                legacy=assess(candidate['groups']['original29']['summary']['metrics'],frozen['legacy_original29_gates'])
                checks['original29_legacy_gates']=legacy['passed']
            for name,c,b in subsets:
                for k in (0,1):
                    cc=c['brightness']['Lab_Lstar'][k];bb=b['brightness']['Lab_Lstar'][k]
                    checks[f'{scope}:{name}:Lab{k}_TP_nonregression']=cc['gt_pixels']>0 and cc['tp']>=bb['tp']
            checks[f'{scope}:dark_FN_strict_reduction']=candidate['brightness']['Lab_Lstar'][0]['fn']<base['brightness']['Lab_Lstar'][0]['fn']
            for cls in ('stone','fiber'):
                m=base['summary']['metrics'];n=candidate['summary']['metrics'];changes=candidate['paired_changes'][cls]
                checks[f'{scope}:{cls}:grossTP_limit']=changes['lost_tp']<=math.floor(.001*m[cls+'_tp'])
                checks[f'{scope}:{cls}:netTP_nondecrease']=n[cls+'_tp']>=m[cls+'_tp']
                bad=[]
                for i,(r,f) in enumerate(zip(candidate['records'],candidate['summary']['frames'])):
                    gt=sum(f['confusion_matrix'][1 if cls=='stone' else 2]);limit=max(1,math.floor(.005*gt))
                    if r['paired_'+cls]['lost_tp']>limit:bad.append({'scope':scope,'index':i,'sample_id':r['sample_id'],'class':cls,'lost_tp':r['paired_'+cls]['lost_tp'],'limit':limit})
                checks[f'{scope}:{cls}:perframeTP_limit']=not bad;violations.extend(bad)
            if scope=='video_native':
                index=next(i for i,r in enumerate(candidate['records']) if r['video_id']=='video_2' and r['frame_index']==1208)
                checks['fixed_video2_display1209_dark_FN_strict_reduction']=candidate['records'][index]['brightness']['Lab_Lstar'][0]['fn']<base['records'][index]['brightness']['Lab_Lstar'][0]['fn']
        results[arm]={'checks':checks,'passed':all(checks.values()),'rejected_checks':[k for k,v in checks.items() if not v],
                      'perframeTP_violations':violations,'original29':legacy,'tip_metrics':None}
    return results


def checked_writer(path,fps,size):
    writer=cv2.VideoWriter(str(path),cv2.CAP_MSMF,cv2.VideoWriter_fourcc(*'H264'),fps,size)
    if not writer.isOpened():raise RuntimeError('H264 encoder unavailable')
    return writer


def verify_video(path,expected):
    cap=cv2.VideoCapture(str(path));n=0
    while cap.read()[0]:n+=1
    cap.release()
    if n!=expected:raise ValueError('Incomplete output video')
    return {'path':str(path),'frames':n,'sha256':sha256(path)}


def worker(cfg,out,arm,vid):
    pre=read(out/'preflight.json');info=next(v for v in pre['videos'] if v['video_id']==vid)
    for item in [info,*pre['checkpoints'].values()]:
        if sha256(item['path'])!=item['sha256']:raise ValueError('Worker input changed')
    if any(sha256(ROOT/k)!=v for k,v in pre['sources'].items()):raise ValueError('Worker source changed')
    source=read(out/f'source_{vid}.json'); old=json_lines(ROOT/cfg['base']['reference_run']/f'full_video/member1_2net/{vid}/frames.jsonl')
    rows=read(ROOT/cfg['data']['video_manifest'])['val'];selected={r['frame_index']:i for i,r in enumerate(rows) if r['video_id']==vid}
    static=read(out/f'video_native/{arm}.json');rawlog=None if arm=='raw2' else json_lines(out/f'full_video/raw2/{vid}/frames.jsonl')
    seed_training(42);experts=construct(pre);before=model_state_hash(experts);counts,handles=count_forwards(experts)
    folder=out/f'full_video/{arm}/{vid}';folder.mkdir(parents=True);(folder/'masks').mkdir();(folder/'fixed_probabilities').mkdir()
    cap=cv2.VideoCapture(info['path']);ok,first=cap.read();cap.release()
    if not ok:raise ValueError('Empty source')
    warm=cfg['execution']['warmup_frames_per_process']
    for _ in range(warm):infer(experts,first,arm)
    if counts!=[warm,warm]:raise ValueError('Warmup forwards differ')
    counts[:]=[0,0];cap=cv2.VideoCapture(info['path']);h,w=first.shape[:2]
    writer=checked_writer(folder/'replay.mp4',info['fps'],(2*w,h+34))
    times=[];matches=0;i=0;changed=0;cpu_peak=0;wall=time.perf_counter();cpu=time.process_time();torch.cuda.reset_peak_memory_stats()
    try:
        with (folder/'frames.jsonl').open('w',encoding='utf-8') as log:
            while True:
                start=time.perf_counter();ok,frame=cap.read();decoded=time.perf_counter()
                if not ok:break
                mask,sp,fp,t=infer(experts,frame,arm);inferred=time.perf_counter()
                rgb_hash=array_hash(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));mask_hash=array_hash(mask);sp_hash=array_hash(sp);fp_hash=array_hash(fp)
                src=source['frames'][i];pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if rgb_hash!=src['rgb_raw_sha256'] or (src['decoded_pts_seconds'] is not None and abs(pts-src['decoded_pts_seconds'])>1e-9):raise ValueError('Source RGB/PTS differs')
                if arm=='raw2' and mask_hash!=old[i]['mask_raw_sha256']:raise ValueError('Raw2 full native mask differs')
                if rawlog is not None and sp_hash!=rawlog[i]['stone_probability_raw_sha256']:raise ValueError('Full stone probability differs')
                if i in selected:
                    r=static['records'][selected[i]]
                    if any(r[k]!=v for k,v in [('mask_raw_sha256',mask_hash),('stone_probability_raw_sha256',sp_hash),('fiber_probability_raw_sha256',fp_hash)]):raise ValueError('Static/full probability differs')
                    matches+=1
                if i in FIXED[vid]:np.savez_compressed(folder/f'fixed_probabilities/{i:06d}.npz',stone=sp,fiber=fp)
                basehash=old[i]['mask_raw_sha256'];changed+=int(mask_hash!=basehash)
                writer.write(np.hstack([panel(frame,'Original source',None,i),panel(frame,arm,mask,i)]));rendered=time.perf_counter()
                save_png(folder/f'masks/{i:06d}.png',mask)
                log.write(json.dumps({'frame_index':i,'timestamp_seconds':src['timestamp_seconds'],'decoded_pts_seconds':src['decoded_pts_seconds'],
                    'source_rgb_raw_sha256':rgb_hash,'mask_raw_sha256':mask_hash,'stone_probability_raw_sha256':sp_hash,'fiber_probability_raw_sha256':fp_hash,
                    'changed_vs_raw2':mask_hash!=basehash,'stone_pixels':int((mask==1).sum()),'fiber_pixels':int((mask==2).sum()),
                    'approved_semantic_reference':i in selected,'unlabelled_accuracy':None,'tip_metrics':None})+'\n')
                cpu_peak=max(cpu_peak,process_memory()[0]);end=time.perf_counter()
                t.update(decode_ms=(decoded-start)*1000,render_audit_encode_ms=(rendered-inferred)*1000,export_record_ms=(end-rendered)*1000,cycle_ms=(end-start)*1000)
                times.append(t);i+=1
                if i%300==0:print(f'{arm} {vid} {i}/{info["frames"]}',flush=True)
    finally:cap.release();writer.release()
    wall=time.perf_counter()-wall;cpu=time.process_time()-cpu
    if i!=info['frames'] or matches!=len(selected) or counts!=[i,i] or model_state_hash(experts)!=before:raise ValueError('Worker coverage/graph/weights mismatch')
    for handle in handles:handle.remove()
    movie=verify_video(folder/'replay.mp4',i);timings={k:stats([t[k] for t in times]) for k in times[0]}
    result={'arm':arm,'video_id':vid,'pid':os.getpid(),'frames':i,'approved_static_matches':matches,'actual_branch_forwards':counts,'warmup_branch_forwards':[warm,warm],
        'networks':2,'parameters':experts.parameter_count,'model_state_sha256':before,'individual_model_sha256':[model_hash(m) for m in experts.models],
        'runtime':{**runtime_identity(experts.device),'opencv':cv2.__version__,'opencv_threads':cv2.getNumThreads()},'source_fps':info['fps'],'timestamp_mode':source['mode'],
        'timings':timings,'pipeline_p95_within_source_interval':timings['pipeline_total_ms']['p95']<=1000/info['fps'],
        'transform_p95_within_design_target':timings['fixed_transform_ms']['p95']<=5,'changed_frames':changed,
        'wall_seconds_including_decode_audit_export':wall,'cpu_process_seconds':cpu,'offline_fps_including_decode_audit_export':i/wall,
        'cpu_sampled_peak_rss_mib':cpu_peak/2**20,'windows_peak_working_set_mib':process_memory()[1]/2**20,
        'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'cuda_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
        'replay':movie,'whole_video_accuracy':None,'tip_metrics':None,'fallback':'none; fixed input diagnostic, original default remains available'}
    atomic_json(folder/'timings.json',times);atomic_json(folder/'summary.json',result)
    print(json.dumps({'completed':arm,'video':vid,'frames':i,'p95':timings['pipeline_total_ms']['p95']}),flush=True)


def comparison(cfg,out,info):
    vid=info['video_id'];cap=cv2.VideoCapture(info['path']);h,w=info['height'],info['width'];i=0
    path=out/f'{vid}_comparison.mp4';writer=checked_writer(path,info['fps'],(w*4,(h+34)*2))
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            top=[panel(frame,'Original surgical source',None,i)];bottom=[]
            grid=rgb_grid(frame)
            for arm in ARMS:
                mask=load_mask(out/f'full_video/{arm}/{vid}/masks/{i:06d}.png',frame.shape[:2],{0,1,2})
                top.append(panel(frame,arm,mask,i))
                preview=cv2.cvtColor(np.rint(transform_rgb(grid,arm)*255).astype(np.uint8),cv2.COLOR_RGB2BGR)
                preview=cv2.resize(preview,(w,h),interpolation=cv2.INTER_NEAREST)
                bottom.append(panel(preview,'Fiber input256 '+arm,None,i))
            legend=np.zeros_like(frame)
            for j,s in enumerate(['Fiber input previews only','Red: stone; cyan: fiber','No dense whole-video GT','No confirmed tip reference','Gamma = 0.7 (fixed)']):
                cv2.putText(legend,s,(10,50+j*45),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
            bottom.insert(0,panel(legend,'Diagnostic input views',None,i))
            writer.write(np.vstack([np.hstack(top),np.hstack(bottom)]));i+=1
    finally:cap.release();writer.release()
    if i!=info['frames']:raise ValueError('Comparison source incomplete')
    return verify_video(path,i)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',nargs=2,choices=None);args=parser.parse_args()
    cfg=read(CONFIG);out=ROOT/cfg['planned_output']
    if tuple(cfg['arms'])!=ARMS or cfg['input']['gamma']!=GAMMA:raise ValueError('Frozen design differs')
    if args.worker:
        if args.worker[0] not in ARMS or args.worker[1] not in FIXED:raise ValueError('Unknown worker')
        return worker(cfg,out,*args.worker)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);job={'pid':os.getpid(),'training_updates':0,'default_changed':False}
    def progress(phase,**extra):
        job.update(phase=phase,**extra);atomic_json(out/'job.json',job);print(phase+' '+json.dumps(extra),flush=True)
    try:
        progress('preflight');protocol=read(ROOT/'runs/protocols/fiber_dark_input_control_20261030_v1/identity.json')
        if any(sha256(ROOT/k)!=v for k,v in protocol['files'].items()):raise ValueError('Frozen protocol/source differs')
        prior=read(Path(protocol['frozen_base_preflight']))
        if sha256(protocol['frozen_base_preflight'])!=protocol['frozen_base_preflight_sha256']:raise ValueError('Old preflight differs')
        if any(sha256(ROOT/k)!=v for k,v in prior['sources'].items()):raise ValueError('Old source differs')
        vm=read(ROOT/cfg['data']['video_manifest']);im=read(ROOT/cfg['data']['image_manifest'])
        if len(vm['val'])!=70 or len(im['val'])!=132:raise ValueError('Reference count differs')
        inputs=input_identity(ROOT/cfg['data']['image_manifest'],Path(im['root']),{'train':{'labeled':[],'unlabeled':[]},'val':vm['val']+im['val']})
        if inputs!=prior['inputs']:raise ValueError('Reference identity differs')
        sources={**prior['sources']}
        for p in list((ROOT/'tools').glob('*.py'))+list((ROOT/'tests').glob('*.py')):sources[p.relative_to(ROOT).as_posix()]=sha256(p)
        related={str(CONFIG):sha256(CONFIG),str(ROOT/cfg['design_document']):sha256(ROOT/cfg['design_document'])}
        previous=ROOT/cfg['base']['reference_run']
        for scope in ('video_native','image_native'):
            for arm in ('member1_2net','classwise3'):
                p=previous/f'{scope}/{arm}.json';related[str(p)]=sha256(p)
        for info in prior['videos']:
            if sha256(info['path'])!=info['sha256']:raise ValueError('Video differs')
            progress('source_timestamp_audit',video_id=info['video_id']);source_scan(out,info,previous)
            related[str(out/f'source_{info["video_id"]}.json')]=sha256(out/f'source_{info["video_id"]}.json')
            p=previous/f'full_video/member1_2net/{info["video_id"]}/frames.jsonl';related[str(p)]=sha256(p)
        for cp in prior['checkpoints'].values():
            if sha256(cp['path'])!=cp['sha256']:raise ValueError('Weight differs')
        pre={'config':cfg,'config_sha256':sha256(CONFIG),'sources':sources,'related':related,'inputs':inputs,
             'checkpoints':prior['checkpoints'],'videos':prior['videos'],'train_pixels_read':0,'reserved_test_pixels_read':0,
             'mechanism_tests_passed':10,'no_threshold_or_gamma_search':True}
        atomic_json(out/'preflight.json',pre)
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in sources:z.write(ROOT/name,name)
            z.write(CONFIG,CONFIG.relative_to(ROOT).as_posix());z.write(ROOT/cfg['design_document'],cfg['design_document'])
        for arm in ARMS:progress('static_arm_start',arm=arm);static_arm(cfg,pre,out,arm,progress)
        gates=quality(cfg,out);atomic_json(out/'static_quality.json',gates);full={}
        for arm in ARMS:
            full[arm]={}
            for info in pre['videos']:
                vid=info['video_id'];progress('full_video_fresh_process',arm=arm,video_id=vid)
                with (out/f'{arm}_{vid}.log').open('w',encoding='utf-8') as log:
                    subprocess.run([sys.executable,'-m','tools.run_fiber_dark_input_control','--worker',arm,vid],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                   check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                full[arm][vid]=read(out/f'full_video/{arm}/{vid}/summary.json')
        exports={}
        for info in pre['videos']:progress('comparison_export',video_id=info['video_id']);exports[info['video_id']]=comparison(cfg,out,info)
        if any(sha256(ROOT/k)!=v for k,v in sources.items()) or any(sha256(k)!=v for k,v in related.items()):raise ValueError('Frozen source/reference differs at completion')
        if inputs!=input_identity(ROOT/cfg['data']['image_manifest'],Path(im['root']),{'train':{'labeled':[],'unlabeled':[]},'val':vm['val']+im['val']}):raise ValueError('Reference identity changed')
        if any(sha256(x['path'])!=x['sha256'] for x in [*pre['checkpoints'].values(),*pre['videos']]):raise ValueError('Weights/video changed')
        passed={a:gates[a]['passed'] and all(v['pipeline_p95_within_source_interval'] and v['transform_p95_within_design_target'] for v in full[a].values()) for a in ARMS[1:]}
        atomic_json(out/'summary.json',{'static_quality':gates,'full_video':full,'comparison_videos':exports,'passing_dark_candidate':passed,
            'actual_static_paired_frame_inferences':606,'actual_full_paired_frame_inferences':8082,'full_processes':6,
            'warmup_paired_frames':72,'training_updates':0,'train_pixels_read':0,'reserved_test_pixels_read':0,
            'adopted':False,'default_changed':False,'tip_metrics':None,'whole_video_accuracy':None,'s3_started':False})
        progress('completed',static_frames=606,full_frames=8082)
    except Exception as error:
        progress('failed',error=str(error));(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
