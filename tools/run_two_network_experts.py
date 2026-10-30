"""Fixed existing-member experiment: fresh static inference and six complete-video workers."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
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

from ureter_semis.experiment import atomic_json, input_identity, runtime_identity, seed_training, source_identity
from ureter_semis.supervised_trainer import assess
from .classwise_experts import ClasswiseExperts, native_outputs
from .member_experts import MemberExperts
from .fiber_reference_workspace import ROOT, read, sha256, load_mask
from .run_fiber_blue_aplus import cached_paths, frame_read, panel, comparison_video
from .replay_classwise_experts import pil_input, save_png, stats, model_state_hash
from .run_classwise_expert_control import summarize
from .diagnose_joint_failures import paired_changes

CONFIG = ROOT / 'configs/compare_two_network_experts_20261029_v2.json'


def array_hash(array): return hashlib.sha256(array.tobytes()).hexdigest()


def model_hash(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def construct(config, cp, name):
    arm = config['models'][name]
    if arm['fiber_member'] is None: experts = ClasswiseExperts(cp['old190']['path'], cp['A0']['path'])
    else: experts = MemberExperts(cp['old190']['path'], cp['A0']['path'], arm['fiber_member'])
    if experts.model_count != arm['networks'] or experts.parameter_count != arm['parameters']:
        raise ValueError('Actual constructed model budget differs from frozen arm')
    if experts.size != 256 or any(m.training for m in experts.models): raise ValueError('Frozen inference mode differs')
    return experts


@torch.inference_mode()
def infer(experts, frame):
    """Same native mask and both native3-class maps transferred in every arm."""
    experts.sync(); start = time.perf_counter()
    tensor = pil_input(frame, experts.size, experts.device)
    experts.sync(); prepared = time.perf_counter()
    sp, fp = experts.probabilities(tensor)
    experts.sync(); forwarded = time.perf_counter()
    result = native_outputs(sp[0], fp[0], frame.shape[:2])
    mask = result['mask'].cpu().numpy().copy()
    stone = result['stone_probability'].cpu().numpy().copy()
    fiber = result['fiber_probability'].cpu().numpy().copy()
    experts.sync(); end = time.perf_counter()
    return mask, stone, fiber, {'preprocess_ms': (prepared-start)*1000, 'all_models_ms': (forwarded-prepared)*1000,
        'native_mapping_fusion_transfer_ms': (end-forwarded)*1000, 'pipeline_ms': (end-start)*1000}


def cache_for(row, index, video, old_indices):
    path, raw = cached_paths(row, index, video, old_indices)
    if path is None: path = ROOT / f'runs/a1_fiber_blue_aplus_20261029/video_native/new_probabilities/{index:04d}.npz'
    return path, raw


def static_arm(config, experts, name, rows, video, out, old_indices, progress):
    folder = out / ('video_native' if video else 'image_native')
    (folder / name).mkdir(parents=True)
    targets = []; predictions = []; baselines = []; records = []; expected_key = 'a0_native' if name == 'classwise3' else 'a0_member' + str(config['models'][name]['fiber_member'][-1]) + '_native'
    max_error = 0.
    for i, row in enumerate(rows):
        frame = frame_read(row['image']); target = load_mask(row['mask'], frame.shape[:2], {0,1,2,255})
        pred, stone, fiber, _ = infer(experts, frame)
        path, raw_path = cache_for(row, i, video, old_indices)
        with np.load(path) as cache:
            old_sp = cache['old190_native']; old_fp = cache[expected_key]
        error = max(float(np.abs(stone-old_sp).max()), float(np.abs(fiber-old_fp).max())); max_error = max(max_error, error)
        expected = np.zeros(pred.shape, np.uint8); expected[old_sp.argmax(0)==1] = 1; expected[old_fp.argmax(0)==2] = 2
        baseline = load_mask(raw_path, pred.shape, {0,1,2})
        if error > 2e-6 or not np.array_equal(pred, expected): raise ValueError('Fresh model differs from frozen individual-member probability reference')
        if name == 'classwise3' and not np.array_equal(pred, baseline): raise ValueError('Three-network baseline reproduction differs')
        if any(a.dtype != np.float32 or not np.isfinite(a).all() or not np.allclose(a.sum(0), 1, atol=2e-5) for a in (stone, fiber)):
            raise ValueError('Invalid native FP32 maps')
        save_png(folder / name / f'{i:04d}.png', pred)
        records.append({'sample_id': row['sample_id'], 'video_id': row.get('video_id'), 'frame_index': row.get('frame_index'),
            'probability_reference': str(path), 'stone_probability_raw_sha256': array_hash(stone), 'fiber_probability_raw_sha256': array_hash(fiber),
            'native_probability_max_absolute_error_vs_reference': error, 'mask_raw_sha256': array_hash(pred),
            'paired_stone': paired_changes(target, baseline, pred, 1), 'paired_fiber': paired_changes(target, baseline, pred, 2),
            'tip_metrics': None})
        targets.append(target); predictions.append(pred); baselines.append(baseline)
        if (i+1)%30 == 0: progress('static_' + ('video' if video else 'image'), scheme=name, samples_completed=i+1, samples_expected=len(rows))
    groups = {}
    if video:
        indices = {v:[i for i,r in enumerate(rows) if r['video_id']==v] for v in ('video_1','video_2')}
        indices.update({cohort:[i for i,r in enumerate(rows) if r['evaluation_cohort']==cohort] for cohort in ('original29','new_user_approved')})
    else:
        exposed = set(read(ROOT/'runs/a1_classwise_experts_20261028/summary.json')['image']['historical_exposure_cohorts']['historical_old190_exposed']['sample_ids'])
        indices = {k:[i for i,r in enumerate(rows) if r['source_dataset']==k] for k in sorted({r['source_dataset'] for r in rows})}
        indices.update({k:[i for i,r in enumerate(rows) if (r['sample_id'] in exposed)==flag] for k,flag in [('old190_exposed25',True),('not_matched107',False)]})
    for group, ids in indices.items(): groups[group] = summarize([rows[i] for i in ids], [targets[i] for i in ids], [predictions[i] for i in ids])
    result = {'summary': summarize(rows,targets,predictions), 'groups':groups, 'records':records,
        'network_count':experts.model_count, 'parameter_count':experts.parameter_count, 'fresh_inferences':len(rows),
        'max_native_probability_error_vs_reference':max_error, 'expected_member_mask_different_pixels':0,
        'paired_changes':{c:{k:sum(r['paired_'+c][k] for r in records) for k in ('lost_tp','gained_tp','added_fp','removed_fp')} for c in ('stone','fiber')},
        'tip_metrics':None}
    atomic_json(folder / f'{name}.json', result)
    return result


def screening(config, video, image):
    result = {}; epsilon = config['recommendation_checks']['epsilon']
    for name in config['schemes'][1:]:
        legacy = assess(video[name]['groups']['original29']['metrics'], config['legacy_original29_gates'])
        checks = {}
        for label, data in [('video70',video),('image132',image)]:
            reference = data['classwise3']['summary']; candidate = data[name]['summary']
            for metric in config['recommendation_checks']['metrics']:
                if metric.endswith('_boundary_f'):
                    c = '1' if metric.startswith('stone') else '2'; a = reference['boundaries'][c]['f_score']; b = candidate['boundaries'][c]['f_score']
                else: a = reference['metrics'][metric]; b = candidate['metrics'][metric]
                checks[label+'_'+metric] = a is not None and b is not None and b >= a-epsilon
        result[name] = {'legacy_original29_screening':legacy, 'current_nonregression_checks':checks,
            'current_nonregression_passed':all(checks.values()), 'rejected_checks':[k for k,v in checks.items() if not v],
            'quality_recommendation_passed':legacy['passed'] and all(checks.values())}
    return result


def video_worker(config, out, name, vid):
    pre = read(out/'preflight.json'); info = next(v for v in pre['videos'] if v['video_id']==vid)
    for item in [info,*pre['checkpoints'].values()]:
        if sha256(item['path'])!=item['sha256']: raise ValueError('Frozen worker input changed')
    folder = out/f'full_video/{name}/{vid}'; folder.mkdir(parents=True); (folder/'masks').mkdir()
    seed_training(42); experts = construct(config,pre['checkpoints'],name); before = model_state_hash(experts)
    individual = [model_hash(m) for m in experts.models]
    atomic_json(folder/'runtime.json', {'pid':os.getpid(), 'individual_model_sha256':individual, **runtime_identity(experts.device)})
    rows = read(ROOT/config['video_manifest'])['val']; selected = {r['frame_index']:r for r in rows if r['video_id']==vid}
    static_indices = {(r['video_id'],r['frame_index']):i for i,r in enumerate(rows)}
    source_log = [__import__('json').loads(s) for s in (ROOT/f'runs/a1_fiber_tip_d0_20261029/full_video/{vid}/frames.jsonl').read_text().splitlines()]
    cap = cv2.VideoCapture(info['path']); ok, first = cap.read()
    if not ok: raise ValueError('Empty video')
    for _ in range(config['full_video']['warmup_frames']): infer(experts,first)
    cap.release(); cap = cv2.VideoCapture(info['path']); fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(str(folder/'replay.mp4'),cv2.CAP_MSMF,cv2.VideoWriter_fourcc(*'H264'),fps,(800,434))
    if not writer.isOpened(): raise RuntimeError('H264 encoder unavailable')
    timing=[]; i=0; static_matches=0; baseline_matches=0; changed=0; tp_lost_total_not_known=0
    torch.cuda.reset_peak_memory_stats(); wall_start=time.perf_counter()
    try:
        with (folder/'frames.jsonl').open('w',encoding='utf-8') as log:
            while True:
                start=time.perf_counter(); ok,frame=cap.read(); decoded=time.perf_counter()
                if not ok: break
                mask,sp,fp,t=infer(experts,frame); inferred=time.perf_counter()
                writer.write(np.hstack([panel(frame,'Input',None,i),panel(frame,name,mask,i)])); rendered=time.perf_counter()
                rgb_hash=array_hash(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); mask_hash=array_hash(mask)
                if source_log[i]['frame_index']!=i or source_log[i]['rgb_raw_bytes_sha256']!=rgb_hash: raise ValueError('Sequential source RGB differs')
                if name=='classwise3':
                    if source_log[i]['mask_raw_bytes_sha256']!=mask_hash: raise ValueError('Full three-network reference differs from history')
                    baseline_matches+=1
                else: changed+=int(source_log[i]['mask_raw_bytes_sha256']!=mask_hash)
                if i in selected:
                    if selected[i]['rgb_raw_bytes_sha256']!=rgb_hash: raise ValueError('Approved reference RGB differs')
                    static=load_mask(out/f'video_native/{name}/{static_indices[(vid,i)]:04d}.png',mask.shape,{0,1,2})
                    if not np.array_equal(mask,static): raise ValueError('Static and sequential predictions differ')
                    static_matches+=1
                save_png(folder/'masks'/f'{i:06d}.png',mask)
                record={'frame_index':i,'source_rgb_raw_sha256':rgb_hash,'mask_raw_sha256':mask_hash,
                    'stone_pixels':int((mask==1).sum()),'fiber_pixels':int((mask==2).sum()),
                    'approved_semantic_reference':i in selected,'unlabelled_accuracy':None,'tip_metrics':None}
                log.write(__import__('json').dumps(record)+'\n'); end=time.perf_counter()
                t.update(decode_ms=(decoded-start)*1000,render_encode_ms=(rendered-inferred)*1000,audit_export_ms=(end-rendered)*1000,cycle_ms=(end-start)*1000)
                timing.append(t); i+=1
                if i%250==0: print(f'{name} {vid} {i}/{info["frames"]}',flush=True)
    finally: cap.release(); writer.release()
    wall=time.perf_counter()-wall_start
    if i!=info['frames'] or static_matches!=len(selected): raise ValueError('Incomplete worker coverage')
    if model_state_hash(experts)!=before: raise ValueError('Weights changed')
    check=cv2.VideoCapture(str(folder/'replay.mp4')); encoded=0
    while check.read()[0]: encoded+=1
    check.release()
    if encoded!=i: raise ValueError('Incomplete H264 replay')
    result={'scheme':name,'video_id':vid,'pid':os.getpid(),'networks':experts.model_count,'parameters':experts.parameter_count,
        'individual_model_sha256':individual,'model_state_sha256':before,'frames':i,'encoded_frames':encoded,'fps':fps,
        'warmup_frames':config['full_video']['warmup_frames'],'approved_static_matches':static_matches,
        'baseline_history_matches':baseline_matches,'changed_frames_vs_classwise3':changed,
        'timings':{k:stats([t[k] for t in timing]) for k in timing[0]},'offline_fps_including_all_exports':i/wall,
        'wall_seconds':wall,'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'realtime_p95_passed':stats([t['pipeline_ms'] for t in timing])['p95']<=1000/fps,
        'replay_sha256':sha256(folder/'replay.mp4'),'full_video_accuracy':None,'tip_metrics':None}
    atomic_json(folder/'timings.json',timing); atomic_json(folder/'summary.json',result)
    print(__import__('json').dumps({'completed':name,'video':vid,'frames':i,'p95':result['timings']['pipeline_ms']['p95']}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--worker',nargs=2,metavar=('SCHEME','VIDEO')); args=parser.parse_args()
    config=read(CONFIG); out=ROOT/config['output']
    if args.worker: return video_worker(config,out,*args.worker)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(); job={'pid':os.getpid(),'training_updates':0,'default_changed':False,'tip_metrics':None}
    def progress(phase,**extra):
        job.update(phase=phase,**extra); atomic_json(out/'job.json',job); print(phase+' '+__import__('json').dumps(extra),flush=True)
    try:
        progress('preflight')
        vm_path=ROOT/config['video_manifest']; vm=read(vm_path); im_path=ROOT/config['image_manifest']; im=read(im_path)
        if len(vm['val'])!=70 or len(im['val'])!=132: raise ValueError('Frozen sample scope changed')
        original=read(ROOT/'datasets/ureter_semis_kidney_joint_20261028/video_validation_manifest.json')['val']
        old_indices={(r['video_id'],r['frame_index']):i for i,r in enumerate(original)}
        cp={k:v for k,v in read(ROOT/config['experts'])['checkpoints'].items() if k in {'old190','A0'}}
        videos=read(ROOT/'runs/baselines/a0_20261027/config.json')['inputs']
        for x in [*cp.values(),*videos]:
            if sha256(x['path'])!=x['sha256']: raise ValueError('Frozen checkpoint or video changed')
        original_protocol=read(ROOT/config['old_protocol'])
        if original_protocol['gates']!=config['legacy_original29_gates']: raise ValueError('Legacy gates changed')
        identity=input_identity(im_path,Path(im['root']),{'train':{'labeled':[],'unlabeled':[]},'val':vm['val']+im['val']})
        extra=[CONFIG,vm_path,ROOT/config['old_protocol'],ROOT/config['experts']]
        for video,rows in [(True,vm['val']),(False,im['val'])]:
            for i,row in enumerate(rows): extra.extend(cache_for(row,i,video,old_indices))
        related={str(p):sha256(p) for p in extra}
        sources={Path(k).as_posix():v for k,v in source_identity().items()}
        for p in list((ROOT/'tools').glob('*.py'))+list((ROOT/'tests').glob('*.py')): sources[p.relative_to(ROOT).as_posix()]=sha256(p)
        pre={'config':config,'config_sha256':sha256(CONFIG),'sources':sources,'related':related,'inputs':identity,
             'checkpoints':cp,'videos':videos,'train_pixels_read':0,'reserved_test_pixels_read':0}
        atomic_json(out/'preflight.json',pre)
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for name in sources: archive.write(ROOT/name,name)
            archive.write(CONFIG,CONFIG.relative_to(ROOT).as_posix())
        static_video={}; static_image={}; model_identities={}
        for name in config['schemes']:
            seed_training(42); experts=construct(config,cp,name); before=model_state_hash(experts)
            model_identities[name]={'individual_model_sha256':[model_hash(m) for m in experts.models],
                'networks':experts.model_count,'parameters':experts.parameter_count}
            atomic_json(out/f'runtime_static_{name}.json',runtime_identity(experts.device))
            progress('static_arm',scheme=name)
            static_video[name]=static_arm(config,experts,name,vm['val'],True,out,old_indices,progress)
            static_image[name]=static_arm(config,experts,name,im['val'],False,out,old_indices,progress)
            if before!=model_state_hash(experts): raise ValueError('Static inference changed weights')
            del experts; torch.cuda.empty_cache()
        atomic_json(out/'model_identities.json',model_identities)
        # Same-interface baseline metrics must reproduce the full earlier summaries, including all grouping details.
        for scope,rows in [('video',vm['val']),('image',im['val'])]:
            old=read(ROOT/f'runs/a1_fiber_blue_aplus_20261029/{scope}_native/summary.json')['models']['raw']
            current=(static_video if scope=='video' else static_image)['classwise3']['summary']
            if old!=current: raise ValueError('Three-network static reference metrics differ')
        quality=screening(config,static_video,static_image); atomic_json(out/'static_quality.json',quality)
        full={}
        for name in config['schemes']:
            full[name]={}
            for info in videos:
                vid=info['video_id']; progress('full_video_fresh_process',scheme=name,video_id=vid)
                with (out/f'{name}_{vid}.log').open('w',encoding='utf-8') as log:
                    subprocess.run([sys.executable,'-m','tools.run_two_network_experts','--worker',name,vid],cwd=ROOT,
                        stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                full[name][vid]=read(out/f'full_video/{name}/{vid}/summary.json')
        exports={}
        for info in videos:
            progress('comparison_video_export',video_id=info['video_id']); exports[info['video_id']]=comparison_video(out,config,info)
        if any(sha256(ROOT/k)!=v for k,v in sources.items()) or any(sha256(k)!=v for k,v in related.items()): raise ValueError('Frozen sources/inputs changed')
        if input_identity(im_path,Path(im['root']),{'train':{'labeled':[],'unlabeled':[]},'val':vm['val']+im['val']})!=identity: raise ValueError('Validation references changed')
        recommendation={name:quality[name]['quality_recommendation_passed'] and all(x['realtime_p95_passed'] for x in full[name].values()) for name in config['schemes'][1:]}
        result={'static_quality':quality,'replacement_recommendation_passed':recommendation,'full_video':full,
            'comparison_videos':exports,'models':model_identities,'static_video_frames':70,'static_image_frames':132,
            'actual_static_forward_frames':606,'full_frames_per_scheme':2694,'full_frames_all_schemes':8082,
            'default_changed':False,'adopted':False,'training_updates':0,'train_pixels_read':0,'reserved_test_pixels_read':0,
            'tip_metrics':None,'whole_video_accuracy_estimates':None,'stone_memory_started':False}
        atomic_json(out/'summary.json',result); progress('completed',static_frames_per_arm=202,full_frames_all_schemes=8082)
    except Exception as error:
        progress('failed',error=str(error)); (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8'); raise


if __name__=='__main__': main()
