"""Frozen geometry-only and causal fiber refinement, validation only."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import time
import traceback
import zipfile

import cv2
import numpy as np
from PIL import Image
import torch

from ureter_semis.experiment import atomic_json, input_identity, runtime_identity, seed_training, sha256, source_identity
from ureter_semis.video_metrics import estimated_fov
from ureter_semis.supervised_trainer import assess
from .classwise_experts import ClasswiseExperts
from .fiber_shape_prior import CausalShapeTracker, fit_shape, visible_fiber, combine
from .replay_classwise_experts import pil_input, intervals, mask_diagnostics, summarize_diagnostics, stats, save_png, model_state_hash
from .run_classwise_expert_control import summarize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/a1_fiber_geometry_20261028'
NAMES = ('raw', 'geometry', 'geometry_causal')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


@torch.inference_mode()
def core(experts, frame):
    experts.sync(); started = time.perf_counter()
    image = pil_input(frame, experts.size, experts.device)
    result = experts.predict_native(image, [frame.shape[:2]])[0]
    raw = result['mask'].cpu().numpy().copy(); stone = result['stone_mask'].cpu().numpy().copy()
    a0 = result['fiber_mask'].cpu().numpy().copy(); probability = result['fiber_probability'][2].cpu().numpy().copy()
    experts.sync()
    return raw, stone, a0, probability, (time.perf_counter() - started) * 1000


def refine(frame, raw, stone, probability, protocol, tracker=None, dt=1/30):
    start = time.perf_counter(); fov = estimated_fov(frame, raw.shape)
    fit = fit_shape(frame, probability, raw == 2, fov, protocol['fit'])
    fitted = time.perf_counter()
    state = fit['state'] if fit['accepted'] else None
    masks = {'raw': raw, 'geometry': combine(stone, visible_fiber(probability, raw == 2, state, protocol['fit']['visible_probability_min']))}
    static = time.perf_counter()
    tracked = None
    if tracker is not None:
        tracked = tracker.step(state, raw.shape, dt)
        current = tracked['state'] if tracked['apply'] else None
        masks['geometry_causal'] = combine(stone, visible_fiber(probability, raw == 2, current, protocol['fit']['visible_probability_min']))
    finished = time.perf_counter()
    timing = {'fit_and_fov_ms': (fitted - start) * 1000, 'geometry_raster_fusion_ms': (static - fitted) * 1000,
              'causal_update_raster_fusion_ms': (finished - static) * 1000 if tracker is not None else None}
    deltas = {}
    for name, mask in masks.items():
        if name == 'raw': continue
        deltas[name] = {'fiber_added': int(((mask == 2) & (raw != 2)).sum()), 'fiber_removed': int(((raw == 2) & (mask != 2)).sum()),
                        'stone_overwritten': int(((raw == 1) & (mask != 1)).sum()), 'stone_restored': int(((raw != 1) & (mask == 1)).sum())}
    return masks, fov, {'fit': fit, 'tracked': tracked, 'deltas': deltas}, timing


def render(frame, masks, title):
    panels = []
    for name in ('Input', *NAMES):
        panel = frame.copy()
        if name != 'Input':
            for k, color in [(1, (40,40,235)), (2,(235,210,20))]:
                selected = masks[name] == k
                panel[selected] = (.48 * frame[selected] + .52 * np.asarray(color)).astype(np.uint8)
        header = np.zeros((34, frame.shape[1], 3), np.uint8)
        cv2.putText(header, f'{name} | {title}', (5,21), cv2.FONT_HERSHEY_SIMPLEX, .42, (255,255,255), 1, cv2.LINE_AA)
        panels.append(np.vstack([header, panel]))
    return np.hstack(panels)


def evaluate_images(rows, experts, protocol, exposed):
    folder = OUT / 'image_native'; folder.mkdir()
    for n in ('raw', 'geometry'): (folder / n).mkdir()
    targets = []; predictions = {n: [] for n in ('raw', 'geometry')}; records = []
    for i, row in enumerate(rows):
        rgb = np.asarray(Image.open(row['image']).convert('RGB')); frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        raw, stone, a0, probability, core_ms = core(experts, frame)
        masks, _, record, timing = refine(frame, raw, stone, probability, protocol)
        record.update(sample_id=row['sample_id'], timing={'raw_core_ms': core_ms, **timing})
        records.append(record); targets.append(np.asarray(Image.open(row['mask'])).copy())
        for n, a in masks.items():
            predictions[n].append(a); save_png(folder / n / f'{i:04d}.png', a)
    result = {'models': {n: summarize(rows, targets, a) for n, a in predictions.items()}, 'records': records, 'cohorts': {}, 'sources': {}}
    for name, seen in [('historical_old190_exposed', True), ('not_matched_to_old190_exposure', False)]:
        ids = [i for i, row in enumerate(rows) if (row['group_id'] in exposed) == seen]
        result['cohorts'][name] = {'images': len(ids), 'sample_ids': [rows[i]['sample_id'] for i in ids],
            'models': {n: summarize([rows[i] for i in ids], [targets[i] for i in ids], [p[i] for i in ids])['metrics'] for n,p in predictions.items()}}
    for source in sorted({r['source_dataset'] for r in rows}):
        ids = [i for i,r in enumerate(rows) if r['source_dataset'] == source]
        result['sources'][source] = {n: summarize([rows[i] for i in ids], [targets[i] for i in ids], [p[i] for i in ids])['metrics'] for n,p in predictions.items()}
    result['accepted_images'] = sum(r['fit']['accepted'] for r in records)
    result['fit_reasons'] = dict(Counter(r['fit']['reason'] for r in records))
    atomic_json(folder / 'comparison.json', result)
    return result


def run_video(item, experts, protocol, replay_protocol, reviewed, progress):
    vid = item['video_id']; folder = OUT / vid; folder.mkdir()
    for n in NAMES: (folder / n).mkdir()
    (folder / 'clips').mkdir(); (folder / 'keyframes').mkdir()
    cap = cv2.VideoCapture(item['path']); writers = []
    if not cap.isOpened(): raise RuntimeError('Cannot open video')
    fps = float(cap.get(cv2.CAP_PROP_FPS)); width = int(cap.get(3)); height = int(cap.get(4)); count = item['frames']
    events = intervals(replay_protocol, vid, count); tracker = CausalShapeTracker(protocol['tracking'])
    def writer(path):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 4, height + 34))
        if not w.isOpened(): raise RuntimeError('Cannot open preview writer')
        writers.append(w); return w
    try:
        ok, frame = cap.read()
        if not ok: raise ValueError('Empty video')
        for _ in range(12): core(experts, frame)
        a = core(experts, frame); b = core(experts, frame)
        if not all(np.array_equal(a[i], b[i]) for i in range(4)): raise ValueError('Expert repeat failure')
        fa = fit_shape(frame, a[3], a[0] == 2, estimated_fov(frame, frame.shape[:2]), protocol['fit'])
        fb = fit_shape(frame, a[3], a[0] == 2, estimated_fov(frame, frame.shape[:2]), protocol['fit'])
        if fa != fb: raise ValueError('Geometry repeat failure')
        del a,b
        cap.release(); cap = cv2.VideoCapture(item['path'])
        preview = writer(folder / 'comparison.mp4')
        clip_writers = []
        for i,e in enumerate(events):
            e['clip'] = f'clips/{i:02d}_{e["event"]}_{e["one_based_inclusive"][0]}-{e["one_based_inclusive"][1]}.mp4'
            clip_writers.append(writer(folder / e['clip']))
        selected = {r['frame_index']: r for r in reviewed if r['video_id'] == vid}
        previous = {n: None for n in NAMES}; logs = []; timings = []; alignment = []; selected_masks = {}
        cache_diff = {'classwise': 0, 'old190': 0, 'A0': 0}
        torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
        with (folder / 'frames.jsonl').open('w', encoding='utf-8') as log:
            while True:
                t0 = time.perf_counter(); ok, frame = cap.read(); decoded = time.perf_counter()
                if not ok: break
                i = len(logs)
                if i >= count or frame.shape != (height,width,3): raise ValueError('Unexpected geometry/count')
                raw, stone, a0, probability, core_ms = core(experts, frame)
                masks, fov, record, timing = refine(frame, raw, stone, probability, protocol, tracker, 1/fps)
                timing.update(raw_core_ms=core_ms, geometry_derived_ms=core_ms+timing['fit_and_fov_ms']+timing['geometry_raster_fusion_ms'],
                    geometry_causal_derived_ms=core_ms+timing['fit_and_fov_ms']+timing['causal_update_raster_fusion_ms'], decode_ms=(decoded-t0)*1000)
                diagnostics_start = time.perf_counter()
                diagnostics = {n: mask_diagnostics(mask, fov, previous[n]) for n, mask in masks.items()}
                for n, mask in [('classwise',raw),('old190',stone),('A0',a0)]:
                    cached = np.asarray(Image.open(ROOT / f'runs/a1_classwise_replay_20261028/{vid}/{n}/{i:06d}.png'))
                    cache_diff[n] += int((cached != mask).sum())
                if i in selected:
                    row = selected[i]; original = np.asarray(Image.open(row['image']).convert('RGB'))
                    exact = np.array_equal(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB), original)
                    alignment.append({'sample_id':row['sample_id'],'frame_index':i,'pixel_exact':exact})
                    if not exact: raise ValueError('Keyframe decode alignment differs')
                    selected_masks[row['sample_id']] = {n: mask.copy() for n,mask in masks.items()}
                diagnosed = time.perf_counter()
                for n,mask in masks.items(): save_png(folder / n / f'{i:06d}.png',mask)
                canvas = render(frame,masks,f'{vid} frame{i+1}'); preview.write(canvas)
                for event,w in zip(events,clip_writers):
                    if event['start_frame'] <= i < event['end_frame_exclusive']: w.write(canvas)
                if i in selected: save_png(folder/'keyframes'/f'{i:06d}.png',canvas)
                exported = time.perf_counter()
                record.update(frame_index=i,one_based_frame=i+1,timestamp_s=i/fps,models=diagnostics,
                    fit_state_is_visible_truth=False, dense_ground_truth_available=False)
                log.write(json.dumps(record,allow_nan=False)+'\n'); log.flush()
                timing.update(diagnostic_and_cache_check_ms=(diagnosed-diagnostics_start)*1000, render_and_exports_ms=(exported-diagnosed)*1000,
                    actual_all_branch_cycle_ms=(time.perf_counter()-t0)*1000)
                logs.append(record); timings.append(timing); previous=masks
                if (i+1)%200==0:
                    progress('replaying',video_id=vid,frames_completed=i+1,expected_frames=count)
                    print(f'{vid}: {i+1}/{count}',flush=True)
        for w in writers: w.release()
        writers.clear(); elapsed = time.perf_counter()-started
        if len(logs)!=count or len(alignment)!=len(selected): raise ValueError('Final frame count differs')
        tracker.reset()
        result={'video_id':vid,'frames':count,'fps':fps,'repeat_exact':True,'tracker_reset_between_cases':tracker.state is None,
            'cache_different_pixels':cache_diff,'reviewed_alignment':alignment,
            'accepted_fits':sum(r['fit']['accepted'] for r in logs),'fit_reasons':dict(Counter(r['fit']['reason'] for r in logs)),
            'tracker_statuses':dict(Counter(r['tracked']['status'] for r in logs)),
            'models':{n:summarize_diagnostics([{'frame_index':r['frame_index'],**r['models'][n]} for r in logs],fps) for n in NAMES},
            'timings':{k:stats([r[k] for r in timings]) for k in timings[0]},'offline_all_branches_fps':count/elapsed,
            'wall_seconds_including_writer_flush':elapsed,'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'source_frame_budget_ms':1000/fps,'user_intervals':events,'dense_accuracy_metrics':None,
            'limitations':'Paired per-arm derived cost includes only required stages of shared-core replay; real cycle includes all branches and audit exports. No live display,3D geometry,tip truth,stone identity or calibrated uncertainty'}
        for event in events:
            part=logs[event['start_frame']:event['end_frame_exclusive']]
            event['models']={n:summarize_diagnostics([{'frame_index':r['frame_index'],**r['models'][n]} for r in part],fps) for n in NAMES}
        atomic_json(folder/'timings.json',timings);atomic_json(folder/'summary.json',result)
        return result,selected_masks
    finally:
        cap.release()
        for w in writers:w.release()


def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir();job={'pid':os.getpid(),'training_updates':0,'default_model_changed':False,'s2_accepted':False,'stone_memory_started':False}
    def progress(phase,**extra):job.update(phase=phase,**extra);atomic_json(OUT/'job.json',job)
    progress('preflight')
    try:
        config_path=ROOT/'configs/fiber_geometry_prior_20261028.json';protocol=read(config_path)
        replay_path=ROOT/'configs/replay_classwise_experts_20261028.json';replay_protocol=read(replay_path)
        source_path=ROOT/'runs/baselines/a0_20261027/config.json';inputs=read(source_path)['inputs']
        predictor=read(ROOT/protocol['experts']);checkpoints={n:predictor['checkpoints'][n] for n in ('old190','A0')}
        for n,key in [('old190','stone_sha256'),('A0','fiber_sha256')]:
            if sha256(Path(checkpoints[n]['path']))!=protocol[key]:raise ValueError('Expert SHA differs')
        for item in inputs:
            if sha256(Path(item['path']))!=item['sha256']:raise ValueError('Video SHA differs')
        manifest_path=ROOT/'datasets/ureter_semis_kidney_joint_20261028/manifest.json';m=read(manifest_path)
        vm_path=manifest_path.parent/'video_validation_manifest.json';vm=read(vm_path)
        identity=input_identity(manifest_path,Path(m['root']),{'train':{'labeled':[],'unlabeled':[]},'val':m['val']+vm['val']})
        related=[config_path,replay_path,source_path,ROOT/protocol['experts'],vm_path,ROOT/'runs/a1_classwise_experts_20261028/summary.json']
        for item in inputs:
            for n in ('A0','old190','classwise'):related+=sorted((ROOT/f'runs/a1_classwise_replay_20261028/{item["video_id"]}/{n}').glob('*.png'))
        sources=source_identity()
        extra=['tools/run_fiber_shape_control.py','tools/fiber_shape_prior.py','tests/test_fiber_shape_prior.py','tools/replay_classwise_experts.py','tools/classwise_experts.py','tools/run_classwise_expert_control.py','tools/diagnose_joint_failures.py','tools/evaluate_native_checkpoints.py','tools/evaluate_spatial_detail.py']
        sources.update({n:sha256(ROOT/n) for n in extra})
        preflight={'protocol':protocol,'checkpoints':checkpoints,'video_inputs':inputs,'inputs':identity,'sources':sources,
            'related':{str(p):sha256(p) for p in related},'training_updates':0,'reserved_test_pixels_read':False}
        atomic_json(OUT/'preflight.json',preflight)
        with zipfile.ZipFile(OUT/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in sources:z.write(ROOT/name,name.replace('\\','/'))
            z.write(config_path,str(config_path.relative_to(ROOT)))
        seed_training(42);experts=ClasswiseExperts(checkpoints['old190']['path'],checkpoints['A0']['path'])
        if experts.model_count!=3 or experts.parameter_count!=73804185:raise ValueError('Model count differs')
        before=model_state_hash(experts);atomic_json(OUT/'runtime.json',runtime_identity(experts.device))
        cohorts=read(ROOT/'runs/a1_classwise_experts_20261028/summary.json')['image']['historical_exposure_cohorts']
        exposed_ids=set(cohorts['historical_old190_exposed']['sample_ids']);exposed={r['group_id'] for r in m['val'] if r['sample_id'] in exposed_ids}
        progress('image132_stateless');image=evaluate_images(m['val'],experts,protocol,exposed)
        videos={};selected={}
        for item in inputs:
            progress('loading_video',video_id=item['video_id'],frames_completed=0,expected_frames=item['frames'])
            result,frames=run_video(item,experts,protocol,replay_protocol,vm['val'],progress)
            videos[item['video_id']]=result;selected.update(frames)
        progress('reviewed29_evaluation');targets=[np.asarray(Image.open(r['mask'])).copy() for r in vm['val']]
        video={n:summarize(vm['val'],targets,[selected[r['sample_id']][n] for r in vm['val']]) for n in NAMES}
        changes={}
        for name in ('geometry','geometry_causal'):
            changes[name]={}
            for k,label in [(1,'stone'),(2,'fiber')]:
                total=Counter()
                for row,target in zip(vm['val'],targets):
                    a=selected[row['sample_id']]['raw'];b=selected[row['sample_id']][name];valid=target!=255;truth=target==k
                    total.update(lost_tp=int(((a==k)&(b!=k)&truth).sum()),gained_tp=int(((a!=k)&(b==k)&truth).sum()),
                        added_fp=int(((a!=k)&(b==k)&~truth&valid).sum()),removed_fp=int(((a==k)&(b!=k)&~truth&valid).sum()))
                changes[name][label]=dict(total)
        assert model_state_hash(experts)==before and all(not model.training for model in experts.models)
        assert input_identity(manifest_path,Path(m['root']),{'train':{'labeled':[],'unlabeled':[]},'val':m['val']+vm['val']})==identity
        assert all(sha256(ROOT/n)==h for n,h in sources.items()) and all(sha256(Path(n))==h for n,h in preflight['related'].items())
        assert all(sha256(Path(v['path']))==v['sha256'] for v in checkpoints.values()) and all(sha256(Path(v['path']))==v['sha256'] for v in inputs)
        result={'image':image,'video29':video,'videos':videos,'video29_paired_changes':changes,
            'screening':{n:assess(v['metrics'],protocol['gates']) for n,v in video.items()},
            'frames_total':sum(v['frames'] for v in videos.values()),'model_count':3,'parameter_count':73804185,
            'model_state_sha256_unchanged':before,'training_updates':0,'default_model_changed':False,'s2_accepted':False,
            'reserved_test_pixels_read':False,'stone_memory_started':False,'tip_and_angle_reference_available':False}
        atomic_json(OUT/'summary.json',result)
        progress('completed',frames_completed=result['frames_total'],expected_frames=2694)
        print(json.dumps({'phase':'completed','video':{n:r['metrics'] for n,r in video.items()},'image_accepted':image['accepted_images'],
            'video_accepted':{v:r['accepted_fits'] for v,r in videos.items()},'screening':result['screening']},indent=2),flush=True)
    except Exception as error:
        progress('failed',error=str(error));(OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
