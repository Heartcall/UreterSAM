"""Frozen static screening followed by conditional full causal video replay."""
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import zipfile
import cv2
import numpy as np
from PIL import Image
import torch
from ureter_semis.experiment import atomic_json, input_identity, seed_training, sha256, source_identity, runtime_identity
from ureter_semis.supervised_trainer import assess
from .classwise_experts import ClasswiseExperts
from .direct_neural_observation import IMAGE, NEURAL, locate
from .run_fiber_shape_control import core
from .run_classwise_expert_control import summarize
from .run_direct_fiber_template import pair_summary
from .replay_classwise_experts import save_png, model_state_hash

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'configs/run_direct_neural_observation_20261029.json'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def load(p):
    with Image.open(p) as im: return np.asarray(im).copy()


def evaluate(rows, experts, prior, config, out, video=False, progress=None):
    names = ('raw_classwise', IMAGE, NEURAL); out.mkdir()
    for n in (*names, 'probabilities', 'hypotheses_image', 'hypotheses_neural'): (out/n).mkdir()
    targets = []; predictions = {n:[] for n in names}; records = []; equality = Counter()
    for i, row in enumerate(rows):
        rgb = np.asarray(Image.open(row['image']).convert('RGB')); frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        raw, stone, _, p, core_ms = core(experts, frame)
        outputs, timing = locate(frame, prior, config, p)
        masks = {'raw_classwise':raw, **{n:np.where(v['fiber'],2,np.where(stone==1,1,0)).astype(np.uint8) for n,v in outputs.items()}}
        # Persist predictions before loading reference masks. GT never enters localization.
        for n,a in masks.items(): save_png(out/n/f'{i:04d}.png', a); predictions[n].append(a)
        np.savez_compressed(out/'probabilities'/f'{i:04d}.npz', fiber_probability=p)
        for n,label in [(IMAGE,'image'), (NEURAL,'neural')]: save_png(out/f'hypotheses_{label}'/f'{i:04d}.png', outputs[n]['hypothesis'].astype(np.uint8))
        old = ROOT/'runs/a1_direct_fiber_template_20261029'
        if video: old = old/row['video_id']; filename = f'{row["frame_index"]:06d}.png'
        else: old = old/'image_native'; filename = f'{i:04d}.png'
        for n,previous in [('raw_classwise','raw_classwise'), (IMAGE,'image_template')]:
            equality[n] += int((masks[n] != load(old/previous/filename)).sum())
        target = load(row['mask']); targets.append(target)
        records.append({'sample_id':row['sample_id'], 'outputs':{n:v['record'] for n,v in outputs.items()}, 'timings':{'shared3model_core_ms':core_ms,**timing}})
        if progress and (i+1)%25 == 0: progress('static_video29' if video else 'static_image132', samples_completed=i+1, expected_samples=len(rows))
    groups = {}
    if video:
        selections = {v:[i for i,r in enumerate(rows) if r['video_id']==v] for v in ('video_1','video_2')}
    else:
        cohorts = read(ROOT/'runs/a1_classwise_experts_20261028/summary.json')['image']['historical_exposure_cohorts']
        exposed = set(cohorts['historical_old190_exposed']['sample_ids'])
        selections = {label:[i for i,r in enumerate(rows) if (r['sample_id'] in exposed)==seen] for label,seen in [('historical_old190_exposed',True), ('not_matched_to_old190_exposure',False)]}
    for label, indices in selections.items():
        groups[label] = {'images':len(indices),'sample_ids':[rows[i]['sample_id'] for i in indices], 'models':{n:summarize([rows[i] for i in indices],[targets[i] for i in indices],[a[i] for i in indices])['metrics'] for n,a in predictions.items()}}
    conditional = {}
    for n in names[1:]:
        ids = [i for i,r in enumerate(records) if r['outputs'][n]['accepted']]
        conditional[n] = {'accepted':len(ids),'rejected':len(rows)-len(ids),'rejection_checks':dict(Counter(k for r in records for k,v in r['outputs'][n]['checks'].items() if not v)),
            'accepted_only_metrics_diagnostic':summarize([rows[i] for i in ids],[targets[i] for i in ids],[predictions[n][i] for i in ids])['metrics'] if ids else None}
    if any(equality.values()): raise ValueError(f'Historical baseline reproduction differs: {dict(equality)}')
    result = {'models':{n:summarize(rows,targets,a) for n,a in predictions.items()}, 'paired_changes':pair_summary(rows,targets,predictions), 'groups':groups,
        'records':records, 'acceptance':conditional, 'historical_different_pixels':dict(equality)}
    atomic_json(out/'comparison.json',result); return result


def main():
    run = read(RUN); out = ROOT/run['output']
    if out.exists(): raise FileExistsError('Never overwrite an existing observation-control run')
    out.mkdir(); job = {'pid':os.getpid(), 'gradient_updates':0, 'template_calibration_updates':0, 'default_model_changed':False, 's2_accepted':False, 'stone_memory_started':False}
    def progress(phase,**extra): job.update(phase=phase,**extra); atomic_json(out/'job.json',job)
    progress('preflight')
    try:
        config = read(ROOT/run['base_config']); prior = read(ROOT/run['template']); mpath = ROOT/config['manifest']; m = read(mpath)
        vpath = mpath.parent/'video_validation_manifest.json'; vm = read(vpath)
        validation = {'train':{'labeled':[],'unlabeled':[]},'val':m['val']+vm['val']}
        identity = input_identity(mpath,Path(m['root']),validation)
        checkpoints = {n:v for n,v in read(ROOT/config['experts'])['checkpoints'].items() if n in ('old190','A0')}
        videos = read(ROOT/'runs/baselines/a0_20261027/config.json')['inputs']
        for v in list(checkpoints.values())+videos:
            if sha256(Path(v['path'])) != v['sha256']: raise ValueError('Checkpoint/video identity differs')
        extra = ['tools/direct_neural_observation.py','tools/run_direct_neural_observation.py','tools/benchmark_direct_neural_observation.py','tests/test_direct_neural_observation.py',
                 'tools/direct_fiber_template.py','tools/run_direct_fiber_template.py','tools/classwise_experts.py','tools/run_fiber_shape_control.py','tools/fiber_shape_prior.py',
                 'tools/run_classwise_expert_control.py','tools/diagnose_joint_failures.py','tools/replay_classwise_experts.py','tools/evaluate_native_checkpoints.py','tools/evaluate_spatial_detail.py']
        sources = source_identity(); sources.update({n:sha256(ROOT/n) for n in extra})
        related = [RUN,ROOT/run['protocol'],ROOT/run['base_config'],ROOT/run['template'],vpath,ROOT/config['experts'],
                   ROOT/'runs/a1_direct_fiber_template_20261029/calibration.json',ROOT/'runs/a1_direct_fiber_template_20261029/preflight.json',
                   ROOT/'runs/a1_direct_fiber_template_20261029/summary.json',ROOT/'runs/a1_classwise_experts_20261028/summary.json',ROOT/'runs/baselines/a0_20261027/config.json',ROOT/'configs/replay_classwise_experts_20261028.json']
        # Freeze the exact predecessor prediction files used for paired reproduction.
        previous = ROOT/'runs/a1_direct_fiber_template_20261029'
        for row_index,row in enumerate(m['val']):
            related.extend(previous/'image_native'/n/f'{row_index:04d}.png' for n in ('raw_classwise','image_template'))
        for row in vm['val']:
            related.extend(previous/row['video_id']/n/f'{row["frame_index"]:06d}.png' for n in ('raw_classwise','image_template'))
        preflight = {'execution':run,'base_config':config,'template':prior,'inputs':identity,'sources':sources,'related':{str(p):sha256(p) for p in related},'checkpoints':checkpoints,'videos':videos,
                     'train_pixels_read':0,'reserved_test_pixels_read':False,'gradient_updates':0,'new_shared_calibration':False}
        atomic_json(out/'preflight.json',preflight)
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for n in sources: z.write(ROOT/n,n.replace('\\','/'))
            for p in related[:14]: z.write(p,str(p.relative_to(ROOT)))
        seed_training(42); experts = ClasswiseExperts(checkpoints['old190']['path'],checkpoints['A0']['path']); before = model_state_hash(experts)
        atomic_json(out/'runtime.json',runtime_identity(experts.device))
        progress('static_image132'); image = evaluate(m['val'],experts,prior,config,out/'image_native',progress=progress)
        progress('static_video29'); video = evaluate(vm['val'],experts,prior,config,out/'video_native',video=True,progress=progress)
        screening = {n:assess(v['metrics'],config['gates']) for n,v in video['models'].items()}; eligible = screening[NEURAL]['passed']; full = {}
        atomic_json(out/'static_summary.json',{'image':image,'video29':video,'screening':screening,'full_replay_eligible':eligible})
        if eligible:
            # Reuse the frozen sequential decoder/export/tracker-reset runner, with explicit new observation implementation.
            from . import run_direct_fiber_template as replay
            replay.OUT = out; replay.NAMES = ('raw_classwise',IMAGE,NEURAL,'causal_template')
            def causal_locate(frame,prior,config,p,tracker=None,requested=None):
                return locate(frame,prior,config,p,tracker,requested if requested is not None else [IMAGE,NEURAL,'causal_template'])
            replay.locate = causal_locate
            for item in videos:
                progress('conditional_full_replay',video_id=item['video_id'])
                result,_ = replay.run_video(item,experts,prior,config,vm['val'],progress); full[item['video_id']] = result
        assert before == model_state_hash(experts) and all(not model.training for model in experts.models)
        del experts; torch.cuda.empty_cache(); performance = {}
        for name in run['arms']:
            progress('isolated_cost',arm=name)
            with (out/f'benchmark_{name}.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-m','tools.benchmark_direct_neural_observation','--arm',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            performance[name] = read(out/f'performance_{name}.json')
        assert input_identity(mpath,Path(m['root']),validation) == identity
        assert all(sha256(ROOT/n)==h for n,h in sources.items()) and all(sha256(Path(n))==h for n,h in preflight['related'].items())
        assert all(sha256(Path(v['path']))==v['sha256'] for v in list(checkpoints.values())+videos)
        result = {'image':image,'video29':video,'screening':screening,'full_replay_eligible':eligible,'full_replay_executed':bool(full),'videos':full,'performance':performance,
                  'model_state_sha256_unchanged':before,'gradient_updates':0,'template_calibration_updates':0,'train_pixels_read':0,'reserved_test_pixels_read':False,
                  'default_model_changed':False,'s2_accepted':False,'stone_memory_started':False,'tip_angle_reference_available':False}
        atomic_json(out/'summary.json',result); progress('completed',static_samples_completed=161,full_replay_executed=bool(full))
        print(json.dumps({'phase':'completed','video29':{n:v['metrics'] for n,v in video['models'].items()},'screening':screening,'full_replay_executed':bool(full)},indent=2),flush=True)
    except Exception as error:
        progress('failed',error=str(error)); (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8'); raise

if __name__ == '__main__': main()
