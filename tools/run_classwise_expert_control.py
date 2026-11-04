"""Frozen inference-only classwise comparison on video29/image132 and total cost."""
from __future__ import annotations

from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from ureter_semis.experiment import atomic_json,input_identity,seed_training,sha256,source_identity
from ureter_semis.labels import read_class_mask
from ureter_semis.metrics import confusion_matrix,metrics_from_confusion
from ureter_semis.native_validation import NativeValidationDataset,collate_native
from ureter_semis.supervised_trainer import assess
from .classwise_experts import ClasswiseExperts
from .diagnose_joint_failures import paired_changes
from .evaluate_native_checkpoints import evaluate
from .evaluate_spatial_detail import boundary_counts,boundary_summary

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/a1_classwise_experts_20261028'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def summarize(rows,targets,predictions):
    matrix=torch.zeros(3,3,dtype=torch.long);frames=[];neg_images=neg_fp_images=neg_pixels=neg_fp=0
    for row,target,pred in zip(rows,targets,predictions):
        valid=target!=255;truth=target==1
        current=confusion_matrix(torch.from_numpy(pred.copy()),torch.from_numpy(target.copy()));matrix+=current
        presence=row.get('stone_presence_review','unknown')
        if presence=='unknown':presence='present' if truth.any() else 'absent' if valid.all() else 'unknown'
        fp=int(((pred==1)&valid&~truth).sum())
        if presence=='absent' and valid.any():
            neg_images+=1;neg_fp_images+=int(fp>0);neg_pixels+=int(valid.sum());neg_fp+=fp
        frames.append({'sample_id':row['sample_id'],'native_shape':list(target.shape),'gt_stone_pixels':int(truth.sum()),
            'stone_tp':int(((pred==1)&truth).sum()),'stone_fp':fp,'fiber_fp':int(((pred==2)&(target!=2)&valid).sum()),
            'fiber_fn':int(((pred!=2)&(target==2)).sum()),'confusion_matrix':current.tolist(),
            'boundaries':{str(c):boundary_counts(pred,target,c,3) for c in (1,2)}})
    metrics=metrics_from_confusion(matrix)
    metrics.update(stone_negative_images=neg_images,stone_negative_images_with_fp=neg_fp_images,
        stone_negative_fp_image_rate=neg_fp_images/neg_images if neg_images else None,stone_negative_valid_pixels=neg_pixels,
        stone_negative_fp_pixels=neg_fp,stone_negative_fp_pixel_fraction=neg_fp/neg_pixels if neg_pixels else None)
    groups={}
    for lower,upper in ((0,1024),(1024,4096),(4096,None)):
        selected=[r for r in frames if r['gt_stone_pixels']>lower and (upper is None or r['gt_stone_pixels']<=upper)]
        pixels=sum(r['gt_stone_pixels'] for r in selected);tp=sum(r['stone_tp'] for r in selected)
        groups[f'({lower},{upper if upper is not None else "inf"}]']={'frames':len(selected),'target_pixels':pixels,'tp':tp,
            'pixel_recall':tp/pixels if pixels else None,'frames_any_tp':sum(r['stone_tp']>0 for r in selected)}
    return {'metrics':metrics,'boundaries':{str(c):boundary_summary([r['boundaries'][str(c)] for r in frames]) for c in (1,2)},
        'visible_stone_area_groups':groups,'frames':frames}


def evaluate_set(rows,root,experts,folder,joint_cache,exposed_groups):
    folder.mkdir();names=('A0','old190','joint521','classwise')
    for name in names:(folder/name).mkdir()
    dataset=NativeValidationDataset(root,rows,experts.size)
    loader=DataLoader(dataset,batch_size=2,shuffle=False,num_workers=0,collate_fn=collate_native)
    targets=[];predictions={name:[] for name in names};conflicts=[];changes=[];index=0
    for batch in loader:
        results=experts.predict_native(batch['image'],[tuple(t.shape) for t in batch['targets']])
        for result,target_cpu in zip(results,batch['targets']):
            target=target_cpu.numpy().astype(np.uint8);row=rows[index]
            masks={'A0':result['fiber_mask'].cpu().numpy(),'old190':result['stone_mask'].cpu().numpy(),
                'classwise':result['mask'].cpu().numpy(),'joint521':np.asarray(Image.open(joint_cache/f'{index:04d}.png')).copy()}
            overlap=(masks['old190']==1)&(masks['A0']==2)
            conflicts.append({'sample_id':row['sample_id'],'overlap_pixels':int(overlap.sum()),
                'overlap_by_reference':{str(c):int((overlap&(target==c)).sum()) for c in (0,1,2,255)}})
            changes.append({'sample_id':row['sample_id'],
                'stone_vs_old190':paired_changes(target,masks['old190'],masks['classwise'],1),
                'fiber_vs_old190':paired_changes(target,masks['old190'],masks['classwise'],2),
                'stone_vs_a0':paired_changes(target,masks['A0'],masks['classwise'],1)})
            for name,mask in masks.items():
                if mask.shape!=target.shape:raise ValueError('Cached/native output geometry differs')
                predictions[name].append(mask);Image.fromarray(mask).save(folder/name/f'{index:04d}.png')
            targets.append(target);index+=1
    result={'models':{name:summarize(rows,targets,preds) for name,preds in predictions.items()},
        'conflicts':{'total_pixels':sum(r['overlap_pixels'] for r in conflicts),
            'by_reference':{str(c):sum(r['overlap_by_reference'][str(c)] for r in conflicts) for c in (0,1,2,255)},'frames':conflicts},
        'paired_changes':{key:{item:sum(r[key][item] for r in changes) for item in changes[0][key]} for key in ('stone_vs_old190','fiber_vs_old190','stone_vs_a0')},
        'changes_per_frame':changes,'by_source_or_video':{},'historical_exposure_cohorts':{}}
    group_names=sorted({row.get('video_id') or row.get('source_dataset') for row in rows})
    for key in group_names:
        selected=[i for i,r in enumerate(rows) if (r.get('video_id') or r.get('source_dataset'))==key]
        result['by_source_or_video'][key]={name:summarize([rows[i] for i in selected],[targets[i] for i in selected],[predictions[name][i] for i in selected])['metrics'] for name in names}
    if any(row.get('group_id') for row in rows):
        for cohort,seen in [('historical_old190_exposed',True),('not_matched_to_old190_exposure',False)]:
            selected=[i for i,row in enumerate(rows) if (row.get('group_id') in exposed_groups)==seen]
            result['historical_exposure_cohorts'][cohort]={'images':len(selected),'sample_ids':[rows[i]['sample_id'] for i in selected],
                'models':{name:summarize([rows[i] for i in selected],[targets[i] for i in selected],[predictions[name][i] for i in selected])['metrics'] for name in names}}
    atomic_json(folder/'comparison.json',result)
    return result


def main():
    protocol_path=ROOT/'configs/compare_classwise_experts_20261028.json';protocol=read(protocol_path)
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir();job={'pid':os.getpid(),'training_updates':0,'default_model_changed':False,'reserved_test_pixels_read':False}
    def phase(value,**extra):job.update(phase=value,**extra);atomic_json(OUT/'job.json',job)
    phase('preflight')
    manifest_path=ROOT/'datasets/ureter_semis_kidney_joint_20261028/manifest.json';m=read(manifest_path);root=Path(m['root'])
    video_manifest_path=manifest_path.parent/'video_validation_manifest.json';vm=read(video_manifest_path)
    raw=read(root/'raw_inventory.json')['records'];groups=read(root/'groups.json')
    old=read(ROOT/'datasets/ureter_semis_a1_sup_kidney_only_20261027/manifest.json')
    old_hashes={r['image_sha256'] for r in old['train']['labeled']}
    exposed={g['group_id'] for g in groups if any(raw[i]['image_sha256'] in old_hashes for i in g['member_indices'])}
    assert len([r for r in m['val'] if r['group_id'] in exposed])==25 and len(m['val'])==132 and len(vm['val'])==29
    checkpoints={'A0':ROOT/protocol['fiber_expert'],'old190':ROOT/protocol['stone_expert'],'joint521':ROOT/'runs/a1_sup_kidney_joint_20261028/training/last.pt'}
    paths=[Path(__file__).resolve(),ROOT/'tools/classwise_experts.py',ROOT/'tools/benchmark_classwise_experts.py',ROOT/'tools/evaluate_native_checkpoints.py',ROOT/'tools/evaluate_spatial_detail.py',ROOT/'tools/benchmark_native.py',ROOT/'tools/diagnose_joint_failures.py',ROOT/'tests/test_classwise_experts.py']
    sources=source_identity();sources.update({str(p.relative_to(ROOT)):sha256(p) for p in paths})
    identity=input_identity(manifest_path,root,{'train':{'labeled':[],'unlabeled':[]},'val':m['val']+vm['val']})
    cached_dirs=[ROOT/'runs/a1_sup_kidney_joint_20261028/image_native/joint_final',ROOT/'runs/a1_sup_kidney_joint_20261028/video_native/joint_final']
    related=[video_manifest_path,root/'groups.json',root/'raw_inventory.json']+[p for folder in cached_dirs for p in sorted(folder.glob('*.png'))]
    preflight={'protocol':protocol,'protocol_sha256':sha256(protocol_path),'checkpoints':{name:{'path':str(p),'sha256':sha256(p)} for name,p in checkpoints.items()},
        'inputs':identity,'sources':sources,'related':{str(p):sha256(p) for p in related},
        'policy_frozen_before_results':True,'image_cohorts':{'historical_old190_exposed':25,'not_matched_to_old190_exposure':107},
        'training_updates':0,'reserved_test_pixels_read':False}
    atomic_json(OUT/'preflight.json',preflight)
    with zipfile.ZipFile(OUT/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in sources:z.write(ROOT/name,name.replace('\\','/'))
        z.write(protocol_path,str(protocol_path.relative_to(ROOT)))
    try:
        phase('fresh_a0_gate_reference')
        baseline=evaluate(video_manifest_path,{'A0':checkpoints['A0']},OUT/'baseline_native')
        a0=baseline['models']['A0']['metrics']
        gates={'stone_dice_min':a0['stone_dice'],'fiber_dice_min':a0['fiber_dice']-.01,'negative_fp_fraction_max':a0['stone_negative_fp_pixel_fraction']}
        preflight.update(native_video_gates=gates,gates_frozen_before_fused_metrics=True);atomic_json(OUT/'preflight.json',preflight)
        seed_training(42);experts=ClasswiseExperts(checkpoints['old190'],checkpoints['A0'])
        if experts.parameter_count!=protocol['parameter_count_total'] or experts.model_count!=3:raise ValueError('Frozen model budget differs')
        phase('video29_accuracy')
        video=evaluate_set(vm['val'],root,experts,OUT/'video_native',cached_dirs[1],exposed)
        historical=read(ROOT/'runs/a1_sup_kidney_resolution_20261028/native_comparison/comparison.json')['models']
        assert video['models']['A0']['metrics']==a0==historical['A0']['metrics']
        assert video['models']['old190']['metrics']==historical['unet256']['metrics']
        assert video['models']['joint521']['metrics']==read(ROOT/'runs/a1_sup_kidney_joint_20261028/video_native/comparison.json')['models']['joint_final']['metrics']
        phase('image132_accuracy_and_cohorts')
        image=evaluate_set(m['val'],root,experts,OUT/'image_native',cached_dirs[0],exposed)
        assert image['models']['joint521']['metrics']==read(ROOT/'runs/a1_sup_kidney_joint_20261028/image_native/comparison.json')['models']['joint_final']['metrics']
        del experts;torch.cuda.empty_cache()
        performance={}
        for name in ('A0','old190','joint521','classwise'):
            phase('matched_fresh_process_cost',current_cost_variant=name)
            if name=='classwise':
                args=[sys.executable,'-m','tools.benchmark_classwise_experts','--manifest',str(video_manifest_path),'--stone',str(checkpoints['old190']),
                    '--fiber',str(checkpoints['A0']),'--output',str(OUT/f'performance_{name}.json')]
            else:
                args=[sys.executable,'-m','tools.benchmark_native','--manifest',str(video_manifest_path),'--checkpoint',str(checkpoints[name]),'--output',str(OUT/f'performance_{name}.json')]
            with (OUT/f'benchmark_{name}.log').open('w',encoding='utf-8') as log:
                subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            performance[name]=read(OUT/f'performance_{name}.json')
        assert input_identity(manifest_path,root,{'train':{'labeled':[],'unlabeled':[]},'val':m['val']+vm['val']})==identity
        assert all(sha256(ROOT/name)==digest for name,digest in sources.items()) and sha256(protocol_path)==preflight['protocol_sha256']
        assert all(sha256(Path(path))==digest for path,digest in preflight['related'].items())
        assert all(sha256(p)==preflight['checkpoints'][name]['sha256'] for name,p in checkpoints.items())
        screening=assess(video['models']['classwise']['metrics'],gates)
        result={'protocol':protocol,'video':video,'image':image,'performance':performance,'screening':screening,
            'adopted':False,'s2_accepted':False,'training_updates':0,'default_model_changed':False,
            'scope':'Frozen inference-only development comparison; neither sparse video29 nor image cohorts establish independent-case validity; classwise output has no calibrated joint probability distribution'}
        atomic_json(OUT/'summary.json',result)
        atomic_json(OUT/'predictor_manifest.json',{'type':'classwise_expert_inference_control','checkpoints':preflight['checkpoints'],
            'rule':protocol['rule'],'model_count':3,'parameter_count':73804185,'default_model_changed':False,
            'implementation':'tools/classwise_experts.py','role':'development_control_not_deployed',
            'uncertainty':'Separate expert probability maps; no calibrated fused class probabilities'})
        phase('completed',screening_passed=screening['passed'],adopted=False,s2_accepted=False)
        print(json.dumps({'phase':'completed','classwise_metrics':video['models']['classwise']['metrics'],'screening':screening,
            'performance':{n:r['timings']['pipeline_ms'] for n,r in performance.items()}},indent=2))
    except Exception as error:
        phase('failed',error=str(error));(OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
