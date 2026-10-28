"""Verify joint groups, frozen video29, matched sampling and independent metrics."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import zipfile
import hashlib

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from ureter_semis.experiment import atomic_json,input_identity,sha256,source_identity
from ureter_semis.labels import read_class_mask
from ureter_semis.metrics import confusion_matrix,metrics_from_confusion
from ureter_semis.source_policy import require_training_sources


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


if __name__=='__main__':
    folder=ROOT/'datasets/ureter_semis_kidney_joint_20261028';out=ROOT/'runs/a1_sup_kidney_joint_20261028'
    m=read(folder/'manifest.json');old=read(ROOT/'datasets/ureter_semis_a1_sup_kidney_only_20261027/manifest.json')
    vm=read(folder/'video_validation_manifest.json');groups=read(folder/'groups.json');raw=read(folder/'raw_inventory.json')
    preflight=read(out/'preflight.json');summary=read(out/'summary.json');history=read(out/'training/history.json');run=read(out/'training/run_config.json')
    require_training_sources(m)
    group_splits={};all_indices=[]
    for g in groups:
        for i in g['member_indices']:group_splits[i]=g['split'];all_indices.append(i)
    pairs=read(folder/'near_copy_pairs.json')
    all_group_ids=[r['group_id'] for r in m['train']['labeled']+m['val']+m['reserved_test']]
    sampled=[sid for h in history for sid in h['sampled_training_sample_ids']]
    train_ids={r['sample_id'] for r in m['train']['labeled']}
    matrices_match=True;native_shapes=True;strata={}
    for name,manifest,comparison in [('image',m,out/'image_native/comparison.json'),('video',vm,out/'video_native/comparison.json')]:
        result=read(comparison)['models']['joint_final'];matrix=torch.zeros(3,3,dtype=torch.long)
        by_group=defaultdict(lambda:torch.zeros(3,3,dtype=torch.long))
        for i,row in enumerate(manifest['val']):
            target=read_class_mask(Path(row['mask']));pred=np.asarray(Image.open(comparison.parent/'joint_final'/f'{i:04d}.png'))
            native_shapes &= target.shape==pred.shape
            current=confusion_matrix(torch.from_numpy(pred),torch.from_numpy(target));matrix+=current
            by_group[row['source_dataset'] if name=='image' else row['video_id']]+=current
        matrices_match &= matrix.tolist()==result['metrics']['confusion_matrix']
        strata[name]={k:metrics_from_confusion(v) for k,v in by_group.items()}
    conflict_counts=Counter()
    for g in groups:
        signatures={(raw['records'][i]['annotation_counts'].get('stone',0),raw['records'][i]['annotation_counts'].get('fiber',0)) for i in g['member_indices']}
        presence={(a>0,b>0) for a,b in signatures}
        conflict_counts['foreground_count_conflicts']+=len(signatures)>1
        conflict_counts['foreground_presence_conflicts']+=len(presence)>1
        conflict_counts['fiber_presence_conflicts']+=len({b for a,b in presence})>1
    old_train_hashes={r['image_sha256'] for r in old['train']['labeled']}
    old_reserve_hashes={r['image_sha256'] for r in old['reserved_unlabeled']}
    group_by_id={g['group_id']:g for g in groups}
    old_exposure={}
    for split,rows in [('train',m['train']['labeled']),('image_val',m['val']),('reserved_test',m['reserved_test'])]:
        old_exposure[split]={label:sum(any(raw['records'][i]['image_sha256'] in hashes for i in group_by_id[row['group_id']]['member_indices']) for row in rows)
            for label,hashes in [('contains_historical190_image',old_train_hashes),('contains_historical800_reserve_image',old_reserve_hashes)]}
    atomic_json(out/'stratified_metrics.json',{'metrics_by_source_or_video':strata,'conflict_counts':dict(conflict_counts),
        'historical_exposure_group_counts':old_exposure,'image_pixel_weighting':'Global native confusion weights640 exports4x320 export area; report per source separately'})
    encoder=Path(torch.hub.get_dir())/'checkpoints/resnet34-333f7ec4.pth'
    checks={
        'all2387_raw_records_in_one_component_each':sorted(all_indices)==list(range(2387)) and len(set(all_indices))==2387,
        'all_near_copy_edges_within_one_purpose':all(group_splits[p['query_index']]==group_splits[p['reference_index']] for p in pairs),
        'augmentation_families_within_one_component':all(len({g['group_id'] for g in groups if any(raw['records'][i]['source_dataset']==source and raw['records'][i]['family']==family for i in g['member_indices'])})==1 for source,family in {(r['source_dataset'],r['family']) for r in raw['records']}),
        'all_original_test_associated_groups_reserved':all(g['split']=='reserved_test' for g in groups if g['contains_original_test']),
        'one_canonical_per_disjoint_train_val_test_group':len(all_group_ids)==len(set(all_group_ids))==797,
        'new521_train132_imageval144_reserved':len(m['train']['labeled'])==521 and len(m['val'])==132 and len(m['reserved_test'])==144,
        'all_raw_inputs_and_labels_unchanged':all(sha256(Path(p))==digest for p,digest in raw['input_hashes'].items()),
        'video29_rows_and_original_snapshot_unchanged':m['video_validation']==vm['val']==old['val'] and input_identity(folder/'video_validation_manifest.json',folder,vm)==preflight['video_inputs'],
        'full2694_video_frames_screened_no_candidates':read(folder/'video_overlap.json')['frames_compared']==2694 and read(folder/'video_overlap.json')['pairs']==[],
        'source_exclusion_and_pending_labels_preserved':{r['source_dataset'] for r in m['train']['labeled']}=={'kidney_stones_fiber_v1','kidney_stones_fiber_v2'} and all(r['review_status']=='pending' and not r['source_group_verified'] for r in m['train']['labeled']) and m['training_ready'] is False,
        'zero_unlabeled_pseudo_or_video_training':m['train']['unlabeled']==[] and m['reserved_unlabeled']==[] and all(r.get('video_id') is None for r in m['train']['labeled']) and run['config']['unlabeled_images_used']==0,
        'frozen_inputs_config_source_and_encoder':input_identity(folder/'manifest.json',folder,m)==preflight['inputs'] and source_identity()==preflight['source_hashes'] and sha256(encoder)==preflight['encoder_sha256'] and all(sha256(Path(p))==h for p,h in preflight['related_inputs'].items()),
        'exact1425_updates2850_exposures':len(history)==15 and sum(h['optimizer_update_steps'] for h in history)==1425 and len(sampled)==2850,
        '190_distinct_samples_each_budget_epoch':all(h['sampled_training_images']==190 and len(h['sampled_training_sample_ids'])==len(set(h['sampled_training_sample_ids']))==190 and h['optimizer_update_steps']==95 for h in history),
        'sampled_ids_only_in_frozen_train':set(sampled)<=train_ids,
        'clean_initialization_fixed_architecture_and_schedule':run['config']['pretrained'] and run['config']['architecture']=='unet_resnet34' and run['config']['size']==256 and run['config']['seed']==42 and run['config']['epochs']==15 and run['config']['images_per_epoch']==190 and run['config']['class_weights']==[1.,2.,4.],
        'same_video_a0_and_old190_metrics_reproduced':read(out/'baseline_video_native/comparison.json')['models']['A0']['metrics']==read(ROOT/'runs/a1_sup_kidney_resolution_20261028/native_comparison/comparison.json')['models']['A0']['metrics'] and read(out/'baseline_video_native/comparison.json')['models']['kidney190']['metrics']==read(ROOT/'runs/a1_sup_kidney_resolution_20261028/native_comparison/comparison.json')['models']['unet256']['metrics'],
        'native_prediction_geometry_and_independent_confusion':bool(native_shapes and matrices_match),
        'final_image_metrics_equal_final_epoch':summary['image_validation']['metrics']==history[-1]['metrics'],
        'video_regression_not_checkpoint_selection':all(h['metrics']['valid_pixels']==history[-1]['metrics']['valid_pixels'] for h in history) and summary['fixed_final_epoch']==15 and run['inputs']['files']==preflight['inputs']['files'],
        'reserved_test_masks_not_in_training_or_evaluation_identity':not {r['mask'] for r in m['reserved_test']} & (set(run['inputs']['files'])|set(preflight['video_inputs']['files'])),
        'fresh_cost87_samples12_warmup':summary['performance']['samples']==87 and summary['performance']['warmup_frames']==12,
        'a0_sha_preserved':sha256(ROOT/'runs/endosemis/best.pt')=='e4df36dde654ed5f5aa4e93125b256f22312c66bc8945a7fdcfa9bf2b8bf27f7'
    }
    audit={'checks':checks,'all_passed':all(checks.values()),'tests':read(ROOT/'runs/engineering/joint_tests_20261028.json'),
        'unique_training_images_seen':len(set(sampled)),'sampling_histogram':dict(Counter(Counter(sampled).values())),
        'scope':'Single-seed matched-update development check, not independent case or clinical acceptance'}
    atomic_json(ROOT/'runs/engineering/kidney_joint_checks_20261028.json',audit)
    print(json.dumps(audit,indent=2));assert audit['all_passed']

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    comparison={**summary['video_baselines'],'joint521':summary['video_validation']}
    names=list(comparison);x=np.arange(len(names));width=.35
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for offset,metric,label in [(-width/2,'stone_dice','Stone'),(width/2,'fiber_dice','Fiber')]:
        axes[0].bar(x+offset,[comparison[n]['metrics'][metric] for n in names],width,label=label)
    axes[0].set_xticks(x,names);axes[0].set_ylim(0,1);axes[0].set_ylabel('Native400 Dice on frozen29 video frames');axes[0].legend()
    axes[1].plot([h['epoch'] for h in history],[h['metrics']['stone_dice'] for h in history],label='Image-dev stone')
    axes[1].plot([h['epoch'] for h in history],[h['metrics']['fiber_dice'] for h in history],label='Image-dev fiber')
    axes[1].set_xlabel('Budget epoch (95 updates,190 sampled images)');axes[1].set_ylabel('132-image native export Dice');axes[1].legend()
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Joint-data baseline:1425updates, image-dev and video-regression metrics kept separate')
    fig.savefig(out/'joint_baseline_comparison.png',dpi=150);plt.close(fig)
