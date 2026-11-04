"""Paired native errors, active annotation conflicts and exposure diagnosis."""
from __future__ import annotations

from collections import Counter,defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from ureter_semis.experiment import atomic_json,input_identity,sha256,source_identity
from ureter_semis.labels import read_class_mask
from ureter_semis.metrics import confusion_matrix,metrics_from_confusion
from .audit_kidney_v2 import feature,variants,verified_pairs,read_image
from .convert_dataset import read_yolo_sample
from .evaluate_native_checkpoints import evaluate

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def class_error_counts(target,prediction,cls,radius=3):
    if target.shape!=prediction.shape or target.ndim!=2 or radius<0 or cls not in (1,2):
        raise ValueError('Equal2D geometry, foreground class and nonnegative radius required')
    valid=target!=255;truth=target==cls;pred=prediction==cls
    fp=valid&pred&~truth;fn=truth&~pred
    distance=cv2.distanceTransform((~truth).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE) if truth.any() else np.full(truth.shape,np.inf)
    other=3-cls
    return {'target_pixels':int(truth.sum()),'tp':int((truth&pred).sum()),'fp':int(fp.sum()),'fn':int(fn.sum()),
        'fp_over_background':int((fp&(target==0)).sum()),'fp_over_other_foreground':int((fp&(target==other)).sum()),
        'fp_near':int((fp&(distance<=radius)).sum()),'fp_far':int((fp&(distance>radius)).sum()),
        'fn_to_background':int((fn&(prediction==0)).sum()),'fn_to_other_foreground':int((fn&(prediction==other)).sum())}


def paired_changes(target,old,new,cls):
    if target.shape!=old.shape or target.shape!=new.shape:raise ValueError('Paired geometry mismatch')
    truth=target==cls;valid=target!=255
    old_fp=(old==cls)&~truth&valid;new_fp=(new==cls)&~truth&valid
    return {'lost_tp':int((truth&(old==cls)&(new!=cls)).sum()),'gained_tp':int((truth&(old!=cls)&(new==cls)).sum()),
        'added_fp':int((new_fp&~old_fp).sum()),'removed_fp':int((old_fp&~new_fp).sum())}


def error_totals(rows,key):
    return {k:sum(r[key][k] for r in rows) for k in rows[0][key]} if rows else {}


def frame_descriptors(image,target):
    rgb=np.asarray(image.resize((256,256),Image.Resampling.BILINEAR))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    crop=rgb[25:231,25:231];g=gray[25:231,25:231]
    hsv=cv2.cvtColor(crop,cv2.COLOR_RGB2HSV)
    mask=np.asarray(Image.fromarray(target).resize((256,256),Image.Resampling.NEAREST))
    valid=mask!=255
    return {'luminance_mean':float(g.mean()),'saturation_mean':float(hsv[:,:,1].mean()/255),
        'laplacian_variance':float(cv2.Laplacian(g,cv2.CV_32F).var()),
        'annotated_stone_fraction':float(((mask==1)&valid).sum()/valid.sum()),
        'annotated_fiber_fraction':float(((mask==2)&valid).sum()/valid.sum()),
        'has_stone_annotation':bool((target==1).any()),'has_fiber_annotation':bool((target==2).any())}


def describe(rows):
    keys=('luminance_mean','saturation_mean','laplacian_variance','annotated_stone_fraction','annotated_fiber_fraction')
    return {'images':len(rows),'has_stone_annotation':sum(r['has_stone_annotation'] for r in rows),
        'has_fiber_annotation':sum(r['has_fiber_annotation'] for r in rows),
        'statistics':{k:{'mean':float(np.mean([r[k] for r in rows])),'p25':float(np.percentile([r[k] for r in rows],25)),
            'median':float(np.median([r[k] for r in rows])),'p75':float(np.percentile([r[k] for r in rows],75))} for k in keys}}


def main():
    config_path=ROOT/'configs/diagnose_kidney_joint_failures_20261028.json';config=read(config_path)
    out=ROOT/config['output']
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    def phase(name):atomic_json(out/'job.json',{'phase':name,'training_updates':0,'labels_changed':False,'reserved_test_pixels_read':False})
    phase('preflight')
    manifest_path=ROOT/config['manifest'];m=read(manifest_path);folder=manifest_path.parent
    groups=read(folder/'groups.json');raw_inventory=read(folder/'raw_inventory.json');raw=raw_inventory['records']
    history=read(ROOT/config['joint_run']/'training/history.json')
    rows=m['train']['labeled']+m['val'];active_ids={r['group_id'] for r in rows}
    conflicts=[g for g in groups if len({raw[i]['annotation_counts'].get('fiber',0)>0 for i in g['member_indices']})>1]
    conflict_members={i for g in conflicts if g['group_id'] in active_ids for i in g['member_indices']}
    protected={raw[i][k] for g in groups if g['split']=='reserved_test' for i in g['member_indices'] for k in ('image','source_label')}
    input_manifest={'train':m['train'],'val':m['val']+m['video_validation']}
    inputs=input_identity(manifest_path,folder,input_manifest)
    sources=source_identity();sources.update({str(Path(__file__).resolve().relative_to(ROOT)):sha256(Path(__file__).resolve())})
    other_inputs={str(ROOT/config['joint_checkpoint']):sha256(ROOT/config['joint_checkpoint']),str(folder/'groups.json'):sha256(folder/'groups.json'),
        str(folder/'raw_inventory.json'):sha256(folder/'raw_inventory.json'),str(ROOT/config['joint_run']/'training/history.json'):sha256(ROOT/config['joint_run']/'training/history.json')}
    # Only active conflict members may be opened; reserved test groups stay metadata-only.
    for i in sorted(conflict_members):
        for key in ('image','source_label'):
            path=Path(raw[i][key]);assert str(path) not in protected;other_inputs[str(path)]=sha256(path)
    preflight={'config':config,'config_sha256':sha256(config_path),'inputs':inputs,'other_inputs':other_inputs,
        'source_hashes':sources,'conflict_groups':len(conflicts),'active_conflict_groups':sum(g['group_id'] in active_ids for g in conflicts),
        'reserved_conflict_groups_metadata_only':sum(g['split']=='reserved_test' for g in conflicts),'frozen_before_diagnosis':True}
    atomic_json(out/'preflight.json',preflight)
    phase('paired_video_native_errors')
    video=[]
    for index,row in enumerate(m['video_validation']):
        target=read_class_mask(Path(row['mask']))
        old=np.asarray(Image.open(ROOT/config['old_video_predictions']/f'{index:04d}.png')).copy()
        new=np.asarray(Image.open(ROOT/config['new_video_predictions']/f'{index:04d}.png')).copy()
        frame={'index':index,'sample_id':row['sample_id'],'video_id':row['video_id'],'frame_index':row['frame_index']}
        for cls,label in ((1,'stone'),(2,'fiber')):
            frame['old_'+label]=class_error_counts(target,old,cls,3);frame['new_'+label]=class_error_counts(target,new,cls,3)
            frame[label+'_changes']=paired_changes(target,old,new,cls)
        frame['stone_fn_change']=frame['new_stone']['fn']-frame['old_stone']['fn']
        frame['fiber_fp_change']=frame['new_fiber']['fp']-frame['old_fiber']['fp']
        video.append(frame)
    video_summary={key:error_totals(video,key) for key in ('old_stone','new_stone','old_fiber','new_fiber','stone_changes','fiber_changes')}
    for kind in ('stone_fn_change','fiber_fp_change'):
        positive=sorted([r for r in video if r[kind]>0],key=lambda r:r[kind],reverse=True)
        total=sum(r[kind] for r in positive)
        video_summary[kind]={'net_change':sum(r[kind] for r in video),'total_positive_change':total,
            'top3_share_of_positive_change':sum(r[kind] for r in positive[:3])/total if total else None,
            'ranked_sample_ids':[r['sample_id'] for r in positive]}
    atomic_json(out/'video_errors.json',{'protocol':config['video_errors'],'overall':video_summary,
        'by_video':{v:{key:error_totals([r for r in video if r['video_id']==v],key) for key in ('old_stone','new_stone','old_fiber','new_fiber')} for v in ('video_1','video_2')},'frames':video})
    phase('training_label_agreement')
    # New model only on active521 training samples; source-provided labels are not reviewed truth.
    train_manifest={'root':m['root'],'train':{'labeled':[],'unlabeled':[]},'val':m['train']['labeled'],'test':[],'role':'training_agreement_only_not_validation'}
    atomic_json(out/'train_agreement_manifest.json',train_manifest)
    train_result=evaluate(out/'train_agreement_manifest.json',{'joint_final':ROOT/config['joint_checkpoint']},out/'train_native')
    phase('source_and_exposure_analysis')
    exposure=Counter(sid for h in history for sid in h['sampled_training_sample_ids'])
    old=read(ROOT/'datasets/ureter_semis_a1_sup_kidney_only_20261027/manifest.json')
    old_hash_counts=Counter(r['image_sha256'] for r in old['train']['labeled'])
    group_for_hash={raw[i]['image_sha256']:g['group_id'] for g in groups for i in g['member_indices']}
    old_group_exposure=Counter()
    for digest,count in old_hash_counts.items():old_group_exposure[group_for_hash[digest]]+=15*count
    source_records=[];all_descriptors=[];source_matrix=defaultdict(lambda:torch.zeros(3,3,dtype=torch.long))
    for split,samples,pred_dir in [('train',m['train']['labeled'],out/'train_native/joint_final'),
        ('image_val',m['val'],ROOT/config['joint_run']/'image_native/joint_final')]:
        for index,row in enumerate(samples):
            target=read_class_mask(Path(row['mask']));prediction=np.asarray(Image.open(pred_dir/f'{index:04d}.png')).copy()
            key=split+':'+row['source_dataset'];matrix=confusion_matrix(torch.from_numpy(prediction),torch.from_numpy(target))
            source_matrix[key]+=matrix
            desc=frame_descriptors(Image.open(row['image']).convert('RGB'),target)
            count=exposure[row['sample_id']];bucket='0' if count==0 else '1-3' if count<4 else '4-6' if count<7 else '7+'
            record={'sample_id':row['sample_id'],'group_id':row['group_id'],'split':split,'source_dataset':row['source_dataset'],
                'source_split':row['source_split'],'exposures':count if split=='train' else None,'exposure_bucket':bucket if split=='train' else 'not_trained',
                'old_group_exposures':old_group_exposure[row['group_id']],'active_presence_conflict':row['group_id'] in {g['group_id'] for g in conflicts},
                'stone':class_error_counts(target,prediction,1),'fiber':class_error_counts(target,prediction,2),
                'descriptors':desc,'confusion_matrix':matrix.tolist()}
            laser=0
            if row.get('auxiliary_masks',{}).get('laser_dot'):
                auxiliary=np.asarray(Image.open(row['auxiliary_masks']['laser_dot']))>0
                laser=int(((prediction==2)&(target!=2)&(target!=255)&auxiliary).sum())
            record['fiber_fp_over_auxiliary_laser_dot']=laser
            source_records.append(record);all_descriptors.append({'split_source':key,**desc})
    for row in m['video_validation']:
        desc=frame_descriptors(Image.open(row['image']).convert('RGB'),read_class_mask(Path(row['mask'])))
        all_descriptors.append({'split_source':row['video_id'],**desc})
    sampling={'total_exposures':sum(exposure.values()),'new_train_pool':len(m['train']['labeled']),'unique_seen':len(exposure),
        'new_exposure_histogram':dict(Counter(exposure.get(r['sample_id'],0) for r in m['train']['labeled'])),
        'old190_conservative_groups':len(old_group_exposure),'old_group_exposure_histogram':dict(Counter(old_group_exposure.values())),
        'old_total_exposures':sum(old_group_exposure.values()),
        'new_exposures_on_historical_groups':sum(exposure[r['sample_id']] for r in m['train']['labeled'] if r['group_id'] in old_group_exposure),
        'historical_groups_in_new_train':sum(r['group_id'] in old_group_exposure for r in m['train']['labeled']),
        'old_groups_reassigned_image_val_or_reserved_test':sum(r['group_id'] in old_group_exposure for r in m['val']+m['reserved_test'])}
    def subset_metrics(records):
        if not records:return None
        matrix=sum((torch.tensor(r['confusion_matrix'],dtype=torch.long) for r in records),torch.zeros(3,3,dtype=torch.long))
        return {'images':len(records),'metrics':metrics_from_confusion(matrix),
            'fiber_fp_over_auxiliary_laser_dot':sum(r['fiber_fp_over_auxiliary_laser_dot'] for r in records),
            'exposures':sum(r['exposures'] or 0 for r in records)}
    strata={key:subset_metrics([r for r in source_records if r['split']+':'+r['source_dataset']==key]) for key in source_matrix}
    exposure_strata={bucket:subset_metrics([r for r in source_records if r['split']=='train' and r['exposure_bucket']==bucket]) for bucket in ('0','1-3','4-6','7+')}
    descriptors={key:describe([r for r in all_descriptors if r['split_source']==key]) for key in sorted({r['split_source'] for r in all_descriptors})}
    atomic_json(out/'source_exposure.json',{'sampling':sampling,'source_metrics':strata,'exposure_bucket_metrics':exposure_strata,
        'descriptor_groups':descriptors,'source_records':source_records,
        'interpretation':'Training/source-label agreement only; exposure-stratum associations are not randomized or causal; native pixel-weighted metrics separated by source'})
    phase('active_conflict_alignment')
    names={'kidney_stones_fiber_v1':{0:'fiber',1:'laser',2:'stone',3:'tissue'},'kidney_stones_fiber_v2':{0:'fiber',1:'stone',2:'tissue'}}
    conflicts_out=[]
    for group in conflicts:
        canonical=raw[group['canonical_index']]
        entry={'group_id':group['group_id'],'split':group['split'],'canonical_index':group['canonical_index'],
            'canonical_source':canonical['source_dataset'],'canonical_family':canonical['family'],
            'canonical_counts':canonical['annotation_counts'],'members_metadata':[raw[i] for i in group['member_indices']]}
        if group['group_id'] not in active_ids:
            entry['pixel_analysis']='skipped_reserved_test_metadata_only';conflicts_out.append(entry);continue
        canon_image=read_image(Path(canonical['image']));canon_feature=feature(canon_image)
        canon_mask,_,canon_aux=read_yolo_sample(Path(canonical['image']),Path(canonical['source_label']),names[canonical['source_dataset']])
        choices=[]
        for index in group['member_indices']:
            member=raw[index]
            if bool(member['annotation_counts'].get('fiber',0))==bool(canonical['annotation_counts'].get('fiber',0)):continue
            image=read_image(Path(member['image']))
            matches=verified_pairs([[feature(v) for v in variants(image)]],[canon_feature])
            if matches:choices.append((matches[0]['rgb_mae'],index,matches[0]))
        if not choices:
            entry['pixel_analysis']='no_verified_direct_D4_match_no_label_overlap_claim';conflicts_out.append(entry);continue
        _,index,match=min(choices,key=lambda item:(item[0],item[1]));member=raw[index]
        member_mask,_,member_aux=read_yolo_sample(Path(member['image']),Path(member['source_label']),names[member['source_dataset']])
        def small(array):return np.asarray(Image.fromarray(array).resize((256,256),Image.Resampling.NEAREST)).copy()
        cm=small(canon_mask);mm=variants(small(member_mask))[match['d4_orientation']]
        laser=variants(small(member_aux.get('laser_dot',np.zeros_like(member_mask))))[match['d4_orientation']]>0
        canon_laser=small(canon_aux.get('laser_dot',np.zeros_like(canon_mask)))>0
        both_fiber=(cm==2)|(mm==2);source_laser=laser|canon_laser
        location=out/'conflicts'/group['group_id'];location.mkdir(parents=True)
        for name,array in [('canonical_mask',cm),('member_aligned_mask',mm),('laser_aligned',source_laser.astype(np.uint8))]:Image.fromarray(array).save(location/(name+'.png'))
        entry.update(pixel_analysis='verified_D4_source_disagreement_only',member_index=index,alignment=match,
            canonical_fiber_pixels_256=int((cm==2).sum()),aligned_member_fiber_pixels_256=int((mm==2).sum()),
            fiber_disagreement_pixels_256=int(((cm==2)^(mm==2)).sum()),
            fiber_union_overlap_with_other_source_laser_pixels_256=int((both_fiber&source_laser).sum()),
            fiber_union_pixels_256=int(both_fiber.sum()),
            laser_overlap_fraction=float((both_fiber&source_laser).sum()/both_fiber.sum()) if both_fiber.any() else None)
        conflicts_out.append(entry)
    atomic_json(out/'conflicts.json',{'groups':conflicts_out,'reserved_test_pixels_read':False,
        'interpretation':'D4 geometry and source annotation disagreement only; laser/fiber overlap cannot identify true physical object'})
    assert input_identity(manifest_path,folder,input_manifest)==inputs
    assert all(sha256(Path(p))==h for p,h in other_inputs.items())
    assert all(sha256(ROOT/p)==h for p,h in sources.items()) and sha256(config_path)==preflight['config_sha256']
    summary={'config':config,'video_error_summary':video_summary,'sampling':sampling,'source_metrics':strata,'descriptor_groups':descriptors,
        'conflict_summary':{'total':len(conflicts_out),'active':sum(g['split']!='reserved_test' for g in conflicts_out),
            'reserved_metadata_only':sum(g['split']=='reserved_test' for g in conflicts_out),
            'verified_direct_alignment':sum(g['pixel_analysis']=='verified_D4_source_disagreement_only' for g in conflicts_out),
            'active_laser_overlap_groups':sum((g.get('fiber_union_overlap_with_other_source_laser_pixels_256') or 0)>0 for g in conflicts_out)},
        'training_updates':0,'labels_changed':False,'split_changed':False,'all_frozen_identities_unchanged':True,'reserved_test_pixels_read':False}
    atomic_json(out/'summary.json',summary);phase('completed')
    print(json.dumps({'phase':'completed','video_error_summary':video_summary,'sampling':sampling,'conflicts':summary['conflict_summary']},indent=2))


if __name__=='__main__':main()
