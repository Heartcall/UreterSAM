"""Independent diagnostic accounting and representative visual evidence."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image,ImageDraw
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from ureter_semis.experiment import atomic_json,input_identity,sha256
from ureter_semis.labels import read_class_mask
from ureter_semis.metrics import confusion_matrix
from tools.audit_kidney_v2 import variants
from tools.convert_dataset import read_yolo_sample
from tools.diagnose_joint_failures import class_error_counts,paired_changes,frame_descriptors


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


if __name__=='__main__':
    out=ROOT/'runs/a1_sup_kidney_joint_diagnosis_20261028';p=read(out/'preflight.json');s=read(out/'summary.json')
    cfg=s['config'];manifest_path=ROOT/cfg['manifest'];m=read(manifest_path);folder=manifest_path.parent
    v=read(out/'video_errors.json');source=read(out/'source_exposure.json');conflicts=read(out/'conflicts.json')['groups']
    raw=read(folder/'raw_inventory.json')['records'];groups=read(folder/'groups.json')
    history=read(ROOT/cfg['joint_run']/'training/history.json')
    old_history=read(ROOT/'runs/a1_sup_kidney_resolution_20261028/unet256/history.json')
    matrices={name:torch.zeros(3,3,dtype=torch.long) for name in ('old','new')}
    paired_match=True
    for i,row in enumerate(m['video_validation']):
        target=read_class_mask(Path(row['mask']))
        a=np.asarray(Image.open(ROOT/cfg['old_video_predictions']/f'{i:04d}.png')).copy()
        b=np.asarray(Image.open(ROOT/cfg['new_video_predictions']/f'{i:04d}.png')).copy()
        matrices['old']+=confusion_matrix(torch.from_numpy(a),torch.from_numpy(target))
        matrices['new']+=confusion_matrix(torch.from_numpy(b),torch.from_numpy(target))
        frame=v['frames'][i]
        for cls,label in ((1,'stone'),(2,'fiber')):
            paired_match &= paired_changes(target,a,b,cls)==frame[label+'_changes']
            paired_match &= class_error_counts(target,a,cls)==frame['old_'+label] and class_error_counts(target,b,cls)==frame['new_'+label]
    old_metrics=read(ROOT/'runs/a1_sup_kidney_resolution_20261028/native_comparison/comparison.json')['models']['unet256']['metrics']
    new_metrics=read(ROOT/cfg['joint_run']/'video_native/comparison.json')['models']['joint_final']['metrics']
    exposure=Counter(x for h in history for x in h['sampled_training_sample_ids'])
    old_ids={r['image_sha256'] for r in read(ROOT/'datasets/ureter_semis_a1_sup_kidney_only_20261027/manifest.json')['train']['labeled']}
    old_groups={g['group_id'] for g in groups if any(raw[i]['image_sha256'] in old_ids for i in g['member_indices'])}
    protected_raw={raw[i][key] for g in groups if g['split']=='reserved_test' for i in g['member_indices'] for key in ('image','source_label')}
    protected_canonical={r[key] for r in m['reserved_test'] for key in ('image','mask','source_label')}
    files=set(p['inputs']['files'])|set(p['other_inputs'])
    partitions=True
    for row in v['frames']:
        for key in ('old_stone','new_stone','old_fiber','new_fiber'):
            r=row[key];partitions &= r['fp']==r['fp_near']+r['fp_far']==r['fp_over_background']+r['fp_over_other_foreground']
            partitions &= r['fn']==r['fn_to_background']+r['fn_to_other_foreground']
    new_source_totals={split:sum((torch.tensor(r['confusion_matrix']) for r in source['source_records'] if r['split']==split),torch.zeros(3,3,dtype=torch.long)) for split in ('train','image_val')}
    active={r['group_id'] for r in m['train']['labeled']+m['val']}
    checks={
        'frozen_config_and_active_inputs_unchanged':sha256(ROOT/'configs/diagnose_kidney_joint_failures_20261028.json')==p['config_sha256'] and input_identity(manifest_path,folder,{'train':m['train'],'val':m['val']+m['video_validation']})==p['inputs'],
        'frozen_sources_and_other_inputs_unchanged':all(sha256(ROOT/path)==digest for path,digest in p['source_hashes'].items()) and all(sha256(Path(path))==digest for path,digest in p['other_inputs'].items()),
        'native29_old_and_new_confusions_exactly_reproduced':matrices['old'].tolist()==old_metrics['confusion_matrix'] and matrices['new'].tolist()==new_metrics['confusion_matrix'],
        'paired_class_changes_and_distance_counts_recomputed':bool(paired_match),
        'near_far_background_foreground_partitions_exhaustive':bool(partitions),
        'paired_lost_gained_equals_signed_fn_change':v['overall']['stone_changes']['lost_tp']-v['overall']['stone_changes']['gained_tp']==new_metrics['stone_fn']-old_metrics['stone_fn'],
        'paired_added_removed_equals_signed_fp_change':v['overall']['fiber_changes']['added_fp']-v['overall']['fiber_changes']['removed_fp']==new_metrics['fiber_fp']-old_metrics['fiber_fp'],
        'same2850_actual_exposures_and_histogram':sum(exposure.values())==2850 and len(exposure)==519 and dict(Counter(exposure.get(r['sample_id'],0) for r in m['train']['labeled']))=={int(k):value for k,value in s['sampling']['new_exposure_histogram'].items()},
        'historical190_groups_and_shared_exposures_recomputed':len(old_groups)==150 and sum(exposure[r['sample_id']] for r in m['train']['labeled'] if r['group_id'] in old_groups)==536,
        'source_training_confusion_matches_independent_native_result':new_source_totals['train'].tolist()==read(out/'train_native/comparison.json')['models']['joint_final']['metrics']['confusion_matrix'],
        'source_imageval_confusion_matches_frozen_final_epoch':new_source_totals['image_val'].tolist()==history[-1]['metrics']['confusion_matrix'],
        'source_records_exactly521train132imageval':Counter(r['split'] for r in source['source_records'])=={'train':521,'image_val':132},
        'all11_conflict_groups_inventoried_only6_active':len(conflicts)==11 and sum(g['group_id'] in active for g in conflicts)==6,
        'reserved5_conflicts_metadata_only_no_pixel_inputs':sum(g['pixel_analysis']=='skipped_reserved_test_metadata_only' for g in conflicts)==5 and not files&(protected_raw|protected_canonical),
        'unverified_alignments_have_no_overlap_claim':all('fiber_disagreement_pixels_256' not in g for g in conflicts if g['pixel_analysis']!='verified_D4_source_disagreement_only'),
        'strict_verified_alignment_count_two':sum(g['pixel_analysis']=='verified_D4_source_disagreement_only' for g in conflicts)==2,
        'no_training_label_or_split_changes':s['training_updates']==0 and not s['labels_changed'] and not s['split_changed'] and not s['reserved_test_pixels_read'],
        'a0_preserved':sha256(ROOT/'runs/endosemis/best.pt')=='e4df36dde654ed5f5aa4e93125b256f22312c66bc8945a7fdcfa9bf2b8bf27f7'
    }
    # Additional descriptive statistics; no model or threshold selection.
    conditional={}
    for key in ('train:kidney_stones_fiber_v1','train:kidney_stones_fiber_v2','image_val:kidney_stones_fiber_v1','image_val:kidney_stones_fiber_v2','video_1','video_2'):
        if key.startswith('video'):
            ds=[frame_descriptors(Image.open(r['image']).convert('RGB'),read_class_mask(Path(r['mask']))) for r in m['video_validation'] if r['video_id']==key]
        else:ds=[r['descriptors'] for r in source['source_records'] if r['split']+':'+r['source_dataset']==key]
        conditional[key]={}
        for label in ('stone','fiber'):
            values=[d['annotated_'+label+'_fraction'] for d in ds if d['has_'+label+'_annotation']]
            conditional[key][label]={'positive_images':len(values),'conditional_median_fraction':float(np.median(values)) if values else None}
    pixel_exposure={}
    for name,hist in [('historical190',old_history),('joint521',history)]:
        totals={str(c):sum(h['sampled_label_pixels'][str(c)] for h in hist) for c in (0,1,2,255)}
        valid=sum(totals[str(c)] for c in (0,1,2))
        pixel_exposure[name]={'counts':totals,'valid':valid,'stone_fraction':totals['1']/valid,'fiber_fraction':totals['2']/valid}
    atomic_json(out/'supplemental_descriptors.json',{'positive_class_conditional':conditional,'actual256_training_pixel_exposure':pixel_exposure,
        'interpretation':'Post-hoc descriptive summaries only; no labels/thresholds changed or causal claim'})
    audit={'checks':checks,'all_passed':all(checks.values()),'new_unit_tests':{'passed':3,'failed':0,
        'command':'python -m unittest discover -s tests -p test_joint_failure_diagnosis.py -v','evidence':'All3 passed in this turn before diagnostic execution'},
        'scope':'Read-only diagnostic correctness; not clinical, independent-case or causal validation'}
    atomic_json(ROOT/'runs/engineering/kidney_joint_diagnosis_checks_20261028.json',audit)
    print(json.dumps(audit,indent=2));assert audit['all_passed']

    # Representative video errors in the native400 grid, resized only for display.
    chosen=[]
    for field in ('stone_fn_change','fiber_fp_change'):
        for r in sorted(v['frames'],key=lambda x:x[field],reverse=True)[:3]:
            if r['index'] not in chosen:chosen.append(r['index'])
    width=220;canvas=Image.new('RGB',(5*width,len(chosen)*250+28),'white');draw=ImageDraw.Draw(canvas)
    draw.text((4,6),'Raw | GT green=stone blue=fiber | Old190 | Joint521 | New errors: fiberFP magenta / stoneFN yellow',fill='black')
    def overlay(image,mask,laser=None):
        a=np.asarray(image).copy()
        for cls,color in ((1,np.array([50,230,70])),(2,np.array([40,170,255]))):
            hit=mask==cls;a[hit]=(.4*a[hit]+.6*color).astype(np.uint8)
        if laser is not None:a[laser]=(.35*a[laser]+.65*np.array([255,230,0])).astype(np.uint8)
        return Image.fromarray(a)
    for j,index in enumerate(chosen):
        row=m['video_validation'][index];image=Image.open(row['image']).convert('RGB');target=read_class_mask(Path(row['mask']))
        old=np.asarray(Image.open(ROOT/cfg['old_video_predictions']/f'{index:04d}.png'));new=np.asarray(Image.open(ROOT/cfg['new_video_predictions']/f'{index:04d}.png'))
        error=np.asarray(image).copy();valid=target!=255;error[(new==2)&(target!=2)&valid]=[240,0,210];error[(new!=1)&(target==1)]=[255,220,0]
        panels=[image,overlay(image,target),overlay(image,old),overlay(image,new),Image.fromarray(error)]
        frame=v['frames'][index];y=28+j*250
        draw.text((4,y),f"{row['video_id']} zero-index{row['frame_index']}: stone FN delta{frame['stone_fn_change']:+d}; fiber FP delta{frame['fiber_fp_change']:+d}",fill='black')
        for k,panel in enumerate(panels):canvas.paste(panel.resize((width,width)),(k*width,y+25))
    canvas.save(out/'video_failure_changes.png')

    # Show all six active conflicts; unmatched members remain in their OWN coordinates.
    names={'kidney_stones_fiber_v1':{0:'fiber',1:'laser',2:'stone',3:'tissue'},'kidney_stones_fiber_v2':{0:'fiber',1:'stone',2:'tissue'}}
    active_conflicts=[g for g in conflicts if g['group_id'] in active]
    canvas=Image.new('RGB',(4*256,len(active_conflicts)*294+30),'white');draw=ImageDraw.Draw(canvas)
    draw.text((4,5),'Canonical raw / source labels | Opposite-presence member raw / source labels; green stone, blue fiber, yellow laser',fill='black')
    def load_member(index):
        r=raw[index];image=Image.open(r['image']).convert('RGB').resize((256,256),Image.Resampling.BILINEAR)
        mask,_,aux=read_yolo_sample(Path(r['image']),Path(r['source_label']),names[r['source_dataset']])
        mask=np.asarray(Image.fromarray(mask).resize((256,256),Image.Resampling.NEAREST))
        laser=np.asarray(Image.fromarray(aux.get('laser_dot',np.zeros((r['height'],r['width']),np.uint8))).resize((256,256),Image.Resampling.NEAREST))>0
        return image,mask,laser
    for j,g in enumerate(active_conflicts):
        c=raw[g['canonical_index']]
        other_index=g.get('member_index')
        if other_index is None:
            other_index=next(r['index'] for r in g['members_metadata'] if bool(r['annotation_counts'].get('fiber',0))!=bool(c['annotation_counts'].get('fiber',0)))
        other=raw[other_index];image,mask,laser=load_member(g['canonical_index']);other_image,other_mask,other_laser=load_member(other_index)
        if g['pixel_analysis']=='verified_D4_source_disagreement_only':
            orientation=g['alignment']['d4_orientation'];other_image=Image.fromarray(variants(np.asarray(other_image))[orientation]);other_mask=variants(other_mask)[orientation];other_laser=variants(other_laser)[orientation]
            status='VERIFIED D4 common coordinates'
        else:status='UNMATCHED: each source shown in own coordinates'
        y=30+j*294
        draw.text((4,y),f"{g['split']} {c['source_dataset'][-2:]} {c['family']} vs {other['source_dataset'][-2:]} {other['family']} | {status}",fill='black')
        for k,panel in enumerate([image,overlay(image,mask,laser),other_image,overlay(other_image,other_mask,other_laser)]):canvas.paste(panel,(k*256,y+28))
    canvas.save(out/'active_label_conflicts.png')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    old_hist=s['sampling']['old_group_exposure_histogram'];new_hist=s['sampling']['new_exposure_histogram']
    for histogram,total,label in [(old_hist,150,'Old190:150groups'),(new_hist,521,'Joint521:521groups')]:
        x=sorted(map(int,histogram));axes[0].plot(x,[histogram[str(k)]/total for k in x],marker='o',label=label)
    axes[0].set_xlabel('Actual exposures per conservative group');axes[0].set_ylabel('Fraction of groups');axes[0].legend()
    labels=['train:v1','train:v2','image-val:v1','image-val:v2','video1','video2'];keys=list(conditional)
    for offset,cls,label in [(-.18,'stone','Stone present'),(.18,'fiber','Fiber present')]:
        axes[1].bar(np.arange(6)+offset,[conditional[k][cls]['conditional_median_fraction']*100 for k in keys],.36,label=label)
    axes[1].set_xticks(np.arange(6),labels,rotation=35,ha='right');axes[1].set_ylabel('Median annotated area on positive images (%)');axes[1].legend()
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Descriptive exposure and target-scale differences, not causal attribution')
    fig.savefig(out/'exposure_target_scale.png',dpi=150);plt.close(fig)
    with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in [ROOT/'configs/diagnose_kidney_joint_failures_20261028.json',ROOT/'tools/diagnose_joint_failures.py',ROOT/'tests/test_joint_failure_diagnosis.py',Path(__file__).resolve()]:z.write(path,str(path.relative_to(ROOT)))
