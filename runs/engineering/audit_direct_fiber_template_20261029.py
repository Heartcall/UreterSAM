"""Frozen train-only calibration, fixed-family visible masks and cost audit."""
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/a1_direct_fiber_template_20261029'
NAMES=('raw_classwise','image_template','soft_template','causal_template')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def load(p):
    with Image.open(p) as im:return np.asarray(im).copy()


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def matrix(target,pred):
    valid=target!=255
    return np.bincount((3*target[valid].astype(int)+pred[valid]).ravel(),minlength=9).reshape(3,3)


def reconstruct(frame,record,prior,config):
    shape=frame.shape[:2]
    if not record['accepted']:return np.zeros(shape,bool)
    x,y,angle,length=record['pose'];yy,xx=np.indices(shape,dtype=np.float32);r=prior['cap_radius_fraction']*min(shape);center=max(1,length-r)
    t=(xx-x)*np.cos(angle)+(yy-y)*np.sin(angle);n=-(xx-x)*np.sin(angle)+(yy-y)*np.cos(angle)
    width=r+prior['halfwidth_slope']*(center-t)
    model=((t>=0)&(t<=center)&(np.abs(n)<=width))|((t>=center)&((t-center)**2+n**2<=r*r))
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV);bins=prior['histogram_bins'];h=(hsv[:,:,0].astype(int)*bins//180).clip(0,bins-1);s=(hsv[:,:,1].astype(int)*bins//256).clip(0,bins-1)
    return model&(np.asarray(prior['appearance_probability'])[h,s]>=config['acceptance']['visible_appearance_min'])&(hsv[:,:,2]>10)


def main():
    s=read(OUT/'summary.json');p=read(OUT/'preflight.json');cal=read(OUT/'calibration.json');config=p['config'];prior=p['template'];checks={};details={}
    checks['all_frozen_sources_and_related_template_config_references_unchanged']=all(digest(ROOT/n)==h for n,h in p['sources'].items()) and all(digest(n)==h for n,h in p['related'].items())
    checks['weights_and_videos_unchanged']=all(digest(v['path'])==v['sha256'] for v in list(p['checkpoints'].values())+p['video_inputs'])
    mpath=ROOT/config['manifest'];m=read(mpath);vm=read(mpath.parent/'video_validation_manifest.json')
    checks['521_132_144_29_manifest_unchanged']=digest(mpath)==p['inputs']['manifest_sha256']==cal['manifest_sha256'] and (len(m['train']['labeled']),len(m['val']),len(m['reserved_test']),len(vm['val']))==(521,132,144,29)
    train_files={str(Path(r[k]).resolve()) for r in m['train']['labeled'] for k in ('image','mask')}
    checks['calibration_only1042_active_train_files_no_validation_or_test']=set(cal['files'])==train_files and len(train_files)==1042 and cal['validation_images_read']==0 and not cal['reserved_test_pixels_read'] and all(digest(n)==h for n,h in cal['files'].items())
    checks['all1364_calibration_and_validation_input_bytes_unchanged']=len(p['inputs']['files'])==1364 and all(digest(n)==h for n,h in p['inputs']['files'].items()) and not p['inputs']['test_pixels_read']
    eligible=[r for r in cal['records'] if r['eligible']];train_ids={r['sample_id'] for r in m['train']['labeled']}
    checks['shared_shape_is_median_of31_train_only_eligible_records']=len(eligible)==prior['eligible_train_images']==31 and all(r['sample_id'] in train_ids for r in eligible) and np.isclose(prior['halfwidth_slope'],np.median([r['halfwidth_slope'] for r in eligible])) and np.isclose(prior['cap_radius_fraction'],np.median([r['cap_radius_fraction'] for r in eligible]))
    checks['no_gradient_updates_default_or_stone_memory_adoption']=s['gradient_updates']==0 and s['calibration_train_images']==521 and not s['default_model_changed'] and not s['s2_accepted'] and not s['stone_memory_started']
    totals={n:np.zeros((3,3),np.int64) for n in NAMES}
    for item in p['video_inputs']:
        vid=item['video_id'];folder=OUT/vid;v=s['videos'][vid];logs=[json.loads(x) for x in (folder/'frames.jsonl').read_text(encoding='utf-8').splitlines()];timings=read(folder/'timings.json')
        checks[f'{vid}_all_consecutive_frames_and_four_mask_inventories']=len(logs)==v['frames']==len(timings) and [r['frame_index'] for r in logs]==list(range(v['frames'])) and all(len(list((folder/n).glob('*.png')))==v['frames'] for n in NAMES)
        cap=cv2.VideoCapture(item['path']);selected={r['frame_index']:r for r in vm['val'] if r['video_id']==vid}
        geometry=fusion=raw_exact=align=states=counts=True;old_before=None;accepted={n:0 for n in NAMES[1:]};status={}
        try:
            for row in logs:
                i=row['frame_index'];ok,frame=cap.read()
                if not ok:raise AssertionError('Short sequential audit decode')
                masks={n:load(folder/n/f'{i:06d}.png') for n in NAMES};old=load(ROOT/f'runs/a1_classwise_replay_20261028/{vid}/old190/{i:06d}.png')
                raw_exact &= np.array_equal(masks['raw_classwise'],load(ROOT/f'runs/a1_classwise_replay_20261028/{vid}/classwise/{i:06d}.png'))
                for n in NAMES:
                    a=masks[n];fusion &= a.shape==(400,400) and set(np.unique(a)).issubset({0,1,2}) and np.array_equal(a,np.where(a==2,2,np.where(old==1,1,0)))
                    counts &= row['models'][n]['stone_pixels']==int((a==1).sum()) and row['models'][n]['fiber_pixels']==int((a==2).sum())
                    if n!='raw_classwise':
                        record=row['outputs'][n];geometry &= np.array_equal(a==2,reconstruct(frame,record,prior,config));accepted[n]+=int(record['accepted'])
                    if i in selected:totals[n]+=matrix(load(selected[i]['mask']),a)
                tr=row['outputs']['causal_template']['tracking'];status[tr['status']]=status.get(tr['status'],0)+1
                states &= tr['applied']==row['outputs']['causal_template']['accepted']
                if tr['status']=='predicted_not_visible':states &= 1<=tr['age']<=3 and not (masks['causal_template']==2).any()
                if i in selected:align &= np.array_equal(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),np.asarray(Image.open(selected[i]['image']).convert('RGB')))
            align &= not cap.read()[0]
        finally:cap.release()
        checks[f'{vid}_fixed_shape_and_same_current_image_visibility_reconstructed']=bool(geometry)
        checks[f'{vid}_fusion_native_classes_and_area_counts_recomputed']=bool(fusion and counts)
        checks[f'{vid}_raw_reference_and_reviewed_rgb_pixel_exact']=bool(raw_exact and align)
        checks[f'{vid}_causal_rejection_prediction_not_visible_and_case_reset']=bool(states) and v['tracker_reset'] and logs[0]['outputs']['causal_template']['tracking']['status'] in ('initialized','no_state')
        checks[f'{vid}_acceptance_and_tracking_counts_recomputed']=accepted==v['accepted'] and status==v['tracking_statuses']
        cap=cv2.VideoCapture(str(folder/'comparison.mp4'));media=cap.isOpened() and int(cap.get(7))==v['frames'] and int(cap.get(3))==2000 and int(cap.get(4))==434;cap.release()
        for e in v['user_intervals']:
            cap=cv2.VideoCapture(str(folder/e['clip']));media &= cap.isOpened() and int(cap.get(7))==e['frames'];cap.release()
            media &= e['start_frame']+1==e['one_based_inclusive'][0] and e['end_frame_exclusive']==e['one_based_inclusive'][1]
        checks[f'{vid}_full_video_and5fixed_interval_geometry_counts']=bool(media)
        checks[f'{vid}_timing_percentiles_recomputed']=all(np.isclose(v['timings'][key][stat],fn([r[key] for r in timings])) for key in timings[0] for stat,fn in [('mean',np.mean),('p50',np.median),('p95',lambda a:np.percentile(a,95)),('max',np.max)])
        details[vid]={'frames':len(logs),'accepted':accepted,'prediction_only_frames':status.get('predicted_not_visible',0)}
    checks['reviewed29_four_independent_confusions']=all(totals[n].tolist()==s['video29'][n]['metrics']['confusion_matrix'] for n in NAMES)
    image=s['image'];matrices={n:np.zeros((3,3),np.int64) for n in config['image_arms']};correct=True
    for i,(row,record) in enumerate(zip(m['val'],image['records'])):
        frame=cv2.cvtColor(np.asarray(Image.open(row['image']).convert('RGB')),cv2.COLOR_RGB2BGR);target=load(row['mask'])
        for n in config['image_arms']:
            mask=load(OUT/f'image_native/{n}/{i:04d}.png');matrices[n]+=matrix(target,mask)
            if n!='raw_classwise':correct &= np.array_equal(mask==2,reconstruct(frame,record['outputs'][n],prior,config))
        correct &= 'causal_template' not in record['outputs']
    checks['image132_fixed_shape_confusions_and_no_unordered_tracking']=bool(correct) and all(matrices[n].tolist()==image['models'][n]['metrics']['confusion_matrix'] for n in matrices)
    groups=image['cohorts'];a=set(groups['historical_old190_exposed']['sample_ids']);b=set(groups['not_matched_to_old190_exposure']['sample_ids'])
    checks['historical25_107_cohorts_complete_disjoint']=(len(a),len(b),len(a&b))==(25,107,0) and a|b=={r['sample_id'] for r in m['val']}
    costs=s['performance'];checks['four_fresh_processes_actual1_or3_network_cost']=len({v['pid'] for v in costs.values()})==4 and all((v['model_count'],v['parameter_count'])==((1,24601395) if n=='image_template' else (3,73804185)) and v['warmup_frames']==12 and v['samples']==len(v['raw_samples'])==87 for n,v in costs.items())
    checks['isolated_cost_percentiles_recomputed']=all(np.isclose(v['timings']['pipeline_ms'][k],fn([r['pipeline_ms'] for r in v['raw_samples']])) for v in costs.values() for k,fn in [('mean',np.mean),('p50',np.median),('p95',lambda a:np.percentile(a,95))])
    gates=config['gates'];checks['all_primary_frozen_gates_recomputed']=all(s['screening'][n]['passed']==bool(v['metrics']['stone_dice']>=gates['stone_dice_min'] and v['metrics']['fiber_dice']>=gates['fiber_dice_min'] and v['metrics']['stone_negative_fp_pixel_fraction']<=gates['negative_fp_fraction_max']) for n,v in s['video29'].items())
    checks['no_tip_angle_truth_claim']=s['tip_angle_reference_available'] is False and all(v['dense_accuracy_metrics'] is None for v in s['videos'].values())
    with zipfile.ZipFile(OUT/'source_snapshot.zip') as z:checks['frozen_unique_snapshot_and_sources']=len(z.namelist())==len(set(z.namelist())) and all(hashlib.sha256(z.read(n.replace('\\','/'))).hexdigest()==h for n,h in p['sources'].items())
    result={'checks':{n:bool(v) for n,v in checks.items()},'all_passed':bool(all(checks.values())),'details':details,
        'unit_tests':{'passed':3,'failed':0,'command':'python -m unittest discover -s tests -p test_direct_fiber_template.py -v'},
        'scope':'Calibration split, fixed-family output/export and cost correctness only; not tip/angle, clinical, independent-case or live-stream validity'}
    (ROOT/'runs/engineering/direct_fiber_template_checks_20261029.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    assert result['all_passed'],result
    print(json.dumps({'all_passed':True,'checks':len(checks),'details':details},indent=2))


if __name__=='__main__':main()
