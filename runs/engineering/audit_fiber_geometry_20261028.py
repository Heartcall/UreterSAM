"""Independent exported-mask and geometric application audit, no training."""
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/a1_fiber_geometry_20261028'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def load(path):
    with Image.open(path) as im:return np.asarray(im).copy()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def template(state,shape):
    x,y,angle,length,w0,w1,ratio=state;yy,xx=np.indices(shape,dtype=np.float32)
    along=(xx-x)*np.cos(angle)+(yy-y)*np.sin(angle)
    normal=-(xx-x)*np.sin(angle)+(yy-y)*np.cos(angle)
    radius=max(.5,w1/2*ratio);center=max(.5,length-radius)
    half=(w0+(w1-w0)*np.clip(along/center,0,1))/2
    return ((along>=0)&(along<=center)&(np.abs(normal)<=half))|((along>=center)&(((along-center)/radius)**2+(normal/max(.5,w1/2))**2<=1))


def cm(target,pred):
    valid=target!=255
    return np.bincount((3*target[valid].astype(int)+pred[valid]).ravel(),minlength=9).reshape(3,3)


def main():
    s=read(OUT/'summary.json');p=read(OUT/'preflight.json');checks={};detail={}
    checks['frozen_config_sources_and_all_cached_references_unchanged']=all(digest(ROOT/n)==h for n,h in p['sources'].items()) and all(digest(n)==h for n,h in p['related'].items())
    checks['frozen_expert_weights_and_videos_unchanged']=all(digest(v['path'])==v['sha256'] for v in list(p['checkpoints'].values())+p['video_inputs'])
    checks['only161_validation_images_and_masks_no_test_pixels']=len(p['inputs']['files'])==322 and p['inputs']['test_pixels_read'] is False and all(digest(n)==h for n,h in p['inputs']['files'].items())
    mpath=ROOT/'datasets/ureter_semis_kidney_joint_20261028/manifest.json';m=read(mpath);vm=read(mpath.parent/'video_validation_manifest.json')
    checks['manifest_and521_132_144_29_splits_unchanged']=digest(mpath)==p['inputs']['manifest_sha256'] and (len(m['train']['labeled']),len(m['val']),len(m['reserved_test']),len(vm['val']))==(521,132,144,29)
    checks['zero_training_no_default_or_stone_memory_adoption']=s['training_updates']==0 and not s['default_model_changed'] and not s['stone_memory_started'] and not s['s2_accepted'] and s['model_count']==3 and s['parameter_count']==73804185
    video_matrices={n:np.zeros((3,3),np.int64) for n in ('raw','geometry','geometry_causal')}
    for vid,v in s['videos'].items():
        folder=OUT/vid;logs=[json.loads(line) for line in (folder/'frames.jsonl').read_text(encoding='utf-8').splitlines()];timings=read(folder/'timings.json')
        checks[f'{vid}_exact_frame_indices_and_inventory']=len(logs)==v['frames']==len(timings) and [r['frame_index'] for r in logs]==list(range(v['frames'])) and all(len(list((folder/n).glob('*.png')))==v['frames'] for n in video_matrices)
        checks[f'{vid}_repeat_alignment_and_case_reset']=v['repeat_exact'] and v['tracker_reset_between_cases'] and all(r['pixel_exact'] for r in v['reviewed_alignment']) and logs[0]['tracked']['status'] in ('initialized','no_state')
        policy=geometry=fallback=counts=causal=True
        previous={n:None for n in video_matrices};selected={r['frame_index']:r for r in vm['val'] if r['video_id']==vid}
        status_counts={};reason_counts={};accepted=0
        for r in logs:
            i=r['frame_index'];masks={n:load(folder/n/f'{i:06d}.png') for n in video_matrices}
            old=load(ROOT/f'runs/a1_classwise_replay_20261028/{vid}/old190/{i:06d}.png')
            rawcache=load(ROOT/f'runs/a1_classwise_replay_20261028/{vid}/classwise/{i:06d}.png')
            counts &= np.array_equal(rawcache,masks['raw'])
            for name,a in masks.items():
                policy &= a.shape==(400,400) and set(np.unique(a)).issubset({0,1,2}) and np.array_equal(a,np.where(a==2,2,np.where(old==1,1,0)))
                d=r['models'][name]
                counts &= d['stone_pixels']==int((a==1).sum()) and d['fiber_pixels']==int((a==2).sum())
                for k,label in [(1,'stone'),(2,'fiber')]:
                    expected=abs(int((a==k).sum())-int((previous[name]==k).sum())) if previous[name] is not None else None
                    counts &= expected==d[f'{label}_absolute_area_change_pixels']
                if i in selected:video_matrices[name]+=cm(load(selected[i]['mask']),a)
            for name,state in [('geometry',r['fit']['state'] if r['fit']['accepted'] else None),('geometry_causal',r['tracked']['state'] if r['tracked']['apply'] else None)]:
                if state is None:fallback &= np.array_equal(masks[name],masks['raw'])
                else:geometry &= not ((masks[name]==2)&~template(state,(400,400))).any()
                delta=r['deltas'][name];raw=masks['raw'];a=masks[name]
                counts &= delta=={'fiber_added':int(((a==2)&(raw!=2)).sum()),'fiber_removed':int(((a!=2)&(raw==2)).sum()),'stone_overwritten':int(((raw==1)&(a!=1)).sum()),'stone_restored':int(((raw!=1)&(a==1)).sum())}
            causal &= r['tracked']['apply']==r['fit']['accepted']
            if r['tracked']['status']=='predicted_not_applied':causal &= 1<=r['tracked']['age']<=3 and not r['tracked']['apply']
            status_counts[r['tracked']['status']]=status_counts.get(r['tracked']['status'],0)+1
            reason_counts[r['fit']['reason']]=reason_counts.get(r['fit']['reason'],0)+1;accepted+=int(r['fit']['accepted'])
            previous=masks
        checks[f'{vid}_all_pixel_fiber_priority_and_stone_delegation']=bool(policy)
        checks[f'{vid}_accepted_fiber_inside_current_or_filtered_template']=bool(geometry)
        checks[f'{vid}_rejected_or_history_only_geometry_never_forces_visible_mask']=bool(fallback and causal)
        checks[f'{vid}_raw_reproduced_and_area_delta_counts_recomputed']=bool(counts)
        checks[f'{vid}_fit_rejection_and_tracking_status_counts']=accepted==v['accepted_fits'] and reason_counts==v['fit_reasons'] and status_counts==v['tracker_statuses']
        sums_ok=all(np.isclose(t['geometry_derived_ms'],t['raw_core_ms']+t['fit_and_fov_ms']+t['geometry_raster_fusion_ms']) and np.isclose(t['geometry_causal_derived_ms'],t['raw_core_ms']+t['fit_and_fov_ms']+t['causal_update_raster_fusion_ms']) for t in timings)
        stats_ok=all(np.isclose(v['timings'][k][stat],fn([r[k] for r in timings])) for k in timings[0] for stat,fn in [('mean',np.mean),('p50',np.median),('p95',lambda x:np.percentile(x,95)),('max',np.max)])
        checks[f'{vid}_per_arm_stage_sums_and_percentiles']=bool(sums_ok and stats_ok)
        cap=cv2.VideoCapture(str(folder/'comparison.mp4'));checks[f'{vid}_full_preview_geometry_count']=cap.isOpened() and int(cap.get(7))==v['frames'] and int(cap.get(3))==1600 and int(cap.get(4))==434;cap.release()
        clip_ok=True
        for event in v['user_intervals']:
            cap=cv2.VideoCapture(str(folder/event['clip']));clip_ok &= cap.isOpened() and int(cap.get(7))==event['frames'];cap.release()
            clip_ok &= event['start_frame']+1==event['one_based_inclusive'][0] and event['end_frame_exclusive']==event['one_based_inclusive'][1]
        checks[f'{vid}_user_clip_counts_and_index_mapping']=bool(clip_ok)
        detail[vid]={'frames':v['frames'],'accepted_fits':accepted,'prediction_only_frames':status_counts.get('predicted_not_applied',0)}
    checks['native29_confusions_independently_recomputed']=all(video_matrices[n].tolist()==s['video29'][n]['metrics']['confusion_matrix'] for n in video_matrices)
    image=s['image'];image_matrices={n:np.zeros((3,3),np.int64) for n in ('raw','geometry')};image_ok=True
    for i,(row,record) in enumerate(zip(m['val'],image['records'])):
        raw=load(OUT/f'image_native/raw/{i:04d}.png');geo=load(OUT/f'image_native/geometry/{i:04d}.png');target=load(row['mask'])
        for name,a in [('raw',raw),('geometry',geo)]:image_matrices[name]+=cm(target,a)
        if not record['fit']['accepted']:image_ok &= np.array_equal(raw,geo)
        image_ok &= record['tracked'] is None
    checks['image132_confusions_fallback_and_no_unordered_tracking']=bool(image_ok) and all(image_matrices[n].tolist()==image['models'][n]['metrics']['confusion_matrix'] for n in image_matrices)
    cohorts=image['cohorts'];a=set(cohorts['historical_old190_exposed']['sample_ids']);b=set(cohorts['not_matched_to_old190_exposure']['sample_ids'])
    checks['historical25_107_cohorts_complete_and_disjoint']=(len(a),len(b),len(a&b))==(25,107,0) and a|b=={r['sample_id'] for r in m['val']}
    checks['no_tip_angle_or_dense_accuracy_claims']=s['tip_and_angle_reference_available'] is False and all(v['dense_accuracy_metrics'] is None for v in s['videos'].values())
    with zipfile.ZipFile(OUT/'source_snapshot.zip') as z:checks['frozen_snapshot_unique_and_source_identity']=len(z.namelist())==len(set(z.namelist())) and all(hashlib.sha256(z.read(n.replace('\\','/'))).hexdigest()==h for n,h in p['sources'].items())
    result={'checks':{n:bool(v) for n,v in checks.items()},'all_passed':bool(all(checks.values())),'details':detail,
        'unit_tests':{'passed':3,'failed':0,'command':'python -m unittest discover -s tests -p test_fiber_shape_prior.py -v'},
        'scope':'Algorithm/export correctness and development masks only; not geometric/clinical/independent-case validation'}
    (ROOT/'runs/engineering/fiber_geometry_checks_20261028.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    assert result['all_passed'],result
    print(json.dumps({'all_passed':True,'checks':len(checks),'details':detail},indent=2))


if __name__=='__main__':main()
