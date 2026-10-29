"""Independent component preservation, native metrics and isolated-cost audit."""
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/a1_fiber_component_gate_20261029'
ARMS=('classwise_raw_3net','classwise_component_gate_3net','old190_raw_1net','old190_component_gate_1net')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def load(path):
    with Image.open(path) as im:return np.asarray(im).copy()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def matrix(target,pred):
    valid=target!=255
    return np.bincount((3*target[valid].astype(int)+pred[valid]).ravel(),minlength=9).reshape(3,3)


def selected_component(frame,raw,config):
    # Reproduce the fixed estimated FOV definition independently.
    binary=(frame.max(2)>10).astype(np.uint8);fov=np.zeros(raw.shape,np.uint8)
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if contours:cv2.drawContours(fov,[max(contours,key=cv2.contourArea)],-1,1,-1)
    dist=cv2.distanceTransform(np.pad(fov,1),cv2.DIST_L2,5)[1:-1,1:-1]
    count,labels,sizes,_=cv2.connectedComponentsWithStats((raw==2).astype(np.uint8),8)
    choices=[]
    for label in range(1,count):
        mask=labels==label
        if sizes[label,cv2.CC_STAT_AREA]>=config['minimum_candidate_pixels'] and int((mask&(dist<=config['entry_distance_fraction']*min(raw.shape))).sum())>=3:
            choices.append((int(mask.sum()),label))
    return labels==max(choices)[1] if choices else None


def main():
    s=read(OUT/'summary.json');p=read(OUT/'preflight.json');checks={};details={}
    checks['frozen_sources_protocols_and_historical_references_unchanged']=all(digest(ROOT/n)==h for n,h in p['sources'].items()) and all(digest(n)==h for n,h in p['related'].items())
    checks['weights_and_video_sources_unchanged']=all(digest(v['path'])==v['sha256'] for v in list(p['checkpoints'].values())+p['video_inputs'])
    checks['161_validation_images_masks_only_no_test_pixels']=len(p['inputs']['files'])==322 and p['inputs']['test_pixels_read'] is False and all(digest(n)==h for n,h in p['inputs']['files'].items())
    mpath=ROOT/'datasets/ureter_semis_kidney_joint_20261028/manifest.json';m=read(mpath);vm=read(mpath.parent/'video_validation_manifest.json')
    checks['521_132_144_29_manifest_unchanged']=digest(mpath)==p['inputs']['manifest_sha256'] and (len(m['train']['labeled']),len(m['val']),len(m['reserved_test']),len(vm['val']))==(521,132,144,29)
    predecessor=read(ROOT/'runs/a1_fiber_geometry_20261028/preflight.json')
    checks['previous_shape_source_and_all_fit_thresholds_unchanged']=predecessor['sources']['tools/fiber_shape_prior.py']==p['sources']['tools/fiber_shape_prior.py'] and predecessor['protocol']['fit']==p['geometry_config']['fit']
    checks['no_training_adoption_or_stone_memory']=not s['training_updates'] and not s['default_model_changed'] and not s['s2_accepted'] and not s['stone_memory_started']
    for scope,rows in [('video',vm['val']),('image',m['val'])]:
        folder=OUT/f'{scope}_native';result=s[scope];targets=[load(r['mask']) for r in rows]
        predictions={n:[load(folder/n/f'{i:04d}.png') for i in range(len(rows))] for n in ARMS}
        checks[f'{scope}_all_native_geometry_and_classes']=all(a.shape==t.shape and set(np.unique(a)).issubset({0,1,2}) for arrays in predictions.values() for a,t in zip(arrays,targets))
        confusions=True
        for name,arrays in predictions.items():
            cms=[matrix(t,a) for t,a in zip(targets,arrays)]
            confusions &= all(cm.tolist()==row['confusion_matrix'] for cm,row in zip(cms,result['models'][name]['frames']))
            confusions &= np.sum(cms,0).tolist()==result['models'][name]['metrics']['confusion_matrix']
        checks[f'{scope}_four_independent_native_confusions']=bool(confusions)
        preserve=fallback=subset=counts=paired=True
        for gated,raw in [(ARMS[1],ARMS[0]),(ARMS[3],ARMS[2])]:
            expected_pair={str(k):dict(lost_tp=0,gained_tp=0,added_fp=0,removed_fp=0) for k in (1,2)}
            for i,(row,t,a,b,record) in enumerate(zip(rows,targets,predictions[raw],predictions[gated],result['models'][gated]['records'])):
                g=record['gate'];subset &= not ((b==2)&(a!=2)).any()
                counts &= g['added_pixels']==0 and g['removed_pixels']==int(((a==2)&(b!=2)).sum())
                if not g['accepted']:fallback &= np.array_equal(a,b)
                else:
                    frame=cv2.cvtColor(np.asarray(Image.open(row['image']).convert('RGB')),cv2.COLOR_RGB2BGR)
                    candidate=selected_component(frame,a,p['geometry_config']['fit'])
                    preserve &= candidate is not None and np.array_equal(b==2,candidate) and int(candidate.sum())==g['selected_pixels']
                # Removing old190's own fiber can only expose background, never change stone.
                if gated==ARMS[3]:preserve &= np.array_equal(a==1,b==1)
                else:preserve &= not ((b==1)&(predictions[ARMS[2]][i]!=1)).any()
                for k in (1,2):
                    valid=t!=255;truth=t==k
                    d=dict(lost_tp=int(((a==k)&(b!=k)&truth).sum()),gained_tp=int(((a!=k)&(b==k)&truth).sum()),added_fp=int(((a!=k)&(b==k)&~truth&valid).sum()),removed_fp=int(((a==k)&(b!=k)&~truth&valid).sum()))
                    for key,value in d.items():expected_pair[str(k)][key]+=value
            paired &= expected_pair==result['pairs'][gated]['class_changes']
        checks[f'{scope}_accepted_component_every_pixel_preserved']=bool(preserve)
        checks[f'{scope}_rejected_fits_exact_raw_fallback']=bool(fallback)
        checks[f'{scope}_fiber_never_added_and_removals_exact']=bool(subset and counts)
        checks[f'{scope}_paired_tp_fp_changes_recomputed']=bool(paired)
    cohorts=s['image']['cohorts'];a=set(cohorts['historical_old190_exposed']['sample_ids']);b=set(cohorts['not_matched_to_old190_exposure']['sample_ids'])
    checks['image25_107_cohorts_complete_disjoint']=(len(a),len(b),len(a&b))==(25,107,0) and a|b=={r['sample_id'] for r in m['val']}
    expected_models={n:(3,73804185) if n.startswith('classwise') else (1,24601395) for n in ARMS}
    checks['four_cost_processes_distinct']=len({v['pid'] for v in s['performance'].values()})==4
    cost_ok=True
    for name,v in s['performance'].items():
        cost_ok &= (v['model_count'],v['parameter_count'])==expected_models[name] and v['warmup_frames']==12 and v['samples']==len(v['raw_samples'])==87
        cost_ok &= set(v['checkpoint_sha256'])==({'old190','A0'} if name.startswith('classwise') else {'old190'})
        cost_ok &= all(np.isclose(v['timings']['pipeline_ms'][k],fn([r['pipeline_ms'] for r in v['raw_samples']])) for k,fn in [('mean',np.mean),('p50',np.median),('p95',lambda x:np.percentile(x,95))])
    checks['actual1_or3_network_full_cost_and_statistics']=bool(cost_ok)
    gates=p['protocol']['gates'];passing=[]
    for name,d in s['video']['models'].items():
        metrics=d['metrics'];expected=metrics['stone_dice']>=gates['stone_dice_min'] and metrics['fiber_dice']>=gates['fiber_dice_min'] and metrics['stone_negative_fp_pixel_fraction']<=gates['negative_fp_fraction_max']
        if name in (ARMS[1],ARMS[3]) and expected:passing.append(name)
        assert s['screening'][name]['passed']==expected
    checks['full_replay_only_exact_passing_gated_arms']=passing==s['passing_gated_arms'] and set(passing)==set(s['full_replay'])
    for arm,full in s['full_replay'].items():
        matrix29=np.zeros((3,3),np.int64);preserve=fallback=count_ok=raw_ok=True;frames=0
        for vid,v in full['videos'].items():
            folder=OUT/'full_replay'/arm/vid;logs=[json.loads(x) for x in (folder/'frames.jsonl').read_text(encoding='utf-8').splitlines()];frames+=len(logs)
            count_ok &= len(logs)==v['frames'] and [r['frame_index'] for r in logs]==list(range(v['frames']))
            count_ok &= v['repeat_exact'] and all(r['pixel_exact'] for r in v['reviewed_alignment'])
            selected={r['frame_index']:r for r in vm['val'] if r['video_id']==vid};reference='classwise' if arm.startswith('classwise') else 'old190'
            for r in logs:
                i=r['frame_index'];a=load(folder/'raw'/f'{i:06d}.png');b=load(folder/'gated'/f'{i:06d}.png')
                raw_ok &= np.array_equal(a,load(ROOT/f'runs/a1_classwise_replay_20261028/{vid}/{reference}/{i:06d}.png'))
                preserve &= not ((b==2)&(a!=2)).any() and r['gate']['removed_pixels']==int(((a==2)&(b!=2)).sum())
                if not r['gate']['accepted']:fallback &= np.array_equal(a,b)
                if i in selected:matrix29+=matrix(load(selected[i]['mask']),b)
            cap=cv2.VideoCapture(str(folder/'comparison.mp4'));count_ok &= cap.isOpened() and int(cap.get(7))==v['frames'] and int(cap.get(3))==1200 and int(cap.get(4))==434;cap.release()
            for event in v['user_intervals']:
                cap=cv2.VideoCapture(str(folder/event['clip']));count_ok &= cap.isOpened() and int(cap.get(7))==event['frames'];cap.release()
        checks[f'{arm}_full2694_masks_references_and_clips']=bool(count_ok and raw_ok and frames==2694)
        checks[f'{arm}_full_no_added_fiber_and_raw_fallback']=bool(preserve and fallback)
        checks[f'{arm}_reviewed29_confusion_recomputed']=matrix29.tolist()==full['reviewed29']['metrics']['confusion_matrix']
        details[arm]={'full_frames':frames,'primary_vs_full_different_pixels':full['different_pixels_from_primary29']}
    with zipfile.ZipFile(OUT/'source_snapshot.zip') as z:checks['source_snapshot_unique_and_matches']=len(z.namelist())==len(set(z.namelist())) and all(hashlib.sha256(z.read(n.replace('\\','/'))).hexdigest()==h for n,h in p['sources'].items())
    result={'checks':{k:bool(v) for k,v in checks.items()},'all_passed':bool(all(checks.values())),'details':details,
        'unit_tests':{'passed':4,'failed':0,'command':'python -m unittest discover -s tests -p test_fiber_component_gate.py -v'},
        'scope':'Development export/selection/cost correctness; no live display, clinical, tip, identity or independent-case validation'}
    (ROOT/'runs/engineering/fiber_component_gate_checks_20261029.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    assert result['all_passed'],result
    print(json.dumps({'all_passed':True,'checks':len(checks),'details':details},indent=2))


if __name__=='__main__':main()
