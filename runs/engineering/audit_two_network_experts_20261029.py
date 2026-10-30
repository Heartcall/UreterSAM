"""Independent post-run checks; no model construction or training."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image
from tools.fiber_reference_workspace import read, sha256, load_mask
from tools.run_classwise_expert_control import summarize
from tools.diagnose_joint_failures import paired_changes
from ureter_semis.experiment import atomic_json

OUT = ROOT / 'runs/a1_two_network_experts_20261029'
checks = {}
def check(name, value):
    checks[name] = bool(value)
    if not value: raise AssertionError(name)
def raw(a): return hashlib.sha256(a.tobytes()).hexdigest()
def decode_count(path, shape):
    cap = cv2.VideoCapture(str(path)); n = 0
    while True:
        ok, a = cap.read()
        if not ok: break
        assert a.shape[:2] == shape
        n += 1
    cap.release()
    return n

def main():
    pre = read(OUT/'preflight.json'); cfg = pre['config']; result = read(OUT/'summary.json')
    check('job_completed', read(OUT/'job.json')['phase'] == 'completed')
    check('frozen_config', sha256(ROOT/'configs/compare_two_network_experts_20261029_v2.json') == pre['config_sha256'])
    check('sources_unchanged', all(sha256(ROOT/p)==h for p,h in pre['sources'].items()))
    check('related_caches_manifests_unchanged', all(sha256(p)==h for p,h in pre['related'].items()))
    check('approved_inputs_unchanged', all(sha256(p)==h for p,h in pre['inputs']['files'].items()))
    check('weights_videos_unchanged', all(sha256(x['path'])==x['sha256'] for x in [*pre['checkpoints'].values(), *pre['videos']]))
    with zipfile.ZipFile(OUT/'source_snapshot.zip') as z:
        check('source_archive_matches', all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in pre['sources'].items()))
        check('config_archive_matches', hashlib.sha256(z.read('configs/compare_two_network_experts_20261029_v2.json')).hexdigest()==pre['config_sha256'])
    check('legacy_gates_unchanged', cfg['legacy_original29_gates']==read(ROOT/cfg['old_protocol'])['gates'])
    models = read(OUT/'model_identities.json'); base = models['classwise3']['individual_model_sha256']
    check('member1_exact_subset', models['member1_2net']['individual_model_sha256']==base[:2])
    check('member2_exact_subset', models['member2_2net']['individual_model_sha256']==[base[0],base[2]])
    rows_by_scope = {}
    targets_by_scope = {}
    for scope, key, count in [('video_native','video_manifest',70),('image_native','image_manifest',132)]:
        rows = read(ROOT/cfg[key])['val']; rows_by_scope[scope] = rows
        targets = [np.array(Image.open(r['mask'])) for r in rows]; targets_by_scope[scope] = targets
        check(scope+'_count', len(rows)==count)
        baseline = [np.array(Image.open(OUT/f'{scope}/classwise3/{i:04d}.png')) for i in range(count)]
        for name in cfg['schemes']:
            data = read(OUT/f'{scope}/{name}.json')
            preds = [np.array(Image.open(OUT/f'{scope}/{name}/{i:04d}.png')) for i in range(count)]
            check(scope+'_'+name+'_recomputed_summary', summarize(rows,targets,preds)==data['summary'])
            paired_ok = True; probabilities_ok = True
            for r,t,p,b in zip(data['records'], targets, preds, baseline):
                paired_ok &= raw(p)==r['mask_raw_sha256'] and all(paired_changes(t,b,p,c)==r['paired_'+label] for c,label in [(1,'stone'),(2,'fiber')])
                keyp = 'a0_native' if name=='classwise3' else 'a0_member'+str(cfg['models'][name]['fiber_member'][-1])+'_native'
                with np.load(r['probability_reference']) as cache:
                    sp, fp = cache['old190_native'],cache[keyp]
                    expected=np.zeros(p.shape,np.uint8); expected[sp.argmax(0)==1]=1;expected[fp.argmax(0)==2]=2
                    probabilities_ok &= raw(sp)==r['stone_probability_raw_sha256'] and raw(fp)==r['fiber_probability_raw_sha256'] and np.array_equal(p,expected)
            check(scope+'_'+name+'_paired_pixels_hashes', paired_ok)
            check(scope+'_'+name+'_native_probabilities', probabilities_ok and data['max_native_probability_error_vs_reference']==0)
        old = read(ROOT/f'runs/a1_fiber_blue_aplus_20261029/{scope}/summary.json')['models']['raw']
        check(scope+'_historical_baseline_exact', read(OUT/f'{scope}/classwise3.json')['summary']==old)
    video_rows=rows_by_scope['video_native']
    check('video_cohorts_43_27_29_41', [sum(r['video_id']==v for r in video_rows) for v in ['video_1','video_2']]==[43,27] and sum(r['evaluation_cohort']=='original29' for r in video_rows)==29)
    pids=[]; total=0
    for vid,n in [('video_1',722),('video_2',1972)]:
        original=[json.loads(s) for s in (ROOT/f'runs/a1_fiber_tip_d0_20261029/full_video/{vid}/frames.jsonl').read_text().splitlines()]
        ids={r['frame_index']:i for i,r in enumerate(video_rows) if r['video_id']==vid}
        for name in cfg['schemes']:
            folder=OUT/f'full_video/{name}/{vid}'; data=read(folder/'summary.json'); pids.append(data['pid']); times=read(folder/'timings.json')
            logs=[json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()]
            changed=0; exact=0
            for i,r in enumerate(logs):
                a=np.array(Image.open(folder/f'masks/{i:06d}.png'))
                assert r['frame_index']==i and raw(a)==r['mask_raw_sha256']
                assert r['source_rgb_raw_sha256']==original[i]['rgb_raw_bytes_sha256']
                assert r['stone_pixels']==int((a==1).sum()) and r['fiber_pixels']==int((a==2).sum())
                assert r['tip_metrics'] is None and r['unlabelled_accuracy'] is None
                changed+=int(raw(a)!=original[i]['mask_raw_bytes_sha256'])
                if i in ids:
                    assert np.array_equal(a,np.array(Image.open(OUT/f'video_native/{name}/{ids[i]:04d}.png')))
                    exact+=1
            label=name+'_'+vid
            check(label+'_complete_native_masks_and_rgb', len(logs)==n and len(list((folder/'masks').glob('*.png')))==n)
            check(label+'_selected_static_and_history', exact==len(ids) and changed==data['changed_frames_vs_classwise3'])
            check(label+'_actual_model_budget', data['networks']==models[name]['networks'] and data['parameters']==models[name]['parameters'] and data['individual_model_sha256']==models[name]['individual_model_sha256'])
            check(label+'_timing_percentiles', len(times)==n and all(abs(float(np.percentile([t[k] for t in times],q))-v[p])<1e-10 for k,v in data['timings'].items() for q,p in [(50,'p50'),(95,'p95')]))
            check(label+'_throughput', abs(n/data['wall_seconds']-data['offline_fps_including_all_exports'])<1e-10)
            check(label+'_replay', sha256(folder/'replay.mp4')==data['replay_sha256'] and decode_count(folder/'replay.mp4',(434,800))==n)
            total+=n
        c=result['comparison_videos'][vid]
        check(vid+'_four_panel_complete', decode_count(c['path'],(434,1600))==n and sha256(c['path'])==c['sha256'])
    check('six_distinct_processes_8082_frames', len(set(pids))==6 and total==8082)
    check('legacy_pass_strict_nonregression_fail_both', all(v['legacy_original29_screening']['passed'] and not v['current_nonregression_passed'] for v in result['static_quality'].values()))
    check('no_adoption_training_or_whole_video_claim', not result['adopted'] and not result['default_changed'] and not result['stone_memory_started'] and result['training_updates']==result['train_pixels_read']==result['reserved_test_pixels_read']==0 and result['whole_video_accuracy_estimates'] is None and result['tip_metrics'] is None)
    # Same preexisting reference-derived proxy, evaluated after all timed workers.
    spec=importlib.util.spec_from_file_location('centerline',ROOT/'runs/engineering/fiber_centerline_proxy_20261029.py'); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    center={'role':'supplementary reference-mask-derived visible centerline coverage; not anatomical tip truth or adoption gate','method_source_sha256':sha256(Path(module.__file__)),'sets':{}}
    check('centerline_helper_self_tests',module.self_test()==3)
    for scope,targets in targets_by_scope.items():
        sums={name:[0,0] for name in cfg['schemes']}; frames=[]
        for i,t in enumerate(targets):
            valid=cv2.erode((t!=255).astype(np.uint8),np.ones((7,7),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
            skeleton=module.thin(t==2)&valid; count=int(skeleton.sum()); f={'sample_id':rows_by_scope[scope][i]['sample_id'],'reference_pixels':count,'covered':{}}
            for name in cfg['schemes']:
                p=np.array(Image.open(OUT/f'{scope}/{name}/{i:04d}.png')); covered=int((skeleton&(p==2)).sum());sums[name][0]+=covered;sums[name][1]+=count;f['covered'][name]=covered
            frames.append(f)
        center['sets'][scope]={'models':{k:{'covered':v[0],'reference_pixels':v[1],'coverage':v[0]/v[1] if v[1] else None} for k,v in sums.items()},'frames':frames}
    atomic_json(OUT/'centerline_coverage_proxy.json',center)
    atomic_json(OUT/'engineering_checks.json',{'passed':sum(checks.values()),'checks':checks,'auditor_sha256':sha256(Path(__file__)),'full_native_masks_rechecked':total,'static_masks_rechecked':606,'model_forwards_in_audit':0})
    print(json.dumps({'passed':sum(checks.values()),'full_frames':total,'static':606,'centerline':{s:d['models'] for s,d in center['sets'].items()}}))

if __name__=='__main__':main()
