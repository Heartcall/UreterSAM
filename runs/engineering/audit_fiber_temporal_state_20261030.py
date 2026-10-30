"""Post-benchmark audit: immutable masks, independent legacy replay, gated causality."""
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image
from tools.fiber_reference_workspace import read,sha256
from tools.fiber_shape_prior import CausalShapeTracker
from tools.fiber_temporal_state import GeometryStateControl,valid_geometry
from tools.run_classwise_expert_control import summarize
from ureter_semis.experiment import atomic_json
OUT=ROOT/'runs/a1_fiber_temporal_state_gate_20261030'
checks={}
def check(name,value):
    checks[name]=bool(value)
    if not value:raise AssertionError(name)
def raw(a):return hashlib.sha256(a.tobytes()).hexdigest()
def count_video(path,shape):
    cap=cv2.VideoCapture(str(path));n=0
    while True:
        ok,a=cap.read()
        if not ok:break
        assert a.shape[:2]==shape;n+=1
    cap.release();return n
def distribution(values):
    return {'n':len(values),'p50':float(np.percentile(values,50)),'p95':float(np.percentile(values,95)),'max':float(np.max(values))} if values else {'n':0,'p50':None,'p95':None,'max':None}
def motion(records,selected=None):
    endpoint_speed=[];angle_speed=[];length_speed=[];acceleration=[];previous_velocity=None;previous_i=None
    for i in range(1,len(records)):
        a,b=records[i-1],records[i]
        if selected is not None and not (selected[i-1] and selected[i]):previous_velocity=None;previous_i=None;continue
        if a['tracked']['geometry'] is None or b['tracked']['geometry'] is None:previous_velocity=None;previous_i=None;continue
        dt=b['timestamp_seconds']-a['timestamp_seconds'];p=np.array(a['tracked']['geometry']['tip_xy']);q=np.array(b['tracked']['geometry']['tip_xy']);velocity=(q-p)/dt
        endpoint_speed.append(float(np.linalg.norm(velocity)))
        delta=b['tracked']['state'][2]-a['tracked']['state'][2];delta=(delta+np.pi)%(2*np.pi)-np.pi
        angle_speed.append(float(abs(np.rad2deg(delta))/dt));length_speed.append(float(abs(b['tracked']['state'][3]-a['tracked']['state'][3])/dt))
        if previous_velocity is not None and previous_i==i-1:acceleration.append(float(np.linalg.norm(velocity-previous_velocity)/dt))
        previous_velocity=velocity;previous_i=i
    return {'tip_speed_px_per_s':distribution(endpoint_speed),'axis_speed_deg_per_s':distribution(angle_speed),'length_speed_px_per_s':distribution(length_speed),'tip_acceleration_px_per_s2':distribution(acceleration)}

def main():
    pre=read(OUT/'preflight.json');cfg=pre['config'];s=read(OUT/'summary.json');previous=ROOT/cfg['base']['reference_run'];legacy=read(ROOT/'configs/fiber_geometry_prior_20261028.json')
    check('job_completed',read(OUT/'job.json')['phase']=='completed')
    check('sources_frozen',all(sha256(ROOT/p)==h for p,h in pre['sources'].items()))
    check('related_and_native_reference_files_frozen',all(sha256(p)==h for p,h in pre['related'].items()))
    check('validation_rgb_labels_frozen',all(sha256(p)==h for p,h in pre['inputs']['files'].items()))
    check('weights_source_videos_frozen',all(sha256(x['path'])==x['sha256'] for x in [*pre['checkpoints'].values(),*pre['videos']]))
    with zipfile.ZipFile(OUT/'source_snapshot.zip') as z:
        check('source_archive',all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in pre['sources'].items()))
        check('config_archive',hashlib.sha256(z.read('configs/fiber_temporal_state_gate_20261030.json')).hexdigest()==pre['config_sha256'])
    for scope,key,n in [('video_native','video_manifest',70),('image_native','image_manifest',132)]:
        d=read(OUT/(scope+'.json'));rows=read(ROOT/cfg['data'][key])['val'];targets=[];preds=[]
        check(scope+'_count',len(d['records'])==n)
        for i,(row,r) in enumerate(zip(rows,d['records'])):
            target=np.array(Image.open(row['mask']));pred=np.array(Image.open(OUT/f'{scope}/shared_raw2/{i:04d}.png'))
            assert raw(pred)==r['mask_raw_sha256'];targets.append(target);preds.append(pred)
            for arm in cfg['arms']:
                state=r['states'][arm]
                assert state['apply_to_mask'] is False and state['prediction_only'] is False and state['pending_count']==0
                if arm!='raw2':
                    expected=r['fit']['state'] if r['fit']['accepted'] and valid_geometry(r['fit']['state']) else None
                    if expected is None:assert state['state'] is None
                    else:np.testing.assert_allclose(state['state'],expected,rtol=0,atol=1e-12)
        check(scope+'_summary_and_historical_match',summarize(rows,targets,preds)==d['summary']==read(previous/f'{scope}/member1_2net.json')['summary'])
        check(scope+'_groups_preserved',d['groups']==read(previous/f'{scope}/member1_2net.json')['groups'])
        check(scope+'_all_tp_fp_fn_deltas_zero',all(v==0 for c in d['paired_changes_all_arms'].values() for v in c.values()))
    selected_rows=read(ROOT/cfg['data']['video_manifest'])['val'];static=read(OUT/'video_native.json');selected={(r['video_id'],r['frame_index']):i for i,r in enumerate(selected_rows)}
    model=read(OUT/'static_model_identity.json');pids=[];total=0;analysis={'role':'uncalibrated geometry diagnostics only; motion/residual reductions do not establish accuracy','videos':{}}
    for vid,n in [('video_1',722),('video_2',1972)]:
        logs={arm:[json.loads(x) for x in (OUT/f'full_video/{arm}/{vid}/frames.jsonl').read_text().splitlines()] for arm in cfg['arms']}
        source=read(OUT/f'source_{vid}.json');base=logs['raw2'];gated_control=GeometryStateControl('gated_state_only',cfg,legacy,1/s['full_video']['raw2'][vid]['source_fps']);legacy_tracker=CausalShapeTracker(legacy['tracking'])
        check(vid+'_identical_current_fit_across_geometry_arms',all(a['fit']==b['fit']==c['fit'] for a,b,c in zip(logs['current_observation'],logs['legacy_state_only'],logs['gated_state_only'])))
        events=[]
        for arm,records in logs.items():
            info=s['full_video'][arm][vid];pids.append(info['pid']);timings=read(OUT/f'full_video/{arm}/{vid}/timings.json');matches=0
            for i,r in enumerate(records):
                pred=np.array(Image.open(r['mask_reference']))
                assert raw(pred)==r['mask_raw_sha256']==base[i]['mask_raw_sha256']
                assert r['frame_index']==i and r['source_rgb_raw_sha256']==source['frames'][i]['rgb_raw_sha256'] and r['timestamp_seconds']==source['frames'][i]['timestamp_seconds']
                st=r['tracked'];assert st['apply_to_mask'] is False and st['tip_metrics'] is None
                if st['state'] is not None:
                    assert valid_geometry(st['state']);z=st['state'];expected=np.array(z[:2])+z[3]*np.array([np.cos(z[2]),np.sin(z[2])])
                    np.testing.assert_allclose(st['geometry']['tip_xy'],expected,atol=1e-10,rtol=0)
                if st['prediction_only']:assert 1<=st['age']<=3 and not st['uses_current_observation']
                if (vid,i) in selected:assert r['mask_raw_sha256']==static['records'][selected[(vid,i)]]['mask_raw_sha256'];matches+=1
                measured=r['fit']['state'] if r['fit'] and r['fit']['accepted'] and valid_geometry(r['fit']['state']) else None
                if arm=='legacy_state_only':
                    old=legacy_tracker.step(measured,pred.shape,st['dt_seconds']);assert st['state']==(old['state'] if valid_geometry(old['state']) else None)
                if arm=='gated_state_only':
                    replay=gated_control.step(measured,pred.shape,r['timestamp_seconds'],vid);assert replay==st
                    assert 0<=st['pending_count']<3
                    if st['status']=='confirmed_relocation':
                        assert i>=2 and records[i-1]['tracked']['pending_count']==2 and st['uses_current_observation']
                        np.testing.assert_allclose(st['state'],measured,rtol=0,atol=1e-12)
                    if st['status']=='rejected_jump_pending':assert not st['uses_current_observation'] and st['pending_count'] in [1,2]
                    if st['status'] in ['confirmed_relocation','rejected_jump_pending','expired_reset_with_pending','expired_reset','invalid_prediction_reset']:
                        events.append({'frame_index':i,'display_frame':i+1,'status':st['status'],'age':st['age'],'pending_count':st['pending_count'],'current_fit':measured,'state':st['state']})
            label=arm+'_'+vid
            check(label+'_complete_native_hashes_states',len(records)==len(timings)==n and matches==info['approved_static_matches'])
            check(label+'_actual_model_identity',info['networks']==2 and info['parameters']==49202790 and info['individual_model_sha256']==model['individual_model_sha256'] and info['model_state_sha256']==model['model_state_sha256'])
            check(label+'_timing_percentiles',all(abs(float(np.percentile([r[k] for r in timings],q))-v[p])<1e-9 for k,v in info['timings'].items() for q,p in [(50,'p50'),(95,'p95')]))
            check(label+'_pipeline_stage_sum',all(abs(t['pipeline_total_ms']-sum(t[k] for k in ['coarse_pipeline_ms','fit_fov_ms','state_update_ms']))<1e-8 for t in timings))
            check(label+'_replay_video',sha256(OUT/f'full_video/{arm}/{vid}/state_replay.mp4')==info['video_sha256'] and count_video(OUT/f'full_video/{arm}/{vid}/state_replay.mp4',(434,800))==n)
            total+=n
        # A prefix replay cannot read or revise any later observations.
        prefix=GeometryStateControl('gated_state_only',cfg,legacy,1/s['full_video']['raw2'][vid]['source_fps'])
        for r in logs['gated_state_only'][:n//2]:
            measured=r['fit']['state'] if r['fit'] and r['fit']['accepted'] else None
            assert prefix.step(measured,(400,400),r['timestamp_seconds'],vid)==r['tracked']
        check(vid+'_prefix_causality',True)
        common=[all(logs[a][i]['tracked']['state'] is not None for a in cfg['arms'][1:]) for i in range(n)]
        all_pair_motion={arm:motion(logs[arm]) for arm in cfg['arms'][1:]};common_motion={arm:motion(logs[arm],common) for arm in cfg['arms'][1:]}
        residuals={}
        for arm in cfg['arms'][1:]:
            values=[]
            for r in logs[arm]:
                if r['fit'] and r['fit']['accepted'] and r['tracked']['geometry']:
                    values.append(float(np.linalg.norm(np.array(r['fit']['tip_xy'])-r['tracked']['geometry']['tip_xy'])))
            residuals[arm]=distribution(values)
        analysis['videos'][vid]={'gated_events':events,'common_available_frames':sum(common),'motion_all_available_pairs':all_pair_motion,'motion_common_available_pairs':common_motion,'state_to_current_fit_endpoint_residual_px_NOT_gt_error':residuals}
        c=s['comparison_videos'][vid];check(vid+'_comparison_complete',sha256(c['path'])==c['sha256'] and count_video(c['path'],(434,2000))==n)
    check('eight_distinct_processes_10776frames',len(set(pids))==8 and total==10776)
    check('no_new_accuracy_training_adoption_or_t1',s['all_masks_unchanged'] and s['state_only'] and not s['default_changed'] and not s['adopted'] and not s['t1_search_started'] and not s['s3_started'] and s['training_updates']==s['train_pixels_read']==s['reserved_test_pixels_read']==0 and s['tip_metrics'] is None and s['whole_video_accuracy'] is None)
    atomic_json(OUT/'geometry_diagnostics.json',analysis)
    atomic_json(OUT/'engineering_checks.json',{'passed':sum(checks.values()),'checks':checks,'auditor_sha256':sha256(Path(__file__)),'native_mask_records_rechecked':total,'static_masks_rechecked':202,'extra_model_forwards':0,'geometry_accuracy_claim':False})
    print(json.dumps({'passed':sum(checks.values()),'frames':total,'events':{v:len(d['gated_events']) for v,d in analysis['videos'].items()}}))

if __name__=='__main__':main()
