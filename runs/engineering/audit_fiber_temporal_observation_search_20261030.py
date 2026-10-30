"""Independent post-run T1 identity, replay, causality and evidence audit. No NN."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image
from tools.fiber_reference_workspace import read,sha256
from tools.fiber_evidence_tracker import EvidenceTracker,valid_group,stateless_result
from tools.fiber_fast_observation import FastFiberObserver,public_observation
from tools.run_classwise_expert_control import summarize
from tools.run_two_network_experts import cache_for
from tools.run_fiber_blue_aplus import frame_read
from ureter_semis.experiment import atomic_json
OUT=ROOT/'runs/a1_fiber_temporal_observation_search_20261030'
checks={}


def check(name,value):
    checks[name]=bool(value)
    if not value:raise AssertionError(name)


def same(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    elif isinstance(a,float):np.testing.assert_allclose(a,b,atol=1e-10,rtol=0)
    else:assert a==b,(a,b)


def raw(a):return hashlib.sha256(a.tobytes()).hexdigest()
def records(arm,vid):return [json.loads(s) for s in (OUT/f'full_video/{arm}/{vid}/frames.jsonl').read_text().splitlines()]


def count_video(path,shape):
    cap=cv2.VideoCapture(str(path));n=0
    while True:
        ok,f=cap.read()
        if not ok:break
        assert f.shape[:2]==shape;n+=1
    cap.release();return n


def distribution(values):
    return {'n':len(values),'p50':float(np.percentile(values,50)),'p95':float(np.percentile(values,95)),'max':float(max(values))} if values else {'n':0,'p50':None,'p95':None,'max':None}


def thin(mask):
    """Zhang-Suen thinning of visible semantic foreground; not anatomical axis GT."""
    a=np.pad(mask.astype(np.uint8),1)
    while True:
        changed=0
        for phase in (0,1):
            p=[a[:-2,1:-1],a[:-2,2:],a[1:-1,2:],a[2:,2:],a[2:,1:-1],a[2:,:-2],a[1:-1,:-2],a[:-2,:-2]]
            neighbors=sum(p)
            transitions=sum(((p[k]==0)&(p[(k+1)%8]==1)).astype(np.uint8) for k in range(8))
            if phase==0:allowed=(p[0]*p[2]*p[4]==0)&(p[2]*p[4]*p[6]==0)
            else:allowed=(p[0]*p[2]*p[6]==0)&(p[0]*p[4]*p[6]==0)
            remove=(a[1:-1,1:-1]==1)&(neighbors>=2)&(neighbors<=6)&(transitions==1)&allowed
            changed+=int(remove.sum());a[1:-1,1:-1][remove]=0
        if not changed:return a[1:-1,1:-1].astype(bool)


def motion(rows,selected=None):
    speeds=[];angles=[];lengths=[]
    for i,(a,b) in enumerate(zip(rows,rows[1:]),1):
        if selected is not None and not(selected[i-1] and selected[i]):continue
        sa,sb=a['tracked']['groups']['shaft']['value'],b['tracked']['groups']['shaft']['value']
        ea,eb=a['tracked']['groups']['endpoint']['value'],b['tracked']['groups']['endpoint']['value']
        dt=b['timestamp_seconds']-a['timestamp_seconds']
        if sa is not None and sb is not None:
            delta=(sb[2]-sa[2]+np.pi)%(2*np.pi)-np.pi;angles.append(abs(float(np.rad2deg(delta)))/dt)
        pa,pb=a['tracked']['visible_front_candidate'],b['tracked']['visible_front_candidate']
        if pa is not None and pb is not None:speeds.append(float(np.linalg.norm(np.array(pb)-pa))/dt)
        if ea is not None and eb is not None:lengths.append(abs(eb[0]-ea[0])/dt)
    return {'front_speed_px_per_s':distribution(speeds),'axis_speed_deg_per_s':distribution(angles),'length_speed_px_per_s':distribution(lengths)}


def main():
    pre=read(OUT/'preflight.json');cfg=pre['config'];s=read(OUT/'summary.json');observer=FastFiberObserver(cfg)
    previous=ROOT/cfg['base']['reference_run'];pids=[];total=0;analysis={'role':'descriptive geometry diagnostics, not accuracy','videos':{}}
    check('job_completed',read(OUT/'job.json')['phase']=='completed')
    check('source_frozen',all(sha256(ROOT/p)==h for p,h in pre['sources'].items()))
    check('related_frozen',all(sha256(p)==h for p,h in pre['related'].items()))
    check('references_frozen',all(sha256(p)==h for p,h in pre['inputs']['files'].items()))
    check('videos_weights_frozen',all(sha256(x['path'])==x['sha256'] for x in [*pre['videos'],*pre['checkpoints'].values()]))
    with zipfile.ZipFile(OUT/'source_snapshot.zip') as z:
        check('source_archive_exact',all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in pre['sources'].items()))
        check('config_archive_exact',hashlib.sha256(z.read('configs/fiber_temporal_observation_search_20261030.json')).hexdigest()==pre['config_sha256'])
    old29=read(ROOT/'datasets/ureter_semis_kidney_joint_20261028/video_validation_manifest.json')['val']
    old_indices={(r['video_id'],r['frame_index']):i for i,r in enumerate(old29)}
    centerline={};semantic={};static_probabilities={}
    for scope,key,n in [('video_native','video_manifest',70),('image_native','image_manifest',132)]:
        d=read(OUT/(scope+'.json'));rows=read(ROOT/cfg['data'][key])['val'];targets=[];preds=[];cl=[]
        for i,(row,r) in enumerate(zip(rows,d['records'])):
            pred=np.array(Image.open(OUT/f'{scope}/shared_raw2/{i:04d}.png'));target=np.array(Image.open(row['mask']))
            assert raw(pred)==r['mask_raw_sha256'];targets.append(target);preds.append(pred)
            path,_=cache_for(row,i,scope=='video_native',old_indices)
            with np.load(path) as z:fp=z['a0_member1_native'].copy()
            assert raw(fp)==r['fiber_probability_raw_sha256']
            f=frame_read(row['image']);obs=observer.current(f,fp);obs=public_observation({k:v for k,v in obs.items() if k!='timings'})
            for arm in cfg['arms']:
                o=r['observations'][arm];assert o['history'] is None and not o['tracked']['apply_to_mask']
                if arm!='raw2':same(o['current'],obs)
                if arm in ('evidence_state','history_search'):
                    t=EvidenceTracker(cfg);same(t.update(obs,None,t.prepare(pred.shape,0,row['sample_id'])),o['tracked'])
                else:same(stateless_result(o['current']),o['tracked'])
            valid=target!=255;truth=(target==2)&valid;prediction=(pred==2)&valid
            ts,ps=thin(truth),thin(prediction)
            cl.append({'sample_id':row['sample_id'],'gt_skeleton_pixels':int(ts.sum()),'pred_skeleton_pixels':int(ps.sum()),
                       'gt_skeleton_in_pred':int((ts&prediction).sum()),'pred_skeleton_in_gt':int((ps&truth).sum())})
            if scope=='video_native':static_probabilities[(row['video_id'],row['frame_index'])]=(f,fp)
        check(scope+'_202static_counts',len(d['records'])==n)
        check(scope+'_fresh_metrics_exact',summarize(rows,targets,preds)==d['summary']==read(previous/f'{scope}/member1_2net.json')['summary'])
        check(scope+'_groups_exact',d['groups']==read(previous/f'{scope}/member1_2net.json')['groups'])
        check(scope+'_candidate_and_fresh_state_reproduction',True)
        check(scope+'_all_pixel_changes_zero',all(v==0 for x in d['paired_changes_all_arms'].values() for v in x.values()))
        denom=sum(x['gt_skeleton_pixels'] for x in cl);pd=sum(x['pred_skeleton_pixels'] for x in cl)
        recall=sum(x['gt_skeleton_in_pred'] for x in cl)/denom if denom else None
        precision=sum(x['pred_skeleton_in_gt'] for x in cl)/pd if pd else None
        centerline[scope]={'method':'Zhang-Suen visible semantic skeleton, exact-pixel coverage; descriptive post-run metric, not anatomical axis or new adoption threshold',
                          'gt_centerline_recall':recall,'pred_centerline_precision':precision,'clDice':2*precision*recall/(precision+recall) if precision and recall else 0.,
                          'all_arms_same':True,'records':cl}
        semantic[scope]=d
    model=read(OUT/'static_model_identity.json')
    for vid,n in [('video_1',722),('video_2',1972)]:
        logs={a:records(a,vid) for a in cfg['arms']};base=logs['raw2'];source=read(OUT/f'source_{vid}.json')
        check(vid+'_same_current_evidence_all_geometry_arms',all(a['current']==b['current']==c['current'] for a,b,c in zip(logs['light_current'],logs['evidence_state'],logs['history_search'])))
        # Decode source once again, independently checking sequential hashes and native mask references.
        cap=cv2.VideoCapture(next(v['path'] for v in pre['videos'] if v['video_id']==vid));decoded=0
        while True:
            ok,f=cap.read()
            if not ok:break
            assert raw(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))==base[decoded]['source_rgb_raw_sha256']
            pred=np.array(Image.open(base[decoded]['mask_reference']));assert raw(pred)==base[decoded]['mask_raw_sha256'];decoded+=1
        cap.release();check(vid+'_source_native_masks_independently_redecoded',decoded==n)
        events={};common=[all(logs[a][i]['tracked']['visible_front_candidate'] is not None for a in cfg['arms'][1:]) for i in range(n)]
        for arm,rr in logs.items():
            info=s['full_video'][arm][vid];timings=read(OUT/f'full_video/{arm}/{vid}/timings.json');pids.append(info['pid']);t=EvidenceTracker(cfg,1/info['source_fps']);events[arm]=[]
            for i,r in enumerate(rr):
                assert r['frame_index']==i and r['mask_raw_sha256']==base[i]['mask_raw_sha256'] and r['source_rgb_raw_sha256']==source['frames'][i]['rgb_raw_sha256'] and r['timestamp_seconds']==source['frames'][i]['timestamp_seconds']
                cur,history,tracked=r['current'],r['history'],r['tracked'];assert not tracked['apply_to_mask'] and tracked['tip_metrics'] is None
                assert len(cur['candidates'])<=2
                for c in cur['candidates']:
                    assert c['provenance']=='independent_current'
                    if c['level']!='rejected':assert c['edge_pairs']>=5 and c['qN']>=.10 and c['interior_valid_triplets']>=5
                    if c['level']=='strong':assert c['qN']>=.5 and c['fiber_argmax_fraction']>=.75
                if arm in ('evidence_state','history_search'):
                    prepared=t.prepare((400,400),r['timestamp_seconds'],vid);same(t.update(cur,history,prepared),tracked)
                else:same(stateless_result(cur),tracked)
                if history is not None:
                    assert arm=='history_search' and cur['strong_label'] is None and tracked['prior']['shaft'] is not None
                    assert history['provenance']=='history_conditioned' and history['level']!='strong'
                    if (vid,i) in static_probabilities:
                        f,fp=static_probabilities[(vid,i)];same(public_observation(observer.history(f,fp,tracked['prior'])),history)
                if cur['strong_label'] is not None and arm in ('evidence_state','history_search'):
                    assert tracked['selected_label']==cur['strong_label'] and tracked['independent_strong_used']
                for g,v in tracked['groups'].items():
                    if v['value'] is not None:assert valid_group(g,v['value'])
                    if v['predicted']:assert 1<=v['age']<=3
                if tracked.get('strong_override_groups') or r['history_search_attempted'] or any(v['status'] in ('expired','invalid_prediction_reset') for v in tracked['groups'].values()):
                    events[arm].append({'display_frame':i+1,'strong_override_groups':tracked.get('strong_override_groups'),
                                        'history_attempted':r['history_search_attempted'],'used_provenance':tracked['used_provenance'],
                                        'statuses':{g:v['status'] for g,v in tracked['groups'].items()}})
            prefix=EvidenceTracker(cfg,1/info['source_fps'])
            if arm in ('evidence_state','history_search'):
                for r in rr[:n//2]:same(prefix.update(r['current'],r['history'],prefix.prepare((400,400),r['timestamp_seconds'],vid)),r['tracked'])
            label=arm+'_'+vid
            check(label+'_complete_invariance_and_state_replay',len(rr)==len(timings)==n)
            check(label+'_prefix_causality',True)
            check(label+'_actual_model',info['networks']==2 and info['parameters']==49202790 and info['model_state_sha256']==model['model_state_sha256'] and info['individual_model_sha256']==model['individual_model_sha256'])
            check(label+'_all_stage_percentiles',all(abs(float(np.percentile([t[k] for t in timings],q))-v[p])<1e-9 for k,v in info['timings'].items() for q,p in [(50,'p50'),(95,'p95')]))
            # Geometry's inner timer excludes public-metadata conversion/return;
            # the directly measured full pipeline includes it. Do not hide this gap.
            check(label+'_total_stage_accounting',all(t['pipeline_total_ms']+1e-8>=t['coarse_pipeline_ms']+t['geometry_total_ms'] for t in timings))
            check(label+'_complete_encoded_video',sha256(OUT/f'full_video/{arm}/{vid}/state_replay.mp4')==info['video_sha256'] and count_video(OUT/f'full_video/{arm}/{vid}/state_replay.mp4',(462,800))==n)
            total+=n
        c=s['comparison_videos'][vid];check(vid+'_comparison_video_complete',sha256(c['path'])==c['sha256'] and count_video(c['path'],(462,2000))==n)
        analysis['videos'][vid]={'events':events,'common_front_available_frames':sum(common),
                               'motion_all_available_pairs':{a:motion(logs[a]) for a in cfg['arms'][1:]},
                               'motion_common_available_pairs':{a:motion(logs[a],common) for a in cfg['arms'][1:]}}
    check('eight_unique_processes_full10776',len(set(pids))==8 and total==10776)
    check('no_training_adoption_truth_claims',s['all_masks_unchanged'] and s['state_only'] and not s['default_changed'] and not s['adopted'] and not s['s3_started'] and s['training_updates']==s['train_pixels_read']==s['reserved_test_pixels_read']==0 and s['tip_metrics'] is None and s['whole_video_accuracy'] is None)
    atomic_json(OUT/'geometry_diagnostics.json',analysis);atomic_json(OUT/'visible_centerline_diagnostics.json',centerline)
    atomic_json(OUT/'engineering_checks.json',{'passed':sum(checks.values()),'checks':checks,'auditor_sha256':sha256(Path(__file__)),
                                             'static_masks_rechecked':202,'full_masks_rechecked':total,'extra_network_forwards':0,'geometry_accuracy_claim':False})
    print(json.dumps({'passed':sum(checks.values()),'full_records':total}))


if __name__=='__main__':main()
