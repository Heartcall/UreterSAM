"""Independent read-only recount and identity audit; no model construction."""
import hashlib,json,math,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image
from tools.fiber_reference_workspace import read,sha256
from tools.run_two_network_experts import cache_for
from ureter_semis.experiment import atomic_json
OUT=ROOT/'runs/a1_fiber_t1_failure_diagnosis_20261030'
checks={}


def check(name,ok):
    checks[name]=bool(ok)
    if not ok:raise AssertionError(name)


def raw(x):return hashlib.sha256(x.tobytes()).hexdigest()


def main():
    identity=read(OUT/'identity.json');summary=read(OUT/'summary.json');forensic=read(OUT/'forensic_summary.json')
    task=read(ROOT/'configs/fiber_t1_failure_diagnosis_20261030.json');old=ROOT/task['reference_run'];pre=read(old/'preflight.json');cfg=pre['config']
    planpath=ROOT/'configs/fiber_local_evidence_control_20261030.json';plan=read(planpath)
    check('frozen_T1_sources_unchanged',all(sha256(ROOT/k)==v for k,v in identity['frozen_t1_sources'].items()))
    check('frozen_input_files_unchanged',all(sha256(k)==v for k,v in identity['input_files'].items()))
    check('all202reference_bytes_unchanged',all(sha256(k)==v for k,v in identity['validation_input_files'].items()))
    check('weights_unchanged',all(sha256(v['path'])==v['sha256'] for v in identity['weights'].values()))
    check('diagnostic_initial_outputs_unchanged',all(sha256(OUT/p)==v for p,v in identity['new_outputs'].items()))
    check('diagnostic_config_matches',sha256(ROOT/'configs/fiber_t1_failure_diagnosis_20261030.json')==identity['config_sha256'])
    original=read(ROOT/'datasets/ureter_semis_kidney_joint_20261028/video_validation_manifest.json')['val'];indices={(r['video_id'],r['frame_index']):i for i,r in enumerate(original)}
    manifest=read(ROOT/cfg['data']['image_manifest']);train={r['image'] for r in manifest['train']['labeled']};test={r['image'] for r in manifest['reserved_test'] if 'image'in r}
    static_records=0
    for scope,key,n in [('video_native','video_manifest',70),('image_native','image_manifest',132)]:
        data=read(OUT/f'{scope}.json');rows=read(ROOT/cfg['data'][key])['val'];olddata=read(old/f'{scope}.json');records=data['records'];total={'tp':0,'fp':0,'fn':0,'fn_below01':0,'fn_01_to09':0,'fp_ge09':0,'fp_below09':0};opportunity={'eligible_pixels':0,'eligible_reference_tp_at_risk':0,'eligible_reference_fp':0,'eligible_reference_fn':0}
        check(scope+'_counts',len(records)==len(rows)==n)
        check(scope+'_allowlist_excludes521and144',all(r['image'] not in train|test for r in rows))
        for i,(row,r) in enumerate(zip(rows,records)):
            gt=np.array(Image.open(row['mask']));pred=np.array(Image.open(old/f'{scope}/shared_raw2/{i:04d}.png'))
            path,_=cache_for(row,i,scope=='video_native',indices)
            with np.load(path) as z:p=z['a0_member1_native']
            assert raw(p)==olddata['records'][i]['fiber_probability_raw_sha256']
            truth=gt==2;valid=gt!=255;tp=truth&(pred==2);fp=(pred==2)&valid&~truth;fn=truth&(pred!=2)
            counts=dict(tp=int(tp.sum()),fp=int(fp.sum()),fn=int(fn.sum()),fn_below01=int((fn&(p[2]<.1)).sum()),fn_01_to09=int((fn&(p[2]>=.1)&(p[2]<.9)).sum()),fp_ge09=int((fp&(p[2]>=.9)).sum()),fp_below09=int((fp&(p[2]<.9)).sum()))
            assert all(r['neural'][k]==v for k,v in counts.items())
            for k,v in counts.items():total[k]+=v
            assert not(truth&(p.argmax(0)==2)&(pred!=2)).any()
            scale=min(gt.shape)/400;radius=max(1,int(math.floor(2*scale+.5)));kernel=np.ones((2*radius+1,2*radius+1),np.uint8);fiber=(pred==2).astype(np.uint8)
            band=cv2.dilate(fiber,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)!=cv2.erode(fiber,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
            eligible=band&(pred!=1)&(p[2]>=.1)&(p[2]<.9)
            assert not(eligible&(pred==1)).any() and not(eligible&(p[2]>=.9)).any() and not(eligible&(p[2]<.1)).any()
            entry=forensic[scope]['records'][i];assert int(eligible.sum())==entry['eligible_pixels']
            if eligible.sum()<=math.floor(.05*gt.size):
                for k,v in [('eligible_pixels',eligible.sum()),('eligible_reference_tp_at_risk',(eligible&tp).sum()),('eligible_reference_fp',(eligible&fp).sum()),('eligible_reference_fn',(eligible&fn).sum())]:opportunity[k]+=int(v)
            for c in r['candidate_diagnostics']:
                assert len(c['station_trace'])==8
                if c['reason']=='width_profile' and c['forensic_width_ratio']>0:
                    assert abs(c['forensic_reversed_ratio']*c['forensic_width_ratio']-1)<1e-12
                for st in c['station_trace']:
                    assert st['support_before_cap']>=st['support_after_cap']
            for name,aux in r['auxiliary'].items():
                region=np.array(Image.open(row['auxiliary_masks'][name]))>0
                assert aux['raw_fiber_overlap']==int((region&(pred==2)).sum()) and aux['provided_annotation_unreviewed']
        check(scope+'_neural_recount_exact',all(total[k]==data['neural_counts'][k] for k in total))
        check(scope+'_TPFPFN_match_frozen_baseline',all(total[k]==olddata['summary']['metrics']['fiber_'+k] for k in ['tp','fp','fn']))
        check(scope+'_prospective_scope_count_only_reproduced',opportunity==forensic[scope]['after_presampling_budget_totals'])
        check(scope+'_plan_frozen_before_scope_measurement',forensic[scope]['eligible_scope_plan_sha256_before_calculation']==sha256(planpath))
        check(scope+'_unoriented_quartile_denominator',sum(x['pixels'] for x in forensic[scope]['unoriented_reference_PCA_quartiles_NOT_tip_segments'])==data['neural_counts']['gt_fiber_pixels'])
        check(scope+'_SVbin_denominator',sum(x['pixels'] for x in forensic[scope]['reference_SV_bins_S_thenV_0to84_85to169_170to255_NOT_tip_metrics'])==data['neural_counts']['gt_fiber_pixels'])
        check(scope+'_positive_stage_denominator',sum(data['positive_stage_counts'].values())==data['reference_positive_frames'])
        static_records+=n
    snaps=read(OUT/'video_snapshots.json');model=read(old/'static_model_identity.json')
    check('five_original_model_snapshots_only',snaps['actual_original_model_forward_calls']==len(snaps['snapshots'])==5 and [x['display_frame'] for x in snaps['snapshots']]==[311,439,622,643,1266])
    check('same_original_model_state',snaps['model_state_sha256']==model['model_state_sha256'] and snaps['individual_model_sha256']==model['individual_model_sha256'])
    actual_indices={(r['video_id'],r['frame_index']) for r in read(ROOT/cfg['data']['video_manifest'])['val']}
    for r in snaps['snapshots']:
        with np.load(r['probability_file']) as z:p=z['fiber'];stone=z['stone'];assert raw(p)==r['fiber_probability_raw_sha256']
        assert p.dtype==np.float32 and stone.dtype==np.float32 and np.allclose(p.sum(0),1,atol=2e-5)
        mask=np.zeros(p.shape[1:],np.uint8);mask[stone.argmax(0)==1]=1;mask[p.argmax(0)==2]=2;assert raw(mask)==r['mask_raw_sha256']
        assert(r['video_id'],r['frame_index'])not in actual_indices and r['pixel_truth_available'] is False and r['tip_metrics'] is None
    check('five_probability_snapshots_reconstruct_original_masks_no_truth',True)
    check('both_full_video_metadata_all_four_arms',all(v[a]['frames']==n for v,n in [(summary['full_video_metadata']['video_1'],722),(summary['full_video_metadata']['video_2'],1972)] for a in cfg['arms']))
    check('no_new_masks_or_training_default_changes',summary['new_candidate_masks']==summary['training_updates']==summary['train_pixels_read']==summary['reserved_test_pixels_read']==0 and not summary['default_changed'] and not summary['threshold_changes'])
    check('T2P_not_implemented',not(ROOT/plan['planned_output']).exists() and all(not(ROOT/p).exists() for p in plan['planned_files']))
    check('finite_three_prospective_arms',plan['arms']==['raw2','spatial5','rgb_edge5'] and len(plan['filter']['neighbor_offsets'])==5)
    check('all_six_processes_full_videos_even_failure',plan['execution']['expected_processes']==6 and plan['execution']['expected_full_forward_frames']==3*(722+1972)==8082 and plan['execution']['all_arms_complete_even_static_or_latency_failure'])
    original_control=read(ROOT/'runs/a1_two_network_experts_20261029/preflight.json')
    check('legacy_gate_reference_unchanged',sha256(ROOT/plan['base']['frozen_protocol'])==original_control['config_sha256'] and read(ROOT/plan['base']['frozen_protocol'])==original_control['config'])
    check('separate_new_geometry_and_tip_accuracy_not_claimed',summary['tip_metrics'] is None and summary['whole_video_accuracy'] is None)
    atomic_json(OUT/'engineering_checks.json',{'passed':sum(checks.values()),'checks':checks,'scope':'read-only attribution and protocol consistency, NOT pixel-method unit tests or accuracy validation',
                                               'auditor_sha256':sha256(Path(__file__)),'static_reference_records_recounted':static_records,'additional_audit_model_forwards':0})
    print(json.dumps({'checks':sum(checks.values()),'static_records':static_records,'T2P_executed':False}))


if __name__=='__main__':main()
