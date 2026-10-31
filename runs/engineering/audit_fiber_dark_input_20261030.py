"""Independent result/identity audit. Never constructs a neural network."""
from pathlib import Path
import json
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.fiber_reference_workspace import read,sha256,load_mask
from tools.run_two_network_experts import array_hash
from tools.fiber_photometric_input import ARMS
from tools.run_fiber_dark_input_control import quality,json_lines
from ureter_semis.experiment import atomic_json

OUT=ROOT/'runs/a1_fiber_dark_input_control_20261030'


def main():
    checks={}
    def check(name,value):
        checks[name]=bool(value)
        if not value:raise AssertionError(name)
    pre=read(OUT/'preflight.json');cfg=pre['config'];summary=read(OUT/'summary.json')
    check('source_hashes',all(sha256(ROOT/k)==v for k,v in pre['sources'].items()))
    check('reference_hashes',all(sha256(k)==v for k,v in pre['inputs']['files'].items()))
    check('related_hashes',all(sha256(k)==v for k,v in pre['related'].items()))
    check('checkpoint_and_video_hashes',all(sha256(x['path'])==x['sha256'] for x in [*pre['checkpoints'].values(),*pre['videos']]))
    with zipfile.ZipFile(OUT/'source_snapshot.zip') as z:
        check('archive_roundtrip',all(z.read(k)==(ROOT/k).read_bytes() for k in z.namelist()))
    protocol=read(ROOT/'runs/protocols/fiber_dark_input_control_20261030_v1/identity.json')
    check('protocol_remains_frozen',all(sha256(ROOT/k)==v for k,v in protocol['files'].items()))
    check('same_606_static_8082_full',summary['actual_static_paired_frame_inferences']==606 and summary['actual_full_paired_frame_inferences']==8082)
    check('no_train_test_pixels_or_training',all(summary[k]==0 for k in ['training_updates','train_pixels_read','reserved_test_pixels_read']))
    check('no_adoption_or_S3',not any(summary[k] for k in ['adopted','default_changed','s3_started']))
    check('no_unsupported_accuracy',summary['tip_metrics'] is None and summary['whole_video_accuracy'] is None)
    pids=[];total=0
    for scope,key in [('video_native','video_manifest'),('image_native','image_manifest')]:
        rows=read(ROOT/cfg['data'][key])['val'];base=read(ROOT/cfg['base']['reference_run']/f'{scope}/member1_2net.json')
        raw=read(OUT/f'{scope}/raw2.json')
        check(scope+'_raw_exact_summary',raw['summary']==base['summary'])
        for arm in ARMS:
            result=read(OUT/f'{scope}/{arm}.json');matrix=np.zeros((3,3),np.int64);changes={c:{k:0 for k in ['lost_tp','gained_tp','added_fp','removed_fp']} for c in ['stone','fiber']}
            check(scope+'_'+arm+'_count',len(result['records'])==len(rows)==result['actual_paired_frame_inferences'])
            brightness_totals=np.zeros((2,3,4),np.int64)
            import cv2
            from tools.run_fiber_blue_aplus import frame_read
            for i,(row,record) in enumerate(zip(rows,result['records'])):
                target=load_mask(row['mask'],tuple(result['summary']['frames'][i]['native_shape']),{0,1,2,255});shape=target.shape
                mask=load_mask(OUT/f'{scope}/{arm}/{i:04d}.png',shape,{0,1,2});original=load_mask(OUT/f'{scope}/raw2/{i:04d}.png',shape,{0,1,2})
                assert array_hash(mask)==record['mask_raw_sha256']
                valid=target!=255;cm=np.bincount(target[valid].astype(np.int64)*3+mask[valid],minlength=9).reshape(3,3);matrix+=cm
                assert cm.tolist()==result['summary']['frames'][i]['confusion_matrix']
                for c,name in [(1,'stone'),(2,'fiber')]:
                    truth=target==c;a=original==c;b=mask==c
                    counts={'lost_tp':int((truth&a&~b).sum()),'gained_tp':int((truth&~a&b).sum()),
                        'added_fp':int((valid&~truth&~a&b).sum()),'removed_fp':int((valid&~truth&a&~b).sum())}
                    assert counts==record['paired_'+name]
                    for k,v in counts.items():changes[name][k]+=v
                with np.load(OUT/f'{scope}/{arm}/probabilities/{i:04d}.npz') as z:
                    sp=z['stone'];fp=z['fiber']
                    assert array_hash(sp)==record['stone_probability_raw_sha256']==base['records'][i]['stone_probability_raw_sha256']
                    assert array_hash(fp)==record['fiber_probability_raw_sha256']
                    assert fp.dtype==np.float32 and np.isfinite(fp).all() and np.allclose(fp.sum(0),1,atol=2e-5)
                    if arm=='raw2':assert array_hash(fp)==base['records'][i]['fiber_probability_raw_sha256']
                frame=frame_read(row['image']);fields=[cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)[:,:,2],cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)[:,:,0]]
                for j,field in enumerate(fields):
                    for k,(a,b) in enumerate([(0,85),(85,170),(170,256)]):
                        region=(field>=a)&(field<b);truth=(target==2)&region;pred=mask==2
                        brightness_totals[j,k]+=[truth.sum(),(truth&pred).sum(),(valid&(target!=2)&pred&region).sum(),(truth&~pred).sum()]
            check(f'{scope}_{arm}_independent_confusion',matrix.tolist()==result['summary']['metrics']['confusion_matrix'])
            check(f'{scope}_{arm}_independent_paired_changes',changes==result['paired_changes'])
            check(f'{scope}_{arm}_independent_brightness',all(brightness_totals[j,k].tolist()==[result['brightness'][feature][k][n] for n in ['gt_pixels','tp','fp','fn']] for j,feature in enumerate(['HSV_V','Lab_Lstar']) for k in range(3)))
            check(f'{scope}_{arm}_group_counts',sum(v['frames'] for k,v in result['groups'].items() if k in ['video_1','video_2'] or k.startswith('kidney_'))==len(rows))
    for arm in ARMS:
        static=read(OUT/f'static_identity_{arm}.json')
        check(arm+'_static_graph',static['actual_branch_forwards']==[202,202] and static['networks']==2 and static['parameters']==49202790)
        for info in pre['videos']:
            vid=info['video_id'];folder=OUT/f'full_video/{arm}/{vid}';result=read(folder/'summary.json');records=json_lines(folder/'frames.jsonl');raw=json_lines(OUT/f'full_video/raw2/{vid}/frames.jsonl')
            source=read(OUT/f'source_{vid}.json')['frames'];n=info['frames'];pids.append(result['pid']);total+=n
            check(arm+'_'+vid+'_complete',len(records)==n==result['frames']==result['replay']['frames'])
            check(arm+'_'+vid+'_graph',result['actual_branch_forwards']==[n,n] and result['warmup_branch_forwards']==[12,12] and result['networks']==2 and result['parameters']==49202790)
            check(arm+'_'+vid+'_same_weights',result['model_state_sha256']==static['model_state_sha256'] and result['individual_model_sha256']==static['individual_model_sha256'])
            check(arm+'_'+vid+'_source_and_stone',all(r['frame_index']==i and r['source_rgb_raw_sha256']==source[i]['rgb_raw_sha256'] and r['stone_probability_raw_sha256']==raw[i]['stone_probability_raw_sha256'] for i,r in enumerate(records)))
            check(arm+'_'+vid+'_mask_file_identity',all(array_hash(load_mask(folder/f'masks/{i:06d}.png',(info['height'],info['width']),{0,1,2}))==r['mask_raw_sha256'] for i,r in enumerate(records)))
            check(arm+'_'+vid+'_video_identity',sha256(folder/'replay.mp4')==result['replay']['sha256'])
            timings=read(folder/'timings.json');p95=float(np.percentile([r['pipeline_total_ms'] for r in timings],95))
            check(arm+'_'+vid+'_direct_latency',len(timings)==n and p95==result['timings']['pipeline_total_ms']['p95'])
            check(arm+'_'+vid+'_selected_matches',result['approved_static_matches']==sum(r['approved_semantic_reference'] for r in records))
    check('six_distinct_processes',len(set(pids))==6 and total==8082)
    check('quality_gate_recomputation',quality(cfg,OUT)==summary['static_quality'])
    for vid,row in summary['comparison_videos'].items():check(vid+'_comparison_identity',sha256(row['path'])==row['sha256'])
    check('T2P_still_absent',not (ROOT/'tools/fiber_local_evidence.py').exists())
    identity=read(OUT/'analysis/identity.json')
    check('analysis_artifact_identity',all(sha256(OUT/k)==v for k,v in identity['artifacts'].items()))
    atomic_json(OUT/'engineering_checks.json',{'passed':len(checks),'checks':checks,'auditor_sha256':sha256(Path(__file__)),
        'no_model_forwards':True,'scope':'Independent counts, fixed gates, coverage and identities; not clinical validation or anatomical tip truth.'})
    print(json.dumps({'checks_passed':len(checks),'complete_frames':total,'distinct_workers':len(set(pids))}),flush=True)


if __name__=='__main__':main()
