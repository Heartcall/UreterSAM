"""Independent counts, original identities and full-video proxy audit."""
from pathlib import Path
import hashlib,json,sys,zipfile
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image
from tools.fiber_reference_workspace import read,sha256
from tools.fiber_refinement_feasibility import fiber_rois
from ureter_semis.experiment import atomic_json
OUT=ROOT/'runs/a1_fiber_refinement_feasibility_20261030'
checks={}
def check(k,v):
    checks[k]=bool(v)
    if not v:raise AssertionError(k)
def raw(a):return hashlib.sha256(a.tobytes()).hexdigest()
def rectangle_union(shape,boxes):
    y,x=np.indices(shape);union=np.zeros(shape,bool)
    for x0,y0,x1,y1 in boxes:
        assert 0<=x0<x1<=shape[1] and 0<=y0<y1<=shape[0]
        union|=(x>=x0)&(x<x1)&(y>=y0)&(y<y1)
    return union
def video_count(path,shape):
    cap=cv2.VideoCapture(str(path));n=0
    while True:
        ok,a=cap.read()
        if not ok:break
        assert a.shape[:2]==shape;n+=1
    cap.release();return n
def main():
    pre=read(OUT/'preflight.json');cfg=pre['config'];result=read(OUT/'summary.json');previous=ROOT/cfg['reference_run']
    check('completed',read(OUT/'job.json')['phase']=='completed')
    check('sources_frozen',all(sha256(ROOT/k)==v for k,v in pre['sources'].items()))
    check('related_frozen',all(sha256(k)==v for k,v in pre['related'].items()))
    check('labels_rgb_frozen',all(sha256(k)==v for k,v in pre['inputs']['files'].items()))
    check('checkpoints_videos_frozen',all(sha256(x['path'])==x['sha256'] for x in [*pre['checkpoints'].values(),*pre['videos']]))
    with zipfile.ZipFile(OUT/'source_snapshot.zip') as z:check('source_archive',all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in pre['sources'].items()))
    extra={'role':'Post-hoc diagnostic only, no new gate or inference change; distances to reference fiber for FP are reference-conditioned, not inference features','sets':{}}
    static={}
    for scope,key,n in [('video_native','video_manifest',70),('image_native','image_manifest',132)]:
        rows=read(ROOT/cfg[key])['val'];d=read(OUT/(scope+'.json'));m1=read(previous/f'{scope}/member1_2net.json');static[scope]=(rows,d)
        check(scope+'_reference_count',len(rows)==n)
        hist=[];matrices={p:np.zeros((3,3),np.int64) for p in cfg['profiles']};sums={p:{k:[0,0,0] for k in d['profiles'][p]['aggregate']['regions']} for p in cfg['profiles']}
        pooled={'reference_fp_pixels':0,'fp_in_no_reference_fiber_images':0,'fp_in_components_without_fiber_overlap':0,'fp_distance_gt_le3':0,'fp_distance_gt_3_to16':0,'fp_distance_gt_gt16_or_no_gt':0,'fp_probability_ge09':0,'no_reference_fiber_images':0,'predicted_fiber_in_no_reference_fiber_images':0,'reference_present_but_no_tp_frames':0}
        for i,row in enumerate(rows):
            target=np.array(Image.open(row['mask']));coarse=np.array(Image.open(previous/f'{scope}/member1_2net/{i:04d}.png'));reference=np.array(Image.open(previous/f'{scope}/classwise3/{i:04d}.png'))
            with np.load(m1['records'][i]['probability_reference']) as a:p=a['a0_member1_native'][2]
            valid=target!=255;truth=target==2;fp=valid&~truth&(coarse==2)
            distance=cv2.distanceTransform((~truth).astype(np.uint8),cv2.DIST_L2,5) if truth.any() else np.full(target.shape,np.inf)
            nc,labels,_,_=cv2.connectedComponentsWithStats((coarse==2).astype(np.uint8),8)
            supported=set(np.unique(labels[truth&(coarse==2)]));supported.discard(0)
            unsupported=(coarse==2)&~np.isin(labels,list(supported))
            row_extra={'sample_id':row['sample_id'],'video_id':row.get('video_id'),'frame_index':row.get('frame_index'),'reference_fiber_pixels':int(truth.sum()),
                'fp_pixels':int(fp.sum()),'fp_in_no_overlap_components':int((fp&unsupported).sum()),'fp_probability_ge09':int((fp&(p>=.9)).sum()),
                'fp_distance_gt_le3':int((fp&(distance<=3)).sum()),'fp_distance_gt_3_to16':int((fp&(distance>3)&(distance<=16)).sum()),'fp_distance_gt_gt16_or_no_gt':int((fp&(distance>16)).sum())}
            pooled['reference_fp_pixels']+=row_extra['fp_pixels'];pooled['fp_probability_ge09']+=row_extra['fp_probability_ge09']
            pooled['fp_in_components_without_fiber_overlap']+=row_extra['fp_in_no_overlap_components']
            for k in ['fp_distance_gt_le3','fp_distance_gt_3_to16','fp_distance_gt_gt16_or_no_gt']:pooled[k]+=row_extra[k]
            if not truth.any():
                pooled['no_reference_fiber_images']+=1;pooled['fp_in_no_reference_fiber_images']+=int(fp.sum());pooled['predicted_fiber_in_no_reference_fiber_images']+=int((coarse==2).any())
            elif not (truth&(coarse==2)).any():pooled['reference_present_but_no_tp_frames']+=1
            for profile in cfg['profiles']:
                record=d['profiles'][profile]['frames'][i];boxes=record['boxes'];union=rectangle_union(target.shape,boxes)
                assert [list(b) for b in fiber_rois(coarse,p,cfg['proposal'],profile,cap=2)]==boxes
                assert len(boxes)<=2
                regions={'target':truth,'tp':truth&(coarse==2),'fn':truth&(coarse!=2),'fp':fp,'lost_tp_vs_classwise':truth&(reference==2)&(coarse!=2),'added_fp_vs_classwise':fp&(reference!=2),'gt_fiber_blocked_by_stone':truth&(coarse==1)}
                for name,area in regions.items():
                    nums=[int(area.sum()),int((area&union).sum()),int((area&~union).sum())]
                    assert nums==[record['regions'][name][k] for k in ['total','inside','outside']]
                    sums[profile][name]=[x+y for x,y in zip(sums[profile][name],nums)]
                expected=coarse.copy();modifiable=union&valid&(coarse!=1)
                expected[modifiable]=np.where(target[modifiable]==2,2,0)
                assert np.array_equal(expected==1,coarse==1)
                matrices[profile]+=np.bincount((target[valid].astype(int)*3+expected[valid]).ravel(),minlength=9).reshape(3,3)
                row_extra[profile]={'outside_fn':record['regions']['fn']['outside'],'roi_count':len(boxes),'inaccessible_gt_fiber_on_stone':int((truth&(coarse==1)).sum())}
            hist.append(row_extra)
        for profile in cfg['profiles']:
            a=d['profiles'][profile]['aggregate']['regions']
            check(scope+'_'+profile+'_independent_all_region_counts',all(sums[profile][name]==[v[k] for k in ['total','inside','outside']] for name,v in a.items()))
            check(scope+'_'+profile+'_protected_oracle_confusion',matrices[profile].tolist()==d['profiles'][profile]['diagnostic_oracle_not_model']['metrics']['confusion_matrix'])
            check(scope+'_'+profile+'_stone_metrics_unchanged',all(d['profiles'][profile]['diagnostic_oracle_not_model']['metrics'][k]==d['coarse_actual_model']['metrics'][k] for k in ['stone_tp','stone_fp','stone_fn','stone_dice','stone_precision','stone_recall']))
            check(scope+'_'+profile+'_no_posthoc_gate',d['profiles'][profile]['coverage_feasible']==all(d['profiles'][profile]['coverage_checks'].values()))
        extra['sets'][scope]={'summary':pooled,'frames':hist}
    rows,d=static['video_native'];ids={(r['video_id'],r['frame_index']):i for i,r in enumerate(rows)};pids=[];total=0;selected=0
    state=read(previous/'full_video/member1_2net/video_1/summary.json')['model_state_sha256']
    for name in cfg['full_video']['arms']:
        for vid,n in [('video_1',722),('video_2',1972)]:
            folder=OUT/f'full_video/{name}/{vid}';s=read(folder/'summary.json');pids.append(s['pid']);t=read(folder/'timings.json')
            records=[json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()];old=[json.loads(line) for line in (previous/f'full_video/member1_2net/{vid}/frames.jsonl').read_text().splitlines()]
            for i,r in enumerate(records):
                assert r['frame_index']==i and r['source_rgb_raw_sha256']==old[i]['source_rgb_raw_sha256'] and r['coarse_mask_raw_sha256']==old[i]['mask_raw_sha256']
                a=np.array(Image.open(previous/f'full_video/member1_2net/{vid}/masks/{i:06d}.png'));assert raw(a)==r['coarse_mask_raw_sha256']
                rectangle_union(a.shape,r['boxes']);assert r['crop_proxy_calls']==len(r['boxes'])<=2
                if name=='member1_raw':assert not r['boxes']
                if name=='hard2':assert r['boxes']==[list(b) for b in fiber_rois(a,None,cfg['proposal'],name,cap=2)]
                if name!='member1_raw' and (vid,i) in ids:assert r['boxes']==d['profiles'][name]['frames'][ids[(vid,i)]]['boxes'];selected+=1
                assert r['refined_mask'] is None and r['segmentation_accuracy'] is None and r['tip_metrics'] is None
            label=name+'_'+vid
            check(label+'_complete_baseline_masks_roi_logs',len(records)==len(t)==n)
            check(label+'_actual_graph_and_state',s['model_count']==2 and s['parameters']==49202790 and s['model_state_sha256']==state)
            check(label+'_timings',all(abs(float(np.percentile([r[k] for r in t],q))-v[p])<1e-9 for k,v in s['timings'].items() for q,p in [(50,'p50'),(95,'p95')]))
            check(label+'_crop_calls',sum(r['crop_proxy_calls'] for r in records)==s['total_crop_proxy_calls'])
            check(label+'_video_complete',sha256(folder/'diagnostic.mp4')==s['video_sha256'] and video_count(folder/'diagnostic.mp4',(434,800))==n)
            total+=n
    check('six_processes_8082frames_selected140',len(set(pids))==6 and total==8082 and selected==140)
    for vid,n in [('video_1',722),('video_2',1972)]:
        c=result['comparison_videos'][vid];check(vid+'_comparison_complete',sha256(c['path'])==c['sha256'] and video_count(c['path'],(434,1600))==n)
    check('not_accuracy_training_or_adoption',result['adopted'] is False and result['default_changed'] is False and result['trained_refiner'] is False and result['training_updates']==result['train_pixels_read']==result['reserved_test_pixels_read']==0 and result['tip_metrics'] is None and result['whole_video_accuracy'] is None)
    atomic_json(OUT/'error_spatial_diagnosis.json',extra)
    atomic_json(OUT/'engineering_checks.json',{'passed':sum(checks.values()),'checks':checks,'auditor_sha256':sha256(Path(__file__)),'full_coarse_mask_records_rechecked':total,'full_hard_proposals_recomputed':2694,'selected_soft_proposals_compared':70,'new_model_forwards':0,'limitation':'Full soft probabilities not cached; soft proposals independently compared on all70 approved frames and checked for bounds/calls on all2694; no all-frame soft probability replay in audit'})
    print(json.dumps({'passed':sum(checks.values()),'full_frames':total,'spatial':{scope:v['summary'] for scope,v in extra['sets'].items()}}))
if __name__=='__main__':main()
