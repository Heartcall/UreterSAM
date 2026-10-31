"""Frozen T1 failure attribution. No model updates, mask correction or parameter search."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image
from tools.fiber_reference_workspace import read,sha256,load_mask
from tools.fiber_fast_observation import FastFiberObserver,public_observation,grid_to_native,sample
from tools.run_two_network_experts import cache_for,infer,array_hash,model_hash
from tools.run_fiber_temporal_state_control import construct
from tools.run_fiber_blue_aplus import frame_read
from tools.replay_classwise_experts import model_state_hash
from ureter_semis.experiment import atomic_json,seed_training,runtime_identity
CONFIG=ROOT/'configs/fiber_t1_failure_diagnosis_20261030.json'


def distribution(values):
    x=np.asarray(values,float)
    return {'n':len(x),'mean':float(x.mean()),'p05':float(np.percentile(x,5)),'p50':float(np.median(x)),'p95':float(np.percentile(x,95)),'max':float(x.max())} if len(x) else {'n':0,'mean':None,'p05':None,'p50':None,'p95':None,'max':None}


def color_summary(frame,region):
    n=int(region.sum())
    if n==0:return {'n':0}
    bgr=frame[region].reshape(-1,1,3);hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV).reshape(-1,3);lab=cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB).reshape(-1,3)
    rgb=bgr[:,0,::-1].astype(float)/255;chroma=rgb/np.maximum(rgb.sum(1,keepdims=True),1e-6)
    return {'n':n,'RGB_mean':rgb.mean(0).tolist(),'normalized_RGB_mean':chroma.mean(0).tolist(),
            'HSV_opencv_mean':hsv.astype(float).mean(0).tolist(),'HSV_S_q05_q50_q95':np.percentile(hsv[:,1],[5,50,95]).tolist(),
            'HSV_V_q05_q50_q95':np.percentile(hsv[:,2],[5,50,95]).tolist(),'Lab_opencv_mean':lab.astype(float).mean(0).tolist(),
            'hue_note':'HSV mean hue is descriptive linear encoding only, not circular hue model or identity threshold'}


def stage(obs):
    if obs['tracked']['groups']['shaft']['value'] is not None:return 'available'
    current=obs['current'];candidates=current['candidates']
    if not candidates:return 'no_proposal'
    if current['ambiguous']:return 'ambiguous'
    if any(c['shaft_observed'] for c in candidates):return 'qualified_unoriented'
    return 'all_rejected:'+','.join(sorted({c['reason'] for c in candidates}))


def neural_counts(mask,fp,gt):
    valid=gt!=255;truth=gt==2;prediction=mask==2;fn=truth&~prediction;false=valid&prediction&~truth
    values=fp[2,truth]
    return {'gt_fiber_pixels':int(truth.sum()),'tp':int((truth&prediction).sum()),'fp':int(false.sum()),'fn':int(fn.sum()),
            'fn_below01':int((fn&(fp[2]<.1)).sum()),'fn_01_to09':int((fn&(fp[2]>=.1)&(fp[2]<.9)).sum()),'fn_ge09':int((fn&(fp[2]>=.9)).sum()),
            'fp_below09':int((false&(fp[2]<.9)).sum()),'fp_ge09':int((false&(fp[2]>=.9)).sum()),
            'gt_fiber_probability':distribution(values),'fn_probability':distribution(fp[2,fn]),'fp_probability':distribution(fp[2,false]),
            'fiber_argmax_but_missing_in_raw_mask':int((truth&(fp.argmax(0)==2)&~prediction).sum())}


def trace(frame,fp,proposal,candidate,cfg,gt=None):
    """Same frozen strips, plus separately marked GT-conditional measurements.

    No modified proposal/observation is passed to any inference or tracker.
    """
    p=proposal;o=cfg['observation'];h,w=frame.shape[:2];scale=min(h,w)/400;origin,u,n=p['origin'],p['u'],p['normal']
    start,end=p['ends'];length=end-start;stations=np.linspace(*o['shaft_axial_fraction'],o['shaft_stations'])*length+start
    support=p['support'];along=(support-origin)@u;across=(support-origin)@n
    ids=np.array(p['_support_ids']);side=cfg['proposal']['grid_side'];full=grid_to_native(np.stack([ids%side,ids//side],-1),(h,w),side)
    fullalong=(full-origin)@u;result=[];delta=o['rgb_difference_offset_at400']*scale
    for i,t in enumerate(stations):
        active=np.abs(along-t)<=length/8;count=int(active.sum());fullcount=int((np.abs(fullalong-t)<=length/8).sum())
        r={'station':i,'native_center_xy':(origin+t*u).tolist(),'support_after_cap':count,'support_before_cap':fullcount,
           'seed_evidence':count>=4,'edge_pair_passed':candidate['selected_edge_pairs'][i]}
        if count>=4:
            boundary=np.percentile(across[active],[2,98]);width=boundary[1]-boundary[0]
            half=float(np.clip(o['search_halfwidth_fraction_of_local_width']*width,o['search_halfwidth_min_at400']*scale,o['search_halfwidth_max_at400']*scale))
            positions=boundary[:,None]+half*np.linspace(-1,1,o['transverse_samples_per_side'])
            xy=origin+t*u+positions[...,None]*n
            a,v1=sample(frame,xy-delta*n);b,v2=sample(frame,xy+delta*n)
            contrast=np.linalg.norm(a-b,axis=-1)/(255*math.sqrt(3));valid=v1&v2
            r.update(seed_boundary_normal_offsets=boundary.tolist(),search_halfwidth_px=half,
                     valid_samples_per_side=valid.sum(1).tolist(),maximum_contrast_per_side=np.where(valid,contrast,-1).max(1).tolist())
        if gt is not None:
            # Reference crossings along the SAME axis/station. Multiple visible
            # intervals are retained separately; no bridging or physical-tip label.
            q=np.arange(-math.ceil(math.hypot(h,w)),math.ceil(math.hypot(h,w))+1,dtype=float)
            xy=origin+t*u+q[:,None]*n;truth,inbounds=sample((gt==2).astype(np.float32),xy);inside=inbounds&(truth>=.5)
            changes=np.diff(np.pad(inside.astype(np.int8),1));left=np.flatnonzero(changes==1);right=np.flatnonzero(changes==-1)-1
            intervals=[]
            for low,high in zip(left,right):
                offsets=np.array([q[low]-.5,q[high]+.5]);ends=origin+t*u+offsets[:,None]*n
                a,v1=sample(frame,ends-delta*n);b,v2=sample(frame,ends+delta*n)
                con=np.linalg.norm(a-b,axis=-1)/(255*math.sqrt(3));good=v1&v2
                intervals.append({'normal_offsets':offsets.tolist(),'width_px':float(offsets[1]-offsets[0]),
                                  'contrast':con.tolist(),'valid':good.tolist(),
                                  'inside_seed_search_strips':[bool(count>=4 and abs(offsets[k]-boundary[k])<=half) for k in range(2)]})
            r['semantic_reference_intervals_NOT_physical_boundary_truth']=intervals
        result.append(r)
    return result


def main():
    task=read(CONFIG);old=ROOT/task['reference_run'];pre=read(old/'preflight.json');cfg=pre['config'];out=ROOT/task['output']
    if out.exists():raise FileExistsError(out)
    assert all(sha256(ROOT/k)==v for k,v in pre['sources'].items())
    assert all(sha256(k)==v for k,v in pre['inputs']['files'].items())
    assert all(sha256(x['path'])==x['sha256'] for x in pre['checkpoints'].values())
    out.mkdir(parents=True);(out/'probabilities').mkdir();observer=FastFiberObserver(cfg);allscopes={};files={str(CONFIG):sha256(CONFIG),str(old/'preflight.json'):sha256(old/'preflight.json')}
    for name in ['video_native.json','image_native.json','summary.json','engineering_checks.json','known_frame_diagnostics.json']:
        p=old/name;files[str(p)]=sha256(p)
    manifest=read(ROOT/cfg['data']['image_manifest']);training={str(Path(r['image']).resolve()) for r in manifest['train']['labeled']}
    reserved={str(Path(r['image']).resolve()) for r in manifest['reserved_test'] if 'image' in r}
    original29=read(ROOT/'datasets/ureter_semis_kidney_joint_20261028/video_validation_manifest.json')['val'];old_indices={(r['video_id'],r['frame_index']):i for i,r in enumerate(original29)}
    for scope,key in [('video_native','video_manifest'),('image_native','image_manifest')]:
        rows=read(ROOT/cfg['data'][key])['val'];baseline=read(old/f'{scope}.json');records=[]
        for i,(row,saved) in enumerate(zip(rows,baseline['records'])):
            assert str(Path(row['image']).resolve()) not in training|reserved
            frame=frame_read(row['image']);gt=load_mask(row['mask'],frame.shape[:2],{0,1,2,255});mask=load_mask(old/f'{scope}/shared_raw2/{i:04d}.png',frame.shape[:2],{0,1,2})
            path,_=cache_for(row,i,scope=='video_native',old_indices);files[str(path)]=sha256(path)
            with np.load(path) as z:fp=z['a0_member1_native'].copy()
            assert array_hash(fp)==saved['fiber_probability_raw_sha256'] and array_hash(mask)==saved['mask_raw_sha256']
            obs=observer.current(frame,fp);clean=public_observation({k:v for k,v in obs.items() if k!='timings'})
            assert clean==saved['observations']['light_current']['current']
            proposals,_=observer.proposals(frame,fp);bylabel={p['label']:p for p in proposals};candidates=[]
            for c in clean['candidates']:
                width0=c.get('width0_px');width1=c.get('far_seed_width_px');ratio=width1/width0 if width0 and width1 is not None else None
                reversed_ratio=1/ratio if ratio is not None and ratio>0 else None
                nc={**c,'forensic_width_ratio':ratio,'forensic_reversed_ratio':reversed_ratio,
                    'reversal_ratio_alone_in_frozen_range_NOT_revalidated':reversed_ratio is not None and cfg['observation']['distal_proximal_width_ratio'][0]<=reversed_ratio<=cfg['observation']['distal_proximal_width_ratio'][1],
                    'station_trace':trace(frame,fp,bylabel[c['label']],c,cfg,gt)}
                if c['front_observed']:
                    point=np.array(c['visible_front_candidate']);x,y=np.rint(point).astype(int);h,w=gt.shape
                    nc['front_semantic_label_at_nearest_pixel_NOT_tip_error']=int(gt[y,x]) if 0<=x<w and 0<=y<h else None
                    nc['front_native_probability']=float(sample(fp[2],point)[0])
                candidates.append(nc)
            stats=neural_counts(mask,fp,gt);obsrecord=saved['observations']['light_current'];aux={}
            for name,path in row.get('auxiliary_masks',{}).items():
                if name!='laser_dot':continue
                ap=Path(path);files[str(ap)]=sha256(ap);region=np.array(Image.open(ap))>0
                assert region.shape==gt.shape
                aux[name]={'pixels':int(region.sum()),'raw_fiber_overlap':int((region&(mask==2)).sum()),
                           'provided_annotation_unreviewed':True,'not_anatomical_tip_truth':True,
                           'candidate_front_inside':[{ 'label':c['label'],'inside':bool(region[int(round(c['visible_front_candidate'][1])),int(round(c['visible_front_candidate'][0]))]) if 0<=round(c['visible_front_candidate'][1])<gt.shape[0] and 0<=round(c['visible_front_candidate'][0])<gt.shape[1] else False} for c in candidates if c['front_observed']]}
            rawcolors={'reference_fiber':color_summary(frame,gt==2),'raw_fiber_reference_fp':color_summary(frame,(mask==2)&(gt!=2)&(gt!=255)),
                       'reference_fiber_fn':color_summary(frame,(gt==2)&(mask!=2))}
            records.append({'index':i,'sample_id':row['sample_id'],'source_dataset':row.get('source_dataset'),'source_split':row.get('source_split'),
                            'image_path':row['image'],'mask_path':row['mask'],'native_shape':list(gt.shape),'review_status':row.get('review_status'),
                            'background_basis':row.get('background_basis'),'fiber_annotation_completeness':row.get('fiber_annotation_completeness'),
                            'stage':stage(obsrecord),'neural':stats,'candidate_diagnostics':candidates,'appearance':rawcolors,'auxiliary':aux})
        positive=[r for r in records if r['neural']['gt_fiber_pixels']>0]
        counts={k:sum(r['neural'][k] for r in records) for k in ['gt_fiber_pixels','tp','fp','fn','fn_below01','fn_01_to09','fn_ge09','fp_below09','fp_ge09','fiber_argmax_but_missing_in_raw_mask']}
        allscopes[scope]={'frames':len(records),'reference_positive_frames':len(positive),'positive_stage_counts':dict(Counter(r['stage'] for r in positive)),
                          'neural_counts':counts,'records':records}
        atomic_json(out/f'{scope}.json',allscopes[scope]);print(scope+' diagnostic complete',flush=True)
    fullmeta={}
    for vid in ['video_1','video_2']:
        fullmeta[vid]={}
        for arm in cfg['arms']:
            path=old/f'full_video/{arm}/{vid}/frames.jsonl';files[str(path)]=sha256(path);rs=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
            counts=Counter(c['reason'] for r in rs for c in r['current']['candidates'])
            fronts=[c for r in rs for c in r['current']['candidates'] if c['front_observed']]
            fullmeta[vid][arm]={'frames':len(rs),'candidate_reasons':dict(counts),'fronts':len(fronts),
                                'front_profile_exact_range_boundary_hits':sum(any(abs(abs(x)-cfg['observation']['front_axial_half_range_at400'])<1e-9 for x in c['front_profile_offsets_px']) for c in fronts),
                                'history_used_frames':sum(r['tracked'].get('used_provenance')=='history_conditioned' for r in rs),
                                'no_truth_accuracy':None}
    # Five frozen original-model snapshots: not new candidates or performance trials.
    seed_training(42);experts=construct(pre);before=model_state_hash(experts);snapshots=[]
    for vid,displays in task['extra_original_model_snapshot_display_frames'].items():
        rows=[json.loads(s) for s in (old/f'full_video/history_search/{vid}/frames.jsonl').read_text(encoding='utf-8').splitlines()]
        for display in displays:
            i=display-1;path=ROOT/f'runs/fiber_reference_d1_20261029/images/{vid}/{i:06d}.png';files[str(path)]=sha256(path)
            frame=frame_read(path);mask,sp,fp,_=infer(experts,frame);record=rows[i]
            assert array_hash(mask)==record['mask_raw_sha256'] and array_hash(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))==record['source_rgb_raw_sha256']
            obs=observer.current(frame,fp);clean=public_observation({k:v for k,v in obs.items() if k!='timings'});assert clean==record['current']
            proposals,_=observer.proposals(frame,fp);lookup={p['label']:p for p in proposals}
            traces=[{'label':c['label'],'trace':trace(frame,fp,lookup[c['label']],c,cfg)} for c in clean['candidates']]
            history=observer.history(frame,fp,record['tracked']['prior']) if record['history_search_attempted'] else None
            assert public_observation(history)==record['history']
            snapshot=out/f'probabilities/{vid}_{i:06d}.npz';np.savez_compressed(snapshot,stone=sp,fiber=fp)
            snapshots.append({'display_frame':display,'video_id':vid,'frame_index':i,'raw_rgb_path':str(path),'mask_raw_sha256':array_hash(mask),
                              'probability_file':str(snapshot),'fiber_probability_raw_sha256':array_hash(fp),'current':clean,'traces':traces,'history':record['history'],
                              'tracked':record['tracked'],'pixel_truth_available':False,'tip_metrics':None})
    assert model_state_hash(experts)==before
    atomic_json(out/'video_snapshots.json',{'actual_original_model_forward_calls':len(snapshots),'model_state_sha256':before,
                'individual_model_sha256':[model_hash(m) for m in experts.models],'runtime':runtime_identity(experts.device),'snapshots':snapshots,
                'inference_purpose':'five diagnostic original-probability snapshots, no new method or latency benchmark'})
    assert all(sha256(ROOT/k)==v for k,v in pre['sources'].items()) and all(sha256(k)==v for k,v in files.items())
    assert all(sha256(k)==v for k,v in pre['inputs']['files'].items())
    atomic_json(out/'summary.json',{'static':{k:{a:v for a,v in d.items() if a!='records'} for k,d in allscopes.items()},'full_video_metadata':fullmeta,
                                  'actual_new_diagnostic_forward_calls':5,'new_candidate_masks':0,'training_updates':0,'train_pixels_read':0,'reserved_test_pixels_read':0,
                                  'default_changed':False,'threshold_changes':False,'tip_metrics':None,'whole_video_accuracy':None})
    atomic_json(out/'identity.json',{'config_sha256':sha256(CONFIG),'generator_sha256':sha256(Path(__file__)),'frozen_t1_sources':pre['sources'],
                                   'input_files':files,'validation_input_files':pre['inputs']['files'],'weights':pre['checkpoints'],
                                   'new_outputs':{p.relative_to(out).as_posix():sha256(p) for p in out.rglob('*') if p.is_file()},
                                   'read_only_original_data':True,'actual_new_original_forward_calls':5,'train_pixels_read':0,'reserved_test_pixels_read':0})
    print(json.dumps({'completed':True,'static_frames':202,'new_original_snapshot_forwards':5,'new_methods':0}))


if __name__=='__main__':main()
