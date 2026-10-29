"""Independent saved-output audit and post-hoc foreground/geometry loss attribution."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'runs/a1_direct_neural_observation_20261029'
IMAGE='fixed_image_observation_template'; NEURAL='fixed_neural_observation_template'; NAMES=('raw_classwise',IMAGE,NEURAL)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def load(p):
    with Image.open(p) as im:return np.asarray(im).copy()
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def matrix(t,p):
    a=t!=255;return np.bincount((3*t[a].astype(int)+p[a]).ravel(),minlength=9).reshape(3,3)
def shape_mask(shape,pose,accepted,prior):
    if not accepted:return np.zeros(shape,bool)
    x,y,angle,length=pose;r=prior['cap_radius_fraction']*min(shape);center=max(1,length-r);yy,xx=np.indices(shape,dtype=np.float32)
    t=(xx-x)*np.cos(angle)+(yy-y)*np.sin(angle);n=-(xx-x)*np.sin(angle)+(yy-y)*np.cos(angle)
    width=r+prior['halfwidth_slope']*(center-t)
    return ((t>=0)&(t<=center)&(np.abs(n)<=width))|((t>=center)&((t-center)**2+n*n<=r*r))
def changes(t,a,b,k):
    valid=t!=255;positive=t==k;old=a==k;new=b==k
    return {'lost_tp':int((valid&positive&old&~new).sum()),'gained_tp':int((valid&positive&~old&new).sum()),'added_fp':int((valid&~positive&~old&new).sum()),'removed_fp':int((valid&~positive&old&~new).sum())}


def main():
    summary=read(OUT/'summary.json'); p=read(OUT/'preflight.json');config=p['base_config'];prior=p['template'];checks={}
    mpath=ROOT/config['manifest'];m=read(mpath);vm=read(mpath.parent/'video_validation_manifest.json')
    checks['all_sources_related_inputs_and_weights_unchanged']=all(digest(ROOT/n)==h for n,h in p['sources'].items()) and all(digest(n)==h for n,h in p['related'].items()) and all(digest(v['path'])==v['sha256'] for v in list(p['checkpoints'].values())+p['videos'])
    checks['validation_only322_files_and_manifest_unchanged']=len(p['inputs']['files'])==322 and all(digest(n)==h for n,h in p['inputs']['files'].items()) and digest(mpath)==p['inputs']['manifest_sha256'] and not p['inputs']['test_pixels_read']
    checks['521_132_144_29_unchanged']=(len(m['train']['labeled']),len(m['val']),len(m['reserved_test']),len(vm['val']))==(521,132,144,29)
    oldpre=read(ROOT/'runs/a1_direct_fiber_template_20261029/preflight.json');oldsummary=read(ROOT/'runs/a1_direct_fiber_template_20261029/summary.json')
    checks['frozen_shape_search_weights_acceptance_tracking_and_original_gates_exact']=prior==oldpre['template'] and config==oldpre['config'] and read(ROOT/p['execution']['protocol'])['gates']==config['gates']
    checks['no_training_shared_calibration_test_access_or_default_adoption']=summary['gradient_updates']==summary['template_calibration_updates']==summary['train_pixels_read']==0 and all(summary[k] is False for k in ('reserved_test_pixels_read','default_model_changed','s2_accepted','stone_memory_started','tip_angle_reference_available'))
    diagnostics={}; visual=[]
    oldlogs={v:{r['frame_index']:r for r in [json.loads(x) for x in (ROOT/f'runs/a1_direct_fiber_template_20261029/{v}/frames.jsonl').read_text(encoding='utf-8').splitlines()]} for v in ('video_1','video_2')}
    for label,rows,folder,key in [('image132',m['val'],'image_native','image'),('video29',vm['val'],'video_native','video29')]:
        saved=summary[key];conf={n:np.zeros((3,3),np.int64) for n in NAMES};pairs={n:{str(k):Counter() for k in (1,2)} for n in (IMAGE,NEURAL)}
        reconstruction=fusion=record_parity=acceptance=probabilities=stone_unchanged=True;accepted=Counter();failures={n:Counter() for n in (IMAGE,NEURAL)};details=[];all_masks=[];all_targets=[]
        for i,(row,record) in enumerate(zip(rows,saved['records'])):
            frame=cv2.cvtColor(np.asarray(Image.open(row['image']).convert('RGB')),cv2.COLOR_RGB2BGR);target=load(row['mask']);masks={n:load(OUT/folder/n/f'{i:04d}.png') for n in NAMES}
            probability=np.load(OUT/folder/'probabilities'/f'{i:04d}.npz')['fiber_probability'];hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV);bins=prior['histogram_bins']
            color=np.asarray(prior['appearance_probability'])[(hsv[:,:,0].astype(int)*bins//180).clip(0,bins-1),(hsv[:,:,1].astype(int)*bins//256).clip(0,bins-1)]
            probabilities &= probability.shape==target.shape and probability.dtype==np.float32 and np.isfinite(probability).all() and probability.min()>=0 and probability.max()<=1
            oldrecord=oldsummary['image']['records'][i] if label=='image132' else oldlogs[row['video_id']][row['frame_index']]
            record_parity &= record['outputs'][IMAGE]==oldrecord['outputs']['image_template']
            for n,field,suffix in [(IMAGE,color,'image'),(NEURAL,probability,'neural')]:
                r=record['outputs'][n];hypothesis=shape_mask(target.shape,r['pose'],r['accepted'],prior);support=(field>=config['acceptance']['visible_appearance_min'])&(hsv[:,:,2]>10)
                reconstruction &= np.array_equal(masks[n]==2,hypothesis&support) and np.array_equal(load(OUT/folder/f'hypotheses_{suffix}'/f'{i:04d}.png')>0,hypothesis)
                accepted[n]+=int(r['accepted']);failures[n].update(k for k,v in r['checks'].items() if not v)
                f=r['features'];a=config['acceptance']
                expected={'appearance':f['inside']>=a['inside_appearance_min'],'region_contrast':f['contrast']>=a['region_contrast_min'],'boundary':f['boundary']>=a['boundary_contrast_min'],'rgb_edge':f['rgb']>=a['rgb_edge_min'],'fov':f['support']>=a['fov_support_min']} if f else {'fov':False}
                acceptance &= r['checks']==expected and r['accepted']==all(expected.values()) and (not f or np.isclose(f['score'],f['soft']))
                for k in (1,2):pairs[n][str(k)].update(changes(target,masks['raw_classwise'],masks[n],k))
                # Recover old190 stone semantics on pixels not occupied by either fiber arm.
                shared=(masks[n]!=2)&(masks['raw_classwise']!=2)
                stone_unchanged &= np.array_equal(masks[n][shared]==1,masks['raw_classwise'][shared]==1)
            for n,mask in masks.items():
                conf[n]+=matrix(target,mask);fusion &= mask.shape==target.shape and set(np.unique(mask)).issubset({0,1,2})
            truth=target==2;valid=target!=255;support=(probability>=.5)&(hsv[:,:,2]>10);hyp=load(OUT/folder/'hypotheses_neural'/f'{i:04d}.png')>0
            lost=changes(target,masks['raw_classwise'],masks[NEURAL],2)
            detail={'sample_id':row['sample_id'],'index':i,'accepted':record['outputs'][NEURAL]['accepted'],'reference_fiber_pixels':int(truth.sum()),
                'neural_field_supported_reference':int((truth&support).sum()),'reference_outside_neural_field':int((truth&~support).sum()),
                'supported_reference_lost_to_shape_or_rejection':int((truth&support&~hyp).sum()),
                'supported_reference_lost_to_shape':int((truth&support&~hyp).sum()) if record['outputs'][NEURAL]['accepted'] else 0,
                'supported_reference_lost_to_rejection':int((truth&support).sum()) if not record['outputs'][NEURAL]['accepted'] else 0,
                'visible_tp':int((truth&(masks[NEURAL]==2)).sum()),**lost}
            details.append(detail);visual.append((detail['lost_tp'],label,i,row,record,folder))
            all_masks.append(masks);all_targets.append(target)
        checks[label+'_complete_inventories_all_native_classes']=len(saved['records'])==len(rows) and bool(fusion) and all(len(list((OUT/folder/n).glob('*.png')))==len(rows) for n in NAMES)
        checks[label+'_independent_confusions']=all(conf[n].tolist()==saved['models'][n]['metrics']['confusion_matrix'] for n in NAMES)
        checks[label+'_fixed_geometry_and_foreground_visibility_exact']=bool(reconstruction and probabilities)
        checks[label+'_color_records_and_masks_reproduce_predecessor']=bool(record_parity) and not any(saved['historical_different_pixels'].values())
        checks[label+'_acceptance_checks_and_no_soft_bonus']=bool(acceptance) and all(accepted[n]==saved['acceptance'][n]['accepted'] and dict(failures[n])==saved['acceptance'][n]['rejection_checks'] for n in (IMAGE,NEURAL))
        # Historical Counter addition omits zero-valued keys; compare all four numerical counts.
        checks[label+'_independent_paired_tp_fp_counts']=all(pairs[n][str(k)][field]==saved['paired_changes'][n][str(k)].get(field,0) for n in (IMAGE,NEURAL) for k in (1,2) for field in ('lost_tp','gained_tp','added_fp','removed_fp'))
        checks[label+'_unoccupied_stone_semantics_unchanged']=bool(stone_unchanged)
        groupok=True
        for group in saved['groups'].values():
            ids=set(group['sample_ids']);indices=[i for i,r in enumerate(rows) if r['sample_id'] in ids]
            groupok &= len(indices)==group['images'] and all(sum((matrix(all_targets[i],all_masks[i][n]) for i in indices),np.zeros((3,3),np.int64)).tolist()==group['models'][n]['confusion_matrix'] for n in NAMES)
        checks[label+'_subgroup_confusions_recomputed']=bool(groupok)
        aggregate={k:sum(d[k] for d in details) for k in details[0] if k not in ('sample_id','index','accepted')}
        aggregate['field_reference_recall']=aggregate['neural_field_supported_reference']/max(1,aggregate['reference_fiber_pixels'])
        aggregate['accepted']=accepted[NEURAL];aggregate['total']=len(rows)
        aggregate['reference_positive_frames']=sum(d['reference_fiber_pixels']>0 for d in details)
        aggregate['accepted_reference_positive_frames']=sum(d['accepted'] and d['reference_fiber_pixels']>0 for d in details)
        source_groups={}
        if label=='image132':
            for source in sorted({r['source_dataset'] for r in rows}):
                ids=[i for i,r in enumerate(rows) if r['source_dataset']==source];models={}
                for n in NAMES:
                    cm=sum((matrix(all_targets[i],all_masks[i][n]) for i in ids),np.zeros((3,3),np.int64));metrics={}
                    for k,name in [(1,'stone'),(2,'fiber')]:
                        tp=int(cm[k,k]);fp=int(cm[:,k].sum()-tp);fn=int(cm[k,:].sum()-tp)
                        metrics[name+'_dice']=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
                    models[n]={'metrics':metrics,'confusion_matrix':cm.tolist()}
                source_groups[source]={'images':len(ids),'models':models}
        diagnostics[label]={'aggregate':aggregate,'per_frame_posthoc':details,'source_groups':source_groups,'scope':'Reference-based attribution AFTER frozen inference only; not a pose/tip oracle, parameter selection or training'}
    groups=summary['image']['groups'];a=set(groups['historical_old190_exposed']['sample_ids']);b=set(groups['not_matched_to_old190_exposure']['sample_ids'])
    checks['25_107_complete_disjoint']=(len(a),len(b),len(a&b))==(25,107,0) and a|b=={r['sample_id'] for r in m['val']}
    gates=config['gates'];screen={n:bool(v['metrics']['stone_dice']>=gates['stone_dice_min'] and v['metrics']['fiber_dice']>=gates['fiber_dice_min'] and v['metrics']['stone_negative_fp_pixel_fraction']<=gates['negative_fp_fraction_max']) for n,v in summary['video29']['models'].items()}
    checks['original_joint_gates_and_conditional_replay_decision']=all(summary['screening'][n]['passed']==v for n,v in screen.items()) and summary['full_replay_eligible']==screen[NEURAL] and summary['full_replay_executed']==screen[NEURAL]
    if not screen[NEURAL]:checks['failed_static_has_no_full_video_or_causal_outputs']=not summary['videos'] and not (OUT/'video_1').exists() and not (OUT/'video_2').exists()
    costs=summary['performance'];checks['three_fresh_processes_actual1_or3_networks']=len({v['pid'] for v in costs.values()})==3 and all((v['model_count'],v['parameter_count'])==((1,24601395) if n==IMAGE else (3,73804185)) and v['warmup_frames']==12 and v['samples']==len(v['raw_samples'])==87 for n,v in costs.items())
    checks['all_cost_statistics_recomputed']=all(np.isclose(v['timings'][key][stat],fn([r[key] for r in v['raw_samples']])) for v in costs.values() for key in v['timings'] for stat,fn in [('mean',np.mean),('p50',np.median),('p95',lambda a:np.percentile(a,95))])
    with zipfile.ZipFile(OUT/'source_snapshot.zip') as z:checks['frozen_snapshot_sources_unique_and_hash_exact']=len(z.namelist())==len(set(z.namelist())) and all(hashlib.sha256(z.read(n.replace('\\','/'))).hexdigest()==h for n,h in p['sources'].items())
    (OUT/'failure_attribution.json').write_text(json.dumps(diagnostics,indent=2)+'\n',encoding='utf-8')
    selected=[]
    for label in ('video29','image132'):
        subset=sorted((v for v in visual if v[1]==label),reverse=True,key=lambda v:v[0]);selected.extend(subset[:2])
        accepted=[v for v in subset if v[4]['outputs'][NEURAL]['accepted']]
        if accepted:selected.append(min(accepted,key=lambda v:v[0]))
    panels=[];selection=[]
    for loss,label,i,row,record,folder in selected:
        frame=cv2.cvtColor(np.asarray(Image.open(row['image']).convert('RGB')),cv2.COLOR_RGB2BGR);tiles=[]
        for title,mask in [('RGB',None),('Reference',load(row['mask'])),('Raw3',load(OUT/folder/'raw_classwise'/f'{i:04d}.png')),('Color',load(OUT/folder/IMAGE/f'{i:04d}.png')),('Neural',load(OUT/folder/NEURAL/f'{i:04d}.png'))]:
            panel=frame.copy()
            if mask is not None:
                for k,color in [(1,(40,40,235)),(2,(235,210,20))]:
                    a=mask==k;panel[a]=(.45*panel[a]+.55*np.asarray(color)).astype(np.uint8)
            if title=='Neural':
                hyp=load(OUT/folder/'hypotheses_neural'/f'{i:04d}.png');contours,_=cv2.findContours(hyp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(panel,contours,-1,(0,255,0),1)
            panel=cv2.resize(panel,(256,256));header=np.zeros((36,256,3),np.uint8)
            cv2.putText(header,f'{title} {label} #{i}',(3,15),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1)
            cv2.putText(header,f'neural accept={record["outputs"][NEURAL]["accepted"]}',(3,30),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
            tiles.append(np.vstack([header,panel]))
        panels.append(np.hstack(tiles));selection.append({'sample_id':row['sample_id'],'split':label,'index':i,'raw_tp_removed':loss})
    if panels:cv2.imwrite(str(OUT/'reviewed_examples.jpg'),np.vstack(panels))
    result={'checks':{k:bool(v) for k,v in checks.items()},'all_passed':bool(all(checks.values())),'unit_tests':{'passed':7,'failed':0,'scope':'4 observation-control and3 frozen-template tests'},'visual_selection':selection,'scope':'Output/inference/source/cost engineering verification; not clinical, independent-case, physical tip/angle or real-time validation'}
    (ROOT/'runs/engineering/direct_neural_observation_checks_20261029.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    assert result['all_passed'],result
    print(json.dumps({'all_passed':True,'checks':len(checks),'attribution':{k:v['aggregate'] for k,v in diagnostics.items()}},indent=2))

if __name__=='__main__':main()
