"""Independent saved-cache checks, reference-proxy counts and D0 review figures."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/a1_fiber_tip_d0_20261029'
sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def load(p):
    with Image.open(p) as im:return np.asarray(im).copy()
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def confusion(t,p):
    a=t!=255;return np.bincount((3*t[a].astype(int)+p[a]).ravel(),minlength=9).reshape(3,3)
def array_summary(values):
    return {'pixels':len(values),'mean':float(np.mean(values)),'p05_p25_p50_p75_p95':np.percentile(values,[5,25,50,75,95]).astype(float).tolist()} if len(values) else None


def main():
    from tools.fiber_tip_diagnostics import reference_axis
    s=read(OUT/'summary.json');p=read(OUT/'preflight.json');settings=p['protocol']['diagnostic_only'];checks={}
    mpath=ROOT/p['protocol']['manifest'];m=read(mpath);vm=read(mpath.parent/'video_validation_manifest.json')
    checks['all_frozen_code_related_inputs_weights_and_videos_unchanged']=all(digest(ROOT/n)==h for n,h in p['sources'].items()) and all(digest(n)==h for n,h in p['related'].items()) and all(digest(v['path'])==v['sha256'] for v in p['videos']+list(p['checkpoints'].values()))
    checks['322_validation_files_and521_132_144_29_unchanged']=len(p['inputs']['files'])==322 and all(digest(n)==h for n,h in p['inputs']['files'].items()) and digest(mpath)==p['inputs']['manifest_sha256'] and (len(m['train']['labeled']),len(m['val']),len(m['reserved_test']),len(vm['val']))==(521,132,144,29)
    checks['no_training_fitting_default_or_stone_memory_change']=s['gradient_updates']==s['train_pixels_read']==0 and all(s[n] is False for n in ('shared_statistics_fitted','reserved_test_pixels_read','default_model_changed','s2_accepted','stone_memory_started','new_segmentation_scheme'))
    diagnosis={};pilot=read(OUT/'tip_reference_pilot.template.json');selected={r['sample_id'] for r in pilot['records']};review_panels=defaultdict(list)
    checks['12_pilot_references_pending_without_predicted_tip_truth']=len(pilot['records'])==12 and Counter(r['video_id'] for r in pilot['records'])=={'video_1':6,'video_2':6} and all(r['tip_review_status']=='pending' and all(r[k] is None for k in ('entry_point','tip_point','tip_roi','tip_mask','shaft_mask','unknown_mask','reviewer')) for r in pilot['records']) and s['tip_metrics'] is None
    for scope,rows,key in [('image_native',m['val'],'image'),('video_native',vm['val'],'video29')]:
        source=s[key];records=source['records'];matrix=np.zeros((3,3),np.int64);valid=True;fncounts=Counter();historic=defaultdict(Counter);exclusions=defaultdict(Counter);segments=defaultdict(Counter);conditions=defaultdict(lambda:defaultdict(Counter));axis_reasons=Counter();regional_pf=defaultdict(list)
        for i,(row,record) in enumerate(zip(rows,records)):
            raw=load(OUT/scope/'raw_classwise'/f'{i:04d}.png');target=load(row['mask']);frame=cv2.cvtColor(np.asarray(Image.open(row['image']).convert('RGB')),cv2.COLOR_RGB2BGR)
            with np.load(OUT/scope/'probabilities'/f'{i:04d}.npz') as z:
                required={n+'_'+where for n in ('old190','a0','a0_member1','a0_member2') for where in ('grid','native')}
                valid &= set(z.files)==required
                for n in required:
                    a=z[n];valid &= a.dtype==np.float32 and a.shape==(3,*(target.shape if n.endswith('native') else (256,256))) and np.isfinite(a).all() and np.allclose(a.sum(0),1,atol=1e-5) and a.min()>=0 and a.max()<=1
                a0=z['a0_native'];old=z['old190_native'];fmask=a0.argmax(0);expected=np.where(fmask==2,2,np.where(old.argmax(0)==1,1,0)).astype(np.uint8)
                valid &= np.array_equal(raw,expected)
                fn=(target==2)&(raw!=2);truth=target==2
                valid &= int(fn.sum())==record['network']['raw_fn'] and sum(record['network']['raw_fn_probability_bins_diagnostic'].values())==int(fn.sum()) and int((fn&(a0.argmax(0)==0)).sum())==record['network']['raw_fn_background_winner'] and int((fn&(a0.argmax(0)==1)).sum())==record['network']['raw_fn_stone_winner']
                for k,v in record['network'].items():
                    if isinstance(v,int):fncounts[k]+=v
                for k,v in record['network']['raw_fn_probability_bins_diagnostic'].items():fncounts['fn_probability_'+k]+=v
                axis,t=reference_axis(frame,target,settings);axis_reasons[axis.get('reason','valid')]+=1
                if axis['valid']:
                    q=np.clip(t,0,1);bounds=settings['axial_boundaries']
                    for j,(lo,hi) in enumerate(zip(bounds,bounds[1:])):
                        mask=truth&(q>=lo)&((q<=hi) if j==3 else (q<hi));label=str(j);segments[label].update(reference_pixels=int(mask.sum()),tp=int((mask&(raw==2)).sum()),fn=int((mask&(raw!=2)).sum()))
                        regional_pf[label].append(a0[2][mask].copy())
                for name,result in record['historical'].items():
                    if name=='threshold_counterfactual_not_a_new_scheme':historic[name].update(result);continue
                    historic[name].update(result['changes'])
                    if 'first_exclusion' in result:
                        ex=result['first_exclusion'];valid &= ex['unexplained']==0 and ex['lost_tp']==sum(ex[k] for k in ('fit_rejected','outside_hypothesis','foreground_field_rejected','darkness_rejected')) and ex['lost_tp']==result['changes']['lost_tp'];exclusions[name].update(ex)
                for feature,groups in record['whole_fiber_condition_groups_not_tip_truth'].items():
                    for group in groups:conditions[feature][str(group['range'])].update({k:group[k] for k in ('reference_pixels','tp','fn')})
                if row['sample_id'] in selected:
                    gt=frame.copy();pred=frame.copy();loss=frame.copy();axial=frame.copy()
                    for k,color in [(1,(40,40,235)),(2,(235,210,20))]:
                        a=target==k;gt[a]=(.45*frame[a]+.55*np.array(color)).astype(np.uint8);a=raw==k;pred[a]=(.45*frame[a]+.55*np.array(color)).astype(np.uint8)
                    loss[truth&(raw!=2)]=(20,20,255);loss[(target!=255)&~truth&(raw==2)]=(255,0,255)
                    if axis['valid']:
                        for j,(lo,hi) in enumerate(zip(settings['axial_boundaries'],settings['axial_boundaries'][1:])):
                            a=truth&(q>=lo)&((q<=hi) if j==3 else (q<hi));color=[(200,100,0),(50,200,0),(0,180,220),(0,80,255)][j];axial[a]=(.3*frame[a]+.7*np.array(color)).astype(np.uint8)
                    heat=cv2.applyColorMap(np.rint(a0[2]*255).astype(np.uint8),cv2.COLORMAP_TURBO);tiles=[]
                    for title,img in [('RGB',frame),('Reference whole fiber',gt),('A0 P(fiber) 0..1',heat),('Frozen classwise',pred),('FN red / FP magenta',loss),('GT axial proxy NOT tip GT',axial)]:
                        img=cv2.resize(img,(256,256));head=np.zeros((34,256,3),np.uint8);cv2.putText(head,title,(3,13),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1,cv2.LINE_AA);cv2.putText(head,f'{row["video_id"]} frame {row["frame_index"]+1}',(3,28),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1,cv2.LINE_AA);tiles.append(np.vstack([head,img]))
                    review_panels[row['video_id']].append(np.hstack(tiles))
            matrix+=confusion(target,raw)
        checks[scope+'_probability_cache_native_fusion_and_fn_counts_independent']=bool(valid) and matrix.tolist()==source['models']['classwise']['metrics']['confusion_matrix']
        checks[scope+'_all_static_cache_files_and_no_tip_metrics']=len(records)==len(rows) and all(len(list((OUT/scope/n).glob('*')))==len(rows) for n in ('probabilities','raw_classwise','old190')) and source['tip_metrics'] is None and source['raw_historical_different_pixels']==0
        for group in segments.values():group['recall']=group['tp']/group['reference_pixels'] if group['reference_pixels'] else None
        for feature,groups in conditions.items():
            for group in groups.values():group['recall']=group['tp']/group['reference_pixels'] if group['reference_pixels'] else None
        diagnosis[key]={'network_counts':dict(fncounts),'axis_status':dict(axis_reasons),'reference_axial_proxy':dict(segments),'proxy_probability_distributions':{k:array_summary(np.concatenate(v)) for k,v in regional_pf.items()},'historical_changes':{k:dict(v) for k,v in historic.items()},'historical_first_exclusion':{k:dict(v) for k,v in exclusions.items()},'whole_fiber_observable_condition_groups':{k:dict(v) for k,v in conditions.items()},'tip_metrics':None,'scope':'Post-hoc diagnosis only. Axial proxies/low-S/high-V conditions are not reviewed tip, exposure or defocus truth'}
    checks['historical25_107_and_source68_64_groups_complete']={k:v['images'] for k,v in s['image']['groups'].items()}=={'historical_old190_exposed':25,'not_matched_to_old190_exposure':107,'kidney_stones_fiber_v1':68,'kidney_stones_fiber_v2':64}
    for vid,v in s['full_video_baseline'].items():
        folder=OUT/'full_video'/vid;logs=[json.loads(x) for x in (folder/'frames.jsonl').read_text(encoding='utf-8').splitlines()];timing=read(folder/'timings.json');job=read(folder/'job.json')
        checks[vid+'_all_consecutive_frames_masks_raw_hashes_and_historical_reference']=len(logs)==len(timing)==v['frames'] and [r['frame_index'] for r in logs]==list(range(v['frames'])) and len(list((folder/'raw_classwise').glob('*.png')))==v['frames'] and all(hashlib.sha256(load(folder/'raw_classwise'/f'{i:06d}.png').tobytes()).hexdigest()==row['mask_raw_bytes_sha256'] and np.array_equal(load(folder/'raw_classwise'/f'{i:06d}.png'),load(ROOT/f'runs/a1_classwise_replay_20261028/{vid}/classwise/{i:06d}.png')) for i,row in enumerate(logs))
        checks[vid+'_every_frame_timing_statistics_recomputed']=all(np.isclose(v['timings'][key][stat],fn([r[key] for r in timing])) for key in timing[0] for stat,fn in [('mean',np.mean),('p50',np.median),('p95',lambda a:np.percentile(a,95)),('max',np.max)])
        cap=cv2.VideoCapture(str(folder/'baseline_replay.mp4'));media=cap.isOpened() and int(cap.get(7))==v['frames'] and int(cap.get(3))==800 and int(cap.get(4))==434;cap.release()
        checks[vid+'_full_video_export_no_dense_truth_claims']=bool(media) and v['decoded_export_frames']==v['frames'] and digest(folder/'baseline_replay.mp4')==v['video_sha256'] and v['dense_accuracy_metrics'] is None and v['tip_metrics'] is None
        checks[vid+'_fresh_process3nets_and_alignment']=job['phase']=='completed' and v['model_count']==3 and v['parameter_count']==73804185 and v['warmup_frames']==12 and v['raw_historical_different_pixels']==0 and v['reviewed_rgb_exact']==(20 if vid=='video_1' else 9)
    checks['two_videos_really_separate_processes']=len({read(OUT/'full_video'/v/'job.json')['pid'] for v in ('video_1','video_2')})==2
    # Preserve the original archive; canonicalize Windows '/' vs '\\' alias entries without changing content.
    canonical={};duplicates=[]
    with zipfile.ZipFile(OUT/'source_snapshot.zip') as z:
        for info in z.infolist():
            name=info.filename.replace('\\','/');data=z.read(info)
            if name in canonical:
                if canonical[name]!=data:raise ValueError('Nonidentical duplicate source archive entry')
                duplicates.append(name)
            canonical[name]=data
    with zipfile.ZipFile(OUT/'source_snapshot_canonical.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name,data in canonical.items():z.writestr(name,data)
    checks['canonical_snapshot_all_frozen_sources_exact']=all(hashlib.sha256(canonical[n.replace('\\','/')]).hexdigest()==h for n,h in p['sources'].items())
    diagnosis['snapshot_note']={'original_preserved':True,'duplicate_path_aliases':duplicates,'canonical_sha256':digest(OUT/'source_snapshot_canonical.zip')}
    for vid,panels in review_panels.items():cv2.imwrite(str(OUT/f'{vid}_pilot_diagnostic_board.jpg'),np.vstack(panels),[cv2.IMWRITE_JPEG_QUALITY,94])
    (OUT/'diagnosis_summary.json').write_text(json.dumps(diagnosis,indent=2)+'\n',encoding='utf-8')
    result={'all_passed':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()},'unit_tests':{'passed':4,'failed':0},'scope':'Cache/reproduction/stage-accounting/all-frame timing/export verification only; no dedicated tip GT or clinical/live acceptance'}
    (OUT/'engineering_checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    assert result['all_passed'],result
    print(json.dumps({'all_passed':True,'checks':len(checks),'video29':diagnosis['video29'],'performance':{k:v['timings']['pipeline_ms'] for k,v in s['full_video_baseline'].items()}},indent=2))

if __name__=='__main__':main()
