"""Read-only darkness strata on frozen semantic references, no inferred tip GT."""
from pathlib import Path
import json,math,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from tools.fiber_reference_workspace import read,sha256
from tools.run_two_network_experts import cache_for,array_hash
from tools.run_fiber_blue_aplus import frame_read,panel
from ureter_semis.experiment import atomic_json
OUT=ROOT/'runs/a1_fiber_dark_appearance_diagnosis_20261030'


def metrics(truth,pred,p,valid,region):
    t=truth&region;tp=t&pred;fn=t&~pred;fp=pred&~truth&valid&region
    return {'gt_pixels':int(t.sum()),'tp':int(tp.sum()),'fn':int(fn.sum()),'fp':int(fp.sum()),
            'fn_Pbelow01':int((fn&(p<.1)).sum()),'gt_probability_sum':float(p[t].sum(dtype=np.float64))}


def finish(d):
    d['recall']=d['tp']/d['gt_pixels'] if d['gt_pixels'] else None
    d['mean_probability']=d['gt_probability_sum']/d['gt_pixels'] if d['gt_pixels'] else None
    return d


def main():
    if OUT.exists():raise FileExistsError(OUT)
    pre=read(ROOT/'runs/a1_fiber_temporal_observation_search_20261030/preflight.json');cfg=pre['config'];plan=read(ROOT/'configs/fiber_local_evidence_control_20261030.json')
    assert all(sha256(ROOT/k)==v for k,v in pre['sources'].items())
    assert all(sha256(k)==v for k,v in pre['inputs']['files'].items())
    OUT.mkdir(parents=True);results={};files={};cases={}
    original=read(ROOT/'datasets/ureter_semis_kidney_joint_20261028/video_validation_manifest.json')['val'];indices={(r['video_id'],r['frame_index']):i for i,r in enumerate(original)}
    for scope,key in [('video_native','video_manifest'),('image_native','image_manifest')]:
        rows=read(ROOT/cfg['data'][key])['val'];base={m:read(ROOT/f'runs/a1_two_network_experts_20261029/{scope}/{m}.json') for m in ['member1_2net','classwise3']}
        records=[]
        for i,row in enumerate(rows):
            frame=frame_read(row['image']);gt=np.array(Image.open(row['mask']));truth=gt==2;valid=gt!=255;h,w=gt.shape
            hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV);lab=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)
            # Equal-width encoded-image bins fixed before looking at outcomes.
            # L* is represented by OpenCV0..255; not radiometric illumination.
            grouping={'HSV_V':np.minimum(hsv[:,:,2]//85,2),'Lab_Lstar':np.minimum(lab[:,:,0]//85,2)}
            path,_=cache_for(row,i,scope=='video_native',indices);files[str(path)]=sha256(path)
            with np.load(path) as z:stone=z['old190_native'].copy();fiber={'member1_2net':z['a0_member1_native'].copy(),'classwise3':z['a0_native'].copy()}
            record={'sample_id':row['sample_id'],'index':i,'video_id':row.get('video_id'),'frame_index':row.get('frame_index'),'models':{}}
            for model,prob in fiber.items():
                mask=np.zeros(gt.shape,np.uint8);mask[stone.argmax(0)==1]=1;mask[prob.argmax(0)==2]=2
                assert array_hash(mask)==base[model]['records'][i]['mask_raw_sha256'] and array_hash(prob)==base[model]['records'][i]['fiber_probability_raw_sha256']
                pred=mask==2;stats={}
                distance=cv2.distanceTransform(truth.astype(np.uint8),cv2.DIST_L2,5);interior=distance>3*min(h,w)/400
                rad=max(1,int(math.floor(2*min(h,w)/400+.5)));kernel=np.ones((rad*2+1,rad*2+1),np.uint8)
                band=cv2.dilate(pred.astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)^cv2.erode(pred.astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
                eligible=(band>0)&(mask!=1)&(prob[2]>=.1)&(prob[2]<.9)
                admitted=int(eligible.sum())<=math.floor(.05*h*w)
                for feature,groups in grouping.items():
                    stats[feature]=[]
                    for k in range(3):
                        region=groups==k;d=metrics(truth,pred,prob[2],valid,region)
                        d['interior']=metrics(truth,pred,prob[2],valid,region&interior)
                        d['eligible_FN_scope_only_NOT_recovery']=int((truth&~pred&region&eligible).sum()) if admitted else 0
                        stats[feature].append(d)
                # Reference-conditioned end gaps are only a coverage diagnostic.
                # No end is named an anatomical tip or used to guide a prediction.
                yy,xx=np.nonzero(truth);gap=None
                if len(xx)>=2:
                    pts=np.stack([xx,yy],-1);center=pts.mean(0);_,vectors=np.linalg.eigh((pts-center).T@(pts-center));u=vectors[:,-1]
                    if u[0]<0:u=-u
                    axial=(pts-center)@u;hit=pred[yy,xx]
                    gap={'reference_axial_span_px':float(np.ptp(axial)),'axis_is_unoriented_not_true_entry_or_tip':True,
                         'end0_uncovered_px':float(axial[hit].min()-axial.min()) if hit.any() else None,
                         'end1_uncovered_px':float(axial.max()-axial[hit].max()) if hit.any() else None,
                         'no_reference_true_positive':not bool(hit.any())}
                record['models'][model]={'brightness':stats,'reference_projection_coverage_NOT_tip_error':gap}
                if scope=='video_native' and model=='member1_2net':
                    score=stats['Lab_Lstar'][0]['fn'];vid=row['video_id']
                    if vid not in cases or score>cases[vid][0]:cases[vid]=(score,row,frame.copy(),gt.copy(),mask.copy(),prob.copy(),record)
            records.append(record)
        scopes={}
        for model in fiber:
            scopes[model]={}
            for cohort in ['all']+(['video_1','video_2'] if scope=='video_native' else []):
                rr=[r for r in records if cohort=='all' or r['video_id']==cohort];result={}
                for feature in grouping:
                    bins=[]
                    for k in range(3):
                        dd=[r['models'][model]['brightness'][feature][k] for r in rr]
                        d={name:sum(x[name] for x in dd) for name in ['gt_pixels','tp','fn','fp','fn_Pbelow01','gt_probability_sum','eligible_FN_scope_only_NOT_recovery']}
                        d['interior']=finish({name:sum(x['interior'][name] for x in dd) for name in ['gt_pixels','tp','fn','fp','fn_Pbelow01','gt_probability_sum']})
                        bins.append(finish(d))
                    result[feature]=bins
                scopes[model][cohort]=result
        results[scope]={'frames':len(rows),'models':scopes,'records':records}
        atomic_json(OUT/f'{scope}.json',results[scope])
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15);board=Image.new('RGB',(2000,2*510),(12,12,12));draw=ImageDraw.Draw(board);selection=[]
    for j,vid in enumerate(['video_1','video_2']):
        score,row,frame,gt,mask,prob,record=cases[vid];i=row['frame_index'];fn=(gt==2)&(mask!=2);error=frame.copy();error[fn]=(0,0,255)
        heat=cv2.applyColorMap(np.rint(np.clip(prob[2],0,1)*255).astype(np.uint8),cv2.COLORMAP_VIRIDIS)
        pics=[panel(frame,'RGB',None,i),panel(frame,'Reviewed semantic GT (not tip GT)',gt,i),panel(frame,'Frozen member1 raw mask',mask,i),panel(heat,'P_fiber: dark0 yellow1',None,i),panel(error,'Reference FN in red',None,i)]
        for k,pic in enumerate(pics):board.paste(Image.fromarray(cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)),(k*400,j*510))
        draw.text((4,j*510+443),f'{vid} frame{i+1}: largest low-L* reference-FN count in this case ({score} pixels), not tip-selected truth.',font=font,fill='white')
        gap=record['models']['member1_2net']['reference_projection_coverage_NOT_tip_error'];draw.text((4,j*510+470),f'Unoriented reference-axis uncovered ends: {gap["end0_uncovered_px"]:.2f}/{gap["end1_uncovered_px"]:.2f} px; NOT anatomical-tip error.',font=font,fill='white')
        selection.append({'video_id':vid,'frame_index':i,'display_frame':i+1,'sample_id':row['sample_id'],'selection_rule':'max low-Lstar reference FN in this case, all rows remain reported','low_Lstar_FN':score})
    board.save(OUT/'dark_appearance_reference_board.jpg',quality=94)
    atomic_json(OUT/'selection.json',selection)
    atomic_json(OUT/'summary.json',{'scope':'darkness of visible semantic fiber pixels, not confirmed tip regions or measured scene illumination',
        'HSV_V_bins':[0,85,170,256],'Lab_Lstar_encoded_bins':[0,85,170,256], 'Lab_Lstar_approx_ranges':['0..33','33..67','67..100'],
        'pooled_pixels_not_independent_cases':True,'new_model_forwards':0,'new_corrected_masks':0,'training_updates':0,'tip_truth':False,
        'results':{k:{a:v for a,v in d.items() if a!='records'} for k,d in results.items()}})
    assert all(sha256(ROOT/k)==v for k,v in pre['sources'].items()) and all(sha256(k)==v for k,v in pre['inputs']['files'].items())
    atomic_json(OUT/'identity.json',{'generator_sha256':sha256(Path(__file__)),'cache_files':files,'reference_files':pre['inputs']['files'],
        'frozen_sources':pre['sources'],'frozen_T2P_config_sha256':sha256(ROOT/'configs/fiber_local_evidence_control_20261030.json'),
        'outputs':{p.relative_to(OUT).as_posix():sha256(p) for p in OUT.iterdir() if p.is_file()},'new_model_forwards':0})
    print(json.dumps({'complete':True,'new_forwards':0,'new_corrected_masks':0,'selected_frames':selection}))


if __name__=='__main__':main()
