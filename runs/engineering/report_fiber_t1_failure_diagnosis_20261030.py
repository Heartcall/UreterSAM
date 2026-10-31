"""Read-only attribution tables/figures and prospective scope bounds, no corrected mask."""
from collections import Counter
import json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from tools.fiber_reference_workspace import read,sha256,load_mask
from tools.fiber_fast_observation import FastFiberObserver
from tools.run_two_network_experts import cache_for
from tools.run_fiber_blue_aplus import frame_read,panel
from ureter_semis.experiment import atomic_json
OUT=ROOT/'runs/a1_fiber_t1_failure_diagnosis_20261030'
PLAN=ROOT/'configs/fiber_local_evidence_control_20261030.json'


def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'


def main():
    task=read(ROOT/'configs/fiber_t1_failure_diagnosis_20261030.json');old=ROOT/task['reference_run'];pre=read(old/'preflight.json');cfg=pre['config'];plan=read(PLAN)
    assert not (ROOT/plan['planned_output']).exists() and all(not(ROOT/p).exists() for p in plan['planned_files'])
    plan_hash_before=sha256(PLAN);observer=FastFiberObserver(cfg);analysis={};scope_records={};plot_cases=[]
    original29=read(ROOT/'datasets/ureter_semis_kidney_joint_20261028/video_validation_manifest.json')['val'];indices={(r['video_id'],r['frame_index']):i for i,r in enumerate(original29)}
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    for scope,key in [('video_native','video_manifest'),('image_native','image_manifest')]:
        data=read(OUT/f'{scope}.json');rows=read(ROOT/cfg['data'][key])['val'];records=data['records'];scope_records[scope]=records
        opportunity=[];quartiles=[{'pixels':0,'tp':0,'probability_sum':0.,'S_sum':0.,'V_sum':0.} for _ in range(4)];sv=[{'pixels':0,'tp':0,'probability_sum':0.} for _ in range(9)]
        for i,(row,r) in enumerate(zip(rows,records)):
            frame=frame_read(row['image']);gt=load_mask(row['mask'],frame.shape[:2],{0,1,2,255});mask=load_mask(old/f'{scope}/shared_raw2/{i:04d}.png',frame.shape[:2],{0,1,2})
            path,_=cache_for(row,i,scope=='video_native',indices)
            with np.load(path) as z:fp=z['a0_member1_native'].copy()
            h,w=gt.shape;scale=min(h,w)/400;rad=max(1,int(math.floor(plan['eligible_pixels']['boundary_radius_at400']*scale+.5)))
            kernel=np.ones((2*rad+1,2*rad+1),np.uint8);fiber=(mask==2).astype(np.uint8)
            band=cv2.dilate(fiber,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)^cv2.erode(fiber,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
            eligible=(band>0)&(fp[2]>=plan['eligible_pixels']['fiber_probability_min_inclusive'])&(fp[2]<plan['eligible_pixels']['fiber_probability_max_exclusive'])&(mask!=1)
            count=int(eligible.sum());admitted=count<=math.floor(plan['eligible_pixels']['maximum_eligible_image_fraction']*h*w)
            valid=gt!=255;tp=(mask==2)&(gt==2);fpix=(mask==2)&valid&(gt!=2);fn=(mask!=2)&(gt==2)
            opportunity.append({'sample_id':row['sample_id'],'eligible_pixels':count,'eligible_fraction':count/(h*w),'within_pre_sampling_budget':admitted,
                                'eligible_reference_tp_at_risk':int((eligible&tp).sum()),'eligible_reference_fp':int((eligible&fpix).sum()),'eligible_reference_fn':int((eligible&fn).sum()),
                                'raw_stone_protected_gt_fiber':int(((mask==1)&(gt==2)).sum()),'changed_area_budget_pixels':math.floor(plan['filter']['maximum_changed_image_fraction']*h*w)})
            yy,xx=np.nonzero(gt==2)
            if len(xx):
                points=np.stack([xx,yy],-1);center=points.mean(0);_,vectors=np.linalg.eigh((points-center).T@(points-center));u=vectors[:,-1]
                if u[0]<0 or(u[0]==0 and u[1]<0):u=-u
                axial=(points-center)@u;normalized=(axial-axial.min())/max(float(np.ptp(axial)),1e-9);groups=np.minimum((normalized*4).astype(int),3)
                hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV);pvals=fp[2,yy,xx];correct=mask[yy,xx]==2;S=hsv[yy,xx,1];V=hsv[yy,xx,2]
                for k in range(4):
                    active=groups==k;d=quartiles[k];d['pixels']+=int(active.sum());d['tp']+=int(correct[active].sum());d['probability_sum']+=float(pvals[active].sum(dtype=np.float64));d['S_sum']+=float(S[active].sum(dtype=np.float64));d['V_sum']+=float(V[active].sum(dtype=np.float64))
                svgroup=np.minimum(S//85,2)*3+np.minimum(V//85,2)
                for k in range(9):
                    active=svgroup==k;d=sv[k];d['pixels']+=int(active.sum());d['tp']+=int(correct[active].sum());d['probability_sum']+=float(pvals[active].sum(dtype=np.float64))
        for groups in [quartiles,sv]:
            for d in groups:
                d['recall']=d['tp']/d['pixels'] if d['pixels'] else None;d['mean_probability']=d['probability_sum']/d['pixels'] if d['pixels'] else None
                if 'S_sum' in d:d['mean_S']=d['S_sum']/d['pixels'] if d['pixels'] else None;d['mean_V']=d['V_sum']/d['pixels'] if d['pixels'] else None
        trace_counts=Counter();single=both=outside=0;width_rejected=[];ambiguous=[];gross=[]
        for r in records:
            if r['neural']['gt_fiber_pixels']==0:continue
            for c in r['candidate_diagnostics']:
                if c['reason']=='width_profile':width_rejected.append({'sample_id':r['sample_id'],'ratio':c['forensic_width_ratio'],'reverse_ratio_in_range':c['reversal_ratio_alone_in_frozen_range_NOT_revalidated'],'oriented_by_fov':c['oriented']})
                for t in c['station_trace']:
                    if not t['edge_pair_passed']:
                        why='no_seed_support' if not t['seed_evidence'] else 'no_inbounds_samples_on_one_side' if min(t['valid_samples_per_side'])==0 else 'contrast_or_crossed_edge_selection'
                        trace_counts[why]+=1
                    intervals=t['semantic_reference_intervals_NOT_physical_boundary_truth']
                    if len(intervals)==1 and all(intervals[0]['valid']):
                        single+=1;z=intervals[0]
                        if min(z['contrast'])>=cfg['observation']['rgb_edge_contrast_min']:
                            both+=1
                            if not all(z['inside_seed_search_strips']):outside+=1
        totals={k:sum(r[k] for r in opportunity) for k in ['eligible_pixels','eligible_reference_tp_at_risk','eligible_reference_fp','eligible_reference_fn','raw_stone_protected_gt_fiber']}
        admitted={k:sum(r[k] for r in opportunity if r['within_pre_sampling_budget']) for k in ['eligible_pixels','eligible_reference_tp_at_risk','eligible_reference_fp','eligible_reference_fn']}
        analysis[scope]={'eligible_scope_plan_sha256_before_calculation':plan_hash_before,'scope_only_not_filter_execution':True,
                         'eligible_uncapped_totals':totals,'after_presampling_budget_totals':admitted,'budget_rejected_frames':sum(not r['within_pre_sampling_budget'] for r in opportunity),
                         'records':opportunity,'unoriented_reference_PCA_quartiles_NOT_tip_segments':quartiles,
                         'reference_SV_bins_S_thenV_0to84_85to169_170to255_NOT_tip_metrics':sv,
                         'failed_candidate_station_reasons':dict(trace_counts),'reference_single_intervals':single,'reference_bilateral_contrast_intervals':both,
                         'reference_contrast_intervals_outside_seed_strips':outside,'width_rejection_arithmetic':width_rejected,
                         'point_contact_labels_NOT_tip_accuracy':dict(Counter(str(c['front_semantic_label_at_nearest_pixel_NOT_tip_error']) for r in records for c in r['candidate_diagnostics'] if c['front_observed']))}
        if scope=='image_native':
            predicates=[('width-direction conflict',lambda r:r['stage']=='all_rejected:width_profile' and r['neural']['gt_fiber_pixels']>0),
                        ('short visible extent',lambda r:r['stage']=='all_rejected:shaft_too_short' and r['neural']['gt_fiber_pixels']>0),
                        ('qualified but unoriented',lambda r:r['stage']=='qualified_unoriented'),
                        ('line-residual rejection',lambda r:r['stage']=='all_rejected:line_residual' and r['neural']['gt_fiber_pixels']>0),
                        ('provided laser-dot / accepted pose',lambda r:r['stage']=='available' and r['neural']['gt_fiber_pixels']==0),
                        ('no seed,445 reference pixels',lambda r:r['neural']['gt_fiber_pixels']==445 and not r['candidate_diagnostics'])]
            for title,predicate in predicates:
                r=next(r for r in records if predicate(r));plot_cases.append((title,r,rows[r['index']]))
    assert sha256(PLAN)==plan_hash_before
    atomic_json(OUT/'forensic_summary.json',analysis)
    # Fixed category-first evidence boards, never a success-selected metric cohort.
    selections=[];board=Image.new('RGB',(1600,len(plot_cases)*510),(12,12,12));draw=ImageDraw.Draw(board)
    for j,(title,r,row) in enumerate(plot_cases):
        frame=frame_read(row['image']);gt=np.array(Image.open(row['mask']));i=r['index'];path,_=cache_for(row,i,False,indices)
        with np.load(path) as z:fp=z['a0_member1_native'].copy()
        current=observer.current(frame,fp);proposals,_=observer.proposals(frame,fp)
        heat=cv2.applyColorMap(np.rint(np.clip(fp[2],0,1)*255).astype(np.uint8),cv2.COLORMAP_VIRIDIS);evidence=frame.copy()
        for c,p in zip(current['candidates'],proposals):
            a=np.array(c.get('entry_xy',p['origin']+p['ends'][0]*p['u']));b=np.array(c.get('far_seed_xy',p['origin']+p['ends'][1]*p['u']))
            cv2.arrowedLine(evidence,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),(0,255,255),1,cv2.LINE_AA,tipLength=.12)
            cv2.circle(evidence,tuple(np.rint(a).astype(int)),3,(0,0,255),-1)
            diag=next(d for d in r['candidate_diagnostics'] if d['label']==c['label'])
            for station in diag['station_trace']:
                center=tuple(np.rint(station['native_center_xy']).astype(int));cv2.circle(evidence,center,2,(0,220,0) if station['edge_pair_passed'] else (0,0,255),-1)
        pics=[panel(frame,'RGB',None,i),panel(frame,'Semantic reference (NOT tip GT)',gt,i),panel(heat,'Original P_fiber: dark0 yellow1',None,i),panel(evidence,'Unchanged T1 axis and station evidence',None,i)]
        for k,p in enumerate(pics):board.paste(Image.fromarray(cv2.cvtColor(cv2.resize(p,(400,434)),cv2.COLOR_BGR2RGB)),(k*400,j*510))
        ratio=[round(c['forensic_width_ratio'],3) for c in r['candidate_diagnostics'] if c['forensic_width_ratio'] is not None]
        draw.text((4,j*510+440),f'{title}: {r["sample_id"]}',font=font,fill='white')
        draw.text((4,j*510+464),f'GTfiber={r["neural"]["gt_fiber_pixels"]}, network FN={r["neural"]["fn"]}, stage={r["stage"]}, width ratios={ratio}',font=font,fill='white')
        draw.text((4,j*510+487),'Red=start heuristic; yellow arrow=axis sign; green/red dots=passed/failed stations. No alternative prediction.',font=font,fill='white')
        selections.append({'title':title,'index':r['index'],'sample_id':r['sample_id']})
    board.save(OUT/'static_failure_attribution_board.jpg',quality=94)
    atomic_json(OUT/'figure_selection.json',{'rule':'first manifest row in each fixed failure category; all202 rows retain full statistics','cases':selections})
    snapshots=read(OUT/'video_snapshots.json')['snapshots'];video_board=Image.new('RGB',(1200,2*510),(12,12,12));draw=ImageDraw.Draw(video_board)
    for j,display in enumerate([311,1266]):
        r=next(x for x in snapshots if x['display_frame']==display);frame=frame_read(r['raw_rgb_path'])
        with np.load(r['probability_file']) as z:fp=z['fiber'].copy()
        heat=cv2.applyColorMap(np.rint(fp[2]*255).astype(np.uint8),cv2.COLORMAP_VIRIDIS);evidence=frame.copy()
        for item in r['traces']:
            for t in item['trace']:
                p=np.array(t['native_center_xy']);cv2.circle(evidence,tuple(np.rint(p).astype(int)),3,(0,220,0) if t['edge_pair_passed'] else (0,0,255),-1)
        pics=[panel(frame,'RGB (no pixel GT)',None,display-1),panel(heat,'Original P_fiber, dark0 yellow1',None,display-1),panel(evidence,'Frozen current station support',None,display-1)]
        for k,p in enumerate(pics):video_board.paste(Image.fromarray(cv2.cvtColor(p,cv2.COLOR_BGR2RGB)),(400*k,510*j))
        c=r['current']['candidates'][0];draw.text((4,510*j+442),f'video_2 frame{display}: seed counts all>=4; only {c["edge_pairs"]}/8 bilateral edge pairs.',font=font,fill='white')
        draw.text((4,510*j+468),'One search-side has weak contrast at several stations; no GT => low contrast vs misplaced strip unresolved.',font=font,fill='white')
    video_board.save(OUT/'video_edge_failure_attribution_board.jpg',quality=94)
    print(json.dumps({'read_only_forensic_complete':True,'prospective_plan_unchanged':sha256(PLAN)==plan_hash_before,'new_masks':0,
                      'scope_opportunity':{k:v['after_presampling_budget_totals'] for k,v in analysis.items()}}))


if __name__=='__main__':main()
