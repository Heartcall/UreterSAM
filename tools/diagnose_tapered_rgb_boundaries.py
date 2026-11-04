"""Read-only full-video attribution; no output segmentation is changed."""
import json,copy,zipfile,importlib.util,time
from pathlib import Path
import cv2,numpy as np
from .fiber_reference_workspace import ROOT,read,sha256,load_mask
from .fiber_local_dark_evidence import public,nearest
from .fiber_straight_semicircle import observed_sides
from .fiber_rgb_side_observation import huber_diagnostics,edge_observations,broad_search_domain,terminal_observations
from .run_two_network_experts import array_hash
from .run_fiber_dark_input_control import checked_writer
from .replay_classwise_experts import stats
from ureter_semis.experiment import atomic_json
CONFIG=ROOT/'configs/tapered_rgb_diagnosis_20261104.json'
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def draw_line(im,p,q,c):cv2.line(im,tuple(np.rint(p).astype(int)),tuple(np.rint(q).astype(int)),c,1,cv2.LINE_AA)
def dots(im,p,c):
    for q in p:cv2.circle(im,tuple(np.rint(q).astype(int)),1,c,-1)
def overlay(frame,mask,color):
    a=frame.copy();a[mask]=(.6*a[mask]+.4*np.asarray(color)).astype(np.uint8);return a
def board(frame,old,new,body,blue,domain,broad,eligible,g,tip,L,R,H,held,meta,vid,i,t):
    images=[frame.copy(),overlay(frame,old==2,(255,210,0))]+[frame.copy() for _ in range(4)]+[overlay(frame,domain,(0,220,220)),overlay(frame,broad,(100,100,0)),overlay(frame,new==2,(255,210,0)),overlay(frame,body&blue&(new!=2),(255,0,255)),frame.copy(),frame.copy()]
    titles=['Original RGB','Old S2 segmentation','Raw min/max section points','Huber weights: green=1 / red=low','RGB external-edge CANDIDATES','Original lines / V / axis / T','Old taper-dependent RGB search','Independent envelope: READ ONLY','Current geometric FINAL mask','Deleted SUBJECT blue (proxy)','80-95% held-out RGB edges','Attribution: no segmentation edits']
    texts=['No marks','Frozen historical result','10-80% of old S2 extent','Coherent inward bias can keep weight 1','Green RGB / yellow-magenta old fit','V is NOT the visible tip','Yellow domain; blue evidence cyan','Extra eligible points green','Applied or fallback exactly as saved','Magenta: old subject + RGB support','Candidates are NOT manual truth',f"loss={meta['subject_blue_loss_rate']:.1%} ({meta['subject_blue_deleted']})"]
    if g:
        o=np.asarray(g['origin']);u=np.asarray(g['axis']);n=np.asarray(g['normal']);V=np.asarray(g['intersection']);T=np.asarray(tip['vertex']) if tip and tip.get('vertex') else None
        for side,a,obs,h in zip([-1,1],g['independent_lines'],[L,R],H):
            p=np.asarray(a['point']);v=np.asarray(a['direction']);color=(0,255,255) if side==-1 else (255,0,255)
            dots(images[2],a['points'],color)
            for q,w in zip(h['points'],h['weights']):dots(images[3],[q],(0,int(255*w),int(255*(1-w))))
            for index in [4,5,10]:draw_line(images[index],p-500*v,p+500*v,color)
            dots(images[4],obs['points'][obs['valid']],(0,255,0))
        draw_line(images[5],o-500*u,o+500*u,(255,255,255));dots(images[5],[V],(0,0,255))
        if T is not None:cv2.drawMarker(images[5],tuple(np.rint(T).astype(int)),(0,255,0),cv2.MARKER_CROSS,9,1)
        images[6][blue&domain]=(255,210,0);images[7][eligible>0]=(0,255,0)
        for a in held:dots(images[10],a['points'][a['valid']],(0,255,0))
        texts[5]=f"V=({V[0]:.1f},{V[1]:.1f}) r={meta.get('radius',0):.2f}px"
        for j,s in enumerate([f"Blue lost: {meta['subject_blue_loss_rate']:.2%}",f"old search excludes blue: {meta.get('blue_outside_old_search',0)}",f"extra terminal px: {meta.get('extra_terminal_pixels',0)}",f"read-only extra length: {meta.get('independent_tip_delta',0) or 0:.2f}px"]):cv2.putText(images[11],s,(4,25+22*j),0,.47,(255,255,255),1)
    tiles=[]
    for im,name,text in zip(images,titles,texts):
        head=np.zeros((56,400,3),np.uint8)
        for j,s in enumerate([name,text,f'{vid} frame={i+1} t={t:.3f}s']):cv2.putText(head,s,(3,15+17*j),0,.33 if j else .39,(255,255,255),1)
        tiles.append(np.vstack([head,im]))
    result=np.vstack([np.hstack(tiles[k:k+4]) for k in [0,4,8]])
    assert np.array_equal(result[56:456,:400],frame)
    return result
def signed_edge_distance(obs,line,axis,side):
    v=np.asarray(line['direction']);n=np.array([-v[1],v[0]]);out=np.array([-axis[1],axis[0]])*side
    if n@out<0:n=-n
    q=obs['points'][obs['valid']];return (q-np.asarray(line['point']))@n
def main():
    cfg=read(CONFIG);base=ROOT/cfg['baseline'];bcfg=read(ROOT/cfg['baseline_config']);out=ROOT/cfg['output'];assert not out.exists();out.mkdir();(out/'focus').mkdir();sources={p.relative_to(ROOT).as_posix():sha256(p) for p in [CONFIG,ROOT/cfg['baseline_config'],Path(__file__),ROOT/'tools/fiber_rgb_side_observation.py',ROOT/'tools/fiber_tapered_tangent_cap.py',ROOT/'tools/fiber_straight_semicircle.py',ROOT/'docs/plans/tapered_rgb_diagnosis_20261104.md']};atomic_json(out/'preflight.json',dict(config=cfg,sources=sources,baseline_seal_sha256=sha256(base/'seal.json'),pixel_updates=0,default_changed=False))
    ranking={};selected={}
    for vid,total in [('video_1',722),('video_2',1972)]:
        ranked=[]
        for i in range(total):
            with np.load(base/f'full_video/{vid}/layers/{i:06d}.npz') as z:body=z['subject']>0;blue=z['blue_identity']>0;final=z['final_fiber']>0
            denom=int((body&blue).sum());lost=int((body&blue&~final).sum());ranked.append(dict(video=vid,frame=i+1,subject_blue=denom,subject_blue_deleted=lost,subject_blue_loss_rate=lost/max(denom,1)))
        ranked.sort(key=lambda x:(-x['subject_blue_loss_rate'],-x['subject_blue_deleted'],x['frame']));ranking[vid]=ranked;selected[vid]={x['frame'] for x in [z for z in ranked if z['subject_blue']>=cfg['minimum_subject_blue_pixels_for_ranking']][:cfg['top_per_video']]}
        if vid=='video_2':selected[vid].update(cfg['focus_video2'])
    atomic_json(out/'ranking.json',ranking);media=read(ROOT/bcfg['source_identity']/'source_media_timing.json');spec=importlib.util.spec_from_file_location('timing',ROOT/'runs/engineering/timing_bisector_media_20261103.py');tm=importlib.util.module_from_spec(spec);spec.loader.exec_module(tm);summary={};exports={}
    for vid,total in [('video_1',722),('video_2',1972)]:
        dest=out/vid;dest.mkdir();(dest/'layers').mkdir();data=rows(base/f'full_video/{vid}/frames.jsonl');rankmap={z['frame']:z for z in ranking[vid]};cap=cv2.VideoCapture(str(ROOT/f'data/videos/{vid}.mp4'));writer=checked_writer(out/f'{vid}_diagnosis_draft.mp4',30,(1600,1368));per=[];start=time.perf_counter()
        for i,r in enumerate(data):
            ok,frame=cap.read();assert ok and array_hash(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))==r['source_rgb_sha256'];raw=load_mask(ROOT/cfg['raw_masks']/f'{vid}/masks/{i:06d}.png',(400,400),{0,1,2});assert array_hash(raw)==r['raw_sha256'];old=load_mask(ROOT/bcfg['old_result']/f'full_video/{vid}/masks/{i:06d}.png',(400,400),{0,1,2});new=load_mask(base/f'full_video/{vid}/masks/{i:06d}.png',(400,400),{0,1,2})
            with np.load(base/f'full_video/{vid}/layers/{i:06d}.npz') as a:body=a['subject']>0;blue=a['blue_identity']>0;domain=a['search_domain']>0
            info=r['info'];g=info.get('geometry');tip=info.get('tip');meta=dict(rankmap[i+1],applied=info['applied'],baseline_reason=info['reason']);empty=dict(points=np.empty((0,2)),valid=np.zeros(0,bool));L=R=empty;H=[];held=[];broad=np.zeros(raw.shape,bool);eligible=np.zeros_like(raw);layers=dict(deleted_subject_blue=(body&blue&(new!=2)).astype(np.uint8))
            if g:
                seedrow=next(z for z in info['seeds'] if z['reason']=='qualified');seed=dict(mu=np.asarray(seedrow['mu']),bg=np.asarray(seedrow['background_chroma']),gap=seedrow['relative_blue_advantage']);u=np.asarray(g['axis']);o=np.asarray(g['origin']);n=np.asarray(g['normal']);scale=1.
                obs=[];dist=[]
                for j,(side,line) in enumerate(zip([-1,1],g['independent_lines'])):
                    h=huber_diagnostics(line,bcfg['fit']['huber_delta_at400']);H.append(h);a=edge_observations(frame,raw,line['points'],u,side,seed,bcfg['tip'],cfg['edge'],scale);obs.append(a);d=signed_edge_distance(a,line,u,side);dist.extend(d.tolist());meta['left' if j==0 else 'right']=dict(valid_points=int(a['valid'].sum()),points=len(a['valid']),outward_shift_median=float(np.median(a['outward_offset'][a['valid']])) if a['valid'].any() else None,RGB_outside_line_median=float(np.median(d)) if len(d) else None,RGB_outside_line_fraction=float(np.mean(d>1)) if len(d) else None,Huber_fully_weighted_fraction=h['fully_weighted_fraction'])
                    for key in ['raw_points','points','valid','outward_offset','full_scores','full_valid']:layers[f'{j}_{key}']=a[key]
                    layers[f'{j}_Huber_weights']=h['weights'];layers[f'{j}_Huber_signed_residual']=h['signed_residual']
                L,R=obs;heldcfg=copy.deepcopy(bcfg['fit']);heldcfg['axial_fraction']=[.8,.95];lp,rp,_=observed_sides(body,o,u,heldcfg,1.)
                for side,points,line in zip([-1,1],[lp,rp],g['independent_lines']):
                    if len(points):a=edge_observations(frame,raw,points,u,side,seed,bcfg['tip'],cfg['edge'],1.);held.append(a)
                heldd=np.concatenate([signed_edge_distance(a,line,u,side) for a,line,side in zip(held,g['independent_lines'],[-1,1])]) if len(held)==2 else np.array([])
                btip=dict(bcfg['tip'],independent_side_margin_at400=cfg['independent_side_margin_at400']);broad=broad_search_domain(raw.shape,body,g,btip,1.);eligible,extra=terminal_observations(frame,raw,g,seed,bcfg['tip'],broad,blue);outside=(eligible>0)&~domain
                meta.update(radius=info['cap']['radius'] if info.get('cap') else 0.,V=g['intersection'],O=g['origin'],axis=g['axis'],T=tip.get('vertex') if tip else None,geometry_to_RGB_side_median=float(np.median(dist)) if dist else None,heldout_RGB_outside_line_median=float(np.median(heldd)) if len(heldd) else None,independent_terminal=extra,extra_terminal_pixels=int(outside.sum()),blue_outside_old_search=int((blue&broad&~domain).sum()),independent_tip_delta=float(extra['tip_t']-tip['tip_t']) if extra.get('vertex') is not None and tip and tip.get('vertex') else None)
                if tip and tip.get('vertex') is not None:meta['V_T_distance']=float((np.asarray(g['intersection'])-np.asarray(tip['vertex']))@u)
                # Pixel coverage/width proxies at fixed distances behind T.
                widths={}
                if tip and tip.get('vertex') is not None:
                    for back in [5,10,20]:
                        xy=np.asarray(tip['vertex'])-back*u+np.arange(-80,80.001,.5)[:,None]*n
                        ab,good=nearest((body&blue).astype(np.uint8),xy);nb,ng=nearest((new==2).astype(np.uint8),xy);widths[str(back)]=dict(old_subject_blue=float((ab[good]>0).sum()*.5),current_geometry=float((nb[ng]>0).sum()*.5))
                meta['tip_local_widths_proxy']=widths
            layers.update(old_search=domain.astype(np.uint8),independent_search=broad.astype(np.uint8),independent_eligible=eligible)
            np.savez_compressed(dest/f'layers/{i:06d}.npz',**layers);meta=public(meta);per.append(meta);im=board(frame,old,new,body,blue,domain,broad,eligible,g,tip,L,R,H,held,meta,vid,i,media[vid]['pts_seconds'][i]);writer.write(im)
            if i+1 in selected[vid]:cv2.imwrite(str(out/f'focus/{vid}_frame{i+1:04d}.png'),im);atomic_json(out/f'focus/{vid}_frame{i+1:04d}.json',public(dict(meta=meta,geometry=g,Huber=H,RGB_observations=obs if g else [],heldout=held)))
            if (i+1)%500==0:print('DIAG',vid,i+1,flush=True)
        cap.release();writer.release();assert len(per)==total;atomic_json(dest/'frames.json',per);exports[vid]=tm.retime(out/f'{vid}_diagnosis_draft.mp4',out/f'{vid}_diagnosis.mp4',media[vid]);keys=['subject_blue_loss_rate','geometry_to_RGB_side_median','heldout_RGB_outside_line_median','V_T_distance','radius','independent_tip_delta'];summary[vid]=dict(frames=total,selected_frames=sorted(selected[vid]),top_subject_loss=[x for x in ranking[vid] if x['subject_blue']>=100][:10],measurements={k:stats([x[k] for x in per if x.get(k) is not None]) for k in keys},extra_search_later_tip_frames=sum(x.get('independent_tip_delta',0) is not None and x.get('independent_tip_delta',0)>1 for x in per),pixel_updates=0,wall_seconds=time.perf_counter()-start)
    assert all(sha256(ROOT/k)==v for k,v in sources.items());assert sha256(base/'seal.json')==read(out/'preflight.json')['baseline_seal_sha256'];atomic_json(out/'summary.json',summary);atomic_json(out/'media_identity.json',exports);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
