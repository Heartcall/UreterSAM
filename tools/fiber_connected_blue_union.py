"""Current-RGB4CC extraction before optional contour trimming; shadow only."""
from __future__ import annotations
import math,time
import cv2,numpy as np
from .fiber_aggressive_blue import AggressiveVariant,GrowthPipeline as PriorPipeline,tip_visibility
from .fiber_blue_extent import GrowthPipeline as ExtentPipeline
from .fiber_connected_blue_growth import chroma
from .fiber_fast_observation import sample
from .fiber_local_dark_evidence import public,nearest
from .fiber_reference_workspace import ROOT,read

ARMS=('raw2','aggressive','connected_blue_union','connected_blue_regularized')
LAYERS=('rgb_eligible','blue_body','regularized_body')

def rooted(binary,core,minimum):
    _,labels=cv2.connectedComponents(binary.astype(np.uint8),connectivity=4)
    ids,counts=np.unique(labels[core],return_counts=True);ids=ids[(ids>0)&(counts>=minimum)]
    return ((labels==ids[0]).astype(np.uint8) if len(ids)==1 else np.zeros_like(binary,np.uint8)),int(len(ids))

def trim(body,core,settings,scale):
    smooth=cv2.GaussianBlur(body.astype(np.float32),(0,0),settings['gaussian_sigma_at400']*scale,borderType=cv2.BORDER_REPLICATE)
    result=(body>0)&(smooth>=settings['threshold']);result|=(body>0)&core
    return rooted(result,core,1)[0]

class ConnectedBlue:
    def __init__(self,config,regularized):
        self.config=config;self.settings=config['connected_blue'];self.regularized=regularized
        self.reference=read(ROOT/config['reference_aggressive_config']);self.observer=AggressiveVariant(self.reference,'aggressive');self.layers={}

    def end(self,frame,mask,body,seed,domain,blocked):
        cfg=self.settings['tip'];s=min(mask.shape)/400;tip=dict(state='unknown',reason=blocked or 'insufficient_current_cap',xy=None,end_sign=1,direction='distal')
        if blocked or not body.any():return tip
        yy,xx=np.nonzero(body);points=np.stack([xx,yy],-1).astype(float);u=seed['u'];n=np.array([-u[1],u[0]]);t=points@u;maximum=t.max();cap=points[t>=maximum-cfg['frontier_band_at400']*s];front=np.median(cap,axis=0)
        tip.update(frontier_xy=front,frontier_pixels=len(cap))
        if min(front[0],front[1],mask.shape[1]-1-front[0],mask.shape[0]-1-front[1])<cfg['tip_near_component_at400']*s:tip.update(state='out_of_view',reason='current_cap_near_frame_edge');return tip
        future=front+np.array(cfg['future_offsets_at400'])[:,None]*s*u;frgb,fvalid=sample(frame,future);fv,dvalid=sample(domain.astype(np.float32),future);labels,lv=nearest(mask,future)
        if not fvalid.all():tip.update(state='out_of_view',reason='terminal_background_out_of_view');return tip
        if not dvalid.all() or np.any(fv<.5):tip['reason']='forward_scope_limit';return tip
        if np.any(lv&(labels==1)):tip.update(state='occlusion_suspected',reason='current_stone_contact');return tip
        if np.any(lv&(labels==2)) or self.observer.color(frgb,seed).any():tip.update(state='internal_transition',reason='current_body_color_continues');return tip
        _,noise=chroma(frgb)
        if np.any(noise>self.observer.cfg['color']['max_noise']):tip['reason']='terminal_background_unknown';return tip
        tail=points[t>=maximum-cfg['local_taper_tail_at400']*s];lo,hi=cfg['local_taper_body_range_at400'];shaft=points[(t<maximum-lo*s)&(t>=maximum-hi*s)]
        if min(len(tail),len(shaft))<2:return tip
        tw=float(np.ptp(tail@n)+1);bw=float(np.ptp(shaft@n)+1);tip.update(current_tail_width=tw,current_local_body_width=bw)
        if tw/bw>cfg['maximum_tail_body_width_ratio']:tip['reason']='no_current_local_cap_closure';return tip
        offsets=np.linspace(*np.array(cfg['axial_range_at400'])*s,cfg['profile_samples']);pts=front+np.array(cfg['profile_offsets_at400'])[:,None,None]*s*n+offsets[None,:,None]*u
        a,av=sample(frame,pts-cfg['edge_delta_at400']*s*u);b,bv=sample(frame,pts+cfg['edge_delta_at400']*s*u)
        contrast=np.where(av&bv,np.linalg.norm(a-b,axis=-1)/(255*math.sqrt(3)),-1);ix=contrast.argmax(1);strength=contrast[np.arange(3),ix];supported=strength>=cfg['minimum_RGB_contrast'];tip.update(boundary_profile_contrasts=strength,boundary_profile_offsets_px=offsets[ix])
        if not supported[1] or supported.sum()<cfg['minimum_profiles']:tip['reason']='weak_terminal_RGB_boundary';return tip
        spread=float(np.ptp(offsets[ix[supported]]))
        if spread>cfg['max_profile_spread_at400']*s:tip['reason']='incoherent_end_profiles';return tip
        tip.update(state='visible_end_candidate',reason='current_connected_cap_and_background',xy=front,uncertainty_radius_px=max(cfg['tip_near_component_at400']*s,spread));return tip

    def grow(self,frame,mask,probability,prior=None,age=99):
        start=time.perf_counter();s=min(mask.shape)/400;zero=np.zeros_like(mask,np.uint8);self.layers={k:zero.copy() for k in LAYERS}
        self.observer.chosen=0;self.observer.orientation='unavailable';seeds,records=self.observer.seeds(frame,mask,probability,prior,age);seedend=time.perf_counter()
        m=dict(reason='no_trusted_seed',seeds=records,branches=[],tips=[],apply_to_mask=False,tip_metrics=None,abnormal=False,proposal_kind='RGB_rooted_component',direction_control=dict(selected_sign=1 if len(seeds)==1 else 0,walk_calls=[],orientation_source='fixed_edge_to_center'))
        def done(c,p,conf):
            m['timings']=dict(seed_ms=(seedend-start)*1000,connected_contour_tip_ms=(time.perf_counter()-seedend)*1000,extra_total_ms=(time.perf_counter()-start)*1000)
            return c,p,conf,public(m)
        if len(seeds)!=1:
            if len(seeds)>1:m['reason']='ambiguous_trusted_seeds'
            return done(zero,zero.copy(),zero.copy())
        seed=seeds[0];q=next(z for z in records if z['reason']=='qualified');anchor=np.asarray(q['fixed_entry_anchor']);u=seed['u'];n=np.array([-u[1],u[0]]);yy,xx=np.indices(mask.shape);dx=xx-anchor[0];dy=yy-anchor[1];t=dx*u[0]+dy*u[1];normal=dx*n[0]+dy*n[1]
        sy,sx=np.nonzero(seed['core']);minimum=float(np.min((sx-anchor[0])*u[0]+(sy-anchor[1])*u[1]))
        domain=(t>=minimum-self.settings['proximal_margin_at400']*s)&(np.abs(normal)<=np.maximum(t,0)*math.tan(math.radians(self.settings['forward_cone_half_angle_degrees']))+self.settings['cone_margin_at400']*s)
        eligible=zero.copy();region=domain|seed['core'];eligible[region]=self.observer.color(frame[region],seed).astype(np.uint8);self.layers['rgb_eligible']=eligible.copy()
        minroot=self.observer.cfg['seed']['min_pixels_at400']*s*s;unblocked,_=rooted(eligible,seed['core'],minroot);conflict=((unblocked>0)&(mask==1)).astype(np.uint8)
        eligible[mask==1]=0;body,roots=rooted(eligible,seed['core'],minroot);self.layers['blue_body']=body.copy();processed=trim(body,seed['core'],self.settings['regularization'],s) if self.regularized else body.copy();self.layers['regularized_body']=processed.copy()
        proposal=((body>0)&domain&(mask==0)).astype(np.uint8);candidate=((processed>0)&domain&(mask==0)).astype(np.uint8);blocked=None
        if roots!=1:blocked='color_root_absent_or_ambiguous'
        elif body.sum()>self.settings['component_area_fraction_limit']*mask.size:blocked='connected_component_area_budget'
        elif proposal.sum()>self.settings['new_area_fraction_limit']*mask.size:blocked='new_area_budget'
        if blocked:candidate[:]=0
        m.update(reason=blocked or ('candidate_accepted' if candidate.any() else 'no_continuous_new_candidate'),abnormal=blocked in ['connected_component_area_budget','new_area_budget'],root_components=roots,seed_width=seed['width'],seed_length=seed['length'],seed_origin=seed['origin'],seed_axis=u,distal_sign=1,orientation_source='fixed_edge_to_center',eligible_pixels=int(eligible.sum()),evaluated_pixels=int(region.sum()),blue_component_pixels=int(body.sum()),processed_component_pixels=int(processed.sum()),contour_trimmed_pixels=int(((body>0)&(processed==0)).sum()),contour_trimmed_new_pixels=int((proposal>candidate).sum()),stone_conflict_pixels=int(conflict.sum()),forward_scope_pixels=int(domain.sum()),rgb_support_policy='direct_current_color_component_no_width_or_curvature_veto')
        m['tips']=[self.end(frame,mask,processed,seed,domain,blocked)];return done(candidate,proposal,conflict)

class GrowthPipeline:
    def __init__(self,cfg,arm,t1_config=None,sequential=False):
        if arm not in ARMS:raise ValueError('Unknown arm')
        self.cfg=cfg;self.arm=arm;self.new=arm.startswith('connected_blue_');reference=read(ROOT/cfg['reference_aggressive_config'])
        if self.new:
            self.inner=ExtentPipeline(read(ROOT/cfg['reference_extent_config']),'blue_extent',t1_config,sequential);self.inner.growth=ConnectedBlue(cfg,arm=='connected_blue_regularized')
        else:self.inner=PriorPipeline(reference,arm,t1_config,sequential)
        self.tracker=self.inner.tracker;self.layers={}
    def reset(self):self.inner.reset();self.layers={}
    def run(self,frame,mask,probability,*args,**kwargs):
        c,p,conf,m=self.inner.run(frame,mask,probability,*args,**kwargs)
        if self.new:
            start=time.perf_counter();m['tip_visibility_audit']=tip_visibility(mask,probability,c,m,self.cfg['aggressive_parameters']['tip_visibility_tolerance_at400']*min(mask.shape)/400);ms=(time.perf_counter()-start)*1000;m['timings'].update(tip_visibility_audit_ms=ms,extra_total_ms=m['timings']['extra_total_ms']+ms);self.layers=self.inner.growth.layers
        else:self.layers={k:np.zeros_like(mask,np.uint8) for k in LAYERS}
        return c,p,conf,m
