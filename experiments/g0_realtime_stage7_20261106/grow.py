"""Copied original grow; only region color acquisition is instance-bound."""
import math,time
import numpy as np
from tools.fiber_connected_blue_union import rooted,trim,LAYERS
from tools.fiber_local_dark_evidence import public

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
    eligible=zero.copy();region=domain|seed['core'];eligible[region]=region_color(frame,region,seed).astype(np.uint8);self.layers['rgb_eligible']=eligible.copy()
    minroot=self.observer.cfg['seed']['min_pixels_at400']*s*s;unblocked,_=rooted(eligible,seed['core'],minroot);conflict=((unblocked>0)&(mask==1)).astype(np.uint8)
    eligible[mask==1]=0;body,roots=rooted(eligible,seed['core'],minroot);self.layers['blue_body']=body.copy();processed=trim(body,seed['core'],self.settings['regularization'],s) if self.regularized else body.copy();self.layers['regularized_body']=processed.copy()
    proposal=((body>0)&domain&(mask==0)).astype(np.uint8);candidate=((processed>0)&domain&(mask==0)).astype(np.uint8);blocked=None
    if roots!=1:blocked='color_root_absent_or_ambiguous'
    elif body.sum()>self.settings['component_area_fraction_limit']*mask.size:blocked='connected_component_area_budget'
    elif proposal.sum()>self.settings['new_area_fraction_limit']*mask.size:blocked='new_area_budget'
    if blocked:candidate[:]=0
    m.update(reason=blocked or ('candidate_accepted' if candidate.any() else 'no_continuous_new_candidate'),abnormal=blocked in ['connected_component_area_budget','new_area_budget'],root_components=roots,seed_width=seed['width'],seed_length=seed['length'],seed_origin=seed['origin'],seed_axis=u,distal_sign=1,orientation_source='fixed_edge_to_center',eligible_pixels=int(eligible.sum()),evaluated_pixels=int(region.sum()),blue_component_pixels=int(body.sum()),processed_component_pixels=int(processed.sum()),contour_trimmed_pixels=int(((body>0)&(processed==0)).sum()),contour_trimmed_new_pixels=int((proposal>candidate).sum()),stone_conflict_pixels=int(conflict.sum()),forward_scope_pixels=int(domain.sum()),rgb_support_policy='direct_current_color_component_no_width_or_curvature_veto')
    m['tips']=[self.end(frame,mask,processed,seed,domain,blocked)];return done(candidate,proposal,conflict)
