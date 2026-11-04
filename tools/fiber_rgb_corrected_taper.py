"""RGB observation controls on frozen upstream geometry, no weak-pixel repair.

R: RGB external-edge points replace S2 extrema; identical fit band and Huber.
S: transverse terminal search envelope no longer depends on the line crossing.
The exact ORIGINAL tangent-circle rasterizer remains unchanged for every arm.
Inputs include the frozen upstream identity/geometry and its actual fallback.
"""
import copy,time,math
import cv2,numpy as np
from .fiber_reference_workspace import ROOT,read
from .fiber_local_dark_evidence import public
from .fiber_straight_semicircle import unit,robust_line,blue_identity
from .fiber_tapered_tangent_cap import locate_tip,raster_geometry
from .fiber_rgb_side_observation import edge_observations,broad_search_domain,terminal_observations
ARMS=('RGB_sides','independent_search','RGB_sides_independent_search')

def geometry_from_lines(body,lines,cfg,scale=1.):
    a,b=lines;u=unit(np.asarray(a['direction'])+np.asarray(b['direction']));n=np.array([-u[1],u[0]]);aa=np.asarray(a['direction']);bb=np.asarray(b['direction']);ap=np.asarray(a['point']);bp=np.asarray(b['point'])
    if abs(np.linalg.det(np.c_[aa,-bb]))<cfg['numerical_minimum_line_determinant']:return None,['parallel_or_ill_conditioned_line_intersection']
    coefficients=np.linalg.solve(np.c_[aa,-bb],bp-ap);V=ap+coefficients[0]*aa;y,x=np.nonzero(body);points=np.c_[x,y];o0=points.mean(0);o=V+float((o0-V)@u)*u;dist=[]
    for p,v in [(ap,aa),(bp,bb)]:
        nn=np.array([-v[1],v[0]]);dist.append(float((p-o)@nn/(n@nn)))
    width=dist[1]-dist[0]
    if width<=0 or (V-o)@u<=0:return None,['sides_diverge_in_distal_direction']
    lts=(np.asarray(a['points'])-o)@u;rts=(np.asarray(b['points'])-o)@u;lo=max(lts.min(),rts.min());hi=min(lts.max(),rts.max());ts=(points-o)@u;angle=float(np.arccos(np.clip(aa@bb,-1,1)));errors=[]
    if hi-lo<cfg['minimum_span_at400']*scale:errors.append('insufficient_fitted_span')
    if width<cfg['minimum_width_at400']*scale or width>cfg['maximum_width_frame_fraction']*min(body.shape):errors.append('width_out_of_range')
    if np.ptp(ts)/width<cfg['minimum_observed_aspect']:errors.append('subject_not_slender')
    if any(q['median_error']>max(cfg['maximum_median_line_error_at400']*scale,cfg['maximum_median_line_error_width_fraction']*width) for q in lines):errors.append('nonstraight_side_residual')
    return dict(origin=o,axis=u,normal=n,width=float(width),intersection=V,independent_lines=lines,side_angle_degrees=float(np.degrees(angle)),half_angle_radians=angle/2,fit_span=[float(lo),float(hi)],body_extent=[float(ts.min()),float(ts.max())]),errors

class RGBCorrectedTaper:
    def __init__(self,cfg,arm):
        if arm not in ARMS:raise ValueError(arm)
        self.cfg=copy.deepcopy(cfg);self.base=read(ROOT/cfg['baseline_config']);self.arm=arm;self.reset()
    def reset(self):self.final_mask=None;self.info={};self.layers={}
    def run(self,frame,raw,old,baseline,base_info,subject):
        start=time.perf_counter();self.final_mask=baseline.copy();zero=np.zeros_like(raw,np.uint8);self.layers={k:zero.copy() for k in ['geometric_proposal','rod','cap','search_domain','tip_eligible','blue_identity','final_fiber','edit_add','edit_remove','deleted_subject_blue']};self.info=dict(arm=self.arm,applied=False,status='fallback_to_frozen_G0',reason='upstream_no_geometry',geometry=None,tip=None,cap=None,rejections=[],RGB_edges=[],upstream_reason=base_info['reason'],physical_tip_accuracy=None)
        def done(reason=None):
            if reason:self.info['reason']=reason
            self.layers['final_fiber']=(self.final_mask==2).astype(np.uint8);self.layers['edit_add']=((self.final_mask==2)&(baseline!=2)).astype(np.uint8);self.layers['edit_remove']=((self.final_mask!=2)&(baseline==2)).astype(np.uint8);blue=self.layers['blue_identity']>0;self.layers['deleted_subject_blue']=(subject&blue&(self.final_mask!=2)).astype(np.uint8)
            actual=self.info if self.info['applied'] else base_info if base_info['applied'] else {}
            self.info.update(actual_geometry=actual.get('geometry'),actual_tip=actual.get('tip'),actual_cap=actual.get('cap'))
            denominator=int((subject&blue).sum());lost=int(self.layers['deleted_subject_blue'].sum());self.info.update(subject_blue=denominator,subject_blue_deleted=lost,subject_blue_loss_rate=lost/max(denominator,1),added_pixels=int(self.layers['edit_add'].sum()),removed_pixels=int(self.layers['edit_remove'].sum()),runtime_ms=(time.perf_counter()-start)*1000)
            assert np.array_equal(self.final_mask==1,raw==1);self.info=public(self.info);return self.final_mask
        upstream=base_info.get('geometry')
        if not upstream:return done()
        selected=[z for z in base_info['seeds'] if z['reason']=='qualified']
        if len(selected)!=1:return done('upstream_identity_not_unique')
        row=selected[0];seed=dict(mu=np.asarray(row['mu']),bg=np.asarray(row['background_chroma']),gap=row['relative_blue_advantage']);blue,_,_=blue_identity(frame,seed,self.base['tip']);self.layers['blue_identity']=blue.astype(np.uint8)
        lines=[]
        for side,line in zip([-1,1],upstream['independent_lines']):
            if self.arm=='independent_search':lines.append({k:np.asarray(v) if k in ['point','direction','points'] else v for k,v in line.items()});continue
            a=edge_observations(frame,raw,line['points'],np.asarray(upstream['axis']),side,seed,self.base['tip'],self.cfg['edge'],min(raw.shape)/400)
            ok=a['valid'];p=a['points'][ok];axis=np.asarray(upstream['axis']);old_span=float(np.ptp(np.asarray(line['points'])@axis));span=float(np.ptp(p@axis)) if len(p) else 0
            record={k:v for k,v in a.items() if not k.startswith('full_')};record.update(accepted_points=int(ok.sum()),coverage=span/max(old_span,1e-9));self.info['RGB_edges'].append(record)
            if len(p)<self.cfg['minimum_RGB_edge_points'] or span<self.cfg['minimum_RGB_span_fraction']*old_span:return done('insufficient_RGB_external_side_observations')
            lines.append(robust_line(p,axis,self.base['fit']['huber_delta_at400']*min(raw.shape)/400,self.base['fit']['IRLS_iterations']))
        g,errors=geometry_from_lines(subject,lines,self.base['fit'],min(raw.shape)/400);self.info['geometry']=g;self.info['rejections']=errors
        if g is None:return done(errors[0])
        if self.arm=='RGB_sides':_,domain,eligible,tip=locate_tip(frame,raw,g,seed,self.base['tip'],min(raw.shape)/400)
        else:
            tc=dict(self.base['tip'],independent_side_margin_at400=self.cfg['independent_side_margin_at400']);domain=broad_search_domain(raw.shape,subject,g,tc,min(raw.shape)/400);eligible,tip=terminal_observations(frame,raw,g,seed,self.base['tip'],domain,blue)
        self.layers.update(search_domain=domain.astype(np.uint8),tip_eligible=eligible);self.info['tip']=tip
        if tip['vertex'] is None:errors.append(tip['reason']);return done(errors[0])
        G,rod,cap,ci,fail=raster_geometry(raw.shape,g,tip['vertex'],self.base['geometry']['arc_samples'])
        if fail:errors.append(fail);return done(errors[0])
        self.layers.update(geometric_proposal=G,rod=rod,cap=cap);self.info['cap']=ci
        if tip['reason']=='tip_reaches_search_limit':errors.append(tip['reason'])
        conflict=int(((G>0)&(raw==1)).sum())
        if conflict:errors.append('geometric_stone_conflict')
        if G.sum()>self.base['geometry']['maximum_frame_area_fraction']*G.size:errors.append('geometric_area_budget')
        self.info.update(rejections=errors,proposal_stone_conflicts=conflict)
        if errors:return done(errors[0])
        self.final_mask[baseline==2]=0;self.final_mask[G>0]=2;self.info.update(applied=True,status='applied',reason='RGB_geometry_control_applied');return done()
