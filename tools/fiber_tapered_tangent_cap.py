"""Actual converging straight sides and a circle tangent to BOTH original lines.

The visible apex is an RGB observation projected onto the angle bisector. The
line intersection only determines circle geometry; it is NEVER the visible tip.
No parallel canonical sides, splines, weak-pixel growth or union intersection.
"""
from __future__ import annotations
import copy, math, time
import cv2
import numpy as np
from .fiber_reference_workspace import ROOT, read
from .fiber_weak_stage1 import QualifiedSeedObserver
from .fiber_straight_semicircle import unit, robust_line, observed_sides, blue_identity
from .fiber_fast_observation import sample
from .fiber_local_dark_evidence import public, nearest


def fit_geometry(body, initial_u, cfg, scale):
    y,x=np.nonzero(body); origin=np.c_[x,y].mean(0)
    left,right,_=observed_sides(body,origin,initial_u,cfg,scale)
    if len(left)<cfg['minimum_pairs']:return None,['insufficient_visible_side_pairs']
    lines=[robust_line(a,initial_u,cfg['huber_delta_at400']*scale,cfg['IRLS_iterations']) for a in [left,right]]
    a,b=lines; u=unit(a['direction']+b['direction']); n=np.array([-u[1],u[0]])
    cross=float(np.linalg.det(np.c_[a['direction'],-b['direction']]))
    if abs(cross)<cfg['numerical_minimum_line_determinant']:return None,['parallel_or_ill_conditioned_line_intersection']
    coefficients=np.linalg.solve(np.c_[a['direction'],-b['direction']],b['point']-a['point'])
    V=a['point']+coefficients[0]*a['direction']
    # Exact angle bisector must pass through the ORIGINAL line intersection.
    origin=V+float((origin-V)@u)*u
    offsets=[]
    for line in lines:
        normal=np.array([-line['direction'][1],line['direction'][0]])
        offsets.append(float((line['point']-origin)@normal/(n@normal)))
    width=offsets[1]-offsets[0]
    if width<=0 or float((V-origin)@u)<=0:return None,['sides_diverge_in_distal_direction']
    angle=float(np.arccos(np.clip(a['direction']@b['direction'],-1,1))); half=angle/2
    points=np.c_[x,y]; ts=(points-origin)@u
    lts=(left-origin)@u;rts=(right-origin)@u
    lo=max(lts.min(),rts.min());hi=min(lts.max(),rts.max())
    widths=(right-left)@n
    cv=1.4826*float(np.median(abs(widths-np.median(widths))))/width
    drift=float(np.quantile(widths,.9)-np.quantile(widths,.1))/width
    errors=[]
    if hi-lo<cfg['minimum_span_at400']*scale:errors.append('insufficient_fitted_span')
    if width<cfg['minimum_width_at400']*scale or width>cfg['maximum_width_frame_fraction']*min(body.shape):errors.append('width_out_of_range')
    if np.ptp(ts)/width<cfg['minimum_observed_aspect']:errors.append('subject_not_slender')
    if any(q['median_error']>max(cfg['maximum_median_line_error_at400']*scale,cfg['maximum_median_line_error_width_fraction']*width) for q in lines):errors.append('nonstraight_side_residual')
    return dict(origin=origin,axis=u,normal=n,width=float(width),intersection=V,
                independent_lines=lines,side_angle_degrees=float(np.degrees(angle)),half_angle_radians=half,
                width_robust_CV=cv,width_p90_p10_fraction=drift,width_variation_is_expected_taper=True,
                observed_widths=widths,fit_span=[float(lo),float(hi)],body_extent=[float(ts.min()),float(ts.max())]),errors


def locate_tip(frame,raw,g,seed,cfg,scale):
    o,u,n,V=[g[k] for k in ['origin','axis','normal','intersection']]
    blue,color,noise=blue_identity(frame,seed,cfg)
    yy,xx=np.indices(raw.shape);t=(xx-o[0])*u[0]+(yy-o[1])*u[1];v=(xx-o[0])*n[0]+(yy-o[1])*n[1]
    lo,old_end=g['body_extent'];begin=lo+cfg['minimum_search_axial_fraction']*(old_end-lo);limit=old_end+cfg['maximum_forward_at400']*scale
    # Follow actual taper. A small RGB-only search margin also permits detecting
    # evidence at/beyond V; such evidence is not silently truncated to fit a cap.
    halfwidth=np.maximum(0.,(float((V-o)@u)-t)*math.tan(g['half_angle_radians']))
    domain=(t>=begin)&(t<=limit)&(abs(v)<=halfwidth+cfg['search_side_margin_at400']*scale)
    patch=cv2.filter2D(blue.astype(np.float32),-1,np.ones((3,3),np.float32),borderType=cv2.BORDER_CONSTANT)
    candidate=domain&blue&(patch>=cfg['minimum_local_blue_pixels_3x3'])&(raw!=1)
    eligible=np.zeros_like(raw,np.uint8);y,x=np.nonzero(candidate)
    meta=dict(reason='no_blue_external_terminal',source_pixel=None,vertex=None,qualified_pixels=0,search_limit_t=float(limit),pixel_candidates=int(len(x)),network_probability_gate=False)
    if not len(x):return blue,domain,eligible,meta
    points=np.c_[x,y].astype(float);future=points[:,None,:]+np.asarray(cfg['forward_offsets_at400'])[None,:,None]*scale*u
    bg,valid=sample(frame,future);labels,lv=nearest(raw,future);forward_blue,bc,bnoise=blue_identity(bg,seed,cfg)
    med=np.median(bg,axis=1);contrast=np.linalg.norm(frame[y,x].astype(float)-med,axis=1)/(255*math.sqrt(3));chromatic=np.linalg.norm(color[y,x]-np.median(bc,axis=1),axis=1)
    margin=np.minimum.reduce([points[:,0],points[:,1],raw.shape[1]-1-points[:,0],raw.shape[0]-1-points[:,1]])
    photometric=(contrast>=cfg['minimum_RGB_contrast'])|(chromatic>=cfg['minimum_chroma_contrast'])
    good=valid.all(1)&lv.all(1)&~np.any(labels==1,axis=1)&(bnoise<cfg['maximum_chroma_noise']).all(1)&(forward_blue.mean(1)<=cfg['maximum_forward_blue_fraction'])&photometric&(margin>=cfg['minimum_frame_margin_at400']*scale)
    eligible[y[good],x[good]]=1;ids=np.flatnonzero(good)
    meta.update(qualified_pixels=len(ids),rejected_no_contrast=int((~photometric).sum()),rejected_forward_blue=int((forward_blue.mean(1)>cfg['maximum_forward_blue_fraction']).sum()),rejected_forward_stone=int(np.any(labels==1,axis=1).sum()),rejected_unknown_or_outside=int((~valid.all(1)|(bnoise>=cfg['maximum_chroma_noise']).any(1)).sum()))
    if not len(ids):return blue,domain,eligible,meta
    axial=(points-o)@u;order=np.lexsort((abs((points[ids]-o)@n),-axial[ids]));chosen=int(ids[order[0]]);end=float(axial[chosen]);T=o+end*u
    meta.update(source_pixel=points[chosen],vertex=T,tip_t=end,source_normal_distance=float((points[chosen]-o)@n),RGB_contrast=float(contrast[chosen]),chroma_contrast=float(chromatic[chosen]),forward_blue_fraction=float(forward_blue[chosen].mean()),reason='farthest_current_blue_external_contrast')
    if end>=limit-cfg['search_limit_margin_at400']*scale:meta['reason']='tip_reaches_search_limit'
    return blue,domain,eligible,meta


def raster_geometry(shape,g,T,samples=129):
    """Closed tapered body + minor circle arc through T, exact original tangency.

    If d=projection(V-T,u) and s=sin(half-angle), r=d*s/(1-s).
    C=T-r*u. Arc span is pi-2*half-angle, not a prescribed semicircle.
    """
    u,n,V,o=[np.asarray(g[k],float) for k in ['axis','normal','intersection','origin']];T=np.asarray(T,float)
    d=float((V-T)@u);s=math.sin(g['half_angle_radians'])
    if d<=1e-6 or not 0<s<1:return None,None,None,None,'tip_at_or_beyond_side_intersection'
    r=d*s/(1-s);C=T-r*u;joins=[]
    yy,xx=np.indices(shape);points=np.stack([xx,yy],axis=-1).astype(float);inside=np.ones(shape,bool)
    for line in g['independent_lines']:
        p=np.asarray(line['point']);v=np.asarray(line['direction']);normal=np.array([-v[1],v[0]])
        joins.append(C+float((p-C)@normal)*normal)
        sign=1. if float((o-p)@normal)>0 else -1.
        inside&=((points-p)@normal)*sign>=-1e-9
    join_t=float(np.mean([(q-C)@u for q in joins]));along=(points-C)@u
    rod=inside&(along<=join_t+1e-9)
    cap=(along>=join_t-1e-9)&(np.sum((points-C)**2,axis=-1)<=r*r+1e-9)
    bound=math.pi/2-g['half_angle_radians'];theta=np.linspace(-bound,bound,samples)
    arc=C+r*np.cos(theta)[:,None]*u+r*np.sin(theta)[:,None]*n
    errors=[]
    for line,q in zip(g['independent_lines'],joins):
        v=np.asarray(line['direction']);radial=unit(q-C);tangent=np.array([-radial[1],radial[0]])
        errors.append(float(np.degrees(np.arccos(np.clip(abs(tangent@v),-1,1)))))
    info=dict(center=C,vertex=T,radius=r,curve=arc,join_points=joins,join_t_relative_center=join_t,
              analytic_join_angle_degrees=errors,axis_distance=float(abs((T-o)@n)),arc_span_degrees=float(np.degrees(2*bound)),
              circle_line_distance_errors=[float(abs(np.linalg.norm(q-C)-r)) for q in joins],
              construction='original_converging_lines_tangent_circle_through_RGB_projected_vertex',parallel_projection=False)
    return (rod|cap).astype(np.uint8),rod.astype(np.uint8),cap.astype(np.uint8),info,None


class TaperedTangentCap:
    def __init__(self,cfg):
        self.cfg=copy.deepcopy(cfg);self.seed=QualifiedSeedObserver(read(ROOT/cfg['seed_config']),'aggressive');self.reset()
    def reset(self):self.layers={};self.info={};self.final_mask=None
    def run(self,frame,raw,probability,old):
        start=time.perf_counter();self.final_mask=old.copy();z=np.zeros_like(raw,np.uint8)
        self.layers={k:z.copy() for k in ['subject','blue_identity','search_domain','tip_eligible','rod','cap','geometric_proposal','final_fiber','edit_add','edit_remove']}
        self.info=dict(applied=False,status='fallback',reason='no_trusted_subject',rejections=[],geometry=None,tip=None,cap=None,physical_tip_accuracy=None,old_fiber_retention_required=False,method='converging_straight_sides_tangent_round_cap_no_weak_growth')
        def done(reason=None):
            if reason:self.info['reason']=reason
            self.layers['final_fiber']=(self.final_mask==2).astype(np.uint8);self.layers['edit_add']=((self.final_mask==2)&(old!=2)).astype(np.uint8);self.layers['edit_remove']=((old==2)&(self.final_mask!=2)).astype(np.uint8)
            self.info.update(added_pixels=int(self.layers['edit_add'].sum()),removed_old_pixels=int(self.layers['edit_remove'].sum()),runtime_ms=(time.perf_counter()-start)*1000)
            if np.any(self.final_mask[raw==1]!=1):raise ValueError('Stone changed')
            self.info=public(self.info);return self.final_mask
        if raw.shape!=old.shape or frame.shape[:2]!=raw.shape:raise ValueError('Native shapes differ')
        if np.any(old[raw==1]!=1):raise ValueError('Old stone identity mismatch')
        seeds,records=self.seed.seeds(frame,raw,probability,None,99);self.info['seeds']=records
        if len(seeds)!=1:return done('ambiguous_trusted_subjects' if len(seeds)>1 else 'no_trusted_subject')
        seed=seeds[0];_,labels=cv2.connectedComponents((old==2).astype(np.uint8),connectivity=8);ids=np.unique(labels[seed['core']]);ids=ids[ids>0]
        if len(ids)!=1:return done('old_subject_not_unique')
        body=(labels==ids[0]).astype(np.uint8);self.layers['subject']=body;scale=min(raw.shape)/400
        g,errors=fit_geometry(body,seed['u'],self.cfg['fit'],scale);self.info['geometry']=g;self.info['rejections']=errors
        if g is None:return done(errors[0])
        blue,domain,eligible,tip=locate_tip(frame,raw,g,seed,self.cfg['tip'],scale);self.layers.update(blue_identity=blue.astype(np.uint8),search_domain=domain.astype(np.uint8),tip_eligible=eligible);self.info['tip']=tip
        if tip['vertex'] is None:
            errors.append(tip['reason']);return done(errors[0])
        G,rod,cap,cp,fail=raster_geometry(raw.shape,g,tip['vertex'],self.cfg['geometry']['arc_samples'])
        if fail:errors.append(fail);return done(errors[0])
        self.layers.update(geometric_proposal=G,rod=rod,cap=cap);self.info['cap']=cp
        if tip['reason']=='tip_reaches_search_limit':errors.append(tip['reason'])
        if G.sum()>self.cfg['geometry']['maximum_frame_area_fraction']*G.size:errors.append('geometric_area_budget')
        conflict=int(((G>0)&(raw==1)).sum())
        if conflict:errors.append('geometric_stone_conflict')
        self.info.update(rejections=errors,proposal_stone_conflict_pixels=conflict,geometry_without_blue_pixels=int(((G>0)&~blue).sum()),proposal_old_fiber_removed=int(((old==2)&(G==0)).sum()),proposal_pixels=int(G.sum()))
        if errors:return done(errors[0])
        self.final_mask[old==2]=0;self.final_mask[G>0]=2
        self.info.update(applied=True,status='applied',reason='converging_geometry_reconstruction')
        return done()
