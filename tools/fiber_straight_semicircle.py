"""Geometry-first rod and exact semicircle; independent experiment, no weak growth.

Independent fitted lines qualify the model. Final canonical sides are parallel,
so their join to the semicircle is analytically tangent. Interior pixels are a
geometric hypothesis, not a claim of individually observed visible fiber.
"""
from __future__ import annotations
import copy, math, time
import cv2
import numpy as np
from .fiber_reference_workspace import ROOT, read
from .fiber_weak_stage1 import QualifiedSeedObserver
from .fiber_connected_blue_growth import chroma
from .fiber_fast_observation import sample
from .fiber_local_dark_evidence import public, nearest


def unit(x):
    x = np.asarray(x, float)
    return x / max(float(np.linalg.norm(x)), 1e-12)


def robust_line(points, forward, delta, iterations):
    """Huber IRLS orthogonal straight-line fit, never a spline."""
    points = np.asarray(points, float)
    weights = np.ones(len(points))
    for _ in range(iterations + 1):
        center = np.average(points, axis=0, weights=weights)
        d = points - center
        _, vectors = np.linalg.eigh((d * weights[:, None]).T @ d)
        u = vectors[:, -1]
        if u @ forward < 0: u = -u
        residual = abs(d @ np.array([-u[1], u[0]]))
        weights = np.minimum(1., delta / np.maximum(residual, 1e-12))
    return dict(point=center, direction=u, median_error=float(np.median(residual)),
                RMS=float(np.sqrt(np.mean(residual ** 2))), points=points)


def observed_sides(body, origin, u, cfg, scale):
    y, x = np.nonzero(body)
    points = np.c_[x, y].astype(float)
    n = np.array([-u[1], u[0]])
    t, v = (points-origin) @ u, (points-origin) @ n
    lo, hi = float(t.min()), float(t.max())
    bins = np.floor((t-lo)/(cfg['profile_step_at400']*scale)).astype(int)
    left, right = [], []
    for k in np.unique(bins):
        ix = np.flatnonzero(bins == k)
        if not lo + cfg['axial_fraction'][0]*(hi-lo) <= np.mean(t[ix]) <= lo + cfg['axial_fraction'][1]*(hi-lo): continue
        a, b = points[ix[np.argmin(v[ix])]], points[ix[np.argmax(v[ix])]]
        margin = min(*a, body.shape[1]-1-a[0], body.shape[0]-1-a[1], *b, body.shape[1]-1-b[0], body.shape[0]-1-b[1])
        if margin < cfg['frame_margin_at400']*scale: continue
        left.append(a); right.append(b)
    return np.asarray(left).reshape(-1, 2), np.asarray(right).reshape(-1, 2), (lo, hi)


def fit_geometry(body, initial_u, cfg, scale):
    y, x = np.nonzero(body); origin = np.c_[x, y].mean(0)
    left, right, extent = observed_sides(body, origin, initial_u, cfg, scale)
    if len(left) < cfg['minimum_pairs']: return None, ['insufficient_visible_side_pairs']
    lines = [robust_line(a, initial_u, cfg['huber_delta_at400']*scale, cfg['IRLS_iterations']) for a in [left, right]]
    u = unit(lines[0]['direction'] + lines[1]['direction']); n = np.array([-u[1], u[0]])
    origin = (lines[0]['point']+lines[1]['point'])/2
    limits = [((a-origin)@u).min() for a in [left, right]], [((a-origin)@u).max() for a in [left, right]]
    lo, hi = max(limits[0]), min(limits[1])
    stations = np.linspace(lo, hi, cfg['width_stations'])
    distances = []
    for line in lines:
        normal = np.array([-line['direction'][1], line['direction'][0]])
        distances.append(((line['point']-origin-stations[:, None]*u) @ normal) / (n @ normal))
    widths = distances[1]-distances[0]
    width = float(widths.mean())
    if not np.isfinite(width) or width <= 0: return None, ['crossing_or_invalid_side_lines']
    origin = origin + float(np.mean((distances[0]+distances[1])/2))*n
    observed_widths = (right-left)@n
    def variability(w):
        return 1.4826*float(np.median(abs(w-np.median(w))))/width, float(np.quantile(w,.9)-np.quantile(w,.1))/width
    cv, drift = np.max([variability(widths), variability(observed_widths)], axis=0)
    angle = float(np.degrees(np.arccos(np.clip(lines[0]['direction']@lines[1]['direction'], -1, 1))))
    y, x = np.nonzero(body); ts = (np.c_[x,y]-origin)@u
    errors = []
    if angle > cfg['maximum_side_angle_degrees']: errors.append('sides_not_parallel')
    if cv > cfg['maximum_width_robust_CV'] or drift > cfg['maximum_width_p90_p10_fraction']: errors.append('width_unstable')
    if hi-lo < cfg['minimum_span_at400']*scale: errors.append('insufficient_fitted_span')
    if width < cfg['minimum_width_at400']*scale or width > cfg['maximum_width_frame_fraction']*min(body.shape): errors.append('width_out_of_range')
    if np.ptp(ts)/width < cfg['minimum_observed_aspect']: errors.append('subject_not_slender')
    if any(a['median_error'] > max(cfg['maximum_median_line_error_at400']*scale, cfg['maximum_median_line_error_width_fraction']*width) for a in lines): errors.append('nonstraight_side_residual')
    return dict(origin=origin, axis=u, normal=n, width=width, radius=width/2,
                independent_lines=lines, side_angle_degrees=angle, width_robust_CV=float(cv),
                width_p90_p10_fraction=float(drift), observed_widths=observed_widths,
                fit_span=[float(lo),float(hi)], body_extent=[float(ts.min()),float(ts.max())]), errors


def blue_identity(frame, seed, cfg):
    c, noise = chroma(frame); v = c-1/3
    norm = np.linalg.norm(v, axis=-1); reference = seed['mu']-1/3
    cosine = (v@reference)/np.maximum(norm*np.linalg.norm(reference),1e-12)
    blue = c[...,2]-.5*(c[...,0]+c[...,1]); bg = seed['bg']
    gap = blue-(bg[2]-.5*(bg[0]+bg[1]))
    result = (blue>cfg['minimum_blue_advantage']) & (gap>max(cfg['minimum_blue_advantage'], cfg['minimum_seed_gap_fraction']*seed['gap'])) & (norm>cfg['minimum_chroma_norm']) & (cosine>cfg['minimum_seed_cosine']) & (noise<cfg['maximum_chroma_noise'])
    return result, c, noise


def locate_tip(frame, raw, geometry, seed, cfg, scale):
    o,u,n,w = [geometry[k] for k in ['origin','axis','normal','width']]
    blue, color, noise = blue_identity(frame, seed, cfg)
    yy,xx = np.indices(raw.shape); t=(xx-o[0])*u[0]+(yy-o[1])*u[1]; v=(xx-o[0])*n[0]+(yy-o[1])*n[1]
    lo,old_end=geometry['body_extent']; begin=lo+cfg['minimum_search_axial_fraction']*(old_end-lo); limit=old_end+cfg['maximum_forward_at400']*scale
    patch=cv2.filter2D(blue.astype(np.float32),-1,np.ones((3,3),np.float32),borderType=cv2.BORDER_CONSTANT)
    domain=(t>=begin)&(t<=limit)&(abs(v)<=w/2)
    candidate=domain&blue&(patch>=cfg['minimum_local_blue_pixels_3x3'])&(raw!=1)
    eligible=np.zeros_like(raw,np.uint8); y,x=np.nonzero(candidate)
    meta=dict(reason='no_blue_external_terminal',source_pixel=None,vertex=None,qualified_pixels=0,search_limit_t=float(limit),pixel_candidates=int(len(x)))
    if not len(x): return blue, domain, eligible, meta
    points=np.c_[x,y].astype(float); future=points[:,None,:]+np.asarray(cfg['forward_offsets_at400'])[None,:,None]*scale*u
    bg,valid=sample(frame,future); labels,lv=nearest(raw,future)
    forward_blue,bc,bnoise=blue_identity(bg,seed,cfg)
    current=frame[y,x].astype(float); med=np.median(bg,axis=1)
    contrast=np.linalg.norm(current-med,axis=1)/(255*math.sqrt(3)); chromatic=np.linalg.norm(color[y,x]-np.median(bc,axis=1),axis=1)
    margin=np.minimum.reduce([points[:,0], points[:,1],raw.shape[1]-1-points[:,0],raw.shape[0]-1-points[:,1]])
    good=valid.all(1)&lv.all(1)&~np.any(labels==1,axis=1)&(bnoise<cfg['maximum_chroma_noise']).all(1)&(forward_blue.mean(1)<=cfg['maximum_forward_blue_fraction'])&((contrast>=cfg['minimum_RGB_contrast'])|(chromatic>=cfg['minimum_chroma_contrast']))&(margin>=cfg['minimum_frame_margin_at400']*scale)
    eligible[y[good],x[good]]=1; ids=np.flatnonzero(good)
    meta.update(qualified_pixels=len(ids),rejected_no_contrast=int((~((contrast>=cfg['minimum_RGB_contrast'])|(chromatic>=cfg['minimum_chroma_contrast']))).sum()),rejected_forward_blue=int((forward_blue.mean(1)>cfg['maximum_forward_blue_fraction']).sum()),rejected_forward_stone=int(np.any(labels==1,axis=1).sum()),rejected_unknown_or_outside=int((~valid.all(1)|(bnoise>=cfg['maximum_chroma_noise']).any(1)).sum()))
    if not len(ids): return blue,domain,eligible,meta
    axial=(points-o)@u
    # Literal farthest qualified observed pixel; tie-break by closeness to axis.
    order=np.lexsort((abs((points[ids]-o)@n),-axial[ids])); chosen=int(ids[order[0]]); end=float(axial[chosen]); T=o+end*u
    meta.update(source_pixel=points[chosen],vertex=T,tip_t=end,source_normal_distance=float((points[chosen]-o)@n),RGB_contrast=float(contrast[chosen]),chroma_contrast=float(chromatic[chosen]),forward_blue_fraction=float(forward_blue[chosen].mean()),reason='farthest_current_blue_external_contrast')
    if end>=limit-cfg['search_limit_margin_at400']*scale:meta['reason']='tip_reaches_search_limit'
    return blue,domain,eligible,meta


def raster_geometry(shape, origin, u, width, T, samples=129):
    u=unit(u);n=np.array([-u[1],u[0]]);r=width/2;C=np.asarray(T)-r*u
    yy,xx=np.indices(shape);along=(xx-C[0])*u[0]+(yy-C[1])*u[1];across=(xx-C[0])*n[0]+(yy-C[1])*n[1]
    rod=(along<=0)&(abs(across)<=r);cap=(along>=0)&(along**2+across**2<=r*r+1e-9)
    theta=np.linspace(-math.pi/2,math.pi/2,samples);arc=C+r*np.cos(theta)[:,None]*u+r*np.sin(theta)[:,None]*n
    return (rod|cap).astype(np.uint8),rod.astype(np.uint8),cap.astype(np.uint8),dict(center=C,vertex=T,radius=r,curve=arc,join_points=[C-r*n,C+r*n],analytic_join_angle_degrees=[0.,0.],axis_distance=float(abs((np.asarray(T)-origin)@n)))


class StraightSemicircle:
    def __init__(self,cfg):
        self.cfg=copy.deepcopy(cfg);self.seed=QualifiedSeedObserver(read(ROOT/cfg['seed_config']),'aggressive');self.reset()
    def reset(self):self.layers={};self.info={};self.final_mask=None
    def run(self,frame,raw,probability,old):
        start=time.perf_counter();self.final_mask=old.copy();z=np.zeros_like(raw,np.uint8)
        self.layers={k:z.copy() for k in ['subject','blue_identity','search_domain','tip_eligible','rod','cap','geometric_proposal','final_fiber','edit_add','edit_remove']}
        self.info=dict(applied=False,status='fallback',reason='no_trusted_subject',rejections=[],geometry=None,tip=None,cap=None,physical_tip_accuracy=None,old_fiber_retention_required=False,method='straight_rod_semicircle_no_weak_growth')
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
            self.info['rejections'].append(tip['reason']);return done(errors[0] if errors else tip['reason'])
        G,rod,cap,cap_info=raster_geometry(raw.shape,g['origin'],g['axis'],g['width'],tip['vertex'],self.cfg['geometry']['arc_samples']);self.layers.update(geometric_proposal=G,rod=rod,cap=cap);self.info['cap']=cap_info
        if tip['reason']=='tip_reaches_search_limit':errors.append(tip['reason'])
        if G.sum()>self.cfg['geometry']['maximum_frame_area_fraction']*G.size:errors.append('geometric_area_budget')
        seed_loss=float(((G==0)&seed['core']).sum()/max(int(seed['core'].sum()),1))
        if seed_loss>self.cfg['geometry']['maximum_seed_loss_fraction']:errors.append('geometry_disagrees_with_trusted_subject')
        conflict=int(((G>0)&(raw==1)).sum())
        if conflict:errors.append('geometric_stone_conflict')
        self.info.update(rejections=errors,proposal_stone_conflict_pixels=conflict,seed_loss_fraction=seed_loss,geometry_without_blue_pixels=int(((G>0)&~blue).sum()),proposal_old_fiber_removed=int(((old==2)&(G==0)).sum()),proposal_pixels=int(G.sum()))
        if errors:return done(errors[0])
        self.final_mask[old==2]=0;self.final_mask[G>0]=2
        self.info.update(applied=True,status='applied',reason='straight_geometry_reconstruction')
        return done()
