"""S2 conservative domain plus strict native arithmetic; full graph/gradient."""
import math
import types

import numpy as np

from tools.fiber_reference_workspace import ROOT
from tools.fiber_local_dark_evidence import public
from tools.fiber_weak_stage2 import frontier, terminal
from experiments.g0_realtime_stage2_20261105.runtime import private_module, grouped_profiles
from experiments.g0_realtime_stage3_20261105 import pixels
from experiments.g0_realtime_stage4_20261105.runtime import Runtime as V4
from .native import Native


def rectangle(shape, origin, u, normal, lo, hi, center, lateral):
    if not np.isfinite([*origin,*u,*normal,lo,hi,center,lateral]).all():
        return (0,shape[0],0,shape[1])
    matrix=np.stack([u,normal])
    if np.linalg.cond(matrix)>1e8:return (0,shape[0],0,shape[1])
    corners=np.asarray([[lo,center-lateral],[lo,center+lateral],
                        [hi,center-lateral],[hi,center+lateral]])
    try:points=np.linalg.solve(matrix,corners.T).T+origin
    except np.linalg.LinAlgError:return (0,shape[0],0,shape[1])
    if not np.isfinite(points).all():return (0,shape[0],0,shape[1])
    h,w=shape
    return (max(0,min(h,math.floor(float(points[:,1].min()))-2)),
            max(0,min(h,math.ceil(float(points[:,1].max()))+3)),
            max(0,min(w,math.floor(float(points[:,0].min()))-2)),
            max(0,min(w,math.ceil(float(points[:,0].max()))+3)))


class Runtime(V4):
    def __init__(self):
        super().__init__()
        self.backend=Native(self.chroma)
        self.s2_spatial=dict(domain_full_pixels=0,domain_roi_pixels=0)
        self.colors.fast_evidence=self.backend.evidence
        self.observer.color=lambda bgr,seed,weak=False:self.backend.color(self.observer,bgr,seed,weak)
        self.expand=types.MethodType(expand,self)
        cap=private_module('experiments.g0_realtime_stage5_20261106._cap',
                           ROOT/'experiments/g0_realtime_stage5_20261106/cap.py')
        cap.evidence=self.evidence;cap.connect=pixels.connect;cap.profiles=grouped_profiles
        cap.projection=self.backend.project
        self.stage.fit_cap=types.MethodType(cap.fit_cap,self.stage)

    def run(self,*args,**kwargs):
        self.last_domain=None
        return super().run(*args,**kwargs)

    def reset(self):
        super().reset()
        self.backend.reset()
        for key in self.s2_spatial:self.s2_spatial[key]=0


def expand(self,frame,raw,probability,meta,B,core,u):
    stage=self.stage
    q=next(z for z in meta['seeds'] if z['reason']=='qualified')
    mu=np.asarray(q['mu'],np.float32)
    seed=dict(mu=mu,v=mu-1/3,bg=np.asarray(q['background_chroma'],np.float32),gap=q['relative_blue_advantage'])
    stage.gradient=pixels.native_gradient(frame)
    score,strong,weak,uncertain,edge,noise=self.evidence(frame,seed,stage.q['evidence'],stage.gradient)
    scale=min(raw.shape)/400;cfg=stage.q['search']
    origin=np.asarray(stage.outline['context']['origin']);normal=np.array([-u[1],u[0]])
    by,bx=np.nonzero(B)
    # Preserve exactly the old per-pixel multiply/add order and row-major data
    # supplied to quantile/median. Do not replace these expressions by matmul.
    bt=(bx-origin[0])*u[0]+(by-origin[1])*u[1]
    bv=(bx-origin[0])*normal[0]+(by-origin[1])*normal[1]
    front=float(bt.max())
    limit=min(cfg['maximum_forward_at400']*scale,cfg['maximum_forward_diagonal_fraction']*np.hypot(*raw.shape))
    lateral=(np.quantile(bv,.98)-np.quantile(bv,.02))*cfg['lateral_width_multiplier']/2+cfg['lateral_margin_at400']*scale
    center=float(np.median(bv));lo=bt.min()-2*scale;hi=front+limit
    y0,y1,x0,x1=rectangle(raw.shape,origin,u,normal,lo,hi,center,lateral)
    domain=np.zeros(raw.shape,bool)
    if y0<y1 and x0<x1:
        domain[y0:y1,x0:x1]=self.backend.domain((y1-y0,x1-x0),origin,u,normal,lo,hi,center,lateral,(y0,x0))
    self.last_domain=domain
    self.s2_spatial['domain_full_pixels']+=raw.size
    self.s2_spatial['domain_roi_pixels']+=(y1-y0)*(x1-x0)
    eligible=(strong|weak)&domain&(raw!=1)
    # The entire graph is retained, including B outside the new search domain.
    reachable=pixels.connect(eligible|(B>0),(B>0)&core)
    new=(reachable>0)&(B==0)&(raw==0)
    blocked=(int(reachable.sum())>cfg['maximum_component_fraction']*raw.size
             or int(new.sum())>cfg['maximum_new_fraction']*raw.size)
    recovered=B.copy() if blocked else reachable.copy()
    new=(recovered>0)&(B==0)&(raw==0)
    final=stage.final_mask.copy();final[new]=2;stage.final_mask=final
    cluster,distal=frontier(recovered,origin,u,stage.q['tip'],scale)
    info=stage.outline;info['distal']=distal;xy=distal['candidate_xy']
    ry,rx=np.nonzero(recovered)
    rt=(rx-origin[0])*u[0]+(ry-origin[1])*u[1]
    limited=bool(np.any(rt>=front+limit-1.5*scale))
    reason='area_budget_fallback' if blocked else 'evidence_graph_recovered' if new.any() else 'no_additional_reachable_evidence'
    info.update(recovery=dict(reason=reason,diagnostic_policy='functional_runtime_only'),
                candidate_tip_xy=xy,reason=reason)
    stage.layers['processed_body']=((stage.layers['root_instance']>0)|(recovered>0)).astype(np.uint8)
    stage.layers['reachable_blue']=recovered
    if xy is None:
        info['tip_observation']=dict(state='unknown',reason=distal['reason']);stage.outline=public(info);return
    observation=terminal(frame,raw,np.asarray(xy),u,seed,stage.q['evidence'],stage.q['tip'],scale,distal,limited or blocked,stage.gradient)
    info['tip_observation']=observation
    info['cap'].update(candidate_xy=xy,state=observation['state'],reason=observation['reason'],observable=observation['state']=='current_RGB_confirmed')
    if observation['state']=='current_RGB_confirmed':
        info['final_tip_xy']=xy
        stage.fit_cap(frame,raw,probability,recovered,core,meta,eligible,origin,u,xy,seed)
    stage.outline=public(stage.outline)
