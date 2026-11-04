"""One frozen factor: do not search the other end of an oriented visible shaft."""
from __future__ import annotations
import math
import numpy as np
from .fiber_blue_extent import BlueExtent,GrowthPipeline as ExtentPipeline,compatible_history
from .fiber_reference_workspace import ROOT,read

ARMS=('raw2','blue_extent','distal_only')

def direction(seed,shape,prior,age,settings):
    h,w=shape;s0,s1=seed['start_sections'][0]['center'],seed['start_sections'][1]['center']
    border=lambda p:min(p[0],p[1],w-1-p[0],h-1-p[1]);diff=border(s1)-border(s0)
    if abs(diff)>=settings['orientation_border_distance_difference_diagonal_fraction']*math.hypot(h,w):return (1 if diff>0 else -1),'current_border_hint'
    if compatible_history(prior,age,seed['origin'],seed['u'],seed['width'],shape,settings):
        sign=1 if np.dot(seed['u'],[math.cos(prior[2]),math.sin(prior[2])])>=0 else -1
        return sign,'prior_resolved'
    return 0,'ambiguous'

class DistalOnlyExtent(BlueExtent):
    def seeds(self,frame,mask,probability,prior,age):
        seeds,records=super().seeds(frame,mask,probability,prior,age)
        if len(seeds)==1:
            self.chosen,self.orientation=direction(seeds[0],mask.shape,prior,age,self.cfg['history'])
        return seeds,records
    def walk(self,frame,seed,sign,prior,age):
        if self.chosen and sign==self.chosen:
            self.calls.append(sign);return super().walk(frame,seed,sign,prior,age)
        current=seed['start_sections'][0 if sign<0 else 1];u=seed['u']*sign
        current={**current,'sides':current['sides'].copy(),'u':u.copy()}
        if np.dot(seed['start_sections'][0 if sign<0 else 1]['u'],u)<0:current['sides']=current['sides'][::-1].copy()
        return dict(sign=sign,sections=[current],polygons=[],rejected=[],distance=0.,stop_reason='proximal_not_searched' if self.chosen else 'direction_ambiguous',abnormal=None,final_u=u)
    def tip(self,frame,mask,seed,branch,oriented_sign):
        if branch['stop_reason'] in ['proximal_not_searched','direction_ambiguous']:
            return dict(state='unknown',reason=branch['stop_reason'],xy=None,end_sign=branch['sign'],direction='unsearched')
        return super().tip(frame,mask,seed,branch,oriented_sign)
    def grow(self,frame,mask,probability,prior=None,age=99):
        self.calls=[];self.chosen=0;self.orientation='unavailable'
        c,p,conf,m=super().grow(frame,mask,probability,prior,age)
        m['direction_control']=dict(walk_calls=self.calls.copy(),selected_sign=self.chosen,orientation_source=self.orientation)
        if m['seeds'] and len([s for s in m['seeds'] if s['reason']=='qualified'])==1:
            if 'distal_sign' in m and m['distal_sign']!=self.chosen:raise ValueError('Pre-walk orientation differs from frozen post-walk orientation')
            if self.orientation=='ambiguous':
                if c.any() or p.any() or any(t['xy'] is not None for t in m['tips']):raise ValueError('Ambiguous direction generated pixels/tip')
                m['reason']='direction_ambiguous'
        return c,p,conf,m

class GrowthPipeline:
    def __init__(self,cfg,arm,t1_config=None,sequential=False):
        if arm not in ARMS:raise ValueError('Unknown arm')
        reference=read(ROOT/cfg['reference_extent_config'])
        if cfg['extent']!=reference['extent']:raise ValueError('A numeric threshold changed in a direction-only control')
        self.pipeline=ExtentPipeline(reference,'raw2' if arm=='raw2' else 'blue_extent',t1_config,sequential)
        if arm=='distal_only':self.pipeline.growth=DistalOnlyExtent(reference['extent'])
        self.tracker=self.pipeline.tracker
    def reset(self):self.pipeline.reset()
    def run(self,*args,**kwargs):return self.pipeline.run(*args,**kwargs)
