"""S1: validate each seed before declaring subject ambiguity; frozen thresholds."""
import time
import cv2,numpy as np
from .fiber_aggressive_blue import AggressiveVariant
from .fiber_connected_blue_union import ConnectedBlue
from .fiber_weak_blue_evidence import GrowthPipeline as FrozenPipeline

class QualifiedSeedObserver(AggressiveVariant):
    def seeds(self,frame,mask,probability,prior,age):
        cfg=self.cfg['seed'];s=min(mask.shape)/400
        _,labels,stats,_=cv2.connectedComponentsWithStats(((mask==2)&(probability[2]>=cfg['P_min'])).astype(np.uint8),connectivity=4)
        ids=[k for k in range(1,len(stats)) if stats[k,4]>=cfg['min_pixels_at400']*s*s]
        self.chosen=0;self.orientation='unavailable'
        if len(ids)>32:return [],[dict(reason='seed_computation_budget',components=len(ids),limit=32)]
        qualified=[];records=[]
        for k in sorted(ids,key=lambda k:(-stats[k,4],k)):
            isolated=mask.copy();isolated[(mask==2)&(labels!=k)]=0
            seeds,rows=super().seeds(frame,isolated,probability,prior,age)
            for q in seeds:q['label']=k;qualified.append(q)
            for r in rows:r.update(label=k,raw_candidate_count=len(ids));records.append(r)
        self.chosen=1 if len(qualified)==1 else 0;self.orientation='fixed_edge_to_center' if len(qualified)==1 else 'ambiguous_qualified_subjects' if len(qualified)>1 else 'unavailable'
        return qualified,records

def install_seed_fix(pipeline):
    node=pipeline;seen=set()
    while id(node) not in seen:
        seen.add(id(node))
        if isinstance(node,ConnectedBlue):
            old=node.observer;node.observer=QualifiedSeedObserver(old.full,'aggressive');return
        if hasattr(node,'growth') and isinstance(node.growth,ConnectedBlue):node=node.growth
        elif hasattr(node,'inner'):node=node.inner
        else:break
    raise RuntimeError('Frozen seed observer path missing')

class GrowthPipeline(FrozenPipeline):
    def __init__(self,cfg,arm='weak_blue_extension_tip',t1_config=None,sequential=False):
        super().__init__(cfg,arm,t1_config,sequential);install_seed_fix(self)
