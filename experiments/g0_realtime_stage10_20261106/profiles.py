"""Batch exact group extrema; retain each original mean reduction unchanged."""
import numpy as np
from experiments.g0_realtime_stage2_20261105.runtime import grouped_profiles as reference


def grouped_profiles(body,origin,u,step,frame_margin=0):
    yy,xx=np.nonzero(body)
    if not len(xx):return reference(body,origin,u,step,frame_margin)
    points=np.stack([xx,yy],-1).astype(float)
    n=np.array([-u[1],u[0]])
    t,v=(points-origin)@u,(points-origin)@n
    if not np.isfinite(t).all() or not np.isfinite(v).all() or not np.isfinite(step) or step<=0:
        return reference(body,origin,u,step,frame_margin)
    lo=float(t.min());bins=np.floor((t-lo)/step).astype(int)
    order=np.argsort(bins,kind='stable')
    cut=np.flatnonzero(np.diff(bins[order]))+1
    starts=np.r_[0,cut];ends=np.r_[cut,len(order)];lengths=ends-starts
    values=v[order];positions=np.arange(len(order))
    low=np.minimum.reduceat(values,starts);high=np.maximum.reduceat(values,starts)
    # First equal position reproduces np.argmin/argmax tie order, including
    # signed zero. The selected original v value is written to the result.
    z=order[np.minimum.reduceat(np.where(values==np.repeat(low,lengths),positions,len(order)),starts)]
    w=order[np.minimum.reduceat(np.where(values==np.repeat(high,lengths),positions,len(order)),starts)]
    means=np.asarray([float(np.mean(t[order[a:b]])) for a,b in zip(starts,ends)])
    def valid(ids):
        p=points[ids]
        return np.minimum.reduce((p[:,0],p[:,1],body.shape[1]-1-p[:,0],body.shape[0]-1-p[:,1]))>=frame_margin
    return np.column_stack((means,v[z],v[w],valid(z),valid(w))),float(t.min()),float(t.max())
