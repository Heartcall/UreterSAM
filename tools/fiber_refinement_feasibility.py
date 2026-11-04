"""Label-free fiber ROI proposals and stone-protected diagnostic upper bounds."""
from __future__ import annotations
import cv2
import numpy as np
from .refinement_roi import _axis_starts, roi_union

def fiber_rois(mask, probability, settings, profile, cap=2):
    if profile not in {'hard2','soft02_2'}: raise ValueError('Unknown frozen profile')
    mask=np.asarray(mask)
    if mask.ndim!=2 or not np.isin(mask,[0,1,2]).all(): raise ValueError('Expected semantic mask')
    if profile=='hard2': seed=mask==2
    else:
        probability=np.asarray(probability)
        if probability.shape!=mask.shape or not np.isfinite(probability).all() or (probability<0).any() or (probability>1).any():raise ValueError('Expected native fiber probability')
        seed=probability>=settings['soft_probability_threshold']
    side,margin,stride,minimum=(int(settings[k]) for k in ['native_crop_side','margin_native_pixels','max_tile_stride','minimum_component_pixels'])
    if side<=0 or margin<0 or not 0<stride<=side or minimum<1:raise ValueError('Invalid geometry')
    if cap is not None and (not isinstance(cap,int) or cap<0):raise ValueError('Invalid cap')
    h,w=mask.shape; clean=np.zeros_like(seed); candidates=set()
    count,labels,stats,_=cv2.connectedComponentsWithStats(seed.astype(np.uint8),8)
    for i in range(1,count):
        x,y,ww,hh,area=map(int,stats[i])
        if area<minimum:continue
        clean[labels==i]=True
        for y0 in _axis_starts(y-margin,y+hh+margin,h,side,stride):
            for x0 in _axis_starts(x-margin,x+ww+margin,w,side,stride):candidates.add((x0,y0,x0+min(w,side),y0+min(h,side)))
    covered=np.zeros_like(seed);remaining=sorted(candidates);result=[]
    while remaining and (cap is None or len(result)<cap):
        scores=[int((clean[y0:y1,x0:x1]&~covered[y0:y1,x0:x1]).sum()) for x0,y0,x1,y1 in remaining]
        index=int(np.argmax(scores))
        if not scores[index]:break
        box=remaining.pop(index);result.append(box);x0,y0,x1,y1=box;covered[y0:y1,x0:x1]=True
    return result

def stone_protected_upper_bound(coarse,target,boxes):
    """GT allowed ONLY for explicit post-proposal oracle; never an inference output."""
    if coarse.shape!=target.shape:raise ValueError('Native geometry mismatch')
    change=roi_union(boxes,target.shape)&(target!=255)&(coarse!=1)
    result=coarse.copy();result[change&(coarse==2)]=0;result[change&(target==2)]=2
    assert np.array_equal(result==1,coarse==1)
    return result

def region_counts(target,coarse,reference,boxes):
    union=roi_union(boxes,target.shape);valid=target!=255;truth=target==2
    regions={'target':truth,'tp':truth&(coarse==2),'fn':truth&(coarse!=2),'fp':valid&~truth&(coarse==2),
        'lost_tp_vs_classwise':truth&(reference==2)&(coarse!=2),'added_fp_vs_classwise':valid&~truth&(reference!=2)&(coarse==2),
        'gt_fiber_blocked_by_stone':truth&(coarse==1)}
    return {'regions':{k:{'total':int(v.sum()),'inside':int((v&union).sum()),'outside':int((v&~union).sum())} for k,v in regions.items()},
        'roi_count':len(boxes),'roi_area_union':int(union.sum()),'image_pixels':int(target.size),'valid_pixels':int(valid.sum())}
