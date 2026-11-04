"""Read-only RGB side/terminal measurements. Never produces a segmentation edit.

Blue evidence and photometric limits are the frozen tangent-control values.
Current RGB inside/outside transitions determine edge candidates; S2 points only
locate bounded measurement windows. An RGB candidate is NOT manual ground truth.
"""
import math
import cv2,numpy as np
from .fiber_fast_observation import sample
from .fiber_local_dark_evidence import nearest
from .fiber_straight_semicircle import blue_identity

def huber_diagnostics(line,delta):
    p=np.asarray(line['points']);o=np.asarray(line['point']);v=np.asarray(line['direction']);n=np.array([-v[1],v[0]])
    signed=(p-o)@n;r=abs(signed);w=np.minimum(1.,delta/np.maximum(r,1e-12))
    return dict(points=p,signed_residual=signed,weights=w,fully_weighted_fraction=float(np.mean(w>=.999)))

def edge_observations(frame,raw,points,axis,side,seed,tipcfg,cfg,scale):
    points=np.asarray(points,float);n=np.array([-axis[1],axis[0]])*side
    offsets=np.arange(-cfg['window_inward_at400'],cfg['window_outward_at400']+.001,cfg['sample_step_at400'])*scale
    centers=points[:,None,:]+offsets[None,:,None]*n
    inside_xy=centers[:,:,None,:]-np.array(cfg['support_offsets_at400'])[None,None,:,None]*scale*n
    outside_xy=centers[:,:,None,:]+np.array(cfg['support_offsets_at400'])[None,None,:,None]*scale*n
    a,av=sample(frame,inside_xy);b,bv=sample(frame,outside_xy)
    abi,ac,an=blue_identity(a,seed,tipcfg);bbi,bc,bn=blue_identity(b,seed,tipcfg)
    contrast=np.linalg.norm(np.median(a,axis=2)-np.median(b,axis=2),axis=-1)/(255*math.sqrt(3))
    chromatic=np.linalg.norm(np.median(ac,axis=2)-np.median(bc,axis=2),axis=-1)
    inside_blue=abi.mean(2);outside_blue=bbi.mean(2)
    labels,lv=nearest(raw,inside_xy)
    good=av.all(2)&bv.all(2)&lv.all(2)&(inside_blue>=2/3)&(outside_blue<=1/3)&~np.any(labels==1,axis=2)&((contrast>=tipcfg['minimum_RGB_contrast'])|(chromatic>=tipcfg['minimum_chroma_contrast']))
    margin=np.minimum.reduce([centers[...,0],centers[...,1],frame.shape[1]-1-centers[...,0],frame.shape[0]-1-centers[...,1]])
    good&=margin>=cfg['minimum_frame_margin_at400']*scale
    # Normal gradients are sampled in ORIGINAL native coordinates, not an ROI.
    rgb=frame.astype(np.float32)/255;gx=cv2.Sobel(rgb,cv2.CV_32F,1,0,ksize=3)/8;gy=cv2.Sobel(rgb,cv2.CV_32F,0,1,ksize=3)/8
    gradient=np.sqrt(np.sum((n[0]*gx+n[1]*gy)**2,axis=-1))/math.sqrt(3)
    grad,gv=sample(gradient,centers);good&=gv
    score=contrast+chromatic+grad
    selection=np.argmax(np.where(good,score,-1),axis=1);ii=np.arange(len(points));valid=good.any(1)
    chosen=centers[ii,selection];chosen[~valid]=points[~valid]
    return dict(raw_points=points,points=chosen,valid=valid,outward_offset=offsets[selection],
                contrast=contrast[ii,selection],chroma_contrast=chromatic[ii,selection],gradient=grad[ii,selection],
                inside_blue_fraction=inside_blue[ii,selection],outside_blue_fraction=outside_blue[ii,selection],
                full_offsets=offsets,full_scores=score,full_valid=good)

def broad_search_domain(shape,body,g,cfg,scale):
    """Observation envelope independent of the fitted convergence and V."""
    o=np.asarray(g['origin']);u=np.asarray(g['axis']);n=np.asarray(g['normal']);y,x=np.nonzero(body)
    if not len(x):return np.zeros(shape,bool)
    p=np.c_[x,y]-o;t=p@u;v=p@n;yy,xx=np.indices(shape);a=(xx-o[0])*u[0]+(yy-o[1])*u[1];b=(xx-o[0])*n[0]+(yy-o[1])*n[1]
    # Same axial interval as frozen locate_tip; ONLY transverse envelope changes.
    lo,hi=g['body_extent'];begin=lo+cfg['minimum_search_axial_fraction']*(hi-lo);limit=hi+cfg['maximum_forward_at400']*scale
    vlo,vhi=np.quantile(v,[.01,.99]);pad=cfg['independent_side_margin_at400']*scale
    return (a>=begin)&(a<=limit)&(b>=vlo-pad)&(b<=vhi+pad)

def terminal_observations(frame,raw,g,seed,tipcfg,domain,blue=None):
    """Frozen RGB external-terminal tests, domain supplied independently."""
    o,u,n=[np.asarray(g[k]) for k in ['origin','axis','normal']]
    if blue is None:blue,color,noise=blue_identity(frame,seed,tipcfg)
    else:_,color,noise=blue_identity(frame,seed,tipcfg)
    patch=cv2.filter2D(blue.astype(np.float32),-1,np.ones((3,3),np.float32),borderType=cv2.BORDER_CONSTANT)
    candidate=domain&blue&(patch>=tipcfg['minimum_local_blue_pixels_3x3'])&(raw!=1)
    eligible=np.zeros_like(raw,np.uint8);y,x=np.nonzero(candidate);meta=dict(vertex=None,source_pixel=None,reason='no_blue_external_terminal',qualified_pixels=0)
    if not len(x):return eligible,meta
    scale=min(raw.shape)/400;points=np.c_[x,y].astype(float);future=points[:,None,:]+np.asarray(tipcfg['forward_offsets_at400'])[None,:,None]*scale*u
    bg,valid=sample(frame,future);labels,lv=nearest(raw,future);forward_blue,bc,bnoise=blue_identity(bg,seed,tipcfg)
    contrast=np.linalg.norm(frame[y,x].astype(float)-np.median(bg,axis=1),axis=1)/(255*math.sqrt(3));chromatic=np.linalg.norm(color[y,x]-np.median(bc,axis=1),axis=1)
    margin=np.minimum.reduce([points[:,0],points[:,1],raw.shape[1]-1-points[:,0],raw.shape[0]-1-points[:,1]])
    good=valid.all(1)&lv.all(1)&~np.any(labels==1,axis=1)&(bnoise<tipcfg['maximum_chroma_noise']).all(1)&(forward_blue.mean(1)<=tipcfg['maximum_forward_blue_fraction'])&((contrast>=tipcfg['minimum_RGB_contrast'])|(chromatic>=tipcfg['minimum_chroma_contrast']))&(margin>=tipcfg['minimum_frame_margin_at400']*scale)
    eligible[y[good],x[good]]=1;ids=np.flatnonzero(good);meta['qualified_pixels']=len(ids)
    if not len(ids):return eligible,meta
    axial=(points-o)@u;order=np.lexsort((abs((points[ids]-o)@n),-axial[ids]));chosen=int(ids[order[0]]);end=float(axial[chosen]);limit=g['body_extent'][1]+tipcfg['maximum_forward_at400']*scale
    meta.update(source_pixel=points[chosen],vertex=o+end*u,tip_t=end,RGB_contrast=float(contrast[chosen]),chroma_contrast=float(chromatic[chosen]),search_limit_t=float(limit),reason='farthest_current_blue_external_contrast')
    if end>=limit-tipcfg['search_limit_margin_at400']*scale:meta['reason']='tip_reaches_search_limit'
    return eligible,meta
