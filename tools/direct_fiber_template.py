"""Fixed shared shape family; pose hypotheses scored directly against RGB evidence."""
from __future__ import annotations

import itertools
import cv2
import numpy as np


def wrap(x):return (x+np.pi)%(2*np.pi)-np.pi


def train_geometry(frame,mask,config):
    from ureter_semis.video_metrics import estimated_fov
    from .fiber_shape_prior import robust_line
    fiber=mask==2;h,w=fiber.shape;size=min(h,w)
    fov=estimated_fov(frame,(h,w));dist=cv2.distanceTransform(np.pad(fov.astype(np.uint8),1),cv2.DIST_L2,5)[1:-1,1:-1]
    count,labels,stats,_=cv2.connectedComponentsWithStats(fiber.astype(np.uint8),8)
    choices=[]
    for label in range(1,count):
        a=labels==label
        if stats[label,cv2.CC_STAT_AREA]>=config['minimum_fiber_pixels'] and int((a&(dist<=size*config['entry_fraction'])).sum())>=3:choices.append((int(a.sum()),label))
    if not choices:return None
    selected=labels==max(choices)[1];yy,xx=np.where(selected);points=np.column_stack([xx,yy]).astype(np.float32)
    origin=points.mean(0);_,vectors=cv2.PCACompute(points,mean=None,maxComponents=1);u=vectors[0];s=(points-origin)@u
    low,high=np.percentile(s,[5,95]);a=np.abs(s-low)<max(2,np.std(s)*.1);b=np.abs(s-high)<max(2,np.std(s)*.1)
    if np.median(dist[yy[a],xx[a]])>np.median(dist[yy[b],xx[b]]):u=-u
    v=np.array([-u[1],u[0]]);s=(points-origin)@u;n=(points-origin)@v;start,end=np.percentile(s,[.5,99.5]);length=end-start
    if length<size*config['minimum_length_fraction']:return None
    ts=np.linspace(start+length*config['body_range'][0],start+length*config['body_range'][1],config['shaft_bins']);pairs=[]
    for t in ts:
        a=np.abs(s-t)<max(1.5,length/config['shaft_bins']*.6)
        if a.sum()>=4:
            lo,hi=np.percentile(n[a],[3,97]);pairs.append([t,(hi-lo)/2])
    if len(pairs)<8:return None
    pairs=np.asarray(pairs);coef,residual=robust_line(pairs[:,0],pairs[:,1],{'robust_iterations':4,'huber_mad_multiplier':2.5})
    slope=-float(coef[1]);radius=float((coef[0]+coef[1]*end)/max(1-slope,.1));fraction=radius/size
    if length/(2*np.median(pairs[:,1]))<config['minimum_aspect_ratio']:return None
    if not config['slope_range'][0]<=slope<=config['slope_range'][1] or not config['radius_fraction_range'][0]<=fraction<=config['radius_fraction_range'][1]:return None
    if residual>max(size/400, np.median(pairs[:,1])*.2):return None
    return {'halfwidth_slope':slope,'cap_radius_fraction':fraction,'residual_px':residual,'selected':selected}


def appearance_hist(frame,fg,bins):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV);h=(hsv[:,:,0].astype(int)*bins//180).clip(0,bins-1);s=(hsv[:,:,1].astype(int)*bins//256).clip(0,bins-1)
    a=np.bincount((h[fg]*bins+s[fg]).ravel(),minlength=bins*bins).reshape(bins,bins).astype(float)
    return a/max(a.sum(),1)


def observation(frame,prior,config):
    size=config['search']['size'];height,width=frame.shape[:2];ratio=size/max(height,width);h=max(8,round(height*ratio));w=max(8,round(width*ratio))
    small=cv2.resize(frame,(w,h),interpolation=cv2.INTER_AREA);hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV);bins=prior['histogram_bins']
    hs=(hsv[:,:,0].astype(int)*bins//180).clip(0,bins-1);ss=(hsv[:,:,1].astype(int)*bins//256).clip(0,bins-1)
    probability=np.asarray(prior['appearance_probability'],np.float32)[hs,ss]
    probability=np.where(hsv[:,:,2]>10,probability,0).astype(np.float32)
    from ureter_semis.video_metrics import estimated_fov
    fov=estimated_fov(small,(h,w));image=cv2.GaussianBlur(small.astype(np.float32)/255,(3,3),.8)
    gx=cv2.Sobel(image,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(image,cv2.CV_32F,0,1,ksize=3)
    contours,_=cv2.findContours(fov.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours:return None
    boundary=max(contours,key=cv2.contourArea)[:,0,:].astype(np.float32);full_boundary=boundary.copy()
    boundary=boundary[np.linspace(0,len(boundary)-1,config['search']['boundary_samples'],dtype=int)]
    center=np.array([w/2,h/2]);normal=np.arctan2(center[1]-boundary[:,1],center[0]-boundary[:,0])
    params=[]
    for entry,angle in zip(boundary,normal):
        for offset,length in itertools.product(config['search']['angles_degrees'],config['search']['length_fractions']):
            params.append([entry[0],entry[1],angle+np.deg2rad(offset),length*min(h,w)])
    return {'image':small,'appearance':probability,'fov':fov.astype(np.float32),'gx':gx,'gy':gy,'poses':np.asarray(params,np.float32),'ratio':ratio,'shape':(h,w),'boundary':full_boundary}


def sample(array,xy):
    return cv2.remap(array,xy[:,:,0].astype(np.float32),xy[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)


def score_poses(obs,poses,prior,config,network=None):
    poses=np.asarray(poses,np.float32);u=np.column_stack([np.cos(poses[:,2]),np.sin(poses[:,2])]);v=np.column_stack([-u[:,1],u[:,0]])
    radius=prior['cap_radius_fraction']*min(obs['shape']);slope=prior['halfwidth_slope'];center=np.maximum(poses[:,3]-radius,1)
    t=center[:,None]*np.linspace(.10,.90,config['search']['body_samples'])[None,:];width=radius+slope*(center[:,None]-t)
    anchor=poses[:,:2,None].transpose(0,2,1)+t[:,:,None]*u[:,None,:]
    inside=np.concatenate([anchor+width[:,:,None]*f*v[:,None,:] for f in (-.65,0,.65)],axis=1)
    outside=np.concatenate([anchor+width[:,:,None]*f*v[:,None,:] for f in (-1.35,1.35)],axis=1)
    side=np.concatenate([anchor-width[:,:,None]*v[:,None,:],anchor+width[:,:,None]*v[:,None,:]],axis=1)
    side_normals=np.concatenate([-np.repeat(v[:,None,:],t.shape[1],1),np.repeat(v[:,None,:],t.shape[1],1)],axis=1)
    angles=np.linspace(-np.pi/2,np.pi/2,config['search']['cap_samples'])
    cap=poses[:,:2,None].transpose(0,2,1)+center[:,None,None]*u[:,None,:]+radius*(np.cos(angles)[None,:,None]*u[:,None,:]+np.sin(angles)[None,:,None]*v[:,None,:])
    cap_normals=np.cos(angles)[None,:,None]*u[:,None,:]+np.sin(angles)[None,:,None]*v[:,None,:]
    contour=np.concatenate([side,cap],1);normals=np.concatenate([side_normals,cap_normals],1)
    p_in=sample(obs['appearance'],inside).mean(1);p_out=sample(obs['appearance'],outside).mean(1)
    contrast=(sample(obs['appearance'],contour-normals)-sample(obs['appearance'],contour+normals)).mean(1)
    dx=sample(obs['gx'],contour);dy=sample(obs['gy'],contour)
    rgb=np.sqrt(np.sum((dx*normals[:,:,0,None]+dy*normals[:,:,1,None])**2,axis=2)).mean(1)
    support=sample(obs['fov'],inside).mean(1)
    weights=config['score'];score=weights['inside']*p_in+weights['region_contrast']*(p_in-p_out)+weights['boundary_contrast']*contrast+weights['rgb_edge']*np.minimum(rgb,1)
    soft=score.copy()
    if network is not None:soft+=weights['network_soft']*(sample(network,inside).mean(1)-.5)
    return {'score':score,'soft':soft,'inside':p_in,'contrast':p_in-p_out,'boundary':contrast,'rgb':rgb,'support':support}


def fine_poses(poses,score,config,shape,boundary=None):
    indices=np.argsort(score,kind='stable')[-config['search']['topk']:];result=[];size=min(shape)
    for p in poses[indices]:
        # Tangential entrance motion keeps the search local to the view boundary approximation.
        tangent=np.array([-np.sin(p[2]),np.cos(p[2])])
        for shift,angle,length in itertools.product(config['search']['refine_entry_fraction'],config['search']['refine_angles_degrees'],config['search']['refine_length_fraction']):
            result.append([*(p[:2]+shift*size*tangent),p[2]+np.deg2rad(angle),max(2,p[3]+length*size)])
    result=np.asarray(result,np.float32)
    if boundary is not None:
        nearest=np.sum((result[:,:2,None]-boundary.T[None,:,:])**2,axis=1).argmin(1)
        result[:,:2]=boundary[nearest]
    return result


def pick(obs,poses,scores,key,config,history=None):
    objective=scores[key].copy()
    if history is not None:
        c=config['tracking'];size=min(obs['shape']);p=np.asarray(history)
        penalty=np.sum(((poses[:,:2]-p[:2])/(size*c['entry_scale_fraction']))**2,1)+((poses[:,3]-p[3])/(size*c['length_scale_fraction']))**2+(wrap(poses[:,2]-p[2])/np.deg2rad(c['angle_scale_degrees']))**2
        objective-=c['temporal_weight']*np.minimum(penalty,c['penalty_cap'])
    i=int(np.argmax(objective));a=config['acceptance']
    checks={'appearance':scores['inside'][i]>=a['inside_appearance_min'],'region_contrast':scores['contrast'][i]>=a['region_contrast_min'],
        'boundary':scores['boundary'][i]>=a['boundary_contrast_min'],'rgb_edge':scores['rgb'][i]>=a['rgb_edge_min'],'fov':scores['support'][i]>=a['fov_support_min']}
    return {'accepted':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()},'pose':poses[i].astype(float).tolist(),
        'objective':float(objective[i]),'features':{k:float(v[i]) for k,v in scores.items()},'role':'complete2D_hypothesis_not_visible_truth'}


def native_pose(record,ratio):
    p=np.asarray(record['pose'],float);p[[0,1,3]]/=ratio
    return p.tolist()


def visible(frame,pose,accepted,prior,config):
    h,w=frame.shape[:2]
    if not accepted:return np.zeros((h,w),bool),np.zeros((h,w),bool)
    x,y,angle,length=pose;radius=prior['cap_radius_fraction']*min(h,w);center=max(1,length-radius);yy,xx=np.indices((h,w),dtype=np.float32)
    t=(xx-x)*np.cos(angle)+(yy-y)*np.sin(angle);n=-(xx-x)*np.sin(angle)+(yy-y)*np.cos(angle)
    width=radius+prior['halfwidth_slope']*(center-t)
    template=((t>=0)&(t<=center)&(np.abs(n)<=width))|((t>=center)&((t-center)**2+n**2<=radius**2))
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV);bins=prior['histogram_bins'];hh=(hsv[:,:,0].astype(int)*bins//180).clip(0,bins-1);ss=(hsv[:,:,1].astype(int)*bins//256).clip(0,bins-1)
    evidence=np.asarray(prior['appearance_probability'])[hh,ss]>=config['acceptance']['visible_appearance_min']
    return template & evidence & (hsv[:,:,2]>10),template


class DirectTracker:
    def __init__(self,config):self.config=config;self.reset()
    def reset(self):self.pose=None;self.velocity=None;self.age=0
    def prediction(self):
        return None if self.pose is None else self.pose+self.velocity
    def update(self,pose,accepted):
        if accepted:
            measured=np.asarray(pose,float)
            if self.pose is None:self.velocity=np.zeros(4);status='initialized'
            else:
                d=measured-self.pose;d[2]=wrap(d[2]);self.velocity=(1-self.config['velocity_beta'])*self.velocity+self.config['velocity_beta']*d;status='updated'
            self.pose=measured.copy();self.age=0
            return {'status':status,'pose':self.pose.tolist(),'applied':True,'age':0}
        if self.pose is None:return {'status':'no_state','pose':None,'applied':False,'age':0}
        self.age+=1
        if self.age>self.config['prediction_only_frames']:
            self.reset();return {'status':'expired','pose':None,'applied':False,'age':0}
        self.pose=self.prediction();self.pose[2]=wrap(self.pose[2])
        return {'status':'predicted_not_visible','pose':self.pose.tolist(),'applied':False,'age':self.age}


def locate(frame,prior,config,network_probability=None,tracker=None,requested=None):
    import time
    started=time.perf_counter();requested=set(requested or ('image_template','soft_template','causal_template'));obs=observation(frame,prior,config)
    if obs is None:
        out={};timings={'shared_ms':(time.perf_counter()-started)*1000}
        for name in requested:
            if name=='soft_template' and network_probability is None:continue
            if name=='causal_template' and tracker is None:continue
            record={'accepted':False,'checks':{'fov':False},'pose':None,'objective':None,'features':{},'role':'no_observation'}
            if name=='causal_template':record['tracking']=tracker.update(None,False)
            out[name]={'record':record,'fiber':np.zeros(frame.shape[:2],bool),'hypothesis':np.zeros(frame.shape[:2],bool)};timings[name+'_ms']=0.
        return out,timings
    network=cv2.resize(network_probability,(obs['shape'][1],obs['shape'][0]),interpolation=cv2.INTER_LINEAR) if network_probability is not None else None
    coarse=score_poses(obs,obs['poses'],prior,config,network);shared=time.perf_counter();out={};timings={'shared_ms':(shared-started)*1000}
    for name,key in [('image_template','score'),('soft_template','soft')]:
        if name not in requested:continue
        if key=='soft' and network is None:continue
        t=time.perf_counter();poses=fine_poses(obs['poses'],coarse[key],config,obs['shape'],obs['boundary']);scores=score_poses(obs,poses,prior,config,network if key=='soft' else None)
        record=pick(obs,poses,scores,key,config);record['pose']=native_pose(record,obs['ratio'])
        mask,hypothesis=visible(frame,record['pose'],record['accepted'],prior,config)
        out[name]={'record':record,'fiber':mask,'hypothesis':hypothesis};timings[name+'_ms']=(time.perf_counter()-t)*1000
    if tracker is not None and 'causal_template' in requested:
        t=time.perf_counter();poses=fine_poses(obs['poses'],coarse['soft'],config,obs['shape'],obs['boundary']);prediction=tracker.prediction();history=None
        if prediction is not None:
            history=prediction.copy();history[[0,1,3]]*=obs['ratio']
            poses=np.vstack([poses,fine_poses(history[None,:],np.array([0.]),{**config,'search':{**config['search'],'topk':1}},obs['shape'],obs['boundary'])])
        scores=score_poses(obs,poses,prior,config,network);record=pick(obs,poses,scores,'soft',config,history);record['pose']=native_pose(record,obs['ratio'])
        tracked=tracker.update(record['pose'],record['accepted']);record['tracking']=tracked
        mask,hypothesis=visible(frame,record['pose'],record['accepted'],prior,config)
        out['causal_template']={'record':record,'fiber':mask,'hypothesis':hypothesis};timings['causal_template_ms']=(time.perf_counter()-t)*1000
    return out,timings
