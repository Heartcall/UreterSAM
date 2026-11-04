"""Bounded current-image evidence only. No mask edits, history, labels or tip claims."""
from __future__ import annotations

import math
import time
import cv2
import numpy as np

from .fiber_fast_observation import FastFiberObserver, sample

ARMS=('raw2','geometry_window','local_current_evidence')


def public(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:public(v) for k,v in value.items() if not k.startswith('_')}
    if isinstance(value,(tuple,list)):return [public(v) for v in value]
    return value


def nearest(image,xy):
    """Nearest pixel with explicit validity; never turn out-of-image into truth."""
    xy=np.asarray(xy,float);h,w=image.shape[:2]
    valid=np.isfinite(xy).all(-1)&(xy[...,0]>=0)&(xy[...,0]<=w-1)&(xy[...,1]>=0)&(xy[...,1]<=h-1)
    q=np.floor(np.nan_to_num(xy)+.5).astype(int)
    values=image[np.clip(q[...,1],0,h-1),np.clip(q[...,0],0,w-1)]
    return values,valid


def features(bgr):
    rgb=np.asarray(bgr,np.float32)[...,::-1]/np.float32(255.)
    c=rgb/(rgb.sum(-1,keepdims=True)+np.float32(1e-6));v=rgb.max(-1);lo=rgb.min(-1)
    saturation=np.divide(v-lo,v,out=np.zeros_like(v),where=v>0)
    color_available=~((v<.03)|((v>=.98)&(saturation<.10)))
    hsv=cv2.cvtColor(rgb.reshape(-1,1,3),cv2.COLOR_RGB2HSV).reshape(rgb.shape)
    lab=cv2.cvtColor(rgb.reshape(-1,1,3),cv2.COLOR_RGB2LAB).reshape(rgb.shape)
    return {'rgb':rgb,'chromaticity':c,'blue_excess':c[...,2]-.5*(c[...,0]+c[...,1]),
            'hsv_float_Hdeg_SV01':hsv,'lab_float_L100_ab':lab,'color_available':color_available}


def color_evidence(bgr,mu_f,mu_b,rho):
    f=features(bgr);d_f=np.linalg.norm(f['chromaticity']-mu_f,axis=-1);d_b=np.linalg.norm(f['chromaticity']-mu_b,axis=-1)
    f['qC']=(d_b-d_f)/(d_b+d_f+1e-6);f['rho_pixel']=f['color_available'].astype(float)*rho
    return f


def empty_observation():
    return {'candidates':[],'proposal_info':{},'hypothesis':'none','qualified_axis_groups':[],
            'ambiguous':False,'apply_to_mask':False,'tip_metrics':None,'timings':{'observer_total_ms':0.}}


class LocalDarkEvidence:
    def __init__(self,config,seed_config):
        self.config=config;self.seeds=FastFiberObserver(seed_config)

    @staticmethod
    def fov(frame):
        rgb=cv2.resize(frame,(160,160),interpolation=cv2.INTER_AREA);fov=(rgb.max(2)>10).astype(np.uint8)
        contours,_=cv2.findContours(fov,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);fov[:]=0
        if contours:cv2.drawContours(fov,[max(contours,key=cv2.contourArea)],-1,1,-1)
        return fov

    @staticmethod
    def valid_fov(xy,shape,fov):
        xy=np.asarray(xy,float);h,w=shape
        valid=np.isfinite(xy).all(-1)&(xy[...,0]>=0)&(xy[...,0]<=w-1)&(xy[...,1]>=0)&(xy[...,1]<=h-1)
        q=np.floor((np.nan_to_num(xy)+.5)*np.array([160/w,160/h])).astype(int)
        return valid&(fov[np.clip(q[...,1],0,159),np.clip(q[...,0],0,159)]>0)

    def anchor(self,frame,mask,probability,p,fov,appearance):
        cfg=self.config['proposal'];start,end=p['ends'];length=end-start
        t=start+np.array(cfg['anchor_axial_fractions'])*length
        xy=p['origin']+t[:,None,None]*p['u']+np.array(cfg['anchor_interior_normal_width_fractions'])[None,:,None]*p['width']*p['normal']
        pp,good=sample(probability.transpose(1,2,0),xy);labels,lg=nearest(mask,xy)
        valid=good&lg&self.valid_fov(xy,mask.shape,fov)
        high=valid&(pp[...,2]>=cfg['anchor_neural_min'])&(pp.argmax(-1)==2)&(labels==2)
        side=p['origin']+t[:,None,None]*p['u']+np.array(cfg['anchor_background_normal_width_fractions'])[None,:,None]*p['width']*p['normal']
        bgr,sg=sample(frame,side);sv=sg&self.valid_fov(side,mask.shape,fov)
        result={'qualified':bool(high.sum()>=cfg['anchor_min_highconfidence_points'] and sv.sum()>=6),
            'highconfidence_points':int(high.sum()),'valid_side_points':int(sv.sum()),'interior_xy':xy,'qualified_points':high,
            'background_xy':side,'background_valid':sv,'reason':'qualified' if high.sum()>=cfg['anchor_min_highconfidence_points'] and sv.sum()>=6 else 'insufficient_highconfidence_or_side_anchor'}
        if not result['qualified'] or not appearance:return result,None
        rgb,_=sample(frame,xy);mu_f=np.median(features(rgb)['chromaticity'][high],axis=0);mu_b=np.median(features(bgr)['chromaticity'][sv],axis=0)
        rho=float(np.clip(np.linalg.norm(mu_f-mu_b)/.10,0,1)*sv.mean())
        result.update(shaft_chromaticity=mu_f,background_chromaticity=mu_b,rhoC=rho)
        return result,(mu_f,mu_b,rho)

    def profile(self,frame,mask,probability,p,fov,sign,appearance,colors):
        cfg=self.config['search'];h,w=mask.shape;s=min(h,w)/400;length=float(p['ends'][1]-p['ends'][0]);width=float(p['width'])
        extension=min(.20*math.hypot(h,w),max(2*width,.5*length));n=min(cfg['maximum_stations_per_end'],int(math.ceil(extension/(2*s)))+1)
        d=np.linspace(0,extension,n);t=(p['ends'][0] if sign==-1 else p['ends'][1])+sign*d
        centers=p['origin']+t[:,None]*p['u'];half=.75*width+4*s
        fixed_offsets=np.array(self.config['proposal']['anchor_interior_normal_width_fractions'])*width
        xy=centers[:,None,:]+fixed_offsets[None,:,None]*p['normal']
        pp,pg=sample(probability.transpose(1,2,0),xy);raw,mg=nearest(mask,xy);good=pg&mg&self.valid_fov(xy,mask.shape,fov)
        result={'end_sign':sign,'distance_from_seed_end':d,'axial_t':t,'centers_xy':centers,'extension_px':extension,
            'actual_step_px':float(d[1]-d[0]) if n>1 else 0.,'halfwidth_px':half,
            'corridor_corners':np.array([centers[0]-half*p['normal'],centers[0]+half*p['normal'],centers[-1]+half*p['normal'],centers[-1]-half*p['normal']]),
            'primary_xy':xy,'primary_valid':good,'primary_originalP2':pp[...,2],'primary_raw_label':raw,
            'supported_probes':np.zeros(good.shape,bool),'protected_conflict':np.zeros(good.shape,bool),
            'station_supported':np.zeros(n,bool),'hypothesis_supported':False,'asserted_tip':None,'end_status':'geometry_opportunity_only'}
        if not appearance:return result
        normal_offsets=np.linspace(-half,half,cfg['transverse_samples']);edge_xy=centers[:,None,:]+normal_offsets[None,:,None]*p['normal']
        delta=cfg['edge_difference_offset_at400']*s
        a=edge_xy-delta*p['normal'];b=edge_xy+delta*p['normal'];a_rgb,av=sample(frame,a);b_rgb,bv=sample(frame,b)
        contrast=np.linalg.norm(a_rgb-b_rgb,axis=-1)/(255*math.sqrt(3));valid=av&bv&self.valid_fov(a,mask.shape,fov)&self.valid_fov(b,mask.shape,fov)&self.valid_fov(edge_xy,mask.shape,fov)
        selected=[];strength=[];edge_valid=[]
        for side in (-1,1):
            order=sorted(np.flatnonzero(normal_offsets*side>0),key=lambda k:(abs(normal_offsets[k]-side*width/2),normal_offsets[k]))
            values=np.where(valid[:,order],contrast[:,order],-1);idx=np.asarray(order)[values.argmax(1)]
            selected.append(normal_offsets[idx]);strength.append(contrast[np.arange(n),idx]);edge_valid.append(valid[np.arange(n),idx])
        chosen=np.stack(selected,1);strength=np.stack(strength,1);edge_valid=np.stack(edge_valid,1)
        bilateral=edge_valid.all(1)&(strength>=cfg['rgb_edge_contrast_min']).all(1)&(chosen[:,0]<chosen[:,1])
        bgr,rg=sample(frame,xy);color=color_evidence(bgr,*colors)
        inside=(fixed_offsets[None,:]>=chosen[:,0,None])&(fixed_offsets[None,:]<=chosen[:,1,None])
        c_support=(color['rho_pixel']>=.5)&(color['qC']>0);n_support=pp[...,2]>=.1
        point_support=good&rg&inside&bilateral[:,None]&(c_support|n_support)
        station=point_support.sum(1)>=2;point_support &= station[:,None]
        conflict=point_support&(raw==1)
        secondary_offsets=chosen[:,0,None]+(chosen[:,1]-chosen[:,0])[:,None]*np.array(cfg['interior_normal_fractions_between_edges'])[None,:]
        secondary_xy=centers[:,None,:]+secondary_offsets[...,None]*p['normal'];secondary_rgb,sg=sample(frame,secondary_xy);secondary_p,spg=sample(probability.transpose(1,2,0),secondary_xy)
        secondary_valid=sg&spg&self.valid_fov(secondary_xy,mask.shape,fov)&bilateral[:,None]
        # Changes are descriptive: every P decrease is retained, without fitting a drop cutoff.
        axial_rgb=np.linalg.norm(np.diff(color['rgb'],axis=0),axis=-1).mean(1)/math.sqrt(3)
        axial_p=np.diff(pp[...,2],axis=0).mean(1)
        later=np.array([bilateral[j+1:].sum()>=2 for j in range(n)])
        transition=np.r_[False,(axial_p<0)|(axial_rgb>=cfg['rgb_edge_contrast_min'])]&later
        if station[-1]:end_status='supported_at_search_limit_censored'
        elif not good[-1].any():end_status='outside_image_or_estimatedFOV_unknown'
        else:end_status='current_contrast_or_appearance_limit_unknown'
        result.update(primary_features=color,primary_color_support=c_support,primary_neural_support=n_support,
            primary_inside_witnessed_edges=inside,side_edge_xy=centers[:,None,:]+chosen[...,None]*p['normal'],side_edge_contrast=strength,
            side_edge_valid=edge_valid,bilateral_supported=bilateral,observed_width_px=chosen[:,1]-chosen[:,0],
            supported_probes=point_support,protected_conflict=conflict,station_supported=station,
            hypothesis_supported=bool(np.sum(station&(d>0))>=2),
            secondary_xy=secondary_xy,secondary_valid=secondary_valid,secondary_features=features(secondary_rgb),secondary_originalP2=secondary_p[...,2],
            axial_rgb_change=np.r_[0.,axial_rgb],axial_P2_change=np.r_[0.,axial_p],internal_transition_candidate=transition,end_status=end_status)
        return result

    def _observe_proposals(self,frame,mask,probability,arm,proposals,info):
        appearance=arm=='local_current_evidence';fov=self.fov(frame);candidates=[];groups=[]
        anchor_ms=profile_ms=0.
        for p in proposals:
            start=time.perf_counter();anchor,colors=self.anchor(frame,mask,probability,p,fov,appearance);anchor_ms+=(time.perf_counter()-start)*1000
            group=p['label'];support=set(p['_support_ids']);theta=math.atan2(p['u'][1],p['u'][0])
            for old in groups:
                iou=len(support&old['ids'])/max(len(support|old['ids']),1);angle=abs((theta-old['theta']+math.pi/2)%math.pi-math.pi/2)
                if iou>=.8 and angle<=math.radians(5):group=old['group'];break
            groups.append({'ids':support,'theta':theta,'group':group})
            candidate={'label':p['label'],'axis_group':int(group),'seed_origin_xy':p['origin'],'axis_u':p['u'],'seed_ends':p['ends'],
                'width_px':p['width'],'oriented_hint_only':p['oriented'],'fov_end_distances':p['fov_end_distances'],'anchor':anchor,'profiles':[],
                'axis_hypothesis_supported':False,'reason':anchor['reason']}
            if anchor['qualified']:
                start=time.perf_counter()
                candidate['profiles']=[self.profile(frame,mask,probability,p,fov,sign,appearance,colors) for sign in (-1,1)]
                profile_ms+=(time.perf_counter()-start)*1000
                candidate['axis_hypothesis_supported']=any(x['hypothesis_supported'] for x in candidate['profiles'])
                candidate['reason']='current_supported_hypothesis' if candidate['axis_hypothesis_supported'] else 'window_only' if not appearance else 'no_supported_extension'
            candidates.append(candidate)
        qualified=sorted({c['axis_group'] for c in candidates if c['axis_hypothesis_supported']});ambiguous=len(qualified)>1
        return {'candidates':candidates,'proposal_info':info,'qualified_axis_groups':qualified,'ambiguous':ambiguous,
            'hypothesis':'ambiguous' if ambiguous else 'unambiguous_supported_hypothesis' if qualified else 'none',
            'apply_to_mask':False,'tip_metrics':None,'timings':{'anchor_ms':anchor_ms,'profiles_evidence_ms':profile_ms}}

    def observe(self,frame,mask,probability,arm):
        if arm not in ARMS:raise ValueError('Unknown fixed arm')
        if arm=='raw2':return empty_observation()
        start=time.perf_counter();proposals,info=self.seeds.proposals(frame,probability);proposed=time.perf_counter()
        result=self._observe_proposals(frame,mask,probability,arm,proposals,info)
        result=public(result);ended=time.perf_counter()
        result['timings'].update(seed_proposal_ms=(proposed-start)*1000,observer_total_ms=(ended-start)*1000)
        return result
