"""Prespecified cumulative controls; experimental candidate layers only."""
from __future__ import annotations
import copy,math,time
import cv2,numpy as np
from .fiber_blue_extent import BlueExtent,GrowthPipeline as ExtentPipeline,compatible_history
from .fiber_distal_only import DistalOnlyExtent,GrowthPipeline as DistalPipeline,direction
from .fiber_connected_blue_growth import _pca,angle,chroma
from .fiber_fast_observation import sample
from .fiber_reference_workspace import ROOT,read

ARMS=('raw2','distal_only','visible_bg','fixed_axis','color_edges','broad_blue','aggressive')

def blue(c):return c[...,2]-.5*(c[...,0]+c[...,1])

def fixed_axis(points,shape,settings):
    h,w=shape;d=np.min(np.stack([points[:,0],points[:,1],w-1-points[:,0],h-1-points[:,1]],-1),axis=1)
    k=min(len(points),max(settings['minimum_anchor_points'],int(math.ceil(settings['nearest_seed_fraction']*len(points)))))
    near=points[np.argsort(d,kind='stable')[:k]];anchor=np.median(near,axis=0);anchor=np.clip(anchor,[0,0],[w-1,h-1])
    side=int(np.argmin([anchor[0],anchor[1],w-1-anchor[0],h-1-anchor[1]]))
    anchor[0 if side in [0,2] else 1]=[0,0,w-1,h-1][side]
    vector=np.array([(w-1)/2,(h-1)/2])-anchor;u=vector/np.linalg.norm(vector)
    return anchor,u

class AggressiveVariant(DistalOnlyExtent):
    def __init__(self,config,arm):
        self.full=config;self.arm=arm;idx=ARMS.index(arm);self.fixed=idx>=3;self.edges=idx>=4;self.broad=idx>=5;self.geometry=idx>=6
        settings=copy.deepcopy(config['extent']);params=config['aggressive_parameters']
        if self.edges:
            settings['section']['strong_RGB_contrast']=0.
            settings['seed']['normal_min_sections']=params['edge_policy']['minimum_paired_sections']
        if self.broad:settings['color'].update(params['broad_color'])
        if self.geometry:settings['geometry'].update(params['geometry'])
        super().__init__(settings)

    def section(self,*args,**kwargs):
        section,why=BlueExtent.section(self,*args,**kwargs)
        if section is not None and self.edges:
            threshold=self.full['aggressive_parameters']['edge_policy']['original_strong_RGB_reference']
            section['states']=[('observed_strong' if contrast>=threshold else 'color_supported') if state=='observed_strong' else state for state,contrast in zip(section['states'],section['contrasts'])]
        return section,why

    def seeds(self,frame,mask,probability,prior,age):
        cfg=self.cfg['seed'];s=min(mask.shape)/400;binary=((mask==2)&(probability[2]>=cfg['P_min'])).astype(np.uint8)
        count,labels,stats,_=cv2.connectedComponentsWithStats(binary,connectivity=4);ids=[k for k in range(1,count) if stats[k,cv2.CC_STAT_AREA]>=cfg['min_pixels_at400']*s*s]
        if len(ids)>cfg['max_components']:return [],[dict(reason='excessive_seed_hypotheses',components=len(ids))]
        records=[];qualified=[];bgcfg=self.full['aggressive_parameters']['background']
        for k in sorted(ids,key=lambda k:(-stats[k,cv2.CC_STAT_AREA],k)):
            yy,xx=np.nonzero(labels==k);points=np.stack([xx,yy],-1).astype(float);sub=points[np.linspace(0,len(points)-1,min(len(points),cfg['max_points'])).astype(int)]
            origin,u,ends,width=_pca(sub);anchor=None
            if self.fixed:
                anchor,u=fixed_axis(sub,mask.shape,self.full['aggressive_parameters']['fixed_direction']);n=np.array([-u[1],u[0]]);axis=(sub-origin)@u;normal=(sub-origin)@n;ends=np.percentile(axis,[2,98]);width=max(float(np.ptp(np.percentile(normal,[2,98]))),1.)
            length=float(ends[1]-ends[0]);record=dict(label=k,pixels=len(points),length=length,width=width,reason='short_or_nonelongated_seed')
            if anchor is not None:record.update(fixed_entry_anchor=anchor,fixed_axis_proposal=u.copy())
            if length<cfg['min_length_at400']*s or length/width<cfg['min_aspect']:records.append(record);continue
            pixels=frame[sub[:,1].astype(int),sub[:,0].astype(int)];mu=np.median(chroma(pixels)[0],axis=0);n=np.array([-u[1],u[0]])
            fractions=np.linspace(*bgcfg['axial_fractions'],bgcfg['stations']);bg_centers=origin+(ends[0]+fractions*length)[:,None]*u
            bp=bg_centers[:,None,:]+np.array([-1,1])[None,:,None]*(.9*width+4*s)*n;bg_rgb,bv=sample(frame,bp);valid_fractions=np.broadcast_to(fractions[:,None],bv.shape)[bv];coverage=float(np.ptp(valid_fractions)) if len(valid_fractions)>1 else 0.
            record.update(background_valid_points=int(bv.sum()),background_side_counts=bv.sum(0),background_axial_coverage=coverage)
            if bv.sum()<bgcfg['min_points'] or coverage<bgcfg['min_axial_coverage']:record['reason']='insufficient_seed_background';records.append(record);continue
            bg_values=chroma(bg_rgb[bv])[0];bg=np.median(bg_values,axis=0);values=blue(bg_values);gap=float(blue(mu)-blue(bg));limit=cfg['background_MAD_multiplier']*max(1.4826*float(np.median(abs(values-np.median(values)))),cfg['background_MAD_floor'])
            record.update(mu=mu,background_chroma=bg,relative_blue_advantage=gap,relative_blue_required=limit)
            if gap<=limit or np.linalg.norm(mu-1/3)<cfg['min_color_vector_norm']:record['reason']='no_separable_blue_identity';records.append(record);continue
            seed=dict(core=labels==k,mu=mu,v=mu-1/3,bg=bg,gap=gap,origin=origin,u=u,width=width,length=length,label=k)
            centers=origin+(ends[0]+np.array(cfg['fractions'])*length)[:,None]*u;sections=[];failures=[]
            for center in centers:
                sec,why=self.section(frame,center,u,width,seed)
                if sec is not None:sections.append(sec)
                else:failures.append(why)
            paired=sum(not z['weak'] for z in sections);record.update(paired_sections=paired,usable_sections=len(sections),section_failures=failures)
            normal=paired>=cfg['normal_min_sections'];weak=len(sections)>=cfg['weak_min_sections'] and paired>=cfg['weak_min_paired']
            if not normal and not weak:record['reason']='insufficient_external_sections';records.append(record);continue
            if not normal and not compatible_history(prior,age,origin,u,width,mask.shape,self.cfg['history']):record['reason']='weak_seed_requires_compatible_history';records.append(record);continue
            _,edge_u,_,_=_pca(np.array([z['center'] for z in sections]))
            if np.dot(edge_u,u)<0:edge_u=-edge_u
            if angle(edge_u,u)>self.cfg['geometry']['normal_total_turn_degrees']:record['reason']='external_axis_conflict';records.append(record);continue
            normalvec=np.array([-edge_u[1],edge_u[0]]);fits=[];minpoints=self.full['aggressive_parameters']['edge_policy']['minimum_fit_points'] if self.edges else 3
            for side in range(2):
                ps=np.array([z['sides'][side] for z in sections if z['states'][side] in (['observed_strong','color_supported'] if self.edges else ['observed_strong'])])
                if len(ps)<minpoints:continue
                ts=(ps-origin)@edge_u;ys=(ps-origin)@normalvec;coeff=np.polyfit(ts,ys,1);fits.append(float(np.sqrt(np.mean((ys-np.polyval(coeff,ts))**2))))
            if not fits or max(fits)>max(cfg['fit_rms_at400']*s,cfg['fit_rms_width_fraction']*width):record['reason']='external_line_residual';records.append(record);continue
            seed.update(u=edge_u,start_sections=[sections[0],sections[-1]],width=float(np.median([z['width'] for z in sections])))
            record.update(reason='qualified',mode='normal' if normal else 'weak_history',fit_rms=fits);qualified.append(seed);records.append(record)
        if len(qualified)==1:
            if self.fixed:self.chosen=1;self.orientation='fixed_edge_to_center'
            else:self.chosen,self.orientation=direction(qualified[0],mask.shape,prior,age,self.cfg['history'])
        return qualified,records

    def tip(self,frame,mask,seed,branch,oriented_sign):
        return DistalOnlyExtent.tip(self,frame,mask,seed,branch,self.chosen)

    def grow(self,frame,mask,probability,prior=None,age=99):
        self.calls=[];self.chosen=0;self.orientation='unavailable';c,p,conf,m=BlueExtent.grow(self,frame,mask,probability,prior,age)
        m['direction_control']=dict(walk_calls=self.calls.copy(),selected_sign=self.chosen,orientation_source=self.orientation)
        if 'distal_sign' in m:m.update(distal_sign=self.chosen,orientation_source=self.orientation)
        if self.orientation=='ambiguous':
            if c.any() or p.any():raise ValueError('Ambiguous direction generated pixels')
            m['reason']='direction_ambiguous'
        return c,p,conf,m

def tip_visibility(mask,probability,candidate,meta,tolerance):
    audits=[];tips=meta['tips'];component=None
    for tip in tips:
        result=dict(raw_state=tip['state'],raw_xy=tip['xy'],valid_current_mask_support=False,xy=None,reason='no_tip_candidate')
        if tip['xy'] is not None:
            if component is None:
                selected=[x for x in meta['seeds'] if x['reason']=='qualified'];binary=((mask==2)&(probability[2]>=.9)).astype(np.uint8);_,seeds=cv2.connectedComponents(binary,connectivity=4)
                yy,xx=np.nonzero(seeds==selected[0]['label']) if len(selected)==1 else (np.array([],int),np.array([],int))
                visible=((mask==2)|(candidate>0)).astype(np.uint8);_,labels=cv2.connectedComponents(visible,connectivity=4)
                if len(yy) and labels[yy[0],xx[0]]:component=labels==labels[yy[0],xx[0]]
                else:component=np.zeros_like(mask,bool)
                distance=cv2.distanceTransform((~component).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE) if component.any() else np.full(mask.shape,np.inf)
            d,inside=sample(distance,np.asarray(tip['xy'],float));valid=bool(inside and d<=tolerance)
            result.update(valid_current_mask_support=valid,xy=tip['xy'] if valid else None,reason='root_visible_support' if valid else 'terminal_not_on_root_connected_visible_body')
        audits.append(result)
    return audits

class GrowthPipeline:
    def __init__(self,cfg,arm,t1_config=None,sequential=False):
        if arm not in ARMS:raise ValueError('Unknown arm')
        self.cfg=cfg;reference=read(ROOT/cfg['reference_distal_config'])
        if cfg['extent']!=reference['extent']:raise ValueError('Base thresholds changed outside registered factor patches')
        if arm in ['raw2','distal_only']:self.inner=DistalPipeline(reference,arm,t1_config,sequential)
        else:
            previous=read(ROOT/cfg['reference_extent_config']);self.inner=ExtentPipeline(previous,'blue_extent',t1_config,sequential);self.inner.growth=AggressiveVariant(cfg,arm)
        self.tracker=self.inner.tracker
    def reset(self):self.inner.reset()
    def run(self,frame,mask,probability,*args,**kwargs):
        c,p,conf,m=self.inner.run(frame,mask,probability,*args,**kwargs);start=time.perf_counter()
        m['tip_visibility_audit']=tip_visibility(mask,probability,c,m,self.cfg['aggressive_parameters']['tip_visibility_tolerance_at400']*min(mask.shape)/400)
        ms=(time.perf_counter()-start)*1000;m['timings']['tip_visibility_audit_ms']=ms;m['timings']['extra_total_ms']+=ms
        return c,p,conf,m
