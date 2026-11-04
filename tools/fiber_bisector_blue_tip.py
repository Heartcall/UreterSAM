"""Research-only visible blue endpoint and explicit elliptic cap; union immutable."""
from __future__ import annotations
import math,time,copy
import cv2,numpy as np
from .fiber_reference_workspace import ROOT,read
from .fiber_blue_outline_repaired import GrowthPipeline as PreviousPipeline,RepairedOutline
from .fiber_blue_outline import LAYERS as OLD_LAYERS,root_instance,natural_values,boundary_distance
from .fiber_fast_observation import sample
from .fiber_connected_blue_growth import chroma
from .fiber_connected_blue_union import rooted
from .fiber_local_dark_evidence import public,nearest

ARMS=('connected_blue_union','contour_sides_cap','bisector_guided_blue_tip')
LAYERS=(*OLD_LAYERS,'distal_cluster','side_fill','cap_fill','union_fiber')

def unit(x):
    x=np.asarray(x,float);return x/max(float(np.linalg.norm(x)),1e-12)

def direction(info,cfg):
    c=info['context'];u=np.asarray(c['axis']);n=np.array([-u[1],u[0]]);models=info['side_models'];join=c['join_t']
    if len(models)!=2 or not all(m.get('valid') for m in models):return None,dict(reason='insufficient_sides')
    ds=[float(natural_values(m['knots'],m['values'],join,True)) for m in models];ts=[unit(u+d*n) for d in ds];angle=float(np.degrees(np.arccos(np.clip(np.dot(*ts),-1,1))));v=unit(ts[0]+ts[1]);mode='angle_bisector'
    if angle<=cfg['parallel_degrees'] or any(m['mode']=='spline' for m in models):
        p=np.asarray(c['profile']);f=cfg['local_centerline_axial_fraction'];a=c['min_t'];length=c['max_t']-a;z=p[(p[:,0]>=a+f[0]*length)&(p[:,0]<=a+f[1]*length)&(p[:,3]>0)&(p[:,4]>0)]
        if len(z)<cfg['minimum_centerline_points']:return None,dict(reason='insufficient_local_centerline',side_angle_degrees=angle)
        x=z[:,0];y=(z[:,1]+z[:,2])/2;A=np.c_[x-x.mean(),np.ones(len(x))];beta=np.linalg.lstsq(A,y,rcond=None)[0]
        for _ in range(12):
            residual=y-A@beta;w=np.minimum(1,1.5*c['scale']/np.maximum(abs(residual),1e-9));beta=np.linalg.lstsq(A*w[:,None]**.5,y*w**.5,rcond=None)[0]
        v=unit(u+beta[0]*n);mode='local_centerline_curved_or_parallel'
    if np.dot(v,u)<0:v=-v
    deviation=float(np.degrees(np.arccos(np.clip(np.dot(v,u),-1,1))))
    return (None if deviation>cfg['maximum_axis_deviation_degrees'] else v),dict(reason='direction_qualified' if deviation<=cfg['maximum_axis_deviation_degrees'] else 'axis_direction_uncertain',mode=mode,side_angle_degrees=angle,axis_deviation_degrees=deviation,side_unit_tangents=ts,direction=v)

def distal_cluster(body,origin,u,cfg,s):
    yy,xx=np.nonzero(body);pts=np.stack([xx,yy],-1);t=(pts-origin)@u;empty=np.zeros_like(body);meta=dict(reason='empty_blue',candidate_xy=None)
    if not len(t):return empty,meta
    end=float(t.max());band=np.zeros_like(body);keep=t>=end-cfg['depth_at400']*s;band[yy[keep],xx[keep]]=1;qualified=[]
    # Explicit connectivity argument: OpenCV's second positional argument is labels.
    _,labels,stats,_=cv2.connectedComponentsWithStats(band,connectivity=4)
    for k in range(1,len(stats)):
        by,bx=np.nonzero(labels==k);q=np.stack([bx,by],-1);qt=(q-origin)@u
        if len(q)<cfg['minimum_area_at400']*s*s or np.ptp(qt)<cfg['minimum_axial_span_at400']*s:continue
        if ((qt>=end-cfg['distal_band_at400']*s).sum()<cfg['minimum_distal_pixels_at400']*s*s):continue
        if qt.max()<end-1e-7:continue
        qualified.append((k,q,qt))
    if len(qualified)!=1:return empty,dict(meta,reason='unsupported_or_ambiguous_farthest_cluster',blue_front_t=end,qualifying_clusters=len(qualified))
    k,q,qt=qualified[0];cluster=(labels==k).astype(np.uint8)
    if end-np.quantile(qt,.90)>cfg['maximum_spike_beyond_q90_at400']*s:return cluster,dict(meta,reason='single_extreme_spike',blue_front_t=end)
    front=q[qt>=end-cfg['frontier_band_at400']*s];median=np.median(front,axis=0);xy=front[np.argmin(np.sum((front-median)**2,axis=1))].astype(float)
    return cluster,dict(reason='supported_farthest_blue_cluster',candidate_xy=xy,blue_front_t=end,area=len(q),span_px=float(np.ptp(qt)),frontier_points=len(front),cluster_centroid=np.mean(q,axis=0))

def visibility(frame,raw,xy,u,observer,seed,cfg,s):
    n=np.array([-u[1],u[0]]);off=np.asarray(cfg['normal_offsets_at400'])*s;inside=xy+np.asarray(cfg['inside_offsets_at400'])[:,None,None]*s*u+off[None,:,None]*n;outside=xy+np.asarray(cfg['outside_offsets_at400'])[:,None,None]*s*u+off[None,:,None]*n
    a,av=sample(frame,inside);b,bv=sample(frame,outside);labels,lv=nearest(raw,outside);margin=min(xy[0],xy[1],raw.shape[1]-1-xy[0],raw.shape[0]-1-xy[1]);meta=dict(state='unknown',reason='terminal_RGB_insufficient')
    if margin<cfg['minimum_frame_margin_at400']*s or not (av.all() and bv.all()):return dict(meta,state='out_of_view',reason='terminal_out_of_view')
    if np.any(lv&(labels==1)):return dict(meta,state='occlusion_suspected',reason='current_stone_contact')
    ai=observer.color(a,seed);bo=observer.color(b,seed);unknown=float(np.r_[chroma(a.reshape(-1,3))[1],chroma(b.reshape(-1,3))[1]].__gt__(observer.cfg['color']['max_noise']).mean());contrast=float(np.median(np.linalg.norm(a.mean(0)-b.mean(0),axis=-1)/(255*math.sqrt(3))))
    meta.update(inside_blue_fraction=float(ai.mean()),outside_blue_fraction=float(bo.mean()),unknown_fraction=unknown,RGB_contrast=contrast)
    if bo.mean()>cfg['maximum_outside_blue_fraction']:return dict(meta,state='internal_transition',reason='blue_continues')
    if unknown>cfg['maximum_unknown_fraction']:return dict(meta,reason='terminal_background_unknown')
    if ai.mean()<cfg['minimum_inside_blue_fraction'] or contrast<cfg['minimum_RGB_contrast']:return meta
    return dict(meta,state='current_RGB_confirmed',reason='current_visible_blue_terminal')

class BisectorFitter:
    def __init__(self,cfg):self.cfg=cfg;self.q=cfg['bisector_tip'];self.side=RepairedOutline(read(ROOT/'configs/blue_outline_control_20261102_v2.json'))
    def run(self,frame,raw,p,body,core,meta,union):
        _,evidence,sideprop,_,info=self.side.run(frame,raw,p,body,core,meta,'contour_sides');info=copy.deepcopy(info);info.update(stage=ARMS[-1],sides_applied=False,reason='unchanged_union',final_tip_xy=None);info['cap']=dict(attempted=False,accepted=False,observable=False,state='unknown',reason='tip_not_confirmed',curve=[],candidate_xy=None,accuracy=None);empty=np.zeros_like(body);cluster=empty.copy();sidefill=empty.copy();capfill=empty.copy();capprop=body.copy();u,d=direction(info,self.q['direction']);info['bisector']=d
        if u is None:return union,evidence,cluster,sidefill,capfill,sideprop,capprop,public(info)
        c=info['context'];o=np.asarray(c['origin']);s=c['scale'];n=np.array([-u[1],u[0]]);cluster,front=distal_cluster(body,o,u,self.q['cluster'],s);info['distal']=front
        if front['candidate_xy'] is None:return union,evidence,cluster,sidefill,capfill,sideprop,capprop,public(info)
        q=next(z for z in meta['seeds'] if z['reason']=='qualified');mu=np.asarray(q['mu'],np.float32);seed=dict(mu=mu,v=mu-1/3,bg=np.asarray(q['background_chroma'],np.float32),gap=q['relative_blue_advantage'])
        xy=np.asarray(front['candidate_xy']);state=visibility(frame,raw,xy,u,self.side.relaxed,seed,self.q['visibility'],s);info['tip_observation']=state;info['candidate_tip_xy']=xy;info['cap'].update(state=state['state'],reason=state['reason'],candidate_xy=xy,observable=state['state']=='current_RGB_confirmed')
        if state['state']!='current_RGB_confirmed':return union,evidence,cluster,sidefill,capfill,sideprop,capprop,public(info)
        info['final_tip_xy']=xy;info['cap']['attempted']=True;curves=[np.asarray(z,float) for z in info['side_curves']];projected=[]
        for z in curves:
            t=(z-o)@u;v=(z-o)@n;order=np.argsort(t);projected.append((t[order],v[order]))
        end=front['blue_front_t'];joinbase=min(z[0].max() for z in projected);cfg=self.q['cap'];candidates=[];bd=boundary_distance(body);yy,xx=np.indices(raw.shape);tt=(xx-o[0])*u[0]+(yy-o[1])*u[1];vv=(xx-o[0])*n[0]+(yy-o[1])*n[1]
        # Fitting observations use actual RGB-body extrema in the new coordinate.
        from .fiber_blue_outline import profiles
        pr,_,_=profiles(body,o,u,self.cfg['outline']['profile_step_at400']*s,self.cfg['outline']['side_frame_margin_at400']*s)
        for retreat in cfg['join_retreat_at400']:
            join=joinbase-retreat*s
            if not all(z[0].min()<=join<=z[0].max() for z in projected):continue
            sides=[float(np.interp(join,*z)) for z in projected];derivatives=[float(np.interp(join,z[0],np.gradient(z[1],z[0]))) for z in projected];half=(sides[1]-sides[0])/2;center=sum(sides)/2;length=end-join
            if half<=0 or length<=0:continue
            slope=(float((xy-o)@n)-center)/length;tail=pr[pr[:,0]>=join]
            if len(tail)<4:continue
            for r in cfg['r_candidates']:
                a=length/(1-r);ct=join-a*r;b=half/math.sqrt(1-r*r);theta=np.linspace(-math.acos(r),math.acos(r),cfg['curve_samples']);t=ct+a*np.cos(theta);v=center+slope*(t-join)+b*np.sin(theta);curve=o+t[:,None]*u+v[:,None]*n;w=b*np.sqrt(np.clip(1-((tail[:,0]-ct)/a)**2,0,1));cy=center+slope*(tail[:,0]-join);residual=np.r_[tail[:,1]-(cy-w),tail[:,2]-(cy+w)];rms=float(np.sqrt(np.mean(residual**2)));wp=-half*r/(length*(1+r));jumps=np.degrees(abs(np.arctan(derivatives)-np.arctan([slope-wp,slope+wp])));relative=v-center-slope*(t-join);normals=((t-ct)/a**2-slope*relative/b**2)[:,None]*u+(relative/b**2)[:,None]*n;normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-9);inside,iv=sample(frame,curve-1.5*s*normals);outside,ov=sample(frame,curve+1.5*s*normals);dist,dv=sample(bd,curve);contrast=np.linalg.norm(inside-outside,axis=-1)/(255*math.sqrt(3));valid=iv&ov&dv;unknown=(chroma(inside)[1]>self.side.relaxed.cfg['color']['max_noise'])|(chroma(outside)[1]>self.side.relaxed.cfg['color']['max_noise']);supported=valid&~unknown&self.side.relaxed.color(inside,seed)&((~self.side.relaxed.color(outside,seed))|(contrast>=self.q['visibility']['minimum_RGB_contrast']))&(dist<=cfg['maximum_boundary_distance_at400']*s)
                row=dict(r=r,join_retreat=retreat,tip_t=end,join=join,a=a,b=b,center_t=ct,center_n=center,slope=slope,RMS=rms,score=float(np.median(abs(residual))),tangent_jumps=jumps,curve=curve,candidate_xy=xy,apex_xy=o+end*u+(center+slope*length)*n,RGB_supported_fraction=float(supported.mean()),unknown_fraction=float(unknown.mean()),accepted=False,state='unknown',reason='ellipse_fit_residual',observable=True,attempted=True,accuracy=None)
                if rms>cfg['maximum_RMS_at400']*s:candidates.append(row);continue
                if max(jumps)>cfg['maximum_model_join_angle_degrees']:row['reason']='model_join_tangent_mismatch';candidates.append(row);continue
                if supported.mean()<cfg['minimum_current_RGB_support'] or unknown.mean()>self.q['visibility']['maximum_unknown_fraction']:row['reason']='insufficient_current_RGB_arc';candidates.append(row);continue
                mask=(tt>=join)&(tt<=end)&(abs(vv-center-slope*(tt-join))<=b*np.sqrt(np.clip(1-((tt-ct)/a)**2,0,1)));addition=(evidence>0)&mask&(union==0);fill=(evidence>0)&(sideprop>0)&(tt<join)&(union==0);reach,_=rooted(((body>0)|addition|fill),(body>0)&core,1);addition&=reach>0;fill&=reach>0;processed=((body>0)|(union==2)|addition|fill).astype(np.uint8);ad,av=sample(boundary_distance(processed),curve);match=float((av&(ad<=cfg['maximum_boundary_distance_at400']*s)).mean());row['actual_arc_match_fraction']=match
                if match<cfg['minimum_actual_arc_match']:row['reason']='protected_raster_not_following_arc';candidates.append(row);continue
                row.update(accepted=True,state='RGB_supported_ellipse_candidate',reason='current_cluster_anchored_ellipse',cap_mask=addition,side_mask=fill);candidates.append(row)
        if candidates:
            feasible=[z for z in candidates if z['accepted']];chosen=min(feasible or candidates,key=lambda z:(z['score'],z['RMS'],z['join_retreat'],z['r']));accepted=chosen['accepted'];sidefill=chosen.pop('side_mask',empty.copy()).astype(np.uint8);capfill=chosen.pop('cap_mask',empty.copy()).astype(np.uint8);info['cap']=chosen;info['cap']['candidate_count']=len(candidates);info['cap']['feasible_count']=len(feasible);info['cap_coordinate_context']=dict(origin=o,axis=u,join_t=chosen['join'],scale=s,min_t=float(tt[body>0].min()),max_t=end);capprop=(capfill|(body>0)).astype(np.uint8)
            if accepted:union=union.copy();union[(sidefill>0)|(capfill>0)]=2;info.update(sides_applied=True,reason='cluster_anchored_visible_cap')
        else:info['cap']['reason']='no_observed_side_join'
        return union,evidence,cluster,sidefill,capfill,sideprop,capprop,public(info)

class GrowthPipeline:
    def __init__(self,cfg,arm,t1_config=None,sequential=False):
        if arm not in ARMS:raise ValueError(arm)
        self.arm=arm;self.cfg=cfg;self.inner=PreviousPipeline(read(ROOT/'configs/blue_outline_control_20261102_v2.json'),'connected_blue_union' if arm==ARMS[-1] else arm,t1_config,sequential);self.tracker=self.inner.tracker;self.fitter=BisectorFitter(cfg);self.layers={};self.outline={};self.final_mask=None
    def reset(self):self.inner.reset();self.layers={};self.outline={};self.final_mask=None
    def run(self,frame,raw,p,*args,**kwargs):
        c,proposal,conflict,meta=self.inner.run(frame,raw,p,*args,**kwargs);start=time.perf_counter();self.layers={k:v.copy() for k,v in self.inner.layers.items()};self.outline=copy.deepcopy(self.inner.outline);self.final_mask=self.inner.final_mask.copy();B=self.layers['blue_body'];empty=np.zeros_like(B)
        self.layers.update(distal_cluster=empty.copy(),side_fill=empty.copy(),cap_fill=empty.copy(),union_fiber=((raw==2)|((B>0)&(raw==0)&(not meta['abnormal']))).astype(np.uint8))
        if self.arm==ARMS[-1] and self.outline['eligible']:
            before=self.final_mask.copy();core,_=root_instance(raw,p,meta);final,E,cluster,sf,cf,sp,cp,info=self.fitter.run(frame,raw,p,B,core,meta,before);self.final_mask=final;self.outline=info;self.layers.update(rgb_evidence=E,processed_body=((self.layers['root_instance']>0)|(B>0)|(sf>0)|(cf>0)).astype(np.uint8),side_proposal=sp,cap_proposal=cp,distal_cluster=cluster,side_fill=sf,cap_fill=cf)
            if np.any(final[before==2]!=2) or np.any(final[B>0]!=2):raise ValueError('Union/blue preservation violated')
        add=((raw==0)&(self.final_mask==2)).astype(np.uint8);self.layers['edit_add']=add;self.layers['edit_remove']=((raw==2)&(self.final_mask!=2)).astype(np.uint8)
        if np.any(self.final_mask[raw==1]!=1):raise ValueError('Stone overwritten')
        extra=(time.perf_counter()-start)*1000;meta['timings']['bisector_stage_ms']=extra if self.arm==ARMS[-1] else 0.;meta['timings']['outline_ms']+=extra;meta['timings']['extra_total_ms']+=extra
        return add,proposal,conflict,meta
