"""Experimental RGB-body replacement, robust separate sides and explicit ellipse cap."""
from __future__ import annotations
import copy,math,time
import cv2,numpy as np
from .fiber_connected_blue_union import GrowthPipeline as UnionPipeline,rooted
from .fiber_connected_blue_growth import _pca,chroma
from .fiber_blue_extent import BlueExtent
from .fiber_fast_observation import sample
from .fiber_local_dark_evidence import public,nearest
from .fiber_reference_workspace import ROOT,read

ARMS=('connected_blue_union','blue_body_only','contour_sides','contour_sides_cap')
LAYERS=('rgb_eligible','blue_body','rgb_evidence','processed_body','side_proposal','cap_proposal','root_instance','edit_add','edit_remove')

def root_instance(raw,probability,meta):
    selected=[s for s in meta['seeds'] if s['reason']=='qualified'];empty=np.zeros_like(raw,bool)
    if len(selected)!=1:return empty,empty.copy()
    _,labels=cv2.connectedComponents(((raw==2)&(probability[2]>=.9)).astype(np.uint8),connectivity=4);core=labels==selected[0]['label']
    yy,xx=np.nonzero(core)
    if not len(yy):return empty,empty.copy()
    _,groups=cv2.connectedComponents((raw==2).astype(np.uint8),connectivity=4);return core,groups==groups[yy[0],xx[0]]

def natural_values(knots,values,x,derivative=False):
    knots=np.asarray(knots,float);values=np.asarray(values,float);x=np.asarray(x,float);k=len(knots);h=np.diff(knots)
    if k==2:
        slope=(values[1]-values[0])/h[0]
        return np.full(x.shape,slope) if derivative else values[0]+(x-knots[0])*slope
    mat=np.eye(k);rhs=np.zeros(k)
    for j in range(1,k-1):mat[j,j-1:j+2]=[h[j-1],2*(h[j-1]+h[j]),h[j]];rhs[j]=6*((values[j+1]-values[j])/h[j]-(values[j]-values[j-1])/h[j-1])
    second=np.linalg.solve(mat,rhs);i=np.clip(np.searchsorted(knots,x,side='right')-1,0,k-2);width=h[i];a=(knots[i+1]-x)/width;b=(x-knots[i])/width
    if derivative:return (values[i+1]-values[i])/width+width*((1-3*a*a)*second[i]+(3*b*b-1)*second[i+1])/6
    return a*values[i]+b*values[i+1]+((a*a*a-a)*second[i]+(b*b*b-b)*second[i+1])*width*width/6

def hat_matrix(x,knots):
    i=np.clip(np.searchsorted(knots,x,side='right')-1,0,len(knots)-2);z=(x-knots[i])/(knots[i+1]-knots[i]);a=np.zeros((len(x),len(knots)));a[np.arange(len(x)),i]=1-z;a[np.arange(len(x)),i+1]=z;return a

def robust_model(t,y,weights,cfg,s):
    def solve(knots,penalty):
        a=np.stack([natural_values(knots,np.eye(len(knots))[j],t) for j in range(len(knots))],1);d=np.diff(np.eye(len(knots)),2,axis=0);reg=penalty*d.T@d+cfg['ridge']*np.eye(len(knots));w=weights.copy();beta=np.linalg.solve(a.T@(w[:,None]*a)+reg,a.T@(w*y))
        for _ in range(cfg['IRLS_iterations']):
            residual=y-natural_values(knots,beta,t);w=weights*np.minimum(1,cfg['Huber_delta_at400']*s/np.maximum(abs(residual),1e-9));beta=np.linalg.solve(a.T@(w[:,None]*a)+reg,a.T@(w*y))
        residual=y-natural_values(knots,beta,t);return dict(knots=knots,values=beta,median_abs_residual=float(np.median(abs(residual))),RMS=float(np.sqrt(np.mean(residual**2))))
    line=solve(np.array([t.min(),t.max()]),0.);curve=solve(np.linspace(t.min(),t.max(),cfg['spline_knots']),cfg['spline_D2_penalty']);deviation=float(np.ptp(natural_values(curve['knots'],curve['values'],t)-natural_values(line['knots'],line['values'],t)));improvement=(line['median_abs_residual']-curve['median_abs_residual'])/max(line['median_abs_residual'],1e-9)
    mode='spline' if improvement>=cfg['spline_required_residual_improvement'] and deviation>=cfg['spline_required_deviation_at400']*s else 'line';chosen=curve if mode=='spline' else line
    return dict(**chosen,mode=mode,line_residual=line['median_abs_residual'],spline_residual=curve['median_abs_residual'],curve_deviation=deviation,residual_improvement=improvement,observed_min=float(t.min()),observed_max=float(t.max()),valid=True)

def profiles(body,origin,u,step,frame_margin=0):
    yy,xx=np.nonzero(body);points=np.stack([xx,yy],-1).astype(float);n=np.array([-u[1],u[0]]);t=(points-origin)@u;v=(points-origin)@n;lo=float(t.min());idx=np.floor((t-lo)/step).astype(int);rows=[]
    for k in np.unique(idx):
        which=np.flatnonzero(idx==k);z=which[np.argmin(v[which])];w=which[np.argmax(v[which])];a,b=points[z],points[w];dist=lambda p:min(p[0],p[1],body.shape[1]-1-p[0],body.shape[0]-1-p[1]);rows.append([float(np.mean(t[which])),float(v[z]),float(v[w]),float(dist(a)>=frame_margin),float(dist(b)>=frame_margin)])
    return np.asarray(rows),float(t.min()),float(t.max())

def boundary_distance(binary):
    boundary=(binary>0)&~(cv2.erode(binary.astype(np.uint8),np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT)>0)
    return cv2.distanceTransform((~boundary).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)

def retained(body,reference,core,u,cfg,s):
    result,_=rooted(body,core,1);total=max(int(reference.sum()),1);fraction=float(((result>0)&(reference>0)).sum()/total)
    yy,xx=np.nonzero(reference);t=np.stack([xx,yy],-1)@u;ry,rx=np.nonzero(result)
    loss=float(t.max()-(np.stack([rx,ry],-1)@u).max()) if len(ry) else float(np.ptp(t)+cfg['maximum_length_loss_at400']*s)
    return result,fraction,loss,bool(fraction>=cfg['minimum_blue_fraction'] and loss<=cfg['maximum_length_loss_at400']*s)

class Outline:
    def __init__(self,config):
        self.config=config;self.cfg=config['outline'];reference=read(ROOT/config['reference_aggressive_config']);base=BlueExtent({**reference['extent'],'color':{**reference['extent']['color'],**reference['aggressive_parameters']['broad_color']}});settings=copy.deepcopy(base.cfg);settings['color']['background_distance_margin_normal']=self.cfg['evidence']['relaxed_background_distance_margin'];self.relaxed=BlueExtent(settings)

    def evidence(self,frame,raw,body,core,meta,s):
        q=next(z for z in meta['seeds'] if z['reason']=='qualified');mu=np.asarray(q['mu'],np.float32);seed=dict(mu=mu,v=mu-1/3,bg=np.asarray(q['background_chroma'],np.float32),gap=q['relative_blue_advantage']);distance=cv2.distanceTransform((body==0).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE);band=(distance<=self.cfg['evidence']['band_distance_at400']*s)&(body==0)&(raw!=1)
        image=frame.astype(np.float32)/255.;gx=cv2.Sobel(image,cv2.CV_32F,1,0,ksize=3)/8;gy=cv2.Sobel(image,cv2.CV_32F,0,1,ksize=3)/8;gradient=np.sqrt((gx*gx+gy*gy).sum(-1)/3)
        eligible=body.astype(bool).copy();eligible[band]=self.relaxed.color(frame[band],seed)&(gradient[band]>=self.cfg['evidence']['minimum_RGB_gradient']);eligible[raw==1]=False;result,_=rooted(eligible,core,1);return result,seed

    def cap(self,frame,raw,body,sides,models,context,evidence,seed,core):
        cfg=self.cfg['cap'];s=context['scale'];o=np.asarray(context['origin']);u=np.asarray(context['axis']);n=np.array([-u[1],u[0]]);p=np.asarray(context['profile']);join=context['join_t'];end=context['max_t'];answer=dict(attempted=False,accepted=False,state='unknown',reason='side_join_not_observed',curve=[],candidate_xy=None,accuracy=None,observable=False)
        if not all(m['valid'] and m['observed_min']<=join<=m['observed_max']+1e-8 for m in models):return sides,body.copy(),answer
        left,right=[float(natural_values(m['knots'],m['values'],join)) for m in models];dl,dr=[float(natural_values(m['knots'],m['values'],join,True)) for m in models];half=(right-left)/2;center=(left+right)/2;slope=(dl+dr)/2
        if half<=0:answer['reason']='crossing_side_join';return sides,body.copy(),answer
        yy,xx=np.nonzero(body);points=np.stack([xx,yy],-1);ts=(points-o)@u;cappts=points[ts>=end-s];front=np.median(cappts,axis=0);future=front+np.asarray(cfg['future_offsets_at400'])[:,None]*s*u;rgb,valid=sample(frame,future);labels,lv=nearest(raw,future)
        if min(front[0],front[1],raw.shape[1]-1-front[0],raw.shape[0]-1-front[1])<cfg['minimum_frame_margin_at400']*s or not valid.all():answer.update(reason='terminal_out_of_view',state='out_of_view');return sides,body.copy(),answer
        if np.any(lv&(labels==1)):answer.update(reason='current_stone_contact',state='occlusion_suspected');return sides,body.copy(),answer
        if self.relaxed.color(rgb,seed).any():answer.update(reason='blue_continues_after_frontier',state='internal_transition');return sides,body.copy(),answer
        if np.any(chroma(rgb)[1]>self.relaxed.cfg['color']['max_noise']):answer['reason']='terminal_background_unknown';return sides,body.copy(),answer
        answer['observable']=True;tail=p[p[:,0]>=join]
        if len(tail)*2<cfg['minimum_tail_points']:answer['reason']='insufficient_visible_tail';return sides,body.copy(),answer
        answer['attempted']=True;candidates=[]
        for r in cfg['r_candidates']:
            for retreat in cfg['front_retreat_at400']:
                tip=end-retreat*s;d=tip-join
                if d<=0:continue
                a=d/(1-r);ct=join-a*r;b=half/math.sqrt(1-r*r);z=(tail[:,0]-ct)/a;w=b*np.sqrt(np.clip(1-z*z,0,1));cy=center+slope*(tail[:,0]-join);residual=np.r_[tail[:,1]-(cy-w),tail[:,2]-(cy+w)];rms=float(np.sqrt(np.mean(residual**2)));score=float(np.median(abs(residual)));wp=-half*r/(d*(1+r));jumps=np.degrees(np.abs(np.arctan([dl,dr])-np.arctan([slope-wp,slope+wp])));theta=np.linspace(-math.acos(r),math.acos(r),cfg['curve_samples']);t=ct+a*np.cos(theta);v=center+slope*(t-join)+b*np.sin(theta);curve=o+t[:,None]*u+v[:,None]*n;candidates.append(dict(r=r,retreat=retreat,tip_t=tip,a=a,b=b,center_t=ct,center_n=center,slope=slope,half=half,join=join,RMS=rms,score=score,tangent_jumps=jumps,curve=curve))
        if not candidates:answer['reason']='no_ellipse_candidate';return sides,body.copy(),answer
        chosen=min(candidates,key=lambda z:(z['score'],z['RMS'],z['retreat'],z['r']));answer.update(chosen,curve=chosen['curve'],candidate_xy=o+chosen['tip_t']*u+(center+slope*(chosen['tip_t']-join))*n,candidate_count=len(candidates),reason='ellipse_fit_residual')
        if chosen['RMS']>cfg['maximum_fit_RMS_at400']*s:return sides,body.copy(),answer
        if max(chosen['tangent_jumps'])>cfg['maximum_tangent_jump_degrees']:answer['reason']='join_tangent_mismatch';return sides,body.copy(),answer
        curve=chosen['curve'];tt=(curve-o)@u;vv=(curve-o)@n;relative=vv-center-slope*(tt-join);dt=(tt-chosen['center_t'])/chosen['a']**2-slope*relative/chosen['b']**2;dn=relative/chosen['b']**2;normal=dt[:,None]*u+dn[:,None]*n;normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-9)
        inside,iv=sample(frame,curve-cfg['edge_sample_offset_at400']*s*normal);outside,ov=sample(frame,curve+cfg['edge_sample_offset_at400']*s*normal);contrast=np.linalg.norm(inside-outside,axis=-1)/(255*math.sqrt(3));dist,dv=sample(boundary_distance(body),curve);unknown=~(iv&ov&dv)|(chroma(inside)[1]>self.relaxed.cfg['color']['max_noise'])|(chroma(outside)[1]>self.relaxed.cfg['color']['max_noise']);supported=(~unknown)&self.relaxed.color(inside,seed)&((~self.relaxed.color(outside,seed))|(contrast>=cfg['minimum_RGB_contrast']))&(dist<=cfg['maximum_boundary_distance_at400']*s);answer.update(RGB_supported_fraction=float(supported.mean()),unknown_fraction=float(unknown.mean()),observed_arc_match_fraction=float((dv&(dist<=cfg['maximum_boundary_distance_at400']*s)).mean()))
        if supported.mean()<cfg['minimum_RGB_supported_fraction'] or unknown.mean()>cfg['maximum_unknown_fraction']:answer['reason']='insufficient_current_RGB_arc';return sides,body.copy(),answer
        yy,xx=np.indices(raw.shape);t=(xx-o[0])*u[0]+(yy-o[1])*u[1];v=(xx-o[0])*n[0]+(yy-o[1])*n[1];z=(t-chosen['center_t'])/chosen['a'];width=chosen['b']*np.sqrt(np.clip(1-z*z,0,1));in_ellipse=(t<=chosen['tip_t'])&(np.abs(v-center-slope*(t-join))<=width);proposal=sides.copy();tailregion=t>=join;proposal[tailregion]=in_ellipse[tailregion].astype(np.uint8);processed=(proposal>0)&(evidence>0);processed[raw==1]=0;processed,_=rooted(processed,core,1);oldcap=(body>0)&tailregion;retention=float(((processed>0)&oldcap).sum()/max(int(oldcap.sum()),1));answer['cap_blue_retention']=retention
        if retention<cfg['minimum_cap_blue_retention']:answer['reason']='cap_visible_pixels_lost';return sides,proposal,answer
        actualdist=boundary_distance(processed);ad,av=sample(actualdist,curve);match=float((av&(ad<=cfg['maximum_boundary_distance_at400']*s)).mean());answer['actual_arc_match_fraction']=match
        if match<self.config['geometry_quality']['minimum_actual_arc_match_fraction']:answer['reason']='raster_cap_not_following_arc';return sides,proposal,answer
        processed,ret,loss,ok=retained(processed,body,core,u,self.cfg['retention'],s);answer.update(blue_retention=ret,length_loss_px=loss)
        if not ok:answer['reason']='cap_visibility_retention_fallback';return sides,proposal,answer
        answer.update(accepted=True,state='RGB_supported_ellipse_candidate',reason='current_visible_ellipse_arc');return processed,proposal,answer

    def run(self,frame,raw,probability,body,core,meta,arm):
        start=time.perf_counter();s=min(raw.shape)/400;yy,xx=np.nonzero(body);points=np.stack([xx,yy],-1).astype(float);o,u,_,_=_pca(points)
        if np.dot(u,np.asarray(meta['seed_axis']))<0:u=-u
        n=np.array([-u[1],u[0]]);p,tmin,tmax=profiles(body,o,u,self.cfg['profile_step_at400']*s,self.cfg['side_frame_margin_at400']*s);length=tmax-tmin;region=p[(p[:,0]>=tmin+.65*length)&(p[:,0]<=tmin+.80*length)];width=float(np.median(region[:,2]-region[:,1]+1)) if len(region) else float(np.median(p[:,2]-p[:,1]+1));caplength=min(max(self.cfg['cap_length_width_fraction']*width,self.cfg['cap_length_min_at400']*s),self.cfg['cap_length_max_length_fraction']*length);jointarget=tmax-caplength;joint=p[(p[:,0]<=jointarget)&(p[:,3]>0)&(p[:,4]>0)];join=float(joint[-1,0]) if len(joint) else jointarget
        context=dict(origin=o,axis=u,profile=p,min_t=tmin,max_t=tmax,join_t=join,scale=s,visible_length=length,cap_length=tmax-join);evidence,seed=self.evidence(frame,raw,body,core,meta,s);models=[];curves=[]
        for side in range(2):
            rows=p[(p[:,0]<=join)&(p[:,3+side]>0)]
            if len(rows)<self.cfg['minimum_side_points'] or np.ptp(rows[:,0])<self.cfg['minimum_side_span_fraction']*length:models.append(dict(valid=False,mode='unchanged',reason='insufficient_visible_side'));curves.append([]);continue
            xy=o+rows[:,0,None]*u+rows[:,1+side,None]*n;a,av=sample(frame,xy-s*n);b,bv=sample(frame,xy+s*n);contrast=np.where(av&bv,np.linalg.norm(a-b,axis=-1)/(255*math.sqrt(3)),0);weights=self.cfg['minimum_edge_weight']+np.clip(contrast/self.cfg['edge_weight_scale'],0,1);model=robust_model(rows[:,0],rows[:,1+side],weights,self.cfg,s);models.append(model);tt=np.linspace(rows[0,0],rows[-1,0],max(2,int(math.ceil(np.ptp(rows[:,0])/s))));curves.append(o+tt[:,None]*u+natural_values(model['knots'],model['values'],tt)[:,None]*n)
        yy,xx=np.nonzero(evidence);ep=np.stack([xx,yy],-1);t=(ep-o)@u;v=(ep-o)@n;member=body[yy,xx]>0;keep=member.copy();safes=[];preds=[]
        for side,model in enumerate(models):
            pred=natural_values(model['knots'],model['values'],t) if model['valid'] else np.interp(t,p[:,0],p[:,1+side]);observed=np.interp(t,p[:,0],p[:,1+side]);safe=(t<=join)&(t>=model.get('observed_min',math.inf))&(t<=model.get('observed_max',-math.inf))&(abs(pred-observed)<=self.cfg['maximum_local_fit_displacement_at400']*s);keep[member&safe&((v<pred) if side==0 else (v>pred))]=False;safes.append(safe);preds.append(pred)
        add=(~member)&safes[0]&safes[1]&(v>=preds[0])&(v<=preds[1]);keep|=add;proposal=np.zeros_like(body);proposal[yy[keep],xx[keep]]=1;processed,ret,loss,ok=retained(proposal,body,core,u,self.cfg['retention'],s);applied=bool(any(m['valid'] for m in models) and ok);reason='fitted_sides' if applied else ('side_visibility_retention_fallback' if not ok else 'no_side_fit');processed=processed if applied else body.copy();sides=processed.copy();capinfo=dict(attempted=False,accepted=False,state='unknown',reason='cap_not_requested',curve=[],candidate_xy=None,observable=False,accuracy=None);capproposal=processed.copy()
        if arm=='contour_sides_cap':processed,capproposal,capinfo=self.cap(frame,raw,body,sides,models,context,evidence,seed,core)
        info=dict(eligible=True,stage=arm,reason=reason,sides_applied=applied,side_models=models,side_curves=curves,side_local_fallback_counts=[int(((t<=join)&~v).sum()) for v in safes],context=context,side_blue_retention=ret,side_length_loss_px=loss,cap=capinfo,timing_ms=(time.perf_counter()-start)*1000)
        return processed,evidence,proposal,capproposal,public(info)

class GrowthPipeline:
    def __init__(self,cfg,arm,t1_config=None,sequential=False):
        if arm not in ARMS:raise ValueError('Unknown arm')
        self.cfg=cfg;self.arm=arm;self.inner=UnionPipeline(read(ROOT/cfg['reference_union_config']),'connected_blue_union',t1_config,sequential);self.tracker=self.inner.tracker;self.fitter=Outline(cfg);self.layers={};self.final_mask=None;self.outline={}
    def reset(self):self.inner.reset();self.layers={};self.final_mask=None;self.outline={}
    def run(self,frame,raw,probability,*args,**kwargs):
        c,p,conf,m=self.inner.run(frame,raw,probability,*args,**kwargs);start=time.perf_counter();source=self.inner.layers;body=source['blue_body'].copy();core,root=root_instance(raw,probability,m);eligible=bool(body.any() and not m['abnormal'] and m.get('root_components')==1 and core.any());evidence=body.copy();sideproposal=body.copy();capproposal=body.copy();processed=body.copy();info=dict(eligible=eligible,stage=self.arm,sides_applied=False,side_curves=[[],[]],side_models=[],cap=dict(attempted=False,accepted=False,observable=False,state='unknown',reason='control',curve=[],candidate_xy=None,accuracy=None),reason='control',timing_ms=0.)
        final=raw.copy()
        if self.arm=='connected_blue_union':final[(raw==0)&(c>0)]=2;processed=((body>0)|root).astype(np.uint8) if eligible else np.zeros_like(body)
        elif eligible:
            if self.arm in ARMS[2:]:processed,evidence,sideproposal,capproposal,info=self.fitter.run(frame,raw,probability,body,core,m,self.arm)
            final[root]=0;final[(processed>0)&(raw!=1)]=2
        add=((raw==0)&(final==2)).astype(np.uint8);remove=((raw==2)&(final!=2)).astype(np.uint8);self.final_mask=final;self.layers=dict(rgb_eligible=source['rgb_eligible'].copy(),blue_body=body,rgb_evidence=evidence,processed_body=processed,side_proposal=sideproposal,cap_proposal=capproposal,root_instance=root.astype(np.uint8),edit_add=add,edit_remove=remove);self.outline=info
        if np.any(final[raw==1]!=1):raise ValueError('Original stone overwritten')
        ms=(time.perf_counter()-start)*1000;m['timings']['outline_ms']=ms;m['timings']['extra_total_ms']+=ms
        return add,p,conf,m
