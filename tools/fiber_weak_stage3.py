"""S3: qualified seeds/native gradients plus weak leaves/coherent frontier.

No historical pixel filling, threshold sweep, model or default modification.
"""
from __future__ import annotations
import copy,math,time
import cv2,numpy as np
from .fiber_reference_workspace import ROOT,read
from .fiber_bisector_blue_tip import GrowthPipeline as FrozenPipeline,LAYERS as OLD_LAYERS,direction,unit
from .fiber_blue_outline_repaired import RepairedOutline
from .fiber_blue_outline import root_instance,profiles,boundary_distance,natural_values
from .fiber_connected_blue_growth import chroma
from .fiber_fast_observation import sample
from .fiber_local_dark_evidence import public,nearest
from .fiber_weak_stage1 import install_seed_fix

ARMS=('connected_blue_union','bisector_guided_blue_tip','weak_blue_extension','weak_blue_extension_tip')
LAYERS=(*OLD_LAYERS,'strong_blue','weak_blue','uncertain_blue','weak_candidate','reachable_blue','unconnected_blue','search_domain','evidence_score','RGB_edge','recovered_shaft','recovered_distal','cap_boundary_evidence','pruned_weak')
STAGE=read(ROOT/'configs/weak_blue_staged_control_20261104.json')['s3']

def connect(binary,anchor):
    """8-neighbor evidence graph; never manufactures a connecting pixel."""
    _,labels=cv2.connectedComponents(binary.astype(np.uint8),connectivity=8)
    ids=np.unique(labels[anchor&(binary>0)]);ids=ids[ids>0]
    return np.isin(labels,ids).astype(np.uint8) if len(ids) else np.zeros_like(binary,np.uint8)

def native_gradient(frame):
    image=frame.astype(np.float32)/255.
    gx=cv2.Sobel(image,cv2.CV_32F,1,0,ksize=3)/8
    gy=cv2.Sobel(image,cv2.CV_32F,0,1,ksize=3)/8
    return np.sqrt((gx*gx+gy*gy).sum(-1)/3)

def evidence(frame,seed,cfg,gradient=None):
    c,noise=chroma(frame);v=c-1/3;norm=np.linalg.norm(v,axis=-1)
    cosine=np.sum(v*seed['v'],axis=-1)/np.maximum(norm*np.linalg.norm(seed['v']),1e-9)
    blue=c[...,2]-.5*(c[...,0]+c[...,1]);bg=seed['bg'];gap=blue-(bg[2]-.5*(bg[0]+bg[1]))
    separation=np.linalg.norm(c-bg,axis=-1)-np.linalg.norm(c-seed['mu'],axis=-1)
    edge=native_gradient(frame) if gradient is None else np.asarray(gradient,np.float32)
    if edge.shape!=frame.shape[:-1]:raise ValueError('Gradient/sample shape mismatch')
    a=np.clip((cosine-cfg['cosine_floor'])/(1-cfg['cosine_floor']),0,1)
    b=np.clip(gap/max(seed['gap'],cfg['minimum_blue_excess']),0,1)
    d=np.clip((separation+cfg['distance_margin'])/(2*cfg['distance_margin']),0,1)
    reliability=np.clip(1-noise/cfg['max_chroma_noise'],0,1)
    score=(.4*a+.35*b+.25*d)*np.sqrt(reliability)
    # Independent positive blue identity plus local contrast/edge is mandatory;
    # low score is not rescued by probability or historical pose.
    identity=(blue>cfg['minimum_blue_excess'])&(gap>max(cfg['minimum_blue_excess'],cfg['blue_seed_gap_fraction']*seed['gap']))&(cosine>cfg['cosine_floor'])&(norm>cfg['minimum_chroma_norm'])&(noise<cfg['max_chroma_noise'])
    support=(separation>-cfg['distance_margin'])|(edge>=cfg['minimum_gradient'])
    weak=identity&support&(score>=cfg['weak_score'])
    strong=weak&(score>=cfg['strong_score'])
    uncertain=identity&~weak
    return score,strong,weak&~strong,uncertain,edge,noise

def legacy_frontier(body,o,u,cfg,s):
    z=np.zeros_like(body);yy,xx=np.nonzero(body)
    if not len(xx):return z,dict(candidate_xy=None,reason='empty_recovered_body',strong_cluster=False)
    pts=np.c_[xx,yy];t=(pts-o)@u;end=float(t.max());band=z.copy();keep=t>=end-cfg['depth_at400']*s;band[yy[keep],xx[keep]]=1
    _,labels,stats,_=cv2.connectedComponentsWithStats(band,connectivity=8)
    ids=np.unique(labels[yy[t>=end-1e-6],xx[t>=end-1e-6]]);ids=ids[ids>0]
    if len(ids)!=1:return z,dict(candidate_xy=None,reason='ambiguous_farthest_cluster',strong_cluster=False)
    cluster=(labels==ids[0]).astype(np.uint8);y,x=np.nonzero(cluster);q=np.c_[x,y];qt=(q-o)@u
    front=q[qt>=end-cfg['frontier_band_at400']*s];med=np.median(front,axis=0);xy=front[np.argmin(np.sum((front-med)**2,axis=1))].astype(float)
    area=len(q);span=float(np.ptp(qt));spike=float(end-np.quantile(qt,.9));good=bool(area>=cfg['minimum_area_at400']*s*s and span>=cfg['minimum_span_at400']*s and len(front)>=cfg['minimum_front_pixels']*s*s and spike<=cfg['maximum_spike_at400']*s)
    return cluster,dict(candidate_xy=xy,blue_front_t=end,area=area,span_px=span,frontier_points=len(front),spike_beyond_q90=spike,strong_cluster=good,reason='coherent_recovered_front' if good else 'weak_small_or_spiked_front')

def prune_new_weak(body,protected,strong,weak,score,edge,anchor):
    result=body.copy();removed=np.zeros_like(body);rollbacks=0
    for _ in range(STAGE['prune_iterations']):
        count=cv2.filter2D(result.astype(np.float32),-1,np.ones((3,3),np.float32),borderType=cv2.BORDER_CONSTANT)-result
        cut=(result>0)&~protected&~strong&weak&(score<STAGE['maximum_prune_score'])&(edge<STAGE['maximum_prune_gradient'])&(count<=STAGE['maximum_neighbor_count'])
        if not cut.any():break
        proposed=result.copy();proposed[cut]=0;reachable=connect(proposed,anchor)
        # Do not sever a bridge or silently discard any retained distal region.
        if np.any((proposed>0)&(reachable==0)):rollbacks+=1;break
        result=proposed;removed[cut]=1
    return result,removed,dict(removed_pixels=int(removed.sum()),connectivity_rollbacks=rollbacks,iterations_limit=STAGE['prune_iterations'])

def frontier(body,o,u,cfg,s):
    z=np.zeros_like(body);yy,xx=np.nonzero(body)
    if not len(xx):return z,dict(candidate_xy=None,reason='empty_recovered_body',strong_cluster=False)
    pts=np.c_[xx,yy];t=(pts-o)@u;raw_end=float(t.max());step=STAGE['tip_bin_at400']*s;lo=float(t.min());bins=np.floor((t-lo)/step).astype(int);counts=np.bincount(bins);depth=max(1,int(round(STAGE['front_band_at400']/STAGE['tip_bin_at400'])));mass=np.convolve(counts,np.ones(depth,dtype=int),'full')[:len(counts)]
    # A singleton forward bin cannot become a determined tip just because
    # its neighbor bin contains many pixels. Derive density from frozen mass/depth.
    front_density=max(2,int(math.ceil(STAGE['minimum_supported_band_pixels']*s*s/depth)))
    goodbins=np.flatnonzero((counts>=front_density)&(mass>=STAGE['minimum_supported_band_pixels']*s*s))
    if not len(goodbins):return z,dict(candidate_xy=None,reason='no_supported_continuous_front_band',strong_cluster=False,raw_extreme_t=raw_end)
    end=float(t[bins==goodbins[-1]].max());band=z.copy();select=(t>=end-cfg['depth_at400']*s)&(t<=end+1e-6);band[yy[select],xx[select]]=1
    _,labels,stats,_=cv2.connectedComponentsWithStats(band,connectivity=8);front_ids=np.unique(labels[yy[(t>=end-step)&select],xx[(t>=end-step)&select]]);front_ids=front_ids[front_ids>0]
    if not len(front_ids):return z,dict(candidate_xy=None,reason='no_connected_supported_front',strong_cluster=False,raw_extreme_t=raw_end)
    sizes=[int(stats[k,4]) for k in front_ids];maxsize=max(sizes);ids=[k for k,n in zip(front_ids,sizes) if n==maxsize]
    if len(ids)!=1:return z,dict(candidate_xy=None,reason='ambiguous_supported_front_clusters',strong_cluster=False,raw_extreme_t=raw_end)
    cluster=(labels==ids[0]).astype(np.uint8);y,x=np.nonzero(cluster);q=np.c_[x,y];qt=(q-o)@u;end=float(qt.max());front=q[qt>=end-STAGE['front_band_at400']*s];median=np.median(front,axis=0);xy=front[np.argmin(((front-median)**2).sum(1))].astype(float);farther=t>end+1e-6;span=float(np.ptp(qt));spike=float(end-np.quantile(qt,.9));good=bool(len(q)>=cfg['minimum_area_at400']*s*s and span>=cfg['minimum_span_at400']*s and len(front)>=cfg['minimum_front_pixels']*s*s and spike<=cfg['maximum_spike_at400']*s and not farther.any())
    return cluster,dict(candidate_xy=xy,blue_front_t=end,raw_extreme_t=raw_end,area=len(q),span_px=span,frontier_points=len(front),spike_beyond_q90=spike,strong_cluster=good,unresolved_farther_visible_pixels=int(farther.sum()),farther_extent_px=max(0.,raw_end-end),reason='coherent_supported_front' if good else 'farther_blue_not_resolved' if farther.any() else 'weak_small_or_spiked_front')

def terminal(frame,raw,xy,u,seed,e_cfg,cfg,s,front,search_limited,gradient,body):
    n=np.array([-u[1],u[0]]);off=np.asarray(cfg['normal_offsets_at400'])*s
    inside=xy+np.asarray(cfg['inside_offsets_at400'])[:,None,None]*s*u+off[None,:,None]*n
    outside=xy+np.asarray(cfg['outside_offsets_at400'])[:,None,None]*s*u+off[None,:,None]*n
    yy,xx=np.nonzero(body);pts=np.c_[xx,yy];sections=[]
    for row,axial in enumerate(cfg['inside_offsets_at400']):
        center=xy+axial*s*u;delta=pts-center;near=delta[abs(delta@u)<=.75*s]
        if len(near)<2:return dict(state='unknown',reason='interior_cross_section_unavailable',accuracy=None)
        low,high=np.quantile(near@n,[.05,.95]);middle=(low+high)/2;half=(high-low)/2;offset=float(np.clip(half*STAGE['inside_halfwidth_fraction'],STAGE['inside_offset_min_at400']*s,STAGE['inside_offset_max_at400']*s));inside[row]=center+middle*n+np.asarray(cfg['normal_offsets_at400'])[:,None]*offset*n;sections.append(dict(center_xy=(center+middle*n).tolist(),half_width_px=float(half),normal_offset_px=offset))
    a,av=sample(frame,inside);b,bv=sample(frame,outside);labels,lv=nearest(raw,outside)
    answer=dict(state='unknown',reason='terminal_RGB_insufficient',accuracy=None)
    if not (av.all() and bv.all()) or min(xy[0],xy[1],raw.shape[1]-1-xy[0],raw.shape[0]-1-xy[1])<cfg['minimum_frame_margin_at400']*s:return dict(answer,reason='terminal_out_of_view')
    if np.any(lv&(labels==1)):return dict(answer,reason='current_stone_contact')
    ag,agv=sample(gradient,inside);bg,bgv=sample(gradient,outside)
    if not (agv.all() and bgv.all()):return dict(answer,reason='gradient_sample_out_of_view')
    ai=evidence(a.reshape(-1,1,3),seed,e_cfg,ag.reshape(-1,1));bo=evidence(b.reshape(-1,1,3),seed,e_cfg,bg.reshape(-1,1))
    ac=(ai[1]|ai[2]).reshape(a.shape[:-1]);bc=(bo[1]|bo[2]).reshape(b.shape[:-1]);asc=ai[0].reshape(a.shape[:-1]);bsc=bo[0].reshape(b.shape[:-1]);unknown=float(np.r_[ai[-1].ravel(),bo[-1].ravel()].__ge__(e_cfg['max_chroma_noise']).mean())
    rgb=float(np.median(np.linalg.norm(a.mean(0)-b.mean(0),axis=-1)/(255*math.sqrt(3))))
    ca,_=chroma(a);cb,_=chroma(b);cc=float(np.median(np.linalg.norm(ca.mean(0)-cb.mean(0),axis=-1)))
    drop=float(np.median(asc)-np.median(bsc))
    answer.update(inside_blue_fraction=float(ac.mean()),outside_blue_fraction=float(bc.mean()),inside_score=float(np.median(asc)),outside_score=float(np.median(bsc)),terminal_score_drop=drop,RGB_contrast=rgb,chroma_contrast=cc,unknown_fraction=unknown,inside_points=inside,inside_sections=sections)
    if front.get('unresolved_farther_visible_pixels',0):return dict(answer,state='unknown' if front['farther_extent_px']>STAGE['maximum_ignored_weak_spur_at400']*s else 'weak_candidate',reason='farther_blue_not_resolved')
    if search_limited:return dict(answer,reason='continuous_blue_reaches_search_limit')
    if unknown>cfg['maximum_unknown_fraction']:return dict(answer,reason='terminal_background_unknown')
    if bc.mean()>cfg['maximum_continuation_fraction']:return dict(answer,state='weak_candidate',reason='forward_blue_evidence_unresolved')
    boundary=(rgb>=cfg['minimum_RGB_contrast'] or cc>=cfg['minimum_chroma_contrast']) and drop>=cfg['minimum_terminal_score_drop']
    if ac.mean()>=cfg['minimum_inside_fraction'] and boundary:
        if front['strong_cluster']:return dict(answer,state='current_RGB_confirmed',reason='recovered_visible_external_terminal')
        return dict(answer,state='weak_candidate',reason='small_cluster_with_terminal_evidence')
    return dict(answer,state='weak_candidate',reason='coherent_front_but_weak_external_boundary')

class GrowthPipeline:
    def __init__(self,cfg,arm,t1_config=None,sequential=False):
        if arm not in ARMS:raise ValueError(arm)
        self.cfg=cfg;self.arm=arm;self.new=arm in ARMS[2:];self.q=cfg['weak_blue'];old=read(ROOT/'configs/bisector_blue_tip_control_20261103.json')
        self.inner=FrozenPipeline(old,ARMS[0] if self.new else arm,t1_config,sequential);install_seed_fix(self.inner);self.tracker=self.inner.tracker;self.side=RepairedOutline(read(ROOT/'configs/blue_outline_control_20261102_v2.json'));self.layers={};self.outline={};self.final_mask=None;self.gradient=None
    def reset(self):self.inner.reset();self.layers={};self.outline={};self.final_mask=None;self.gradient=None
    def run(self,frame,raw,p,*args,**kwargs):
        result=self.inner.run(frame,raw,p,*args,**kwargs);c,proposal,conflict,meta=result;start=time.perf_counter();self.layers={k:v.copy() for k,v in self.inner.layers.items()};self.outline=copy.deepcopy(self.inner.outline);self.final_mask=self.inner.final_mask.copy();z=np.zeros_like(raw,np.uint8)
        self.layers.update({k:z.copy() for k in LAYERS if k not in self.layers})
        if not self.new:return result
        if self.outline['eligible']:
            B=self.layers['blue_body'];core,_=root_instance(raw,p,meta);_,_,_,_,info=self.side.run(frame,raw,p,B,core,meta,'contour_sides');u,d=direction(info,self.cfg['bisector_tip']['direction']);info=copy.deepcopy(info);info.update(stage=self.arm,sides_applied=False,final_tip_xy=None,bisector=d,cap=dict(attempted=False,accepted=False,observable=False,state='unknown',reason='tip_not_confirmed',curve=[],candidate_xy=None,accuracy=None));self.outline=info
            if u is not None:self.expand(frame,raw,p,meta,B,core,u)
            else:self.outline['recovery']=dict(reason='direction_unknown',stage_rejections={'direction_unknown':1})
        extra=(time.perf_counter()-start)*1000;meta['timings']['weak_blue_ms']=extra;meta['timings']['outline_ms']+=extra;meta['timings']['extra_total_ms']+=extra
        c=((raw==0)&(self.final_mask==2)).astype(np.uint8);self.layers['edit_add']=c;self.layers['edit_remove']=((raw==2)&(self.final_mask!=2)).astype(np.uint8)
        if np.any(self.final_mask[raw==1]!=1) or np.any(self.final_mask[self.inner.final_mask==2]!=2):raise ValueError('Protected union or stone changed')
        return c,proposal,conflict,public(meta)
    def expand(self,frame,raw,p,meta,B,core,u):
        q=next(z for z in meta['seeds'] if z['reason']=='qualified');mu=np.asarray(q['mu'],np.float32);seed=dict(mu=mu,v=mu-1/3,bg=np.asarray(q['background_chroma'],np.float32),gap=q['relative_blue_advantage']);self.gradient=native_gradient(frame);score,strong,weak,uncertain,edge,noise=evidence(frame,seed,self.q['evidence'],self.gradient);s=min(raw.shape)/400;sc=self.q['search'];o=np.asarray(self.outline['context']['origin']);n=np.array([-u[1],u[0]]);yy,xx=np.indices(raw.shape);t=(xx-o[0])*u[0]+(yy-o[1])*u[1];v=(xx-o[0])*n[0]+(yy-o[1])*n[1];bt=t[B>0];bv=v[B>0];front0=float(bt.max());limit=min(sc['maximum_forward_at400']*s,sc['maximum_forward_diagonal_fraction']*np.hypot(*raw.shape));lateral=(np.quantile(bv,.98)-np.quantile(bv,.02))*sc['lateral_width_multiplier']/2+sc['lateral_margin_at400']*s;center=float(np.median(bv));domain=(t>=bt.min()-2*s)&(t<=front0+limit)&(abs(v-center)<=lateral)
        eligible=(strong|weak)&domain&(raw!=1);graph=eligible|(B>0);reachable=connect(graph,(B>0)&core);new=(reachable>0)&(B==0)&(raw==0);blocked=int(reachable.sum())>sc['maximum_component_fraction']*raw.size or int(new.sum())>sc['maximum_new_fraction']*raw.size
        recovered=B.copy() if blocked else reachable.copy()
        recovered,pruned,pruning=prune_new_weak(recovered,self.final_mask==2,strong,weak,score,edge,(B>0)&core)
        self.layers['pruned_weak']=pruned;self.outline['pruning']=pruning
        new=(recovered>0)&(B==0)&(raw==0);final=self.final_mask.copy();final[new]=2;self.final_mask=final
        cluster,distal=frontier(recovered,o,u,self.q['tip'],s);info=self.outline;info['distal']=distal;xy=distal['candidate_xy'];search_limited=bool(np.any((recovered>0)&(t>=front0+limit-1.5*s)))
        rejected=dict(uncertain_in_domain=int((uncertain&domain).sum()),weak_unconnected=int(((strong|weak)&domain&(reachable==0)&(raw!=1)).sum()),outside_search=int(((strong|weak)&~domain).sum()),stone_evidence_blocked=int(((strong|weak)&domain&(raw==1)).sum()),area_budget=int(blocked),search_limit=int(search_limited))
        recovery=dict(reason='area_budget_fallback' if blocked else 'evidence_graph_recovered' if new.any() else 'no_additional_reachable_evidence',stage_rejections=rejected,forward_limit_px=limit,old_blue_front_t=front0,recovered_front_t=distal.get('blue_front_t'),distal_gain_px=max(0,float(t[recovered>0].max())-front0),new_pixels=int(new.sum()),new_Pbelow01=int((new&(p[2]<.1)).sum()),new_lowLab=int((new&(cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)[...,0]<85)).sum()),new_weak=int((new&weak).sum()),new_strong=int((new&strong).sum()),new_RGB_edge_support=float((edge[new]>=self.q['evidence']['minimum_gradient']).mean()) if new.any() else None,new_mean_score=float(score[new].mean()) if new.any() else None,union_retention=1.0,blue_retention=1.0,connectivity=8,temporal_prior=meta.get('t1_advice'))
        prior=meta.get('t1_advice',{}).get('prior_shaft')
        if isinstance(prior,dict):recovery['temporal_diagnostic_only']=prior
        info.update(recovery=recovery,candidate_tip_xy=xy,reason=recovery['reason']);self.layers.update(strong_blue=(strong&domain).astype(np.uint8),weak_blue=(weak&domain).astype(np.uint8),uncertain_blue=(uncertain&domain).astype(np.uint8),weak_candidate=(weak&domain&(B==0)&(raw!=1)).astype(np.uint8),reachable_blue=recovered,rgb_evidence=eligible.astype(np.uint8),unconnected_blue=((strong|weak)&domain&(reachable==0)&(raw!=1)).astype(np.uint8),search_domain=domain.astype(np.uint8),evidence_score=np.rint(np.clip(score,0,1)*255).astype(np.uint8),RGB_edge=(edge>=self.q['evidence']['minimum_gradient']).astype(np.uint8),distal_cluster=cluster,recovered_shaft=(new&(t<=front0)).astype(np.uint8),recovered_distal=(new&(t>front0)).astype(np.uint8),processed_body=((self.layers['root_instance']>0)|(recovered>0)).astype(np.uint8))
        if xy is None:info['tip_observation']=dict(state='unknown',reason=distal['reason']);self.outline=public(info);return
        observation=terminal(frame,raw,np.asarray(xy),u,seed,self.q['evidence'],self.q['tip'],s,distal,search_limited or blocked,self.gradient,recovered);info['tip_observation']=observation;info['cap'].update(candidate_xy=xy,state=observation['state'],reason=observation['reason'],observable=observation['state']=='current_RGB_confirmed')
        if observation['state']=='current_RGB_confirmed':info['final_tip_xy']=xy
        if self.arm==ARMS[-1] and observation['state']=='current_RGB_confirmed':self.fit_cap(frame,raw,p,recovered,core,meta,eligible,o,u,xy,seed)
        self.outline=public(self.outline)
    def fit_cap(self,frame,raw,p,B,core,meta,E,o,u,xy,seed):
        """Observed-front constrained ellipse, only evidence-supported concavities.

        All retained pixels remain. A smooth proposal alone is not an applied cap.
        """
        s=min(raw.shape)/400;cfg=self.q['cap'];n=np.array([-u[1],u[0]]);end=float(((np.c_[np.nonzero(B)[1],np.nonzero(B)[0]]-o)@u).max());pr,_,_=profiles(B,o,u,s,1.5*s);candidates=[];yy,xx=np.indices(raw.shape);tt=(xx-o[0])*u[0]+(yy-o[1])*u[1];vv=(xx-o[0])*n[0]+(yy-o[1])*n[1];bd=boundary_distance(B);ev=evidence(frame,seed,self.q['evidence'],self.gradient);fringe=ev[3]&(ev[4]>=self.q['evidence']['minimum_gradient'])&(bd<=cfg['maximum_boundary_distance_at400']*s)&(raw!=1);cap_evidence=connect(E|fringe|(B>0),B>0)>0;self.layers['cap_boundary_evidence']=cap_evidence.astype(np.uint8)
        for retreat in cfg['retreat_at400']:
            join=end-retreat*s;near=pr[(pr[:,0]>=join-10*s)&(pr[:,0]<=join)&(pr[:,3]>0)&(pr[:,4]>0)];tail=pr[pr[:,0]>=join]
            if len(near)<4 or len(tail)<3:continue
            # Robust side tangent fitting is based on recovered RGB body.
            slopes=[];sides=[]
            for k in [1,2]:
                A=np.c_[near[:,0]-join,np.ones(len(near))];beta=np.linalg.lstsq(A,near[:,k],rcond=None)[0]
                for _ in range(12):
                    residual=near[:,k]-A@beta;w=np.minimum(1,1.5*s/np.maximum(abs(residual),1e-9));beta=np.linalg.lstsq(A*w[:,None]**.5,near[:,k]*w**.5,rcond=None)[0]
                slopes.append(beta[0]);sides.append(beta[1])
            half=(sides[1]-sides[0])/2;center=sum(sides)/2;length=end-join
            if half<=0:continue
            slope=(float((np.asarray(xy)-o)@n)-center)/length
            for r in cfg['r_candidates']:
                a=length/(1-r);ct=join-a*r;b=half/math.sqrt(1-r*r);theta=np.linspace(-math.acos(r),math.acos(r),cfg['samples']);t=ct+a*np.cos(theta);v=center+slope*(t-join)+b*np.sin(theta);curve=o+t[:,None]*u+v[:,None]*n;wp=-half*r/(length*(1+r));jumps=np.degrees(abs(np.arctan(slopes)-np.arctan([slope-wp,slope+wp])));w=b*np.sqrt(np.clip(1-((tail[:,0]-ct)/a)**2,0,1));cy=center+slope*(tail[:,0]-join);residual=np.r_[tail[:,1]-(cy-w),tail[:,2]-(cy+w)];rms=float(np.sqrt(np.mean(residual**2)));relative=v-center-slope*(t-join);normals=((t-ct)/a**2-slope*relative/b**2)[:,None]*u+(relative/b**2)[:,None]*n;normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-9);inside,iv=sample(frame,curve-1.5*s*normals);outside,ov=sample(frame,curve+1.5*s*normals);dist,dv=sample(bd,curve);ag,agv=sample(self.gradient,curve-1.5*s*normals);bg,bgv=sample(self.gradient,curve+1.5*s*normals);ai=evidence(inside[:,None,:],seed,self.q['evidence'],ag[:,None]);bo=evidence(outside[:,None,:],seed,self.q['evidence'],bg[:,None]);contrast=np.linalg.norm(inside-outside,axis=-1)/(255*math.sqrt(3));supported=iv&ov&dv&agv&bgv&((ai[1]|ai[2]).ravel())&((~(bo[1]|bo[2]).ravel())|(contrast>=self.q['tip']['minimum_RGB_contrast']))&(dist<=cfg['maximum_boundary_distance_at400']*s);ideal=(tt>=join)&(tt<=end)&(abs(vv-center-slope*(tt-join))<=b*np.sqrt(np.clip(1-((tt-ct)/a)**2,0,1)));addition=ideal&cap_evidence&(self.final_mask==0);result=((B>0)|addition).astype(np.uint8);ad,av=sample(boundary_distance(result),curve);match=float((av&(ad<=cfg['maximum_boundary_distance_at400']*s)).mean());reason='accepted_current_RGB_cap'
                if rms>cfg['maximum_RMS_at400']*s:reason='ellipse_fit_residual'
                elif max(jumps)>cfg['maximum_model_join_degrees']:reason='model_join_tangent_mismatch'
                elif supported.mean()<cfg['minimum_RGB_support']:reason='insufficient_current_RGB_arc'
                elif match<cfg['minimum_actual_arc_match']:reason='protected_raster_not_following_arc'
                candidates.append(dict(attempted=True,accepted=reason=='accepted_current_RGB_cap',observable=True,state='current_RGB_confirmed',reason=reason,curve=curve,candidate_xy=xy,apex_xy=o+end*u+(center+slope*length)*n,tip_t=end,join=join,a=a,b=b,r=r,center_t=ct,center_n=center,slope=slope,RMS=rms,score=float(np.median(abs(residual))),tangent_jumps=jumps,RGB_supported_fraction=float(supported.mean()),actual_arc_match_fraction=match,addition=addition,accuracy=None))
        if candidates:
            feasible=[x for x in candidates if x['accepted']];best=min(feasible or candidates,key=lambda x:(x['score'],x['RMS'],x['r']));addition=best.pop('addition');best.update(candidate_count=len(candidates),feasible_count=len(feasible));self.outline['cap']=best
            if best['accepted']:
                self.final_mask[addition]=2;self.layers['cap_fill']=addition.astype(np.uint8);self.layers['processed_body']|=addition.astype(np.uint8);self.outline['cap_coordinate_context']=dict(origin=o,axis=u,join_t=best['join'],scale=s,min_t=float(tt[B>0].min()),max_t=end);self.outline['reason']='recovered_body_with_observed_cap'
        else:self.outline['cap'].update(attempted=True,reason='insufficient_recovered_side_join')
