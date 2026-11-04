"""Current-RGB rooted growth candidates only; actual semantic output is immutable."""
from __future__ import annotations
import math,time
import cv2
import numpy as np
from .fiber_fast_observation import sample,FastFiberObserver
from .fiber_evidence_tracker import EvidenceTracker
from .fiber_local_dark_evidence import public,nearest

ARMS=('raw2','connected_rgb','connected_rgb_t1')


def chroma(bgr):
    rgb=np.asarray(bgr,np.float32)[...,::-1];total=rgb.sum(-1)
    c=rgb/(total[...,None]+1e-6)
    return c,3./(total+3.)


def color_ok(bgr,mu,sigma,settings,background=None):
    c,noise=chroma(bgr);scale=np.maximum(sigma,noise[...,None])
    ok=(np.sqrt(np.mean(((c-mu)/scale)**2,axis=-1))<=settings['chroma_MAD_multiplier'])&(noise<=settings['maximum_chroma_noise'])
    if background is not None:
        ok &= np.linalg.norm(c-background,axis=-1)>np.linalg.norm(c-mu,axis=-1)+settings['background_distance_margin']
    return ok


def angle(a,b):return math.degrees(math.acos(float(np.clip(np.dot(a,b),-1,1))))


def section(frame,center,u,width,mu,sigma,cfg,background=None,require_color=True):
    s=min(frame.shape[:2])/400;g=cfg['geometry'];normal=np.array([-u[1],u[0]])
    offsets=np.linspace(-(.75*width+4*s),.75*width+4*s,g['normal_samples']);xy=center+offsets[:,None]*normal
    delta=g['difference_offset_at400']*s;a,av=sample(frame,xy-delta*normal);b,bv=sample(frame,xy+delta*normal)
    contrast=np.linalg.norm(a-b,axis=-1)/(255*math.sqrt(3));valid=av&bv
    chosen=[];values=[]
    for sign in (-1,1):
        order=sorted(np.flatnonzero(offsets*sign>0),key=lambda k:(abs(offsets[k]-sign*width/2),offsets[k]))
        scores=np.where(valid[order],contrast[order],-1);k=order[int(np.argmax(scores))]
        chosen.append(offsets[k]);values.append(float(contrast[k]) if valid[k] else 0.)
    left,right=chosen
    if left>=right or min(values)<g['edge_contrast_min']:return None
    sides=center+np.array(chosen)[:,None]*normal
    interior=center+(left+(right-left)*np.array(g['interior_fractions']))[:,None]*normal
    rgb,ig=sample(frame,interior)
    if not ig.all() or (require_color and np.count_nonzero(color_ok(rgb,mu,sigma,cfg['appearance'],background))<g['required_interior_support']):return None
    outside=center+np.array([left-2*delta,right+2*delta])[:,None]*normal;bg,bg_good=sample(frame,outside)
    return {'center':sides.mean(0),'sides':sides,'width':float(right-left),'contrast':values,'u':u.copy(),
            'background_rgb':bg[bg_good],'interior_xy':interior}


def _pca(points):
    center=points.mean(0);_,vectors=np.linalg.eigh((points-center).T@(points-center));u=vectors[:,-1]
    if u[0]<0 or (u[0]==0 and u[1]<0):u=-u
    n=np.array([-u[1],u[0]]);a=(points-center)@u;b=(points-center)@n
    return center,u,np.percentile(a,[2,98]),max(float(np.ptp(np.percentile(b,[2,98]))),1.)


class ConnectedBlueGrowth:
    def __init__(self,config):self.cfg=config

    def trusted_seeds(self,frame,mask,probability):
        cfg=self.cfg;s=min(mask.shape)/400;seedcfg=cfg['seed'];binary=((mask==2)&(probability[2]>=seedcfg['P2_min'])).astype(np.uint8)
        count,labels,stats,_=cv2.connectedComponentsWithStats(binary,connectivity=4)
        eligible=[k for k in range(1,count) if stats[k,cv2.CC_STAT_AREA]>=seedcfg['minimum_pixels_at400']*s*s]
        eligible.sort(key=lambda k:(-stats[k,cv2.CC_STAT_AREA],k));records=[];qualified=[]
        if len(eligible)>seedcfg['maximum_components_examined']:return [],[{'reason':'excessive_seed_hypotheses','components':len(eligible)}]
        for k in eligible:
            yy,xx=np.nonzero(labels==k);points=np.stack([xx,yy],-1).astype(float);core=labels==k
            subset=points[np.linspace(0,len(points)-1,min(len(points),seedcfg['maximum_statistic_points'])).astype(int)]
            c,_=chroma(frame[subset[:,1].astype(int),subset[:,0].astype(int)]);mu=np.median(c,axis=0)
            sigma=np.maximum(1.4826*np.median(np.abs(c-mu),axis=0),cfg['appearance']['chroma_MAD_floor'])
            origin,u,ends,width=_pca(subset);length=float(ends[1]-ends[0]);record={'label':k,'pixels':len(points),'length':length,'width':width,'mu':mu,'sigma':sigma,'reason':'short_or_nonelongated_seed'}
            if length<seedcfg['minimum_length_at400']*s or length/width<seedcfg['minimum_length_width_ratio']:records.append(record);continue
            stations=ends[0]+np.array(seedcfg['seed_axial_fractions'])*length
            sections=[section(frame,origin+t*u,u,width,mu,sigma,cfg) for t in stations];good=[x for x in sections if x is not None]
            record['edge_pairs']=len(good)
            if len(good)<seedcfg['required_edge_pairs']:record['reason']='insufficient_seed_edges';records.append(record);continue
            bg=np.concatenate([x['background_rgb'] for x in good],axis=0)
            if len(bg)<cfg['geometry']['minimum_color_valid_background_points']:record['reason']='insufficient_seed_background';records.append(record);continue
            bg_c,_=chroma(bg);mu_b=np.median(bg_c,axis=0);blue=lambda z:z[...,2]-.5*(z[...,0]+z[...,1]);b=blue(bg_c)
            advantage=float(blue(mu)-np.median(b));limit=seedcfg['blue_background_mad_multiplier']*max(1.4826*float(np.median(np.abs(b-np.median(b)))),seedcfg['blue_background_mad_floor'])
            record.update(background_chroma=mu_b,relative_blue_advantage=advantage,relative_blue_required=limit)
            if advantage<=limit:record['reason']='no_separable_blue_identity';records.append(record);continue
            centers=np.array([x['center'] for x in good]);_,edge_u,_,_=_pca(centers)
            if np.dot(edge_u,u)<0:edge_u=-edge_u
            if angle(u,edge_u)>cfg['geometry']['maximum_total_turn_degrees']:record['reason']='seed_edges_direction_conflict';records.append(record);continue
            width=float(np.median([x['width'] for x in good]));record['reason']='qualified';records.append(record)
            qualified.append({'core':core,'origin':origin,'u':edge_u,'width':width,'length':length,'mu':mu,'sigma':sigma,'background':mu_b,
                'start_sections':[good[0],good[-1]],'label':k})
        return qualified,records

    def walk(self,frame,seed,sign):
        cfg=self.cfg;g=cfg['geometry'];s=min(frame.shape[:2])/400;u=seed['u']*sign;initial=u.copy()
        current=seed['start_sections'][0 if sign==-1 else 1];width=current['width'];center=current['center'];step=max(2*s,seed['width']/4)
        maximum=min(.2*math.hypot(*frame.shape[:2]),4*seed['width']);sections=[current];distance=0.;polygons=[];abnormal=None;reason='section_budget_limit'
        for _ in range(g['maximum_sections_per_end']):
            if distance+step>maximum:reason='search_limit';break
            next_section=section(frame,center+step*u,u,width,seed['mu'],seed['sigma'],cfg,seed['background'])
            if next_section is None:reason='current_RGB_evidence_missing';break
            move=next_section['center']-center;move_length=float(np.linalg.norm(move))
            if move_length<1e-6:abnormal='zero_step';break
            direction=move/move_length;updated=(1-g['heading_blend'])*u+g['heading_blend']*direction;updated/=np.linalg.norm(updated)
            ratio=next_section['width']/width
            if not g['width_step_ratio'][0]<=ratio<=g['width_step_ratio'][1] or next_section['width']>g['maximum_width_seed_ratio']*seed['width']:
                abnormal='width_expansion_or_jump';break
            if angle(u,updated)>g['maximum_turn_step_degrees'] or angle(initial,updated)>g['maximum_total_turn_degrees']:
                abnormal='direction_jump';break
            polygons.append(np.array([current['sides'][0],current['sides'][1],next_section['sides'][1],next_section['sides'][0]]))
            current=next_section;center=current['center'];width=current['width'];u=updated;current['u']=u.copy();sections.append(current);distance+=move_length
        if abnormal:reason=abnormal
        return {'sign':sign,'sections':sections,'polygons':polygons,'distance':distance,'stop_reason':reason,'abnormal':abnormal,'final_u':u}

    def tip(self,frame,mask,seed,branch):
        cfg=self.cfg;tip=cfg['tip'];s=min(mask.shape)/400;last=branch['sections'][-1];u=branch['final_u'];n=np.array([-u[1],u[0]]);w=last['width'];center=last['center']
        result={'state':'unknown','reason':branch['stop_reason'],'xy':None,'direction':'unoriented_end','end_sign':branch['sign']}
        if branch['stop_reason'] in ['search_limit','section_budget_limit'] or branch['abnormal']:return result
        offsets=np.linspace(tip['axial_range_widths'][0]*w,tip['axial_range_widths'][1]*w,tip['axial_samples'])
        xy=center+np.array(tip['offset_width_fractions'])[:,None,None]*w*n+offsets[None,:,None]*u
        a,av=sample(frame,xy-2*s*u);b,bv=sample(frame,xy+2*s*u);contrast=np.linalg.norm(a-b,axis=-1)/(255*math.sqrt(3));valid=av&bv
        score=np.where(valid,contrast,-1);idx=score.argmax(1);strength=score[np.arange(3),idx];supported=strength>=cfg['geometry']['edge_contrast_min']
        result.update(boundary_profile_contrasts=strength.tolist(),boundary_profile_offsets_px=offsets[idx].tolist())
        if not supported[1] or supported.sum()<tip['minimum_boundary_profiles']:result['reason']='insufficient_current_terminal_boundary';return result
        front=center+offsets[idx[1]]*u
        beyond=[section(frame,front+d*s*u,u,w,seed['mu'],seed['sigma'],cfg,require_color=False) for d in (2,4)]
        if all(x is not None for x in beyond) and all(x['width']>tip['closure_width_fraction']*w for x in beyond):
            result.update(state='internal_transition',reason='current_side_edges_continue');return result
        labels,lg=nearest(mask,np.array([front+2*s*u,front+4*s*u]))
        if np.any(lg&(labels==1)):result.update(state='occlusion_suspected',reason='current_stone_contact_and_RGB_boundary');return result
        if any(x is not None and x['width']<=tip['closure_width_fraction']*w for x in beyond):
            result.update(state='visible_end_candidate',reason='current_closure_and_terminal_boundary',xy=front.tolist())
        else:result['reason']='no_independent_terminal_closure'
        return result

    def grow(self,frame,mask,probability):
        start=time.perf_counter();cfg=self.cfg;h,w=mask.shape;empty=np.zeros_like(mask,np.uint8);records=[]
        meta={'reason':'no_trusted_seed','seeds':[],'branches':[],'tips':[],'apply_to_mask':False,'tip_metrics':None,'abnormal':False,'evaluated_pixels':0}
        seeds,records=self.trusted_seeds(frame,mask,probability);meta['seeds']=records;seed_end=time.perf_counter()
        if len(seeds)!=1:
            meta['reason']='ambiguous_trusted_seeds' if len(seeds)>1 else 'no_trusted_seed'
            meta['timings']={'seed_ms':(seed_end-start)*1000,'growth_tip_ms':0.,'extra_total_ms':(time.perf_counter()-start)*1000}
            return empty,empty.copy(),empty.copy(),public(meta)
        seed=seeds[0];branches=[self.walk(frame,seed,sign) for sign in (-1,1)]
        allowed=seed['core'].astype(np.uint8);background=np.zeros((h,w,3),np.float32);background[:]=seed['background']
        for branch in branches:
            for polygon,sec in zip(branch['polygons'],branch['sections'][1:]):
                region=np.zeros_like(mask,np.uint8);cv2.fillConvexPoly(region,np.rint(polygon).astype(np.int32),1);allowed |= region
                if len(sec['background_rgb']):background[region>0]=np.median(chroma(sec['background_rgb'])[0],axis=0)
        area=int(allowed.sum());meta.update(evaluated_pixels=area,seed_width=seed['width'],seed_length=seed['length'],seed_axis=seed['u'],seed_origin=seed['origin'])
        meta['branches']=[{'sign':x['sign'],'distance':x['distance'],'stop_reason':x['stop_reason'],'abnormal':x['abnormal'],
            'centers':[z['center'] for z in x['sections']],'sides':[z['sides'] for z in x['sections']],'widths':[z['width'] for z in x['sections']],'final_u':x['final_u']} for x in branches]
        if area>math.floor(cfg['growth']['maximum_evaluated_fraction']*h*w):
            meta.update(reason='evaluation_area_budget',abnormal=True,proposal_kind='unevaluated_search_scope');proposal=allowed.copy();accepted=empty;conflict=empty.copy()
        else:
            yy,xx=np.nonzero(allowed);rgb=frame[yy,xx];good=color_ok(rgb,seed['mu'],seed['sigma'],cfg['appearance'],background[yy,xx]);eligible=np.zeros_like(mask,np.uint8);eligible[yy[good],xx[good]]=1
            roots=(eligible>0)&seed['core'];root_count,root_labels,root_stats,_=cv2.connectedComponentsWithStats(roots.astype(np.uint8),connectivity=4)
            roots_ids=[k for k in range(1,root_count) if root_stats[k,cv2.CC_STAT_AREA]>=cfg['seed']['minimum_pixels_at400']*(min(h,w)/400)**2]
            def reachable(binary):
                if len(roots_ids)!=1:return empty.copy()
                _,labels=cv2.connectedComponents(binary,connectivity=4);root_y,root_x=np.argwhere(root_labels==roots_ids[0])[0];k=labels[root_y,root_x]
                return ((labels==k)&(binary>0)).astype(np.uint8) if k else empty.copy()
            unblocked=reachable(eligible);blocked=eligible.copy();blocked[mask==1]=0;actual=reachable(blocked)
            conflict=(unblocked>0)&(mask==1);proposal=((actual>0)&(mask==0)).astype(np.uint8);accepted=proposal.copy()
            abnormal=next((x['abnormal'] for x in branches if x['abnormal']),None)
            reason='candidate_accepted' if proposal.any() else 'no_continuous_new_candidate'
            if len(roots_ids)!=1:reason='color_root_absent_or_ambiguous';accepted[:]=0
            if abnormal:reason=abnormal;accepted[:]=0
            if proposal.sum()>math.floor(cfg['growth']['maximum_added_fraction']*h*w):reason='new_area_budget';abnormal=reason;accepted[:]=0
            meta.update(reason=reason,abnormal=bool(abnormal),proposal_kind='RGB_reachable_new_pixels',root_components=len(roots_ids),pre_stone_reachable_pixels=int(unblocked.sum()),
                eligible_pixels=int(eligible.sum()),root_connected_pixels=int(actual.sum()),stone_conflict_pixels=int(conflict.sum()))
            if accepted.any():meta['tips']=[self.tip(frame,mask,seed,b) for b in branches]
        ended=time.perf_counter();meta['timings']={'seed_ms':(seed_end-start)*1000,'growth_tip_ms':(ended-seed_end)*1000,'extra_total_ms':(ended-start)*1000}
        return accepted.astype(np.uint8),proposal.astype(np.uint8),np.asarray(conflict,np.uint8),public(meta)


class GrowthPipeline:
    def __init__(self,cfg,arm,t1_config=None,sequential=False):
        if arm not in ARMS:raise ValueError('Unknown arm')
        self.cfg=cfg;self.arm=arm;self.growth=ConnectedBlueGrowth(cfg);self.sequential=sequential
        self.observer=FastFiberObserver(t1_config) if sequential and arm=='connected_rgb_t1' else None
        self.tracker=EvidenceTracker(t1_config) if self.observer else None

    def reset(self):
        if self.tracker:self.tracker.reset()

    def run(self,frame,mask,probability,timestamp=0.,source='static'):
        if self.arm=='raw2':
            a=np.zeros_like(mask,np.uint8);return a,a.copy(),a.copy(),{'reason':'raw_control','seeds':[],'branches':[],'tips':[],'abnormal':False,'apply_to_mask':False,'tip_metrics':None,'timings':{'extra_total_ms':0.}}
        start=time.perf_counter();candidate,proposal,conflict,meta=self.growth.grow(frame,mask,probability);t1start=time.perf_counter()
        advice={'direction':'unknown','abnormal_shortening':False,'source':'disabled_static_or_spatial'}
        if self.tracker:
            previous_length=None
            if self.tracker.source==source and self.tracker.groups['endpoint'] is not None and self.tracker.age['endpoint']<=3:previous_length=float(self.tracker.groups['endpoint'][0])
            prepared=self.tracker.prepare(mask.shape,timestamp,source);current=self.observer.current(frame,probability);tracked=self.tracker.update(current,None,prepared)
            shaft=tracked['groups']['shaft'];advice.update(source='original_probability_T1_only',shaft_status=shaft['status'],shaft_age=shaft['age'],front_predicted=tracked['front_is_prediction'])
            if shaft['value'] is not None and shaft['age']<=3 and 'seed_axis' in meta:
                theta=shaft['value'][2];axis=np.array(meta['seed_axis']);dot=float(np.dot(axis,[math.cos(theta),math.sin(theta)]))
                if abs(dot)>=math.cos(math.radians(self.cfg['t1']['direction_agreement_degrees'])):
                    advice['direction']='positive_end' if dot>=0 else 'negative_end'
                else:advice['direction']='current_history_conflict'
                if previous_length is not None:
                    gap=previous_length-meta['seed_length'];advice['abnormal_shortening']=gap>max(self.cfg['t1']['shortening_fraction']*previous_length,self.cfg['t1']['minimum_shortening_at400']*min(mask.shape)/400)
        meta['t1_advice']=advice;ended=time.perf_counter();meta['timings'].update(t1_ms=(ended-t1start)*1000,extra_total_ms=(ended-start)*1000)
        return candidate,proposal,conflict,meta
