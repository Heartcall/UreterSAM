"""Frozen blue external-body shadow candidate; never mutates semantic output."""
from __future__ import annotations
import math,time
import cv2,numpy as np
from .fiber_connected_blue_growth import _pca,angle,chroma,GrowthPipeline as LegacyPipeline
from .fiber_fast_observation import sample,FastFiberObserver
from .fiber_evidence_tracker import EvidenceTracker,valid_group
from .fiber_local_dark_evidence import public,nearest
from .fiber_reference_workspace import ROOT,read

ARMS=('raw2','connected_rgb','blue_extent')

def blue(c):return c[...,2]-.5*(c[...,0]+c[...,1])

def compatible_history(prior,age,center,u,width,shape,cfg):
    if prior is None or age>cfg['max_age'] or not valid_group('shaft',prior):return False
    p=np.array(prior,float);v=np.array([math.cos(p[2]),math.sin(p[2])]);normal=np.array([-v[1],v[0]])
    return bool(abs(np.dot(u,v))>=math.cos(math.radians(cfg['max_axis_difference_degrees'])) and
        abs(np.dot(center-p[:2],normal))<=cfg['max_normal_distance_diagonal_fraction']*math.hypot(*shape) and
        cfg['width_ratio'][0]<=width/p[3]<=cfg['width_ratio'][1])

class BlueExtent:
    def __init__(self,config):self.cfg=config

    def color_score(self,bgr,seed,weak=False):
        c,noise=chroma(bgr);v=c-1/3;norm=np.linalg.norm(v,axis=-1);a=self.cfg['color'];suffix='weak' if weak else 'normal'
        cosine=np.sum(v*seed['v'],axis=-1)/np.maximum(norm*np.linalg.norm(seed['v']),1e-9)
        return np.minimum.reduce([a['max_noise']-noise,norm-noise,cosine-math.cos(math.radians(a['angle_'+suffix+'_degrees'])),
            blue(c)-blue(seed['bg'])-a['blue_gap_'+suffix+'_fraction']*seed['gap'],
            np.linalg.norm(c-seed['bg'],axis=-1)-np.linalg.norm(c-seed['mu'],axis=-1)-a['background_distance_margin_'+suffix]])

    def color(self,bgr,seed,weak=False):return self.color_score(bgr,seed,weak)>0

    def section(self,frame,center,u,width,seed):
        g=self.cfg['section'];s=min(frame.shape[:2])/400;n=np.array([-u[1],u[0]])
        offsets=np.linspace(-g['halfwidth_scale']*width-g['halfwidth_add_at400']*s,g['halfwidth_scale']*width+g['halfwidth_add_at400']*s,g['samples'])
        pts=center+offsets[:,None]*n;rgb,valid=sample(frame,pts);scores=self.color_score(rgb,seed);good=valid&(scores>0)
        starts=np.flatnonzero(good&~np.r_[False,good[:-1]]);ends=np.flatnonzero(good&~np.r_[good[1:],False])
        runs=[(a,b) for a,b in zip(starts,ends) if offsets[a]<=g['center_band_width_fraction']*width and offsets[b]>=-g['center_band_width_fraction']*width]
        if len(runs)!=1:return None,'no_unique_current_blue_interval'
        a,b=runs[0];positions=[];states=[];contrasts=[]
        for k,outside in [(a,a-1),(b,b+1)]:
            if 0<=outside<len(offsets) and valid[outside]:
                fraction=float(np.clip(scores[k]/max(scores[k]-scores[outside],1e-12),0,1))
                p=center+(offsets[k]+fraction*(offsets[outside]-offsets[k]))*n
                left,lv=sample(frame,p-g['edge_delta_at400']*s*n);right,rv=sample(frame,p+g['edge_delta_at400']*s*n)
                contrast=float(np.linalg.norm(left-right)/(255*math.sqrt(3))) if lv and rv else 0.
                state='observed_strong' if contrast>=g['strong_RGB_contrast'] else 'observed_weak'
            elif (outside<0 or outside>=len(offsets)) and valid[k]:
                return None,'normal_search_exhausted'
            else:p=pts[k];contrast=0.;state='out_of_view'
            positions.append(p);states.append(state);contrasts.append(contrast)
        count=states.count('observed_strong')
        if count==0:return None,'both_sides_not_strong'
        sides=np.array(positions);w=float(np.linalg.norm(sides[1]-sides[0]));c=sides.mean(0)
        if w<1*s:return None,'insufficient_observed_width'
        weak=count<2;interior=sides[0]+np.array(g['interior_fractions'])[:,None]*(sides[1]-sides[0]);pixels,iv=sample(frame,interior)
        supported=iv&self.color(pixels,seed,weak)
        required=g['weak_min_interior'] if weak else g['normal_min_interior']
        if supported.sum()<required:return None,'insufficient_current_interior'
        return dict(center=c,sides=sides,width=w,u=u.copy(),states=states,contrasts=contrasts,weak=weak),None

    def seeds(self,frame,mask,probability,prior,age):
        cfg=self.cfg['seed'];s=min(mask.shape)/400;binary=((mask==2)&(probability[2]>=cfg['P_min'])).astype(np.uint8)
        count,labels,stats,_=cv2.connectedComponentsWithStats(binary,connectivity=4);ids=[k for k in range(1,count) if stats[k,cv2.CC_STAT_AREA]>=cfg['min_pixels_at400']*s*s]
        if len(ids)>cfg['max_components']:return [],[dict(reason='excessive_seed_hypotheses',components=len(ids))]
        records=[];qualified=[]
        for k in sorted(ids,key=lambda k:(-stats[k,cv2.CC_STAT_AREA],k)):
            yy,xx=np.nonzero(labels==k);points=np.stack([xx,yy],-1).astype(float);sub=points[np.linspace(0,len(points)-1,min(len(points),cfg['max_points'])).astype(int)]
            origin,u,ends,width=_pca(sub);length=float(ends[1]-ends[0]);record=dict(label=k,pixels=len(points),length=length,width=width,reason='short_or_nonelongated_seed')
            if length<cfg['min_length_at400']*s or length/width<cfg['min_aspect']:records.append(record);continue
            pixels=frame[sub[:,1].astype(int),sub[:,0].astype(int)];mu=np.median(chroma(pixels)[0],axis=0);n=np.array([-u[1],u[0]])
            centers=origin+(ends[0]+np.array(cfg['fractions'])*length)[:,None]*u
            bp=centers[:,None,:]+np.array([-1,1])[None,:,None]*(.9*width+4*s)*n;bg_rgb,bv=sample(frame,bp)
            if bv.sum()<6:record['reason']='insufficient_seed_background';records.append(record);continue
            bg_values=chroma(bg_rgb[bv])[0];bg=np.median(bg_values,axis=0);bv_blue=blue(bg_values);gap=float(blue(mu)-blue(bg))
            limit=cfg['background_MAD_multiplier']*max(1.4826*float(np.median(abs(bv_blue-np.median(bv_blue)))),cfg['background_MAD_floor'])
            record.update(mu=mu,background_chroma=bg,relative_blue_advantage=gap,relative_blue_required=limit)
            if gap<=limit or np.linalg.norm(mu-1/3)<cfg['min_color_vector_norm']:record['reason']='no_separable_blue_identity';records.append(record);continue
            seed=dict(core=labels==k,mu=mu,v=mu-1/3,bg=bg,gap=gap,origin=origin,u=u,width=width,length=length,label=k)
            sections=[];failures=[]
            for center in centers:
                section,reason=self.section(frame,center,u,width,seed)
                if section is not None:sections.append(section)
                else:failures.append(reason)
            paired=sum(not z['weak'] for z in sections);record.update(paired_sections=paired,usable_sections=len(sections),section_failures=failures)
            normal=paired>=cfg['normal_min_sections'];weak=(len(sections)>=cfg['weak_min_sections'] and paired>=cfg['weak_min_paired'])
            if not normal and not weak:record['reason']='insufficient_external_sections';records.append(record);continue
            if not normal and not compatible_history(prior,age,origin,u,width,mask.shape,self.cfg['history']):record['reason']='weak_seed_requires_compatible_history';records.append(record);continue
            cs=np.array([z['center'] for z in sections]);_,edge_u,_,_=_pca(cs)
            if np.dot(edge_u,u)<0:edge_u=-edge_u
            if angle(edge_u,u)>self.cfg['geometry']['normal_total_turn_degrees']:record['reason']='external_axis_conflict';records.append(record);continue
            normalvec=np.array([-edge_u[1],edge_u[0]]);fits=[]
            for side in range(2):
                ps=np.array([z['sides'][side] for z in sections if z['states'][side]=='observed_strong'])
                if len(ps)<3:continue
                ts=(ps-origin)@edge_u;ys=(ps-origin)@normalvec;coeff=np.polyfit(ts,ys,1);fits.append(float(np.sqrt(np.mean((ys-np.polyval(coeff,ts))**2))))
            if not fits or max(fits)>max(cfg['fit_rms_at400']*s,cfg['fit_rms_width_fraction']*width):record['reason']='external_line_residual';records.append(record);continue
            seed.update(u=edge_u,start_sections=[sections[0],sections[-1]],width=float(np.median([z['width'] for z in sections])))
            record.update(reason='qualified',mode='normal' if normal else 'weak_history',fit_rms=fits);records.append(record);qualified.append(seed)
        return qualified,records

    def walk(self,frame,seed,sign,prior,age):
        g=self.cfg['geometry'];s=min(frame.shape[:2])/400;u=seed['u']*sign;initial=u.copy();current=seed['start_sections'][0 if sign<0 else 1]
        current={**current,'sides':current['sides'].copy()}
        if np.dot(current['u'],u)<0:current['sides']=current['sides'][::-1].copy()
        current['u']=u.copy()
        center=current['center'];width=current['width'];step=max(g['step_at400']*s,g['step_width_fraction']*seed['width']);sections=[current];polygons=[];rejected=[];distance=weak_distance=0.;weak_run=0;reason='section_budget_limit';abnormal=None
        for _ in range(g['max_sections_per_end']):
            if distance+step>min(g['max_extension_diagonal_fraction']*math.hypot(*frame.shape[:2]),g['max_extension_seed_widths']*seed['width']):reason='search_limit';break
            proposed_center=center+step*u
            if not (0<=proposed_center[0]<=frame.shape[1]-1 and 0<=proposed_center[1]<=frame.shape[0]-1):reason='out_of_view';break
            sec,why=self.section(frame,proposed_center,u,width,seed)
            if sec is None:reason=why;break
            move=sec['center']-center;length=float(np.linalg.norm(move))
            if length<1e-6:reason=abnormal='zero_step';break
            direction=move/length;updated=(1-g['heading_blend'])*u+g['heading_blend']*direction;updated/=np.linalg.norm(updated)
            mode='weak' if sec['weak'] else 'normal';turn=angle(u,updated);ratio=sec['width']/width
            if not g[mode+'_width_ratio'][0]<=ratio<=g[mode+'_width_ratio'][1] or sec['width']>g['max_seed_width_ratio']*seed['width']:reason=abnormal='width_expansion_or_jump';break
            if turn>g[mode+'_turn_degrees'] or angle(initial,updated)>g[mode+'_total_turn_degrees'] or math.radians(turn)*width/length>g[mode+'_curvature_width']:reason=abnormal='direction_or_curvature_jump';break
            # A dense axial witness stops before a true gap, independently of final4CC.
            offsets=np.linspace(0,1,max(2,int(math.ceil(length/max(s,1e-3)))+1));n=np.array([-u[1],u[0]])
            path=center+offsets[:,None]*move;triplets=path[:,None,:]+np.array([-.15,0,.15])[None,:,None]*min(width,sec['width'])*n
            rgb,iv=sample(frame,triplets);support=(iv&self.color(rgb,seed,sec['weak'])).sum(axis=1)
            if np.any(support<(3 if sec['weak'] else 2)):reason='current_axial_gap';break
            polygon=np.array([current['sides'][0],current['sides'][1],sec['sides'][1],sec['sides'][0]])
            if sec['weak']:
                weak_run+=1;weak_distance+=length
                if not compatible_history(prior,age,sec['center'],updated,sec['width'],frame.shape[:2],self.cfg['history']):rejected.append((polygon,True));reason='weak_requires_compatible_history';break
                if weak_run>g['max_consecutive_weak_sections'] or weak_distance>g['max_weak_run_seed_widths']*seed['width']:rejected.append((polygon,True));reason='weak_run_limit';break
            else:weak_run=0;weak_distance=0.
            polygons.append((polygon,sec['weak']));sections.append(sec);distance+=length;current=sec;center=sec['center'];width=sec['width'];u=updated;current['u']=u.copy()
        return dict(sign=sign,sections=sections,polygons=polygons,rejected=rejected,distance=distance,stop_reason=reason,abnormal=abnormal,final_u=u)

    def tip(self,frame,mask,seed,branch,oriented_sign):
        p=self.cfg['tip'];s=min(mask.shape)/400;last=branch['sections'][-1];w=last['width'];u=branch['final_u'];n=np.array([-u[1],u[0]]);center=last['center']
        result=dict(state='unknown',reason=branch['stop_reason'],xy=None,end_sign=branch['sign'],direction='distal' if branch['sign']==oriented_sign else 'unoriented_or_proximal')
        if branch['stop_reason']=='out_of_view':result['state']='out_of_view';return result
        stops=['search_limit','section_budget_limit','weak_requires_compatible_history','weak_run_limit','normal_search_exhausted']
        if branch['abnormal'] or branch['stop_reason'] in stops:return result
        offsets=np.linspace(p['axial_range_widths'][0]*w,p['axial_range_widths'][1]*w,p['profile_samples']);pts=center+np.array(p['profile_offsets_width'])[:,None,None]*w*n+offsets[None,:,None]*u
        a,av=sample(frame,pts-2*s*u);b,bv=sample(frame,pts+2*s*u);score=np.where(av&bv,np.linalg.norm(a-b,axis=-1)/(255*math.sqrt(3)),-1);ix=score.argmax(1);strength=score[np.arange(3),ix];supported=strength>=p['contrast']
        result.update(boundary_profile_contrasts=strength,boundary_profile_offsets_px=offsets[ix])
        if not supported[1] or supported.sum()<p['minimum_profiles']:return result
        if np.ptp(offsets[ix[supported]])>p['max_profile_spread_widths']*w:result['reason']='incoherent_end_profiles';return result
        front=center+offsets[ix[1]]*u;future=front+np.array(p['background_check_at400'])[:,None]*s*u;rgb,inbounds=sample(frame,future);labels,lv=nearest(mask,future)
        if not inbounds.all():result.update(state='out_of_view',reason='terminal_background_out_of_view');return result
        if np.any(lv&(labels==1)):result.update(state='occlusion_suspected',reason='current_stone_contact');return result
        if np.any(lv&(labels==2)) or self.color(rgb,seed).any():result.update(state='internal_transition',reason='body_continues_after_transition');return result
        _,noise=chroma(rgb)
        if np.any(noise>self.cfg['color']['max_noise']):result['reason']='terminal_background_unknown';return result
        if last['width']>p['closure_width_fraction']*seed['width'] or abs(offsets[ix[1]])>p['max_center_advance_widths']*w:result['reason']='no_supported_cap_closure';return result
        if branch['sign']!=oriented_sign:result['reason']='not_reliably_distal';return result
        result.update(state='visible_end_candidate',reason='current_cap_and_background_supported',xy=front,uncertainty_radius_px=max(2*s,float(np.ptp(offsets[ix[supported]]))));return result

    def grow(self,frame,mask,probability,prior=None,age=99):
        start=time.perf_counter();empty=np.zeros_like(mask,np.uint8);meta=dict(reason='no_trusted_seed',seeds=[],branches=[],tips=[],apply_to_mask=False,tip_metrics=None,abnormal=False,evaluated_pixels=0)
        seeds,records=self.seeds(frame,mask,probability,prior,age);meta['seeds']=records;seed_time=time.perf_counter()
        if len(seeds)!=1:
            if len(seeds)>1:meta['reason']='ambiguous_trusted_seeds'
            meta['timings']=dict(seed_ms=(seed_time-start)*1000,growth_tip_ms=0.,extra_total_ms=(time.perf_counter()-start)*1000);return empty,empty.copy(),empty.copy(),public(meta)
        seed=seeds[0];branches=[self.walk(frame,seed,sign,prior,age) for sign in (-1,1)];allowed=seed['core'].astype(np.uint8);weak_map=np.zeros_like(mask,np.uint8);proposed_allowed=allowed.copy();proposed_weak=weak_map.copy()
        for branch in branches:
            for polygon,weak in branch['polygons']:
                cv2.fillConvexPoly(allowed,np.rint(polygon).astype(np.int32),1)
                if weak:cv2.fillConvexPoly(weak_map,np.rint(polygon).astype(np.int32),1)
            for polygon,weak in branch['rejected']:
                cv2.fillConvexPoly(proposed_allowed,np.rint(polygon).astype(np.int32),1)
                if weak:cv2.fillConvexPoly(proposed_weak,np.rint(polygon).astype(np.int32),1)
        proposed_allowed|=allowed;proposed_weak|=weak_map;area=int(proposed_allowed.sum());meta.update(evaluated_pixels=area,seed_width=seed['width'],seed_length=seed['length'],seed_axis=seed['u'],seed_origin=seed['origin'])
        meta['branches']=[dict(sign=b['sign'],centers=[z['center'] for z in b['sections']],sides=[z['sides'] for z in b['sections']],widths=[z['width'] for z in b['sections']],side_states=[z['states'] for z in b['sections']],weak_sections=sum(z['weak'] for z in b['sections']),stop_reason=b['stop_reason'],distance=b['distance'],abnormal=b['abnormal'],final_u=b['final_u']) for b in branches]
        if area>self.cfg['geometry']['max_evaluated_fraction']*mask.size:
            meta.update(reason='evaluation_area_budget',abnormal=True,proposal_kind='unevaluated_search_scope');candidate=empty;proposal=proposed_allowed;conflict=empty.copy()
        else:
            def eligible(scope,weak):
                out=empty.copy();yy,xx=np.nonzero(scope);rgb=frame[yy,xx];good=self.color(rgb,seed);isweak=weak[yy,xx]>0
                if isweak.any():good[isweak]&=self.color(rgb[isweak],seed,True)
                out[yy[good],xx[good]]=1;return out
            admitted=eligible(allowed,weak_map);diagnostic=eligible(proposed_allowed,proposed_weak);roots=(admitted>0)&seed['core'];nr,rl,rs,_=cv2.connectedComponentsWithStats(roots.astype(np.uint8),connectivity=4)
            ids=[k for k in range(1,nr) if rs[k,cv2.CC_STAT_AREA]>=self.cfg['seed']['min_pixels_at400']*(min(mask.shape)/400)**2]
            def reachable(binary):
                if len(ids)!=1:return empty.copy()
                y,x=np.argwhere(rl==ids[0])[0];_,labels=cv2.connectedComponents(binary,connectivity=4);return ((labels==labels[y,x])&(binary>0)).astype(np.uint8) if labels[y,x] else empty.copy()
            unblocked=reachable(diagnostic);conflict=((unblocked>0)&(mask==1)).astype(np.uint8)
            admitted[mask==1]=0;diagnostic[mask==1]=0;candidate=((reachable(admitted)>0)&(mask==0)).astype(np.uint8);proposal=((reachable(diagnostic)>0)&(mask==0)).astype(np.uint8)
            abnormal=next((b['abnormal'] for b in branches if b['abnormal']),None);reason='candidate_accepted' if candidate.any() else 'no_continuous_new_candidate'
            if len(ids)!=1:reason='color_root_absent_or_ambiguous';candidate[:]=0
            if abnormal:reason=abnormal;candidate[:]=0
            if candidate.sum()>self.cfg['geometry']['max_new_fraction']*mask.size:reason=abnormal='new_area_budget';candidate[:]=0
            meta.update(reason=reason,abnormal=bool(abnormal),proposal_kind='RGB_reachable_new_pixels',root_components=len(ids),eligible_pixels=int(admitted.sum()),pre_stone_reachable_pixels=int(unblocked.sum()),stone_conflict_pixels=int(conflict.sum()))
            s0,s1=seed['start_sections'][0]['center'],seed['start_sections'][1]['center'];h,w=mask.shape
            border=lambda p:min(p[0],p[1],w-1-p[0],h-1-p[1]);diff=border(s1)-border(s0);oriented=1 if diff>0 else -1;orientation='current_border_hint'
            if abs(diff)<self.cfg['history']['orientation_border_distance_difference_diagonal_fraction']*math.hypot(h,w):
                if compatible_history(prior,age,seed['origin'],seed['u'],seed['width'],mask.shape,self.cfg['history']):oriented=1 if np.dot(seed['u'],[math.cos(prior[2]),math.sin(prior[2])])>=0 else -1;orientation='prior_resolved'
                else:oriented=0;orientation='ambiguous'
            meta.update(distal_sign=oriented,orientation_source=orientation)
            meta['tips']=[self.tip(frame,mask,seed,b,oriented) for b in branches]
        end=time.perf_counter();meta['timings']=dict(seed_ms=(seed_time-start)*1000,growth_tip_ms=(end-seed_time)*1000,extra_total_ms=(end-start)*1000);return candidate,proposal,conflict,public(meta)

class GrowthPipeline:
    def __init__(self,cfg,arm,t1_config=None,sequential=False):
        self.arm=arm;self.tracker=None
        if arm not in ARMS:raise ValueError('Unknown arm')
        if arm!='blue_extent':self.legacy=LegacyPipeline(read(ROOT/cfg['legacy_config']),arm,t1_config,sequential);return
        self.growth=BlueExtent(cfg['extent']);self.observer=FastFiberObserver(t1_config) if sequential else None;self.tracker=EvidenceTracker(t1_config) if sequential else None
    def reset(self):
        if self.tracker:self.tracker.reset()
    def run(self,frame,mask,probability,timestamp=0.,source='static'):
        if self.arm!='blue_extent':return self.legacy.run(frame,mask,probability,timestamp,source)
        start=time.perf_counter();prepared=None;prior=None;age=99
        if self.tracker:
            prepared=self.tracker.prepare(mask.shape,timestamp,source);prior=prepared['prior']['shaft'];age=self.tracker.age['shaft']
        c,p,conflict,meta=self.growth.grow(frame,mask,probability,prior,age);t1start=time.perf_counter()
        if self.tracker:
            tracked=self.tracker.update(self.observer.current(frame,probability),None,prepared)
            meta['t1_advice']=dict(direction=meta.get('orientation_source','unknown'),prior_shaft=prior,prior_age=age,current_shaft_status=tracked['groups']['shaft']['status'],context_reset_reason=prepared['context_reset_reason'],source='previous_raw_probability_T1_only')
        end=time.perf_counter();meta['timings'].update(t1_ms=(end-t1start)*1000,extra_total_ms=(end-start)*1000);return c,p,conflict,meta
