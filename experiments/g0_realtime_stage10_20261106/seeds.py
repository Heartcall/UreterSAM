"""Reuse the outer 4-connected component; preserve the frozen evaluator math."""
import cv2
import numpy as np
from tools.fiber_aggressive_blue import (
    _pca,angle,chroma,sample,blue,fixed_axis,compatible_history,direction)


def evaluate(self,frame,mask,probability,prior,age,labels,stats,label,roi,counts):
    # Same one-component input as AggressiveVariant.seeds called by the
    # QualifiedSeedObserver. Its local label is always 1; outer relabeling
    # remains in qualified_seeds. Only component discovery/coordinates differ.
    cfg=self.cfg['seed'];s=min(mask.shape)/400;ids=[label]
    if len(ids)>cfg['max_components']:return [],[dict(reason='excessive_seed_hypotheses',components=len(ids))]
    records=[];qualified=[];bgcfg=self.full['aggressive_parameters']['background']
    for k in ids:
        if roi:
            x,y,w,h=map(int,stats[k,:4]);yy,xx=np.nonzero(labels[y:y+h,x:x+w]==k);yy+=y;xx+=x
            counts['coordinate_evaluated_pixels']+=w*h
        else:
            yy,xx=np.nonzero(labels==k)
            counts['coordinate_evaluated_pixels']+=mask.size
        counts['coordinate_full_pixels']+=mask.size
        points=np.stack([xx,yy],-1).astype(float);sub=points[np.linspace(0,len(points)-1,min(len(points),cfg['max_points'])).astype(int)]
        origin,u,ends,width=_pca(sub);anchor=None
        if self.fixed:
            anchor,u=fixed_axis(sub,mask.shape,self.full['aggressive_parameters']['fixed_direction']);n=np.array([-u[1],u[0]]);axis=(sub-origin)@u;normal=(sub-origin)@n;ends=np.percentile(axis,[2,98]);width=max(float(np.ptp(np.percentile(normal,[2,98]))),1.)
        length=float(ends[1]-ends[0]);record=dict(label=1,pixels=len(points),length=length,width=width,reason='short_or_nonelongated_seed')
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
        seed=dict(core=labels==k,mu=mu,v=mu-1/3,bg=bg,gap=gap,origin=origin,u=u,width=width,length=length,label=1)
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


def qualified_seeds(self,frame,mask,probability,prior,age,roi,counts):
    cfg=self.cfg['seed'];s=min(mask.shape)/400
    _,labels,stats,_=cv2.connectedComponentsWithStats(((mask==2)&(probability[2]>=cfg['P_min'])).astype(np.uint8),connectivity=4)
    counts['outer_component_passes']+=1
    ids=[k for k in range(1,len(stats)) if stats[k,4]>=cfg['min_pixels_at400']*s*s]
    self.chosen=0;self.orientation='unavailable'
    if len(ids)>32:return [],[dict(reason='seed_computation_budget',components=len(ids),limit=32)]
    qualified=[];records=[]
    for k in sorted(ids,key=lambda k:(-stats[k,4],k)):
        counts['inner_component_passes_saved']+=1
        seeds,rows=evaluate(self,frame,mask,probability,prior,age,labels,stats,k,roi,counts)
        for q in seeds:q['label']=k;qualified.append(q)
        for r in rows:r.update(label=k,raw_candidate_count=len(ids));records.append(r)
    self.chosen=1 if len(qualified)==1 else 0;self.orientation='fixed_edge_to_center' if len(qualified)==1 else 'ambiguous_qualified_subjects' if len(qualified)>1 else 'unavailable'
    return qualified,records
