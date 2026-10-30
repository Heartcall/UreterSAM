"""GT-conditioned measurements only. These functions never refine a segmentation."""
from collections import Counter
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from .classwise_experts import native_outputs
from .replay_classwise_experts import pil_input
from ureter_semis.video_metrics import estimated_fov


@torch.inference_mode()
def frozen_probabilities(experts, frame):
    """Exactly the original3 forwards and logit fusion, with retained diagnostic fields."""
    image=pil_input(frame,experts.size,experts.device)
    old=experts.stone.models[0](image)
    first=experts.fiber.models[0](image);second=experts.fiber.models[1](image)
    grids={'old190':old.softmax(1)[0],'a0':((first+second)/2).softmax(1)[0],
           'a0_member1':first.softmax(1)[0],'a0_member2':second.softmax(1)[0]}
    native=native_outputs(grids['old190'],grids['a0'],frame.shape[:2])
    arrays={name+'_grid':value.cpu().numpy().copy() for name,value in grids.items()}
    for name,value in grids.items():
        p=native['stone_probability'] if name=='old190' else native['fiber_probability'] if name=='a0' else F.interpolate(value[None].float(),size=frame.shape[:2],mode='bilinear',align_corners=False)[0]
        arrays[name+'_native']=p.cpu().numpy().copy()
    arrays['raw_classwise']=native['mask'].cpu().numpy().copy()
    arrays['old190_mask']=native['stone_mask'].cpu().numpy().copy()
    return arrays


def reference_axis(frame,target,settings):
    fg=target==2;yy,xx=np.where(fg)
    if len(xx)<settings['minimum_reference_pixels']:return {'valid':False,'reason':'too_few_reference_pixels'},None
    points=np.column_stack([xx,yy]).astype(np.float32);origin=points.mean(0)
    _,vectors,values=cv2.PCACompute2(points,mean=None,maxComponents=2)
    ratio=float(values[0,0]/max(float(values[1,0]),1e-6));u=vectors[0];s=(points-origin)@u
    fov=estimated_fov(frame,fg.shape);distance=cv2.distanceTransform(np.pad(fov.astype(np.uint8),1),cv2.DIST_L2,5)[1:-1,1:-1]
    low,high=np.percentile(s,[5,95]);band=max(2*min(fg.shape)/400,float(np.std(s))*.1)
    da=float(np.median(distance[yy[np.abs(s-low)<=band],xx[np.abs(s-low)<=band]]));db=float(np.median(distance[yy[np.abs(s-high)<=band],xx[np.abs(s-high)<=band]]))
    if ratio<settings['minimum_axis_eigenvalue_ratio']:return {'valid':False,'reason':'reference_axis_not_elongated','eigenvalue_ratio':ratio},None
    if abs(da-db)<min(fg.shape)*settings['minimum_end_distance_difference_fraction']:return {'valid':False,'reason':'reference_axis_orientation_ambiguous','end_distances':[da,db],'eigenvalue_ratio':ratio},None
    if da>db:u=-u;s=-s
    start,end=float(s.min()),float(s.max())
    if end<=start:return {'valid':False,'reason':'zero_reference_length'},None
    full_y,full_x=np.indices(fg.shape,dtype=np.float32);t=((full_x-origin[0])*u[0]+(full_y-origin[1])*u[1]-start)/(end-start)
    return {'valid':True,'origin':origin.astype(float).tolist(),'direction':u.astype(float).tolist(),'projected_start':start,'projected_end':end,'eigenvalue_ratio':ratio,'role':'reference_conditioned_axis_NOT_anatomical_tip'},t


def appearance_fields(frame,probabilities,axis):
    bgr=frame.astype(np.float32)/255;rgb=bgr[:,:,::-1];hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV);lab=cv2.cvtColor(bgr,cv2.COLOR_BGR2Lab)
    p=probabilities['a0_native'];p1=probabilities['a0_member1_native'];p2=probabilities['a0_member2_native']
    competitor=np.maximum(p[0],p[1]);h=hsv[:,:,0].astype(np.float32)*2*np.pi/180
    total=rgb.sum(2);gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY);blur=cv2.GaussianBlur(gray,(0,0),max(1,min(gray.shape)/400*3))
    gx=cv2.Sobel(bgr,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(bgr,cv2.CV_32F,0,1,ksize=3)
    gradient=np.sqrt((gx*gx+gy*gy).sum(2))
    fields={'fiber_probability':p[2],'competitor_probability':competitor,'fiber_margin':p[2]-competitor,
        'entropy':-(p*np.log(np.maximum(p,1e-8))).sum(0)/np.log(3),'member_fiber_difference':np.abs(p1[2]-p2[2]),
        'red':rgb[:,:,0],'green':rgb[:,:,1],'blue':rgb[:,:,2],'saturation':hsv[:,:,1].astype(np.float32)/255,'value':hsv[:,:,2].astype(np.float32)/255,
        'hue_sin':np.sin(h),'hue_cos':np.cos(h),'lab_l':lab[:,:,0],'lab_a':lab[:,:,1],'lab_b':lab[:,:,2],
        'chromatic_red':rgb[:,:,0]/np.maximum(total,1e-6),'chromatic_green':rgb[:,:,1]/np.maximum(total,1e-6),'chromatic_blue':rgb[:,:,2]/np.maximum(total,1e-6),
        'blue_excess_diagnostic_not_probability':(rgb[:,:,2]-.5*(rgb[:,:,0]+rgb[:,:,1]))/np.maximum(total,1e-6),
        'local_luminance_residual':gray-blur,'color_gradient':gradient}
    if axis['valid']:
        vx,vy=-axis['direction'][1],axis['direction'][0]
        fields['normal_color_gradient']=np.sqrt(((gx*vx+gy*vy)**2).sum(2))
    return fields


def distributions(fields,selection,quantiles):
    out={'pixels':int(selection.sum())}
    for name,a in fields.items():
        values=a[selection]
        out[name]={'n':int(len(values)),'mean':float(values.mean()),'quantiles':np.percentile(values,quantiles).astype(float).tolist()} if len(values) else None
    if selection.any():
        # Circular hue is descriptive; low-saturation pixels are downweighted, not classified as background.
        weight=fields['saturation'][selection];ss=float((fields['hue_sin'][selection]*weight).sum());cc=float((fields['hue_cos'][selection]*weight).sum())
        out['saturation_weighted_hue_degrees']=float((np.rad2deg(np.arctan2(ss,cc))+360)%360) if weight.sum()>1e-8 else None
        out['hue_resultant_length']=float(np.hypot(ss,cc)/max(float(weight.sum()),1e-8))
    return out


def loss_partition(target,raw,new,accepted,hypothesis,field,threshold,value):
    """Mutually exclusive first exclusion stage for original true-positive pixels."""
    original=(target==2)&(raw==2);lost=original&(new!=2)
    if not accepted:
        counts={'fit_rejected':int(lost.sum()),'outside_hypothesis':0,'foreground_field_rejected':0,'darkness_rejected':0}
    else:
        counts={'fit_rejected':0,'outside_hypothesis':int((lost&~hypothesis).sum()),
            'foreground_field_rejected':int((lost&hypothesis&(field<threshold)).sum()),
            'darkness_rejected':int((lost&hypothesis&(field>=threshold)&(value<=10)).sum())}
    counts['unexplained']=int(lost.sum())-sum(counts.values());counts['lost_tp']=int(lost.sum())
    counts['gained_tp']=int(((target==2)&(raw!=2)&(new==2)).sum())
    counts['removed_fp']=int(((target!=255)&(target!=2)&(raw==2)&(new!=2)).sum())
    counts['added_fp']=int(((target!=255)&(target!=2)&(raw!=2)&(new==2)).sum())
    return counts


def whole_fiber_diagnosis(target,arrays,settings):
    p=arrays['a0_native'];raw=arrays['raw_classwise'];truth=target==2;fn=truth&(raw!=2);winner=p.argmax(0)
    member1=arrays['a0_member1_native'].argmax(0);member2=arrays['a0_member2_native'].argmax(0)
    nearest=cv2.resize(arrays['a0_grid'].argmax(0).astype(np.uint8),(target.shape[1],target.shape[0]),interpolation=cv2.INTER_NEAREST)
    edges=settings['foreground_probability_bins'];bins={}
    for i,(lo,hi) in enumerate(zip(edges,edges[1:])):
        bins[f'[{lo},{hi}{"]" if i==len(edges)-2 else ")"}']=int((fn&(p[2]>=lo)&((p[2]<=hi) if i==len(edges)-2 else (p[2]<hi))).sum())
    return {'reference_pixels':int(truth.sum()),'raw_tp':int((truth&(raw==2)).sum()),'raw_fn':int(fn.sum()),
        'raw_fn_background_winner':int((fn&(winner==0)).sum()),'raw_fn_stone_winner':int((fn&(winner==1)).sum()),'raw_fn_probability_bins_diagnostic':bins,
        'fn_either_member_predicts_fiber':int((fn&((member1==2)|(member2==2))).sum()),'fn_both_members_reject_fiber':int((fn&(member1!=2)&(member2!=2)).sum()),
        'nearest_argmax_counterfactual_lost_tp':int((truth&(raw==2)&(nearest!=2)).sum()),'nearest_argmax_counterfactual_gained_tp':int((truth&(raw!=2)&(nearest==2)).sum()),
        'tip_metrics':None,'role':'Whole-fiber measured errors; no anatomical tip labels; nearest-label mapping is a diagnostic counterfactual, not a changed baseline'}
