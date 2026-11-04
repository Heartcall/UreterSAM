"""Synchronous native-resolution six-panel research video export. No inference."""
from pathlib import Path
import csv,json
import cv2,numpy as np
from .fiber_reference_workspace import ROOT,read,load_mask,sha256
from .fiber_bisector_blue_tip import ARMS
from .run_two_network_experts import array_hash
from .run_fiber_dark_input_control import checked_writer,verify_video
from .replay_classwise_experts import save_png
from ureter_semis.experiment import atomic_json

NAMES={ARMS[0]:'connected_blue_union',ARMS[1]:'contour_sides_cap',ARMS[2]:'bisector_guided_blue_tip'}
def tint(frame,mask):
    image=frame.copy()
    for k,color in [(1,(0,0,255)),(2,(255,210,0))]:
        z=mask==k;image[z]=np.rint(.65*image[z]+.35*np.asarray(color)).astype(np.uint8)
    return image
def dashed(image,points,color,thick=1):
    q=np.rint(points).astype(np.int32)
    for j in range(0,len(q)-1,4):cv2.line(image,tuple(q[j]),tuple(q[min(j+2,len(q)-1)]),color,thick,cv2.LINE_AA)
def analysis(frame,row,cluster):
    image=frame.copy();o=row['outline'];s=min(frame.shape[:2])/400
    if cluster.any():image[cluster>0]=np.rint(.65*image[cluster>0]+.35*np.array([230,0,210])).astype(np.uint8)
    for side,z in enumerate(o.get('side_curves',[])):
        q=np.rint(z).astype(np.int32)
        if len(q)>1:cv2.polylines(image,[q],False,[(0,255,255),(0,140,255)][side],1,cv2.LINE_AA)
    c=o.get('context',{});d=o.get('bisector',{}).get('direction')
    if c and d is not None:
        u=np.asarray(d);origin=np.asarray(c['origin']);xy=np.asarray(o.get('candidate_tip_xy',origin));start=origin+min(0,c['min_t']*.6)*u;end=xy+12*s*u;points=np.linspace(start,end,max(8,int(np.linalg.norm(end-start)/s)));dashed(image,points,(255,255,255))
    cap=o.get('cap',{});q=np.rint(cap.get('curve',[])).astype(np.int32)
    if len(q)>1:
        if cap.get('accepted'):cv2.polylines(image,[q],False,(255,0,230),2,cv2.LINE_AA)
        else:dashed(image,q,(155,155,155))
    final=o.get('final_tip_xy');candidate=o.get('candidate_tip_xy')
    for xy,color,name in [(candidate,(155,155,155),'proposal'),(final,(255,255,255),'RGB tip')]:
        if xy is not None:
            x,y=np.rint(xy).astype(int);cv2.drawMarker(image,(x,y),color,cv2.MARKER_CROSS,9,1);cv2.putText(image,f'{name} ({xy[0]:.1f},{xy[1]:.1f})',(max(2,min(x+6,image.shape[1]-180)),max(12,y-8)),cv2.FONT_HERSHEY_SIMPLEX,.34,color,1,cv2.LINE_AA)
    return image
def grid(frame,raw,masks,rows,cluster,vid,i,t):
    h,w=frame.shape[:2];new=masks[ARMS[-1]];union=masks[ARMS[0]];last=tint(frame,new)
    added=((new==2)&(union!=2)).astype(np.uint8);removed=((union==2)&(new!=2)).astype(np.uint8)
    for z,color in [(added,(0,255,0)),(removed,(0,165,255))]:
        cs,_=cv2.findContours(z,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(last,cs,-1,color,1)
    images=[frame.copy(),tint(frame,raw),tint(frame,masks[ARMS[0]]),tint(frame,masks[ARMS[1]]),last,analysis(frame,rows[ARMS[-1]],cluster)]
    names=['Original RGB - unmarked','Initial two-network masks',NAMES[ARMS[0]],NAMES[ARMS[1]],NAMES[ARMS[2]],'Tip / bisector / cap analysis'];tiles=[]
    for k,(im,name) in enumerate(zip(images,names)):
        header=np.zeros((48,w,3),np.uint8);cv2.putText(header,name,(4,17),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1,cv2.LINE_AA);cap=rows[ARMS[k-2 if k<5 else -1]]['outline']['cap'] if k>=2 else {};state=('cap=APPLIED' if cap.get('accepted') else 'cap=NO / proposal') if k>=3 else 'cap=n/a';cv2.putText(header,f'{vid} frame {i+1}  t={t:.3f}s | {state}',(4,36),cv2.FONT_HERSHEY_SIMPLEX,.34,(255,255,255),1,cv2.LINE_AA);tiles.append(np.vstack([header,im]))
    legend=np.zeros((64,w*3,3),np.uint8);o=rows[ARMS[-1]]['outline'];state=o.get('tip_observation',{}).get('state','unknown');cap=o['cap'];cv2.putText(legend,f'Tip: {state} (current RGB only; no independent tip GT) | Cap: {cap["reason"]}',(5,19),cv2.FONT_HERSHEY_SIMPLEX,.41,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(legend,'Cyan fiber / red stone | green add / orange remove | purple distal cluster | white bisector + RGB tip',(5,40),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1,cv2.LINE_AA);cv2.putText(legend,'Yellow/orange side fits | magenta solid APPLIED cap / gray dashed PROPOSAL | RESEARCH ONLY; default unchanged',(5,59),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1,cv2.LINE_AA)
    image=np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:]),legend]);assert np.array_equal(image[48:48+h,:w],frame),'Raw panel was marked or shifted';return image

def comparison(cfg,out,info):
    vid=info['video_id'];h,w=info['height'],info['width'];cap=cv2.VideoCapture(info['path']);streams={a:(out/f'full_video/{a}/{vid}/frames.jsonl').open(encoding='utf-8') for a in ARMS};path=out/f'{vid}_six_panel.mp4';size=(w*3,(h+48)*2+64);writer=checked_writer(path,info['fps'],size);focus=None;single=None;rowscsv=[];records=[];count=0;focuscount=0;boarddir=out/'focus_screenshots';boarddir.mkdir(exist_ok=True);rawdir=out/f'full_video/raw2/{vid}/masks';rawdir.mkdir(parents=True);source=read(out/f'source_{vid}.json')['frames'];special=[1209,1233,1250,1266,1291]
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            rows={a:json.loads(next(f)) for a,f in streams.items()};rgb_hash=array_hash(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));assert rgb_hash==source[count]['rgb_raw_sha256'];assert all(r['source_rgb_raw_sha256']==rgb_hash and r['frame_index']==count and r['timestamp_seconds']==source[count]['timestamp_seconds'] for r in rows.values());raw=load_mask(rows[ARMS[0]]['raw_mask_reference'],(h,w),{0,1,2});save_png(rawdir/f'{count:06d}.png',raw);assert array_hash(raw)==rows[ARMS[0]]['mask_raw_sha256'];masks={a:load_mask(out/f'full_video/{a}/{vid}/experimental/{count:06d}.png',(h,w),{0,1,2}) for a in ARMS};assert all(array_hash(masks[a])==rows[a]['final_mask_sha256'] for a in ARMS);cluster=load_mask(out/f'full_video/{ARMS[-1]}/{vid}/distal_cluster/{count:06d}.png',(h,w),{0,1});image=grid(frame,raw,masks,rows,cluster,vid,count,source[count]['timestamp_seconds']);writer.write(image)
            if vid=='video_2' and (count+1==1209 or 1233<=count+1<=1291):
                save_png(boarddir/f'video2_frame{count+1:04d}.png',image)
                if count+1==1209:single=checked_writer(out/'video_2_frame1209_six_panel.mp4',info['fps'],size);single.write(image);single.release();single=None
                else:
                    if focus is None:focus=checked_writer(out/'video_2_frames1233-1291_six_panel.mp4',info['fps'],size)
                    focus.write(image);focuscount+=1
            for a in ARMS:
                o=rows[a]['outline'];c=o['cap'];xy=o.get('final_tip_xy') if a==ARMS[-1] else c.get('candidate_xy') if c.get('accepted') else None;proposal=o.get('candidate_tip_xy',c.get('candidate_xy'));apex=c.get('apex_xy',c.get('candidate_xy'));state=o.get('tip_observation',{}).get('state',c.get('state','unknown'));rowscsv.append(dict(video_id=vid,frame_index=count,display_frame=count+1,timestamp_seconds=source[count]['timestamp_seconds'],arm=a,tip_state=state,final_x=xy[0] if xy is not None else None,final_y=xy[1] if xy is not None else None,proposal_x=proposal[0] if proposal is not None else None,proposal_y=proposal[1] if proposal is not None else None,model_apex_x=apex[0] if apex is not None else None,model_apex_y=apex[1] if apex is not None else None,cap_applied=c['accepted'],reason=c['reason'],tip_accuracy=None))
            records.append(dict(frame_index=count,source_rgb_sha256=rgb_hash,raw_panel_sha256=array_hash(cv2.cvtColor(image[48:48+h,:w],cv2.COLOR_BGR2RGB)),timestamp_seconds=source[count]['timestamp_seconds'],mask_sha256={a:array_hash(v) for a,v in masks.items()}));count+=1
            if count%500==0:print('six_panel_export',vid,count,flush=True)
    finally:
        cap.release();writer.release()
        if focus is not None:focus.release()
        if single is not None:single.release()
        for f in streams.values():f.close()
    assert count==info['frames']
    result=verify_video(path,count);result.update(width=size[0],height=size[1],fps=info['fps'],native_panel_size=[w,h],raw_panel_unmarked=True,source_sync_records=records)
    with (out/f'{vid}_tip_coordinates.csv').open('w',encoding='utf-8-sig',newline='') as f:z=csv.DictWriter(f,fieldnames=list(rowscsv[0]));z.writeheader();z.writerows(rowscsv)
    atomic_json(out/f'{vid}_tip_coordinates.json',rowscsv)
    if vid=='video_2':
        assert focuscount==59;result['focus59']=verify_video(out/'video_2_frames1233-1291_six_panel.mp4',59);result['frame1209']=verify_video(out/'video_2_frame1209_six_panel.mp4',1);result['fixed_screenshots']={str(i):dict(path=str(boarddir/f'video2_frame{i:04d}.png'),sha256=sha256(boarddir/f'video2_frame{i:04d}.png')) for i in special}
    atomic_json(out/f'{vid}_media_identity.json',result);return {k:v for k,v in result.items() if k!='source_sync_records'}
