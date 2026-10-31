"""Offline reporting on the frozen input control; no model or training calls."""
from pathlib import Path
import csv
import json
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from tools.fiber_reference_workspace import read,load_mask,sha256
from tools.fiber_photometric_input import ARMS,rgb_grid,transform_rgb
from tools.run_fiber_blue_aplus import frame_read,panel
from tools.run_fiber_dark_input_control import FIXED
from ureter_semis.experiment import atomic_json

OUT=ROOT/'runs/a1_fiber_dark_input_control_20261030'
SHORT={'raw2':'Raw2','fiber_vlift':'Vlift','fiber_channel_gamma':'ChannelGamma'}


def write_jpg(path,image):
    ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,92])
    if not ok:raise RuntimeError('Board encoding failed')
    encoded.tofile(str(path))


def probability_stats(p,region):
    x=p[region]
    return {'pixels':int(len(x)),'mean':float(x.mean(dtype=np.float64)) if len(x) else None,
            'p05_p25_p50_p75_p95':np.percentile(x,[5,25,50,75,95]).tolist() if len(x) else None,
            'counts_Pbelow01':int((x<.1).sum()),'counts_Patleast09':int((x>=.9).sum())}


def heat(p):return cv2.applyColorMap(np.rint(np.clip(p,0,1)*255).astype(np.uint8),cv2.COLORMAP_VIRIDIS)


def error_view(frame,target,mask):
    image=(frame*.5).astype(np.uint8);image[(target==2)&(mask!=2)]=(0,0,255)
    image[(target!=255)&(target!=2)&(mask==2)]=(255,0,255)
    return image


def board(frame,target,masks,probabilities,index,label):
    grid=rgb_grid(frame);h,w=frame.shape[:2]
    previews=[]
    for arm in ARMS[1:]:
        preview=cv2.cvtColor(np.rint(transform_rgb(grid,arm)*255).astype(np.uint8),cv2.COLOR_RGB2BGR)
        previews.append(cv2.resize(preview,(w,h),interpolation=cv2.INTER_NEAREST))
    lab=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)[:,:,0];levels=np.minimum(lab//85,2)
    colors=np.array([[150,50,20],[30,150,170],[180,230,230]],np.uint8);bright=colors[levels]
    if target is None:
        gt=np.zeros_like(frame);cv2.putText(gt,'NO PIXEL REFERENCE',(20,200),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
    else:gt=panel(frame,'GT',target,index)[34:]
    top=[panel(frame,label,None,index),panel(gt,'Reviewed GT (no tip GT)',None,index),panel(bright,'Original Lab: low/mid/high',None,index)]
    top.extend(panel(preview,'Fiber input256 '+SHORT[arm],None,index) for preview,arm in zip(previews,ARMS[1:]))
    middle=[panel(frame,SHORT[arm],masks[arm],index) for arm in ARMS]
    for arm in ARMS[1:]:
        delta=(frame*.5).astype(np.uint8);delta[(masks['raw2']!=2)&(masks[arm]==2)]=(0,255,0);delta[(masks['raw2']==2)&(masks[arm]!=2)]=(0,0,255)
        middle.append(panel(delta,SHORT[arm]+' green+/red-',None,index))
    bottom=[panel(heat(probabilities[arm]),'P_fiber '+SHORT[arm]+': dark0/yellow1',None,index) for arm in ARMS]
    for arm in ARMS[1:]:
        image=error_view(frame,target,masks[arm]) if target is not None else np.zeros_like(frame)
        bottom.append(panel(image,SHORT[arm]+' red FN/pink FP',None,index))
    return np.vstack([np.hstack(top),np.hstack(middle),np.hstack(bottom)])


def main():
    summary=read(OUT/'summary.json');cfg=read(OUT/'preflight.json')['config']
    folder=OUT/'analysis';folder.mkdir(exist_ok=False);(folder/'all_static_boards').mkdir()
    tables=[];diagnoses={};files={}
    for scope,key in [('video_native','video_manifest'),('image_native','image_manifest')]:
        rows=read(ROOT/cfg['data'][key])['val'];results={a:read(OUT/f'{scope}/{a}.json') for a in ARMS}
        diagnoses[scope]=[]
        for i,row in enumerate(rows):
            frame=frame_read(row['image']);target=load_mask(row['mask'],frame.shape[:2],{0,1,2,255});masks={};probs={}
            for arm in ARMS:
                masks[arm]=load_mask(OUT/f'{scope}/{arm}/{i:04d}.png',target.shape,{0,1,2})
                with np.load(OUT/f'{scope}/{arm}/probabilities/{i:04d}.npz') as z:probs[arm]=z['fiber'][2].copy()
            originalFN=(target==2)&(masks['raw2']!=2);below=originalFN&(probs['raw2']<.1);low=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)[:,:,0]<85
            diagnosis={'sample_id':row['sample_id'],'video_id':row.get('video_id'),'frame_index':row.get('frame_index'),'arms':{}}
            for arm in ARMS:
                record=results[arm]['records'][i];f=results[arm]['summary']['frames'][i];cm=np.array(f['confusion_matrix']);metrics={}
                for c,name in [(1,'stone'),(2,'fiber')]:
                    tp=int(cm[c,c]);fp=int(cm[:,c].sum()-tp);fn=int(cm[c,:].sum()-tp)
                    metrics.update({name+'_TP':tp,name+'_FP':fp,name+'_FN':fn,name+'_Dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                        name+'_Precision':tp/(tp+fp) if tp+fp else None,name+'_Recall':tp/(tp+fn) if tp+fn else None,
                        name+'_boundary_F':results[arm]['summary']['frames'][i]['boundaries'][str(c)]})
                newFP=(target!=255)&(target!=2)&(masks[arm]==2)&(masks['raw2']!=2)
                diagnosis['arms'][arm]={'baseline_FN':probability_stats(probs[arm],originalFN),
                    'baseline_FN_Pbelow01':probability_stats(probs[arm],below),'baseline_FN_lowLab':probability_stats(probs[arm],originalFN&low),
                    'introduced_FP':probability_stats(probs[arm],newFP),'highconfidence_FP':int(((target!=255)&(target!=2)&(masks[arm]==2)&(probs[arm]>=.9)).sum())}
                table={'scope':scope,'arm':arm,'index':i,'sample_id':row['sample_id'],'video_id':row.get('video_id'),'frame_index':row.get('frame_index'),
                    **{k:v for k,v in metrics.items() if not isinstance(v,dict)},
                    **{c+'_'+k:v for c in ['stone','fiber'] for k,v in record['paired_'+c].items()},
                    'lowLab_GT':record['brightness']['Lab_Lstar'][0]['gt_pixels'],'lowLab_TP':record['brightness']['Lab_Lstar'][0]['tp'],
                    'lowLab_FP':record['brightness']['Lab_Lstar'][0]['fp'],'lowLab_FN':record['brightness']['Lab_Lstar'][0]['fn'],
                    'lowLab_Recall':record['brightness']['Lab_Lstar'][0]['recall'],'recovered_baselineFN_Pbelow01':record['baseline_FN_Pbelow01_recovered_TP'],
                    'lost_baseline_highconfidence_TP':record['baseline_highconfidence_TP_lost'],'changed_pixels':record['changed_pixels']}
                for c,name in [('1','stone'),('2','fiber')]:
                    b=f['boundaries'][c];p=b['matched_predicted']/b['predicted_boundary'] if b['predicted_boundary'] else None;r=b['matched_target']/b['target_boundary'] if b['target_boundary'] else None
                    table[name+'_boundary_F']=2*p*r/(p+r) if p is not None and r is not None and p+r else 0 if b['target_boundary'] else None
                tables.append(table)
            diagnoses[scope].append(diagnosis)
            label=(row.get('video_id','image')+' '+str(row.get('frame_index',i)+1)) if row.get('video_id') else 'Image-dev '+str(i+1)
            image=board(frame,target,masks,probs,row.get('frame_index',i),label)
            path=folder/f'all_static_boards/{scope}_{i:04d}.jpg';write_jpg(path,image);files[str(path.relative_to(OUT))]=sha256(path)
            if scope=='video_native' and row['frame_index'] in FIXED[row['video_id']]:
                path=folder/f'{row["video_id"]}_frame{row["frame_index"]+1}_comparison.jpg';write_jpg(path,image);files[str(path.relative_to(OUT))]=sha256(path)
            if (i+1)%40==0:print(scope,i+1,flush=True)
    for info in read(OUT/'preflight.json')['videos']:
        vid=info['video_id'];cap=cv2.VideoCapture(info['path']);i=0
        references={r['frame_index'] for r in read(ROOT/cfg['data']['video_manifest'])['val'] if r['video_id']==vid}
        while True:
            ok,frame=cap.read()
            if not ok:break
            if i in FIXED[vid] and i not in references:
                masks={a:load_mask(OUT/f'full_video/{a}/{vid}/masks/{i:06d}.png',frame.shape[:2],{0,1,2}) for a in ARMS};probs={}
                for a in ARMS:
                    with np.load(OUT/f'full_video/{a}/{vid}/fixed_probabilities/{i:06d}.npz') as z:probs[a]=z['fiber'][2].copy()
                image=board(frame,None,masks,probs,i,vid+' UNLABELLED')
                path=folder/f'{vid}_frame{i+1}_comparison.jpg';write_jpg(path,image);files[str(path.relative_to(OUT))]=sha256(path)
            i+=1
        cap.release()
    with (folder/'per_frame_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(tables[0]));writer.writeheader();writer.writerows(tables)
    atomic_json(folder/'probability_response.json',diagnoses)
    compact={}
    for scope in ['video_native','image_native']:
        compact[scope]={}
        for arm in ARMS:
            d=read(OUT/f'{scope}/{arm}.json')
            compact[scope][arm]={'metrics':d['summary']['metrics'],'boundaries':d['summary']['boundaries'],'brightness':d['brightness'],
                'paired_changes':d['paired_changes'],'semantic_centerline':d['semantic_centerline'],
                'changed_frames':sum(r['changed_pixels']>0 for r in d['records']),
                'baselineFN_Pbelow01_recovered_TP':sum(r['baseline_FN_Pbelow01_recovered_TP'] for r in d['records']),
                'baseline_highconfidence_TP_lost':sum(r['baseline_highconfidence_TP_lost'] for r in d['records']),
                'groups':{k:{s:v[s] for s in ['frames','brightness','semantic_centerline']} for k,v in d['groups'].items()}}
    fixed=[]
    rows=read(ROOT/cfg['data']['video_manifest'])['val']
    for i,row in enumerate(rows):
        if row['frame_index'] not in FIXED[row['video_id']]:continue
        fixed.append({'video_id':row['video_id'],'display_frame':row['frame_index']+1,'models':{a:read(OUT/f'video_native/{a}.json')['records'][i] for a in ARMS}})
    atomic_json(folder/'compact_results.json',compact);atomic_json(folder/'fixed_reference_results.json',fixed)
    atomic_json(folder/'identity.json',{'generator_sha256':sha256(Path(__file__)),'frozen_result_summary_sha256':sha256(OUT/'summary.json'),
        'artifacts':{**files,**{str(p.relative_to(OUT)):sha256(p) for p in folder.iterdir() if p.is_file()}},
        'new_model_forwards':0,'tip_metrics':None,'whole_video_accuracy':None})
    print(json.dumps({'boards':len(files),'rows':len(tables),'new_model_forwards':0}),flush=True)


if __name__=='__main__':main()
