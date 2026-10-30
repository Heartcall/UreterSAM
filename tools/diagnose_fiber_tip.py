"""D0 frozen inference, cached historical-output attribution and full-video cost."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

import cv2
import numpy as np
from PIL import Image
import torch

from ureter_semis.experiment import atomic_json,input_identity,sha256,source_identity,seed_training,runtime_identity
from .classwise_experts import ClasswiseExperts
from .direct_fiber_template import visible
from .fiber_tip_diagnostics import frozen_probabilities,reference_axis,appearance_fields,distributions,whole_fiber_diagnosis,loss_partition
from .run_classwise_expert_control import summarize
from .diagnose_joint_failures import paired_changes
from .run_fiber_shape_control import core
from .replay_classwise_experts import model_state_hash,save_png,stats

ROOT=Path(__file__).resolve().parents[1];CONFIG=ROOT/'configs/fiber_tip_d0_20261029.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def load(p):
    with Image.open(p) as im:return np.asarray(im).copy()
def frame_read(p):return cv2.cvtColor(np.asarray(Image.open(p).convert('RGB')),cv2.COLOR_RGB2BGR)


def historical_paths(row,i,video):
    filename=f'{i:04d}.png';g=ROOT/'runs/a1_fiber_geometry_20261028';d=ROOT/'runs/a1_direct_fiber_template_20261029';c=ROOT/'runs/a1_fiber_component_gate_20261029';n=ROOT/'runs/a1_direct_neural_observation_20261029'
    if video:
        vid=row['video_id'];index=f'{row["frame_index"]:06d}.png'
        paths={'geometry':g/vid/'geometry'/index,'geometry_causal':g/vid/'geometry_causal'/index,
            'fixed_color':d/vid/'image_template'/index,'fixed_soft':d/vid/'soft_template'/index,'fixed_causal':d/vid/'causal_template'/index}
        scope='video_native'
    else:
        paths={'geometry':g/'image_native/geometry'/filename,'fixed_color':d/'image_native/image_template'/filename,'fixed_soft':d/'image_native/soft_template'/filename};scope='image_native'
    paths.update({'component3':c/scope/'classwise_component_gate_3net'/filename,'component1':c/scope/'old190_component_gate_1net'/filename,
        'fixed_neural':n/scope/'fixed_neural_observation_template'/filename,'raw_classwise':n/scope/'raw_classwise'/filename})
    return paths


def history_context():
    old=read(ROOT/'runs/a1_direct_fiber_template_20261029/summary.json');new=read(ROOT/'runs/a1_direct_neural_observation_20261029/summary.json')
    logs={v:{r['frame_index']:r for r in [json.loads(line) for line in (ROOT/f'runs/a1_direct_fiber_template_20261029/{v}/frames.jsonl').read_text(encoding='utf-8').splitlines()]} for v in ('video_1','video_2')}
    return old,new,logs


def historical_diagnosis(row,i,video,frame,target,arrays,axis,t,context,prior,config):
    old,new,logs=context;paths=historical_paths(row,i,video);records=logs[row['video_id']][row['frame_index']]['outputs'] if video else old['image']['records'][i]['outputs']
    neural=(new['video29'] if video else new['image'])['records'][i]['outputs']['fixed_neural_observation_template']
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV);bins=prior['histogram_bins'];color=np.asarray(prior['appearance_probability'])[(hsv[:,:,0].astype(int)*bins//180).clip(0,bins-1),(hsv[:,:,1].astype(int)*bins//256).clip(0,bins-1)]
    record_names={'fixed_color':'image_template','fixed_soft':'soft_template','fixed_causal':'causal_template'};result={}
    for name,path in paths.items():
        if name=='raw_classwise':continue
        mask=load(path);raw=arrays['old190_mask'] if name=='component1' else arrays['raw_classwise'];changes=paired_changes(target,raw,mask,2)
        item={'baseline':'old190' if name=='component1' else 'classwise','changes':changes,'whole_fiber_fn':int(((target==2)&(mask!=2)).sum()),'tip_metrics':None}
        if axis['valid']:
            fg=target==2;z=np.clip(t,0,1);bounds=[0,.25,.5,.85,1.]
            item['reference_axis_proxy_changes']={}
            for j,(lo,hi) in enumerate(zip(bounds,bounds[1:])):
                selected=fg&(z>=lo)&((z<=hi) if j==3 else (z<hi))
                item['reference_axis_proxy_changes'][str(j)]={'reference_pixels':int(selected.sum()),'lost_tp':int((selected&(raw==2)&(mask!=2)).sum()),'gained_tp':int((selected&(raw!=2)&(mask==2)).sum())}
        if name.startswith('fixed_'):
            record=neural if name=='fixed_neural' else records[record_names[name]]
            _,hyp=visible(frame,record['pose'],record['accepted'],prior,config)
            field=arrays['a0_native'][2] if name=='fixed_neural' else color
            item['first_exclusion']=loss_partition(target,raw,mask,record['accepted'],hyp,field,config['acceptance']['visible_appearance_min'],hsv[:,:,2])
            if item['first_exclusion']['unexplained']:raise ValueError('Historical fixed-template exclusion not explained')
        result[name]=item
    support=(color>=.5)&(hsv[:,:,2]>10);p=arrays['a0_native'][2];raw=arrays['raw_classwise'];truth=target==2
    result['threshold_counterfactual_not_a_new_scheme']={'reference_supported_color':int((truth&support).sum()),
        'raw_tp_removed_by_probability_05_only':int((truth&(raw==2)&(p<.5)).sum()),
        'raw_tp_removed_by_darkness_after_probability_05':int((truth&(raw==2)&(p>=.5)&(hsv[:,:,2]<=10)).sum()),
        'reference_supported_neural_05_and_value':int((truth&(p>=.5)&(hsv[:,:,2]>10)).sum())}
    return result


def static_evaluate(rows,video,experts,out,settings,context,prior,base,progress):
    folder=out/('video_native' if video else 'image_native');folder.mkdir()
    for n in ('probabilities','raw_classwise','old190'):(folder/n).mkdir()
    records=[];targets=[];predictions={'classwise':[],'old190':[]};pool=defaultdict(lambda:defaultdict(list));equal=0
    for i,row in enumerate(rows):
        frame=frame_read(row['image']);arrays=frozen_probabilities(experts,frame)
        np.savez_compressed(folder/'probabilities'/f'{i:04d}.npz',**{k:v for k,v in arrays.items() if k not in ('raw_classwise','old190_mask')})
        save_png(folder/'raw_classwise'/f'{i:04d}.png',arrays['raw_classwise']);save_png(folder/'old190'/f'{i:04d}.png',arrays['old190_mask'])
        equal+=int((arrays['raw_classwise']!=load(historical_paths(row,i,video)['raw_classwise'])).sum())
        if equal:raise ValueError('Frozen classwise output differs from historical native baseline')
        target=load(row['mask']);axis,t=reference_axis(frame,target,settings);fields=appearance_fields(frame,arrays,axis)
        truth=target==2;raw=arrays['raw_classwise'];selections={'whole_reference_fiber':truth,'raw_fiber_false_positive':(target!=255)&~truth&(raw==2)}
        if axis['valid']:
            z=np.clip(t,0,1);bound=settings['axial_boundaries']
            for j,(lo,hi) in enumerate(zip(bound,bound[1:])):selections[f'axial_proxy_{j}']=truth&(z>=lo)&((z<=hi) if j==len(bound)-2 else (z<hi))
        grouped={name:distributions(fields,selection,settings['quantiles']) for name,selection in selections.items()}
        for name,selection in selections.items():
            for k,a in fields.items():
                if selection.any():pool[name][k].append(a[selection].astype(np.float32))
        conditions={}
        for feature,bins in [('saturation',settings['saturation_bins']),('value',settings['value_bins'])]:
            conditions[feature]=[]
            for j,(lo,hi) in enumerate(zip(bins,bins[1:])):
                selected=truth&(fields[feature]>=lo)&((fields[feature]<=hi) if j==len(bins)-2 else (fields[feature]<hi));tp=int((selected&(raw==2)).sum());count=int(selected.sum())
                conditions[feature].append({'range':[lo,hi],'reference_pixels':count,'tp':tp,'fn':count-tp,'recall':tp/count if count else None})
        boundary=truth&~cv2.erode(truth.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        record={'sample_id':row['sample_id'],'reference_axis':axis,'network':whole_fiber_diagnosis(target,arrays,settings),'appearance':grouped,
            'reference_boundary_appearance':distributions(fields,boundary,settings['quantiles']),'whole_fiber_condition_groups_not_tip_truth':conditions,
            'historical':historical_diagnosis(row,i,video,frame,target,arrays,axis,t,context,prior,base),'dedicated_tip_metrics':None}
        records.append(record);targets.append(target);predictions['classwise'].append(raw);predictions['old190'].append(arrays['old190_mask'])
        if (i+1)%20==0:progress('static_video29' if video else 'static_image132',samples_completed=i+1,expected_samples=len(rows))
    pooled={}
    for name,features in pool.items():
        arrays={k:np.concatenate(v) for k,v in features.items()};normal=arrays.pop('normal_color_gradient',None)
        selection=np.ones(len(next(iter(arrays.values()))),bool);pooled[name]=distributions(arrays,selection,settings['quantiles'])
        if normal is not None:pooled[name]['normal_color_gradient_valid_axes_only']={'n':int(len(normal)),'mean':float(normal.mean()),'quantiles':np.percentile(normal,settings['quantiles']).astype(float).tolist()}
    metrics={n:summarize(rows,targets,a) for n,a in predictions.items()};groups={}
    if video:ids={n:[i for i,row in enumerate(rows) if row['video_id']==n] for n in ('video_1','video_2')}
    else:
        exposed=set(read(ROOT/'runs/a1_classwise_experts_20261028/summary.json')['image']['historical_exposure_cohorts']['historical_old190_exposed']['sample_ids'])
        ids={n:[i for i,row in enumerate(rows) if (row['sample_id'] in exposed)==seen] for n,seen in [('historical_old190_exposed',True),('not_matched_to_old190_exposure',False)]}
        ids.update({n:[i for i,row in enumerate(rows) if row['source_dataset']==n] for n in sorted({r['source_dataset'] for r in rows})})
    for label,index in ids.items():groups[label]={'images':len(index),'models':{n:summarize([rows[i] for i in index],[targets[i] for i in index],[a[i] for i in index])['metrics'] for n,a in predictions.items()}}
    result={'models':metrics,'groups':groups,'records':records,'appearance_pooled_reference_conditioned':pooled,'raw_historical_different_pixels':equal,'tip_metrics':None,'reference_axis_is_tip_truth':False}
    atomic_json(folder/'summary.json',result);return result


def full_video(vid,out,protocol):
    p=read(out/'preflight.json');item=next(v for v in p['videos'] if v['video_id']==vid);cp=p['checkpoints'];folder=out/'full_video'/vid;folder.mkdir(parents=True)
    (folder/'raw_classwise').mkdir();job={'pid':os.getpid(),'video_id':vid,'model_count':3,'phase':'loading','full_video_all_frames':True};atomic_json(folder/'job.json',job)
    for v in list(cp.values())+[item]:
        if sha256(Path(v['path']))!=v['sha256']:raise ValueError('Worker checkpoint/video identity differs')
    seed_training(42);experts=ClasswiseExperts(cp['old190']['path'],cp['A0']['path']);before=model_state_hash(experts);atomic_json(folder/'runtime.json',runtime_identity(experts.device))
    vm=read(ROOT/'datasets/ureter_semis_kidney_joint_20261028/video_validation_manifest.json');selected={r['frame_index']:r for r in vm['val'] if r['video_id']==vid}
    cap=cv2.VideoCapture(item['path']);ok,first=cap.read()
    if not ok:raise ValueError('Empty video')
    for _ in range(protocol['full_video_baseline']['warmup_frames']):core(experts,first)
    cap.release();cap=cv2.VideoCapture(item['path']);fps=float(cap.get(cv2.CAP_PROP_FPS))
    writer=cv2.VideoWriter(str(folder/'baseline_replay.mp4'),cv2.CAP_MSMF,cv2.VideoWriter_fourcc(*'H264'),fps,(800,434))
    if not writer.isOpened():raise RuntimeError('H264 writer unavailable')
    timings=[];different=0;align=0;i=0;torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    try:
        with (folder/'frames.jsonl').open('w',encoding='utf-8') as log:
            while True:
                start=time.perf_counter();ok,frame=cap.read();decoded=time.perf_counter()
                if not ok:break
                raw,stone,_,probability,pipeline_ms=core(experts,frame);inferred=time.perf_counter()
                panels=[]
                for name,mask in [('Input',None),('Frozen classwise 3 nets',raw)]:
                    panel=frame.copy()
                    if mask is not None:
                        for k,color in [(1,(40,40,235)),(2,(235,210,20))]:
                            a=mask==k;panel[a]=(.45*frame[a]+.55*np.asarray(color)).astype(np.uint8)
                    head=np.zeros((34,400,3),np.uint8);cv2.putText(head,f'{name} | {vid} {i+1}',(4,22),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA);panels.append(np.vstack([head,panel]))
                writer.write(np.hstack(panels));rendered=time.perf_counter()
                cached=load(ROOT/f'runs/a1_classwise_replay_20261028/{vid}/classwise/{i:06d}.png');different+=int((raw!=cached).sum())
                if i in selected:
                    if not np.array_equal(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),np.asarray(Image.open(selected[i]['image']).convert('RGB'))):raise ValueError('Sequential RGB alignment differs')
                    align+=1
                save_png(folder/'raw_classwise'/f'{i:06d}.png',raw)
                record={'frame_index':i,'one_based_frame':i+1,'mask_raw_bytes_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),'rgb_raw_bytes_sha256':hashlib.sha256(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).tobytes()).hexdigest(),
                    'fiber_pixels':int((raw==2).sum()),'stone_pixels':int((raw==1).sum()),'dense_ground_truth_available':False,'tip_metrics':None}
                log.write(json.dumps(record)+'\n');end=time.perf_counter()
                timings.append({'pipeline_ms':pipeline_ms,'decode_ms':(decoded-start)*1000,'render_encode_ms':(rendered-inferred)*1000,'audit_and_mask_export_ms':(end-rendered)*1000,'cycle_ms':(end-start)*1000})
                i+=1
                if i%100==0:job.update(phase='full_video_replay',frames_completed=i,expected_frames=item['frames']);atomic_json(folder/'job.json',job);print(f'{vid} {i}/{item["frames"]}',flush=True)
    finally:cap.release();writer.release()
    wall=time.perf_counter()-started
    if i!=item['frames'] or different or align!=len(selected):raise ValueError('Incomplete or nonidentical baseline replay')
    assert before==model_state_hash(experts)
    check=cv2.VideoCapture(str(folder/'baseline_replay.mp4'));decoded_frames=0
    while check.read()[0]:decoded_frames+=1
    check.release()
    if decoded_frames!=i:raise ValueError('Video export incomplete')
    result={'video_id':vid,'frames':i,'decoded_export_frames':decoded_frames,'fps':fps,'warmup_frames':12,'model_count':3,'parameter_count':experts.parameter_count,'raw_historical_different_pixels':different,'reviewed_rgb_exact':align,
        'timings':{k:stats([t[k] for t in timings]) for k in timings[0]},'offline_fps_including_decode_audit_render_exports':i/wall,'wall_seconds_including_writer_flush':wall,
        'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'model_state_sha256_unchanged':before,'video_sha256':sha256(folder/'baseline_replay.mp4'),
        'scope':'All frames, fresh process per video. Pipeline preprocessing+required3 networks+FP32 native outputs+CPU transfer. Full cycle additionally decode/render/audit/export; no live display/queue acceptance','dense_accuracy_metrics':None,'tip_metrics':None}
    atomic_json(folder/'timings.json',timings);atomic_json(folder/'summary.json',result);job.update(phase='completed',frames_completed=i,expected_frames=i);atomic_json(folder/'job.json',job)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--full-video');args=parser.parse_args();protocol=read(CONFIG);out=ROOT/protocol['output']
    if args.full_video:return full_video(args.full_video,out,protocol)
    if out.exists():raise FileExistsError(out)
    out.mkdir();job={'pid':os.getpid(),'gradient_updates':0,'default_model_changed':False,'new_segmentation_scheme':False,'tip_reference_available':False}
    def progress(phase,**extra):job.update(phase=phase,**extra);atomic_json(out/'job.json',job)
    progress('preflight')
    try:
        mpath=ROOT/protocol['manifest'];m=read(mpath);vpath=mpath.parent/'video_validation_manifest.json';vm=read(vpath);identity=input_identity(mpath,Path(m['root']),{'train':{'labeled':[],'unlabeled':[]},'val':m['val']+vm['val']})
        cp={k:v for k,v in read(ROOT/protocol['experts'])['checkpoints'].items() if k in ('old190','A0')};videos=read(ROOT/'runs/baselines/a0_20261027/config.json')['inputs']
        for v in list(cp.values())+videos:
            if sha256(Path(v['path']))!=v['sha256']:raise ValueError('Checkpoint/video identity differs')
        extra=['tools/fiber_tip_diagnostics.py','tools/diagnose_fiber_tip.py','tests/test_fiber_tip_diagnostics.py','tools/classwise_experts.py','tools/run_classwise_expert_control.py','tools/run_fiber_shape_control.py','tools/fiber_shape_prior.py','tools/replay_classwise_experts.py','tools/evaluate_native_checkpoints.py','tools/evaluate_spatial_detail.py','tools/diagnose_joint_failures.py','tools/direct_fiber_template.py']
        sources=source_identity();sources.update({str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'tools').glob('*.py')});sources.update({n:sha256(ROOT/n) for n in extra})
        related=[CONFIG,vpath,ROOT/protocol['experts'],ROOT/'configs/direct_fiber_template_20261029.json',ROOT/'runs/a1_direct_fiber_template_20261029/template.json',ROOT/'runs/a1_direct_fiber_template_20261029/summary.json',ROOT/'runs/a1_direct_neural_observation_20261029/summary.json',ROOT/'runs/a1_classwise_experts_20261028/summary.json']
        for video,rows in [(False,m['val']),(True,vm['val'])]:
            for i,row in enumerate(rows):related.extend(historical_paths(row,i,video).values())
        for item in videos:related.extend(sorted((ROOT/f'runs/a1_classwise_replay_20261028/{item["video_id"]}/classwise').glob('*.png')))
        related.extend(ROOT/f'runs/a1_direct_fiber_template_20261029/{v}/frames.jsonl' for v in ('video_1','video_2'))
        preflight={'protocol':protocol,'inputs':identity,'checkpoints':cp,'videos':videos,'sources':sources,'related':{str(p):sha256(p) for p in related},'train_pixels_read':0,'reserved_test_pixels_read':False};atomic_json(out/'preflight.json',preflight)
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for n in sources:z.write(ROOT/n,n.replace('\\','/'))
            z.write(CONFIG,str(CONFIG.relative_to(ROOT)))
        seed_training(42);experts=ClasswiseExperts(cp['old190']['path'],cp['A0']['path']);before=model_state_hash(experts);atomic_json(out/'runtime.json',runtime_identity(experts.device))
        context=history_context();prior=read(ROOT/'runs/a1_direct_fiber_template_20261029/template.json');base=read(ROOT/'configs/direct_fiber_template_20261029.json')
        progress('static_image132');image=static_evaluate(m['val'],False,experts,out,protocol['diagnostic_only'],context,prior,base,progress)
        progress('static_video29');video=static_evaluate(vm['val'],True,experts,out,protocol['diagnostic_only'],context,prior,base,progress)
        assert before==model_state_hash(experts);del experts;torch.cuda.empty_cache()
        pilot=[]
        for row in vm['val']:
            if row['frame_index'] in protocol['pilot_reference_frames_zero_based'][row['video_id']]:
                pilot.append({**row,'tip_review_status':'pending','entry_point':None,'tip_visibility':'unreviewed','tip_point':None,'tip_roi':None,'tip_mask':None,'shaft_mask':None,'unknown_mask':None,'reviewer':None,'policy':'Visible pixels only; original whole-fiber label is context, not a generated tip label. No predictions prefill the annotations'})
        atomic_json(out/'tip_reference_pilot.template.json',{'status':'PENDING_NOT_GROUND_TRUTH','records':pilot,'tip_metrics':None,'original_masks_modified':False})
        full={}
        for item in videos:
            vid=item['video_id'];progress('full_video_fresh_process',video_id=vid)
            with (out/f'full_video_{vid}.log').open('w',encoding='utf-8') as log:subprocess.run([sys.executable,'-m','tools.diagnose_fiber_tip','--full-video',vid],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            full[vid]=read(out/'full_video'/vid/'summary.json')
        assert all(sha256(ROOT/n)==h for n,h in sources.items()) and all(sha256(Path(n))==h for n,h in preflight['related'].items())
        assert input_identity(mpath,Path(m['root']),{'train':{'labeled':[],'unlabeled':[]},'val':m['val']+vm['val']})==identity
        assert all(sha256(Path(v['path']))==v['sha256'] for v in list(cp.values())+videos)
        result={'image':image,'video29':video,'full_video_baseline':full,'pilot_reference_samples':len(pilot),'tip_metrics':None,'new_segmentation_scheme':False,'train_pixels_read':0,'reserved_test_pixels_read':False,'gradient_updates':0,'shared_statistics_fitted':False,'default_model_changed':False,'s2_accepted':False,'stone_memory_started':False,'static_model_state_sha256_unchanged':before}
        atomic_json(out/'summary.json',result);progress('completed',video_id=None,static_samples=161,full_video_frames=2694)
        print(json.dumps({'phase':'completed','static':{k:v['models']['classwise']['metrics'] for k,v in [('image',image),('video29',video)]},'tip_metrics':None,'full_video_frames':2694},indent=2),flush=True)
    except Exception as error:
        progress('failed',error=str(error));(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':main()
