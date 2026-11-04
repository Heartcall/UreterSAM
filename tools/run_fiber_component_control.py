"""Four frozen component-gate controls; required-model cost and conditional replay."""
from __future__ import annotations

from collections import Counter
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

from ureter_semis.experiment import atomic_json,input_identity,seed_training,sha256,source_identity,runtime_identity
from ureter_semis.supervised_trainer import assess
from ureter_semis.video_metrics import estimated_fov
from .fiber_component_gate import ComponentGatePredictor,ARMS
from .run_classwise_expert_control import summarize
from .diagnose_joint_failures import paired_changes
from .replay_classwise_experts import intervals,mask_diagnostics,summarize_diagnostics,stats,save_png,model_state_hash

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/a1_fiber_component_gate_20261029'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def load(path):
    with Image.open(path) as im:return np.asarray(im).copy()


def evaluate(rows,scope,predictor,folder):
    predictions=[];targets=[];records=[]
    for i,row in enumerate(rows):
        frame=cv2.cvtColor(np.asarray(Image.open(row['image']).convert('RGB')),cv2.COLOR_RGB2BGR)
        result=predictor.predict(frame)
        save_png(folder/f'{i:04d}.png',result['mask'])
        predictions.append(result['mask']);targets.append(load(row['mask']))
        records.append({'sample_id':row['sample_id'],'gate':result['gate']})
    result=summarize(rows,targets,predictions);result['records']=records
    result['accepted']=sum(r['gate']['accepted'] is True for r in records)
    result['reasons']=dict(Counter(r['gate']['reason'] for r in records))
    return result


def compare_scope(rows,scope,models,exposed):
    folder=OUT/f'{scope}_native';targets=[load(r['mask']) for r in rows]
    predictions={n:[load(folder/n/f'{i:04d}.png') for i in range(len(rows))] for n in ARMS}
    result={'models':models,'pairs':{},'cohorts':{},'by_source_or_video':{}}
    for gated,raw in [(ARMS[1],ARMS[0]),(ARMS[3],ARMS[2])]:
        paired={str(k):Counter() for k in (1,2)}
        changes=[]
        for row,target,a,b in zip(rows,targets,predictions[raw],predictions[gated]):
            d={str(k):paired_changes(target,a,b,k) for k in (1,2)}
            changes.append({'sample_id':row['sample_id'],'class_changes':d})
            for k in (1,2):paired[str(k)].update(d[str(k)])
        accepted=[i for i,r in enumerate(models[gated]['records']) if r['gate']['accepted']]
        result['pairs'][gated]={'raw_reference':raw,'class_changes':{k:dict(v) for k,v in paired.items()},'changes_per_frame':changes,
            'accepted_subset':{'frames':len(accepted),'sample_ids':[rows[i]['sample_id'] for i in accepted],
                'models':{n:summarize([rows[i] for i in accepted],[targets[i] for i in accepted],[predictions[n][i] for i in accepted])['metrics'] for n in (raw,gated)}}}
    if scope=='image':
        for label,seen in [('historical_old190_exposed',True),('not_matched_to_old190_exposure',False)]:
            ids=[i for i,r in enumerate(rows) if (r['group_id'] in exposed)==seen]
            result['cohorts'][label]={'images':len(ids),'sample_ids':[rows[i]['sample_id'] for i in ids],
                'models':{n:summarize([rows[i] for i in ids],[targets[i] for i in ids],[predictions[n][i] for i in ids])['metrics'] for n in ARMS}}
    for key in sorted({r.get('video_id') or r.get('source_dataset') for r in rows}):
        ids=[i for i,r in enumerate(rows) if (r.get('video_id') or r.get('source_dataset'))==key]
        result['by_source_or_video'][key]={n:summarize([rows[i] for i in ids],[targets[i] for i in ids],[predictions[n][i] for i in ids])['metrics'] for n in ARMS}
    atomic_json(folder/'comparison.json',result)
    return result


def render(frame,raw,gated,title):
    panels=[]
    for name,mask in [('Input',None),('Raw',raw),('Component gate',gated)]:
        panel=frame.copy()
        if mask is not None:
            for k,color in [(1,(40,40,235)),(2,(235,210,20))]:
                selected=mask==k;panel[selected]=(.48*frame[selected]+.52*np.asarray(color)).astype(np.uint8)
        header=np.zeros((34,frame.shape[1],3),np.uint8)
        cv2.putText(header,f'{name} | {title}',(5,21),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
        panels.append(np.vstack([header,panel]))
    return np.hstack(panels)


def replay(arm,item,checkpoints,fit,reviewed,replay_protocol,progress):
    folder=OUT/'full_replay'/arm/item['video_id'];folder.mkdir(parents=True)
    for n in ('raw','gated','clips','keyframes'):(folder/n).mkdir()
    predictor=ComponentGatePredictor(arm,checkpoints,fit);vid=item['video_id'];cap=cv2.VideoCapture(item['path']);writers=[]
    if not cap.isOpened():raise RuntimeError('Cannot open video')
    fps=float(cap.get(cv2.CAP_PROP_FPS));count=item['frames'];events=intervals(replay_protocol,vid,count)
    selected={r['frame_index']:r for r in reviewed if r['video_id']==vid};samples={};rows=[];timings=[];alignment=[]
    def writer(path):
        w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(item['width']*3,item['height']+34))
        if not w.isOpened():raise RuntimeError('Cannot create replay preview')
        writers.append(w);return w
    try:
        ok,first=cap.read()
        if not ok:raise ValueError('Empty video')
        for _ in range(12):predictor.predict(first)
        a,b=predictor.predict(first),predictor.predict(first)
        if not np.array_equal(a['mask'],b['mask']) or a['gate']!=b['gate']:raise ValueError('Repeat check failed')
        cap.release();cap=cv2.VideoCapture(item['path']);preview=writer(folder/'comparison.mp4');clipwriters=[]
        for i,e in enumerate(events):
            e['clip']=f'clips/{i:02d}_{e["event"]}_{e["one_based_inclusive"][0]}-{e["one_based_inclusive"][1]}.mp4';clipwriters.append(writer(folder/e['clip']))
        previous={'raw':None,'gated':None};different=0;started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        with (folder/'frames.jsonl').open('w',encoding='utf-8') as log:
            while True:
                t0=time.perf_counter();ok,frame=cap.read();decoded=time.perf_counter()
                if not ok:break
                i=len(rows)
                if i>=count or frame.shape[:2]!=(item['height'],item['width']):raise ValueError('Unexpected video geometry/count')
                result=predictor.predict(frame);raw=result['raw_mask'];gated=result['mask'];fov=estimated_fov(frame,raw.shape)
                historical='classwise' if predictor.classwise else 'old190'
                cached=load(ROOT/f'runs/a1_classwise_replay_20261028/{vid}/{historical}/{i:06d}.png');different+=int((cached!=raw).sum())
                if i in selected:
                    r=selected[i];exact=np.array_equal(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),np.asarray(Image.open(r['image']).convert('RGB')))
                    alignment.append({'sample_id':r['sample_id'],'pixel_exact':exact,'frame_index':i})
                    if not exact:raise ValueError('Reviewed RGB differs')
                    samples[r['sample_id']]=gated.copy()
                d={n:mask_diagnostics(mask,fov,previous[n]) for n,mask in [('raw',raw),('gated',gated)]}
                save_png(folder/'raw'/f'{i:06d}.png',raw);save_png(folder/'gated'/f'{i:06d}.png',gated)
                canvas=render(frame,raw,gated,f'{vid} frame{i+1}');preview.write(canvas)
                for e,w in zip(events,clipwriters):
                    if e['start_frame']<=i<e['end_frame_exclusive']:w.write(canvas)
                if i in selected:save_png(folder/'keyframes'/f'{i:06d}.png',canvas)
                record={'frame_index':i,'one_based_frame':i+1,'gate':result['gate'],'models':d,'dense_ground_truth_available':False}
                log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
                timings.append({**result['timing'],'decode_ms':(decoded-t0)*1000,'actual_cycle_including_exports_ms':(time.perf_counter()-t0)*1000})
                rows.append(record);previous={'raw':raw,'gated':gated}
                if (i+1)%200==0:
                    progress('full_replay',arm=arm,video_id=vid,frames_completed=i+1,expected_frames=count);print(f'{arm} {vid}: {i+1}/{count}',flush=True)
        for w in writers:w.release()
        writers.clear();elapsed=time.perf_counter()-started
        if len(rows)!=count or len(alignment)!=len(selected):raise ValueError('Final counts differ')
        result={'arm':arm,'video_id':vid,'frames':count,'fps':fps,'model_count':predictor.model_count,'parameter_count':predictor.parameter_count,
            'repeat_exact':True,'reviewed_alignment':alignment,'raw_historical_different_pixels':different,
            'accepted':sum(r['gate']['accepted'] for r in rows),'reasons':dict(Counter(r['gate']['reason'] for r in rows)),
            'removed_pixels':sum(r['gate']['removed_pixels'] for r in rows),'frames_with_removal':sum(r['gate']['removed_pixels']>0 for r in rows),
            'models':{n:summarize_diagnostics([{'frame_index':r['frame_index'],**r['models'][n]} for r in rows],fps) for n in ('raw','gated')},
            'timings':{k:stats([r[k] for r in timings]) for k in timings[0]},'offline_fps_including_exports':count/elapsed,
            'wall_seconds_including_writer_flush':elapsed,'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'user_intervals':events,'dense_accuracy_metrics':None,'scope':'Stateless validation replay, not live display, identity, tip or occlusion-ground-truth validation'}
        for e in events:
            part=rows[e['start_frame']:e['end_frame_exclusive']]
            e['models']={n:summarize_diagnostics([{'frame_index':r['frame_index'],**r['models'][n]} for r in part],fps) for n in ('raw','gated')}
        atomic_json(folder/'timings.json',timings);atomic_json(folder/'summary.json',result)
        return result,samples
    finally:
        cap.release()
        for w in writers:w.release()


def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir();job={'pid':os.getpid(),'training_updates':0,'default_model_changed':False,'s2_accepted':False,'stone_memory_started':False}
    def progress(phase,**extra):job.update(phase=phase,**extra);atomic_json(OUT/'job.json',job)
    progress('preflight')
    try:
        execution_path=ROOT/'configs/run_fiber_component_gate_20261029.json';execution=read(execution_path)
        protocol_path=ROOT/execution['protocol'];protocol=read(protocol_path);geometry_path=ROOT/execution['geometry_config'];geometry=read(geometry_path)
        expert_path=ROOT/execution['expert_manifest'];checkpoints={n:v for n,v in read(expert_path)['checkpoints'].items() if n in ('old190','A0')}
        previous=read(ROOT/'runs/a1_fiber_geometry_20261028/preflight.json')
        if sha256(ROOT/'tools/fiber_shape_prior.py')!=previous['sources']['tools/fiber_shape_prior.py'] or geometry['fit']!=previous['protocol']['fit']:raise ValueError('Previous fit source/parameters changed')
        for n,v in checkpoints.items():
            if sha256(Path(v['path']))!=v['sha256']:raise ValueError('Checkpoint identity differs')
        manifest_path=ROOT/'datasets/ureter_semis_kidney_joint_20261028/manifest.json';m=read(manifest_path);vm_path=manifest_path.parent/'video_validation_manifest.json';vm=read(vm_path)
        identity=input_identity(manifest_path,Path(m['root']),{'train':{'labeled':[],'unlabeled':[]},'val':m['val']+vm['val']})
        baseline=read(ROOT/'runs/a1_classwise_experts_20261028/summary.json');exposed_ids=set(baseline['image']['historical_exposure_cohorts']['historical_old190_exposed']['sample_ids'])
        exposed={r['group_id'] for r in m['val'] if r['sample_id'] in exposed_ids}
        extra=['tools/run_fiber_component_control.py','tools/fiber_component_gate.py','tools/benchmark_fiber_component_gate.py','tests/test_fiber_component_gate.py','tools/fiber_shape_prior.py','tools/classwise_experts.py','tools/replay_classwise_experts.py','tools/run_classwise_expert_control.py','tools/evaluate_native_checkpoints.py','tools/evaluate_spatial_detail.py','tools/diagnose_joint_failures.py']
        sources=source_identity();sources.update({n:sha256(ROOT/n) for n in extra})
        related=[execution_path,protocol_path,geometry_path,expert_path,vm_path,ROOT/'runs/a1_classwise_experts_20261028/summary.json',ROOT/'runs/baselines/a0_20261027/config.json',ROOT/'configs/replay_classwise_experts_20261028.json']
        video_inputs=read(ROOT/'runs/baselines/a0_20261027/config.json')['inputs']
        for v in video_inputs:
            if sha256(Path(v['path']))!=v['sha256']:raise ValueError('Video identity differs')
            for n in ('classwise','old190'):related+=sorted((ROOT/f'runs/a1_classwise_replay_20261028/{v["video_id"]}/{n}').glob('*.png'))
        preflight={'execution':execution,'protocol':protocol,'geometry_config':geometry,'inputs':identity,'checkpoints':checkpoints,'video_inputs':video_inputs,
            'sources':sources,'related':{str(p):sha256(p) for p in related},'training_updates':0,'reserved_test_pixels_read':False}
        atomic_json(OUT/'preflight.json',preflight)
        with zipfile.ZipFile(OUT/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in sources:z.write(ROOT/name,name.replace('\\','/'))
            for path in (execution_path,protocol_path,geometry_path):z.write(path,str(path.relative_to(ROOT)))
        for scope in ('image','video'):
            (OUT/f'{scope}_native').mkdir()
            for arm in ARMS:(OUT/f'{scope}_native'/arm).mkdir()
        accuracy={'image':{},'video':{}};state_hashes={}
        for arm in ARMS:
            progress('accuracy',arm=arm);seed_training(42);predictor=ComponentGatePredictor(arm,checkpoints,geometry['fit'])
            expected=(3,73804185) if arm.startswith('classwise') else (1,24601395)
            if (predictor.model_count,predictor.parameter_count)!=expected:raise ValueError('Actual model budget differs')
            atomic_json(OUT/f'runtime_{arm}.json',runtime_identity(predictor.device));before=model_state_hash(predictor)
            for scope,rows in [('video',vm['val']),('image',m['val'])]:
                accuracy[scope][arm]=evaluate(rows,scope,predictor,OUT/f'{scope}_native'/arm)
            assert model_state_hash(predictor)==before and all(not model.training for model in predictor.models)
            state_hashes[arm]=before;del predictor;torch.cuda.empty_cache()
        video=compare_scope(vm['val'],'video',accuracy['video'],exposed);image=compare_scope(m['val'],'image',accuracy['image'],exposed)
        screening={n:assess(v['metrics'],protocol['gates']) for n,v in video['models'].items()}
        atomic_json(OUT/'accuracy_summary.json',{'video':video,'image':image,'screening':screening})
        costs={}
        for arm in ARMS:
            progress('isolated_cost',arm=arm)
            with (OUT/f'benchmark_{arm}.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-m','tools.benchmark_fiber_component_gate','--arm',arm,'--output',str(OUT/f'performance_{arm}.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            costs[arm]=read(OUT/f'performance_{arm}.json')
        passing=[n for n in (ARMS[1],ARMS[3]) if screening[n]['passed']];full={}
        for arm in passing:
            selected={};videos={}
            for item in video_inputs:
                progress('loading_full_replay',arm=arm,video_id=item['video_id'],frames_completed=0,expected_frames=item['frames'])
                result,samples=replay(arm,item,checkpoints,geometry['fit'],vm['val'],read(ROOT/'configs/replay_classwise_experts_20261028.json'),progress)
                videos[item['video_id']]=result;selected.update(samples);torch.cuda.empty_cache()
            targets=[load(r['mask']) for r in vm['val']];preds=[selected[r['sample_id']] for r in vm['val']]
            full[arm]={'videos':videos,'reviewed29':summarize(vm['val'],targets,preds),
                'different_pixels_from_primary29':sum(int((preds[i]!=load(OUT/f'video_native/{arm}/{i:04d}.png')).sum()) for i in range(len(vm['val'])))}
        assert input_identity(manifest_path,Path(m['root']),{'train':{'labeled':[],'unlabeled':[]},'val':m['val']+vm['val']})==identity
        assert all(sha256(ROOT/n)==h for n,h in sources.items()) and all(sha256(Path(n))==h for n,h in preflight['related'].items())
        assert all(sha256(Path(v['path']))==v['sha256'] for v in checkpoints.values()) and all(sha256(Path(v['path']))==v['sha256'] for v in video_inputs)
        result={'video':video,'image':image,'screening':screening,'performance':costs,'passing_gated_arms':passing,'full_replay':full,
            'state_hashes_before_after_unchanged':state_hashes,'training_updates':0,'default_model_changed':False,'s2_accepted':False,'reserved_test_pixels_read':False,'stone_memory_started':False}
        atomic_json(OUT/'summary.json',result);progress('completed',passing_gated_arms=passing)
        print(json.dumps({'phase':'completed','video_metrics':{n:v['metrics'] for n,v in video['models'].items()},'screening':screening,
            'performance':{n:v['timings']['pipeline_ms'] for n,v in costs.items()},'full_replayed':passing},indent=2),flush=True)
    except Exception as error:
        progress('failed',error=str(error));(OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
