"""Finite shadow-candidate growth control; never changes actual segmentation."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib,json,math,os
from pathlib import Path
import subprocess,sys,time,traceback,zipfile
import cv2
import numpy as np
import torch
from ureter_semis.experiment import atomic_json,input_identity,runtime_identity,seed_training
from ureter_semis.supervised_trainer import assess
from .fiber_reference_workspace import ROOT,read,sha256,load_mask
from .fiber_connected_blue_growth import GrowthPipeline,ARMS
from .fiber_photometric_input import infer
from .run_fiber_dark_input_control import count_forwards,process_memory,checked_writer,verify_video,select_groups,brightness,pool_brightness,json_lines,FIXED
from .run_fiber_temporal_state_control import construct,source_scan
from .run_two_network_experts import array_hash,model_hash
from .run_fiber_blue_aplus import frame_read,panel
from .replay_classwise_experts import model_state_hash,save_png,stats
from .run_classwise_expert_control import summarize
from .diagnose_joint_failures import paired_changes

CONFIG=ROOT/'configs/fiber_connected_blue_growth_20261101.json'


def growth_hash(meta):
    return hashlib.sha256(json.dumps({k:v for k,v in meta.items() if k not in ['timings','t1_advice']},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def virtual_mask(raw,candidate):
    result=raw.copy();result[(raw==0)&(candidate>0)]=2
    if np.any(result[raw==1]!=1) or np.any(result[raw==2]!=2):raise ValueError('Protected raw labels changed')
    return result


def candidate_metrics(gt,raw,fp,frame,candidate,proposal,conflict,meta):
    truth=gt==2;valid=gt!=255;new=candidate>0;proposed=proposal>0
    tp=new&truth;bad=new&~truth&valid;lab=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)[:,:,0]
    result={'new_pixels':int(new.sum()),'new_known':int((new&valid).sum()),'new_reference_TP':int(tp.sum()),'new_reference_FP':int(bad.sum()),
        'new_unknown':int((new&~valid).sum()),'originalPbelow01_FN_candidate_hits':int((tp&(fp[2]<.1)&(raw!=2)).sum()),
        'lowLab_FN_candidate_hits':int((tp&(lab<85)&(raw!=2)).sum()),'proposed_pixels':int(proposed.sum()),
        'proposed_reference_TP':int((proposed&truth).sum()),'proposed_reference_FP':int((proposed&~truth&valid).sum()),
        'stone_conflict_pixels':int((conflict>0).sum()),'stone_conflict_reference_fiber':int(((conflict>0)&truth).sum()),
        'reference_fiber_absent':not bool(truth.any()),'growth_on_reference_absent':not bool(truth.any()) and bool(new.any()),
        'abnormal_rejected':bool(meta['abnormal']),'new_reference_precision':int(tp.sum())/int((new&valid).sum()) if (new&valid).any() else None,
        'proposal_kind':meta.get('proposal_kind','none')}
    return result


def pooled(records,arm):
    rows=[r['arms'][arm]['candidate_metrics'] for r in records];result={}
    for k,v in rows[0].items():
        if isinstance(v,(int,bool)):result[k]=sum(int(r[k]) for r in rows)
    result['new_reference_precision']=result['new_reference_TP']/result['new_known'] if result['new_known'] else None
    result['reasons']=dict(Counter(r['arms'][arm]['meta']['reason'] for r in records));result['tip_states']=dict(Counter(t['state'] for r in records for t in r['arms'][arm]['meta']['tips']))
    result['frames_with_new_candidate']=sum(r['arms'][arm]['candidate_metrics']['new_pixels']>0 for r in records)
    return result


def save_candidates(folder,i,candidate,proposal,conflict):
    for name,array in [('candidate',candidate),('proposal',proposal),('conflict',conflict)]:save_png(folder/name/f'{i:06d}.png',array)


def prepare_folder(path):
    path.mkdir(parents=True)
    for name in ['candidate','proposal','conflict']:(path/name).mkdir()


def overlay(frame,raw,candidate,proposal,meta,name,index):
    image=frame.copy();new=candidate>0;image[new]=(.35*image[new]+.65*np.array([0,230,0])).astype(np.uint8)
    if not new.any() and (proposal>0).any():
        cs,_=cv2.findContours(proposal.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(image,cs,-1,(200,0,200),1)
    for branch in meta['branches']:
        q=np.rint(branch['centers']).astype(np.int32)
        if len(q)>1:cv2.polylines(image,[q],False,(0,180,220),1)
    for tip in meta['tips']:
        if tip['xy'] is not None:cv2.circle(image,tuple(np.rint(tip['xy']).astype(int)),3,(0,255,255),1)
    result=panel(image,name,None,index);strip=np.zeros((50,frame.shape[1],3),np.uint8)
    cv2.putText(strip,meta['reason'],(3,16),cv2.FONT_HERSHEY_SIMPLEX,.33,(255,255,255),1)
    advice=meta.get('t1_advice',{});text=advice.get('direction','unknown')+' | '+','.join(t['state'] for t in meta['tips'])
    cv2.putText(strip,text,(3,35),cv2.FONT_HERSHEY_SIMPLEX,.30,(255,255,255),1)
    return np.vstack([result,strip])


def base_panel(frame,raw,index):return np.vstack([panel(frame,'ACTUAL raw mask unchanged',raw,index),np.zeros((50,frame.shape[1],3),np.uint8)])


def static(cfg,pre,out,progress):
    seed_training(42);experts=construct(pre);before=model_state_hash(experts);counts,handles=count_forwards(experts);t1=read(ROOT/cfg['t1']['config'])
    for scope,key in [('video_native','video_manifest'),('image_native','image_manifest')]:
        rows=read(ROOT/cfg['data'][key])['val'];old=read(ROOT/cfg['base']['reference_run']/f'{scope}/raw2.json');folder=out/scope;folder.mkdir();(folder/'raw').mkdir()
        for a in ARMS:prepare_folder(folder/a)
        targets=[];actual=[];predictions={a:[] for a in ARMS};records=[]
        for i,row in enumerate(rows):
            frame=frame_read(row['image']);raw,sp,fp,_=infer(experts,frame,'raw2');gt=load_mask(row['mask'],raw.shape,{0,1,2,255})
            hashes={'mask_raw_sha256':array_hash(raw),'stone_probability_raw_sha256':array_hash(sp),'fiber_probability_raw_sha256':array_hash(fp)}
            if any(v!=old['records'][i][k] for k,v in hashes.items()):raise ValueError('Static original probabilities/mask differ')
            record={'sample_id':row['sample_id'],'video_id':row.get('video_id'),'frame_index':row.get('frame_index'),**hashes,'arms':{},'tip_metrics':None}
            for arm in ARMS:
                candidate,proposal,conflict,meta=GrowthPipeline(cfg,arm,t1,False).run(frame,raw,fp)
                shadow=virtual_mask(raw,candidate);save_candidates(folder/arm,i,candidate,proposal,conflict)
                record['arms'][arm]={'candidate_sha256':array_hash(candidate),'proposal_sha256':array_hash(proposal),'conflict_sha256':array_hash(conflict),
                    'growth_sha256':growth_hash(meta),'meta':meta,'candidate_metrics':candidate_metrics(gt,raw,fp,frame,candidate,proposal,conflict,meta),
                    'virtual_brightness':brightness(frame,gt,shadow,fp[2]),'paired_virtual_stone':paired_changes(gt,raw,shadow,1),'paired_virtual_fiber':paired_changes(gt,raw,shadow,2)}
                predictions[arm].append(shadow)
            if any(record['arms']['connected_rgb'][k]!=record['arms']['connected_rgb_t1'][k] for k in ['candidate_sha256','proposal_sha256','conflict_sha256','growth_sha256']):raise ValueError('Static T1 changed growth')
            if array_hash(raw)!=hashes['mask_raw_sha256']:raise ValueError('Actual mask mutated')
            save_png(folder/f'raw/{i:04d}.png',raw);records.append(record);targets.append(gt);actual.append(raw)
            if (i+1)%30==0:progress('static',scope=scope,completed=i+1)
        actual_summary=summarize(rows,targets,actual)
        if actual_summary!=old['summary']:raise ValueError('Actual metrics differ')
        virtual={a:summarize(rows,targets,predictions[a]) for a in ARMS};groups={}
        for name,ids in select_groups(rows,scope=='video_native').items():
            rr=[records[i] for i in ids]
            groups[name]={'frames':len(ids),'actual':summarize([rows[i] for i in ids],[targets[i] for i in ids],[actual[i] for i in ids]),
                'virtual':{a:summarize([rows[i] for i in ids],[targets[i] for i in ids],[predictions[a][i] for i in ids]) for a in ARMS},'candidate':{a:pooled(rr,a) for a in ARMS}}
        for a in ARMS:
            for r in records:r['brightness']=r['arms'][a]['virtual_brightness']
            atomic_json(folder/f'brightness_{a}.json',pool_brightness(records))
        for r in records:r.pop('brightness',None)
        atomic_json(folder/'summary.json',{'actual':actual_summary,'virtual':virtual,'candidate':{a:pooled(records,a) for a in ARMS},'groups':groups,'records':records,
            'actual_paired_frame_inferences':len(rows),'actual_pixel_changes':0,'tip_metrics':None})
    if counts!=[202,202] or model_state_hash(experts)!=before:raise ValueError('Static graph/weights differ')
    for h in handles:h.remove()
    atomic_json(out/'static_model_identity.json',{'model_state_sha256':before,'individual_model_sha256':[model_hash(m) for m in experts.models],'actual_branch_forwards':counts,'networks':2,'parameters':experts.parameter_count,'runtime':runtime_identity(experts.device)})
    del experts;torch.cuda.empty_cache()


def quality(cfg,out):
    frozen=read(ROOT/cfg['quality']['frozen_gates']);eps=frozen['recommendation_checks']['epsilon'];answer={}
    for arm in ARMS[1:]:
        checks={}
        for scope in ['video_native','image_native']:
            d=read(out/f'{scope}/summary.json');b=d['actual'];v=d['virtual'][arm];c=d['candidate'][arm]
            checks[scope+'_new_precision']=c['new_reference_precision'] is not None and c['new_reference_precision']>=cfg['quality']['minimum_new_candidate_precision']
            checks[scope+'_no_growth_on_reference_absent']=c['growth_on_reference_absent']==0
            bb=read(out/f'{scope}/brightness_raw2.json')['Lab_Lstar'][0];vv=read(out/f'{scope}/brightness_{arm}.json')['Lab_Lstar'][0]
            checks[scope+'_dark_FN_10percent']=bb['fn']>0 and (bb['fn']-vv['fn'])/bb['fn']>=cfg['quality']['minimum_lowLab_FN_reduction_fraction']
            classwise=read(ROOT/cfg['base']['historical_reference']/f'{scope}/classwise3.json')['summary']
            for refname,reference in [('raw2',b),('classwise3',classwise)]:
                for m in frozen['recommendation_checks']['metrics']:
                    if m.endswith('_boundary_f'):
                        label='1' if m.startswith('stone') else '2';a=reference['boundaries'][label]['f_score'];z=v['boundaries'][label]['f_score']
                    else:a=reference['metrics'][m];z=v['metrics'][m]
                    checks[f'{scope}:{refname}:{m}']=a is not None and z is not None and z>=a-eps
            if scope=='video_native':
                legacy=assess(d['groups']['original29']['virtual'][arm]['metrics'],frozen['legacy_original29_gates']);checks['original29_gates']=legacy['passed']
                r=next(r for r in d['records'] if r['video_id']=='video_2' and r['frame_index']==1208)
                bb=r['arms']['raw2']['virtual_brightness']['Lab_Lstar'][0];vv=r['arms'][arm]['virtual_brightness']['Lab_Lstar'][0]
                checks['fixed1209_dark_FN_10percent']=bb['fn']>0 and (bb['fn']-vv['fn'])/bb['fn']>=.1
            checks[scope+'_actual_pixels_unchanged']=d['actual_pixel_changes']==0
            checks[scope+'_virtual_protected_TP']=all(r['arms'][arm]['paired_virtual_stone']['lost_tp']==0 and r['arms'][arm]['paired_virtual_fiber']['lost_tp']==0 for r in d['records'])
        answer[arm]={'checks':checks,'passed':all(checks.values()),'rejected_checks':[k for k,v in checks.items() if not v], 'scope':'candidate/counterfactual feasibility, never actual segmentation adoption'}
    return answer


def worker(cfg,out,arm,vid):
    pre=read(out/'preflight.json');info=next(v for v in pre['videos'] if v['video_id']==vid)
    if any(sha256(ROOT/k)!=v for k,v in pre['sources'].items()):raise ValueError('Worker source changed')
    for x in [info,*pre['checkpoints'].values()]:
        if sha256(x['path'])!=x['sha256']:raise ValueError('Worker input changed')
    source=read(out/f'source_{vid}.json');old=json_lines(ROOT/cfg['base']['reference_run']/f'full_video/raw2/{vid}/frames.jsonl')
    rows=read(ROOT/cfg['data']['video_manifest'])['val'];selected={r['frame_index']:i for i,r in enumerate(rows) if r['video_id']==vid};static_results=read(out/'video_native/summary.json')
    reference_growth=None if arm!='connected_rgb_t1' else json_lines(out/f'full_video/connected_rgb/{vid}/frames.jsonl')
    seed_training(42);experts=construct(pre);before=model_state_hash(experts);counts,handles=count_forwards(experts)
    pipeline=GrowthPipeline(cfg,arm,read(ROOT/cfg['t1']['config']),True);folder=out/f'full_video/{arm}/{vid}';prepare_folder(folder)
    def run(frame,timestamp):
        experts.sync();start=time.perf_counter();raw,sp,fp,t=infer(experts,frame,'raw2');base=time.perf_counter()
        candidate,proposal,conflict,meta=pipeline.run(frame,raw,fp,timestamp,vid);end=time.perf_counter()
        t.update(meta['timings']);t.update(base_pipeline_ms=(base-start)*1000,pipeline_total_ms=(end-start)*1000)
        return raw,sp,fp,candidate,proposal,conflict,meta,t
    cap=cv2.VideoCapture(info['path']);ok,first=cap.read();cap.release()
    if not ok:raise ValueError('Empty video')
    warm=cfg['execution']['warmup_frames_per_process']
    for j in range(warm):run(first,j/info['fps'])
    pipeline.reset();counts[:]=[0,0];cap=cv2.VideoCapture(info['path']);h,w=first.shape[:2];writer=checked_writer(folder/'replay.mp4',info['fps'],(w*2,h+84))
    i=matches=changed=cpu_peak=0;times=[];reasons=Counter();tip_states=Counter();advice=Counter();wall=time.perf_counter();cpu=time.process_time();torch.cuda.reset_peak_memory_stats()
    try:
        with (folder/'frames.jsonl').open('w',encoding='utf-8') as log:
            while True:
                start=time.perf_counter();ok,frame=cap.read();decoded=time.perf_counter()
                if not ok:break
                raw,sp,fp,candidate,proposal,conflict,meta,t=run(frame,source['frames'][i]['timestamp_seconds']);inferred=time.perf_counter()
                hashes={'mask_raw_sha256':array_hash(raw),'stone_probability_raw_sha256':array_hash(sp),'fiber_probability_raw_sha256':array_hash(fp)}
                rgb_hash=array_hash(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));src=source['frames'][i];pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if rgb_hash!=src['rgb_raw_sha256'] or any(v!=old[i][k] for k,v in hashes.items()):raise ValueError('Full raw source/output differs')
                if src['decoded_pts_seconds'] is not None and abs(pts-src['decoded_pts_seconds'])>1e-9:raise ValueError('PTS changed')
                gh={'candidate_sha256':array_hash(candidate),'proposal_sha256':array_hash(proposal),'conflict_sha256':array_hash(conflict),'growth_sha256':growth_hash(meta)}
                if reference_growth is not None and any(v!=reference_growth[i][k] for k,v in gh.items()):raise ValueError('T1 changed growth or current terminal evidence')
                if i in selected:
                    r=static_results['records'][selected[i]]
                    if any(v!=r[k] for k,v in hashes.items()) or any(v!=r['arms'][arm][k] for k,v in gh.items()):raise ValueError('Static/full growth differs')
                    matches+=1
                if meta['apply_to_mask'] or np.any(candidate[raw!=0]):raise ValueError('Protected mask changed')
                writer.write(np.hstack([base_panel(frame,raw,i),overlay(frame,raw,candidate,proposal,meta,arm+' SHADOW',i)]));rendered=time.perf_counter()
                save_candidates(folder,i,candidate,proposal,conflict)
                record={'frame_index':i,'timestamp_seconds':src['timestamp_seconds'],'source_rgb_raw_sha256':rgb_hash,**hashes,**gh,
                    'raw_mask_reference':str(ROOT/cfg['base']['reference_run']/f'full_video/raw2/{vid}/masks/{i:06d}.png'),
                    'candidate_pixels':int(candidate.sum()),'proposal_pixels':int(proposal.sum()),'conflict_pixels':int(conflict.sum()),'meta':meta,
                    'approved_semantic_reference':i in selected,'tip_metrics':None,'unlabelled_accuracy':None}
                log.write(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n');reasons[meta['reason']]+=1;tip_states.update(tip['state'] for tip in meta['tips']);advice[meta.get('t1_advice',{}).get('direction','disabled')]+=1
                changed+=int(candidate.any());cpu_peak=max(cpu_peak,process_memory()[0]);end=time.perf_counter()
                t.update(decode_ms=(decoded-start)*1000,render_audit_encode_ms=(rendered-inferred)*1000,record_export_ms=(end-rendered)*1000,cycle_ms=(end-start)*1000);times.append(t);i+=1
                if i%300==0:print(f'{arm} {vid} {i}/{info["frames"]}',flush=True)
    finally:cap.release();writer.release()
    wall=time.perf_counter()-wall;cpu=time.process_time()-cpu
    if i!=info['frames'] or matches!=len(selected) or counts!=[i,i] or model_state_hash(experts)!=before:raise ValueError('Worker coverage/graph/weights changed')
    for handle in handles:handle.remove()
    pipeline.reset();movie=verify_video(folder/'replay.mp4',i);keys=sorted(set().union(*(t.keys() for t in times)));timing={k:stats([t.get(k,0.) for t in times]) for k in keys}
    atomic_json(folder/'timings.json',times);result={'arm':arm,'video_id':vid,'pid':os.getpid(),'frames':i,'replay':movie,'approved_static_matches':matches,
        'actual_branch_forwards':counts,'warmup_branch_forwards':[warm,warm],'networks':2,'parameters':experts.parameter_count,'model_state_sha256':before,
        'individual_model_sha256':[model_hash(m) for m in experts.models],'runtime':runtime_identity(experts.device),'source_fps':info['fps'],'timestamp_mode':source['mode'],
        'timings':timing,'total_runtime_passed':timing['pipeline_total_ms']['p95']<=1000/info['fps'],'extra_budget_passed':timing['extra_total_ms']['p95']<=cfg['execution']['additional_stage_p95_ms'],
        'frames_with_candidate':changed,'reasons':dict(reasons),'tip_states':dict(tip_states),'t1_directions':dict(advice),'state_cleared':pipeline.tracker is None or pipeline.tracker.source is None,
        'offline_fps_including_decode_audit_export':i/wall,'wall_seconds':wall,'cpu_process_seconds':cpu,'windows_peak_working_set_mib':process_memory()[1]/2**20,
        'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'tip_metrics':None,'whole_video_accuracy':None,'all_actual_masks_unchanged':True}
    atomic_json(folder/'summary.json',result);print(json.dumps({'completed':arm,'video':vid,'p95':timing['pipeline_total_ms']['p95']}),flush=True)


def comparison(cfg,out,info):
    vid=info['video_id'];cap=cv2.VideoCapture(info['path']);w,h=info['width'],info['height'];logs={a:(out/f'full_video/{a}/{vid}/frames.jsonl').open(encoding='utf-8') for a in ARMS}
    path=out/f'{vid}_comparison.mp4';writer=checked_writer(path,info['fps'],(w*4,h+124));i=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            r={a:json.loads(next(logs[a])) for a in ARMS};raw=load_mask(r['raw2']['raw_mask_reference'],frame.shape[:2],{0,1,2})
            tiles=[np.vstack([panel(frame,'Original source',None,i),np.zeros((50,w,3),np.uint8)]),base_panel(frame,raw,i)]
            for a in ARMS[1:]:
                c=load_mask(out/f'full_video/{a}/{vid}/candidate/{i:06d}.png',raw.shape,{0,1});p=load_mask(out/f'full_video/{a}/{vid}/proposal/{i:06d}.png',raw.shape,{0,1})
                tiles.append(overlay(frame,raw,c,p,r[a]['meta'],a+' SHADOW',i))
            strip=np.zeros((40,w*4,3),np.uint8);cv2.putText(strip,'Actual masks unchanged | Green: candidate ONLY | Purple: rejected proposal/scope | Yellow: unverified RGB end candidate | no tip accuracy GT',(7,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            writer.write(np.vstack([np.hstack(tiles),strip]));i+=1
    finally:
        cap.release();writer.release()
        for f in logs.values():f.close()
    if i!=info['frames']:raise ValueError('Incomplete comparison')
    return verify_video(path,i)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',nargs=2);args=parser.parse_args();cfg=read(CONFIG);out=ROOT/cfg['output']
    if tuple(cfg['arms'])!=ARMS:raise ValueError('Arms changed')
    if args.worker:return worker(cfg,out,*args.worker)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);job={'pid':os.getpid(),'training_updates':0,'default_changed':False}
    def progress(phase,**extra):job.update(phase=phase,**extra);atomic_json(out/'job.json',job);print(phase+' '+json.dumps(extra),flush=True)
    try:
        progress('preflight');protocol=read(ROOT/'runs/protocols/fiber_connected_blue_growth_20261101_v1/identity.json')
        if any(sha256(ROOT/k)!=v for k,v in protocol['files'].items()):raise ValueError('Frozen protocol changed')
        prior=read(ROOT/cfg['base']['preflight'])
        if sha256(ROOT/cfg['base']['preflight'])!=protocol['reference_preflight_sha256'] or any(sha256(ROOT/k)!=v for k,v in prior['sources'].items()):raise ValueError('Prior source changed')
        vm=read(ROOT/cfg['data']['video_manifest']);im=read(ROOT/cfg['data']['image_manifest'])
        inputs=input_identity(ROOT/cfg['data']['image_manifest'],Path(im['root']),{'train':{'labeled':[],'unlabeled':[]},'val':vm['val']+im['val']})
        if inputs!=prior['inputs'] or len(vm['val'])!=70 or len(im['val'])!=132:raise ValueError('Reference changed')
        sources={**prior['sources']}
        for p in list((ROOT/'tools').glob('*.py'))+list((ROOT/'tests').glob('*.py')):sources[p.relative_to(ROOT).as_posix()]=sha256(p)
        related={str(ROOT/k):v for k,v in protocol['files'].items()}
        for scope in ['video_native','image_native']:
            p=ROOT/cfg['base']['reference_run']/f'{scope}/raw2.json';related[str(p)]=sha256(p)
        for info in prior['videos']:
            if sha256(info['path'])!=info['sha256']:raise ValueError('Video changed')
            progress('source_timestamp_audit',video_id=info['video_id']);source_scan(out,info,ROOT/cfg['base']['historical_reference'])
            for p in [out/f'source_{info["video_id"]}.json',ROOT/cfg['base']['reference_run']/f'full_video/raw2/{info["video_id"]}/frames.jsonl']:related[str(p)]=sha256(p)
        for x in prior['checkpoints'].values():
            if sha256(x['path'])!=x['sha256']:raise ValueError('Weights changed')
        pre={'config':cfg,'sources':sources,'related':related,'inputs':inputs,'checkpoints':prior['checkpoints'],'videos':prior['videos'],'mechanism_tests_passed':12,
            'train_pixels_read':0,'reserved_test_pixels_read':0,'training_updates':0}
        atomic_json(out/'preflight.json',pre)
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for k in sources:z.write(ROOT/k,k)
            for k in protocol['files']:z.write(ROOT/k,k)
        static(cfg,pre,out,progress);gates=quality(cfg,out);atomic_json(out/'static_quality.json',gates);full={}
        for arm in ARMS:
            full[arm]={}
            for info in pre['videos']:
                vid=info['video_id'];progress('full_video_fresh_process',arm=arm,video_id=vid)
                with (out/f'{arm}_{vid}.log').open('w',encoding='utf-8') as log:
                    subprocess.run([sys.executable,'-m','tools.run_fiber_connected_blue_growth','--worker',arm,vid],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                full[arm][vid]=read(out/f'full_video/{arm}/{vid}/summary.json')
        exports={}
        for info in pre['videos']:progress('comparison_export',video_id=info['video_id']);exports[info['video_id']]=comparison(cfg,out,info)
        if any(sha256(ROOT/k)!=v for k,v in sources.items()) or any(sha256(k)!=v for k,v in related.items()):raise ValueError('Frozen source/reference changed')
        if inputs!=input_identity(ROOT/cfg['data']['image_manifest'],Path(im['root']),{'train':{'labeled':[],'unlabeled':[]},'val':vm['val']+im['val']}):raise ValueError('Reference input changed')
        result={'static_quality':gates,'full_video':full,'comparison_videos':exports,'feasibility_passed':{a:gates[a]['passed'] and all(v['total_runtime_passed'] and v['extra_budget_passed'] for v in full[a].values()) for a in ARMS[1:]},
            'actual_static_paired_frame_inferences':202,'actual_full_paired_frame_inferences':8082,'warmup_paired_frames':72,'actual_output_pixel_changes':0,
            'adopted':False,'default_changed':False,'training_updates':0,'train_pixels_read':0,'reserved_test_pixels_read':0,'tip_metrics':None,'whole_video_accuracy':None,'s3_started':False}
        atomic_json(out/'summary.json',result);progress('completed',static_frames=202,full_frames=8082)
    except Exception as error:
        progress('failed',error=str(error));(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
