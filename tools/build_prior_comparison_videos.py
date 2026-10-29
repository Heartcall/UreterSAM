"""Render all completed prior variants together; missing full runs are visualization only."""
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import time
import traceback
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from ureter_semis.experiment import atomic_json, seed_training, sha256, runtime_identity, source_identity
from .classwise_experts import ClasswiseExperts
from .direct_neural_observation import NEURAL, locate
from .fiber_component_gate import gate_fiber
from .fiber_shape_prior import combine
from .replay_classwise_experts import pil_input, model_state_hash, save_png

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/prior_method_comparison_20261029'
FONT=Path('C:/Windows/Fonts/msyh.ttc')
NAMES=('classwise_raw','old190_raw','edge_geometry','edge_geometry_causal','component_gate_3net','component_gate_1net','fixed_color','fixed_color_soft','fixed_color_causal','fixed_neural')
TITLES=('参照：原始三网类别组合','参照：旧190单网','先验1：双边直线＋圆钝末端','先验2：几何拟合＋因果姿态','先验3：候选筛选（三网）','先验4：候选筛选（单网）','先验5：固定模板＋颜色观测','先验6：固定模板＋网络软评分','先验7：固定模板＋因果姿态','先验8：固定模板＋概率观测')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def load(p):
    with Image.open(p) as im:return np.asarray(im).copy()
def cached(vid,i):
    file=f'{i:06d}.png'
    return {
        'classwise_raw':ROOT/f'runs/a1_classwise_replay_20261028/{vid}/classwise'/file,
        'old190_raw':ROOT/f'runs/a1_classwise_replay_20261028/{vid}/old190'/file,
        'edge_geometry':ROOT/f'runs/a1_fiber_geometry_20261028/{vid}/geometry'/file,
        'edge_geometry_causal':ROOT/f'runs/a1_fiber_geometry_20261028/{vid}/geometry_causal'/file,
        'component_gate_3net':ROOT/f'runs/a1_fiber_component_gate_20261029/full_replay/classwise_component_gate_3net/{vid}/gated'/file,
        'fixed_color':ROOT/f'runs/a1_direct_fiber_template_20261029/{vid}/image_template'/file,
        'fixed_color_soft':ROOT/f'runs/a1_direct_fiber_template_20261029/{vid}/soft_template'/file,
        'fixed_color_causal':ROOT/f'runs/a1_direct_fiber_template_20261029/{vid}/causal_template'/file,
    }


@torch.inference_mode()
def infer_missing(videos,experts,prior,config,fit,reviewed,progress):
    result={}
    for item in videos:
        vid=item['video_id'];folder=OUT/vid;folder.mkdir()
        for n in ('component_gate_1net','fixed_neural'):(folder/n).mkdir()
        selected={r['frame_index']:(j,r) for j,r in enumerate(reviewed) if r['video_id']==vid}
        cap=cv2.VideoCapture(item['path']);counts=Counter();parity=Counter();i=0;started=time.perf_counter()
        try:
            with (folder/'supplemental_inference.jsonl').open('w',encoding='utf-8') as log:
                while True:
                    ok,frame=cap.read()
                    if not ok:break
                    if i>=item['frames'] or frame.shape[:2]!=(item['height'],item['width']):raise ValueError('Unexpected video geometry/count')
                    image=pil_input(frame,experts.size,experts.device);native=experts.predict_native(image,[frame.shape[:2]])[0]
                    raw=native['mask'].cpu().numpy();old=native['stone_mask'].cpu().numpy();p=native['fiber_probability'][2].cpu().numpy();oldp=native['stone_probability'][2].cpu().numpy()
                    paths=cached(vid,i)
                    parity['classwise_historical_different_pixels']+=int((raw!=load(paths['classwise_raw'])).sum())
                    parity['old190_historical_different_pixels']+=int((old!=load(paths['old190_raw'])).sum())
                    fiber,gate=gate_fiber(frame,oldp,old==2,fit)
                    outputs,_=locate(frame,prior,config,p,requested=[NEURAL]);record=outputs[NEURAL]['record']
                    masks={'component_gate_1net':combine(old,fiber),'fixed_neural':combine(old,outputs[NEURAL]['fiber'])}
                    for n,mask in masks.items():save_png(folder/n/f'{i:06d}.png',mask)
                    counts['component_gate_1net_accepted']+=int(gate['accepted']);counts['fixed_neural_accepted']+=int(record['accepted'])
                    if i in selected:
                        j,row=selected[i]
                        if not np.array_equal(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),np.asarray(Image.open(row['image']).convert('RGB'))):raise ValueError('Reviewed image alignment differs')
                        previous={
                            'component_gate_1net':ROOT/f'runs/a1_fiber_component_gate_20261029/video_native/old190_component_gate_1net/{j:04d}.png',
                            'fixed_neural':ROOT/f'runs/a1_direct_neural_observation_20261029/video_native/{NEURAL}/{j:04d}.png'}
                        for n,path in previous.items():parity[n+'_static_different_pixels']+=int((masks[n]!=load(path)).sum())
                        counts['reviewed_rgb_exact']+=1
                    log.write(json.dumps({'frame_index':i,'one_based_frame':i+1,'old190_gate':gate,'fixed_neural':record,'scope':'Visualization-only supplement; no threshold change, training or acceptance decision'},allow_nan=False)+'\n')
                    i+=1
                    if i%100==0:progress('supplemental_inference',video_id=vid,frames_completed=i,expected_frames=item['frames']);print(f'infer {vid} {i}/{item["frames"]}',flush=True)
            if i!=item['frames'] or counts['reviewed_rgb_exact']!=len(selected) or any(parity.values()):raise ValueError(f'Full/static reproduction mismatch: {dict(parity)}')
            result[vid]={'frames':i,'counts':dict(counts),'parity':dict(parity),'wall_seconds':time.perf_counter()-started,'scope':'User-requested visualization supplement, not conditional acceptance replay or a timing benchmark'}
            atomic_json(folder/'supplemental_summary.json',result[vid])
        finally:cap.release()
    return result


def text_panel(lines,width,height,font_sizes=(23,17),background=(19,26,37)):
    im=Image.new('RGB',(width,height),background);d=ImageDraw.Draw(im)
    for text,xy,size,color in lines:d.text(xy,text,font=ImageFont.truetype(str(FONT),size),fill=color)
    return cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)


def render(videos,quality,progress):
    report={};width=1600;tileh=452;height=80+3*tileh;size=400
    for item in videos:
        vid=item['video_id'];folder=OUT/vid;cap=cv2.VideoCapture(item['path']);fps=float(cap.get(cv2.CAP_PROP_FPS))
        output=folder/f'{vid}_all_priors_comparison.mp4'
        writer=cv2.VideoWriter(str(output),cv2.CAP_MSMF,cv2.VideoWriter_fourcc(*'H264'),fps,(width,height))
        if not writer.isOpened():raise RuntimeError('H264 Media Foundation encoder unavailable')
        header=text_panel([('原画面',(10,3),22,'white'),('所有面板为同一帧；不填补遮挡区域',(10,29),16,(195,206,222))],size,52)
        headers={n:text_panel([(t,(10,3),21,'white'),(f'29帧光纤Dice {quality[n]:.4f}'+('  · 展示补齐' if n in ('component_gate_1net','fixed_neural') else ''),(10,29),16,(195,206,222))],size,52) for n,t in zip(NAMES,TITLES)}
        legend=text_panel([
            ('阅读说明',(16,20),25,'white'),('红色：结石    青色：光纤',(16,78),23,(235,235,235)),
            ('1–2：随帧拟合几何及其因果跟踪',(16,132),20,(198,211,229)),('3–4：筛选组件，保留原始边界',(16,170),20,(198,211,229)),
            ('5–7：同一固定模板，颜色作可见支持',(16,208),19,(198,211,229)),('8：同一固定模板，概率作可见支持',(16,246),19,(198,211,229)),
            ('Dice仅来自29帧参考，不是全片真值',(16,311),19,(255,211,122)),('失败方案供比较，均未替换默认模型',(16,348),19,(255,211,122)),
            ('因果跟踪仅指光纤姿态，未做结石记忆',(16,385),18,(198,211,229))],size,tileh)
        i=0;identities=hashlib.sha256();selected={0,item['frames']//2,item['frames']-1};sample_files=[]
        try:
            with (folder/'render_sources.jsonl').open('w',encoding='utf-8') as sources:
                while True:
                    ok,frame=cap.read()
                    if not ok:break
                    paths=cached(vid,i);paths.update({n:folder/n/f'{i:06d}.png' for n in ('component_gate_1net','fixed_neural')})
                    masks={};hashes={}
                    for n,path in paths.items():
                        data=path.read_bytes();hashes[n]=hashlib.sha256(data).hexdigest()
                        with Image.open(io.BytesIO(data)) as im:masks[n]=np.asarray(im).copy()
                        if masks[n].shape!=(400,400) or not set(np.unique(masks[n])).issubset({0,1,2}):raise ValueError('Bad native mask')
                    line=json.dumps({'frame_index':i,'source_sha256':hashes},sort_keys=True)+'\n';sources.write(line);identities.update(line.encode())
                    tiles=[np.vstack([header,frame])]
                    for n in NAMES:
                        panel=frame.copy();mask=masks[n]
                        for k,color in [(1,(40,40,235)),(2,(235,210,20))]:
                            a=mask==k;panel[a]=(.43*frame[a]+.57*np.asarray(color)).astype(np.uint8)
                            # A thin outline makes the narrow fiber legible against its own blue RGB appearance.
                            if k==2:
                                contours,_=cv2.findContours(a.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(panel,contours,-1,color,1)
                        tiles.append(np.vstack([headers[n],panel]))
                    tiles.append(legend)
                    canvas=np.vstack([np.hstack(tiles[j:j+4]) for j in (0,4,8)])
                    banner=text_panel([(f'{vid}  |  帧 {i+1:04d}/{item["frames"]}  |  {i/fps:06.2f} 秒  |  同步先验方法对比',(16,6),27,'white'),
                        ('离线展示：沿用原参数与完整可见分割；所有失败方案保留，未改变模型验收或训练状态。',(16,44),20,(195,206,222))],width,80)
                    canvas=np.vstack([banner,canvas]);writer.write(canvas)
                    if i in selected:
                        name=f'comparison_frame_{i+1:06d}.jpg';cv2.imwrite(str(folder/name),canvas,[cv2.IMWRITE_JPEG_QUALITY,95]);sample_files.append(name)
                    i+=1
                    if i%100==0:progress('rendering_h264',video_id=vid,frames_completed=i,expected_frames=item['frames']);print(f'render {vid} {i}/{item["frames"]}',flush=True)
        finally:cap.release();writer.release()
        if i!=item['frames']:raise ValueError('Incomplete render')
        # Decode the entire exported file: metadata-only frame counts do not establish completeness.
        check=cv2.VideoCapture(str(output));decoded=0;actualfps=float(check.get(cv2.CAP_PROP_FPS));samples=[]
        while True:
            ok,frame=check.read()
            if not ok:break
            if frame.shape[:2]!=(height,width):raise ValueError('Wrong encoded geometry')
            if decoded in selected:
                reference=cv2.imread(str(folder/f'comparison_frame_{decoded+1:06d}.jpg'));samples.append({'one_based_frame':decoded+1,'psnr_against_jpeg_reference':float(cv2.PSNR(frame,reference))})
            decoded+=1
        check.release()
        if decoded!=i or abs(actualfps-fps)>.005 or min(x['psnr_against_jpeg_reference'] for x in samples)<25:raise ValueError('Encoded video audit failed')
        report[vid]={'path':str(output),'frames':i,'decoded_frames':decoded,'fps':actualfps,'source_fps':fps,'duration_seconds':i/actualfps,'width':width,'height':height,'codec':'H264 via Windows Media Foundation','sha256':sha256(output),'bytes':output.stat().st_size,'source_mask_identity_chain':identities.hexdigest(),'preview_images':sample_files,'encoded_samples':samples}
        atomic_json(folder/'render_summary.json',report[vid]);progress('video_export_verified',video_id=vid,frames_completed=i,expected_frames=i)
    return report


def main():
    if OUT.exists():raise FileExistsError('Never overwrite prior exports')
    OUT.mkdir();job={'pid':os.getpid(),'scope':'User-requested visual comparison; previous failed static candidates replayed for display only','training_updates':0,'default_model_changed':False,'acceptance_decision_changed':False}
    def progress(phase,**extra):job.update(phase=phase,**extra);atomic_json(OUT/'job.json',job)
    progress('preflight')
    try:
        config=read(ROOT/'configs/direct_fiber_template_20261029.json');prior=read(ROOT/'runs/a1_direct_fiber_template_20261029/template.json');fit=read(ROOT/'configs/fiber_geometry_prior_20261028.json')['fit']
        experts_manifest=read(ROOT/config['experts']);checkpoints={n:v for n,v in experts_manifest['checkpoints'].items() if n in ('old190','A0')}
        videos=read(ROOT/'runs/baselines/a0_20261027/config.json')['inputs'];reviewed=read(ROOT/'datasets/ureter_semis_kidney_joint_20261028/video_validation_manifest.json')['val']
        for v in list(checkpoints.values())+videos:
            if sha256(Path(v['path']))!=v['sha256']:raise ValueError('Video/checkpoint identity differs')
        frozen={n:ROOT/n for n in ['configs/direct_fiber_template_20261029.json','configs/fiber_geometry_prior_20261028.json','runs/a1_direct_fiber_template_20261029/template.json','runs/a1_fiber_geometry_20261028/summary.json','runs/a1_fiber_component_gate_20261029/summary.json','runs/a1_direct_fiber_template_20261029/summary.json','runs/a1_direct_neural_observation_20261029/summary.json','annotations/data_review_status_20261027.json']}
        sources=source_identity();sources.update({n:sha256(ROOT/n) for n in ['tools/build_prior_comparison_videos.py','tools/classwise_experts.py','tools/direct_neural_observation.py','tools/direct_fiber_template.py','tools/fiber_component_gate.py','tools/fiber_shape_prior.py','tools/replay_classwise_experts.py']})
        for item in videos:
            for i in range(item['frames']):
                if not all(p.is_file() for p in cached(item['video_id'],i).values()):raise FileNotFoundError(f'Missing frozen source mask at {item["video_id"]}:{i}')
        preflight={'videos':videos,'checkpoints':checkpoints,'base_config':config,'shape':prior,'fit':fit,'sources':sources,'frozen_metadata':{n:sha256(p) for n,p in frozen.items()},'layout':'4 columns x3 rows, 400x400 native image per panel plus52px title and80px overall header','variants':dict(zip(NAMES,TITLES)),'training_updates':0,'reserved_test_pixels_read':False,'scope':job['scope']}
        atomic_json(OUT/'preflight.json',preflight)
        with zipfile.ZipFile(OUT/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for n in sources:z.write(ROOT/n,n.replace('\\','/'))
        seed_training(42);experts=ClasswiseExperts(checkpoints['old190']['path'],checkpoints['A0']['path']);before=model_state_hash(experts);atomic_json(OUT/'runtime.json',runtime_identity(experts.device))
        supplement=infer_missing(videos,experts,prior,config,fit,reviewed,progress)
        assert before==model_state_hash(experts) and all(not m.training for m in experts.models)
        del experts;torch.cuda.empty_cache()
        geometry=read(frozen['runs/a1_fiber_geometry_20261028/summary.json']);component=read(frozen['runs/a1_fiber_component_gate_20261029/summary.json']);direct=read(frozen['runs/a1_direct_fiber_template_20261029/summary.json']);neural=read(frozen['runs/a1_direct_neural_observation_20261029/summary.json'])
        quality={'classwise_raw':component['video']['models']['classwise_raw_3net']['metrics']['fiber_dice'],'old190_raw':component['video']['models']['old190_raw_1net']['metrics']['fiber_dice'],
            'edge_geometry':geometry['video29']['geometry']['metrics']['fiber_dice'],'edge_geometry_causal':geometry['video29']['geometry_causal']['metrics']['fiber_dice'],
            'component_gate_3net':component['video']['models']['classwise_component_gate_3net']['metrics']['fiber_dice'],'component_gate_1net':component['video']['models']['old190_component_gate_1net']['metrics']['fiber_dice'],
            'fixed_color':direct['video29']['image_template']['metrics']['fiber_dice'],'fixed_color_soft':direct['video29']['soft_template']['metrics']['fiber_dice'],'fixed_color_causal':direct['video29']['causal_template']['metrics']['fiber_dice'],'fixed_neural':neural['video29']['models'][NEURAL]['metrics']['fiber_dice']}
        exports=render(videos,quality,progress)
        assert all(sha256(ROOT/n)==h for n,h in sources.items()) and all(sha256(ROOT/n)==h for n,h in preflight['frozen_metadata'].items())
        assert all(sha256(Path(v['path']))==v['sha256'] for v in list(checkpoints.values())+videos)
        atomic_json(OUT/'summary.json',{'exports':exports,'variants':dict(zip(NAMES,TITLES)),'quality_video29_only':quality,'supplemental_inference':supplement,'model_state_unchanged':before,'training_updates':0,'default_model_changed':False,'acceptance_decision_changed':False,'s2_accepted':False,'stone_memory_started':False,'reserved_test_pixels_read':False,'scope':job['scope']})
        progress('completed',frames_completed=2694,expected_frames=2694,video_id=None)
        print(json.dumps({'phase':'completed','exports':exports},indent=2),flush=True)
    except Exception as error:
        progress('failed',error=str(error));(OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':main()
