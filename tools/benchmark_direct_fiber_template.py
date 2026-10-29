import argparse
import json
import os
from pathlib import Path
import time
import cv2
import torch
import torch.nn.functional as F
import numpy as np

from ureter_semis.experiment import atomic_json,seed_training
from ureter_semis.inference import SegmentationPredictor
from ureter_semis.model import ensemble_logits
from .classwise_experts import ClasswiseExperts
from .direct_fiber_template import DirectTracker,locate
from .replay_classwise_experts import pil_input

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/a1_direct_fiber_template_20261029'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',required=True);args=parser.parse_args()
    config=read(ROOT/'configs/direct_fiber_template_20261029.json');cost=read(ROOT/'configs/benchmark_direct_fiber_template_20261029.json');prior=read(OUT/'template.json')
    if args.arm not in cost['arms']:raise ValueError(args.arm)
    path=OUT/f'performance_{args.arm}.json'
    if path.exists():raise FileExistsError(path)
    source=read(ROOT/'runs/baselines/a0_20261027/config.json')['inputs'][0]['path'];cap=cv2.VideoCapture(source);frames=[]
    for _ in range(cost['cached_frames']):
        ok,frame=cap.read()
        if not ok:raise ValueError('Short benchmark video')
        frames.append(frame)
    cap.release();checkpoints=read(ROOT/config['experts'])['checkpoints'];seed_training(42)
    one=args.arm=='image_template'
    expert=SegmentationPredictor(checkpoints['old190']['path']) if one else ClasswiseExperts(checkpoints['old190']['path'],checkpoints['A0']['path'])
    tracker=DirectTracker(config['tracking']) if args.arm=='causal_template' else None
    def sync():torch.cuda.synchronize(expert.device)
    def infer(frame):
        sync();start=time.perf_counter();image=pil_input(frame,expert.size,expert.device)
        if one:
            p=F.interpolate(ensemble_logits(expert.models,image).softmax(1),size=frame.shape[:2],mode='bilinear',align_corners=False)[0]
            stone=p.argmax(0).byte().cpu().numpy();raw=None;probability=None
        else:
            result=expert.predict_native(image,[frame.shape[:2]])[0];stone=result['stone_mask'].cpu().numpy();raw=result['mask'].cpu().numpy();probability=result['fiber_probability'][2].cpu().numpy()
        sync();inferred=time.perf_counter()
        if args.arm=='raw_classwise':mask=raw
        else:
            outputs,_=locate(frame,prior,config,probability if not one else None,tracker,requested=[args.arm])
            fiber=outputs[args.arm]['fiber'];mask=np.where(fiber,2,np.where(stone==1,1,0)).astype(np.uint8)
        end=time.perf_counter()
        return {'pipeline_ms':(end-start)*1000,'required_model_core_ms':(inferred-start)*1000,'direct_geometry_and_fusion_ms':(end-inferred)*1000}
    for i in range(cost['warmup_frames']):infer(frames[i%len(frames)])
    torch.cuda.reset_peak_memory_stats();samples=[]
    for _ in range(cost['repeats']):
        if tracker:tracker.reset()
        samples.extend(infer(f) for f in frames)
    atomic_json(path,{'arm':args.arm,'pid':os.getpid(),'model_count':len(expert.models),'parameter_count':sum(p.numel() for m in expert.models for p in m.parameters()),
        'warmup_frames':cost['warmup_frames'],'samples':len(samples),'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'scope':cost['scope'],
        'timings':{k:{'mean':float(np.mean([r[k] for r in samples])),'p50':float(np.median([r[k] for r in samples])),'p95':float(np.percentile([r[k] for r in samples],95))} for k in samples[0]},'raw_samples':samples})


if __name__=='__main__':main()
