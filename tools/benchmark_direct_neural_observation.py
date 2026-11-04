"""Isolated, exactly-required-network timing for the observation control."""
import argparse
import json
import os
from pathlib import Path
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ureter_semis.experiment import atomic_json, seed_training
from ureter_semis.inference import SegmentationPredictor
from ureter_semis.model import ensemble_logits
from .classwise_experts import ClasswiseExperts
from .direct_neural_observation import IMAGE, locate
from .replay_classwise_experts import pil_input

ROOT = Path(__file__).resolve().parents[1]
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--arm', required=True); args = parser.parse_args()
    run = read(ROOT/'configs/run_direct_neural_observation_20261029.json'); cost = run['benchmark']
    if args.arm not in run['arms']: raise ValueError(args.arm)
    path = ROOT/run['output']/f'performance_{args.arm}.json'
    if path.exists(): raise FileExistsError(path)
    config = read(ROOT/run['base_config']); prior = read(ROOT/run['template'])
    cap = cv2.VideoCapture(read(ROOT/'runs/baselines/a0_20261027/config.json')['inputs'][0]['path']); frames = []
    for _ in range(cost['cached_frames']):
        ok, frame = cap.read()
        if not ok: raise ValueError('Short benchmark video')
        frames.append(frame)
    cap.release(); checkpoints = read(ROOT/config['experts'])['checkpoints']; seed_training(42)
    one = args.arm == IMAGE
    expert = SegmentationPredictor(checkpoints['old190']['path']) if one else ClasswiseExperts(checkpoints['old190']['path'], checkpoints['A0']['path'])
    def infer(frame):
        torch.cuda.synchronize(expert.device); started = time.perf_counter(); image = pil_input(frame, expert.size, expert.device)
        if one:
            p = F.interpolate(ensemble_logits(expert.models, image).softmax(1), size=frame.shape[:2], mode='bilinear', align_corners=False)[0]
            stone = p.argmax(0).byte().cpu().numpy(); probability = None
        else:
            result = expert.predict_native(image, [frame.shape[:2]])[0]
            stone = result['stone_mask'].cpu().numpy(); raw = result['mask'].cpu().numpy(); probability = result['fiber_probability'][2].cpu().numpy()
        torch.cuda.synchronize(expert.device); inferred = time.perf_counter()
        if args.arm == 'raw_classwise': mask = raw
        else:
            outputs, _ = locate(frame, prior, config, probability, requested=[args.arm])
            mask = np.where(outputs[args.arm]['fiber'], 2, np.where(stone == 1, 1, 0)).astype(np.uint8)
        finished = time.perf_counter()
        return {'pipeline_ms':(finished-started)*1000, 'required_model_core_ms':(inferred-started)*1000, 'geometry_and_fusion_ms':(finished-inferred)*1000}
    for i in range(cost['warmup_frames']): infer(frames[i % len(frames)])
    torch.cuda.reset_peak_memory_stats(); samples = [infer(f) for _ in range(cost['repeats']) for f in frames]
    atomic_json(path, {'arm':args.arm, 'pid':os.getpid(), 'model_count':len(expert.models), 'parameter_count':sum(p.numel() for m in expert.models for p in m.parameters()),
        'warmup_frames':cost['warmup_frames'], 'samples':len(samples), 'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20, 'scope':cost['scope'],
        'timings':{k:{'mean':float(np.mean([r[k] for r in samples])), 'p50':float(np.median([r[k] for r in samples])), 'p95':float(np.percentile([r[k] for r in samples],95))} for k in samples[0]}, 'raw_samples':samples})

if __name__ == '__main__': main()
