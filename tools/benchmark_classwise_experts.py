"""Fresh-process3network shared-input expert forward/native-output/fusion cost."""
import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from ureter_semis.data import load_manifest
from ureter_semis.experiment import atomic_json,seed_training,sha256
from ureter_semis.inference import preprocess
from .classwise_experts import ClasswiseExperts,native_outputs


@torch.inference_mode()
def benchmark(manifest,stone,fiber,output,warmup=12,repeats=3):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    seed_training(42);root,m=load_manifest(manifest)
    frames=[cv2.imread(str(root/row['image'])) for row in m['val']]
    if any(frame is None for frame in frames):raise ValueError('Missing cached frame')
    experts=ClasswiseExperts(stone,fiber)
    def infer(frame):
        experts.sync();start=time.perf_counter()
        image=preprocess(frame,experts.size,experts.device)
        experts.sync();prepared=time.perf_counter()
        sp,fp=experts.probabilities(image)
        experts.sync();forward=time.perf_counter()
        result=native_outputs(sp[0],fp[0],frame.shape[:2])
        mask=result['mask'].cpu().numpy()
        # These are two distinct expert maps, not calibrated fused probabilities.
        stone_map=result['stone_probability'].cpu().numpy();fiber_map=result['fiber_probability'].cpu().numpy()
        experts.sync();finished=time.perf_counter()
        return {'pipeline_ms':(finished-start)*1000,'preprocess_ms':(prepared-start)*1000,
            'model_ms':(forward-prepared)*1000,'native_output_and_fusion_ms':(finished-forward)*1000}
    for i in range(warmup):infer(frames[i%len(frames)])
    torch.cuda.reset_peak_memory_stats()
    samples=[infer(frame) for _ in range(repeats) for frame in frames]
    atomic_json(output,{'checkpoint_sha256':{'stone':sha256(Path(stone)),'fiber':sha256(Path(fiber))},
        'model_count':experts.model_count,'parameter_count':experts.parameter_count,'input_size':experts.size,
        'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'warmup_frames':warmup,'samples':len(samples),
        'scope':'Fresh process batch1 cached BGR -> one shared256 preprocessed tensor -> ALL3U-Nets -> native expert probability maps and fiber-priority fused mask to CPU. Excludes decoding/display/geometry/memory; two expert probabilities not a calibrated fused distribution',
        'timings':{key:{'mean':float(np.mean([r[key] for r in samples])),'p50':float(np.percentile([r[key] for r in samples],50)),
            'p95':float(np.percentile([r[key] for r in samples],95))} for key in samples[0]},'raw_samples':samples})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--stone',required=True)
    p.add_argument('--fiber',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();benchmark(args.manifest,args.stone,args.fiber,args.output)
