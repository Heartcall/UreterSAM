"""One fresh process loads exactly the arm's required1 or3 networks."""
import argparse
from pathlib import Path
import json
import os
import cv2
import numpy as np
from PIL import Image
import torch

from ureter_semis.experiment import atomic_json,seed_training,sha256
from .fiber_component_gate import ComponentGatePredictor,ARMS

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',required=True,choices=ARMS);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    protocol=read(ROOT/'configs/run_fiber_component_gate_20261029.json')
    geometry=read(ROOT/protocol['geometry_config'])['fit'];checkpoints=read(ROOT/protocol['expert_manifest'])['checkpoints']
    manifest=read(ROOT/'datasets/ureter_semis_kidney_joint_20261028/video_validation_manifest.json')
    frames=[cv2.cvtColor(np.asarray(Image.open(row['image']).convert('RGB')),cv2.COLOR_RGB2BGR) for row in manifest['val']]
    seed_training(protocol['seed']);predictor=ComponentGatePredictor(args.arm,checkpoints,geometry)
    for i in range(12):predictor.predict(frames[i%len(frames)])
    torch.cuda.reset_peak_memory_stats();measurements=[predictor.predict(frame)['timing'] for _ in range(3) for frame in frames]
    atomic_json(out,{'arm':args.arm,'pid':os.getpid(),'model_count':predictor.model_count,'parameter_count':predictor.parameter_count,
        'checkpoint_sha256':{n:sha256(Path(checkpoints[n]['path'])) for n in (('old190','A0') if predictor.classwise else ('old190',))},
        'warmup_frames':12,'samples':len(measurements),'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'scope':'Fresh process, cached RGB native frames via PIL256; exactly required1or3models, native outputs and fiber probability CPU transfer, gate if enabled, fusion. No decode/export/display/history',
        'timings':{key:{'mean':float(np.mean([r[key] for r in measurements])),'p50':float(np.median([r[key] for r in measurements])),
            'p95':float(np.percentile([r[key] for r in measurements],95))} for key in measurements[0]},'raw_samples':measurements})


if __name__=='__main__':main()
