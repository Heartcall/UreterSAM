"""Matched PIL/native-output full-frame versus sequential ROI compute proxy."""
import argparse
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from ureter_semis.data import load_manifest
from ureter_semis.experiment import atomic_json, seed_training, sha256, runtime_identity
from ureter_semis.inference import SegmentationPredictor
from .diagnose_refinement_roi import native_probability, read, ROOT
from .refinement_roi import propose_rois


@torch.inference_mode()
def benchmark(name, output):
    config=read(ROOT/"configs/diagnose_refinement_roi_20261028.json")
    seed_training(42)
    root,manifest=load_manifest(ROOT/config["manifest"])
    images=[Image.open(root/r["image"]).convert("RGB") for r in manifest["val"]]
    profile=next((p for p in config["profiles"] if p["name"]==name),None)
    if profile is None and name not in ("full256","full512"):raise ValueError(name)
    checkpoint=ROOT/config["checkpoint"] if name!="full512" else ROOT/"runs/a1_sup_kidney_resolution_20261028/unet512/last.pt"
    predictor=SegmentationPredictor(checkpoint)
    def infer(image):
        predictor._sync();start=time.perf_counter()
        probability=native_probability(predictor,image);mask=probability.argmax(0)
        predictor._sync();coarse=time.perf_counter()
        boxes=propose_rois(probability,config["proposal"],profile["seed"],profile["max_rois"]) if profile else []
        proposed=time.perf_counter()
        for box in boxes:
            crop_probability=native_probability(predictor,image.crop(box));crop_mask=crop_probability.argmax(0)
        predictor._sync();finish=time.perf_counter()
        return {"total_ms":(finish-start)*1000,"coarse_ms":(coarse-start)*1000,
                "proposal_ms":(proposed-coarse)*1000,"crop_calls_ms":(finish-proposed)*1000,"roi_count":len(boxes)}
    for i in range(config["benchmark"]["warmup_frames"]):infer(images[i%len(images)])
    torch.cuda.reset_peak_memory_stats()
    samples=[infer(image) for _ in range(config["benchmark"]["repeats"]) for image in images]
    atomic_json(Path(output),{"name":name,"checkpoint_sha256":sha256(checkpoint),"runtime":runtime_identity(predictor.device),
        "warmup_frames":config["benchmark"]["warmup_frames"],"samples":len(samples),
        "cuda_peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
        "scope":"Fresh process; cached RGB/PIL256 or512; FP32 probabilities to native CPU and NumPy mask; automatic proposal plus sequential192-native crops resized256; same frozen U-Net compute proxy, no crop accuracy/fusion/decode/display/geometry/memory; no concurrent project tests",
        "timings":{k:{"mean":float(np.mean([r[k] for r in samples])),"p50":float(np.percentile([r[k] for r in samples],50)),
            "p95":float(np.percentile([r[k] for r in samples],95))} for k in samples[0] if k!="roi_count"},"raw_samples":samples})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();benchmark(args.name,args.output)
