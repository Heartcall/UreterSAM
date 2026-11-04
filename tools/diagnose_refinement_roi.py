"""Frozen validation-only automatic ROI coverage and compute-proxy diagnosis."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from ureter_semis.data import _tensor, load_manifest
from ureter_semis.experiment import atomic_json, input_identity, runtime_identity, seed_training, sha256, source_identity
from ureter_semis.inference import SegmentationPredictor
from ureter_semis.labels import read_class_mask
from ureter_semis.metrics import confusion_matrix, metrics_from_confusion
from ureter_semis.model import ensemble_logits
from .refinement_roi import propose_rois, roi_union, coverage_counts


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@torch.inference_mode()
def native_probabilities(predictor, images):
    tensor = torch.stack([_tensor(image.resize((predictor.size,predictor.size),Image.Resampling.BILINEAR)) for image in images]).to(predictor.device)
    probability = ensemble_logits(predictor.models,tensor).softmax(1)
    return [F.interpolate(probability[i:i+1].float(),size=(image.height,image.width),mode="bilinear",align_corners=False)[0].cpu().numpy() for i,image in enumerate(images)]


def native_probability(predictor, image):
    return native_probabilities(predictor,[image])[0]


def validation_predictions(predictor, root, rows):
    # Match the frozen reference's batch2 numerical path exactly.
    for start in range(0,len(rows),2):
        group=rows[start:start+2]
        images=[Image.open(root/row["image"]).convert("RGB") for row in group]
        for offset,(row,image,probability) in enumerate(zip(group,images,native_probabilities(predictor,images))):
            yield start+offset,row,image,probability


def aggregate(rows):
    result = {"frames":len(rows), "roi_count_mean":float(np.mean([r["roi_count"] for r in rows])),
        "roi_count_p95":float(np.percentile([r["roi_count"] for r in rows],95)),
        "roi_count_max":max(r["roi_count"] for r in rows),
        "frames_without_roi":sum(r["roi_count"]==0 for r in rows),
        "roi_area_sum_fraction":sum(r["roi_area_sum"] for r in rows)/sum(r["image_pixels"] for r in rows),
        "roi_union_fraction":sum(r["roi_area_union"] for r in rows)/sum(r["image_pixels"] for r in rows),
        "model_input_pixel_ratio_to_coarse":1+float(np.mean([r["roi_count"] for r in rows])),
        "background_fraction_of_valid_roi":sum(r["background_pixels_in_roi"] for r in rows)/sum(r["valid_pixels_in_roi"] for r in rows) if sum(r["valid_pixels_in_roi"] for r in rows) else None}
    for name in ("stone","fiber"):
        present=[r[name] for r in rows if r[name]["target_pixels"]]
        totals={k:sum(r[name][k] for r in rows) for k in rows[0][name]}
        coverages=[r["covered_pixels"]/r["target_pixels"] for r in present]
        totals.update({"pixel_coverage":totals["covered_pixels"]/totals["target_pixels"] if totals["target_pixels"] else None,
            "present_frames":len(present), "minimum_frame_coverage":min(coverages) if present else None,
            "frames_below95pct_coverage":sum(x<.95 for x in coverages),
            "frames_zero_coverage":sum(x==0 for x in coverages),
            "coarse_complete_miss_frames":sum(r["coarse_tp"]==0 for r in present),
            "coarse_complete_miss_frames_any_roi_coverage":sum(r["coarse_tp"]==0 and r["covered_pixels"]>0 for r in present),
            "coarse_fn_coverage":totals["coarse_fn_covered"]/totals["coarse_fn"] if totals["coarse_fn"] else None,
            "coarse_fp_coverage":totals["coarse_fp_covered"]/totals["coarse_fp"] if totals["coarse_fp"] else None})
        result[name]=totals
    return result


def benchmark(config_path, profile_name):
    config=read(config_path);out=ROOT/config["output"]
    profile=next(p for p in config["profiles"] if p["name"]==profile_name)
    seed_training(42)
    root,manifest=load_manifest(ROOT/config["manifest"])
    # No labels read by the cost subprocess.
    images=[Image.open(root/r["image"]).convert("RGB") for r in manifest["val"]]
    predictor=SegmentationPredictor(ROOT/config["checkpoint"])
    side=config["proposal"]["refiner_input_side"]
    @torch.inference_mode()
    def infer(image):
        predictor._sync();start=time.perf_counter()
        probability=native_probability(predictor,image)
        predictor._sync();coarse=time.perf_counter()
        boxes=propose_rois(probability,config["proposal"],profile["seed"],profile["max_rois"])
        proposed=time.perf_counter()
        for box in boxes:
            crop=image.crop(box)
            tensor=_tensor(crop.resize((side,side),Image.Resampling.BILINEAR)).unsqueeze(0).to(predictor.device)
            logits=ensemble_logits(predictor.models,tensor)
            output=F.interpolate(logits.softmax(1).float(),size=(crop.height,crop.width),mode="bilinear",align_corners=False)[0]
            array=output.cpu().numpy();mask=output.argmax(0).byte().cpu().numpy()
        predictor._sync();finish=time.perf_counter()
        return {"pipeline_proxy_ms":(finish-start)*1000,"coarse_ms":(coarse-start)*1000,
            "proposal_ms":(proposed-coarse)*1000,"crop_calls_ms":(finish-proposed)*1000,"roi_count":len(boxes)}
    warmup=config["benchmark"]["warmup_frames"]
    for i in range(warmup):infer(images[i%len(images)])
    torch.cuda.reset_peak_memory_stats()
    samples=[infer(image) for _ in range(config["benchmark"]["repeats"]) for image in images]
    result={"profile":profile_name,"samples":len(samples),"warmup_frames":warmup,
        "checkpoint_sha256":sha256(ROOT/config["checkpoint"]),"scope":config["benchmark"],
        "cuda_peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
        "timings":{k:{"mean":float(np.mean([r[k] for r in samples])),"p50":float(np.percentile([r[k] for r in samples],50)),
            "p95":float(np.percentile([r[k] for r in samples],95))} for k in samples[0] if k!="roi_count"},"raw_samples":samples}
    atomic_json(out/f"performance_{profile_name}.json",result)


def run(config_path):
    config_path=Path(config_path).resolve();config=read(config_path)
    out=ROOT/config["output"]
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);(out/"probabilities").mkdir()
    atomic_json(out/"job.json",{"phase":"preflight","pid":os.getpid(),"training_updates":0,"training_video_frames":0})
    root,manifest=load_manifest(ROOT/config["manifest"])
    val_only={"train":{"labeled":[],"unlabeled":[]},"val":manifest["val"]}
    identities=input_identity(ROOT/config["manifest"],root,val_only)
    sources=source_identity()
    sources.update({str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__).resolve(),ROOT/"tools/refinement_roi.py",ROOT/"tests/test_refinement_roi.py"]})
    preflight={"config":config,"config_sha256":sha256(config_path),"inputs":identities,"source_hashes":sources,
        "checkpoint_sha256":sha256(ROOT/config["checkpoint"]),"native_reference_sha256":sha256(ROOT/config["native_reference"]),
        "frozen_before_predictions_and_coverage":True,"training_updates":0,"training_images_read":0,"reserved_images_read":0}
    atomic_json(out/"preflight.json",preflight)
    seed_training(42);predictor=SegmentationPredictor(ROOT/config["checkpoint"])
    preflight["runtime"]=runtime_identity(predictor.device);atomic_json(out/"preflight.json",preflight)
    baseline=read(ROOT/config["native_reference"])["models"]["unet256"]
    native_gates=read(ROOT/"runs/a1_sup_kidney_resolution_20261028/preflight.json")["native_gates"]
    matrices={p["name"]:torch.zeros((3,3),dtype=torch.long) for p in config["profiles"]}
    matrix=torch.zeros((3,3),dtype=torch.long)
    rows={p["name"]:[] for p in config["profiles"]}
    for index,row,image,probability in validation_predictions(predictor,root,manifest["val"]):
        prediction=probability.argmax(0).astype(np.uint8)
        expected=np.asarray(Image.open(Path(config["native_reference"]).parent/"unet256"/f"{index:04d}.png"))
        if not np.array_equal(prediction,expected):raise ValueError("Coarse native masks differ from frozen reference")
        # Geometry is fully proposed before reading the corresponding GT.
        proposals={p["name"]:propose_rois(probability,config["proposal"],p["seed"],p["max_rois"]) for p in config["profiles"]}
        np.savez_compressed(out/"probabilities"/f"{index:04d}.npz",probabilities=probability)
        target=read_class_mask(root/row["mask"])
        matrix+=confusion_matrix(torch.from_numpy(prediction),torch.from_numpy(target))
        for profile in config["profiles"]:
            name=profile["name"];boxes=proposals[name]
            counts=coverage_counts(prediction,target,boxes)
            union=roi_union(boxes,target.shape)
            perfect=prediction.copy();valid=target!=255;perfect[union & valid]=target[union & valid]
            matrices[name]+=confusion_matrix(torch.from_numpy(perfect),torch.from_numpy(target))
            rows[name].append({"sample_id":row.get("sample_id",str(index)),"video":row.get("video_id",Path(row["image"]).parts[-3]),
                "image":row["image"],"mask":row["mask"],"boxes":[list(b) for b in boxes],**counts})
    assert matrix.tolist()==baseline["metrics"]["confusion_matrix"]
    del predictor
    torch.cuda.empty_cache()
    profiles={}
    rule=config["evaluation"]
    for profile in config["profiles"]:
        name=profile["name"];data=rows[name];totals=aggregate(data)
        checks={f"{c}_pooled_coverage":totals[c]["pixel_coverage"] is not None and totals[c]["pixel_coverage"]>=rule["pixel_coverage_min_per_class"] for c in ("stone","fiber")}
        checks.update({f"{c}_minimum_frame_coverage":totals[c]["minimum_frame_coverage"] is not None and totals[c]["minimum_frame_coverage"]>=rule["minimum_present_frame_coverage_per_class"] for c in ("stone","fiber")})
        checks.update({"p95_roi_count":totals["roi_count_p95"]<=rule["p95_roi_count_max"],
            "mean_compute_pixels":totals["model_input_pixel_ratio_to_coarse"]<rule["mean_model_input_pixels_ratio_to_coarse_max_exclusive"]})
        oracle=metrics_from_confusion(matrices[name])
        oracle_pass=oracle["stone_dice"]>=native_gates["stone_dice_min"] and oracle["fiber_dice"]>=native_gates["fiber_dice_min"]
        bins=rule["area_bins_native_stone_pixels"]
        groups={}
        for lower,upper in zip(bins,(*bins[1:],None)):
            subset=[r for r in data if r["stone"]["target_pixels"]>lower and (upper is None or r["stone"]["target_pixels"]<=upper)]
            groups[f"({lower},{upper if upper is not None else 'inf'}]"]=aggregate(subset) if subset else None
        profiles[name]={"configuration":profile,"aggregate":totals,"feasibility_checks":checks,"feasibility_passed":all(checks.values()),
            "perfect_inside_automatic_roi_upper_bound_metrics":oracle,"upper_bound_stone_fiber_overlap_gates_passed":oracle_pass,
            "by_video":{video:aggregate([r for r in data if r["video"]==video]) for video in sorted({r["video"] for r in data})},
            "stone_visible_area_groups":groups,"frames":data}
    atomic_json(out/"coverage.json",{"config":config,"profiles":profiles,"coarse_metrics":baseline["metrics"]})
    for profile in config["profiles"]:
        atomic_json(out/"job.json",{"phase":"benchmarking","profile":profile["name"],"pid":os.getpid(),"training_updates":0})
        with (out/f"benchmark_{profile['name']}.log").open("w",encoding="utf-8") as log:
            subprocess.run([sys.executable,"-m","tools.diagnose_refinement_roi","--config",str(config_path),"--benchmark-profile",profile["name"]],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,
                creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        profiles[profile["name"]]["performance"]=read(out/f"performance_{profile['name']}.json")
    assert all(sha256(ROOT/p)==digest for p,digest in sources.items())
    assert sha256(config_path)==preflight["config_sha256"] and sha256(ROOT/config["checkpoint"])==preflight["checkpoint_sha256"]
    assert input_identity(ROOT/config["manifest"],root,val_only)==identities
    result={"config":config,"profiles":profiles,"coarse_metrics":baseline["metrics"],"training_updates":0,
        "feasibility_passed":any(p["feasibility_passed"] for p in profiles.values()),
        "all_identities_unchanged":True,"diagnostic_only":True,"refiner_trained":False}
    atomic_json(out/"summary.json",result)
    atomic_json(out/"job.json",{"phase":"completed","pid":os.getpid(),"training_updates":0,"training_video_frames":0,
        "training_images_read":0,"reserved_images_read":0,"validation_frames":len(manifest["val"]),
        "feasibility_passed":result["feasibility_passed"],"refiner_trained":False})
    print(json.dumps({"phase":"completed","feasibility_passed":result["feasibility_passed"]}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/diagnose_refinement_roi_20261028.json")
    parser.add_argument("--benchmark-profile")
    args=parser.parse_args()
    if args.benchmark_profile:benchmark(args.config,args.benchmark_profile)
    else:run(args.config)
