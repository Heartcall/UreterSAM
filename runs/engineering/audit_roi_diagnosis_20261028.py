"""Independent saved-proposal/native-mask accounting and figures."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from ureter_semis.data import load_manifest
from ureter_semis.experiment import atomic_json,input_identity,sha256,source_identity
from ureter_semis.labels import read_class_mask
from ureter_semis.metrics import confusion_matrix,metrics_from_confusion
from tools.refinement_roi import propose_rois,roi_union


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__=="__main__":
    cfg_path=ROOT/"configs/diagnose_refinement_roi_20261028.json";cfg=read(cfg_path)
    output=ROOT/cfg["output"];summary=read(output/"summary.json");preflight=read(output/"preflight.json")
    root,manifest=load_manifest(ROOT/cfg["manifest"])
    baseline=read(ROOT/cfg["native_reference"])["models"]["unet256"]
    costs=read(output/"matched_costs/comparison.json");cost_preflight=read(output/"matched_costs/preflight.json")
    geometry=counts_match=coarse_match=oracle_match=budget_match=cost_counts_match=True
    cached=[]
    for i,row in enumerate(manifest["val"]):
        probability=np.load(output/"probabilities"/f"{i:04d}.npz")["probabilities"]
        target=read_class_mask(root/row["mask"]);pred=probability.argmax(0)
        coarse_match &= np.array_equal(pred,np.asarray(Image.open(Path(cfg["native_reference"]).parent/"unet256"/f"{i:04d}.png")))
        cached.append((probability,pred,target))
    for name,result in summary["profiles"].items():
        profile=result["configuration"];matrix=torch.zeros((3,3),dtype=torch.long)
        sum_target={1:0,2:0};sum_covered={1:0,2:0}
        for row,(probability,pred,target) in zip(result["frames"],cached):
            expected=propose_rois(probability,cfg["proposal"],profile["seed"],profile["max_rois"])
            geometry &= row["boxes"]==[list(b) for b in expected] and all(b[2]-b[0]==192 and b[3]-b[1]==192 for b in expected)
            union=np.zeros_like(target,dtype=bool)
            for x0,y0,x1,y1 in row["boxes"]:
                geometry &= 0<=x0<x1<=target.shape[1] and 0<=y0<y1<=target.shape[0]
                union[y0:y1,x0:x1]=True
            counts_match &= row["roi_area_union"]==int(union.sum())
            for cls,label in ((1,"stone"),(2,"fiber")):
                truth=target==cls;valid=target!=255
                expected_counts={"target_pixels":int(truth.sum()),"covered_pixels":int((truth&union).sum()),
                    "coarse_tp":int(((pred==cls)&truth).sum()),"coarse_fn":int(((pred!=cls)&truth).sum()),
                    "coarse_fn_covered":int(((pred!=cls)&truth&union).sum()),
                    "coarse_fp":int(((pred==cls)&~truth&valid).sum()),
                    "coarse_fp_covered":int(((pred==cls)&~truth&valid&union).sum())}
                counts_match &= row[label]==expected_counts
                sum_target[cls]+=expected_counts["target_pixels"];sum_covered[cls]+=expected_counts["covered_pixels"]
            perfect=pred.copy();valid=target!=255;perfect[union&valid]=target[union&valid]
            matrix+=confusion_matrix(torch.from_numpy(perfect),torch.from_numpy(target))
        for cls,label in ((1,"stone"),(2,"fiber")):
            counts_match &= result["aggregate"][label]["pixel_coverage"]==sum_covered[cls]/sum_target[cls]
        oracle_match &= metrics_from_confusion(matrix)==result["perfect_inside_automatic_roi_upper_bound_metrics"]
        budget_match &= all(profile["max_rois"] is None or len(r["boxes"])<=profile["max_rois"] for r in result["frames"])
        budget_match &= result["aggregate"]["model_input_pixel_ratio_to_coarse"]==1+sum(r["roi_count"] for r in result["frames"])/len(result["frames"])
        cost_counts_match &= [r["roi_count"] for r in costs[name]["raw_samples"]]==[r["roi_count"] for r in result["frames"]]*3
    val_only={"train":{"labeled":[],"unlabeled":[]},"val":manifest["val"]}
    checks={
        "six_predeclared_profiles_and_thresholds":cfg==preflight["config"]==summary["config"] and sha256(cfg_path)==preflight["config_sha256"] and len(summary["profiles"])==6,
        "all_validation_inputs_unchanged":input_identity(ROOT/cfg["manifest"],root,val_only)==preflight["inputs"] and len(preflight["inputs"]["files"])==58,
        "frozen_sources_unchanged":all(sha256(ROOT/p)==h for p,h in preflight["source_hashes"].items()),
        "checkpoint_and_native_reference_unchanged":sha256(ROOT/cfg["checkpoint"])==preflight["checkpoint_sha256"] and sha256(ROOT/cfg["native_reference"])==preflight["native_reference_sha256"],
        "coarse_masks_exactly_match_native_reference":bool(coarse_match),
        "ground_truth_free_deterministic_roi_geometry":bool(geometry),
        "native_per_frame_and_pooled_coverage_recomputed":bool(counts_match),
        "perfect_inside_roi_oracle_only_recomputed":bool(oracle_match),
        "caps_and_pixel_compute_budget_recomputed":bool(budget_match),
        "matched_fresh_process_cost_count_matches_coverage":bool(cost_counts_match),
        "cost_sources_unchanged":all(sha256(ROOT/p)==h for p,h in cost_preflight["sources"].items()),
        "matched8_cost_variants12warmup87samples":len(costs)==8 and all(r["samples"]==87 and r["warmup_frames"]==12 for r in costs.values()),
        "no_training_or_reserve_reads_or_updates":preflight["training_images_read"]==preflight["reserved_images_read"]==summary["training_updates"]==0 and not summary["refiner_trained"],
        "no_failed_candidate_promoted":not summary["feasibility_passed"] and all(not p["feasibility_passed"] for p in summary["profiles"].values()),
        "training_labels_remain_unreviewed":manifest["training_ready"] is False and manifest["role"]=="unreviewed_development_experiment",
        "a0_preserved":sha256(ROOT/"runs/endosemis/best.pt")=="e4df36dde654ed5f5aa4e93125b256f22312c66bc8945a7fdcfa9bf2b8bf27f7"
    }
    audit={"checks":checks,"all_passed":all(checks.values()),"tests":read(ROOT/"runs/engineering/roi_tests_20261028.json"),
        "scope":"Validation-only proposal coverage and compute proxy; no refinement accuracy or independent validation"}
    atomic_json(ROOT/"runs/engineering/kidney_roi_checks_20261028.json",audit)
    print(json.dumps(audit,indent=2));assert audit["all_passed"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    names=list(summary["profiles"])
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout="constrained")
    x=np.arange(len(names));width=.36
    axes[0].bar(x-width/2,[summary['profiles'][n]['aggregate']['stone']['pixel_coverage']*100 for n in names],width,label='Stone')
    axes[0].bar(x+width/2,[summary['profiles'][n]['aggregate']['fiber']['pixel_coverage']*100 for n in names],width,label='Fiber')
    axes[0].axhline(99,color='red',ls='--',lw=1,label='Pooled gate99%');axes[0].set_ylim(85,101)
    axes[0].set_ylabel('Native GT pixel coverage (%)');axes[0].legend(fontsize=8)
    axes[1].bar(x,[summary['profiles'][n]['aggregate']['stone']['minimum_frame_coverage']*100 for n in names])
    axes[1].axhline(95,color='red',ls='--',lw=1);axes[1].set_ylabel('Worst present-frame stone coverage (%)');axes[1].set_ylim(0,101)
    axes[2].bar(x,[costs[n]['timings']['total_ms']['p50'] for n in names])
    axes[2].axhline(costs['full256']['timings']['total_ms']['p50'],ls='--',color='green',label='Matched full256')
    axes[2].axhline(costs['full512']['timings']['total_ms']['p50'],ls='--',color='orange',label='Matched full512')
    axes[2].set_ylabel('Sequential compute proxy p50 (ms)');axes[2].legend(fontsize=8)
    for ax in axes:ax.set_xticks(x,names,rotation=50,ha='right');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Automatic ROI feasibility: coverage is not refinement accuracy (29 development frames)')
    fig.savefig(output/'roi_coverage_cost.png',dpi=150);plt.close(fig)

    indices=[11,23,26,28]
    shown=['hard_cap2','hard_cap4','soft_cap2']
    fig,axes=plt.subplots(len(indices),4,figsize=(13,13),layout='constrained')
    for j,index in enumerate(indices):
        row=manifest['val'][index];image=np.asarray(Image.open(root/row['image']).convert('RGB'));_,pred,target=cached[index]
        axes[j,0].imshow(image);axes[j,0].set_title(row.get('sample_id',str(index)),fontsize=9)
        for k,name in enumerate(shown,1):
            result=summary['profiles'][name]['frames'][index];union=roi_union(result['boxes'],target.shape)
            visual=image.astype(float).copy()
            for cls,color in ((1,np.array([50,240,70])),(2,np.array([50,160,255]))):
                truth=target==cls;visual[truth]=.45*visual[truth]+.55*color
                missing=truth&~union;visual[missing]=.2*visual[missing]+.8*np.array([255,40,40])
            ax=axes[j,k];ax.imshow(visual.astype(np.uint8))
            for x0,y0,x1,y1 in result['boxes']:ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color='yellow',lw=1))
            stone=result['stone'];cov=stone['covered_pixels']/stone['target_pixels'] if stone['target_pixels'] else None
            ax.set_title(f"{name}: {result['roi_count']} ROI; stone {cov:.1%}" if cov is not None else f"{name}: {result['roi_count']} ROI; stone absent",fontsize=9)
        for ax in axes[j]:ax.axis('off')
    fig.suptitle('Automatic boxes (yellow); covered GT stone/fiber (green/blue); GT outside ROI (red)\nGT is used only for this coverage display, never for proposal')
    fig.savefig(output/'roi_failure_examples.png',dpi=140);plt.close(fig)
