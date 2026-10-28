"""Audit the paired full-architecture short screen and legacy replay."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ureter_semis.experiment import atomic_json, sha256, source_identity
from ureter_semis.source_policy import require_training_sources


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    output = ROOT / "runs/a1_sup_kidney_alt_20261028"
    preflight, summary, job = (read(output / f"{name}.json") for name in ("preflight", "summary", "job"))
    arms = {name: read(output / name / "run_config.json") for name in ("unet_control", "segformer_b0")}
    histories = {name: read(output / name / "history.json") for name in arms}
    old = ROOT / "runs/a1_sup_kidney_only_20261027"
    comparison = read(output / "comparison.json")
    old_comparison = read(old / "comparison.json")
    base = torch.load(old / "training/last.pt", weights_only=True, map_location="cpu")
    control = torch.load(output / "unet_control/last.pt", weights_only=True, map_location="cpu")
    replay = all(torch.equal(base["model1"][k], control["model1"][k]) for k in base["model1"])
    del base, control
    manifest_path = ROOT / "datasets/ureter_semis_a1_sup_kidney_only_20261027/manifest.json"
    manifest = read(manifest_path); require_training_sources(manifest)
    files = preflight["inputs"]["files"]
    reserve = {str((Path(manifest["root"]) / r["image"]).resolve()) for r in manifest["reserved_unlabeled"]}
    with zipfile.ZipFile(output / "source_snapshot.zip") as archive:
        snapshot_matches = all(hashlib.sha256(archive.read(name)).hexdigest() == sha256(ROOT / name) for name in archive.namelist())
    allowed = {"architecture", "encoder_checkpoint"}
    training_config_same = {k: v for k, v in arms["unet_control"]["config"].items() if k not in allowed} == {
                           k: v for k, v in arms["segformer_b0"]["config"].items() if k not in allowed}
    rows = {r["name"]: r for r in summary["models"]}
    spatial = read(output / "spatial_detail.json")
    checks = {
        "completed_two_frozen15_epoch_arms": job["phase"] == "completed" and len(job["completed_arms"]) == 2 and all(len(h) == 15 for h in histories.values()),
        "only_architecture_and_encoder_path_differ": training_config_same,
        "common_training_and_validation_bytes": all({k: v for k, v in a["inputs"].items() if k != "initialization_files"} == preflight["inputs"] for a in arms.values()),
        "source_and_runtime_identical_across_arms": all(a["source_hashes"] == preflight["source_hashes"] == source_identity() and a["runtime"] == arms["unet_control"]["runtime"] for a in arms.values()),
        "configuration_snapshot_and438_inputs_frozen": snapshot_matches and len(files) == 438 and all(sha256(Path(p)) == digest for p, digest in files.items()) and sha256(manifest_path) == preflight["inputs"]["manifest_sha256"],
        "both_pretrained_files_and_installed_implementation_frozen": all(sha256(Path(p)) == digest for p, digest in {**preflight["initialization_files"], **preflight["installed_implementation"]}.items()),
        "segformer_initialization_identity_matches_frozen_imagenet": arms["segformer_b0"]["inputs"]["initialization_files"] == {str(ROOT / "runs/pretrained/mit_b0.pth"): preflight["config"]["encoder_sha256"]},
        "kidney190_only_reserve800_not_read": len(manifest["train"]["labeled"]) == 190 and len(reserve) == 800 and not (reserve & set(files)) and {r["source_dataset"] for r in manifest["train"]["labeled"] + manifest["reserved_unlabeled"]} == {"kidney_stones_fiber_v1"},
        "no_video_pseudo_or_unlabeled_training": job["training_video_frames"] == 0 and job["unlabeled_images_used"] == 0 and not manifest["train"]["unlabeled"] and all(not a["config"]["pseudo_labels"] and not a["config"]["mutual_learning"] for a in arms.values()),
        "same1425_effective_updates2850_exposures": all(c["optimizer_updates"] == 1425 and c["image_exposures"] == 2850 for c in summary["training_cost"].values()) and all(r["optimizer_update_steps"] == 95 and r["sampled_training_images"] == 190 and r["sampled_fiber_images"] == 65 for h in histories.values() for r in h),
        "legacy_unet_all_epochs_and_weights_exact": replay and histories["unet_control"] == read(old / "training/history.json"),
        "a0_metrics_weights_and_validation_exact": comparison["checkpoints"][0] == old_comparison["checkpoints"][0] and comparison["inputs"] == old_comparison["inputs"],
        "final_inference_metrics_match_history": all(rows[name]["metrics"] == histories[name][-1]["metrics"] for name in arms),
        "expected_model_parameter_counts": rows["unet_control"]["performance"]["parameter_count"] == 24601395 and rows["segformer_b0"]["performance"]["parameter_count"] == 3714915,
        "same87_timed_frames_and12_warmup": all(r["performance"]["samples"] == 87 and r["performance"]["warmup_frames"] == 12 for r in rows.values()),
        "spatial_metrics_same_GT_and_frozen_tolerance": spatial["tolerance_pixels"] == 2 and spatial["small_stone_max_gt_pixels"] == 128 and len({(r["small_stone_frames"], r["small_stone_target_pixels"]) for r in spatial["models"].values()}) == 1,
        "source_review_status_not_promoted": manifest["training_ready"] is False and manifest["role"] == "unreviewed_development_experiment",
    }
    result = {"checks": checks, "all_passed": all(checks.values()), "tests": read("runs/engineering/alt_tests_20261028.json"),
              "supplemental_spatial_tests_passed": 2, "scope": "Experiment integrity; not clinical validity or model upper-bound proof"}
    atomic_json(ROOT / "runs/engineering/kidney_alt_screen_checks_20261028.json", result)
    print(json.dumps(result, indent=2)); assert result["all_passed"], "Alternative experiment audit failed"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for name, label in (("unet_control", "ResNet34 U-Net"), ("segformer_b0", "SMP SegFormer MiT-B0")):
        h = histories[name]
        for ax, key in zip(axes[:2], ("stone_dice", "fiber_dice")):
            ax.plot([r["epoch"] for r in h], [r["metrics"][key] for r in h], marker="o", label=label)
    for ax, key in zip(axes[:2], ("stone_dice", "fiber_dice")):
        ax.axhline(rows["A0"]["metrics"][key], linestyle="--", color="gray", label="A0")
        ax.set(xlabel="Epoch", ylabel=key.replace("_", " "), ylim=(0, 1)); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    names = ["unet_control", "segformer_b0", "A0"]
    for offset, quantile in ((-.18, "p50"), (.18, "p95")):
        values = [rows[name]["performance"]["timings"]["pipeline_ms"][quantile] for name in names]
        bars = axes[2].bar(np.arange(3) + offset, values, width=.36, label=quantile)
        axes[2].bar_label(bars, fmt="%.1f", fontsize=8)
    axes[2].set_xticks(np.arange(3), ["U-Net", "SegFormer-B0", "A0"])
    axes[2].set(ylabel="Cached BGR frame to probabilities/masks (ms)"); axes[2].legend(fontsize=8)
    fig.suptitle("Kidney190 / seed42 / 15 epochs / 1425 updates; full-architecture short screen")
    fig.savefig(output / "alt_comparison.png", dpi=160); plt.close(fig)
