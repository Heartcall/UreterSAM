"""Audit native probability comparison, fixed budgets and exact legacy weights."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ureter_semis.experiment import atomic_json, sha256, source_identity
from ureter_semis.labels import read_class_mask
from ureter_semis.source_policy import require_training_sources


def read(path): return json.loads((ROOT / path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    output = ROOT / "runs/a1_sup_kidney_resolution_20261028"
    preflight, summary, job = (read(output / f"{name}.json") for name in ("preflight", "summary", "job"))
    histories = {name: read(output / name / "history.json") for name in ("unet256", "unet512")}
    identities = {name: read(output / name / "run_config.json") for name in histories}
    manifest_path = ROOT / "datasets/ureter_semis_a1_sup_kidney_only_20261027/manifest.json"
    manifest = read(manifest_path); require_training_sources(manifest)
    files = preflight["inputs"]["files"]
    reserved = {str((Path(manifest["root"]) / r["image"]).resolve()) for r in manifest["reserved_unlabeled"]}
    with zipfile.ZipFile(output / "source_snapshot.zip") as archive:
        snapshot_matches = all(hashlib.sha256(archive.read(name)).hexdigest() == sha256(ROOT / name) for name in archive.namelist())
    old_path = ROOT / "runs/a1_sup_kidney_only_20261027"
    old = torch.load(old_path / "training/last.pt", map_location="cpu", weights_only=True)
    current = torch.load(output / "unet256/last.pt", map_location="cpu", weights_only=True)
    exact_weights = all(torch.equal(old["model1"][k], current["model1"][k]) for k in old["model1"])
    del old, current
    legacy, old_legacy = read(output / "legacy256_comparison.json"), read(old_path / "comparison.json")
    native = read(output / "native_comparison/comparison.json")
    baseline = read(output / "baseline_native/comparison.json")["models"]["A0"]
    gates, baseline_metrics = preflight["native_gates"], baseline["metrics"]
    rows = {r["name"]: r for r in summary["models"]}
    mask_checks = True
    for name, model in native["models"].items():
        matrix = np.zeros((3, 3), dtype=np.int64)
        for i, row in enumerate(manifest["val"]):
            target = read_class_mask(Path(manifest["root"]) / row["mask"])
            prediction = np.asarray(Image.open(output / "native_comparison" / name / f"{i:04d}.png"))
            mask_checks &= target.shape == prediction.shape == (400, 400)
            valid = target != 255
            matrix += np.bincount((target[valid].astype(int) * 3 + prediction[valid]), minlength=9).reshape(3, 3)
        mask_checks &= matrix.tolist() == model["metrics"]["confusion_matrix"]
    checks = {
        "completed_two_fixed15_epoch_arms": job["phase"] == "completed" and len(job["completed_arms"]) == 2 and all(len(h) == 15 for h in histories.values()),
        "only_input_size_differs_in_training_config": {k: v for k, v in identities["unet256"]["config"].items() if k != "size"} == {k: v for k, v in identities["unet512"]["config"].items() if k != "size"},
        "native_validation_for_both_arms": all(i["config"]["validation_grid"] == "native_probability" and i["config"]["gates"] == gates for i in identities.values()),
        "identical_data_sources_and_runtime": all(i["inputs"] == preflight["inputs"] and i["source_hashes"] == preflight["source_hashes"] == source_identity() and i["runtime"] == identities["unet256"]["runtime"] for i in identities.values()),
        "source_configuration_encoder438_inputs_frozen": snapshot_matches and len(files) == 438 and all(sha256(Path(p)) == digest for p, digest in {**files, **preflight["pretrained_encoder"]}.items()) and sha256(manifest_path) == preflight["inputs"]["manifest_sha256"],
        "kidney190_reserved800_not_read": len(manifest["train"]["labeled"]) == 190 and len(reserved) == 800 and not (reserved & set(files)) and {r["source_dataset"] for r in manifest["train"]["labeled"] + manifest["reserved_unlabeled"]} == {"kidney_stones_fiber_v1"},
        "no_video_unlabeled_pseudo_training": job["training_video_frames"] == 0 and job["unlabeled_images_used"] == 0 and not manifest["train"]["unlabeled"] and all(not i["config"]["pseudo_labels"] and not i["config"]["mutual_learning"] for i in identities.values()),
        "same1425_updates2850_exposures": all(c["optimizer_updates"] == 1425 and c["image_exposures"] == 2850 for c in summary["training_cost"].values()) and all(r["optimizer_update_steps"] == 95 and r["sampled_training_images"] == 190 and r["sampled_fiber_images"] == 65 for h in histories.values() for r in h),
        "unet256_weights_losses_and_sampling_exactly_replay": exact_weights and all(a["train_loss"] == b["train_loss"] and a["sampled_label_pixels"] == b["sampled_label_pixels"] for a, b in zip(histories["unet256"], read(old_path / "training/history.json"))),
        "legacy_a0_and_unet256_metrics_preserved": legacy["checkpoints"][0] == old_legacy["checkpoints"][0] and legacy["checkpoints"][1]["metrics"] == read(old_path / "training/history.json")[-1]["metrics"] and legacy["inputs"] == old_legacy["inputs"],
        "native_a0_reference_reproduced": native["models"]["A0"]["metrics"] == baseline_metrics == preflight["native_a0_metrics"],
        "native_gates_derived_and_frozen_before_training": preflight["gates_frozen_before_training_model_construction"] and gates == {"stone_dice_min": baseline_metrics["stone_dice"], "fiber_dice_min": baseline_metrics["fiber_dice"]-.01, "negative_fp_fraction_max": baseline_metrics["stone_negative_fp_pixel_fraction"]},
        "native_mask_geometry_and_confusion_match": mask_checks,
        "same_native_gt_pixels_for_all_models": len({r["metrics"]["valid_pixels"] for r in rows.values()}) == 1 and all(len(r["frames"]) == 29 for r in native["models"].values()),
        "final_native_metrics_equal_last_epoch": all(rows[name]["metrics"] == histories[name][-1]["metrics"] for name in histories),
        "fresh_process87_frame12_warmup_cost": all(r["performance"]["samples"] == 87 and r["performance"]["warmup_frames"] == 12 for r in rows.values()),
        "parameter_count_same_for_input_resolution": rows["unet256"]["performance"]["parameter_count"] == rows["unet512"]["performance"]["parameter_count"] == 24601395,
        "same_native_area_group_counts": all(rows["unet256"]["visible_stone_area_groups"][key]["target_pixels"] == rows["unet512"]["visible_stone_area_groups"][key]["target_pixels"] and rows["unet256"]["visible_stone_area_groups"][key]["frames"] == rows["unet512"]["visible_stone_area_groups"][key]["frames"] for key in rows["unet256"]["visible_stone_area_groups"]),
        "source_review_status_not_promoted": manifest["training_ready"] is False and manifest["role"] == "unreviewed_development_experiment"
    }
    result = {"checks": checks, "all_passed": all(checks.values()), "tests": read("runs/engineering/resolution_tests_20261028.json"),
              "scope": "Common-native protocol and matched-update verification, not independent or clinical validation"}
    atomic_json(ROOT / "runs/engineering/kidney_resolution_checks_20261028.json", result)
    print(json.dumps(result, indent=2)); assert result["all_passed"], "Resolution audit failed"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for name, label in (("unet256", "U-Net input256"), ("unet512", "U-Net input512")):
        for ax, key in zip(axes[:2], ("stone_dice", "fiber_dice")):
            ax.plot([r["epoch"] for r in histories[name]], [r["metrics"][key] for r in histories[name]], marker="o", label=label)
    for ax, key in zip(axes[:2], ("stone_dice", "fiber_dice")):
        ax.axhline(baseline_metrics[key], linestyle="--", color="gray", label="A0 native reference")
        ax.set(xlabel="Epoch", ylabel=f"Native {key}", ylim=(0,1)); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    names = ["unet256", "unet512", "A0"]
    for offset, quantile in ((-.18, "p50"), (.18, "p95")):
        bars = axes[2].bar(np.arange(3)+offset, [rows[n]["performance"]["timings"]["pipeline_ms"][quantile] for n in names], width=.36, label=quantile)
        axes[2].bar_label(bars, fmt="%.1f", fontsize=8)
    axes[2].set_xticks(np.arange(3), ["Input256", "Input512", "A0"])
    axes[2].set(ylabel="Cached frame to native maps/mask (ms)"); axes[2].legend(fontsize=8)
    fig.suptitle("Same native400 GT; probability-first resize; kidney190 / seed42 / 15 epochs")
    fig.savefig(output / "native_resolution_comparison.png", dpi=160); plt.close(fig)
