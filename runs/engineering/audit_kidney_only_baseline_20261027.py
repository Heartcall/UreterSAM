"""Verify the completed fixed-budget kidney-only development experiment."""
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import zipfile
import hashlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ureter_semis.experiment import sha256, source_identity, atomic_json
from ureter_semis.source_policy import require_training_sources


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    run = ROOT / "runs/a1_sup_kidney_only_20261027"
    job = read(run / "job.json")
    status = read(run / "training/status.json")
    history = read(run / "training/history.json")
    identity = read(run / "training/run_config.json")
    comparison = read(run / "comparison.json")
    config_path = ROOT / "configs/train_a1_sup_kidney_only_20261027.json"
    manifest_path = ROOT / "datasets/ureter_semis_a1_sup_kidney_only_20261027/manifest.json"
    manifest = read(manifest_path)
    require_training_sources(manifest)
    old = read("runs/a1_dev_videos_val_20261027/first_epoch_comparison.json")
    rejected_old = False
    try:
        require_training_sources(read("datasets/ureter_semis_a1_sup_bg_20261027/manifest.json"))
    except ValueError:
        rejected_old = True
    sources = {row["source_dataset"] for row in manifest["train"]["labeled"] + manifest["reserved_unlabeled"]}
    with zipfile.ZipFile(run / "source_snapshot.zip") as archive:
        snapshot_matches = all(
            hashlib.sha256(archive.read(name)).hexdigest() == sha256(ROOT / name)
            for name in archive.namelist()
        )
    frozen_files = identity["inputs"]["files"]
    reserved_paths = {str((Path(manifest["root"]) / r["image"]).resolve()) for r in manifest["reserved_unlabeled"]}
    checks = {
        "completed_frozen_15_epochs": status["state"] == "completed" and status["completed_epoch"] == 15 and len(history) == 15,
        "job_comparison_completed": job["phase"] == "completed_short_screen",
        "only_kidney_190_labeled_800_reserved": sources == {"kidney_stones_fiber_v1"} and len(manifest["train"]["labeled"]) == 190 and len(reserved_paths) == 800,
        "unlabeled_empty_and_no_pseudo_labels": not manifest["train"]["unlabeled"] and identity["config"]["unlabeled_images_used"] == 0 and not identity["config"]["pseudo_labels"],
        "reserved_files_absent_from_training_input_identity": not (reserved_paths & set(frozen_files)),
        "single_network_imagenet_fresh_config": identity["config"]["model_count"] == 1 and identity["config"]["pretrained"] and "no mixed-source checkpoint" in read(config_path)["initialization"],
        "all_epochs_190_images_65_fiber_images_95_updates": all(r["sampled_training_images"] == 190 and r["sampled_fiber_images"] == 65 and r["optimizer_update_steps"] == 95 for r in history),
        "source_and_launch_config_unchanged": source_identity() == identity["source_hashes"] and snapshot_matches and sha256(config_path) == job["config_sha256"],
        "all_active_input_bytes_unchanged": len(frozen_files) == 438 and all(sha256(Path(p)) == digest for p, digest in frozen_files.items()) and sha256(manifest_path) == identity["inputs"]["manifest_sha256"],
        "all_29_validation_images_and_masks_frozen": len(manifest["val"]) == 29 and comparison["inputs"]["files"] == old["inputs"]["files"],
        "no_training_video_rows": job["training_video_frames"] == 0 and all(not row.get("video_id") for row in manifest["train"]["labeled"]),
        "a0_weights_and_all_same_grid_metrics_unchanged": comparison["checkpoints"][0]["checkpoint_sha256"] == "e4df36dde654ed5f5aa4e93125b256f22312c66bc8945a7fdcfa9bf2b8bf27f7" and comparison["checkpoints"][0]["metrics"] == old["checkpoints"][0]["metrics"],
        "old_mixed_manifest_still_rejected": rejected_old,
        "review_status_not_promoted": manifest["training_ready"] is False and manifest["role"] == "unreviewed_development_experiment",
        "selected_metrics_match_frozen_epoch": comparison["checkpoints"][1]["metrics"] == history[status["best_epoch"] - 1]["metrics"],
    }
    result = {"checked_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
              "all_passed": all(checks.values()), "input_files_verified": len(frozen_files),
              "optimizer_updates": sum(r["optimizer_update_steps"] for r in history),
              "selected_epoch": status["best_epoch"], "screening_passed": status["best_passed_screening"],
              "final_epoch_metrics": history[-1]["metrics"],
              "interpretation": "Verification of experiment scope and reproducibility; not proof of clinical validity or independent-case performance"}
    atomic_json(ROOT / "runs/engineering/kidney_only_baseline_checks_20261027.json", result)
    print(json.dumps({"checks": checks, "optimizer_updates": result["optimizer_updates"]}, indent=2))
    assert result["all_passed"], "Experiment audit failed"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for ax, key, label in zip(axes, ("stone_dice", "fiber_dice", "stone_negative_fp_pixel_fraction"),
                              ("Stone Dice", "Fiber Dice", "Stone FP area on 9 negative frames")):
        ax.plot(epochs, [r["metrics"][key] for r in history], marker="o", label="Kidney-only")
        ax.axhline(comparison["checkpoints"][0]["metrics"][key], linestyle="--", color="gray", label="A0")
        ax.axvline(status["best_epoch"], linestyle=":", color="black", label="Selected epoch")
        ax.set(xlabel="Epoch", ylabel=label, xlim=(1, 15), ylim=(0, 1))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("190 kidney images; seed 42; PIL 256; 29 development validation frames")
    fig.savefig(run / "learning_curves.png", dpi=160)
    plt.close(fig)
