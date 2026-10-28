"""Compare fixed-budget kidney-only weight4 and weight2 experiments."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ureter_semis.experiment import atomic_json, sha256, source_identity


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    control = ROOT / "runs/a1_sup_kidney_only_20261027"
    candidate = ROOT / "runs/a1_sup_kidney_weight2_20261027"
    preflight = read("runs/a1_sup_kidney_weight2_20261027.preflight.json")
    base_run, new_run = (read(p / "training/run_config.json") for p in (control, candidate))
    config_differences = {k: [base_run["config"].get(k), new_run["config"].get(k)]
                          for k in set(base_run["config"]) | set(new_run["config"])
                          if base_run["config"].get(k) != new_run["config"].get(k)}
    base_comparison, new_comparison = (read(p / "comparison.json") for p in (control, candidate))
    history = read(candidate / "training/history.json")
    job = read(candidate / "job.json")
    status = read(candidate / "training/status.json")
    diagnostic = read("runs/a1_sup_kidney_diagnosis_20261027/diagnosis.json")
    candidate_diagnostic = read("runs/a1_sup_kidney_weight2_diagnosis_20261027/diagnosis.json")
    with zipfile.ZipFile(candidate / "source_snapshot.zip") as archive:
        snapshot_matches = all(hashlib.sha256(archive.read(name)).hexdigest() == sha256(ROOT / name)
                               for name in archive.namelist())
    frozen_files = new_run["inputs"]["files"]
    checks = {
        "only_class_weights_changed": config_differences == {"class_weights": [[1., 2., 4.], [1., 2., 2.]]},
        "preflight_configuration_frozen_before_start": preflight["frozen_before_candidate_start"] and preflight["candidate_config_sha256"] == job["config_sha256"] == sha256(ROOT / "configs/train_a1_sup_kidney_weight2_20261027.json"),
        "identical_inputs_sources_runtime": base_run["inputs"] == new_run["inputs"] == preflight["control_inputs"] and base_run["source_hashes"] == new_run["source_hashes"] == source_identity() and base_run["runtime"] == new_run["runtime"],
        "snapshot_and_438_active_inputs_unchanged": snapshot_matches and len(frozen_files) == 438 and all(sha256(Path(p)) == digest for p, digest in frozen_files.items()),
        "same_15_epochs_1425_updates_uniform_190_images": status["completed_epoch"] == 15 and status["state"] == "completed" and len(history) == 15 and all(r["optimizer_update_steps"] == 95 and r["sampled_training_images"] == 190 and r["sampled_fiber_images"] == 65 for r in history),
        "a0_and_validation_exactly_reproduced": base_comparison["checkpoints"][0] == new_comparison["checkpoints"][0] and base_comparison["inputs"] == new_comparison["inputs"],
        "selected_checkpoint_matches_history": new_comparison["checkpoints"][1]["metrics"] == history[status["best_epoch"] - 1]["metrics"],
        "validation_not_trained_unlabeled_unused": job["training_video_frames"] == 0 and job["unlabeled_images_used"] == 0 and not new_run["config"]["pseudo_labels"],
        "diagnosis_counts_reproduce_all_reference_fiber_metrics": all(
            diagnostic["models"][name]["splits"]["val"]["overall"][key] == metrics[key]
            for name, metrics in (("a0", base_comparison["checkpoints"][0]["metrics"]),
                                  ("selected", base_comparison["checkpoints"][1]["metrics"]),
                                  ("last", read(control / "training/history.json")[-1]["metrics"]))
            for key in ("fiber_dice", "fiber_precision", "fiber_recall")),
        "candidate_diagnosis_reproduces_fiber_metrics": all(
            candidate_diagnostic["models"][name]["splits"]["val"]["overall"][key] == metrics[key]
            for name, metrics in (("weight2_selected", new_comparison["checkpoints"][1]["metrics"]), ("weight2_last", history[-1]["metrics"]))
            for key in ("fiber_dice", "fiber_precision", "fiber_recall")),
        "job_completed_comparison": job["phase"] == "completed_short_screen",
    }
    arms = []
    for name, path in (("weight4_control", control), ("weight2_candidate", candidate)):
        h, s = read(path / "training/history.json"), read(path / "training/status.json")
        arms.append({"name": name, "selected_epoch": s["best_epoch"], "selected_metrics": h[s["best_epoch"] - 1]["metrics"],
                     "final_epoch_metrics": h[-1]["metrics"], "screening_passed_any_epoch": s["best_passed_screening"],
                     "maximum_fiber_dice_diagnostic_only": max(row["metrics"]["fiber_dice"] for row in h)})
    result = {"checks": checks, "all_passed": all(checks.values()), "arms": arms,
              "primary_comparison": "same final epoch15; selected checkpoints reported separately",
              "scope": "Single seed short screen, provisional source labels, repeated development validation; no independent-case generalization claim"}
    atomic_json(candidate / "matched_comparison.json", result)
    atomic_json(ROOT / "runs/engineering/kidney_weight_screen_checks_20261027.json", {"checks": checks, "all_passed": result["all_passed"]})
    print(json.dumps(result, indent=2))
    assert result["all_passed"], "Matched comparison audit failed"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for name, path in (("Weight 4", control), ("Weight 2", candidate)):
        h = read(path / "training/history.json")
        for ax, key, label in zip(axes, ("stone_dice", "fiber_dice", "stone_negative_fp_pixel_fraction"),
                                  ("Stone Dice", "Fiber Dice", "Stone FP fraction (9 negative frames)")):
            ax.plot([r["epoch"] for r in h], [r["metrics"][key] for r in h], marker="o", label=name)
            ax.set(xlabel="Epoch", ylabel=label, xlim=(1, 15), ylim=(0, 1))
            ax.grid(alpha=.2)
    for ax, key in zip(axes, ("stone_dice", "fiber_dice", "stone_negative_fp_pixel_fraction")):
        ax.axhline(base_comparison["checkpoints"][0]["metrics"][key], color="gray", linestyle="--", label="A0")
    axes[0].legend(fontsize=8)
    fig.suptitle("Same 190 kidney images / seed 42 / 15 epochs; only shared fiber class weight changes")
    fig.savefig(candidate / "matched_learning_curves.png", dpi=160)
    plt.close(fig)
