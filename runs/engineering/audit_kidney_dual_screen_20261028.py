"""Check scope, exact replay, member independence and matched-update budgets."""
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
    run = ROOT / "runs/a1_sup_kidney_dual_20261028"
    preflight, summary, job = (read(run / f"{name}.json") for name in ("preflight", "summary", "job"))
    histories = {a["name"]: read(run / a["name"] / "history.json") for a in preflight["config"]["arms"]}
    identities = {name: read(run / name / "run_config.json") for name in histories}
    old_run = ROOT / "runs/a1_sup_kidney_only_20261027"
    old_history = read(old_run / "training/history.json")
    comparison = read(run / "comparison.json")
    old_comparison = read(old_run / "comparison.json")
    manifest_path = ROOT / "datasets/ureter_semis_a1_sup_kidney_only_20261027/manifest.json"
    manifest = read(manifest_path)
    require_training_sources(manifest)
    cost = summary["training_cost"]
    with zipfile.ZipFile(run / "source_snapshot.zip") as archive:
        snapshot_matches = all(hashlib.sha256(archive.read(name)).hexdigest() == sha256(ROOT / name)
                               for name in archive.namelist())
    configs_match = True
    for arm in preflight["config"]["arms"]:
        expected = dict(identities["single15_seed42"]["config"], seed=arm["seed"], epochs=arm["epochs"])
        configs_match &= identities[arm["name"]]["config"] == expected
    first = torch.load(run / "single15_seed42/last.pt", map_location="cpu", weights_only=True)
    second = torch.load(run / "single15_seed123/last.pt", map_location="cpu", weights_only=True)
    old = torch.load(old_run / "training/last.pt", map_location="cpu", weights_only=True)
    replay_exact = all(torch.equal(first["model1"][k], old["model1"][k]) for k in first["model1"])
    independent_states = (not torch.equal(first["training_state"]["loader_generators"]["labeled"],
                                         second["training_state"]["loader_generators"]["labeled"]) and
                          any(not torch.equal(first["model1"][k], second["model1"][k]) for k in first["model1"]))
    ensemble = torch.load(run / "dual_final.pt", map_location="cpu", weights_only=True)
    exact_members = all(torch.equal(ensemble["model1"][k], first["model1"][k]) and
                        torch.equal(ensemble["model2"][k], second["model1"][k]) for k in first["model1"])
    model_rows = {row["name"]: row for row in summary["models"]}
    by_video = read(run / "per_video.json")
    per_video_confusion_matches = all(
        sum(by_video[name][v]["confusion_matrix"][i][j] for v in ("video_1", "video_2")) == row["metrics"]["confusion_matrix"][i][j]
        for name, row in model_rows.items() for i in range(3) for j in range(3))
    files = preflight["inputs"]["files"]
    reserved = {str((Path(manifest["root"]) / r["image"]).resolve()) for r in manifest["reserved_unlabeled"]}
    checks = {
        "completed_all_three_frozen_arms": job["phase"] == "completed" and len(job["completed_arms"]) == 3 and all(len(histories[a["name"]]) == a["epochs"] for a in preflight["config"]["arms"]),
        "only_predeclared_seed_and_epoch_differences": configs_match,
        "identical_training_data_code_and_runtime": all(r["inputs"] == preflight["inputs"] and r["source_hashes"] == preflight["source_hashes"] == source_identity() and r["runtime"] == identities["single15_seed42"]["runtime"] for r in identities.values()),
        "config_sources_and_438_inputs_frozen": snapshot_matches and len(files) == 438 and all(sha256(Path(p)) == digest for p, digest in files.items()) and sha256(manifest_path) == preflight["inputs"]["manifest_sha256"],
        "kidney190_reserved800_video_training0": len(manifest["train"]["labeled"]) == 190 and len(reserved) == 800 and not (reserved & set(files)) and {r["source_dataset"] for r in manifest["train"]["labeled"] + manifest["reserved_unlabeled"]} == {"kidney_stones_fiber_v1"} and job["training_video_frames"] == 0,
        "no_active_unlabeled_pseudo_or_mutual_learning": not manifest["train"]["unlabeled"] and all(r["config"]["unlabeled_images_used"] == 0 and not r["config"]["pseudo_labels"] and not r["config"]["mutual_learning"] for r in identities.values()),
        "all_epochs_uniform190_images65_fiber95_updates": all(r["sampled_training_images"] == 190 and r["sampled_fiber_images"] == 65 and r["optimizer_update_steps"] == 95 for h in histories.values() for r in h),
        "dual_and_single30_equal2850_updates5700_exposures": cost["single15_seed42"]["optimizer_updates"] + cost["single15_seed123"]["optimizer_updates"] == cost["single30_seed42"]["optimizer_updates"] == 2850 and cost["single15_seed42"]["training_image_exposures"] + cost["single15_seed123"]["training_image_exposures"] == cost["single30_seed42"]["training_image_exposures"] == 5700,
        "seed42_reproduces_historical_weights_and_all_epochs": replay_exact and histories["single15_seed42"] == old_history,
        "members_have_distinct_weights_and_sampling_rng": independent_states,
        "artifact_contains_exact_members_and_no_optimizer": ensemble["model_count"] == 2 and exact_members and "training_state" not in ensemble and [m["seed"] for m in ensemble["ensemble_members"]] == [42, 123],
        "a0_and_validation_exactly_reproduced": comparison["checkpoints"][0] == old_comparison["checkpoints"][0] and comparison["inputs"] == old_comparison["inputs"],
        "single_final_scores_match_frozen_history": all(model_rows[name]["metrics"] == histories[name][-1]["metrics"] for name in histories),
        "per_video_confusions_sum_to_global": per_video_confusion_matches,
        "two_network_parameter_cost_exactly_doubles": model_rows["dual15"]["performance"]["parameter_count"] == 2 * model_rows["single15_seed42"]["performance"]["parameter_count"],
        "all_benchmarks_fixed87_samples_batch1": all(r["performance"]["samples"] == 87 and r["performance"]["warmup_frames"] == 12 for r in model_rows.values()),
        "review_status_not_promoted": manifest["training_ready"] is False and manifest["role"] == "unreviewed_development_experiment",
    }
    result = {"checks": checks, "all_passed": all(checks.values()),
              "scope": "Experiment integrity and budget checks; no clinical or independent-case validity claim"}
    atomic_json(ROOT / "runs/engineering/kidney_dual_screen_checks_20261028.json", result)
    print(json.dumps(result, indent=2))
    assert result["all_passed"], "Dual experiment audit failed"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for name, color in (("single15_seed42", "tab:blue"), ("single15_seed123", "tab:orange"), ("single30_seed42", "tab:green")):
        for ax, metric in zip(axes[:2], ("stone_dice", "fiber_dice")):
            ax.plot([r["epoch"] for r in histories[name]], [r["metrics"][metric] for r in histories[name]], color=color, label=name)
    for ax, metric, label in zip(axes[:2], ("stone_dice", "fiber_dice"), ("Stone Dice", "Fiber Dice")):
        ax.axhline(model_rows["dual15"]["metrics"][metric], color="purple", linestyle="--", label="Dual final (2x15)")
        ax.axhline(model_rows["A0"]["metrics"][metric], color="gray", linestyle=":", label="A0")
        ax.set(xlabel="Epoch", ylabel=label, ylim=(0, 1)); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    names = ["single15_seed42", "single15_seed123", "single30_seed42", "dual15", "A0"]
    positions = np.arange(len(names))
    for offset, quantile in ((-.18, "p50"), (.18, "p95")):
        values = [model_rows[name]["performance"]["timings"]["pipeline_ms"][quantile] for name in names]
        bars = axes[2].bar(positions + offset, values, width=.36, label=quantile)
        axes[2].bar_label(bars, fmt="%.1f", fontsize=7)
    axes[2].set_xticks(positions, ["1x15 s42", "1x15 s123", "1x30 s42", "2x15", "A0"], rotation=25)
    axes[2].set(ylabel="Cached frame to masks / probabilities (ms)")
    axes[2].legend(fontsize=8)
    fig.suptitle("Kidney-only: fixed final checkpoints; 2x15 and 1x30 have equal total updates")
    fig.savefig(run / "dual_budget_comparison.png", dpi=160)
    plt.close(fig)
