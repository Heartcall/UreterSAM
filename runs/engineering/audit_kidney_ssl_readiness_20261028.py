"""Verify read-only reserve diagnosis and its frozen no-go decision."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ureter_semis.experiment import atomic_json, sha256, source_identity
from ureter_semis.source_policy import require_training_sources


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    output = ROOT / "runs/a1_sup_kidney_ssl_diagnosis_20261028"
    preflight, result, job = (read(output / f"{name}.json") for name in ("preflight", "diagnosis", "job"))
    manifest_path = ROOT / "datasets/ureter_semis_a1_sup_kidney_only_20261027/manifest.json"
    manifest = read(manifest_path); require_training_sources(manifest)
    active, reserve_files = preflight["active_inputs"]["files"], preflight["reserve_image_files"]
    labeled_hashes = {r["image_sha256"] for r in manifest["train"]["labeled"]}
    val_hashes = {r["image_sha256"] for r in manifest["val"]}
    pool_hashes = set(reserve_files.values())
    registry = read("annotations/excluded_training_sources_20261027.json")
    with zipfile.ZipFile(output / "source_snapshot.zip") as archive:
        snapshot_matches = all(hashlib.sha256(archive.read(name)).hexdigest() == sha256(ROOT / name) for name in archive.namelist())
    records = result["frames"]
    train, pool = result["summaries"]["train"], result["summaries"]["reserve"]
    primary = train["consensus_strong"]["classes"]
    strict_counts_match = True
    from PIL import Image
    import numpy as np
    for record in records:
        mask = np.asarray(Image.open(output / "diagnostic_masks" / f"{record['index']:04d}_strict.png"))
        strict_counts_match &= [int((mask == c).sum()) for c in range(3)] == record["policies"]["consensus_strong"]["accepted"]
        strict_counts_match &= set(np.unique(mask)).issubset({0, 1, 2, 255})
    checks = {
        "completed190_train800_reserve_no_updates": job["phase"] == "completed" and len(records) == 990 and job["parameter_updates"] == result["parameter_updates"] == 0,
        "all1238_input_files_unchanged": len(active) == 438 and len(reserve_files) == 800 and all(sha256(Path(p)) == digest for p, digest in {**active, **reserve_files}.items()),
        "source_config_and_teacher_bytes_unchanged": snapshot_matches and sha256(ROOT / "configs/diagnose_kidney_ssl_readiness_20261028.json") == preflight["config_sha256"] and source_identity() == preflight["source_hashes"] and all(sha256(Path(p)) == digest for p, digest in preflight["teacher_sha256"].items()),
        "reserve_actual_bytes_disjoint_from_labeled_and_validation": not (pool_hashes & (labeled_hashes | val_hashes)),
        "stone_new_actual_hashes_excluded": not (pool_hashes & set(registry["excluded_image_sha256"])),
        "active_training_manifest_and_pool_not_promoted": sha256(manifest_path) == preflight["active_inputs"]["manifest_sha256"] and not manifest["train"]["unlabeled"] and len(manifest["train"]["labeled"]) == 190 and len(manifest["reserved_unlabeled"]) == 800,
        "no_validation_pseudo_records": {r["sample_id"] for r in records} == {r["sample_id"] for r in manifest["train"]["labeled"] + manifest["reserved_unlabeled"]} and result["validation_predictions_used_as_pseudo_labels"] == 0,
        "reserve_label_statistics_undefined": result["reserve_masks_read"] == 0 and all(sum(r["source_label_pixels"]) == 0 for r in records if r["split"] == "reserve") and all(c["source_label_agreement_precision"] is None and c["source_label_positive_recall"] is None for p in pool.values() for c in p["classes"].values()),
        "histograms_cover_all_pixels": all(sum(c["count"] for c in teachers.values()) == result["summaries"][split]["anchor"]["pixels"] and all(sum(c["confidence_hist"]) == sum(c["entropy_hist"]) == c["count"] for c in teachers.values()) for split, groups in result["histograms"].items() for teachers in groups.values()),
        "strict_mask_files_reproduce_counts": strict_counts_match,
        "consensus_filters_are_nested": all(r["policies"]["consensus_strong"]["accepted"][c] <= r["policies"]["consensus"]["accepted"][c] <= r["policies"]["peers"]["accepted"][c] and r["policies"]["consensus"]["accepted"][c] <= r["policies"]["anchor"]["accepted"][c] for r in records for c in range(3)),
        "frozen_primary_gate_failed_only_fiber_precision": result["readiness"]["passed"] is False and [k for k, v in result["readiness"]["checks"].items() if not v] == ["fiber_provisional_train_precision"] and primary["2"]["source_label_agreement_precision"] < preflight["config"]["readiness_gates"]["each_foreground_provisional_train_precision_min"],
        "paired_protocol_deferred_without_training_launch": read("configs/train_a1_ssl_kidney_pilot_20261028.json")["status"] == "DEFERRED_READINESS_FAILED" and not read("configs/train_a1_ssl_kidney_pilot_20261028.json")["executable"] and job["reserve_images_used_in_training"] == 0,
        "source_review_status_retained": manifest["training_ready"] is False and manifest["role"] == "unreviewed_development_experiment",
        "a0_still_unchanged": sha256(ROOT / "runs/endosemis/best.pt") == "e4df36dde654ed5f5aa4e93125b256f22312c66bc8945a7fdcfa9bf2b8bf27f7"
    }
    audit = {"checks": checks, "all_passed": all(checks.values()), "active_files": len(active),
             "reserve_image_files": len(reserve_files), "reserve_unique_sha256": len(pool_hashes),
             "scope": "Read-only experiment verification, not calibration or clinical validity"}
    atomic_json(ROOT / "runs/engineering/kidney_ssl_readiness_checks_20261028.json", audit)
    print(json.dumps(audit, indent=2)); assert audit["all_passed"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["anchor", "peers", "consensus", "consensus_strong"]
    labels = ["Anchor", "Two peers", "Three models", "+ strong view"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    x = np.arange(4)
    for c, label, color in ((1, "Stone", "tab:red"), (2, "Fiber", "tab:cyan")):
        axes[0].plot(x, [train[n]["classes"][str(c)]["source_label_agreement_precision"] for n in names], marker="o", label=label, color=color)
        axes[1].plot(x, [pool[n]["classes"][str(c)]["accepted_pixels"] / 65536 for n in names], marker="o", label=label, color=color)
    axes[0].axhline(.95, color="gray", linestyle="--", label="Frozen precision gate")
    axes[0].set(ylabel="Agreement with provisional TRAIN labels", ylim=(.75, 1.01))
    axes[1].set(ylabel="Accepted reserve area / image area (not images)", yscale="log")
    bottom = np.zeros(4)
    for c, label, color in ((0, "Background", "lightgray"), (1, "Stone", "tab:red"), (2, "Fiber", "tab:cyan")):
        values = np.array([pool[n]["classes"][str(c)]["accepted_pixels"] / pool[n]["pixels"] for n in names])
        axes[2].bar(x, values, bottom=bottom, label=label, color=color); bottom += values
    axes[2].set(ylabel="Reserve pixel acceptance fraction", ylim=(0, 1))
    for ax in axes:
        ax.set_xticks(x, labels, rotation=20); ax.legend(fontsize=7); ax.grid(alpha=.15)
    fig.suptitle("Kidney190 labeled diagnosis +800 image-only reserve; no parameter updates")
    fig.savefig(output / "readiness_comparison.png", dpi=160); plt.close(fig)
