"""Read-only pseudo-label feasibility screen; never promotes diagnostic masks."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import torch
from torch.utils.data import DataLoader, Dataset

from ureter_semis.data import _tensor, load_manifest
from ureter_semis.experiment import atomic_json, input_identity, require_training_access, seed_training, sha256, source_identity
from ureter_semis.inference import SegmentationPredictor
from ureter_semis.labels import read_class_mask
from ureter_semis.model import ensemble_logits


def policies(probabilities, strong_probability, threshold=.95, entropy_max=.2):
    """Teachers [3,B,3,H,W]; strong is anchor only, same geometric coordinates."""
    if probabilities.ndim != 5 or probabilities.shape[0] != 3 or probabilities.shape[2] != 3 or strong_probability.shape != probabilities[0].shape:
        raise ValueError("Expected three aligned three-class teachers and aligned strong anchor")
    confidence, labels = probabilities.max(2)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(2) / math.log(3)
    reliable = (confidence >= threshold) & (entropy <= entropy_max)
    strong_confidence, strong_label = strong_probability.max(1)
    strong_entropy = -(strong_probability * strong_probability.clamp_min(1e-12).log()).sum(1) / math.log(3)
    peer_agreement = (labels[1] == labels[2]) & reliable[1] & reliable[2]
    consensus = peer_agreement & (labels[0] == labels[1]) & reliable[0]
    strong_keep = (strong_label == labels[0]) & (strong_confidence >= threshold) & (strong_entropy <= entropy_max)
    return {"anchor": (labels[0], reliable[0]), "peers": (labels[1], peer_agreement),
            "consensus": (labels[0], consensus), "consensus_strong": (labels[0], consensus & strong_keep)}, labels, confidence, entropy


def strong_view(image, index, seed):
    rng = random.Random(seed + index)
    result = ImageEnhance.Brightness(image).enhance(rng.uniform(.7, 1.3))
    result = ImageEnhance.Contrast(result).enhance(rng.uniform(.7, 1.3))
    result = ImageEnhance.Color(result).enhance(rng.uniform(.75, 1.25))
    if rng.random() < .3:
        result = result.filter(ImageFilter.GaussianBlur(rng.uniform(.1, 1.5)))
    return result


class ImageOnlyDiagnosis(Dataset):
    def __init__(self, root, rows, size, seed):
        self.root, self.rows, self.size, self.seed = root, rows, size, seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        raw = Image.open(self.root / row["image"]).convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        # Reserve labels are intentionally never opened, even if a row has a mask key.
        target = (np.asarray(Image.fromarray(read_class_mask(self.root / row["mask"])).resize(
            (self.size, self.size), Image.Resampling.NEAREST)).copy() if row["diagnostic_split"] == "train" else
            np.full((self.size, self.size), 255, dtype=np.uint8))
        return {"weak": _tensor(raw), "strong": _tensor(strong_view(raw, i, self.seed)),
                "target": torch.from_numpy(target), "index": i}


def aggregate(records, policy):
    pixels = sum(r["pixels"] for r in records)
    accepted = [sum(r["policies"][policy]["accepted"][c] for r in records) for c in range(3)]
    raw = [sum(r["policies"][policy]["raw"][c] for r in records) for c in range(3)]
    matrix = np.sum([r["policies"][policy]["label_agreement_matrix"] for r in records], axis=0)
    ground_truth = [sum(r["source_label_pixels"][c] for r in records) for c in range(3)]
    return {"images": len(records), "pixels": pixels, "accepted_pixels": sum(accepted),
        "accepted_fraction": sum(accepted) / pixels, "background_share_of_accepted": accepted[0] / sum(accepted) if sum(accepted) else None,
        "classes": {str(c): {"accepted_pixels": accepted[c], "raw_candidate_pixels": raw[c],
            "acceptance_within_predicted_class": accepted[c] / raw[c] if raw[c] else None,
            "images_with_at_least16pixels": sum(r["policies"][policy]["accepted"][c] >= 16 for r in records),
            "source_label_agreement_precision": float(matrix[c, c] / matrix[:, c].sum()) if matrix[:, c].sum() else None,
            "source_label_positive_recall": float(matrix[c, c] / ground_truth[c]) if ground_truth[c] else None}
            for c in range(3)}, "label_agreement_matrix": matrix.tolist()}


def readiness(training, reserve, gates):
    checks = {}
    for c, name in ((1, "stone"), (2, "fiber")):
        t = training["consensus_strong"]["classes"][str(c)]
        r = reserve["consensus_strong"]["classes"][str(c)]
        previous = reserve["consensus"]["classes"][str(c)]["accepted_pixels"]
        checks[f"{name}_provisional_train_precision"] = (t["source_label_agreement_precision"] or 0) >= gates["each_foreground_provisional_train_precision_min"]
        checks[f"{name}_provisional_train_recall"] = (t["source_label_positive_recall"] or 0) >= gates["each_foreground_provisional_train_recall_min"]
        checks[f"{name}_reserve_pixels"] = r["accepted_pixels"] >= gates["each_foreground_reserved_accepted_pixels_min"]
        checks[f"{name}_reserve_images"] = r["images_with_at_least16pixels"] >= gates["each_foreground_reserved_images_with16pixels_min"]
        checks[f"{name}_strong_retention"] = bool(previous and r["accepted_pixels"] / previous >= gates["each_foreground_retention_after_strong_min"])
    share = reserve["consensus_strong"]["background_share_of_accepted"]
    checks["background_not_dominating"] = share is not None and share <= gates["accepted_background_fraction_max"]
    return {"passed": all(checks.values()), "checks": checks,
            "interpretation": "Provisional feasibility gates only; agreement and confidence do not establish truth"}


def contact_sheet(root, rows, records, output, train=False):
    selected = sorted([r for r in records if r["split"] == ("train" if train else "reserve")],
                      key=lambda r: r["anchor_fiber_over_source_background"] if train else r["teacher_disagreement_pixels"], reverse=True)[:6]
    canvas = Image.new("RGB", (1024, 28 + 285 * len(selected)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 5), "Raw | Source label (train only) / teacher disagreement (magenta) | Anchor | Strict accepted FG", fill="black")
    for j, record in enumerate(selected):
        i = record["index"]; row = rows[i]
        raw = np.asarray(Image.open(root / row["image"]).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR))
        anchor = np.asarray(Image.open(output / "diagnostic_masks" / f"{i:04d}_anchor.png"))
        strict = np.asarray(Image.open(output / "diagnostic_masks" / f"{i:04d}_strict.png"))
        reference = (np.asarray(Image.fromarray(read_class_mask(root / row["mask"])).resize((256, 256), Image.Resampling.NEAREST))
                     if train else np.asarray(Image.open(output / "diagnostic_masks" / f"{i:04d}_disagreement.png")))
        panels = [raw.copy() for _ in range(4)]
        for column, labels in ((1, reference), (2, anchor), (3, strict)):
            if column == 1 and not train:
                panels[column][labels != 0] = [240, 0, 200]
                continue
            for c, color in ((1, [240, 40, 40]), (2, [0, 220, 240])):
                region = labels == c
                panels[column][region] = (.4 * panels[column][region] + .6 * np.array(color)).astype(np.uint8)
        y = 28 + j * 285
        draw.text((4, y + 3), f"{record['sample_id'][:85]} | disagreement={record['teacher_disagreement_pixels']}", fill="black")
        for column, panel in enumerate(panels): canvas.paste(Image.fromarray(panel), (column * 256, y + 25))
    canvas.save(output / ("train_high_confidence_disagreements.png" if train else "reserve_teacher_disagreements.png"))


def run(config_path, output_path):
    config_path, output = Path(config_path).resolve(), Path(output_path).resolve()
    if output.exists(): raise FileExistsError(output)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = (config_path.parent / config["manifest"]).resolve()
    root, manifest = load_manifest(manifest_path)
    require_training_access(manifest, True)
    active_identity = input_identity(manifest_path, root, manifest)
    reserve = manifest["reserved_unlabeled"]
    if len(reserve) != 800 or len(manifest["train"]["labeled"]) != 190 or manifest["train"]["unlabeled"]:
        raise ValueError("Expected fixed kidney190 and reserved800, with no active unlabeled pool")
    labeled_hashes = {r["image_sha256"] for r in manifest["train"]["labeled"]}
    validation_hashes = {r["image_sha256"] for r in manifest["val"]}
    reserve_files = {}
    for row in reserve:
        if row["source_dataset"] != "kidney_stones_fiber_v1" or row.get("video_id") or row.get("annotation_basis") == "reviewed_visible_pixels":
            raise ValueError("Reserve contains forbidden source or video derivative")
        path = (root / row["image"]).resolve(); digest = sha256(path)
        if digest != row["image_sha256"] or digest in labeled_hashes | validation_hashes:
            raise ValueError("Reserve byte mismatch or exact overlap with labeled/validation images")
        reserve_files[str(path)] = digest
    teacher_paths = [(config_path.parent / row["checkpoint"]).resolve() for row in config["teachers"]]
    checkpoint_hashes = {str(p): sha256(p) for p in teacher_paths}
    for path in teacher_paths:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("model_count") != 1 or state["run_identity"]["inputs"] != active_identity or state["config"].get("pseudo_labels") is not False:
            raise ValueError("Teacher is not a pure-kidney supervised checkpoint with the frozen inputs")
        del state
    output.mkdir(parents=True); (output / "diagnostic_masks").mkdir()
    preflight = {"config": config, "config_sha256": sha256(config_path), "active_inputs": active_identity,
                 "reserve_image_files": reserve_files, "reserve_unique_sha256": len(set(reserve_files.values())),
                 "teacher_sha256": checkpoint_hashes, "source_hashes": source_identity(),
                 "reserve_masks_read": 0, "frozen_before_inference": True}
    atomic_json(output / "preflight.json", preflight)
    code_root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in [Path(__file__), config_path] + sorted((code_root / "ureter_semis").glob("*.py")):
            archive.write(path, str(path.relative_to(code_root)))
    seed_training(config["seed"])
    predictors = [SegmentationPredictor(path) for path in teacher_paths]
    rows = [{**r, "diagnostic_split": split} for split, original in (("train", manifest["train"]["labeled"]), ("reserve", reserve)) for r in original]
    dataset = ImageOnlyDiagnosis(root, rows, config["size"], config["seed"])
    records = []
    histograms = {split: {t["name"]: {str(c): {"count": 0, "confidence_sum": 0., "entropy_sum": 0.,
                    "confidence_hist": [0] * 7, "entropy_hist": [0] * 7} for c in range(3)} for t in config["teachers"]} for split in ("train", "reserve")}
    confidence_bins = [0, .5, .7, .85, .9, .95, .99, 1.000001]
    entropy_bins = [0, .05, .1, .2, .4, .6, .8, 1.000001]
    atomic_json(output / "job.json", {"phase": "inference", "processed_images": 0, "planned_images": 990, "parameter_updates": 0})
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0):
            weak, strong = batch["weak"].to(predictors[0].device), batch["strong"].to(predictors[0].device)
            probabilities = torch.stack([ensemble_logits(p.models, weak).softmax(1) for p in predictors])
            strong_probability = ensemble_logits(predictors[0].models, strong).softmax(1)
            selected, labels, confidence, entropy = policies(probabilities, strong_probability, config["confidence_min"], config["normalized_entropy_max"])
            selected = {k: (v[0].cpu().numpy(), v[1].cpu().numpy()) for k, v in selected.items()}
            labels, confidence, entropy = (v.cpu().numpy() for v in (labels, confidence, entropy))
            for b, index in enumerate(batch["index"].tolist()):
                row = rows[index]; target = batch["target"][b].numpy(); valid = target != 255
                disagreement = (labels[0, b] != labels[1, b]) | (labels[0, b] != labels[2, b])
                record = {"index": index, "sample_id": row["sample_id"], "split": row["diagnostic_split"], "pixels": target.size,
                    "teacher_disagreement_pixels": int(disagreement.sum()),
                    "anchor_fiber_over_source_background": int(((labels[0, b] == 2) & (target == 0) & selected["anchor"][1][b]).sum()),
                    "source_label_pixels": [int((target == c).sum()) for c in range(3)], "policies": {}}
                for name, (pseudo, keep) in selected.items():
                    accepted_valid = keep[b] & valid
                    matrix = np.bincount((target[accepted_valid].astype(int) * 3 + pseudo[b][accepted_valid]).flatten(), minlength=9).reshape(3, 3)
                    record["policies"][name] = {"accepted": [int((keep[b] & (pseudo[b] == c)).sum()) for c in range(3)],
                        "raw": [int((pseudo[b] == c).sum()) for c in range(3)], "label_agreement_matrix": matrix.tolist()}
                for t, teacher in enumerate(config["teachers"]):
                    for c in range(3):
                        region = labels[t, b] == c; item = histograms[row["diagnostic_split"]][teacher["name"]][str(c)]
                        item["count"] += int(region.sum()); item["confidence_sum"] += float(confidence[t, b][region].astype(float).sum())
                        item["entropy_sum"] += float(entropy[t, b][region].astype(float).sum())
                        for field, array, bins in (("confidence_hist", confidence, confidence_bins), ("entropy_hist", entropy, entropy_bins)):
                            item[field] = (np.asarray(item[field]) + np.histogram(array[t, b][region], bins=bins)[0]).tolist()
                records.append(record)
                Image.fromarray(labels[0, b].astype(np.uint8)).save(output / "diagnostic_masks" / f"{index:04d}_anchor.png")
                strict = np.where(selected["consensus_strong"][1][b], selected["consensus_strong"][0][b], 255).astype(np.uint8)
                Image.fromarray(strict).save(output / "diagnostic_masks" / f"{index:04d}_strict.png")
                Image.fromarray(disagreement.astype(np.uint8)).save(output / "diagnostic_masks" / f"{index:04d}_disagreement.png")
            if len(records) % 50 == 0 or len(records) == 990:
                atomic_json(output / "job.json", {"phase": "inference", "processed_images": len(records), "planned_images": 990, "parameter_updates": 0})
                print(f"diagnosed {len(records)}/990", flush=True)
    summaries = {split: {name: aggregate([r for r in records if r["split"] == split], name) for name in config["policies"]} for split in ("train", "reserve")}
    decision = readiness(summaries["train"], summaries["reserve"], config["readiness_gates"])
    result = {"role": "read_only_diagnostic_not_training_pseudo_dataset", "summaries": summaries, "readiness": decision,
        "histograms": histograms, "histogram_bins": {"confidence": confidence_bins, "normalized_entropy": entropy_bins},
        "disagreement": {split: {"images": sum(r["split"] == split for r in records),
            "fraction": sum(r["teacher_disagreement_pixels"] for r in records if r["split"] == split) / sum(r["pixels"] for r in records if r["split"] == split)} for split in ("train", "reserve")},
        "frames": records, "limitations": config["gate_interpretation"], "parameter_updates": 0,
        "reserve_masks_read": 0, "validation_predictions_used_as_pseudo_labels": 0, "training_manifest_unchanged": sha256(manifest_path) == active_identity["manifest_sha256"]}
    atomic_json(output / "diagnosis.json", result)
    contact_sheet(root, rows, records, output, train=True); contact_sheet(root, rows, records, output)
    if sha256(config_path) != preflight["config_sha256"] or source_identity() != preflight["source_hashes"]:
        raise RuntimeError("Diagnostic configuration or source changed")
    atomic_json(output / "job.json", {"phase": "completed", "processed_images": 990, "parameter_updates": 0,
        "readiness_passed": decision["passed"], "reserve_images_read_for_diagnosis": 800, "reserve_images_used_in_training": 0})
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); run(args.config, args.output)
