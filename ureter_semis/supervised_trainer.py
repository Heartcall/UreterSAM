"""Short, reproducible single-network supervised baseline with collapse checks."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

from .data import UreterDataset, load_manifest
from .experiment import (atomic_checkpoint, atomic_json, capture_rng, input_identity,
                         require_training_access, restore_rng, runtime_identity,
                         seed_training, seed_worker, sha256, source_identity)
from .losses import supervised_multiclass_loss
from .model import build_model
from .native_validation import NativeValidationDataset, collate_native, validate_native
from .supervision import audit_supervision
from .trainer import validate


def assess(metrics: dict, gates: dict) -> dict:
    checks = {"stone_dice": metrics["stone_dice"] is not None and metrics["stone_dice"] >= gates["stone_dice_min"],
              "fiber_dice": metrics["fiber_dice"] is not None and metrics["fiber_dice"] >= gates["fiber_dice_min"],
              "negative_fp_area": metrics.get("stone_negative_fp_pixel_fraction") is not None and
                  metrics["stone_negative_fp_pixel_fraction"] <= gates["negative_fp_fraction_max"]}
    return {"passed": all(checks.values()), "checks": checks}


def collapsed(metrics: dict) -> bool:
    stone_collapse = (metrics.get("stone_negative_fp_pixel_fraction", 0) or 0) > 0.8 and (
        metrics.get("stone_precision", 1) or 0) < 0.05
    matrix = metrics.get("confusion_matrix", [])
    total = sum(sum(row) for row in matrix)
    foreground_collapse = total > 0 and sum(row[0] for row in matrix) / total < .01 and (
        (metrics.get("stone_precision", 1) or 0) < .05 or (metrics.get("fiber_precision", 1) or 0) < .05)
    return stone_collapse or foreground_collapse


def labeled_sampler(audit: dict, strategy: str, fiber_image_fraction: float, generator):
    if strategy == "uniform":
        return None
    if strategy != "fiber_balanced" or not 0 < fiber_image_fraction < 1:
        raise ValueError("Invalid labeled sampling strategy or fiber image fraction")
    positives = [row["pixels"].get(2, 0) > 0 for row in audit["samples"]]
    count = sum(positives)
    if count == 0 or count == len(positives):
        raise ValueError("Fiber-balanced sampling requires labeled images with and without fiber")
    weights = [fiber_image_fraction / count if present else
               (1 - fiber_image_fraction) / (len(positives) - count) for present in positives]
    return WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)


def train_supervised(manifest_path: str, output_dir: str, epochs: int = 5, size: int = 256,
                     batch_size: int = 2, learning_rate: float = 1e-4, device_name: str = "cuda:0",
                     workers: int = 0, pretrained: bool = True, class_weights=(1., 2., 4.),
                     dice_weight: float = 1., seed: int = 42, allow_unreviewed_experiment: bool = False,
                     gates: dict | None = None, resume: str | None = None,
                     stop_after_epoch: int | None = None, sampling_strategy: str = "uniform",
                     fiber_image_fraction: float = 0.7, fiber_background_policy: str = "strict_source",
                     architecture: str = "unet_resnet34", encoder_checkpoint: str | None = None,
                     validation_grid: str = "model", images_per_epoch: int | None = None) -> Path:
    if epochs < 1 or size < 32 or size % 32 or batch_size < 1 or workers < 0:
        raise ValueError("Invalid supervised training dimensions")
    if learning_rate <= 0 or not math.isfinite(learning_rate) or len(class_weights) != 3 or any(
            not math.isfinite(w) or w <= 0 for w in class_weights):
        raise ValueError("Invalid learning rate or class weights")
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= epochs:
        raise ValueError("stop_after_epoch outside the frozen schedule")
    if fiber_background_policy not in {"strict_source", "source_partial"}:
        raise ValueError("Unsupported fiber background policy")
    if validation_grid not in {"model", "native_probability"}:
        raise ValueError("Unsupported validation grid")
    if architecture not in {"unet_resnet34", "segformer_mit_b0"}:
        raise ValueError("Unsupported supervised architecture")
    if architecture == "unet_resnet34" and encoder_checkpoint is not None:
        raise ValueError("U-Net retains its original ImageNet initialization path")
    if architecture == "segformer_mit_b0" and pretrained and not encoder_checkpoint:
        raise ValueError("Pretrained SegFormer needs the frozen local encoder file")
    encoder_checkpoint = str(Path(encoder_checkpoint).resolve()) if encoder_checkpoint else None
    root, manifest = load_manifest(manifest_path)
    require_training_access(manifest, allow_unreviewed_experiment)
    if manifest["train"].get("unlabeled"):
        raise ValueError("Pure supervised training must not include an active unlabeled pool")
    audit = audit_supervision(root, manifest)
    if len(manifest["train"]["labeled"]) < batch_size or not manifest["val"]:
        raise ValueError("Need at least one full labeled batch and validation")
    if images_per_epoch is not None and (not isinstance(images_per_epoch,int) or
            not batch_size <= images_per_epoch <= len(manifest['train']['labeled']) or
            images_per_epoch % batch_size or sampling_strategy != 'uniform'):
        raise ValueError("Fixed image budget requires uniform sampling, full batches and no within-epoch replacement")
    output = Path(output_dir).resolve()
    if any((p / "retired_run.json").exists() for p in (output, output.parent)):
        raise ValueError("Retired experiment cannot be resumed")
    resume_path = Path(resume).resolve() if resume else None
    if output.exists() and any(output.iterdir()) and resume_path != output / "last.pt":
        raise FileExistsError("Nonempty output requires its own complete last.pt resume")
    gates = gates or {"stone_dice_min": 0., "fiber_dice_min": 0., "negative_fp_fraction_max": 1.}
    config = {"mode": "supervised_single_network", "model_count": 1, "epochs": epochs,
              "architecture": architecture, "encoder_checkpoint": encoder_checkpoint,
              "validation_grid": validation_grid,
              "size": size, "batch_size": batch_size, "learning_rate": learning_rate,
              "device_name": device_name, "workers": workers, "pretrained": pretrained,
              "class_weights": list(class_weights), "dice_weight": dice_weight, "seed": seed,
              "allow_unreviewed_experiment": allow_unreviewed_experiment, "gates": gates,
              "unlabeled_images_used": 0, "pseudo_labels": False, "mutual_learning": False,
              "scheduler": "CosineAnnealingLR", "weight_decay": 1e-4,
              "selection": "pass all gates first, then stone Dice, negative FP area, fiber Dice",
              "sampling_strategy": sampling_strategy,
              "fiber_image_fraction": fiber_image_fraction if sampling_strategy == "fiber_balanced" else None,
              "fiber_background_policy": fiber_background_policy,
              "partial_policy_definition": "stone_new background or fiber-unannotated image background permits {background,fiber}; stone and fiber positives unchanged",
              "collapse_policy": "after epoch 1, two consecutive epochs of stone FP area>0.8 and precision<0.05, or predicted background<1% and a foreground precision<0.05"}
    if images_per_epoch is not None:
        config['images_per_epoch']=images_per_epoch
        config['budget_sampling']='uniform without replacement per budget epoch; new random subset each epoch'
    identity = input_identity(Path(manifest_path).resolve(), root, manifest)
    if encoder_checkpoint:
        identity["initialization_files"] = {encoder_checkpoint: sha256(Path(encoder_checkpoint))}
    seed_training(seed)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    run = {"schema_version": 3, "config": config, "inputs": identity,
           "runtime": runtime_identity(device), "source_hashes": source_identity()}
    previous = None
    if resume_path:
        previous = torch.load(resume_path, map_location="cpu", weights_only=True)
        if previous.get("model_count") != 1 or previous.get("run_identity") != run:
            raise ValueError("Resume identity mismatch or wrong model family")
        if previous["epoch"] >= (stop_after_epoch or epochs):
            raise ValueError("No remaining epochs")
    generators = {name: torch.Generator().manual_seed(seed + i + 1) for i, name in enumerate(("labeled", "val"))}
    sampler = labeled_sampler(audit, sampling_strategy, fiber_image_fraction, generators["labeled"])
    if images_per_epoch is not None:
        sampler=RandomSampler(manifest['train']['labeled'],replacement=False,num_samples=images_per_epoch,generator=generators['labeled'])
    budget_sample_ids={str((root/row['image']).resolve()):row.get('sample_id',row['image']) for row in manifest['train']['labeled']} if images_per_epoch is not None else {}
    loaders = {name: DataLoader(NativeValidationDataset(root, rows, size) if name == "val" and validation_grid == "native_probability"
                               else UreterDataset(root, rows, size, "train" if name == "labeled" else "val"),
                               batch_size=batch_size, num_workers=workers,
                               shuffle=name == "labeled" and sampler is None,
                               sampler=sampler if name == "labeled" else None,
                               collate_fn=collate_native if name == "val" and validation_grid == "native_probability" else None,
                               drop_last=name == "labeled", worker_init_fn=seed_worker, generator=generators[name])
               for name, rows in (("labeled", manifest["train"]["labeled"]), ("val", manifest["val"]))}
    model_args = {} if architecture == "unet_resnet34" else {
        "architecture": architecture, "encoder_checkpoint": encoder_checkpoint}
    model = build_model(pretrained=pretrained and previous is None, **model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024.)
    weights = torch.tensor(class_weights, device=device, dtype=torch.float32)
    history, start_epoch, best_epoch, best_score, collapse_streak = [], 0, None, (-1., -1., -1., -1.), 0
    if previous:
        model.load_state_dict(previous["model1"])
        state = previous["training_state"]
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        for name, generator in generators.items(): generator.set_state(state["loader_generators"][name])
        history, start_epoch = previous["history"], previous["epoch"]
        best_epoch, best_score, collapse_streak = state["best_epoch"], tuple(state["best_score"]), state["collapse_streak"]
        selected = output / "selected" / f"epoch_{best_epoch:04d}.pt"
        best = torch.load(selected, map_location="cpu", weights_only=True)
        if best["run_identity"] != run or best["epoch"] != best_epoch:
            raise ValueError("Selected checkpoint identity mismatch")
        restore_rng(state["rng"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "selected").mkdir(exist_ok=True)
    atomic_json(output / "run_config.json", run)
    atomic_json(output / "supervision_audit.json", audit)
    for epoch in range(start_epoch, stop_after_epoch or epochs):
        model.train(); loss_sum = 0.; updates = 0; started = time.perf_counter()
        sampled_pixels = torch.zeros(256, dtype=torch.long)
        sampled_images = sampled_fiber_images = 0
        sampled_ids=[]
        sampled_fiber_unknown_background = 0
        learning_rate_now = optimizer.param_groups[0]["lr"]
        for step, batch in enumerate(loaders["labeled"], 1):
            sampled_pixels += torch.bincount(batch["mask"].flatten(), minlength=256)
            sampled_images += len(batch["mask"])
            if images_per_epoch is not None:
                sampled_ids.extend(budget_sample_ids[str(Path(name).resolve())] for name in batch['name'])
            sampled_fiber_images += int((batch["mask"] == 2).flatten(1).any(1).sum())
            images, targets = batch["image"].to(device), batch["mask"].to(device)
            fiber_unknown = None
            if fiber_background_policy == "source_partial":
                incomplete = torch.tensor([source == "stone_new_v4" or not bool(present)
                    for source, present in zip(batch["source_dataset"], batch["native_fiber_annotation"])], device=device)
                fiber_unknown = (targets == 0) & incomplete[:, None, None]
                sampled_fiber_unknown_background += int(fiber_unknown.sum())
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = supervised_multiclass_loss(model(images), targets, weights, dice_weight,
                                                 fiber_unknown_background=fiber_unknown)
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite supervised loss")
            loss_sum += float(loss.detach()); scale = scaler.get_scale()
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            updates += int(scaler.get_scale() >= scale)
            if step == 1 or step % 25 == 0 or step == len(loaders["labeled"]):
                atomic_json(output / "progress.json", {"state": "training", "epoch": epoch + 1,
                    "planned_epochs": epochs, "step": step, "steps_per_epoch": len(loaders["labeled"]),
                    "loss": float(loss.detach()), "epoch_elapsed_s": time.perf_counter() - started})
        if updates: scheduler.step()
        metrics = (validate_native((model,), loaders["val"], device) if validation_grid == "native_probability"
                   else validate((model,), loaders["val"], device))
        screening = assess(metrics, gates)
        score = (int(screening["passed"]), metrics["stone_dice"] or 0.,
                 -(metrics.get("stone_negative_fp_pixel_fraction") or 0.), metrics["fiber_dice"] or 0.)
        improved = score > best_score
        if improved: best_score, best_epoch = score, epoch + 1
        collapse_streak = collapse_streak + 1 if epoch >= 1 and collapsed(metrics) else 0
        stopped_early = collapse_streak >= 2
        state_name = "stopped_early_collapse" if stopped_early else (
            "completed" if epoch + 1 == epochs else "stopped_at_epoch_boundary")
        row = {"epoch": epoch + 1, "train_loss": loss_sum / len(loaders["labeled"]),
               "optimizer_update_steps": updates, "learning_rate": learning_rate_now,
               "sampled_training_images": sampled_images, "sampled_fiber_images": sampled_fiber_images,
               "sampled_label_pixels": {str(i): int(sampled_pixels[i]) for i in (0, 1, 2, 255)},
               "sampled_fiber_unknown_background_pixels": sampled_fiber_unknown_background,
               "metrics": metrics, "screening": screening}
        if images_per_epoch is not None:
            row['sampled_training_sample_ids']=sampled_ids
        history.append(row)
        checkpoint = {"schema_version": 3, "model_count": 1, "architecture": architecture,
            "epoch": epoch + 1, "model1": model.state_dict(),
            "classes": ["background", "stone", "fiber"], "size": size, "config": config,
            "metrics": metrics, "screening": screening, "run_identity": run,
            "history": history, "training_state": {"optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "rng": capture_rng(), "loader_generators": {k: g.get_state() for k, g in generators.items()},
                "best_epoch": best_epoch, "best_score": best_score, "collapse_streak": collapse_streak}}
        if improved:
            selected = {k: v for k, v in checkpoint.items() if k not in {"history", "training_state"}}
            atomic_checkpoint(output / "selected" / f"epoch_{best_epoch:04d}.pt", selected)
        atomic_checkpoint(output / "last.pt", checkpoint)
        if improved: atomic_checkpoint(output / "best.pt", selected)
        atomic_json(output / "history.json", history)
        atomic_json(output / "status.json", {"completed_epoch": epoch + 1, "planned_epochs": epochs,
                    "state": state_name, "best_epoch": best_epoch, "best_passed_screening": bool(best_score[0]),
                    "resume_granularity": "epoch"})
        atomic_json(output / "progress.json", {"state": state_name, "epoch": epoch + 1,
            "planned_epochs": epochs, "step": len(loaders["labeled"]), "steps_per_epoch": len(loaders["labeled"]),
            "train_loss": row["train_loss"], "stone_dice": metrics["stone_dice"],
            "fiber_dice": metrics["fiber_dice"], "screening_passed": screening["passed"]})
        print(f"supervised epoch={epoch + 1} stone={metrics['stone_dice']} fiber={metrics['fiber_dice']} passed={screening['passed']}", flush=True)
        if stopped_early: break
    return output / "best.pt"
