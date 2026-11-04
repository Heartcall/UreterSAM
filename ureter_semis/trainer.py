from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .data import UreterDataset, load_manifest
from .losses import masked_cross_entropy, supervised_multiclass_loss
from .metrics import confusion_matrix, metrics_from_confusion
from .experiment import (atomic_checkpoint, atomic_json, capture_rng, input_identity,
                         require_training_access, restore_rng, runtime_identity, seed_training, seed_worker, source_identity)
from .model import build_model, enable_dropout, ensemble_logits
from .supervision import audit_supervision


def _repeat(loader: DataLoader) -> Iterator[dict]:
    """Repeat a loader without caching its batches in host memory."""
    while True:
        yield from loader


def _uncertainty(model: nn.Module, images: torch.Tensor, passes: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    enable_dropout(model)
    probabilities = []
    with torch.no_grad():
        for _ in range(passes):
            probabilities.append(torch.softmax(model(images), dim=1))
    stacked = torch.stack(probabilities)
    mean = stacked.mean(dim=0)
    entropy = -(mean * (mean.clamp_min(1e-7).log())).sum(dim=1)
    entropy = entropy / math.log(mean.shape[1])
    return mean, entropy


def _low_uncertainty(entropy: torch.Tensor, quantile: float = 0.8) -> torch.Tensor:
    threshold = torch.quantile(entropy.flatten(1), quantile, dim=1).view(-1, 1, 1)
    return entropy <= threshold


def _reliable_mask(
    probability: torch.Tensor,
    entropy: torch.Tensor,
    confidence_threshold: float,
    uncertainty_quantile: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    confidence, pseudo = probability.max(dim=1)
    low_entropy = _low_uncertainty(entropy, uncertainty_quantile)
    return pseudo.detach(), (confidence >= confidence_threshold) & low_entropy


@torch.no_grad()
def validate(models: tuple[nn.Module, ...], loader: DataLoader, device: torch.device) -> dict:
    for model in models:
        model.eval()
    confusion = torch.zeros((3, 3), dtype=torch.long, device=device)
    negatives = negative_fp_images = negative_pixels = negative_fp_pixels = 0
    for batch in loader:
        images, targets = batch["image"].to(device), batch["mask"].to(device)
        predictions = ensemble_logits(models, images).argmax(dim=1)
        confusion += confusion_matrix(predictions, targets)
        # Presence is established at native resolution, before resizing tiny stones.
        for i, presence in enumerate(batch.get("stone_presence", ["unknown"] * len(targets))):
            valid = targets[i] != 255
            if presence != "absent" or not valid.any():
                continue
            negatives += 1
            false_stone = (predictions[i] == 1) & valid
            negative_fp_images += int(false_stone.any())
            negative_fp_pixels += int(false_stone.sum())
            negative_pixels += int(valid.sum())
    if not int(confusion.sum()):
        raise ValueError("Validation contains no labelled pixels")
    result = metrics_from_confusion(confusion)
    result.update({"stone_negative_images": negatives,
                   "stone_negative_images_with_fp": negative_fp_images,
                   "stone_negative_fp_image_rate": negative_fp_images / negatives if negatives else None,
                   "stone_negative_valid_pixels": negative_pixels,
                   "stone_negative_fp_pixels": negative_fp_pixels,
                   "stone_negative_fp_pixel_fraction": negative_fp_pixels / negative_pixels if negative_pixels else None})
    return result


def train_endosemis(
    manifest_path: str,
    output_dir: str,
    epochs: int = 60,
    size: int = 320,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    device_name: str = "cuda:0",
    workers: int = 0,
    pretrained: bool = True,
    class_weights: tuple[float, float, float] = (1.0, 2.0, 4.0),
    confidence_threshold: float = 0.85,
    uncertainty_quantile: float = 0.8,
    dice_weight: float = 1.0,
    seed: int = 42,
    resume: str | None = None,
    stop_after_epoch: int | None = None,
    allow_unreviewed_experiment: bool = False,
) -> Path:
    root, manifest = load_manifest(manifest_path)
    require_training_access(manifest, allow_unreviewed_experiment)
    audit_supervision(root, manifest)
    if epochs < 1 or size < 32 or size % 32 or batch_size < 1 or workers < 0:
        raise ValueError("Require epochs/batch-size > 0, workers >= 0, size a positive multiple of 32")
    if not 0 <= seed < 2**32 or learning_rate <= 0 or any(w <= 0 for w in class_weights):
        raise ValueError("Invalid seed, learning rate or class weights")
    if not 0 <= confidence_threshold <= 1 or not 0 < uncertainty_quantile <= 1 or dice_weight < 0:
        raise ValueError("Invalid loss/pseudo-label configuration")
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= epochs:
        raise ValueError("stop_after_epoch must lie within the fixed total epoch schedule")
    output = Path(output_dir).expanduser().resolve()
    resume_path = Path(resume).expanduser().resolve() if resume else None
    if output.exists() and any(output.iterdir()):
        if resume_path != output / "last.pt":
            raise FileExistsError("Nonempty output requires --resume pointing to its last.pt")
    # Validate provenance before models, downloads, or writes.
    identity = input_identity(Path(manifest_path).resolve(), root, manifest)
    seed_training(seed)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    config = {"epochs": epochs, "size": size, "batch_size": batch_size,
              "learning_rate": learning_rate, "device": str(device), "workers": workers,
              "pretrained": pretrained, "class_weights": list(class_weights),
              "confidence_threshold": confidence_threshold, "uncertainty_quantile": uncertainty_quantile,
              "dice_weight": dice_weight, "seed": seed, "use_temporal": False,
              "deterministic_algorithms": True, "drop_last": True,
              "scheduler": "CosineAnnealingLR", "weight_decay": 1e-4,
              "selection_metric": "global_foreground_mean_dice_defined_classes",
              "allow_unreviewed_experiment": allow_unreviewed_experiment,
              "dataset_role": manifest.get("role"),
              "labeled_sampling": "weighted_replacement" if any("sampling_weight" in row for row in manifest["train"]["labeled"]) else "shuffle"}
    run = {"schema_version": 2, "config": config, "inputs": identity,
           "runtime": runtime_identity(device), "source_hashes": source_identity()}
    previous = None
    if resume_path:
        previous = torch.load(resume_path, map_location="cpu", weights_only=True)
        if previous.get("schema_version") != 2 or "training_state" not in previous:
            raise ValueError("Legacy weights lack full optimizer/RNG state; cannot resume exactly")
        if previous["run_identity"] != run:
            raise ValueError("Resume identity mismatch: configuration, data, source or runtime changed")
        if previous["epoch"] >= epochs or (stop_after_epoch and previous["epoch"] >= stop_after_epoch):
            raise ValueError("No remaining epochs within the fixed schedule")
    generators = {name: torch.Generator().manual_seed(seed + i + 1)
                  for i, name in enumerate(("labeled", "unlabeled", "val"))}
    labeled = UreterDataset(root, manifest["train"]["labeled"], size, "train")
    unlabeled = UreterDataset(root, manifest["train"]["unlabeled"], size, "unlabeled")
    validation = UreterDataset(root, manifest["val"], size, "val")
    if len(labeled) < batch_size or len(unlabeled) < batch_size or not len(validation):
        raise ValueError("Training subsets must fit batch-size and validation must be nonempty")
    options = {"batch_size": batch_size, "num_workers": workers, "worker_init_fn": seed_worker}
    sampler = None
    if config["labeled_sampling"] == "weighted_replacement":
        sampling_weights = [row.get("sampling_weight", 0) for row in manifest["train"]["labeled"]]
        if any(not math.isfinite(weight) or weight <= 0 for weight in sampling_weights):
            raise ValueError("All labeled sampling weights must be finite and positive")
        sampler = WeightedRandomSampler(sampling_weights, len(labeled), replacement=True, generator=generators["labeled"])
    labeled_loader = DataLoader(labeled, shuffle=sampler is None, sampler=sampler, drop_last=True, generator=generators["labeled"], **options)
    unlabeled_loader = DataLoader(unlabeled, shuffle=True, drop_last=True, generator=generators["unlabeled"], **options)
    val_loader = DataLoader(validation, shuffle=False, generator=generators["val"], **options)
    models = tuple(build_model(pretrained=pretrained and previous is None).to(device) for _ in range(2))
    optimizers = tuple(torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4) for model in models)
    schedulers = tuple(torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6) for opt in optimizers)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=1024.0)
    weights = torch.tensor(class_weights, device=device, dtype=torch.float32)
    best_dice, start_epoch, history, best_epoch = -1.0, 0, [], None
    best_checkpoint = None
    if previous:
        state = previous["training_state"]
        for i, model in enumerate(models):
            model.load_state_dict(previous[f"model{i+1}"])
            optimizers[i].load_state_dict(state["optimizers"][i])
            schedulers[i].load_state_dict(state["schedulers"][i])
        scaler.load_state_dict(state["scaler"])
        for name, generator in generators.items():
            generator.set_state(state["loader_generators"][name])
        best_dice, best_epoch = state["best_dice"], state["best_epoch"]
        start_epoch, history = previous["epoch"], previous["history"]
        # Preserve the earlier best when forking a resumed run into a new directory.
        best_path = resume_path.parent / state["best_checkpoint"]
        if not best_path.exists():
            raise ValueError("Resume needs the recorded selected checkpoint to preserve model selection")
        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
        if best_checkpoint.get("run_identity") != run or best_checkpoint["epoch"] != best_epoch:
            raise ValueError("Selected checkpoint does not match the resumed run")
        restore_rng(state["rng"])
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "run_config.json", run)
    (output / "selected").mkdir(exist_ok=True)
    if best_checkpoint is not None:
        selected_path = output / "selected" / f"epoch_{best_epoch:04d}.pt"
        if not selected_path.exists():
            atomic_checkpoint(selected_path, best_checkpoint)
        atomic_checkpoint(output / "best.pt", best_checkpoint)
        atomic_json(output / "metrics.json", best_checkpoint["metrics"])
    # Do not retain CPU copies of both large optimizer checkpoints during GPU training.
    del previous, best_checkpoint
    steps = max(len(unlabeled_loader), len(labeled_loader))
    final_epoch = stop_after_epoch or epochs

    for epoch in range(start_epoch, final_epoch):
        epoch_updates = 0
        loss_sum = 0.0
        learning_rates = [opt.param_groups[0]["lr"] for opt in optimizers]
        for model in models:
            model.train()
        labeled_iter = _repeat(labeled_loader)
        unlabeled_iter = _repeat(unlabeled_loader)
        epoch_started = time.perf_counter()
        progress = tqdm(range(steps), desc=f"epoch {epoch + 1}/{epochs}", mininterval=5.0)
        for step in progress:
            labeled_batch = next(labeled_iter)
            unlabeled_batch = next(unlabeled_iter)
            images = labeled_batch["image"].to(device)
            masks = labeled_batch["mask"].to(device)
            weak = unlabeled_batch["weak"].to(device)
            strong = unlabeled_batch["strong"].to(device)

            probability1, entropy1 = _uncertainty(models[0], weak)
            probability2, entropy2 = _uncertainty(models[1], weak)
            pseudo1, confident1 = _reliable_mask(probability1, entropy1, confidence_threshold, uncertainty_quantile)
            pseudo2, confident2 = _reliable_mask(probability2, entropy2, confidence_threshold, uncertainty_quantile)
            choose1 = entropy1 <= entropy2
            joint_pseudo = torch.where(choose1, pseudo1, pseudo2)
            joint_confident = torch.where(choose1, confident1, confident2)

            for model in models:
                model.train()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits1, features1 = models[0](images, return_features=True)
                logits2, features2 = models[1](images, return_features=True)
                weak_logits1, weak_logits2 = models[0](weak), models[1](weak)
                strong_logits1, strong_logits2 = models[0](strong), models[1](strong)

                loss1 = supervised_multiclass_loss(logits1, masks, weights, dice_weight)
                loss2 = supervised_multiclass_loss(logits2, masks, weights, dice_weight)
                cross1 = masked_cross_entropy(weak_logits1, pseudo2, confident2)
                cross2 = masked_cross_entropy(weak_logits2, pseudo1, confident1)
                strong1 = masked_cross_entropy(strong_logits1, joint_pseudo, joint_confident)
                strong2 = masked_cross_entropy(strong_logits2, joint_pseudo, joint_confident)
                ramp = min(1.0, (epoch + 1) / 5.0) * 0.5
                mutual = F.mse_loss(logits1.float(), logits2.float())
                mutual = mutual + 0.1 * F.mse_loss(features1[-1].float(), features2[-1].float())
                total = loss1 + loss2 + ramp * (cross1 + cross2 + strong1 + strong2) + 0.1 * mutual

            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}; no checkpoint written")
            loss_sum += float(total.detach())
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            previous_scale = scaler.get_scale()
            scaler.scale(total).backward()
            for optimizer in optimizers:
                scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                epoch_updates += 1
            progress.set_postfix(loss=f"{total.item():.3f}", refresh=False)
            if step % 25 == 0 or step + 1 == steps:
                atomic_json(output / "progress.json", {"state": "training", "epoch": epoch + 1, "planned_epochs": epochs,
                            "step": step + 1, "steps_per_epoch": steps, "loss": float(total.detach()),
                            "epoch_elapsed_s": time.perf_counter() - epoch_started})

        if epoch_updates:
            for scheduler in schedulers:
                scheduler.step()
        metrics = validate(models, val_loader, device)
        selection = metrics["mean_dice"]
        if selection is None:
            raise ValueError("Validation has no defined foreground overlap for model selection")
        improved = selection > best_dice
        if improved:
            best_dice, best_epoch = selection, epoch + 1
        row = {"epoch": epoch + 1, "train_loss": loss_sum / steps,
               "optimizer_update_steps": epoch_updates, "learning_rates": learning_rates,
               "metrics": metrics}
        history.append(row)
        checkpoint = {
            "schema_version": 2, "epoch": epoch + 1,
            "model1": models[0].state_dict(), "model2": models[1].state_dict(),
            "classes": ["background", "stone", "fiber"], "size": size,
            "config": config, "metrics": metrics, "history": history, "run_identity": run,
            "training_state": {"optimizers": [opt.state_dict() for opt in optimizers],
                               "schedulers": [sch.state_dict() for sch in schedulers],
                               "scaler": scaler.state_dict(), "rng": capture_rng(),
                               "loader_generators": {k: g.get_state() for k, g in generators.items()},
                               "best_dice": best_dice, "best_epoch": best_epoch,
                               "best_checkpoint": f"selected/epoch_{best_epoch:04d}.pt"},
        }
        # Immutable selected weights keep a previous last.pt recoverable if saving is interrupted.
        if improved:
            selected = {k: v for k, v in checkpoint.items() if k not in {"training_state", "history"}}
            atomic_checkpoint(output / "selected" / f"epoch_{best_epoch:04d}.pt", selected)
        atomic_checkpoint(output / "last.pt", checkpoint)
        if improved:
            atomic_checkpoint(output / "best.pt", selected)
            atomic_json(output / "metrics.json", metrics)
        atomic_json(output / "history.json", history)
        atomic_json(output / "status.json", {"completed_epoch": epoch + 1, "planned_epochs": epochs,
                    "state": "completed" if epoch + 1 == epochs else "stopped_at_epoch_boundary",
                    "best_epoch": best_epoch, "resume_granularity": "epoch"})
        atomic_json(output / "progress.json", {"state": "completed" if epoch + 1 == epochs else "epoch_completed",
                    "epoch": epoch + 1, "planned_epochs": epochs, "step": steps, "steps_per_epoch": steps,
                    "train_loss": loss_sum / steps, "validation_mean_dice": selection})
        print(f"val epoch={epoch + 1} mean_dice={selection:.6f} negative_fp_images={metrics['stone_negative_images_with_fp']}")
    return output / "best.pt"
