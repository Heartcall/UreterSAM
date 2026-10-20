from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import UreterDataset, load_manifest
from .losses import masked_cross_entropy, supervised_multiclass_loss
from .metrics import multiclass_metrics
from .model import build_model, enable_dropout


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
def validate(models: tuple[nn.Module, nn.Module], loader: DataLoader, device: torch.device) -> dict[str, float]:
    for model in models:
        model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = (models[0](images) + models[1](images)) / 2
        probabilities = torch.softmax(logits, dim=1)
        batch_metrics = multiclass_metrics(probabilities.max(dim=1).indices, masks)
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


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
) -> Path:
    root, manifest = load_manifest(manifest_path)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    labeled = UreterDataset(root, manifest["train"]["labeled"], size, "train")
    unlabeled = UreterDataset(root, manifest["train"]["unlabeled"], size, "unlabeled")
    validation = UreterDataset(root, manifest["val"], size, "val")
    labeled_loader = DataLoader(labeled, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=workers)

    if not len(labeled_loader) or not len(unlabeled_loader):
        raise ValueError("batch-size is larger than a training subset; reduce --batch-size")

    models = (build_model(pretrained=pretrained).to(device), build_model(pretrained=pretrained).to(device))
    optimizers = tuple(torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4) for model in models)
    schedulers = tuple(torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6) for opt in optimizers)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=1024.0)
    weights = torch.tensor(class_weights, device=device, dtype=torch.float32)

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0
    steps = max(len(unlabeled_loader), len(labeled_loader))

    for epoch in range(epochs):
        epoch_updates = 0
        for model in models:
            model.train()
        labeled_iter = _repeat(labeled_loader)
        unlabeled_iter = _repeat(unlabeled_loader)
        progress = tqdm(range(steps), desc=f"epoch {epoch + 1}/{epochs}")
        for _ in progress:
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

            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            previous_scale = scaler.get_scale()
            scaler.scale(total).backward()
            for optimizer in optimizers:
                scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                epoch_updates += 1
            progress.set_postfix(loss=f"{total.item():.3f}")

        if epoch_updates:
            for scheduler in schedulers:
                scheduler.step()
        metrics = validate(models, val_loader, device)
        print("val " + " ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
        checkpoint = {
            "epoch": epoch + 1,
            "model1": models[0].state_dict(),
            "model2": models[1].state_dict(),
            "classes": ["background", "stone", "fiber"],
            "size": size,
            "config": {
                "class_weights": list(class_weights),
                "confidence_threshold": confidence_threshold,
                "uncertainty_quantile": uncertainty_quantile,
                "dice_weight": dice_weight,
                "use_temporal": False,
            },
            "metrics": metrics,
        }
        torch.save(checkpoint, output / "last.pt")
        if metrics["mean_dice"] > best_dice:
            best_dice = metrics["mean_dice"]
            torch.save(checkpoint, output / "best.pt")
            (output / "metrics.json").write_text(json.dumps(checkpoint["metrics"], indent=2), encoding="utf-8")
    return output / "best.pt"
