from __future__ import annotations

import torch
import torch.nn.functional as F


def multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    classes: int = 3,
    class_weights: torch.Tensor | None = None,
    ignore_background: bool = False,
    smooth: float = 1e-6,
    ignore_index: int = 255,
    fiber_unknown_background: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft multiclass Dice loss for logits [B,C,H,W] and target [B,H,W]."""
    if target.ndim != 3 or target.dtype != torch.long:
        raise ValueError("target must have shape [B,H,W] and dtype torch.long")
    valid = target != ignore_index
    if torch.any(valid & ((target < 0) | (target >= classes))):
        raise ValueError("target contains unsupported class ids")
    if not valid.any():
        return logits.sum() * 0.0
    probability = torch.softmax(logits.float(), dim=1) * valid.unsqueeze(1)
    safe_target = torch.where(valid, target, 0)
    one_hot = F.one_hot(safe_target, num_classes=classes)
    one_hot = one_hot.permute(0, 3, 1, 2).to(dtype=probability.dtype)
    one_hot = one_hot * valid.unsqueeze(1)
    if fiber_unknown_background is not None:
        known = torch.ones_like(probability)
        known[:, 2] = ~fiber_unknown_background
        probability = probability * known
        one_hot = one_hot * known
    reduce_dims = (0, 2, 3)
    intersection = (probability * one_hot).sum(dim=reduce_dims)
    denominator = probability.sum(dim=reduce_dims) + one_hot.sum(dim=reduce_dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    selected = dice[1:] if ignore_background else dice
    if class_weights is not None:
        weights = class_weights[1:] if ignore_background else class_weights
        weights = weights.to(device=selected.device, dtype=selected.dtype)
        return 1.0 - (selected * weights).sum() / weights.sum().clamp_min(smooth)
    return 1.0 - selected.mean()


def supervised_multiclass_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor,
    dice_weight: float = 1.0,
    fiber_unknown_background: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted CE plus foreground multiclass Dice."""
    if not (target != 255).any():
        return logits.sum() * 0.0
    # CUDA's reduced NLLLoss2d rejects strict deterministic mode in the pinned
    # runtime. Gather the same weighted log likelihood and reduce explicitly.
    valid = target != 255
    safe_target = torch.where(valid, target, 0)
    log_probability = F.log_softmax(logits.float(), dim=1)
    pixel_loss = -log_probability.gather(1, safe_target.unsqueeze(1)).squeeze(1)
    if fiber_unknown_background is not None:
        if fiber_unknown_background.shape != target.shape or fiber_unknown_background.dtype != torch.bool:
            raise ValueError("Fiber-unknown region must be a boolean mask with the target shape")
        if torch.any(fiber_unknown_background & (target != 0)):
            raise ValueError("Only provisional background can have unknown fiber membership")
        allowed_background_loss = -torch.logsumexp(log_probability[:, (0, 2)], dim=1)
        pixel_loss = torch.where(fiber_unknown_background, allowed_background_loss, pixel_loss)
    pixel_weights = class_weights[safe_target] * valid
    ce = (pixel_loss * pixel_weights).sum() / pixel_weights.sum().clamp_min(1e-12)
    dice = multiclass_dice_loss(logits, target, classes=logits.shape[1], class_weights=class_weights,
                               ignore_background=True, fiber_unknown_background=fiber_unknown_background)
    return ce + dice_weight * dice


def masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor, reliable: torch.Tensor) -> torch.Tensor:
    """Cross entropy over reliable multiclass pseudo-label pixels only."""
    loss = -F.log_softmax(logits.float(), dim=1).gather(1, target.long().unsqueeze(1)).squeeze(1)
    reliable = reliable.to(dtype=loss.dtype)
    return (loss * reliable).sum() / reliable.sum().clamp_min(1.0)
