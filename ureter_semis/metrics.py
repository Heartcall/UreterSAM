from __future__ import annotations

import torch


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    return float((numerator / denominator.clamp_min(1)).item())


def multiclass_metrics(prediction: torch.Tensor, target: torch.Tensor, classes: int = 3) -> dict[str, float]:
    """Aggregate per-pixel metrics for stone (1) and fiber (2)."""
    prediction = prediction.detach().long().flatten()
    target = target.detach().long().flatten()
    confusion = torch.zeros((classes, classes), dtype=torch.long, device=prediction.device)
    confusion.index_put_((target, prediction), torch.ones_like(target), accumulate=True)
    result: dict[str, float] = {}
    dice_values, iou_values = [], []
    for class_id, name in ((1, "stone"), (2, "fiber")):
        tp = confusion[class_id, class_id]
        fp = confusion[:, class_id].sum() - tp
        fn = confusion[class_id, :].sum() - tp
        dice = _safe_divide(2 * tp, 2 * tp + fp + fn)
        iou = _safe_divide(tp, tp + fp + fn)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        result.update({
            f"{name}_dice": dice,
            f"{name}_iou": iou,
            f"{name}_precision": precision,
            f"{name}_recall": recall,
        })
        dice_values.append(dice)
        iou_values.append(iou)
    result["mean_dice"] = sum(dice_values) / len(dice_values)
    result["mean_iou"] = sum(iou_values) / len(iou_values)
    return result
