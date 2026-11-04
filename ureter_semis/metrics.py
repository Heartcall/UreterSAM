from __future__ import annotations

import torch


def confusion_matrix(prediction: torch.Tensor, target: torch.Tensor, classes: int = 3) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must share shape")
    prediction, target = prediction.detach().long().flatten(), target.detach().long().flatten()
    valid = target != 255
    prediction, target = prediction[valid], target[valid]
    if torch.any((target < 0) | (target >= classes) | (prediction < 0) | (prediction >= classes)):
        raise ValueError("prediction/target contains unsupported class ids")
    return torch.bincount(target * classes + prediction, minlength=classes**2).reshape(classes, classes)


def metrics_from_confusion(confusion: torch.Tensor) -> dict:
    """Global foreground metrics. Undefined denominators remain null in JSON."""
    result = {}
    dice_values, iou_values = [], []
    for class_id, name in ((1, "stone"), (2, "fiber")):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum()) - tp
        fn = int(confusion[class_id, :].sum()) - tp
        dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        iou = tp / (tp + fp + fn) if tp + fp + fn else None
        result.update({f"{name}_dice": dice, f"{name}_iou": iou,
                       f"{name}_precision": tp / (tp + fp) if tp + fp else None,
                       f"{name}_recall": tp / (tp + fn) if tp + fn else None,
                       f"{name}_tp": tp, f"{name}_fp": fp, f"{name}_fn": fn})
        if dice is not None:
            dice_values.append(dice)
            iou_values.append(iou)
    result["mean_dice"] = sum(dice_values) / len(dice_values) if dice_values else None
    result["mean_iou"] = sum(iou_values) / len(iou_values) if iou_values else None
    result["foreground_classes_with_defined_overlap"] = len(dice_values)
    result["confusion_matrix"] = confusion.cpu().tolist()
    result["valid_pixels"] = int(confusion.sum())
    return result


def multiclass_metrics(prediction: torch.Tensor, target: torch.Tensor, classes: int = 3) -> dict:
    return metrics_from_confusion(confusion_matrix(prediction, target, classes))
