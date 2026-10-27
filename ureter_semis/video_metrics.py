"""Prediction diagnostics; without annotations these are NOT accuracy metrics."""
from __future__ import annotations

import cv2
import numpy as np


def estimated_fov(frame: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    small = cv2.resize(frame, (shape[1], shape[0]))
    binary = (small.max(axis=2) > 10).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros(shape, np.uint8)
    if contours:
        cv2.drawContours(result, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    return result.astype(bool)


def prediction_diagnostics(mask: np.ndarray, probability: np.ndarray,
                           fov: np.ndarray, previous: np.ndarray | None = None) -> dict:
    valid = int(fov.sum())
    count, _, stats, _ = cv2.connectedComponentsWithStats((mask == 1).astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.array([], dtype=int)
    entropy = -(probability * np.log(np.maximum(probability, 1e-7))).sum(axis=0) / np.log(3)
    return {
        "estimated_fov_pixels": valid,
        "stone_pixels": int((mask == 1).sum()),
        "fiber_pixels": int((mask == 2).sum()),
        "stone_fraction_estimated_fov": float(((mask == 1) & fov).sum() / valid) if valid else None,
        "stone_components_ge_5px": int((areas >= 5).sum()),
        "largest_stone_component_px": int(areas.max()) if areas.size else 0,
        "mean_confidence_estimated_fov": float(probability.max(axis=0)[fov].mean()) if valid else None,
        "mean_entropy_estimated_fov": float(entropy[fov].mean()) if valid else None,
        "uncompensated_label_change_fraction": float((mask[fov] != previous[fov]).mean())
            if previous is not None and valid else None,
    }


def scores_from_confusion(matrix: np.ndarray) -> dict:
    result = {}
    for class_id, name in ((1, "stone"), (2, "fiber")):
        tp = int(matrix[class_id, class_id])
        fp = int(matrix[:, class_id].sum()) - tp
        fn = int(matrix[class_id, :].sum()) - tp
        result[name] = {"tp": tp, "fp": fp, "fn": fn,
                        "dice": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                        "iou": tp/(tp+fp+fn) if tp+fp+fn else None,
                        "precision": tp/(tp+fp) if tp+fp else None,
                        "recall": tp/(tp+fn) if tp+fn else None}
    return result


def confusion_for_valid_pixels(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    if prediction.shape != truth.shape:
        raise ValueError("Prediction and truth must share coordinates and resolution")
    if not set(np.unique(prediction)).issubset({0, 1, 2}):
        raise ValueError("Prediction must contain only classes 0/1/2")
    if not set(np.unique(truth)).issubset({0, 1, 2, 255}):
        raise ValueError("Truth must contain classes 0/1/2 or ignore=255")
    valid = truth != 255
    return np.bincount((truth[valid].astype(np.int64)*3 + prediction[valid]).ravel(),
                       minlength=9).reshape(3, 3)
