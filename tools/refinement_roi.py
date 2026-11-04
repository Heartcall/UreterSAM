"""Ground-truth-free native ROI proposals for bounded refinement feasibility."""
from __future__ import annotations

import math
import cv2
import numpy as np


def _axis_starts(lower, upper, length, side, stride):
    """Cover padded half-open extent using shifted, in-image fixed windows."""
    side = min(side, length)
    lower, upper = max(0, lower), min(length, upper)
    if upper - lower <= side:
        return [max(0, min(length - side, (lower + upper - side) // 2))]
    first, last = min(lower, length-side), max(0, upper-side)
    intervals = math.ceil((last-first) / stride)
    return sorted({round(first + (last-first)*i/intervals) for i in range(intervals+1)})


def propose_rois(probabilities, settings, seed="argmax", max_rois=None):
    """Return half-open (x0,y0,x1,y1) boxes; target labels are never accepted."""
    probability = np.asarray(probabilities)
    if probability.ndim != 3 or probability.shape[0] != 3 or not np.isfinite(probability).all():
        raise ValueError("Expected finite native [3,H,W] probabilities")
    side, margin, stride, minimum = (int(settings[k]) for k in
        ("native_crop_side", "margin_native_pixels", "max_tile_stride", "minimum_component_pixels"))
    if side <= 0 or not 0 < stride <= side or margin < 0 or minimum <= 0:
        raise ValueError("Invalid ROI geometry")
    if max_rois is not None and (not isinstance(max_rois, int) or max_rois < 0):
        raise ValueError("ROI cap must be a nonnegative integer or None")
    if seed not in ("argmax", "probability"):
        raise ValueError("Unknown seed policy")
    height, width = probability.shape[1:]
    hard = probability.argmax(0)
    masks = []
    candidates = set()
    for cls in (1, 2):
        raw = hard == cls if seed == "argmax" else probability[cls] >= settings["soft_probability_threshold"]
        count, labels, stats, _ = cv2.connectedComponentsWithStats(raw.astype(np.uint8), connectivity=8)
        clean = np.zeros_like(raw)
        for index in range(1, count):
            x, y, w, h, area = (int(value) for value in stats[index])
            if area < minimum:
                continue
            clean[labels == index] = True
            xs = _axis_starts(x-margin, x+w+margin, width, side, stride)
            ys = _axis_starts(y-margin, y+h+margin, height, side, stride)
            for y0 in ys:
                for x0 in xs:
                    candidates.add((x0, y0, x0+min(side,width), y0+min(side,height)))
        masks.append(clean)
    seeds = np.stack(masks)
    totals = seeds.sum((1,2))
    covered = np.zeros((height, width), dtype=bool)
    ranked = []
    remaining = sorted(candidates)
    while remaining and (max_rois is None or len(ranked) < max_rois):
        scores = []
        for x0,y0,x1,y1 in remaining:
            fresh = seeds[:,y0:y1,x0:x1] & ~covered[y0:y1,x0:x1]
            scores.append(float(np.divide(fresh.sum((1,2)), totals, out=np.zeros(2), where=totals>0).sum()))
        index = int(np.argmax(scores))
        if scores[index] <= 0:
            break
        box = remaining.pop(index)
        ranked.append(box)
        x0,y0,x1,y1 = box
        covered[y0:y1,x0:x1] = True
    return ranked


def roi_union(boxes, shape):
    mask = np.zeros(shape, dtype=bool)
    height, width = shape
    for x0,y0,x1,y1 in boxes:
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("ROI outside native image")
        mask[y0:y1,x0:x1] = True
    return mask


def coverage_counts(prediction, target, boxes):
    """Coverage measures where a future refiner can act, not its accuracy."""
    if prediction.shape != target.shape:
        raise ValueError("Prediction and GT geometry differ")
    union = roi_union(boxes, target.shape)
    valid = target != 255
    result = {}
    for cls,name in ((1,"stone"),(2,"fiber")):
        truth = target == cls
        tp = (prediction == cls) & truth
        fp = (prediction == cls) & ~truth & valid
        result[name] = {"target_pixels": int(truth.sum()), "covered_pixels": int((truth & union).sum()),
            "coarse_tp": int(tp.sum()), "coarse_fn": int((truth & ~tp).sum()),
            "coarse_fn_covered": int((truth & ~tp & union).sum()),
            "coarse_fp": int(fp.sum()), "coarse_fp_covered": int((fp & union).sum())}
    result.update({"roi_count":len(boxes), "roi_area_sum":sum((b[2]-b[0])*(b[3]-b[1]) for b in boxes),
        "roi_area_union":int(union.sum()), "image_pixels":int(target.size),
        "valid_pixels_in_roi":int((valid & union).sum()),
        "background_pixels_in_roi":int(((target == 0) & union).sum())})
    return result
