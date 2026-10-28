"""Same-grid boundary and small-stone diagnostics for fixed validation frames."""
from __future__ import annotations

import cv2
import numpy as np
from torch.utils.data import DataLoader
import torch

from ureter_semis.data import UreterDataset, load_manifest
from ureter_semis.inference import SegmentationPredictor
from ureter_semis.model import ensemble_logits


def boundary_counts(prediction, target, label, tolerance=2):
    valid = cv2.erode((target != 255).astype(np.uint8), np.ones((2 * tolerance + 1,) * 2, np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    def edge(region):
        eroded = cv2.erode(region.astype(np.uint8), np.ones((3, 3), np.uint8),
                           borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        return region & ~eroded & valid
    predicted, truth = edge(prediction == label), edge(target == label)
    def matched(a, b):
        if not b.any(): return 0
        distance = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        return int((a & (distance <= tolerance)).sum())
    return {"predicted_boundary": int(predicted.sum()), "target_boundary": int(truth.sum()),
            "matched_predicted": matched(predicted, truth), "matched_target": matched(truth, predicted)}


def boundary_summary(counts):
    summed = {k: sum(r[k] for r in counts) for k in ("predicted_boundary", "target_boundary", "matched_predicted", "matched_target")}
    p = summed["matched_predicted"] / summed["predicted_boundary"] if summed["predicted_boundary"] else None
    r = summed["matched_target"] / summed["target_boundary"] if summed["target_boundary"] else None
    f = 2 * p * r / (p + r) if p is not None and r is not None and p + r else (0. if summed["target_boundary"] else None)
    return {**summed, "precision": p, "recall": r, "f_score": f}


def evaluate(manifest_path, checkpoints, tolerance=2, small_max_pixels=128):
    root, manifest = load_manifest(manifest_path)
    results = {}
    for name, path in checkpoints.items():
        predictor = SegmentationPredictor(path)
        dataset = UreterDataset(root, manifest["val"], predictor.size, "val")
        rows = []
        with torch.inference_mode():
            for batch in DataLoader(dataset, batch_size=2, shuffle=False):
                predictions = ensemble_logits(predictor.models, batch["image"].to(predictor.device)).argmax(1).cpu().numpy()
                for prediction, target, presence in zip(predictions, batch["mask"].numpy(), batch["stone_presence"]):
                    index = len(rows); gt_stone = target == 1; valid = target != 255
                    rows.append({"sample_id": manifest["val"][index]["sample_id"],
                        "stone_gt_pixels": int(gt_stone.sum()), "native_stone_presence": presence,
                        "stone_tp": int(((prediction == 1) & gt_stone).sum()),
                        "stone_fp": int(((prediction == 1) & ~gt_stone & valid).sum()),
                        "boundaries": {str(c): boundary_counts(prediction, target, c, tolerance) for c in (1, 2)}})
        small = [r for r in rows if 0 < r["stone_gt_pixels"] <= small_max_pixels]
        tp, total = sum(r["stone_tp"] for r in small), sum(r["stone_gt_pixels"] for r in small)
        results[name] = {"boundaries": {str(c): boundary_summary([r["boundaries"][str(c)] for r in rows]) for c in (1, 2)},
            "small_stone_frames": len(small), "small_stone_target_pixels": total,
            "small_stone_tp": tp, "small_stone_pixel_recall": tp / total if total else None,
            "small_stone_frames_any_tp": sum(r["stone_tp"] > 0 for r in small),
            "native_positive_but_resized_stone_empty": sum(r["native_stone_presence"] == "present" and r["stone_gt_pixels"] == 0 for r in rows),
            "frames": rows}
        del predictor
        torch.cuda.empty_cache()
    return {"scope": "PIL model grid; pooled boundary precision/recall, 255 and image-border uncertainty margin excluded",
            "tolerance_pixels": tolerance, "small_stone_max_gt_pixels": small_max_pixels,
            "interpretation": "Whole-frame small visible stone area, not instance accuracy or physical size; no independent test",
            "models": results}
