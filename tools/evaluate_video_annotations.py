"""Evaluate approved pixel annotations; refuses event-only or pending reviews."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ureter_semis.video_metrics import confusion_for_valid_pixels, scores_from_confusion
from ureter_semis.labels import read_class_mask
from ureter_semis.inference import file_sha256


def evaluate(config_path: str | Path, annotations_path: str | Path, output_path: str | Path) -> dict:
    config_path, annotations_path, output = map(lambda p: Path(p).resolve(),
                                               (config_path, annotations_path, output_path))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    same_baseline = file_sha256(config_path) == annotations.get("baseline_config_sha256")
    if annotations.get("baseline_config_sha256") and not same_baseline and not annotations.get("frozen_inputs"):
        raise ValueError("Different model config needs frozen input identities; regenerate the snapshot")
    approved = [r for r in annotations.get("frames", []) if r.get("review_status") == "approved"]
    if not approved:
        raise ValueError("No approved pixel masks; event intervals do not establish segmentation accuracy")
    inputs = {r["video_id"]: r for r in config["inputs"]}
    for frozen in annotations.get("frozen_inputs", []):
        current = inputs.get(frozen["video_id"], {})
        if any(current.get(key) != value for key, value in frozen.items()):
            raise ValueError("Evaluation video identity differs from the reviewed snapshot")
    matrix = np.zeros((3,3), np.int64)
    negatives, negative_fp, negative_area, rows = 0, 0, [], []
    seen = set()
    for record in approved:
        vid, index = record["video_id"], record["frame_index"]
        if (vid,index) in seen:
            raise ValueError("Duplicate approved frame")
        seen.add((vid,index))
        if not record.get("reviewer", "").strip() or not record.get("mask"):
            raise ValueError("Approved frames require reviewer identity and a pixel mask")
        item = inputs[vid]
        if not isinstance(index,int) or not 0 <= index < item["frames"]:
            raise ValueError("Invalid source frame index")
        if record.get("coordinate_system") != "raw_source_pixels":
            raise ValueError("Pixel truth must be in raw source coordinates")
        path = (annotations_path.parent / record["mask"]).resolve()
        if record.get("mask_sha256") and file_sha256(path) != record["mask_sha256"]:
            raise ValueError("Approved pixel mask changed after review")
        truth = read_class_mask(path)
        if truth.shape != (item["height"],item["width"]):
            raise ValueError("Truth resolution differs from raw source")
        pred_path = config_path.parent / "predictions" / vid / "pred_masks" / f"{index:06d}.png"
        if same_baseline and record.get("a0_prediction_sha256") and file_sha256(pred_path) != record["a0_prediction_sha256"]:
            raise ValueError("Baseline prediction changed since annotation freeze")
        with Image.open(pred_path) as image:
            prediction = np.asarray(image.convert("L"))
        prediction = cv2.resize(prediction,(item["width"],item["height"]),interpolation=cv2.INTER_NEAREST)
        if record.get("fov_mask"):
            with Image.open(annotations_path.parent / record["fov_mask"]) as image:
                valid_fov = np.asarray(image.convert("L")) > 0
            if valid_fov.shape != truth.shape:
                raise ValueError("FOV resolution mismatch")
            truth = np.where(valid_fov,truth,255).astype(np.uint8)
        valid = truth != 255
        if not valid.any():
            raise ValueError("Approved frame has no valid annotated pixels")
        cm = confusion_for_valid_pixels(prediction,truth)
        matrix += cm
        presence = record.get("stone_presence", "unknown")
        if presence not in {"unknown", "present", "absent"} or (presence == "present" and not (truth == 1).any()):
            raise ValueError("Reviewed presence contradicts pixel truth")
        if presence == "absent":
            if (truth == 1).any():
                raise ValueError("Absent-stone annotation contradicts pixel truth")
            negatives += 1
            fp = int(((prediction == 1) & valid).sum())
            negative_fp += int(fp > 0)
            negative_area.append(fp/int(valid.sum()))
        rows.append({"video_id":vid,"frame_index":index,"scores":scores_from_confusion(cm),"confusion_matrix":cm.tolist(),
                     "valid_pixels":int(valid.sum()),"stone_presence":presence})
    result = {"approved_frames":len(rows),"ground_truth_available":True,"coordinate_system":"raw_source_pixels",
              "metrics_scope":"approved frames only, not unannotated intervals or whole-case accuracy",
              "scores":scores_from_confusion(matrix),"confusion_matrix":matrix.tolist(),
              "confirmed_stone_negative_frames":negatives,"negative_frames_with_any_stone_prediction":negative_fp,
              "negative_frame_false_positive_rate":negative_fp/negatives if negatives else None,
              "negative_mean_false_positive_area_fraction":float(np.mean(negative_area)) if negatives else None,
              "temporal_and_instance_accuracy":None,"frames":rows,
              "annotation_sha256":file_sha256(annotations_path),"config_sha256":file_sha256(config_path)}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--annotations",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    evaluate(args.config,args.annotations,args.output)
    print(args.output)


if __name__ == "__main__":
    main()
