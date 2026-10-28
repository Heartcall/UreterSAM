"""Extract user-identified event clips; preserve 1-based reference provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

from .evaluate_video import render, save_png
from .video_frames import read_forward
from ureter_semis.inference import file_sha256


def build(config_path: str | Path, events_path: str | Path, output_path: str | Path) -> Path:
    config_path, events_path, output = map(lambda p: Path(p).resolve(), (config_path, events_path, output_path))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    annotations = json.loads(events_path.read_text(encoding="utf-8"))
    inputs = {r["video_id"]: r for r in config["inputs"]}
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for case in annotations["cases"]:
        item = inputs[case["video_id"]]
        source = Path(item["path"])
        if file_sha256(source) != item["sha256"]:
            raise ValueError(f"Source video changed: {source}")
        reference = cv2.VideoCapture(case["reference_video"])
        if not reference.isOpened() or int(reference.get(7)) != item["frames"]:
            reference.release()
            raise ValueError("Compare-video frame count differs; frame mapping needs review")
        reference.release()
        fps, n = item["fps"], item["frames"]
        events = []
        for event in case["events"]:
            start, end = event["start_frame"]-1, event["end_frame"]
            if not 0 <= start < end <= n:
                raise ValueError("Event outside source video")
            events.append({**event, "start_frame_zero_based": start, "end_frame_exclusive": end,
                           "start_s": start/fps, "end_s": end/fps,
                           "annotation_type": "user_reported_event_not_pixel_ground_truth"})
        # Merge nearby short events into one contextual clip, split long ones.
        windows = []
        for event in events:
            start, end = event["start_frame_zero_based"], event["end_frame_exclusive"]
            if end-start < 3*fps:
                start, end = max(0, start-round(fps)), min(n, end+round(fps))
            if windows and start <= windows[-1][1]:
                windows[-1] = (windows[-1][0], max(end, windows[-1][1]))
            else:
                windows.append((start, end))
        if not windows:
            length = round(5*fps)
            windows = [(center, min(n, center+length)) for center in (0, n//2, max(0,n-length))]
        split_windows = []
        for start, end in windows:
            parts = max(1, int((end-start)/(8*fps))+1)
            for part in range(parts):
                split_windows.append((start+(end-start)*part//parts, start+(end-start)*(part+1)//parts))
        cap = cv2.VideoCapture(str(source))
        next_index = 0
        try:
            for ci, (start, end) in enumerate(split_windows):
                sid = f"{case['video_id']}_{ci:02d}"
                directory = output / sid
                directory.mkdir()
                writer = cv2.VideoWriter(str(directory / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                        fps, (item["width"]*2, item["height"]))
                if not writer.isOpened():
                    raise RuntimeError("Cannot create review clip")
                frame_samples = {start, (start+end)//2, end-1}
                for event in events:
                    a, b = event["start_frame_zero_based"], event["end_frame_exclusive"]
                    frame_samples.update(idx for idx in (a, (a+b)//2, b-1) if start <= idx < end)
                frame_samples = sorted(frame_samples)
                keyframe_hashes = {}
                try:
                    for idx in range(start, end):
                        frame, next_index = read_forward(cap, next_index, idx)
                        mask_path = config_path.parent / "predictions" / case["video_id"] / "pred_masks" / f"{idx:06d}.png"
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if mask is None:
                            raise FileNotFoundError(mask_path)
                        canvas = render(frame, mask, idx)
                        writer.write(canvas)
                        if idx in frame_samples:
                            keyframe_hashes[str(idx)] = hashlib.sha256(frame.tobytes()).hexdigest()
                            save_png(directory / f"frame_{idx:06d}.png", frame)
                            save_png(directory / f"prediction_{idx:06d}.png", canvas)
                            seed = cv2.resize(mask, (item["width"], item["height"]), interpolation=cv2.INTER_NEAREST)
                            save_png(directory / f"model_seed_{idx:06d}.png", seed)
                finally:
                    writer.release()
                included = [e for e in events if e["start_frame_zero_based"] < end and e["end_frame_exclusive"] > start]
                records.append({"clip_id": sid, "video_id": case["video_id"], "patient_id": case["patient_id"],
                                "case_identity_evidence": case["case_identity_evidence"],
                                "independence_from_training_data": "unverified", "role": "development_review",
                                "start_frame": start, "end_frame_exclusive": end, "fps": fps,
                                "source": str(source), "source_sha256": item["sha256"],
                                "decoded_keyframe_bgr_sha256": keyframe_hashes,
                                "preview": f"{sid}/preview.mp4", "events": included,
                                "pixel_ground_truth_available": False, "review_status": "pixel_annotation_pending",
                                "representative_frame": (start+end)//2,
                                "annotation_frames_zero_based": frame_samples})
        finally:
            cap.release()
    result = {"schema_version": 1, "clips": records, "source_events": str(events_path),
              "coordinate_system": "raw_source_pixels", "frame_indexing": "zero_based_end_exclusive",
              "frame_decoding": "sequential_from_frame_zero_no_random_seek",
              "prediction_coordinate_system": "raw_model_grid_resized_for_preview_only",
              "notes": ["Two distinct cases confirmed by user; independence from training images not established.",
                        "Event labels do not supply pixel masks or certify complete intervals as false positives."]}
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    template = {"schema_version": 1, "frames": [
        {"video_id": row["video_id"], "frame_index": idx,
         "image": f"{row['clip_id']}/frame_{idx:06d}.png", "mask": "", "review_status": "pending",
         "model_seed": f"{row['clip_id']}/model_seed_{idx:06d}.png", "model_seed_is_ground_truth": False,
         "reviewer": "", "stone_presence": "unknown", "coordinate_system": "raw_source_pixels"}
        for row in records for idx in row["annotation_frames_zero_based"]]}
    (output / "pixel_review.template.json").write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    # A compact visual index; raw images remain available for annotation.
    canvas = Image.new("RGB", (800, len(records)*220), "#202020")
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(records):
        middle = row["representative_frame"]
        with Image.open(output / row["clip_id"] / f"prediction_{middle:06d}.png") as image:
            canvas.paste(image.resize((400,200)), (0,i*220))
        label = ", ".join(sorted({label for event in row["events"] for label in event["labels"]})) or "Unreviewed control candidate"
        draw.text((408,i*220+20), row["clip_id"], fill="white")
        draw.text((408,i*220+45), f"reference frames {row['start_frame']+1}-{row['end_frame_exclusive']}", fill="white")
        draw.text((408,i*220+70), label, fill="#ffd080")
        draw.text((408,i*220+95), "Pixel annotation pending", fill="#ffd080")
    canvas.save(output / "index.jpg", quality=92)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(build(args.config, args.events, args.output))


if __name__ == "__main__":
    main()
