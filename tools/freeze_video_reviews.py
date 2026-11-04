"""Freeze approved video pixels into a versioned, independent-of-session snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from ureter_semis.experiment import atomic_json, sha256
from ureter_semis.labels import read_class_mask


def freeze(session_path: str | Path, template_path: str | Path, config_path: str | Path,
           output_path: str | Path) -> Path:
    session_path, template_path, config_path, output = map(lambda p: Path(p).resolve(),
                                                         (session_path, template_path, config_path, output_path))
    if output.exists():
        raise FileExistsError(f"Refusing to replace a review snapshot: {output}")
    # session.json is the atomic commit point, not the derived mutable exports.
    session_bytes = session_path.read_bytes()
    state = json.loads(session_bytes)
    template = json.loads(template_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if state["inputs"]["template_sha256"] != sha256(template_path):
        raise ValueError("Template differs from the review session")
    clip_manifest = template_path.parent / "manifest.json"
    clips = json.loads(clip_manifest.read_text(encoding="utf-8"))
    if clips.get("frame_decoding") != "sequential_from_frame_zero_no_random_seek":
        raise ValueError("Review frames must have verified decode-order alignment")
    patients = {r["video_id"]: r["patient_id"] for r in clips["clips"]}
    inputs = {r["video_id"]: r for r in config["inputs"]}
    original = {f"video:{r['video_id']}:{r['frame_index']}": r for r in template["frames"]}
    decisions = [(key, row) for key, row in state["decisions"].items()
                 if key.startswith("video:") and row["review_status"] == "approved"]
    if not decisions:
        raise ValueError("No approved pixel masks to freeze")
    prepared = []
    for key, decision in sorted(decisions, key=lambda pair: (original[pair[0]]["video_id"], original[pair[0]]["frame_index"])):
        source = original[key]
        video, index = source["video_id"], source["frame_index"]
        item = inputs[video]
        if type(index) is not int or not 0 <= index < item["frames"]:
            raise ValueError("Review index outside original video")
        if not decision.get("reviewer", "").strip():
            raise ValueError("Approved review needs reviewer identity")
        image = (template_path.parent / source["image"]).resolve()
        seed = (template_path.parent / source["model_seed"]).resolve()
        mask_path = (session_path.parent / decision["mask"]).resolve()
        if sha256(image) != decision["source_image_sha256"] or sha256(seed) != decision["source_seed_sha256"]:
            raise ValueError("Source image or model seed changed since review")
        if sha256(mask_path) != decision["mask_sha256"]:
            raise ValueError("Approved mask changed since review")
        mask, seed_mask = read_class_mask(mask_path), read_class_mask(seed)
        with Image.open(image) as raw:
            if raw.size != (item["width"], item["height"]):
                raise ValueError("Review image dimensions differ from video")
        if mask.shape != (item["height"], item["width"]) or seed_mask.shape != mask.shape or not (mask != 255).any():
            raise ValueError("Invalid reviewed label dimensions or valid region")
        presence = decision.get("stone_presence", "unknown")
        if presence not in {"present", "absent", "unknown"}:
            raise ValueError("Invalid stone presence")
        if (presence == "absent" and (mask == 1).any()) or (presence == "present" and not (mask == 1).any()):
            raise ValueError("Reviewed presence contradicts the mask")
        relative = Path(video) / f"{index:06d}"
        row = {"video_id": video, "patient_id": patients[video], "frame_index": index,
               "image": str(relative / "image.png"), "mask": str(relative / "mask.png"),
               "image_sha256": sha256(image), "mask_sha256": sha256(mask_path),
               "coordinate_system": "raw_source_pixels", "review_status": "approved",
               "reviewer": decision["reviewer"], "stone_presence": presence,
               "fiber_annotation_completeness": decision["fiber_completeness"],
               "instance_review_status": decision.get("instance_status", "pending"), "instance_mask": "",
               "tracks": [], "saved_utc": decision.get("saved_utc"), "note": decision.get("note", ""),
               "annotation_basis": "human_review_of_model_seed",
               "seed_sha256": decision["source_seed_sha256"],
               "changed_pixels_from_seed": int((mask != seed_mask).sum()),
               "unknown_pixels": int((mask == 255).sum()), "source_revision": decision["revision"],
               "a0_prediction_sha256": sha256(config_path.parent / "predictions" / video / "pred_masks" / f"{index:06d}.png")}
        instance_path = None
        if row["instance_review_status"] == "approved":
            instance_path = (session_path.parent / decision["instances"]).resolve()
            if sha256(instance_path) != decision["instances_sha256"]:
                raise ValueError("Approved instance mask changed since review")
            with Image.open(instance_path) as image_file:
                instances = np.asarray(image_file)
            if instances.shape != mask.shape or np.any((instances > 0) & (mask != 1)) or np.any((mask == 1) & (instances == 0)):
                raise ValueError("Approved identities do not match visible stone pixels")
            row.update(instance_mask=str(relative / "instances.png"), instance_mask_sha256=sha256(instance_path),
                       tracks=decision["tracks"])
        prepared.append((row, image, mask_path, instance_path))
    output.mkdir(parents=True)
    for row, image, mask, instance in prepared:
        directory = (output / row["image"]).parent
        directory.mkdir(parents=True)
        shutil.copyfile(image, output / row["image"])
        shutil.copyfile(mask, output / row["mask"])
        if sha256(output / row["image"]) != row["image_sha256"] or sha256(output / row["mask"]) != row["mask_sha256"]:
            raise ValueError("Source changed while snapshot was copied; snapshot is incomplete")
        if instance:
            shutil.copyfile(instance, output / row["instance_mask"])
    (output / "session_at_freeze.json").write_bytes(session_bytes)
    import hashlib
    payload = {"schema_version": 1, "role": "reviewed_development_keyframes_not_training_or_independent_test",
               "source_session": str(session_path), "source_session_sha256": hashlib.sha256(session_bytes).hexdigest(),
               "template_sha256": sha256(template_path), "clip_manifest_sha256": sha256(clip_manifest),
               "baseline_config": str(config_path), "baseline_config_sha256": sha256(config_path),
               "frozen_inputs": [{k: item[k] for k in ("video_id", "sha256", "width", "height", "frames")}
                                 for item in config["inputs"] if item["video_id"] in {row["video_id"] for row, *_ in prepared}],
               "frame_decoding": "sequential_from_frame_zero_no_random_seek",
               "independence_from_training_data": "unverified", "training_ready": False,
               "frames": [row for row, *_ in prepared]}
    atomic_json(output / "pixel_reviews.json", payload)
    return output / "pixel_reviews.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(freeze(args.session, args.template, args.config, args.output))


if __name__ == "__main__":
    main()
