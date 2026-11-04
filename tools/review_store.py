"""Revisioned local annotation store; source data are read-only."""
from __future__ import annotations

import base64
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import threading
import uuid

import numpy as np
from PIL import Image

from ureter_semis.experiment import atomic_json, sha256
from ureter_semis.labels import read_class_mask


class ConflictError(ValueError):
    pass


def decode_array(value: str, width: int, height: int, dtype: str) -> np.ndarray:
    raw = base64.b64decode(value, validate=True)
    if len(raw) != width * height * np.dtype(dtype).itemsize:
        raise ValueError("像素数量与原图尺寸不一致")
    return np.frombuffer(raw, dtype=dtype).reshape(height, width).copy()


def encode_array(array: np.ndarray) -> str:
    return base64.b64encode(array.tobytes()).decode("ascii")


class ReviewStore:
    def __init__(self, template: str | Path, dataset: str | Path, session: str | Path):
        self.template, self.dataset, self.directory = map(lambda p: Path(p).resolve(), (template, dataset, session))
        self.lock = threading.RLock()
        self.records = {}
        video = json.loads(self.template.read_text(encoding="utf-8"))
        clips = json.loads((self.template.parent / "manifest.json").read_text(encoding="utf-8"))
        if clips.get("frame_decoding") != "sequential_from_frame_zero_no_random_seek":
            raise ValueError("必须使用顺序解码、已对齐的复核集")
        by_clip = {r["clip_id"]: r for r in clips["clips"]}
        for row in video["frames"]:
            image = (self.template.parent / row["image"]).resolve()
            seed = (self.template.parent / row["model_seed"]).resolve()
            clip = by_clip[Path(row["image"]).parts[0]]
            with Image.open(image) as source:
                width, height = source.size
            key = f"video:{row['video_id']}:{row['frame_index']}"
            self.records[key] = {"id": key, "kind": "video", "title": f"{row['video_id']} · 第 {row['frame_index']+1} 帧",
                "image": image, "seed": seed, "image_sha256": sha256(image), "seed_sha256": sha256(seed),
                "width": width, "height": height, "video_id": row["video_id"], "frame_index": row["frame_index"],
                "preview": self.template.parent / clip["preview"],
                "flags": sorted({label for e in clip["events"] for label in e["labels"]}), "original": row}
        draft = json.loads(self.dataset.read_text(encoding="utf-8"))
        if draft.get("schema_version") != 2 or "draft_samples" not in draft:
            raise ValueError("需要版本 2 的数据审核草稿")
        root = Path(draft["root"])
        if not root.is_absolute():
            root = self.dataset.parent / root
        self.dataset_root = root.resolve()
        for row in draft["draft_samples"]:
            key = f"dataset:{row['sample_id']}"
            self.records[key] = {"id": key, "kind": "dataset", "title": Path(row["source_image"]).name,
                "image": (root / row["image"]).resolve(), "seed": (root / row["mask"]).resolve(),
                "image_sha256": row["image_sha256"], "seed_sha256": row["mask_sha256"],
                "width": row["width"], "height": row["height"], "video_id": None,
                "flags": row["review_flags"], "has_stone": row["has_stone_annotation"], "original": row}
        identity = {"template_sha256": sha256(self.template), "dataset_sha256": sha256(self.dataset)}
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / "session.json"
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state["inputs"] != identity:
                raise ValueError("会话来源发生变化，请使用新的会话目录")
        else:
            if any(self.directory.iterdir()):
                raise FileExistsError("会话目录非空且没有可恢复记录")
            self.state = {"schema_version": 1, "inputs": identity, "decisions": {}}
            atomic_json(self.state_path, self.state)
        self.export()

    def list_items(self, kind: str = "video", filter_name: str = "all") -> list[dict]:
        with self.lock:
            rows = []
            for key, row in self.records.items():
                decision = self.state["decisions"].get(key, {})
                pixel_status = decision.get("pixel_review_status", "approved" if decision.get("review_status") == "approved" else "pending")
                if row["kind"] != kind:
                    continue
                if filter_name == "pending" and decision.get("review_status", "pending") != "pending":
                    continue
                if filter_name == "negative" and (kind != "dataset" or row["has_stone"]):
                    continue
                if filter_name == "negative_pending" and (kind != "dataset" or row["has_stone"] or pixel_status == "approved" or decision.get("review_status") == "rejected"):
                    continue
                if filter_name == "labels_pending" and (pixel_status == "approved" or decision.get("review_status") == "rejected"):
                    continue
                if filter_name == "spot" and "laser_spot_without_fiber_annotation" not in row["flags"]:
                    continue
                if filter_name == "unknown" and not any(flag.startswith("unsupported_class:") for flag in row["flags"]):
                    continue
                rows.append({"id": key, "title": row["title"], "flags": row["flags"],
                             "review_status": decision.get("review_status", "pending"), "pixel_review_status": pixel_status})
            return rows

    def verify_source(self, row: dict) -> None:
        if sha256(row["image"]) != row["image_sha256"] or sha256(row["seed"]) != row["seed_sha256"]:
            raise ValueError("原图或预标注已变化，请停止审核并核对来源")

    def asset(self, key: str, asset: str) -> Path:
        row = self.records[key]
        if asset not in {"image", "preview"} or asset not in row:
            raise ValueError("未知素材")
        if asset == "image":
            self.verify_source(row)
        return Path(row[asset])

    def item(self, key: str) -> dict:
        with self.lock:
            row = self.records[key]
            self.verify_source(row)
            decision = self.state["decisions"].get(key, {})
            mask_path = self.directory / decision["mask"] if decision else row["seed"]
            if decision and (sha256(mask_path) != decision["mask_sha256"] or
                             sha256(self.directory / decision["instances"]) != decision["instances_sha256"]):
                raise ValueError("已保存版本被外部修改，请核对审核记录")
            mask = read_class_mask(mask_path)
            if mask.shape != (row["height"], row["width"]):
                raise ValueError("预标注尺寸错误")
            instances = np.zeros(mask.shape, dtype="<u2")
            if decision.get("instances"):
                with Image.open(self.directory / decision["instances"]) as image:
                    instances = np.asarray(image, dtype="<u2")
            original = row["original"]
            return {"id": key, "kind": row["kind"], "title": row["title"], "width": row["width"], "height": row["height"],
                "revision": decision.get("revision", 0), "review_status": decision.get("review_status", "pending"),
                "pixel_review_status": decision.get("pixel_review_status", "approved" if decision.get("review_status") == "approved" else "pending"),
                "reviewer": decision.get("reviewer", ""), "note": decision.get("note", ""),
                "stone_presence": decision.get("stone_presence", "unknown"),
                "fiber_completeness": decision.get("fiber_completeness", "unknown"),
                "source_verified": decision.get("source_verified", False),
                "patient_id": decision.get("patient_id", ""), "source_video_id": decision.get("source_video_id", ""),
                "instance_status": decision.get("instance_status", "pending"), "tracks": decision.get("tracks", []),
                "flags": row["flags"], "video_id": row["video_id"], "frame_index": row.get("frame_index"),
                "source_hint": original.get("group_hint", "用户已确认两个不同病例；训练集独立性未核实"),
                "mask_base64": encode_array(mask), "instances_base64": encode_array(instances),
                "is_model_seed": not bool(decision), "has_preview": "preview" in row}

    def validate_tracks(self, key: str, tracks: list, instances: np.ndarray, mask: np.ndarray, approved: bool) -> None:
        if len(tracks) > 200 or not isinstance(tracks, list):
            raise ValueError("实例记录格式错误")
        identifiers, parents = set(), {}
        for track in tracks:
            identifier = track["track_id"]
            if type(identifier) is not int or not 1 <= identifier <= 65534 or identifier in identifiers:
                raise ValueError("实例编号必须是互不重复的 1–65534 整数")
            identifiers.add(identifier)
            state = track["visibility"]
            if state not in {"visible", "partial_occlusion", "occluded", "lost", "out_of_view", "split_pending"}:
                raise ValueError("实例可见状态错误")
            ancestors = track.get("parent_ids", [])
            if not isinstance(ancestors, list) or any(type(p) is not int or p < 1 or p > 65534 or p == identifier for p in ancestors):
                raise ValueError("父实例编号错误")
            parents[identifier] = set(ancestors)
            visible = bool((instances == identifier).any())
            if approved and visible != (state in {"visible", "partial_occlusion", "split_pending"}):
                raise ValueError("实例像素与可见状态不一致")
        if np.any((instances > 0) & (mask != 1)):
            raise ValueError("实例像素只能属于结石，不能覆盖光纤或未知区域")
        if approved and (np.any((mask == 1) & (instances == 0)) or not set(np.unique(instances)) - {0} <= identifiers):
            raise ValueError("审核实例前，需为每个可见结石像素分配身份并填写状态")
        # Parent assignments are identities over the video, not connected-component ids.
        for other_key, decision in self.state["decisions"].items():
            if other_key == key or self.records[other_key]["video_id"] != self.records[key]["video_id"]:
                continue
            if decision.get("instance_status") != "approved":
                continue
            for track in decision["tracks"]:
                tid, ps = track["track_id"], set(track.get("parent_ids", []))
                if tid in parents and parents[tid] != ps:
                    raise ValueError("同一视频中的实例父子关系不一致")
                parents[tid] = ps
        def visit(identifier, active):
            if identifier in active:
                raise ValueError("实例父子关系存在循环")
            for parent in parents.get(identifier, set()):
                visit(parent, active | {identifier})
        for identifier in parents:
            visit(identifier, set())

    def save(self, key: str, payload: dict) -> dict:
        with self.lock:
            row = self.records[key]
            self.verify_source(row)
            old = self.state["decisions"].get(key, {})
            if payload.get("revision") != old.get("revision", 0):
                raise ConflictError("该样本已被另一页面修改，请重新载入")
            mask = decode_array(payload["mask_base64"], row["width"], row["height"], "u1")
            instances = decode_array(payload["instances_base64"], row["width"], row["height"], "<u2")
            if not set(np.unique(mask)) <= {0,1,2,255}:
                raise ValueError("掩膜必须包含有效的 0/1/2/255 标签")
            status = payload.get("review_status", "pending")
            presence = payload.get("stone_presence", "unknown")
            reviewer = payload.get("reviewer", "").strip()
            completeness = payload.get("fiber_completeness", "unknown")
            pixel_status = "approved" if status == "approved" else payload.get("pixel_review_status", "pending")
            if pixel_status not in {"pending", "approved"}:
                raise ValueError("标签审核状态错误")
            if status not in {"pending", "approved", "rejected"} or presence not in {"present", "absent", "unknown"}:
                raise ValueError("审核状态错误")
            if completeness not in {"complete", "unknown", "incomplete"}:
                raise ValueError("光纤标注完整性错误")
            if presence == "absent" and (mask == 1).any() or presence == "present" and not (mask == 1).any():
                raise ValueError("结石存在性与像素标注矛盾")
            verified = payload.get("source_verified") is True
            patient = payload.get("patient_id", "").strip()
            video = payload.get("source_video_id", "").strip()
            if pixel_status == "approved":
                if not reviewer or not (mask != 255).any():
                    raise ValueError("标签审核通过时需要审核人和有效像素")
                if row["kind"] == "dataset" and (presence == "unknown" or completeness != "complete"):
                    raise ValueError("确认训练标签前需核验结石存在性和光纤标注完整性")
            if status == "approved":
                if not (mask != 255).any():
                    raise ValueError("审核通过的掩膜必须包含有效像素")
                if not reviewer:
                    raise ValueError("审核通过时必须填写审核人")
                if row["kind"] == "dataset" and (not verified or not (patient or video) or presence == "unknown" or completeness != "complete"):
                    raise ValueError("训练样本通过审核还需核实来源、结石存在性和光纤标注完整性；可先保存待审")
            instance_status = payload.get("instance_status", "pending")
            if instance_status not in {"pending", "approved"}:
                raise ValueError("实例审核状态错误")
            if instance_status == "approved" and (row["kind"] != "video" or status != "approved"):
                raise ValueError("实例审核需要视频关键帧的像素标注先审核通过")
            tracks = payload.get("tracks", [])
            self.validate_tracks(key, tracks, instances, mask, instance_status == "approved")
            revision = old.get("revision", 0) + 1
            folder = Path("revisions") / hashlib.sha256(key.encode()).hexdigest()[:20] / f"{revision:05d}_{uuid.uuid4().hex}"
            (self.directory / folder).mkdir(parents=True)
            Image.fromarray(mask).save(self.directory / folder / "mask.png")
            Image.fromarray(instances).save(self.directory / folder / "instances.png")
            decision = {"revision": revision, "mask": str(folder / "mask.png"), "instances": str(folder / "instances.png"),
                "review_status": status, "pixel_review_status": pixel_status, "reviewer": reviewer, "stone_presence": presence,
                "fiber_completeness": completeness, "source_verified": verified,
                "patient_id": patient, "source_video_id": video, "note": payload.get("note", ""),
                "instance_status": instance_status, "tracks": tracks, "source_image_sha256": row["image_sha256"],
                "model_seed_is_ground_truth": False, "saved_utc": datetime.now(timezone.utc).isoformat(),
                "source_seed_sha256": row["seed_sha256"],
                "mask_sha256": sha256(self.directory / folder / "mask.png"),
                "instances_sha256": sha256(self.directory / folder / "instances.png")}
            atomic_json(self.directory / folder / "decision.json", decision)
            next_state = {**self.state, "decisions": {**self.state["decisions"], key: decision}}
            atomic_json(self.state_path, next_state)
            self.state = next_state
            self.export()
            return {"revision": revision, "review_status": status, "saved": True}

    def export(self) -> None:
        frames, datasets = [], []
        for key, row in self.records.items():
            decision = self.state["decisions"].get(key, {})
            if row["kind"] == "video":
                frames.append({"video_id": row["video_id"], "frame_index": row["frame_index"],
                    "image": str(row["image"]), "mask": decision.get("mask", ""),
                    "coordinate_system": "raw_source_pixels", "review_status": decision.get("review_status", "pending"),
                    "reviewer": decision.get("reviewer", ""), "stone_presence": decision.get("stone_presence", "unknown"),
                    "instance_mask": decision.get("instances", ""), "instance_review_status": decision.get("instance_status", "pending"),
                    "tracks": decision.get("tracks", []), "note": decision.get("note", ""),
                    "mask_sha256": decision.get("mask_sha256"), "instance_mask_sha256": decision.get("instances_sha256"),
                    "source_image_sha256": row["image_sha256"], "saved_utc": decision.get("saved_utc")})
            else:
                datasets.append({"sample_id": row["original"]["sample_id"], "review_status": decision.get("review_status", "pending"),
                    "reviewer": decision.get("reviewer", ""), "stone_presence_review": decision.get("stone_presence", "unknown"),
                    "pixel_review_status": decision.get("pixel_review_status", "approved" if decision.get("review_status") == "approved" else "pending"),
                    "fiber_annotation_completeness": decision.get("fiber_completeness", "unknown"),
                    "source_group_verified": str(decision.get("source_verified", False)),
                    "patient_id": decision.get("patient_id", ""), "video_id": decision.get("source_video_id", ""),
                    "corrected_mask": decision.get("mask", ""), "corrected_mask_sha256": decision.get("mask_sha256", ""),
                    "review_note": decision.get("note", "")})
        atomic_json(self.directory / "pixel_reviews.json", {"schema_version": 1, "frames": frames,
                    "frame_decoding": "sequential_from_frame_zero_no_random_seek"})
        buffer = io.StringIO(newline="")
        if datasets:
            writer = csv.DictWriter(buffer, fieldnames=list(datasets[0]))
            writer.writeheader()
            writer.writerows(datasets)
        temp = self.directory / "dataset_reviews.csv.tmp"
        temp.write_text(buffer.getvalue(), encoding="utf-8-sig")
        temp.replace(self.directory / "dataset_reviews.csv")
