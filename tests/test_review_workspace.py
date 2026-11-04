from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

import numpy as np
from PIL import Image

from tools.review_server import create_server
from tools.review_store import ConflictError, ReviewStore, decode_array, encode_array
from tools.evaluate_video_annotations import evaluate
from tools.promote_review_dataset import promote
from ureter_semis.experiment import sha256


def make_workspace_fixture(root: Path) -> tuple[Path, Path]:
    """Synthetic sources only; suitable for UI and storage checks."""
    videos = root / "videos"; videos.mkdir(parents=True)
    frames = []
    mask = np.zeros((32,32),np.uint8); mask[5:12,5:12]=1; mask[5:12,16:23]=1; mask[:,14:16]=2
    image = np.zeros((32,32,3),np.uint8); image[:]=[76,112,143]
    for index in (0,1):
        Image.fromarray(image).save(videos/f"image_{index}.png")
        Image.fromarray(mask).save(videos/f"seed_{index}.png")
        frames.append({"video_id":"synthetic","frame_index":index,"image":f"image_{index}.png","model_seed":f"seed_{index}.png"})
    (videos/"preview.mp4").write_bytes(b"synthetic-test-placeholder")
    (videos/"manifest.json").write_text(json.dumps({"frame_decoding":"sequential_from_frame_zero_no_random_seek",
        "clips":[{"clip_id":"image_0.png","patient_id":"synthetic_case","video_id":"synthetic","preview":"preview.mp4","events":[]},
                 {"clip_id":"image_1.png","patient_id":"synthetic_case","video_id":"synthetic","preview":"preview.mp4","events":[]}]}))
    template=videos/"template.json";template.write_text(json.dumps({"frames":frames}))
    data=root/"data";data.mkdir();Image.fromarray(image).save(data/"image.png");Image.fromarray(mask).save(data/"mask.png")
    rows=[]
    for i in range(3):
        rows.append({"sample_id":f"sample{i}","image":"image.png","mask":"mask.png","source_image":f"source{i}.png",
            "image_sha256":sha256(data/"image.png"),"mask_sha256":sha256(data/"mask.png"),"width":32,"height":32,
            "review_flags":["unsupported_class:0"] if i==2 else ["laser_spot_without_fiber_annotation"],
            "has_stone_annotation":i==0,"group_hint":"unverified_name_hint"})
    dataset=root/"draft.json";dataset.write_text(json.dumps({"schema_version":2,"root":str(data),"draft_samples":rows}))
    return template,dataset


class ReviewWorkspaceTests(unittest.TestCase):
    def setUp(self):
        parent=Path("runs/test_tmp").resolve();parent.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=parent);self.root=Path(self.temp.name)
        self.template,self.dataset=make_workspace_fixture(self.root)
        self.store=ReviewStore(self.template,self.dataset,self.root/"session")

    def tearDown(self):
        self.temp.cleanup()

    def payload(self,key="video:synthetic:0"):
        item=self.store.item(key)
        return {**item,"review_status":"pending","tracks":[],"reviewer":"qa_synthetic_fixture"}

    def test_pending_revisions_are_immutable_and_exported_without_truth(self):
        source=self.root/"videos/seed_0.png";before=sha256(source)
        payload=self.payload();mask=decode_array(payload["mask_base64"],32,32,"u1");mask[0,0]=255
        payload["mask_base64"]=encode_array(mask)
        result=self.store.save(payload["id"],payload)
        first=self.store.state["decisions"][payload["id"]]["mask"]
        self.assertEqual(result["revision"],1);self.assertEqual(sha256(source),before)
        with self.assertRaises(ConflictError):self.store.save(payload["id"],payload)
        payload["revision"]=1;self.store.save(payload["id"],payload)
        self.assertTrue((self.store.directory/first).exists())
        recovered=ReviewStore(self.template,self.dataset,self.store.directory)
        self.assertEqual(recovered.item(payload["id"])["revision"],2)
        exported=json.loads((self.store.directory/"pixel_reviews.json").read_text())
        self.assertTrue(all(r["review_status"]=="pending" for r in exported["frames"]))

    def test_dataset_cannot_be_approved_from_filename_hint(self):
        payload=self.payload("dataset:sample0");payload.update(review_status="approved",stone_presence="present",fiber_completeness="complete")
        with self.assertRaisesRegex(ValueError,"核实来源"):self.store.save(payload["id"],payload)
        payload.update(source_verified=True,source_video_id="verified_synthetic_video")
        self.store.save(payload["id"],payload)
        with (self.store.directory/"dataset_reviews.csv").open(encoding="utf-8-sig",newline="") as handle:
            row=next(csv.DictReader(handle))
        self.assertEqual(row["review_status"],"approved");self.assertEqual(row["source_group_verified"],"True")
        self.assertTrue((self.store.directory/row["corrected_mask"]).exists())

    def test_label_review_can_finish_without_inventing_source_or_training_approval(self):
        payload=self.payload("dataset:sample1")
        payload.update(pixel_review_status="approved",stone_presence="present",fiber_completeness="complete")
        self.store.save(payload["id"],payload)
        item=self.store.item(payload["id"])
        self.assertEqual(item["pixel_review_status"],"approved")
        self.assertEqual(item["review_status"],"pending")
        self.assertFalse(item["source_verified"])
        self.assertEqual(len(self.store.list_items("dataset","negative_pending")),1)
        with self.assertRaisesRegex(ValueError,"No explicitly approved"):
            promote(self.dataset,self.store.directory/"dataset_reviews.csv",self.root/"reviewed.json")
        self.assertFalse((self.root/"reviewed.json").exists())

    def test_approved_video_masks_feed_existing_evaluator(self):
        payload=self.payload();payload.update(review_status="approved",stone_presence="present")
        self.store.save(payload["id"],payload)
        baseline=self.root/"baseline";pred=baseline/"predictions/synthetic/pred_masks";pred.mkdir(parents=True)
        Image.open(self.root/"videos/seed_0.png").save(pred/"000000.png")
        config=baseline/"config.json";config.write_text(json.dumps({"inputs":[{"video_id":"synthetic","frames":2,"height":32,"width":32}]}))
        result=evaluate(config,self.store.directory/"pixel_reviews.json",self.root/"metrics.json")
        self.assertEqual(result["approved_frames"],1);self.assertEqual(result["scores"]["stone"]["dice"],1)

    def test_partial_occlusion_allows_two_visible_parts_of_one_identity(self):
        payload=self.payload();mask=decode_array(payload["mask_base64"],32,32,"u1")
        instances=np.where(mask==1,7,0).astype("<u2")
        payload.update(review_status="approved",instance_status="approved",stone_presence="present",
            tracks=[{"track_id":7,"visibility":"partial_occlusion","parent_ids":[]}],instances_base64=encode_array(instances))
        self.store.save(payload["id"],payload)
        payload=self.payload("video:synthetic:1");mask[:]=2
        payload.update(mask_base64=encode_array(mask),review_status="approved",instance_status="approved",
            tracks=[{"track_id":7,"visibility":"occluded","parent_ids":[]}])
        self.store.save(payload["id"],payload)
        self.assertEqual(self.store.state["decisions"][payload["id"]]["tracks"][0]["track_id"],7)

    def test_bad_instances_and_parent_cycles_are_rejected(self):
        payload=self.payload();instances=np.ones((32,32),dtype="<u2")
        payload.update(instances_base64=encode_array(instances))
        with self.assertRaisesRegex(ValueError,"只能属于结石"):self.store.save(payload["id"],payload)
        payload=self.payload();payload.update(tracks=[{"track_id":1,"visibility":"lost","parent_ids":[2]},
                                                     {"track_id":2,"visibility":"lost","parent_ids":[1]}])
        with self.assertRaisesRegex(ValueError,"循环"):self.store.save(payload["id"],payload)
        payload=self.payload();payload.update(review_status="approved",instance_status="approved")
        with self.assertRaisesRegex(ValueError,"每个可见结石"):self.store.save(payload["id"],payload)

    def test_reviewer_presence_dimensions_and_source_integrity(self):
        payload=self.payload();payload.update(review_status="approved",reviewer="")
        with self.assertRaisesRegex(ValueError,"审核人"):self.store.save(payload["id"],payload)
        payload=self.payload();payload["stone_presence"]="absent"
        with self.assertRaisesRegex(ValueError,"矛盾"):self.store.save(payload["id"],payload)
        payload=self.payload();payload["mask_base64"]=encode_array(np.zeros((1,1),np.uint8))
        with self.assertRaisesRegex(ValueError,"像素数量"):self.store.save(payload["id"],payload)
        Image.new("RGB",(32,32)).save(self.root/"videos/image_0.png")
        with self.assertRaisesRegex(ValueError,"原图或预标注已变化"):self.store.item(payload["id"])

    def test_failed_commit_does_not_change_active_revision(self):
        payload=self.payload()
        from tools import review_store
        actual=review_store.atomic_json
        def fail_session(path,value):
            if path==self.store.state_path:raise OSError("simulated failure")
            actual(path,value)
        with patch("tools.review_store.atomic_json",fail_session),self.assertRaises(OSError):
            self.store.save(payload["id"],payload)
        self.assertEqual(self.store.state["decisions"],{})

    def test_saved_revision_tampering_and_all_unknown_approval_rejected(self):
        payload=self.payload();payload["mask_base64"]=encode_array(np.full((32,32),255,np.uint8))
        self.store.save(payload["id"],payload)
        payload["revision"]=1;payload["review_status"]="approved"
        with self.assertRaisesRegex(ValueError,"有效像素"):self.store.save(payload["id"],payload)
        decision=self.store.state["decisions"][payload["id"]]
        Image.new("L",(32,32)).save(self.store.directory/decision["mask"])
        with self.assertRaisesRegex(ValueError,"外部修改"):self.store.item(payload["id"])

    def test_filters_and_loopback_http(self):
        self.assertEqual(len(self.store.list_items("dataset","negative")),2)
        self.assertEqual(len(self.store.list_items("dataset","unknown")),1)
        self.assertEqual(len(self.store.list_items("dataset","spot")),2)
        server=create_server(self.store,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        url=f"http://127.0.0.1:{server.server_port}"
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(url+"/") as response:
                self.assertIn('UreterSAM 标注复核'.encode(),response.read())
                self.assertIn("frame-ancestors",response.headers["Content-Security-Policy"])
            with opener.open(url+"/api/items") as response:
                self.assertEqual(len(json.load(response)["items"]),2)
            request=urllib.request.Request(url+"/api/save",data=json.dumps(self.payload()).encode(),headers={"Content-Type":"application/json","Origin":"http://example.invalid"})
            with self.assertRaises(urllib.error.HTTPError) as error:opener.open(request)
            self.assertEqual(error.exception.code,403)
            with self.assertRaises(urllib.error.HTTPError):opener.open(url+"/../session.json")
        finally:
            server.shutdown();server.server_close();thread.join()

    def test_context_stream_is_jpeg_without_browser_codec_dependency(self):
        class Capture:
            def __init__(self,*_args):self.remaining=2;self.released=False
            def read(self):
                self.remaining-=1
                return self.remaining>=0,np.zeros((8,8,3),np.uint8)
            def get(self,_key):return 120
            def release(self):self.released=True
        server=create_server(self.store,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with patch("tools.review_server.cv2.VideoCapture",Capture):
                with opener.open(f"http://127.0.0.1:{server.server_port}/context?id=video%3Asynthetic%3A0") as response:
                    self.assertIn("multipart/x-mixed-replace",response.headers["Content-Type"])
                    content=response.read()
                    self.assertEqual(content.count(b"Content-Type: image/jpeg"),2)
                    self.assertTrue(content.endswith(b"--frame--\r\n"))
        finally:
            server.shutdown();server.server_close();thread.join()


if __name__=="__main__":unittest.main()
