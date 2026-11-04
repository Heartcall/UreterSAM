from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.convert_dataset import _target_class, prepare_dataset, read_yolo_sample
from tools.prepare_review_dataset import assign_groups, sha256
from tools.promote_review_dataset import promote
from ureter_semis.labels import read_class_mask


class ReviewTests(unittest.TestCase):
    def setUp(self):
        parent = Path("runs/test_tmp").resolve()
        parent.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.tmp.name).resolve()
        assert self.root.is_relative_to(parent)
        self.addCleanup(self.tmp.cleanup)

    def sample(self, labels: str | None, names: dict):
        image, label = self.root / "image.png", self.root / "image.txt"
        Image.new("RGB", (16,16), (100,100,100)).save(image)
        if labels is not None:
            label.write_text(labels, encoding="utf-8")
        return read_yolo_sample(image, label, names, mark_unknown=True)

    def test_spot_preserves_stone_and_fiber_priority(self):
        text = "0 0 0 1 0 1 1 0 1\n1 .2 .2 .8 .2 .8 .8 .2 .8\n2 .4 0 .6 0 .6 1 .4 1"
        mask, meta, aux = self.sample(text, {0:"stone",1:"laser",2:"fiber"})
        self.assertEqual(mask[5,5],1)
        self.assertEqual(mask[5,8],2)
        self.assertTrue(aux["laser_dot"][5,5])
        self.assertTrue(meta["has_fiber_annotation"])

    def test_aliases_do_not_conflate_laser_with_fiber(self):
        self.assertEqual(_target_class("stone-new"),1)
        self.assertEqual(_target_class("laser-fibre"),2)
        self.assertIsNone(_target_class("laser"))
        self.assertIsNone(_target_class("laser_dot"))

    def test_missing_file_is_unknown_not_negative(self):
        mask, meta, _ = self.sample(None, {0:"stone"})
        self.assertTrue((mask == 255).all())
        self.assertFalse(meta["label_file_present"])

    def test_unknown_regions_do_not_erase_known_stone(self):
        mask, meta, _ = self.sample("0 0 0 1 0 1 1 0 1\n1 .2 .2 .8 .2 .8 .8 .2 .8", {0:"0",1:"stone"})
        self.assertEqual(mask[0,0],255)
        self.assertEqual(mask[8,8],1)
        self.assertIn("unsupported_class:0",meta["review_flags"])

    def test_invalid_annotation_fails_instead_of_silent_background(self):
        with self.assertRaisesRegex(ValueError,"Invalid polygon"):
            self.sample("0 .1 .1 .2 .2",{0:"stone"})
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            self.sample("0 0 0 nan 0 1 1",{0:"stone"})

    def test_groups_union_video_patient_and_duplicate_evidence(self):
        rows = [dict(sample_id=str(i),group_hint=str(i),image_sha256=f"hash{i}",
                     video_id=v,patient_id=p) for i,(v,p) in enumerate([
                         ("v1","p1"),("v1",None),("v2","p1"),("v3","p3")])]
        rows[3]["image_sha256"] = "hash2"
        assign_groups(rows,42,[.7,.15,.15])
        self.assertEqual(len({r["group_id"] for r in rows}),1)
        self.assertEqual(len({r["proposed_split"] for r in rows}),1)

    def test_multi_source_namespaced_and_refuses_overwrite(self):
        sources=[]
        for idx in range(2):
            source=self.root/f"source{idx}"
            source.mkdir()
            (source/"data.yaml").write_text("names: [stone-new, laser-fibre]\n")
            for split in ("train","valid"):
                (source/split/"images").mkdir(parents=True)
                (source/split/"labels").mkdir()
                Image.new("RGB",(16,16),(idx*20,50 if split=="train" else 70,10)).save(source/split/"images/same.png")
                (source/split/"labels/same.txt").write_text("0 0 0 1 0 1 1")
            sources.append(source)
        target=self.root/"converted"
        manifest=prepare_dataset(sources,target,labeled_fraction=.5)
        data=json.loads(manifest.read_text())
        images=[r["image"] for split in (data["train"]["labeled"],data["train"]["unlabeled"],data["val"]) for r in split]
        self.assertEqual(len(images),len(set(images)))
        self.assertFalse(data["training_ready"])
        with self.assertRaises(FileExistsError):
            prepare_dataset(sources,target)

    def make_reviews(self, pending=False):
        records=[]
        for i in range(60):
            image=self.root/f"img{i}.png"
            mask_path=self.root/f"mask{i}.png"
            Image.new("RGB",(4,4),(i,20,30)).save(image)
            mask=np.zeros((4,4),np.uint8)
            if i%2:
                mask[1:3,1:3]=1
            Image.fromarray(mask).save(mask_path)
            records.append({"sample_id":str(i),"image":image.name,"mask":mask_path.name,
                            "width":4,"height":4,"image_sha256":sha256(image),"mask_sha256":sha256(mask_path),
                            "group_hint":f"family{i}","patient_id":None,"video_id":None})
        draft=self.root/"manifest.json"
        draft.write_text(json.dumps({"schema_version":2,"root":str(self.root),"draft_samples":records,
                                    "training_ready":False,"classes":{"0":"background","1":"stone","2":"fiber"}}))
        review=self.root/"review.csv"
        decisions=[{"sample_id":str(i),"review_status":"pending" if pending else "approved","reviewer":"test_fixture",
                    "source_group_verified":"True","patient_id":f"case{i}","video_id":f"video{i}",
                    "stone_presence_review":"present" if i%2 else "absent","fiber_annotation_completeness":"complete"}
                   for i in range(60)]
        with review.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(decisions[0]));w.writeheader();w.writerows(decisions)
        return draft,review

    def test_pending_reviews_cannot_be_promoted(self):
        draft,review=self.make_reviews(pending=True)
        with self.assertRaisesRegex(ValueError,"No explicitly approved"):
            promote(draft,review,self.root/"ready.json")

    def test_promotion_has_disjoint_groups_and_supervised_negatives(self):
        draft,review=self.make_reviews()
        result=json.loads(promote(draft,review,self.root/"ready.json").read_text())
        self.assertTrue(result["training_ready"])
        train=result["train"]["labeled"]+result["train"]["unlabeled"]
        sets=[{r["group_id"] for r in rows} for rows in (train,result["val"],result["test"])]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertEqual({r["stone_presence_review"] for r in result["train"]["labeled"]},{"present","absent"})
        with self.assertRaises(FileExistsError):
            promote(draft,review,self.root/"ready.json")

    def test_unverified_source_identity_rejected(self):
        draft,review=self.make_reviews()
        text=review.read_text().replace("True","False")
        review.write_text(text)
        with self.assertRaisesRegex(ValueError,"Verified source"):
            promote(draft,review,self.root/"ready.json")

    def test_palette_indices_are_not_luminance_labels(self):
        array=np.array([[0,1],[2,255]],np.uint8)
        image=Image.fromarray(array).convert("P")
        palette=[0]*768;palette[3:6]=[255,0,0];palette[6:9]=[0,255,255]
        image.putpalette(palette)
        path=self.root/"palette.png";image.save(path)
        self.assertTrue(np.array_equal(read_class_mask(path),array))
        Image.new("RGB",(4,4),(255,0,0)).save(self.root/"color.png")
        with self.assertRaises(ValueError):read_class_mask(self.root/"color.png")

    def test_changed_mask_without_explicit_correction_is_rejected(self):
        draft,review=self.make_reviews()
        Image.fromarray(np.ones((4,4),np.uint8)).save(self.root/"mask0.png")
        with self.assertRaisesRegex(ValueError,"Mask changed"):
            promote(draft,review,self.root/"ready.json")


if __name__ == "__main__":
    unittest.main()
