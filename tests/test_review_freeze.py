from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.freeze_video_reviews import freeze
from tools.evaluate_video_annotations import evaluate
from tools.review_store import ReviewStore, encode_array
from test_review_workspace import make_workspace_fixture


class ReviewFreezeTests(unittest.TestCase):
    def setUp(self):
        parent=Path("runs/test_tmp").resolve();parent.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=parent);self.root=Path(self.temp.name)
        self.template,self.dataset=make_workspace_fixture(self.root)
        self.store=ReviewStore(self.template,self.dataset,self.root/"session")
        self.payload={**self.store.item("video:synthetic:0"),"reviewer":"synthetic_fixture",
                      "review_status":"approved","stone_presence":"present"}
        self.store.save(self.payload["id"],self.payload)
        baseline=self.root/"baseline";pred=baseline/"predictions/synthetic/pred_masks";pred.mkdir(parents=True)
        Image.open(self.root/"videos/seed_0.png").save(pred/"000000.png")
        self.config=baseline/"config.json"
        self.config.write_text(json.dumps({"inputs":[{"video_id":"synthetic","sha256":"0"*64,"frames":2,"height":32,"width":32}]}))

    def tearDown(self):self.temp.cleanup()

    def test_snapshot_is_independent_and_does_not_promote_training_or_instances(self):
        output=self.root/"snapshot"
        annotations=freeze(self.store.state_path,self.template,self.config,output)
        snapshot=json.loads(annotations.read_text())
        self.assertEqual(len(snapshot["frames"]),1)
        self.assertFalse(snapshot["training_ready"])
        self.assertEqual(snapshot["frames"][0]["instance_review_status"],"pending")
        self.assertEqual(snapshot["frames"][0]["instance_mask"],"")
        mask_path=output/snapshot["frames"][0]["mask"];before=mask_path.read_bytes()
        self.payload.update(revision=1,mask_base64=encode_array(np.ones((32,32),np.uint8)))
        self.store.save(self.payload["id"],self.payload)
        self.assertEqual(mask_path.read_bytes(),before)
        result=evaluate(self.config,annotations,self.root/"metrics.json")
        self.assertEqual(result["scores"]["stone"]["dice"],1)
        with self.assertRaises(FileExistsError):freeze(self.store.state_path,self.template,self.config,output)

    def test_changed_review_mask_or_baseline_prediction_is_rejected(self):
        annotations=freeze(self.store.state_path,self.template,self.config,self.root/"snapshot")
        Image.new("L",(32,32)).save(self.config.parent/"predictions/synthetic/pred_masks/000000.png")
        with self.assertRaisesRegex(ValueError,"Baseline prediction changed"):
            evaluate(self.config,annotations,self.root/"bad_metrics.json")
        self.assertFalse((self.root/"bad_metrics.json").exists())
        decision=self.store.state["decisions"][self.payload["id"]]
        Image.new("L",(32,32)).save(self.store.directory/decision["mask"])
        with self.assertRaisesRegex(ValueError,"Approved mask changed"):
            freeze(self.store.state_path,self.template,self.config,self.root/"bad_snapshot")
        self.assertFalse((self.root/"bad_snapshot").exists())

    def test_other_model_can_be_evaluated_on_same_frozen_video_but_not_other_inputs(self):
        annotations=freeze(self.store.state_path,self.template,self.config,self.root/"snapshot")
        alternative=self.root/"other_model";pred=alternative/"predictions/synthetic/pred_masks";pred.mkdir(parents=True)
        Image.open(self.root/"videos/seed_0.png").save(pred/"000000.png")
        config=json.loads(self.config.read_text());config["checkpoint"]="different_model.pt"
        path=alternative/"config.json";path.write_text(json.dumps(config))
        result=evaluate(path,annotations,self.root/"other_metrics.json")
        self.assertEqual(result["scores"]["stone"]["dice"],1)
        config["inputs"][0]["sha256"]="1"*64;path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError,"video identity differs"):
            evaluate(path,annotations,self.root/"wrong_video_metrics.json")


if __name__=="__main__":unittest.main()
