from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from ureter_semis.demo import _preprocess
from ureter_semis.inference import preprocess
from ureter_semis.losses import supervised_multiclass_loss
from ureter_semis.trainer import train_endosemis
from ureter_semis.video_metrics import confusion_for_valid_pixels, prediction_diagnostics, scores_from_confusion
from tools.evaluate_video import propose_clips
from tools.evaluate_video_annotations import evaluate


class BaselineAndLossTests(unittest.TestCase):
    def test_preprocess_matches_original_on_non_square_frame(self):
        frame=np.random.default_rng(42).integers(0,256,(51,77,3),dtype=np.uint8)
        self.assertTrue(torch.equal(preprocess(frame,32,torch.device("cpu")),
                                    _preprocess(frame,32,torch.device("cpu"))))

    def test_ignored_pixels_have_no_loss_or_gradient(self):
        logits=torch.randn(1,3,2,2,requires_grad=True)
        target=torch.tensor([[[1,255],[0,2]]])
        weights=torch.tensor([1.,2.,4.])
        loss=supervised_multiclass_loss(logits,target,weights)
        changed=logits.detach().clone();changed[:,:,0,1]=torch.tensor([100.,-100.,70.])
        self.assertTrue(torch.allclose(loss,supervised_multiclass_loss(changed,target,weights)))
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[:,:,0,1],torch.zeros(1,3)))

    def test_empty_annotation_has_finite_zero_loss(self):
        logits=torch.randn(1,3,2,2,requires_grad=True)
        loss=supervised_multiclass_loss(logits,torch.full((1,2,2),255,dtype=torch.long),torch.ones(3))
        self.assertEqual(float(loss.detach()),0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_real_negative_penalizes_false_stone(self):
        target=torch.zeros((1,2,2),dtype=torch.long)
        good=torch.zeros(1,3,2,2);good[:,0]=8
        bad=torch.zeros(1,3,2,2);bad[:,1]=8
        self.assertLess(supervised_multiclass_loss(good,target,torch.ones(3)),
                        supervised_multiclass_loss(bad,target,torch.ones(3)))

    def test_review_draft_rejected_before_data_or_gpu_training(self):
        parent=Path("runs/test_tmp").resolve();parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as d:
            path=Path(d)/"manifest.json"
            path.write_text(json.dumps({"root":d,"training_ready":False}))
            with self.assertRaisesRegex(ValueError,"unreviewed draft"):
                train_endosemis(str(path),str(Path(d)/"training"))

    def test_confusion_excludes_unknowns(self):
        prediction=np.array([[2,1],[0,2]],np.uint8)
        truth=np.array([[255,1],[0,2]],np.uint8)
        matrix=confusion_for_valid_pixels(prediction,truth)
        self.assertEqual(matrix.sum(),3)
        self.assertEqual(scores_from_confusion(matrix)["fiber"]["dice"],1)
        self.assertIsNone(scores_from_confusion(np.zeros((3,3),int))["fiber"]["dice"])

    def test_diagnostics_and_candidates_do_not_claim_truth(self):
        mask=np.ones((16,16),np.uint8)
        probability=np.zeros((3,16,16),np.float32);probability[1]=1
        row=prediction_diagnostics(mask,probability,np.ones((16,16),bool))
        self.assertEqual(row["stone_fraction_estimated_fov"],1)
        self.assertNotIn("false_positive",row)
        clips=propose_clips([row]*300,30)
        self.assertTrue(clips)
        self.assertTrue(all(not clip["ground_truth_available"] for clip in clips))
        for a,b in zip(clips,clips[1:]):
            self.assertLessEqual(a["end_frame_exclusive"],b["start_frame"])

    def test_pending_pixel_reviews_cannot_create_accuracy_report(self):
        parent=Path("runs/test_tmp").resolve();parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as d:
            root=Path(d)
            (root/"config.json").write_text(json.dumps({"inputs":[]}))
            (root/"reviews.json").write_text(json.dumps({"frames":[{"review_status":"pending"}]}))
            with self.assertRaisesRegex(ValueError,"No approved pixel masks"):
                evaluate(root/"config.json",root/"reviews.json",root/"summary.json")
            self.assertFalse((root/"summary.json").exists())

    def test_approved_negative_metrics_exclude_unknown_pixels(self):
        parent=Path("runs/test_tmp").resolve();parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as d:
            root=Path(d);mask_dir=root/"predictions/v1/pred_masks";mask_dir.mkdir(parents=True)
            Image.fromarray(np.array([[1,0],[0,2]],np.uint8)).save(mask_dir/"000000.png")
            Image.fromarray(np.array([[255,0],[0,0]],np.uint8)).save(root/"truth.png")
            (root/"config.json").write_text(json.dumps({"inputs":[{"video_id":"v1","frames":1,"height":2,"width":2}]}))
            (root/"reviews.json").write_text(json.dumps({"frames":[{"video_id":"v1","frame_index":0,
                "mask":"truth.png","review_status":"approved","reviewer":"test_fixture",
                "stone_presence":"absent","coordinate_system":"raw_source_pixels"}]}))
            result=evaluate(root/"config.json",root/"reviews.json",root/"summary.json")
            self.assertEqual(result["negative_frame_false_positive_rate"],0)
            self.assertEqual(result["scores"]["fiber"]["fp"],1)
            self.assertEqual(result["frames"][0]["valid_pixels"],3)


if __name__ == "__main__":
    unittest.main()
