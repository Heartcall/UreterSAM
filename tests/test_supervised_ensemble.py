import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tools.package_supervised_ensemble import package
from ureter_semis.experiment import input_identity
from ureter_semis.inference import SegmentationPredictor
from ureter_semis.model import ensemble_logits
from test_training_reproducibility import TinyModel, make_fixture


class SupervisedEnsembleTests(unittest.TestCase):
    def setUp(self):
        parent = Path("runs/test_tmp").resolve(); parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent); self.root = Path(self.temp.name)
        self.manifest = make_fixture(self.root)
        m = json.loads(self.manifest.read_text()); m["train"]["unlabeled"] = []
        self.manifest.write_text(json.dumps(m))
        identity = input_identity(self.manifest, self.root, m)
        self.paths = []
        for seed in (42, 123):
            torch.manual_seed(seed)
            state = {"model_count": 1, "classes": ["background", "stone", "fiber"], "size": 32,
                     "epoch": 1, "model1": TinyModel().state_dict(),
                     "config": {"mode": "supervised_single_network", "seed": seed, "epochs": 1,
                                "pseudo_labels": False, "mutual_learning": False},
                     "run_identity": {"inputs": identity, "source_hashes": {"fixture.py": "same"}},
                     "history": [{"optimizer_update_steps": 2}]}
            path = self.root / f"seed{seed}.pt"; torch.save(state, path); self.paths.append(str(path))

    def tearDown(self):
        self.temp.cleanup()

    def test_packaged_predictor_equals_mean_member_logits(self):
        output = package(str(self.manifest), self.paths, str(self.root / "dual.pt"))
        with patch("ureter_semis.inference.build_model", TinyModel):
            single = [SegmentationPredictor(p, "cpu") for p in self.paths]
            dual = SegmentationPredictor(output, "cpu")
            images = torch.randn(2, 3, 32, 32)
            with torch.inference_mode():
                expected = (single[0].models[0](images) + single[1].models[0](images)) / 2
                self.assertTrue(torch.equal(expected, ensemble_logits(dual.models, images)))
        state = torch.load(output, weights_only=True)
        self.assertEqual(state["model_count"], 2)
        self.assertEqual([m["seed"] for m in state["ensemble_members"]], [42, 123])
        self.assertEqual(sum(m["optimizer_updates"] for m in state["ensemble_members"]), 4)
        self.assertNotIn("training_state", state)
        with self.assertRaises(FileExistsError):
            package(str(self.manifest), self.paths, str(output))

    def test_wrong_inputs_source_or_budget_rejected(self):
        original = torch.load(self.paths[1], weights_only=True)
        for change in ("inputs", "source", "seed", "epoch"):
            state = torch.load(self.paths[1], weights_only=True)
            if change == "inputs": state["run_identity"]["inputs"]["manifest_sha256"] = "wrong"
            if change == "source": state["run_identity"]["source_hashes"] = {"fixture.py": "wrong"}
            if change == "seed": state["config"]["seed"] = 42
            if change == "epoch": state["epoch"] = 2
            altered = self.root / f"{change}.pt"; torch.save(state, altered)
            with self.subTest(change=change), self.assertRaises(ValueError):
                package(str(self.manifest), [self.paths[0], str(altered)], str(self.root / "invalid.pt"))
        self.assertEqual(torch.load(self.paths[1], weights_only=True)["epoch"], original["epoch"])

    def test_same_member_or_active_unlabeled_pool_rejected(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            package(str(self.manifest), [self.paths[0]] * 2, str(self.root / "same.pt"))
        m = json.loads(self.manifest.read_text())
        m["train"]["unlabeled"] = [{"image": "reserved_must_not_be_read.png"}]
        self.manifest.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, "active unlabeled"):
            package(str(self.manifest), self.paths, str(self.root / "unlabeled.pt"))


if __name__ == "__main__":
    unittest.main()
