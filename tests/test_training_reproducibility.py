from __future__ import annotations

import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ureter_semis.data import UreterDataset
from ureter_semis.experiment import capture_rng, restore_rng, seed_training, seed_worker
from ureter_semis.metrics import confusion_matrix, metrics_from_confusion
from ureter_semis.losses import supervised_multiclass_loss, masked_cross_entropy
from ureter_semis.trainer import train_endosemis, validate
from tools.compare_checkpoints import compare_checkpoints
from tools.train_experiment import run as run_experiment


class TinyModel(nn.Module):
    """Exercise batch-norm, MC dropout, features, optimizer and scheduler state."""
    def __init__(self, **_kwargs):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4),
                                  nn.ReLU(), nn.Dropout2d(0.1))
        self.head = nn.Conv2d(4, 3, 1)

    def forward(self, images, return_features=False):
        features = self.body(images)
        logits = self.head(features)
        return (logits, [features]) if return_features else logits


class RandomWorkerDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return torch.tensor([index, random.random(), np.random.random(), torch.rand(()).item()])


def make_fixture(root: Path, size: int = 32) -> Path:
    rows = []
    rng = np.random.default_rng(921)
    for i in range(10):
        image = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        if i % 2:
            mask[3:size//2, 3:size//2] = 1
        mask[-5:-2, 4:-4] = 2
        mask[:2, :] = 255
        Image.fromarray(image).save(root / f"image_{i}.png")
        Image.fromarray(mask).save(root / f"mask_{i}.png")
        rows.append({"image": f"image_{i}.png", "mask": f"mask_{i}.png",
                     "stone_presence_review": "present" if i % 2 else "absent"})
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"root": str(root), "training_ready": True,
        "role": "synthetic_engineering_test_not_medical_data",
        "train": {"labeled": rows[:4], "unlabeled": rows[4:10]}, "val": rows[:3], "test": []}), encoding="utf-8")
    return manifest


class TrainingReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        parent = Path("runs/test_tmp").resolve()
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()
        torch.set_num_threads(self.old_threads)

    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_epoch_resume_matches_uninterrupted_and_preserves_earlier_best(self):
        manifest = make_fixture(self.root)
        args = dict(manifest_path=str(manifest), epochs=2, size=32, batch_size=2,
                    pretrained=False, device_name="cpu", seed=91)
        full, staged, fork = (self.root / name for name in ("full", "staged", "fork"))
        # Stable metric deliberately makes epoch 1 remain best after resuming.
        def fixed_validation(*_args):
            return {"mean_dice": 0.5, "stone_negative_images_with_fp": 0}
        with patch("ureter_semis.trainer.build_model", TinyModel), patch("ureter_semis.trainer.validate", fixed_validation):
            train_endosemis(**args, output_dir=str(full))
            train_endosemis(**args, output_dir=str(staged), stop_after_epoch=1)
            train_endosemis(**args, output_dir=str(fork), resume=str(staged / "last.pt"))
        a = torch.load(full / "last.pt", weights_only=True)
        b = torch.load(fork / "last.pt", weights_only=True)
        self.assert_nested_equal(a, b)
        self.assert_nested_equal(torch.load(full / "best.pt", weights_only=True),
                                 torch.load(fork / "best.pt", weights_only=True))
        self.assertEqual(b["training_state"]["best_epoch"], 1)
        self.assertEqual(b["history"][1]["learning_rates"], [5.05e-5, 5.05e-5])
        self.assertEqual(json.loads((fork / "status.json").read_text())["state"], "completed")
        with patch("ureter_semis.trainer.build_model", TinyModel), patch("ureter_semis.trainer.validate", fixed_validation):
            train_endosemis(**args, output_dir=str(staged), resume=str(staged / "last.pt"))
        self.assert_nested_equal(a, torch.load(staged / "last.pt", weights_only=True))
        with self.assertRaises(FileExistsError):
            train_endosemis(**args, output_dir=str(full))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            train_endosemis(**{**args, "seed": 92}, output_dir=str(self.root / "wrong_seed"),
                            resume=str(staged / "last.pt"))
        changed = self.root / "image_0.png"
        Image.fromarray(np.zeros((32,32,3),np.uint8)).save(changed)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            train_endosemis(**args, output_dir=str(self.root / "changed"), resume=str(staged / "last.pt"))
        self.assertFalse((self.root / "changed").exists())

    def test_rng_state_restores_all_cpu_sources(self):
        seed_training(73)
        state = capture_rng()
        expected = (random.random(), np.random.random(), torch.rand(5))
        restore_rng(state)
        actual = (random.random(), np.random.random(), torch.rand(5))
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_weighted_sampling_remains_exact_across_epoch_resume(self):
        manifest=make_fixture(self.root)
        data=json.loads(manifest.read_text())
        for i,row in enumerate(data['train']['labeled']):row['sampling_weight']=1 if i else 5
        manifest.write_text(json.dumps(data))
        args=dict(manifest_path=str(manifest),epochs=2,size=32,batch_size=2,pretrained=False,device_name='cpu',seed=63)
        def fixed(*_args):return {'mean_dice':.4,'stone_negative_images_with_fp':0}
        with patch('ureter_semis.trainer.build_model',TinyModel),patch('ureter_semis.trainer.validate',fixed):
            train_endosemis(**args,output_dir=str(self.root/'full'))
            train_endosemis(**args,output_dir=str(self.root/'staged'),stop_after_epoch=1)
            train_endosemis(**args,output_dir=str(self.root/'staged'),resume=str(self.root/'staged/last.pt'))
        self.assert_nested_equal(torch.load(self.root/'full/last.pt',weights_only=True),torch.load(self.root/'staged/last.pt',weights_only=True))

    def test_deterministic_ce_matches_weighted_ce_value_and_gradient(self):
        logits = torch.randn(2, 3, 4, 5, requires_grad=True)
        target = torch.randint(0, 3, (2,4,5)); target[0,0,0] = 255
        weights = torch.tensor([1., 2., 4.])
        reference = torch.nn.functional.cross_entropy(logits, target, weight=weights, ignore_index=255)
        actual = supervised_multiclass_loss(logits, target, weights, dice_weight=0)
        self.assertTrue(torch.allclose(reference, actual))
        expected_gradient = torch.autograd.grad(reference, logits)[0]
        actual_gradient = torch.autograd.grad(actual, logits)[0]
        self.assertTrue(torch.allclose(expected_gradient, actual_gradient, atol=1e-7))
        pseudo = torch.randint(0, 3, (2,4,5)); reliable = torch.rand(2,4,5) > 0.5
        expected = (torch.nn.functional.cross_entropy(logits,pseudo,reduction="none") * reliable).sum() / reliable.sum()
        self.assertTrue(torch.allclose(expected, masked_cross_entropy(logits,pseudo,reliable)))

    def test_spawned_worker_restarts_with_restored_generator(self):
        generator = torch.Generator().manual_seed(33)
        loader = DataLoader(RandomWorkerDataset(), batch_size=2, shuffle=True, num_workers=1,
                            worker_init_fn=seed_worker, generator=generator, multiprocessing_context="spawn")
        state = generator.get_state()
        first = torch.cat(list(loader))
        generator.set_state(state)
        second = torch.cat(list(loader))
        self.assertTrue(torch.equal(first, second))

    def test_global_metrics_not_batch_means_and_absence_is_undefined(self):
        target = torch.tensor([[[1, 1], [0, 255]], [[0, 0], [0, 0]]])
        prediction = torch.tensor([[[1, 0], [1, 2]], [[0, 1], [0, 0]]])
        whole = confusion_matrix(prediction, target)
        self.assertTrue(torch.equal(whole, sum(confusion_matrix(p, t) for p, t in zip(prediction, target))))
        result = metrics_from_confusion(whole)
        self.assertEqual(result["stone_dice"], 0.4)
        self.assertIsNone(result["fiber_dice"])
        self.assertEqual(result["mean_dice"], 0.4)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            confusion_matrix(torch.tensor([3]), torch.tensor([0]))

    def test_negative_presence_uses_native_grid_and_partial_unknown_not_absence(self):
        make_fixture(self.root, size=64)
        tiny = np.zeros((64,64),np.uint8); tiny[1,1] = 1
        unknown = np.zeros((64,64),np.uint8); unknown[0,0] = 255
        Image.fromarray(tiny).save(self.root / "tiny.png")
        Image.fromarray(unknown).save(self.root / "unknown.png")
        rows = [{"image":"image_0.png", "mask": name} for name in ("tiny.png", "unknown.png")]
        ds = UreterDataset(self.root, rows, 32, "val")
        self.assertEqual(ds[0]["stone_presence"], "present")
        self.assertEqual(ds[1]["stone_presence"], "unknown")

    def test_validation_negative_fp_and_confusion_exclude_unknown(self):
        class Echo(nn.Module):
            def forward(self, images):
                return images
        logits = torch.zeros(2,3,2,2); logits[:,0] = 5
        logits[0,1,0,0] = 10; logits[0,1,0,1] = 10
        loader = [{"image": logits, "mask": torch.tensor([[[0,255],[0,0]], [[1,1],[0,0]]]),
                   "stone_presence": ["absent", "present"]}]
        result = validate((Echo(), Echo()), loader, torch.device("cpu"))
        self.assertEqual(result["valid_pixels"], 7)
        self.assertEqual(result["stone_fp"], 1)
        self.assertEqual(result["stone_negative_images"], 1)
        self.assertEqual(result["stone_negative_fp_pixels"], 1)
        self.assertEqual(result["stone_negative_fp_image_rate"], 1)
        self.assertEqual(result["stone_negative_fp_pixel_fraction"], 1/3)

    def test_draft_cannot_be_used_for_checkpoint_comparison(self):
        manifest = self.root / "draft.json"
        manifest.write_text(json.dumps({"root":str(self.root), "training_ready":False}))
        with self.assertRaisesRegex(ValueError, "explicitly reviewed"):
            compare_checkpoints(str(manifest), ["a.pt", "b.pt"], str(self.root / "results.json"), "cpu")
        self.assertFalse((self.root / "results.json").exists())

    def test_experiment_resolves_relative_manifest_and_locks_seed_set(self):
        config = self.root / "config.json"
        config.write_text(json.dumps({"schema_version":1, "replication_seeds":[42,123],
                                     "training":{"manifest_path":"manifest.json", "seed":42}}))
        with patch("tools.train_experiment.train_endosemis") as train:
            run_experiment(config, "dummy", seed=123)
            self.assertEqual(train.call_args.kwargs["manifest_path"], str(self.root / "manifest.json"))
            self.assertEqual(train.call_args.kwargs["seed"], 123)
        with self.assertRaisesRegex(ValueError, "frozen replication"):
            run_experiment(config, "dummy", seed=7)


if __name__ == "__main__":
    unittest.main()
