import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from ureter_semis.native_validation import NativeValidationDataset, collate_native, native_argmax, validate_native
from ureter_semis.supervised_trainer import train_supervised
from test_training_reproducibility import TinyModel, make_fixture


class NativeValidationTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads(); torch.set_num_threads(1)
        parent = Path("runs/test_tmp").resolve(); parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup(); torch.set_num_threads(self.threads)

    def equal_nested(self, a, b):
        if isinstance(a, torch.Tensor): self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a: self.equal_nested(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b): self.equal_nested(x, y)
        else: self.assertEqual(a, b)

    def test_resize_probabilities_before_argmax(self):
        fg = torch.tensor([[.51, .01], [.01, .01]])
        probability = torch.stack((1 - fg, fg, torch.zeros_like(fg)))
        self.assertEqual(int(native_argmax(probability, (1, 1))), 0)
        self.assertEqual(int(probability.argmax(0)[0, 0]), 1)
        with self.assertRaises(ValueError): native_argmax(probability[:2], (1, 1))

    def test_native_gt_shape_ignore_and_negative_area(self):
        manifest = make_fixture(self.root, size=40)
        m = json.loads(manifest.read_text())
        dataset = NativeValidationDataset(self.root, m["val"], 32)
        self.assertEqual(dataset[0]["image"].shape, (3, 32, 32))
        self.assertEqual(dataset[0]["target"].shape, (40, 40))
        class ConstantStone(torch.nn.Module):
            def forward(self, x):
                result = torch.zeros(len(x), 3, 32, 32); result[:, 1] = 10
                return result
        metrics = validate_native((ConstantStone(),), DataLoader(dataset, batch_size=2, collate_fn=collate_native), torch.device("cpu"))
        self.assertEqual(metrics["valid_pixels"], 3 * 38 * 40)
        self.assertEqual(metrics["stone_negative_images"], 2)
        self.assertEqual(metrics["stone_negative_fp_pixel_fraction"], 1.)

    def test_native_training_resume_is_exact(self):
        manifest = make_fixture(self.root, size=40); m = json.loads(manifest.read_text())
        m["train"]["unlabeled"] = []; manifest.write_text(json.dumps(m))
        args = dict(manifest_path=str(manifest), epochs=2, size=32, batch_size=2, pretrained=False,
                    device_name="cpu", seed=17, validation_grid="native_probability")
        with patch("ureter_semis.supervised_trainer.build_model", TinyModel):
            train_supervised(**args, output_dir=str(self.root / "full"))
            train_supervised(**args, output_dir=str(self.root / "staged"), stop_after_epoch=1)
            train_supervised(**args, output_dir=str(self.root / "staged"), resume=str(self.root / "staged/last.pt"))
        self.equal_nested(torch.load(self.root / "full/last.pt", weights_only=True),
                          torch.load(self.root / "staged/last.pt", weights_only=True))


if __name__ == "__main__": unittest.main()
