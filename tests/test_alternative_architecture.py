import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from ureter_semis.inference import SegmentationPredictor
from ureter_semis.model import build_model
from ureter_semis.supervised_trainer import train_supervised
from test_training_reproducibility import make_fixture


class AlternativeArchitectureTests(unittest.TestCase):
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

    def test_full_resume_and_architecture_checkpoint_inference(self):
        manifest = make_fixture(self.root); m = json.loads(manifest.read_text())
        m["train"]["unlabeled"] = []; manifest.write_text(json.dumps(m))
        args = dict(manifest_path=str(manifest), epochs=2, size=32, batch_size=2,
                    device_name="cpu", pretrained=False, seed=17, architecture="segformer_mit_b0")
        train_supervised(**args, output_dir=str(self.root / "full"))
        train_supervised(**args, output_dir=str(self.root / "staged"), stop_after_epoch=1)
        train_supervised(**args, output_dir=str(self.root / "staged"), resume=str(self.root / "staged/last.pt"))
        full = torch.load(self.root / "full/last.pt", weights_only=True)
        staged = torch.load(self.root / "staged/last.pt", weights_only=True)
        self.equal_nested(full, staged)
        predictor = SegmentationPredictor(self.root / "full/best.pt", device="cpu")
        self.assertEqual(predictor.architecture, "segformer_mit_b0")
        prediction = predictor.predict(np.zeros((40, 40, 3), dtype=np.uint8))
        self.assertEqual(prediction.probabilities.shape, (3, 32, 32))
        self.assertTrue(np.isfinite(prediction.probabilities).all())

    def test_local_encoder_load_is_exact_and_missing_metadata_rejected(self):
        original = build_model(pretrained=False, architecture="segformer_mit_b0")
        path = self.root / "encoder.pt"; torch.save(original.encoder.state_dict(), path)
        restored = build_model(pretrained=True, architecture="segformer_mit_b0", encoder_checkpoint=str(path))
        self.equal_nested(original.encoder.state_dict(), restored.encoder.state_dict())
        with self.assertRaisesRegex(ValueError, "requires a local"):
            build_model(pretrained=True, architecture="segformer_mit_b0")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            build_model(pretrained=False, architecture="unknown")


if __name__ == "__main__": unittest.main()
