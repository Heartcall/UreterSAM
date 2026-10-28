import random
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from tools.diagnose_pseudo_readiness import ImageOnlyDiagnosis, aggregate, policies, strong_view


class PseudoReadinessTests(unittest.TestCase):
    def test_agreement_and_strong_filters_reject_disagreement_and_low_confidence(self):
        probabilities = torch.tensor([.02, .96, .02]).view(1, 1, 3, 1, 1).repeat(3, 1, 1, 1, 4)
        probabilities[2, 0, :, 0, 1] = torch.tensor([.02, .02, .96])
        probabilities[:, 0, :, 0, 2] = torch.tensor([.2, .6, .2])
        strong = probabilities[0].clone()
        strong[0, :, 0, 3] = torch.tensor([.96, .02, .02])
        output, *_ = policies(probabilities, strong)
        self.assertEqual(output["anchor"][1].flatten().tolist(), [True, True, False, True])
        self.assertEqual(output["consensus"][1].flatten().tolist(), [True, False, False, True])
        self.assertEqual(output["consensus_strong"][1].flatten().tolist(), [True, False, False, False])
        with self.assertRaises(ValueError): policies(probabilities[:2], strong)

    def test_reserve_masks_never_read_and_photometric_rng_is_local(self):
        parent = Path("runs/test_tmp").resolve(); parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            root = Path(temp)
            image = Image.fromarray(np.full((16, 16, 3), 100, dtype=np.uint8))
            image.save(root / "reserve.png")
            rows = [{"image": "reserve.png", "mask": "must_not_be_read.png", "diagnostic_split": "reserve"}]
            item = ImageOnlyDiagnosis(root, rows, 32, 42)[0]
            self.assertTrue(torch.all(item["target"] == 255))
            self.assertEqual(item["weak"].shape, (3, 32, 32))
            state = random.getstate()
            first = np.asarray(strong_view(image, 3, 42))
            self.assertEqual(state, random.getstate())
            self.assertTrue(np.array_equal(first, np.asarray(strong_view(image, 3, 42))))

    def test_global_source_agreement_is_not_confidence_or_mean_image_precision(self):
        def record(matrix, gt):
            return {"pixels": 4, "policies": {"consensus_strong": {"accepted": [0, 2, 0],
                "raw": [0, 4, 0], "label_agreement_matrix": matrix}}, "source_label_pixels": gt}
        records = [record([[0, 1, 0], [0, 1, 0], [0, 0, 0]], [3, 1, 0]),
                   record([[0, 0, 0], [0, 2, 0], [0, 0, 0]], [0, 4, 0])]
        result = aggregate(records, "consensus_strong")
        self.assertEqual(result["classes"]["1"]["source_label_agreement_precision"], .75)
        self.assertEqual(result["classes"]["1"]["source_label_positive_recall"], .6)
        self.assertIsNone(result["classes"]["2"]["source_label_agreement_precision"])


if __name__ == "__main__": unittest.main()
