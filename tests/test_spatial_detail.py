import unittest
import numpy as np

from tools.evaluate_spatial_detail import boundary_counts, boundary_summary


class SpatialDetailTests(unittest.TestCase):
    def test_perfect_shifted_and_missed_boundary(self):
        target = np.zeros((30, 30), dtype=np.uint8); target[10:20, 10:20] = 2
        perfect = boundary_summary([boundary_counts(target, target, 2)])
        self.assertEqual(perfect["f_score"], 1.)
        shifted = np.roll(target, 1, axis=1)
        self.assertEqual(boundary_summary([boundary_counts(shifted, target, 2)])["f_score"], 1.)
        missing = boundary_summary([boundary_counts(np.zeros_like(target), target, 2)])
        self.assertEqual(missing["recall"], 0.); self.assertEqual(missing["f_score"], 0.)

    def test_unknown_margin_and_empty_target_not_false_perfect(self):
        target = np.zeros((30, 30), dtype=np.uint8); target[:, :10] = 255
        prediction = np.zeros_like(target); prediction[8:20, :8] = 2
        counts = boundary_counts(prediction, target, 2)
        self.assertEqual(counts["predicted_boundary"], 0)
        self.assertIsNone(boundary_summary([counts])["f_score"])


if __name__ == "__main__": unittest.main()
