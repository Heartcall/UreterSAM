import unittest
import numpy as np

from tools.diagnose_fiber_false_positives import fiber_errors, summarize


class FiberErrorDiagnosisTests(unittest.TestCase):
    def test_partition_near_far_other_class_and_ignore(self):
        target = np.zeros((12, 12), dtype=np.uint8)
        target[5, 5] = 2
        target[5, 6] = 1
        target[11, 11] = 255
        prediction = np.zeros_like(target)
        prediction[5, 5:7] = 2
        prediction[0, 0] = prediction[11, 11] = 2
        result = fiber_errors(target, prediction)
        self.assertEqual(result, {"tp": 1, "fp": 2, "fn": 0, "fp_over_background": 1,
                                 "fp_over_stone": 1, "fp_within_3px_of_fiber": 1, "fp_farther_than_3px": 1})

    def test_absent_fiber_and_global_counts_not_mean_frame_dice(self):
        target = np.zeros((4, 4), dtype=np.uint8)
        prediction = np.full_like(target, 2)
        absent = fiber_errors(target, prediction)
        self.assertEqual(absent["fp_farther_than_3px"], 16)
        present = fiber_errors(prediction, prediction)
        total = summarize([absent, present])
        self.assertEqual(total["fiber_dice"], 32 / 48)
        self.assertEqual(total["fiber_precision"], .5)
        self.assertEqual(total["fiber_recall"], 1.)


if __name__ == "__main__":
    unittest.main()
