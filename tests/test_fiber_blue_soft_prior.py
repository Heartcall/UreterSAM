import unittest
import numpy as np
from tools.fiber_reference_workspace import ROOT, read
from tools.fiber_blue_soft_prior import blue_evidence, refine

SETTINGS = read(ROOT / 'configs/fiber_blue_aplus_20261029.json')['refinement']


class BlueSoftPriorTests(unittest.TestCase):
    def sample(self):
        raw = np.zeros((100, 100), np.uint8); raw[48:52, :50] = 2; raw[45, 48] = 1
        margin = np.full(raw.shape, -5, np.float32); margin[48:52, 50] = -.1
        frame = np.zeros((100, 100, 3), np.uint8); frame[:] = (255, 30, 0)
        return frame, raw, margin

    def test_color_boost_is_positive_soft_and_white_neutral(self):
        image = np.array([[[255, 0, 0], [255, 255, 255], [0, 0, 0], [0, 0, 255]]], np.uint8)
        fields = blue_evidence(image, SETTINGS)
        self.assertGreater(fields['log_odds_boost'][0, 0], .3)
        self.assertTrue(np.all(fields['log_odds_boost'][0, 1:] == 0))
        self.assertTrue(np.isfinite(fields['blue_score_uncalibrated']).all())

    def test_disconnected_raw_fiber_and_stone_are_protected(self):
        frame, raw, margin = self.sample(); raw[60, 50] = 2; before = raw.copy()
        out, record = refine(frame, raw, margin, SETTINGS)
        self.assertEqual(record['added_pixels'], 4)
        self.assertTrue(np.all(out[raw == 2] == 2)); self.assertTrue(np.array_equal(out == 1, raw == 1))
        np.testing.assert_array_equal(raw, before)

    def test_no_hallucination_from_far_geometry_or_low_probability_or_white(self):
        frame, raw, margin = self.sample(); margin[90, 90] = -.01
        frame[48:52, 50] = 255
        out, _ = refine(frame, raw, margin, SETTINGS)
        np.testing.assert_array_equal(out, raw)
        control, record = refine(frame, raw, margin, SETTINGS, 'neural_relaxation_control')
        self.assertEqual(record['added_pixels'], 4); self.assertEqual(control[90, 90], 0)

    def test_budget_nan_and_no_anchor_fallback_without_topk(self):
        frame, raw, margin = self.sample(); margin[:] = -.01
        out, r = refine(frame, raw, margin, SETTINGS)
        self.assertEqual(r['fallback'], 'addition_budget'); np.testing.assert_array_equal(out, raw)
        margin[0, 0] = np.nan
        self.assertEqual(refine(frame, raw, margin, SETTINGS)[1]['fallback'], 'invalid_neural_margin')
        self.assertEqual(refine(frame, np.zeros_like(raw), margin, SETTINGS)[1]['fallback'], 'insufficient_network_fiber_anchor')


if __name__ == '__main__': unittest.main()
