import json
from pathlib import Path
import unittest
import cv2
import numpy as np

from tools.fiber_shape_prior import CausalShapeTracker, fit_shape, visible_fiber, combine

CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'configs/fiber_geometry_prior_20261028.json').read_text(encoding='utf-8'))


class FiberShapePriorTests(unittest.TestCase):
    def test_image_supported_straight_shaft_rejects_detached_blob(self):
        truth = np.zeros((160, 200), np.uint8)
        cv2.rectangle(truth, (0, 68), (98, 92), 1, -1)
        cv2.ellipse(truth, (98, 80), (12, 12), 0, -90, 90, 1, -1)
        raw = truth.copy(); cv2.circle(raw, (165, 35), 8, 1, -1)
        frame = np.full((160, 200, 3), 100, np.uint8); frame[truth > 0] = (200, 70, 20)
        probability = np.where(raw, .95, .02).astype(np.float32)
        result = fit_shape(frame, probability, raw.astype(bool), np.ones(raw.shape, bool), CONFIG['fit'])
        self.assertTrue(result['accepted'], result)
        refined = visible_fiber(probability, raw.astype(bool), result['state'], .35)
        self.assertFalse(refined[35, 165])
        self.assertGreater((refined & (truth > 0)).sum() / (truth > 0).sum(), .90)
        self.assertLess(np.linalg.norm(np.asarray(result['tip_xy']) - [110, 80]), 5)

    def test_internal_blob_and_flat_image_do_not_get_a_forced_shape(self):
        mask = np.zeros((100, 120), np.uint8); cv2.circle(mask, (60, 50), 15, 1, -1)
        frame = np.full((100, 120, 3), 100, np.uint8); probability = mask.astype(np.float32)
        r = fit_shape(frame, probability, mask > 0, np.ones(mask.shape, bool), CONFIG['fit'])
        self.assertFalse(r['accepted']); self.assertEqual(r['reason'], 'no_edge_attached_candidate')
        mask[:, :] = 0; mask[40:60, :90] = 1
        r = fit_shape(frame, mask.astype(np.float32), mask > 0, np.ones(mask.shape, bool), CONFIG['fit'])
        self.assertFalse(r['accepted']); self.assertEqual(r['reason'], 'insufficient_two_sided_image_edges')
        self.assertTrue(np.array_equal(visible_fiber(probability, mask > 0, None, .35), mask > 0))

    def test_causal_prefix_reset_and_prediction_never_becomes_visible(self):
        states = [[0, 80, 0, 110 + offset, 24, 24, 1] for offset in (0, 2, -1, 1)]
        a = CausalShapeTracker(CONFIG['tracking']); b = CausalShapeTracker(CONFIG['tracking'])
        first = [a.step(x, (160, 200), 1 / 30) for x in states]
        prefix = [b.step(x, (160, 200), 1 / 30) for x in states[:2]]
        self.assertEqual(first[:2], prefix)
        for _ in range(3):
            r = a.step(None, (160, 200), 1 / 30)
            self.assertFalse(r['apply']); self.assertEqual(r['status'], 'predicted_not_applied')
        self.assertEqual(a.step(None, (160, 200), 1 / 30)['status'], 'expired_reset')
        self.assertEqual(a.step(states[0], (160, 200), 1 / 30)['status'], 'initialized')
        a.reset(); self.assertIsNone(a.state)
        stone = np.array([[1, 1], [0, 2]], np.uint8); fiber = np.array([[True, False], [False, False]])
        self.assertTrue(np.array_equal(combine(stone, fiber), [[2, 1], [0, 0]]))


if __name__ == '__main__': unittest.main()
