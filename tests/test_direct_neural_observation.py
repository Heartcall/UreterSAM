import copy
import unittest
import numpy as np
from tools import direct_fiber_template as frozen
from tools.direct_neural_observation import IMAGE, NEURAL, locate, neural_observation, neural_visible
from tests.test_direct_fiber_template import CONFIG, synthetic


class ObservationControlTests(unittest.TestCase):
    def test_image_arm_exact_predecessor_and_probability_independent(self):
        frame, truth, prior = synthetic()
        old, _ = frozen.locate(frame, prior, CONFIG, requested=['image_template'])
        for p in (np.zeros(truth.shape, np.float32), np.ones(truth.shape, np.float32)):
            new, _ = locate(frame, prior, CONFIG, p)
            self.assertEqual(old['image_template']['record'], new[IMAGE]['record'])
            np.testing.assert_array_equal(old['image_template']['fiber'], new[IMAGE]['fiber'])

    def test_only_foreground_field_replaced_without_extra_bonus(self):
        frame, truth, prior = synthetic(); p = truth.astype(np.float32)
        before = frozen.observation(frame, prior, CONFIG); after = neural_observation(frame, prior, CONFIG, p)
        for k in before:
            if k != 'appearance':
                np.testing.assert_equal(before[k], after[k])
        a, _ = locate(frame, prior, CONFIG, p, requested=[NEURAL])
        config = copy.deepcopy(CONFIG); config['score']['network_soft'] = 10000
        b, _ = locate(frame, prior, config, p, requested=[NEURAL])
        self.assertEqual(a[NEURAL]['record'], b[NEURAL]['record'])
        self.assertTrue(a[NEURAL]['record']['accepted'])
        self.assertGreater((a[NEURAL]['fiber'] & truth).sum() / truth.sum(), .9)
        self.assertFalse((a[NEURAL]['fiber'] & ~truth).any())

    def test_color_failure_does_not_suppress_neural_field(self):
        frame, truth, prior = synthetic(); prior['appearance_probability'] = np.zeros((16, 16)).tolist()
        out, _ = locate(frame, prior, CONFIG, truth.astype(np.float32))
        self.assertFalse(out[IMAGE]['record']['accepted'])
        self.assertTrue(out[NEURAL]['record']['accepted'])
        _, full = frozen.visible(frame, [0, 80, 0, 100], True, prior, CONFIG)
        visible, hypothesis = neural_visible(frame, [0, 80, 0, 100], True, prior, CONFIG, truth.astype(np.float32))
        np.testing.assert_array_equal(full, hypothesis)
        np.testing.assert_array_equal(visible, full & truth)

    def test_absence_invalid_probability_and_causal_prefix(self):
        frame, truth, prior = synthetic()
        for p in (np.zeros((1, 1)), np.full(truth.shape, np.nan), np.full(truth.shape, 2)):
            with self.assertRaises(ValueError): locate(frame, prior, CONFIG, p, requested=[NEURAL])
        tracker = frozen.DirectTracker(CONFIG['tracking']); tracker.update([0, 80, 0, 100], True)
        out, _ = locate(np.zeros_like(frame), prior, CONFIG, np.zeros(truth.shape), tracker, ['causal_template'])
        self.assertFalse(out['causal_template']['fiber'].any())
        self.assertEqual(out['causal_template']['record']['tracking']['status'], 'predicted_not_visible')
        a = frozen.DirectTracker(CONFIG['tracking']); b = frozen.DirectTracker(CONFIG['tracking'])
        records = []
        for _ in range(3): records.append(locate(frame, prior, CONFIG, truth.astype(np.float32), a, ['causal_template'])[0]['causal_template']['record'])
        for i in range(2): self.assertEqual(records[i], locate(frame, prior, CONFIG, truth.astype(np.float32), b, ['causal_template'])[0]['causal_template']['record'])


if __name__ == '__main__': unittest.main()
