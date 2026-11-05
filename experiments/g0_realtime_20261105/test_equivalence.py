"""Differential checks for changed mathematical/runtime building blocks."""
import copy
import unittest

import numpy as np

from tools.fiber_reference_workspace import ROOT, read
from tools.fiber_weak_stage2 import evidence, native_gradient, GrowthPipeline
from tools.fiber_blue_extent import BlueExtent
from tools.fiber_straight_semicircle import observed_sides, blue_identity
from tools.fiber_tapered_tangent_cap import TaperedTangentCap, raster_geometry
from tests.test_tapered_tangent_cap import scene, analytic_geometry
from tests.test_weak_blue_evidence import scene as weak_scene
from .fast_evidence import fast_evidence, fast_color, fast_blue_identity
from .fast_geometry import FastGeometry, grouped_observed_sides, cached_raster_geometry
from .fast_stage2 import FastStage2
from .compare import close


class EquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = read(ROOT/'configs/tapered_tangent_cap_no_fallback_20261104.json')
        self.rng = np.random.default_rng(18)
        mu = np.array([.05, .3, .65], np.float32)
        self.seed = dict(mu=mu, v=mu-1/3, bg=np.array([.5, .3, .2], np.float32), gap=.5)

    def test_image_evidence_flags_exact_and_probe_scores_exact(self):
        cfg = read(ROOT/'configs/weak_blue_evidence_control_20261103.json')['weak_blue']['evidence']
        image = self.rng.integers(0, 256, (80, 96, 3), dtype=np.uint8)
        image[0] = 0; image[1] = 255
        gradient = native_gradient(image)
        before = evidence(image, self.seed, cfg, gradient)
        after = fast_evidence(image, self.seed, cfg, gradient)
        for k in (1, 2, 3, 4, 5):
            np.testing.assert_array_equal(before[k], after[k])
        probe = image.reshape(-1, 1, 3)[::19]
        # Terminal decisions use continuous scores even for non-blue samples.
        for a, b in zip(evidence(probe, self.seed, cfg), fast_evidence(probe, self.seed, cfg)):
            np.testing.assert_array_equal(a, b)

    def test_union_color_mask_exact_both_strengths(self):
        cfg = read(ROOT/'configs/fiber_aggressive_blue_control_20261102.json')
        settings = copy.deepcopy(cfg['extent'])
        settings['color'].update(cfg['aggressive_parameters']['broad_color'])
        observer = BlueExtent(settings)
        image = self.rng.integers(0, 256, (20000, 3), dtype=np.uint8)
        image[:100] = 0; image[100:200] = 255
        for weak in (False, True):
            np.testing.assert_array_equal(observer.color(image, self.seed, weak),
                                          fast_color(observer, image, self.seed, weak))

    def test_g0_blue_identity_exact(self):
        image = self.rng.integers(0, 256, (80, 96, 3), dtype=np.uint8)
        for a, b in zip(blue_identity(image, self.seed, self.cfg['tip']),
                        fast_blue_identity(image, self.seed, self.cfg['tip'])):
            np.testing.assert_array_equal(a, b)

    def test_side_grouping_retains_order_extrema_and_bin_means(self):
        for shape in ((80, 96), (160, 160), (400, 400)):
            body = self.rng.random(shape) > .6
            for angle in (0., .3, 1.2, 2.6):
                axis = np.array([np.cos(angle), np.sin(angle)])
                origin = np.array([shape[1]/2, shape[0]/2])
                a = observed_sides(body, origin, axis, self.cfg['fit'], min(shape)/400)
                b = grouped_observed_sides(body, origin, axis, self.cfg['fit'], min(shape)/400)
                for x, y in zip(a, b):
                    np.testing.assert_array_equal(x, y)

    def test_cached_raster_exact_across_reuse_and_impossible_tip(self):
        for angle in (.02, .12, .3, .7):
            g = analytic_geometry(angle)
            for tip in ([270, 200], [290, 200], [310, 200]):
                a = raster_geometry((400, 400), g, tip)
                b = cached_raster_geometry((400, 400), g, tip)
                for x, y in zip(a[:3], b[:3]):
                    np.testing.assert_array_equal(x, y)
                self.assertTrue(close(a[3], b[3])); self.assertEqual(a[4], b[4])

    def test_complete_g0_matches_and_does_not_mutate_inputs(self):
        for mode in ('plain', 'stone', 'no_subject', 'gap'):
            frame, raw, probability, old = scene()
            if mode == 'stone':
                raw[198:202, 220:224] = 1; old[198:202, 220:224] = 1
            elif mode == 'no_subject':
                raw[:] = 0
            elif mode == 'gap':
                old[199:202, 170:174] = 0
            inputs = [x.copy() for x in (frame, raw, probability, old)]
            original, candidate = TaperedTangentCap(self.cfg), FastGeometry(self.cfg)
            original.run(frame, raw, probability, old); candidate.run(frame, raw, probability, old)
            np.testing.assert_array_equal(original.final_mask, candidate.final_mask)
            for key in original.layers:
                np.testing.assert_array_equal(original.layers[key], candidate.layers[key])
            a, b = original.info.copy(), candidate.info.copy()
            a.pop('runtime_ms'); b.pop('runtime_ms'); self.assertTrue(close(a, b))
            for x, y in zip(inputs, (frame, raw, probability, old)):
                np.testing.assert_array_equal(x, y)

    def test_stage2_mask_and_history_reset_equivalence(self):
        cfg = read(ROOT/'configs/weak_blue_evidence_control_20261103.json')
        t1 = read(ROOT/'configs/fiber_temporal_observation_search_20261030.json')
        original = GrowthPipeline(cfg, 'weak_blue_extension_tip', t1, True)
        candidate = FastStage2(cfg, 'weak_blue_extension_tip', t1, True)
        frame, raw, probability, _ = weak_scene()
        for j in range(4):
            original.run(frame, raw, probability, j/30, 'synthetic')
            candidate.run(frame, raw, probability, j/30, 'synthetic')
            np.testing.assert_array_equal(original.final_mask, candidate.final_mask)
            self.assertEqual(original.outline.get('cap', {}).get('accepted', False),
                             candidate.outline.get('cap', {}).get('accepted', False))
        original.reset(); candidate.reset()
        self.assertEqual(original.layers, candidate.layers)
        self.assertIsNone(candidate.final_mask)


if __name__ == '__main__':
    unittest.main()
