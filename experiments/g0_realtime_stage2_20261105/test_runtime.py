import copy
import unittest

import numpy as np

from tools.fiber_reference_workspace import ROOT, read
from tools.fiber_blue_outline import profiles
from tools.fiber_tapered_tangent_cap import locate_tip, fit_geometry
from experiments.g0_realtime_20261105.fast_geometry import FastGeometry
from experiments.g0_realtime_20261105.fast_stage2 import FastStage2
from experiments.g0_realtime_20261105.compare import close
from experiments.g0_realtime_20261105.benchmark import geometry_record
from tests.test_tapered_tangent_cap import scene
from .runtime import Runtime, grouped_profiles, lazy_tip


class RuntimeTests(unittest.TestCase):
    def test_grouped_profiles_exact_with_holes_rotations_and_borders(self):
        rng = np.random.default_rng(7)
        for shape in ((81, 93), (160, 160), (400, 400)):
            body = rng.random(shape) > .6
            for angle in (0., .35, 1.22, 2.7):
                u = np.array([np.cos(angle), np.sin(angle)])
                o = np.array([shape[1]/2, shape[0]/2])
                a = profiles(body, o, u, min(shape)/400, 1.5)
                b = grouped_profiles(body, o, u, min(shape)/400, 1.5)
                for x, y in zip(a, b):
                    np.testing.assert_array_equal(x, y)

    def test_full_stateful_runtime_equal_including_gap_stone_absence(self):
        cfg = read(ROOT/'configs/tapered_tangent_cap_no_fallback_20261104.json')
        stage = FastStage2(read(ROOT/'configs/weak_blue_evidence_control_20261103.json'),
                           'weak_blue_extension_tip',
                           read(ROOT/'configs/fiber_temporal_observation_search_20261030.json'), True)
        geometry = FastGeometry(cfg)
        runtime = Runtime()
        for j, mode in enumerate(('plain', 'gap', 'stone', 'none', 'plain')):
            im, raw, p, old = scene()
            if mode == 'gap':
                im[198:202, 200:204] = 0
            elif mode == 'stone':
                raw[198:202, 220:224] = 1
            elif mode == 'none':
                raw[:] = 0
            inputs = [x.copy() for x in (im, raw, p)]
            stage.run(im, raw, p, j/30, 'synthetic')
            geometry.run(im, raw, p, stage.final_mask)
            runtime.run(im, raw, p, j/30, 'synthetic')
            np.testing.assert_array_equal(stage.final_mask, runtime.stage.final_mask)
            np.testing.assert_array_equal(geometry.final_mask, runtime.g0.final_mask)
            self.assertTrue(close(geometry_record(geometry.info), geometry_record(runtime.g0.info)))
            for a, b in zip(inputs, (im, raw, p)):
                np.testing.assert_array_equal(a, b)
        self.assertGreater(runtime.counters['seed_reuse'], 0)
        self.assertLess(runtime.counters['tip_evaluated_pixels'], runtime.counters['tip_candidate_pixels'])
        runtime.reset()
        self.assertTrue(all(value == 0 for value in runtime.counters.values()))
        self.assertIsNone(runtime._captured)

    def test_weak_history_seed_and_other_frame_force_original_g0_observation(self):
        runtime = Runtime()
        im, raw, p, old = scene()
        seeds, records = runtime.observer.seeds(im, raw, p, None, 99)
        self.assertEqual(len(seeds), 1)
        changed = copy.deepcopy(records)
        next(z for z in changed if z['reason'] == 'qualified')['mode'] = 'weak_history'
        runtime._captured = (im, raw, p, (seeds, changed))
        actual, _ = runtime.g0.seed.seeds(im, raw, p, None, 99)
        self.assertEqual(len(actual), len(seeds))
        self.assertEqual(runtime.counters['seed_reuse'], 0)
        self.assertEqual(runtime.counters['seed_fallback'], 1)
        runtime._captured = (im.copy(), raw, p, (seeds, records))
        runtime.g0.seed.seeds(im, raw, p, None, 99)
        self.assertEqual(runtime.counters['seed_fallback'], 2)

    def test_current_frame_chroma_cache_does_not_reuse_other_image(self):
        runtime = Runtime()
        im, raw, p, old = scene()
        runtime.frame = im
        a = runtime.chroma(im); b = runtime.chroma(im)
        self.assertIs(a, b)
        self.assertEqual(runtime.counters['chroma_reuse'], 1)
        other = im.copy(); other[:] = 0
        c = runtime.chroma(other)
        self.assertFalse(np.array_equal(c[0], a[0]))

    def test_lazy_tip_matches_full_search_when_winner_late_or_no_winner(self):
        runtime = Runtime()
        cfg = read(ROOT/'configs/tapered_tangent_cap_no_fallback_20261104.json')
        im, raw, p, old = scene()
        seeds, _ = runtime.observer.seeds(im, raw, p, None, 99)
        seed = seeds[0]
        g, _ = fit_geometry(old == 2, seed['u'], cfg['fit'], 1.)
        for fill in (False, True):
            frame = im.copy()
            if fill:
                frame[:] = (210, 100, 15)
            original = locate_tip(frame, raw, g, seed, cfg['tip'], 1.)
            actual = lazy_tip(frame, raw, g, seed, cfg['tip'], 1.,
                              runtime.colors.fast_blue_identity, runtime.counters)
            for key in ('reason', 'source_pixel', 'vertex'):
                self.assertTrue(close(original[3][key], actual[3][key]))


if __name__ == '__main__':
    unittest.main()
