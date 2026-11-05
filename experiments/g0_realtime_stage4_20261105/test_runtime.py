import copy
import unittest

import numpy as np

from tools.fiber_tapered_tangent_cap import fit_geometry
from tests.test_tapered_tangent_cap import scene
from experiments.g0_realtime_20261105.compare import close
from experiments.g0_realtime_20261105.benchmark import geometry_record
from experiments.g0_realtime_stage2_20261105.runtime import lazy_tip
from experiments.g0_realtime_stage3_20261105.runtime import Runtime as V3
from .runtime import Runtime
from .spatial import GridCache, search_box, roi_tip


class SpatialTests(unittest.TestCase):
    def test_grid_cache_is_exact_readonly_bounded_and_forwards_other_shapes(self):
        cache = GridCache(2)
        self.assertIs(cache.maximum, np.maximum)
        for shape, dtype in (((31, 57), int), ((31, 57), np.int32)):
            grid = cache.indices(shape, dtype=dtype)
            np.testing.assert_array_equal(grid, np.indices(shape, dtype=dtype))
            self.assertIs(grid, cache.indices(shape, dtype=dtype))
            with self.assertRaises(ValueError): grid[0, 0, 0] = 99
        self.assertEqual(cache.hits, 2)
        cache.indices((12, 13)); self.assertEqual(len(cache.cache), 2)
        np.testing.assert_array_equal(cache.indices((7,)), np.indices((7,)))
        for a, b in zip(cache.indices((4, 5), sparse=True), np.indices((4, 5), sparse=True)):
            np.testing.assert_array_equal(a, b)
        cache.reset_stats(); self.assertEqual(cache.hits+cache.misses, 0)
        self.assertEqual(len(cache.cache), 2)

    def test_analytic_bounds_contain_full_domain_across_orientations_and_intersection(self):
        rng = np.random.default_rng(29)
        shape = (53, 67); yy, xx = np.indices(shape)
        cfg = dict(minimum_search_axial_fraction=.25, maximum_forward_at400=100., search_side_margin_at400=4.)
        for _ in range(160):
            angle = rng.uniform(-np.pi, np.pi)
            u = np.array([np.cos(angle), np.sin(angle)])*rng.uniform(.8, 1.2)
            n = np.array([-u[1], u[0]])*rng.uniform(.8, 1.2)
            origin = rng.uniform(-40, 80, 2)
            g = dict(origin=origin, axis=u, normal=n, intersection=rng.uniform(-40, 80, 2),
                     body_extent=[-30., 40.], half_angle_radians=rng.uniform(0., .7))
            box = search_box(shape, g, cfg, 1.)
            self.assertIsNotNone(box)
            t = (xx-origin[0])*u[0]+(yy-origin[1])*u[1]
            v = (xx-origin[0])*n[0]+(yy-origin[1])*n[1]
            begin = -30+.25*70; limit = 140.
            half = np.maximum(0., (float((g['intersection']-origin) @ u)-t)*np.tan(g['half_angle_radians']))
            domain = (t >= begin) & (t <= limit) & (abs(v) <= half+4.)
            contained = np.zeros(shape, bool); y0, y1, x0, x1 = box
            contained[y0:y1, x0:x1] = True
            self.assertFalse((domain & ~contained).any())
        g['normal'] = g['axis'].copy()
        self.assertIsNone(search_box(shape, g, cfg, 1.))

    def test_roi_tip_equals_full_domain_at_image_edges_and_after_intersection(self):
        runtime = V3(); frame, raw, probability, body = scene()
        seed = runtime.observer.seeds(frame, raw, probability, None, 99)[0][0]
        cfg = runtime.g0.cfg['tip']
        geometry, _ = fit_geometry(body == 2, seed['u'], runtime.g0.cfg['fit'], 1.)
        grid = GridCache()
        for modification in ('normal', 'beyond', 'limit', 'border', 'empty'):
            g = copy.deepcopy(geometry); im = frame.copy(); labels = raw.copy()
            if modification == 'beyond': g['intersection'] = g['origin']+20*g['axis']
            if modification == 'limit': im[:] = (210, 100, 15)
            if modification == 'border': g['origin'] = np.array([0., 0.]); im[0] = (210, 100, 15)
            if modification == 'empty': g['body_extent'] = [1000., 1001.]
            a_count = dict(tip_candidate_pixels=0, tip_evaluated_pixels=0)
            b_count = a_count.copy(); spatial = dict(tip_full_pixels=0, tip_roi_pixels=0, roi_fallback=0)
            original = lambda *args:lazy_tip(*args, runtime.colors.fast_blue_identity, a_count)
            a = original(im, labels, g, seed, cfg, 1.)
            b = roi_tip(im, labels, g, seed, cfg, 1., runtime.colors.fast_blue_identity,
                        b_count, grid, spatial, original)
            for x, y in zip(a[:3], b[:3]): np.testing.assert_array_equal(x, y)
            self.assertTrue(close(a[3], b[3])); self.assertEqual(a_count, b_count)
        bad = copy.deepcopy(geometry); bad['normal'] = bad['axis'].copy()
        called = []
        marker = object()
        result = roi_tip(frame, raw, bad, seed, cfg, 1., runtime.colors.fast_blue_identity,
                         b_count, grid, spatial, lambda *args:(called.append(True) or marker))
        self.assertIs(result, marker); self.assertEqual(called, [True])

    def test_stateful_runtime_and_grid_only_ablation_match_v3_and_do_not_share_caches(self):
        old, new, grid_only, unused = V3(), Runtime(), Runtime(roi=False), Runtime()
        for index, mode in enumerate(('plain', 'gap', 'stone', 'none', 'plain')):
            frame, raw, probability, _ = scene()
            if mode == 'gap': frame[198:202, 200:204] = 0
            if mode == 'stone': raw[198:202, 220:224] = 1
            if mode == 'none': raw[:] = 0
            inputs = [x.copy() for x in (frame, raw, probability)]
            for runtime in (old, new, grid_only): runtime.run(frame, raw, probability, index/30, 'fixture')
            for runtime in (new, grid_only):
                np.testing.assert_array_equal(old.stage.final_mask, runtime.stage.final_mask)
                np.testing.assert_array_equal(old.g0.final_mask, runtime.g0.final_mask)
                np.testing.assert_array_equal(old.g0.layers['search_domain'], runtime.g0.layers['search_domain'])
                self.assertTrue(close(geometry_record(old.g0.info), geometry_record(runtime.g0.info)))
                self.assertTrue(close(old.stage.outline['cap'], runtime.stage.outline['cap']))
                self.assertEqual(old.counters, runtime.counters)
            for a, b in zip(inputs, (frame, raw, probability)): np.testing.assert_array_equal(a, b)
        self.assertGreater(new.grid.hits, 0); self.assertEqual(unused.grid.hits+unused.grid.misses, 0)
        self.assertLess(new.spatial_counters['tip_roi_pixels'], new.spatial_counters['tip_full_pixels'])
        new.reset(); self.assertEqual(sum(new.spatial_counters.values()), 0)


if __name__ == '__main__': unittest.main()
