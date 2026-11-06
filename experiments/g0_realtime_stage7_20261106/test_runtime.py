import unittest
from unittest.mock import patch
import numpy as np

from tests.test_tapered_tangent_cap import scene
from experiments.g0_realtime_20261105.compare import close
from experiments.g0_realtime_20261105.benchmark import geometry_record
from experiments.g0_realtime_stage3_20261105.pixels import chroma
from experiments.g0_realtime_stage4_20261105.spatial import GridCache
from experiments.g0_realtime_stage6_20261106.runtime import Runtime as V6
from .runtime import Runtime, BroadcastNumpy
from .color import from_chroma


def growth(runtime):
    node = runtime.stage.inner
    while hasattr(node, 'inner'): node = node.inner
    return node.growth


class ReuseTests(unittest.TestCase):
    def test_sparse_views_preserve_every_projection_value_and_are_private(self):
        counts = dict(test=0); grid = GridCache(); proxy = BroadcastNumpy(grid, counts, 'test')
        rng = np.random.default_rng(94)
        for shape in ((1, 7), (53, 79), (400, 400)):
            yy, xx = proxy.indices(shape)
            self.assertFalse(yy.flags.writeable); self.assertFalse(xx.flags.writeable)
            dy, dx = np.indices(shape)
            for _ in range(16):
                o, u = rng.normal(size=(2, 2))*70
                np.testing.assert_array_equal((xx-o[0])*u[0]+(yy-o[1])*u[1],
                                              (dx-o[0])*u[0]+(dy-o[1])*u[1])
        self.assertEqual(counts['test'], 3)
        np.testing.assert_array_equal(proxy.indices((7,)), np.indices((7,)))
        original = np.indices
        base, first, second = V6(), Runtime(), Runtime()
        self.assertIs(np.indices, original)
        self.assertIs(base.stage.fit_cap.__func__.__globals__['np'], base.grid)
        self.assertIsNot(first.grid, second.grid)
        self.assertEqual(np.indices((2, 3)).shape, (2, 2, 3))

    def test_region_chroma_and_color_equal_reference_for_both_precisions(self):
        runtime = Runtime(); frame, raw, p, _ = scene()
        seed = runtime.observer.seeds(frame, raw, p, None, 99)[0][0]
        rng = np.random.default_rng(95)
        for mode in ('random', 'zero', 'white', 'blue'):
            image = rng.integers(0, 256, (113, 157, 3), np.uint8)
            if mode == 'zero': image[:] = 0
            if mode == 'white': image[:] = 255
            if mode == 'blue': image[:] = (210, 100, 15)
            region = rng.random(image.shape[:2]) > .25
            runtime.frame = image; runtime._chroma = None
            full_c, full_n = runtime.chroma(image)
            part_c, part_n = chroma(image[region])
            np.testing.assert_array_equal(full_c[region], part_c)
            np.testing.assert_array_equal(full_n[region], part_n)
            for dtype in (np.float32, np.float64):
                s = dict(seed)
                for key in ('mu', 'bg', 'v'): s[key] = np.asarray(seed[key], dtype=dtype)
                for weak in (False, True):
                    expected = runtime.pixels.color(runtime.observer, image[region], s, weak)
                    np.testing.assert_array_equal(from_chroma(runtime.observer, full_c[region], full_n[region], s, weak), expected)
                np.testing.assert_array_equal(runtime.region_color(image, region, s),
                                              runtime.observer.color(image[region], s))
        self.assertEqual(runtime.extra_counts['region_reused'], 8)

    def test_small_float_and_outside_frame_calls_keep_the_reference_path(self):
        runtime = Runtime(); frame, raw, p, _ = scene()
        seed = runtime.observer.seeds(frame, raw, p, None, 99)[0][0]
        for count in (0, 17, 4095, 4096):
            region = np.zeros(frame.shape[:2], bool); region.flat[:count] = True
            for dtype in (np.uint8, np.float32):
                image = frame.astype(dtype); runtime.frame = image; runtime._chroma = None
                with patch.object(runtime.observer, 'color', wraps=runtime.observer.color) as reference:
                    actual = runtime.region_color(image, region, seed)
                    self.assertEqual(reference.call_count, int(count < 4096 or dtype != np.uint8))
                np.testing.assert_array_equal(actual, runtime.observer.color(image[region], seed))
        runtime.frame = None
        region[:] = True
        with patch.object(runtime.observer, 'color', wraps=runtime.observer.color) as reference:
            runtime.region_color(frame, region, seed)
            reference.assert_called_once()

    def test_stateful_ablations_match_masks_layers_caps_and_input_values(self):
        base = V6(); arms = [Runtime(True, False), Runtime(False, True), Runtime()]
        for index, mode in enumerate(('plain', 'gap', 'stone', 'none', 'budget', 'plain', 'same_buffer')):
            if mode != 'same_buffer': frame, raw, p, _ = scene()
            if mode == 'gap': frame[198:202, 200:204] = 0
            if mode == 'stone': raw[198:202, 220:224] = 1
            if mode == 'none': raw[:] = 0
            if mode == 'same_buffer': frame[198:202, 200:204] = 0
            inputs = [a.copy() for a in (frame, raw, p)]
            saved = [r.stage.q['search']['maximum_new_fraction'] for r in [base, *arms]]
            if mode == 'budget':
                for r in [base, *arms]: r.stage.q['search']['maximum_new_fraction'] = -1
            with patch('ctypes.CDLL', side_effect=AssertionError('Custom DLL load attempted')):
                for r in [base, *arms]: r.run(frame, raw, p, index/30, 'fixture')
            for r, value in zip([base, *arms], saved): r.stage.q['search']['maximum_new_fraction'] = value
            for r in arms:
                np.testing.assert_array_equal(base.stage.final_mask, r.stage.final_mask)
                np.testing.assert_array_equal(base.g0.final_mask, r.g0.final_mask)
                self.assertTrue(close(geometry_record(base.g0.info), geometry_record(r.g0.info)))
                self.assertTrue(close(base.stage.outline.get('cap'), r.stage.outline.get('cap')))
                for name in growth(base).layers:
                    np.testing.assert_array_equal(growth(base).layers[name], growth(r).layers[name])
                for name in base.counters:
                    if name != 'chroma_reuse': self.assertEqual(base.counters[name], r.counters[name])
                if base.last_domain is not None: np.testing.assert_array_equal(base.last_domain, r.last_domain)
            for a, b in zip(inputs, (frame, raw, p)): np.testing.assert_array_equal(a, b)
        self.assertGreater(arms[0].extra_counts['region_reused'], 0)
        self.assertGreater(arms[1].extra_counts['grow_broadcast'], 0)
        self.assertEqual(arms[1].extra_counts['region_reused'], 0)
        for r in arms:
            r.reset(); self.assertEqual(sum(r.extra_counts.values()), 0)


if __name__ == '__main__': unittest.main()
