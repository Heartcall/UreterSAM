import unittest
from unittest.mock import patch
import numpy as np

from tests.test_tapered_tangent_cap import scene
from experiments.g0_realtime_20261105.compare import close
from experiments.g0_realtime_20261105.benchmark import geometry_record
from experiments.g0_realtime_stage4_20261105.runtime import Runtime as V4
from .runtime import Runtime
from .spatial import domain_from_body, rectangle
from .audit import pack_masks, unpack_masks


def reference_domain(body, origin, u, cfg, scale, grid=np):
    normal = np.array([-u[1], u[0]])
    yy, xx = grid.indices(body.shape)
    t = (xx-origin[0])*u[0]+(yy-origin[1])*u[1]
    v = (xx-origin[0])*normal[0]+(yy-origin[1])*normal[1]
    bt, bv = t[body > 0], v[body > 0]
    front = float(bt.max())
    limit = min(cfg['maximum_forward_at400']*scale,
                cfg['maximum_forward_diagonal_fraction']*np.hypot(*body.shape))
    lateral = (np.quantile(bv, .98)-np.quantile(bv, .02))*cfg['lateral_width_multiplier']/2+cfg['lateral_margin_at400']*scale
    center = float(np.median(bv))
    return (t >= bt.min()-2*scale) & (t <= front+limit) & (abs(v-center) <= lateral), front, limit


class NumpyDomainTests(unittest.TestCase):
    def test_lossless_audit_storage_keeps_all_labels_and_padding(self):
        rng = np.random.default_rng(93)
        for shape in ((1, 1), (7, 11), (400, 400)):
            s2 = rng.integers(0, 3, size=shape, dtype=np.uint8)
            final = rng.integers(0, 3, size=shape, dtype=np.uint8)
            packed = pack_masks(s2, final)
            a, b = unpack_masks(packed, shape)
            np.testing.assert_array_equal(a, s2); np.testing.assert_array_equal(b, final)
            self.assertEqual(a.dtype, s2.dtype); self.assertEqual(b.dtype, final.dtype)
            self.assertEqual(packed.nbytes, (4*s2.size+7)//8)

    def test_invalid_audit_labels_and_shapes_are_rejected(self):
        labels = np.array([[0, 1, 2]], np.uint8)
        for other in (np.array([[3, 1, 2]], np.uint8), labels.astype(float), np.zeros((2, 2), np.uint8)):
            with self.assertRaises(ValueError): pack_masks(labels, other)

    def test_full_domain_exact_across_random_orientations_and_offsets(self):
        rng = np.random.default_rng(91)
        for index in range(240):
            shape = (int(rng.integers(9, 86)), int(rng.integers(11, 97)))
            body = rng.random(shape) < .04
            body[shape[0]//2, shape[1]//2] = True
            origin = rng.uniform(-70, 120, 2)
            angle = rng.uniform(-np.pi, np.pi)
            u = np.array([np.cos(angle), np.sin(angle)])*rng.uniform(.5, 1.6)
            cfg = dict(maximum_forward_at400=rng.uniform(0, 80),
                       maximum_forward_diagonal_fraction=rng.uniform(0, .8),
                       lateral_width_multiplier=rng.uniform(0, 2),
                       lateral_margin_at400=rng.uniform(0, 7))
            expected, front, limit = reference_domain(body, origin, u, cfg, .7)
            actual, f, l, counters = domain_from_body(body, origin, u, cfg, .7)
            np.testing.assert_array_equal(actual, expected, err_msg=str(index))
            self.assertEqual((f, l), (front, limit))
            self.assertLessEqual(counters['roi_pixels'], counters['full_pixels'])

    def test_boundary_directions_empty_box_and_invalid_matrix(self):
        shape = (31, 47); body = np.zeros(shape, np.uint8); body[0:5, 0:16] = 1
        cfg = dict(maximum_forward_at400=0., maximum_forward_diagonal_fraction=.3,
                   lateral_width_multiplier=1., lateral_margin_at400=0.)
        for u in (np.array([1., 0.]), np.array([-1., 0.]), np.array([0., 1.]), np.array([0., -1.])):
            for origin in (np.array([0., 0.]), np.array([-30., 51.])):
                np.testing.assert_array_equal(domain_from_body(body, origin, u, cfg, 1.)[0],
                                              reference_domain(body, origin, u, cfg, 1.)[0])
        for origin, u, normal in (([0., 0.], [1., 0.], [1., 0.]),
                                  ([float('nan'), 0.], [1., 0.], [0., 1.])):
            self.assertEqual(rectangle(shape, origin, u, normal, 0., 2., 0., 1.), ((0, 31, 0, 47), True))
        box, fallback = rectangle(shape, [1000., 1000.], [1., 0.], [0., 1.], 0., 1., 0., 1.)
        self.assertEqual(box, (31, 31, 47, 47)); self.assertFalse(fallback)

    def test_stateful_masks_caps_and_full_intermediate_domain_equal_v4(self):
        old, new = V4(), Runtime()
        original = old.expand
        observed = {}
        def audited(frame, raw, probability, meta, body, core, u):
            observed['domain'] = reference_domain(body, np.asarray(old.stage.outline['context']['origin']),
                                                  u, old.stage.q['search'], min(raw.shape)/400)[0]
            return original(frame, raw, probability, meta, body, core, u)
        old.expand = audited
        for index, mode in enumerate(('plain', 'gap', 'stone', 'none', 'budget', 'plain')):
            frame, raw, probability, _ = scene()
            if mode == 'gap': frame[198:202, 200:204] = 0
            if mode == 'stone': raw[198:202, 220:224] = 1
            if mode == 'none': raw[:] = 0
            initial = [a.copy() for a in (frame, raw, probability)]
            saved = [r.stage.q['search']['maximum_new_fraction'] for r in (old, new)]
            if mode == 'budget':
                for runtime in (old, new): runtime.stage.q['search']['maximum_new_fraction'] = -1
            observed.clear()
            with patch('ctypes.CDLL', side_effect=AssertionError('Custom DLL load attempted')):
                for runtime in (old, new): runtime.run(frame, raw, probability, index/30, 'fixture')
            for runtime, value in zip((old, new), saved): runtime.stage.q['search']['maximum_new_fraction'] = value
            np.testing.assert_array_equal(old.stage.final_mask, new.stage.final_mask)
            np.testing.assert_array_equal(old.g0.final_mask, new.g0.final_mask)
            self.assertTrue(close(geometry_record(old.g0.info), geometry_record(new.g0.info)))
            self.assertTrue(close(old.stage.outline.get('cap'), new.stage.outline.get('cap')))
            if new.last_domain is None: self.assertNotIn('domain', observed)
            else: np.testing.assert_array_equal(new.last_domain, observed['domain'])
            self.assertEqual(old.counters, new.counters)
            for expected, actual in zip(initial, (frame, raw, probability)): np.testing.assert_array_equal(expected, actual)
        self.assertGreater(new.domain_counters['roi_pixels'], 0)
        self.assertLess(new.domain_counters['roi_pixels'], new.domain_counters['full_pixels'])
        new.reset(); self.assertIsNone(new.last_domain); self.assertEqual(sum(new.domain_counters.values()), 0)


if __name__ == '__main__': unittest.main()
