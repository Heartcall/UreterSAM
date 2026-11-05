import unittest

import cv2
import numpy as np

from tools.fiber_connected_blue_growth import chroma
from tools.fiber_connected_blue_union import rooted
from tools.fiber_weak_stage2 import native_gradient, connect
from experiments.g0_realtime_20261105.compare import close
from experiments.g0_realtime_20261105.benchmark import geometry_record
from experiments.g0_realtime_stage2_20261105.runtime import Runtime as V2
from tests.test_tapered_tangent_cap import scene
from .runtime import Runtime
from . import pixels


class PixelTests(unittest.TestCase):
    def test_chroma_and_gradient_exact_with_borders_and_noncontiguous_inputs(self):
        rng = np.random.default_rng(13)
        for shape in ((17, 29, 3), (400, 400, 3)):
            for mode in ('random', 'black', 'white', 'corners'):
                image = rng.integers(0, 256, shape, np.uint8)
                if mode == 'black': image[:] = 0
                if mode == 'white': image[:] = 255
                if mode == 'corners':
                    image[:] = 0; image[0, 0] = 255; image[-1, -1] = (13, 200, 241)
                for view in (image, image[:, ::-1]):
                    for a, b in zip(chroma(view), pixels.chroma(view)):
                        np.testing.assert_array_equal(a, b)
                    np.testing.assert_array_equal(native_gradient(view), pixels.native_gradient(view))
        probes = rng.random((77, 11, 3)).astype(np.float32)*255
        for a, b in zip(chroma(probes), pixels.chroma(probes)):
            np.testing.assert_array_equal(a, b)

    def test_label_lookup_preserves_four_and_eight_neighbor_rules(self):
        rng = np.random.default_rng(17)
        for shape in ((1, 1), (15, 23), (400, 400)):
            for density in (0., .05, .3, .7, 1.):
                binary = rng.random(shape) < density
                anchor = rng.random(shape) < .3
                np.testing.assert_array_equal(connect(binary, anchor), pixels.connect(binary, anchor))
                for minimum in (0, 1, 3, 3.5, 10000):
                    a, ac = rooted(binary, anchor, minimum)
                    b, bc = pixels.rooted(binary, anchor, minimum)
                    np.testing.assert_array_equal(a, b); self.assertEqual(ac, bc)
        diagonal = np.eye(9, dtype=bool); anchor = np.zeros_like(diagonal); anchor[0, 0] = 1
        self.assertEqual(int(pixels.connect(diagonal, anchor).sum()), 9)
        self.assertEqual(int(pixels.rooted(diagonal, anchor, 1)[0].sum()), 1)

    def test_pixel_evidence_and_color_match_v2_for_random_and_threshold_colors(self):
        rng = np.random.default_rng(19)
        old, new = V2(), Runtime()
        image, raw, probability, _ = scene()
        seeds, _ = old.observer.seeds(image, raw, probability, None, 99)
        seed = seeds[0]
        e_cfg = old.stage.q['evidence']; tip_cfg = old.g0.cfg['tip']
        for index in range(10):
            frame = rng.integers(0, 256, (400, 400, 3), np.uint8)
            if index == 1: frame[:] = 0
            if index == 2: frame[:] = (210, 100, 15)
            gradient = native_gradient(frame)
            for a, b in zip(old.colors.fast_evidence(frame, seed, e_cfg, gradient),
                            new.pixels.evidence(frame, seed, e_cfg, gradient)):
                np.testing.assert_array_equal(a, b)
            for weak in (False, True):
                np.testing.assert_array_equal(old.observer.color(frame.reshape(-1, 3), seed, weak),
                                              new.observer.color(frame.reshape(-1, 3), seed, weak))
            for a, b in zip(old.colors.fast_blue_identity(frame, seed, tip_cfg),
                            new.pixels.blue_identity(frame, seed, tip_cfg)):
                np.testing.assert_array_equal(a, b)
        # Exhaust a 16-level RGB grid, including black/neutral/saturated colors.
        level = np.arange(0, 256, 17, dtype=np.uint8)
        grid = np.stack(np.meshgrid(level, level, level, indexing='ij'), -1).reshape(-1, 3)
        for weak in (False, True):
            np.testing.assert_array_equal(old.observer.color(grid, seed, weak), new.observer.color(grid, seed, weak))

    def test_stateful_pipeline_and_tracker_equal_with_stone_gap_absence_and_reset(self):
        old, new = V2(), Runtime()
        for index, mode in enumerate(('plain', 'gap', 'stone', 'none', 'plain')):
            frame, raw, probability, _ = scene()
            if mode == 'gap': frame[198:202, 200:204] = 0
            if mode == 'stone': raw[198:202, 220:224] = 1
            if mode == 'none': raw[:] = 0
            before = [x.copy() for x in (frame, raw, probability)]
            old.run(frame, raw, probability, index/30, 'fixture')
            new.run(frame, raw, probability, index/30, 'fixture')
            np.testing.assert_array_equal(old.stage.final_mask, new.stage.final_mask)
            np.testing.assert_array_equal(old.g0.final_mask, new.g0.final_mask)
            self.assertTrue(close(geometry_record(old.g0.info), geometry_record(new.g0.info)))
            self.assertTrue(close(old.stage.outline['cap'], new.stage.outline['cap']))
            self.assertEqual(old.stage.tracker.age, new.stage.tracker.age)
            for a, b in zip(before, (frame, raw, probability)):
                np.testing.assert_array_equal(a, b)
        self.assertEqual(old.counters, new.counters)
        old.reset(); new.reset()
        self.assertEqual(old.counters, new.counters)


if __name__ == '__main__':
    unittest.main()
