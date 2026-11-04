import copy
import json
from pathlib import Path
import unittest
import cv2
import numpy as np
from tools.fiber_fast_observation import FastFiberObserver, grid_to_native, sample, select_current, public_observation

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT/'configs/fiber_temporal_observation_search_20261030.json').read_text(encoding='utf-8'))


def scene(y=200, disconnected=False, white=False):
    frame = np.full((400, 400, 3), 60, np.uint8)
    label = np.zeros((400, 400), np.uint8)
    cv2.fillConvexPoly(label, np.array([[0,y-24],[195,y-12],[195,y+12],[0,y+24]]), 1)
    cv2.ellipse(label, (195,y), (12,12), 0, 0, 360, 1, -1)
    if disconnected: label[:,80:100] = 0
    frame[label == 1] = [220,100,35]
    if white: frame[(label == 1) & (np.indices(label.shape)[1] >= 185)] = 240
    p = np.zeros((3,400,400), np.float32); p[0] = .98; p[2] = .01; p[1] = .01
    p[0,label == 1] = .03; p[1,label == 1] = .02; p[2,label == 1] = .95
    return frame, p, label


class FastObservationTests(unittest.TestCase):
    def test_grid_native_centers_and_anisotropic_mapping(self):
        np.testing.assert_allclose(grid_to_native([[0,0],[159,159]],(480,640)), [[1.5,1],[637.5,478]])

    def test_bilinear_invalid_border_never_supports_fake_zero_edge(self):
        a = np.full((5,7,3),100,np.uint8)
        value, good = sample(a, [[0,0],[6,4],[-.1,2],[3,4.01]])
        self.assertEqual(good.tolist(),[True,True,False,False]); np.testing.assert_allclose(value,100)

    def test_current_taper_has_directed_shaft_and_bounded_front(self):
        f,p,_ = scene(); obs = FastFiberObserver(CFG).current(f,p)
        self.assertIsNotNone(obs['strong_label'])
        c = next(c for c in obs['candidates'] if c['label'] == obs['strong_label'])
        self.assertTrue(c['shaft_observed']); self.assertTrue(c['front_observed'])
        self.assertLess(abs(c['visible_front_candidate'][0]-207),10)
        self.assertGreaterEqual(c['interior_valid_triplets'],5)
        json.dumps(public_observation(obs),allow_nan=False)

    def test_white_low_saturation_front_is_not_hue_gated(self):
        f,p,_ = scene(white=True); obs = FastFiberObserver(CFG).current(f,p)
        self.assertTrue(any(c['front_observed'] for c in obs['candidates']))

    def test_flat_rgb_rejects_even_high_neural_support(self):
        f,p,_ = scene(); f[:] = 100
        obs = FastFiberObserver(CFG).current(f,p)
        self.assertTrue(all(not c['shaft_observed'] for c in obs['candidates']))

    def test_disconnected_support_is_not_filled_or_mutated(self):
        f,p,label = scene(disconnected=True); before = p.copy(); before_frame = f.copy()
        obs = FastFiberObserver(CFG).current(f,p)
        np.testing.assert_array_equal(p,before); np.testing.assert_array_equal(f,before_frame)
        self.assertLessEqual(len(obs['candidates']),2); self.assertFalse(label[:,90].any())

    def test_no_seeds_and_no_history_cannot_invent_geometry(self):
        f,p,_ = scene(); p[2] = .0002
        obs = FastFiberObserver(CFG).current(f,p)
        self.assertEqual(obs['candidates'],[]); self.assertIsNone(FastFiberObserver(CFG).history(f,p,None))

    def test_equal_qualified_current_candidates_are_ambiguous(self):
        cs = [dict(label=k, score=.9, level='strong',oriented=True,axis_theta=0.,_support_ids=[k*10+i for i in range(4)]) for k in (1,2)]
        result = select_current(cs, CFG)
        self.assertTrue(result['ambiguous']); self.assertIsNone(result['strong_label'])
        cs[1]['_support_ids'] = list(cs[0]['_support_ids'])
        result = select_current(cs, CFG)
        self.assertFalse(result['ambiguous']); self.assertEqual(result['strong_label'],1)

    def test_history_never_promoted_to_independent_strong_and_needs_current_evidence(self):
        f,p,_ = scene(); observer = FastFiberObserver(CFG)
        prior = {'shaft':[0,200,0,48], 'endpoint':[207,24], 'shaft_extent':200}
        c = observer.history(f,p,prior)
        self.assertEqual(c['provenance'],'history_conditioned'); self.assertNotEqual(c['level'],'strong')
        f[:] = 100; c = observer.history(f,p,prior); self.assertFalse(c['shaft_observed'])

    def test_proposal_cap_and_deterministic_order(self):
        f,p,_ = scene(); cv2.rectangle(p[2],(320,40),(360,150),.9,-1); cv2.rectangle(p[2],(300,260),(350,370),.9,-1)
        o = FastFiberObserver(CFG); a,_ = o.proposals(f,p); b,_ = o.proposals(f,p)
        self.assertEqual(len(a),2); self.assertEqual([x['label'] for x in a],[x['label'] for x in b])
        self.assertTrue(all(len(x['support'])<=2048 for x in a))


if __name__ == '__main__': unittest.main()
