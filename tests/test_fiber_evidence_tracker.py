import copy
import json
from pathlib import Path
import unittest
import numpy as np
from tools.fiber_evidence_tracker import EvidenceTracker, empty_current, stateless_result

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT/'configs/fiber_temporal_observation_search_20261030.json').read_text(encoding='utf-8'))


def observation(y=200, front=True, strong=True, oriented=True, provenance='independent_current'):
    c = dict(label=1,level='strong' if strong else 'weak',score=.9,provenance=provenance,
             oriented=oriented,front_observed=front,shaft_observed=True,pose=[0,y,0,100 if front else None,20,10 if front else None],
             far_seed_xy=[100,y], far_seed_width_px=10, observed_shaft_length_px=100,duplicate_of=None)
    return dict(candidates=[c],strong_label=1 if strong and oriented and provenance=='independent_current' else None,winner_label=1,
                ambiguous=False,qualified_labels=[1],duplicates=[])


def step(t,obs,ts,source='v1',history=None):
    return t.update(obs,history,t.prepare((400,400),ts,source))


class EvidenceTrackerTests(unittest.TestCase):
    def test_strong_sudden_step_overrides_in_same_frame_no_three_wait(self):
        t = EvidenceTracker(CFG); step(t,observation(),0)
        r = step(t,observation(y=320),1/30)
        self.assertEqual(r['groups']['shaft']['value'][1],320)
        self.assertIn('shaft',r['strong_override_groups']); self.assertTrue(r['independent_strong_used'])

    def test_weak_jump_rejected_and_expires_without_persistence_promotion(self):
        t = EvidenceTracker(CFG); step(t,observation(),0)
        outputs = [step(t,observation(y=320,strong=False),i/30) for i in range(1,5)]
        self.assertEqual([r['groups']['shaft']['status'] for r in outputs],['predicted']*3+['expired'])

    def test_weak_compatible_updates_but_cannot_initialize(self):
        t = EvidenceTracker(CFG)
        self.assertIsNone(step(t,observation(strong=False),0)['groups']['shaft']['value'])
        step(t,observation(),1/30)
        r = step(t,observation(y=201,strong=False),2/30)
        self.assertEqual(r['groups']['shaft']['status'],'weak_updated')

    def test_missing_front_does_not_shorten_and_group_expires_independently(self):
        t = EvidenceTracker(CFG); step(t,observation(),0)
        r = step(t,observation(front=False),1/30)
        self.assertEqual(r['groups']['endpoint']['value'][0],100)
        self.assertTrue(r['front_is_prediction']); self.assertTrue(r['groups']['shaft']['observed'])
        for i in range(2,5): r = step(t,observation(front=False),i/30)
        self.assertIsNone(r['groups']['endpoint']['value']); self.assertIsNotNone(r['groups']['shaft']['value'])

    def test_invalid_prediction_cannot_block_strong_current_rescue(self):
        t = EvidenceTracker(CFG); step(t,observation(),0)
        t.velocity['shaft'][3] = -10000
        r = step(t,observation(y=205),1/30)
        self.assertEqual(r['groups']['shaft']['value'][1],205)
        self.assertIn('shaft',r['invalid_predictions']); self.assertTrue(r['groups']['shaft']['observed'])

    def test_source_shape_time_resets_and_static_order_independence(self):
        t = EvidenceTracker(CFG); step(t,observation(),0)
        r = step(t,empty_current(),1/30,source='v2'); self.assertIsNone(r['groups']['shaft']['value'])
        self.assertEqual(r['context_reset_reason'],'source_changed')
        step(t,observation(),2/30,source='v2')
        r = step(t,empty_current(),1.,source='v2'); self.assertEqual(r['context_reset_reason'],'timestamp_gap')
        step(t,observation(),1.1,source='v2'); r = step(t,empty_current(),1.1,source='v2')
        self.assertEqual(r['context_reset_reason'],'nonmonotonic_timestamp')
        step(t,observation(),1.2,source='v2'); r = t.prepare((480,640),1.3,'v2')
        self.assertEqual(r['context_reset_reason'],'shape_changed')
        self.assertEqual(stateless_result(observation()),stateless_result(observation()))

    def test_unoriented_line_cannot_initialize_and_history_resolution_is_weak(self):
        t = EvidenceTracker(CFG); r = step(t,observation(oriented=False),0)
        self.assertIsNone(r['groups']['shaft']['value'])
        step(t,observation(),1/30); r = step(t,observation(oriented=False),2/30)
        self.assertEqual(r['groups']['shaft']['status'],'weak_updated'); self.assertTrue(r['orientation_resolved_by_history'])

    def test_history_never_displaces_strong_and_requires_compatible_current_evidence(self):
        t = EvidenceTracker(CFG); step(t,observation(),0)
        h = observation(strong=False,provenance='history_conditioned')['candidates'][0]; h['label']=-1
        r = step(t,observation(y=320),1/30,history=h)
        self.assertEqual(r['selected_label'],1); self.assertEqual(r['groups']['shaft']['value'][1],320)
        r = step(t,empty_current(),2/30,history=h)
        self.assertIsNone(r['used_provenance'])

    def test_prefix_causality_and_json_finite(self):
        seq = [observation(),observation(y=203),empty_current(),observation(y=320),observation(front=False)]
        def replay(rows):
            t = EvidenceTracker(CFG); return [step(t,copy.deepcopy(o),i/30) for i,o in enumerate(rows)]
        full = replay(seq); self.assertEqual(full[:3],replay(seq[:3])); json.dumps(full,allow_nan=False)
        self.assertTrue(all(not r['apply_to_mask'] for r in full))


if __name__ == '__main__': unittest.main()
