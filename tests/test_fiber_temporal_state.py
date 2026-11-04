import copy,json
from pathlib import Path
import unittest
import numpy as np
from tools.fiber_temporal_state import GatedGeometryTracker,GeometryStateControl,geometry_features,valid_geometry

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/fiber_temporal_state_gate_20261030.json').read_text(encoding='utf-8'))
OLD=json.loads((ROOT/'configs/fiber_geometry_prior_20261028.json').read_text(encoding='utf-8'))
def state(y=200, length=100, angle=0):return [0,y,angle,length,20,10,1]
def tracker():return GatedGeometryTracker(CFG['state'],CFG['innovation_limits'])

class FiberTemporalStateTests(unittest.TestCase):
    def test_isolated_outlier_does_not_replace_known_stationary_state(self):
        t=tracker();t.step(state(),(400,400),1/30)
        result=t.step(state(y=300),(400,400),1/30)
        json.dumps(result,allow_nan=False)
        self.assertEqual(result['status'],'rejected_jump_pending');np.testing.assert_allclose(result['state'],state())
        self.assertEqual(result['pending_count'],1);self.assertFalse(result['uses_current_observation'])
        result=t.step(state(),(400,400),1/30);self.assertEqual(result['pending_count'],0)
        np.testing.assert_allclose(result['state'],state())

    def test_persistent_step_is_confirmed_at_third_observation_only(self):
        t=tracker();t.step(state(),(400,400),1/30)
        outputs=[copy.deepcopy(t.step(state(y=300),(400,400),1/30)) for _ in range(3)]
        self.assertEqual([r['pending_count'] for r in outputs],[1,2,0])
        self.assertEqual(outputs[2]['status'],'confirmed_relocation')
        np.testing.assert_allclose(outputs[0]['state'],state());np.testing.assert_allclose(outputs[1]['state'],state())
        np.testing.assert_allclose(outputs[2]['state'],state(y=300));self.assertEqual(outputs[2]['age'],0)

    def test_missing_expires_fourth_and_breaks_confirmation(self):
        t=tracker();t.step(state(),(400,400),1/30);t.step(state(y=300),(400,400),1/30)
        r=t.step(None,(400,400),1/30);self.assertEqual(r['pending_count'],0);self.assertEqual(r['age'],2)
        self.assertIsNotNone(t.step(None,(400,400),1/30)['state'])
        r=t.step(None,(400,400),1/30);self.assertIsNone(r['state']);self.assertEqual(r['status'],'expired_reset')

    def test_expiry_does_not_bypass_pending_confirmation(self):
        t=tracker();t.step(state(),(400,400),1/30)
        for _ in range(3):t.step(None,(400,400),1/30)
        r=t.step(state(y=300),(400,400),1/30);self.assertIsNone(r['state']);self.assertEqual(r['pending_count'],1)
        r=t.step(state(y=300),(400,400),1/30);self.assertIsNone(r['state']);self.assertEqual(r['pending_count'],2)
        r=t.step(state(y=300),(400,400),1/30);self.assertEqual(r['status'],'confirmed_relocation')

    def test_changing_false_candidates_cannot_confirm(self):
        t=tracker();t.step(state(),(400,400),1/30)
        for y in [300,100,300,100,300,100]:
            r=t.step(state(y=y),(400,400),1/30);self.assertEqual(r['pending_count'],1)
            self.assertNotEqual(r['status'],'confirmed_relocation')

    def test_pending_branch_follows_known_constant_velocity_and_resets_velocity_on_confirmation(self):
        t=tracker();t.step(state(),(400,400),1/30)
        results=[t.step(state(y=y),(400,400),1/30) for y in [300,310,320]]
        self.assertEqual(results[-1]['status'],'confirmed_relocation')
        np.testing.assert_allclose(results[-1]['state'],state(y=320))
        np.testing.assert_allclose(t.velocity,np.zeros(7))

    def test_angle_wrap_and_velocity_dt_units(self):
        t=tracker();t.step(state(angle=np.deg2rad(179)),(400,400),.1)
        r=t.step(state(angle=np.deg2rad(-179)),(400,400),.1)
        self.assertEqual(r['status'],'updated');self.assertLess(abs(abs(r['state'][2])-np.pi),.03)
        a=tracker();b=tracker()
        for t in [a,b]:t.step(state(),(400,400),.1)
        a.step(state(length=102),(400,400),.1);b.step(state(length=102),(400,400),.2)
        self.assertAlmostEqual(a.velocity[3],2*b.velocity[3])
        for _ in range(80):a.step(state(length=102),(400,400),.1)
        self.assertLess(abs(a.state[3]-102),1e-5)

    def test_context_changes_reset_both_branches_and_do_not_extrapolate_large_gap(self):
        for next_source,next_shape,next_time,reason in [('b',(400,400),.1,'source_changed'),('a',(500,400),.1,'shape_changed'),('a',(400,400),0,'nonmonotonic_timestamp'),('a',(400,400),1,'timestamp_gap')]:
            c=GeometryStateControl('gated_state_only',CFG,OLD);c.step(state(),(400,400),0,'a');c.step(state(y=300),(400,400),.03,'a')
            r=c.step(state(y=320),next_shape,next_time,next_source)
            self.assertEqual(r['context_reset_reason'],reason);self.assertEqual(r['status'],'initialized');self.assertEqual(r['pending_count'],0)

    def test_prefix_causality_input_immutability_and_static_permutation(self):
        sequence=[state(length=100+i) for i in range(10)];sequence[5]=state(y=300);original=copy.deepcopy(sequence)
        def run(seq):
            c=GeometryStateControl('gated_state_only',CFG,OLD)
            return [c.step(z,(400,400),i/30,'v') for i,z in enumerate(seq)]
        self.assertEqual(run(sequence)[:6],run(sequence[:6]));self.assertEqual(sequence,original)
        def static(seq):return {str(z):GeometryStateControl('gated_state_only',CFG,OLD).step(z,(400,400),0,str(z)) for z in seq}
        self.assertEqual(static(sequence),static(sequence[::-1]))

    def test_invalid_measurements_predictions_and_joint_geometry(self):
        for a in [state(length=-1),[0,0,0,1,2,10,1],[0,0,0,np.nan,2,1,1]]:
            self.assertFalse(valid_geometry(a));self.assertIsNone(geometry_features(a))
        t=tracker();t.step(state(),(400,400),.1);t.velocity[3]=-2000
        r=t.step(state(),(400,400),.1);self.assertIsNone(r['state']);self.assertEqual(r['status'],'invalid_prediction_reset')
        g=geometry_features(state());np.testing.assert_allclose(g['tip_xy'],[100,200]);np.testing.assert_allclose(g['cap_center_xy'],[95,200])
        self.assertAlmostEqual(g['signed_taper_angle_rad'],2*np.arctan2(10,190))

    def test_controls_do_not_offer_visible_mask_application(self):
        for arm in CFG['arms']:
            c=GeometryStateControl(arm,CFG,OLD);r=c.step(state(),(400,400),0,'v')
            json.dumps(r,allow_nan=False)
            self.assertFalse(r['apply_to_mask']);self.assertIsNone(r['tip_metrics'])
            if arm=='raw2':self.assertIsNone(r['state'])
        c=GeometryStateControl('legacy_state_only',CFG,OLD);c.step(state(),(400,400),0,'v')
        self.assertEqual(c.step(state(y=300),(400,400),1/30,'v')['status'],'measurement_jump_reset')

if __name__=='__main__':unittest.main()
