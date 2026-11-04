import copy,unittest
import numpy as np
from tools.fiber_reference_workspace import ROOT,read
from tools.fiber_tapered_tangent_cap import TaperedTangentCap,raster_geometry as original_raster
from tools.fiber_rgb_corrected_taper import RGBCorrectedTaper,geometry_from_lines,raster_geometry
from tests.test_tapered_tangent_cap import scene
class RGBCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.cfg=read(ROOT/'configs/tapered_rgb_control_20261104.json');self.basecfg=read(ROOT/self.cfg['baseline_config']);self.im,self.raw,self.p,self.old=scene();self.base=TaperedTangentCap(self.basecfg);self.base.run(self.im,self.raw,self.p,self.old)
    def run_arm(self,arm):
        g=RGBCorrectedTaper(self.cfg,arm);g.run(self.im,self.raw,self.old,self.base.final_mask,self.base.info,self.base.layers['subject']>0);return g
    def test_original_circle_formula_is_same_function(self):self.assertIs(raster_geometry,original_raster)
    def test_all_factor_arms_produce_actual_tangent_mask(self):
        for arm in self.cfg['arms']:
            g=self.run_arm(arm);self.assertTrue(g.info['applied'],g.info);np.testing.assert_array_equal(g.final_mask==2,g.layers['geometric_proposal']>0);self.assertLess(max(g.info['cap']['circle_line_distance_errors']),1e-7)
    def test_search_only_preserves_line_direction_and_observations(self):
        g=self.run_arm('independent_search')
        for a,b in zip(self.base.info['geometry']['independent_lines'],g.info['geometry']['independent_lines']):np.testing.assert_array_equal(a['direction'],b['direction']);np.testing.assert_array_equal(a['points'],b['points'])
    def test_RGB_point_replacement_never_forces_expansion(self):
        import cv2
        self.old=(2*cv2.dilate((self.old==2).astype(np.uint8),np.ones((5,5),np.uint8))).astype(np.uint8);self.base.run(self.im,self.raw,self.p,self.old)
        g=self.run_arm('RGB_sides');self.assertEqual(len(g.info['RGB_edges']),2);self.assertTrue(any(np.min(x['outward_offset'])<0 for x in g.info['RGB_edges']))
    def test_insufficient_edges_fallback_actual_coordinates_not_proposal(self):
        cfg=copy.deepcopy(self.cfg);cfg['minimum_RGB_edge_points']=10000;g=RGBCorrectedTaper(cfg,'RGB_sides');g.run(self.im,self.raw,self.old,self.base.final_mask,self.base.info,self.base.layers['subject']>0);self.assertFalse(g.info['applied']);np.testing.assert_array_equal(g.final_mask,self.base.final_mask);self.assertEqual(g.info['actual_tip'],self.base.info['tip'])
    def test_inputs_not_changed_and_no_video_state(self):
        before=[a.copy() for a in [self.im,self.raw,self.old,self.base.final_mask]];g=self.run_arm('RGB_sides_independent_search')
        for a,b in zip(before,[self.im,self.raw,self.old,self.base.final_mask]):np.testing.assert_array_equal(a,b)
        g.reset();self.assertIsNone(g.final_mask);self.assertEqual(g.layers,{})
    def test_native_resolution_scaling_keeps_original_fit_limits(self):
        g=self.base.info['geometry'];body=self.base.layers['subject']>0;lines=[{k:np.asarray(v) if k in ['point','direction','points'] else v for k,v in a.items()} for a in g['independent_lines']];r,e=geometry_from_lines(body,lines,self.basecfg['fit'])
        self.assertFalse(e,e);np.testing.assert_allclose(r['intersection'],g['intersection'],atol=1e-8)
if __name__=='__main__':unittest.main()
