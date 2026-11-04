import unittest
import numpy as np
from tools.fiber_reference_workspace import ROOT,read
from tools.fiber_straight_semicircle import robust_line
from tools.fiber_rgb_side_observation import huber_diagnostics,edge_observations,broad_search_domain,terminal_observations
from tools.fiber_connected_blue_growth import chroma
class RGBObservationTests(unittest.TestCase):
    def setUp(self):
        self.cfg=read(ROOT/'configs/tapered_rgb_diagnosis_20261104.json');self.tip=read(ROOT/'configs/tapered_tangent_cap_control_20261104.json')['tip']
        self.im=np.full((400,400,3),(45,50,80),np.uint8);self.im[180:221,:260]=(210,100,15);self.raw=np.zeros((400,400),np.uint8)
        self.seed=dict(mu=chroma(np.array([210,100,15]))[0],bg=chroma(np.array([45,50,80]))[0],gap=.5)
    def test_coherent_inward_points_keep_full_Huber_weights(self):
        p=np.c_[np.arange(20,150),np.full(130,185.)];line=robust_line(p,np.array([1.,0.]),1.5,15);h=huber_diagnostics(line,1.5)
        self.assertEqual(h['fully_weighted_fraction'],1.)
    def test_RGB_exit_recovers_shift_and_original_coordinates(self):
        p=np.c_[np.arange(20,150,3),np.full(44,185.)];a=edge_observations(self.im,self.raw,p,np.array([1.,0.]),-1,self.seed,self.tip,self.cfg['edge'],1)
        self.assertTrue(a['valid'].all());self.assertAlmostEqual(float(np.median(a['points'][:,1])),179.5,delta=1.5);self.assertGreater(float(np.median(a['gradient'])),0)
    def test_outward_search_can_move_inward_for_oversegmented_old_mask(self):
        p=np.c_[np.arange(20,150,3),np.full(44,175.)];a=edge_observations(self.im,self.raw,p,np.array([1.,0.]),-1,self.seed,self.tip,self.cfg['edge'],1)
        self.assertTrue(a['valid'].all());self.assertLess(np.median(a['outward_offset']),0)
    def test_domain_does_not_use_line_intersection_or_convergence(self):
        body=np.zeros((400,400),bool);body[180:221,:240]=True;g=dict(origin=[100.,200.],axis=[1.,0.],normal=[0.,1.],body_extent=[-100.,140.],intersection=[250,200],half_angle_radians=.5)
        a=broad_search_domain(body.shape,body,g,dict(self.tip,independent_side_margin_at400=4),1);g['intersection']=[150,200];g['half_angle_radians']=.01
        np.testing.assert_array_equal(a,broad_search_domain(body.shape,body,g,dict(self.tip,independent_side_margin_at400=4),1));self.assertTrue(a[200,260])
    def test_external_terminal_and_low_probability_independent(self):
        body=np.zeros((400,400),bool);body[180:221,:240]=True;g=dict(origin=np.array([100.,200.]),axis=np.array([1.,0.]),normal=np.array([0.,1.]),body_extent=[-100.,140.])
        d=broad_search_domain(body.shape,body,g,dict(self.tip,independent_side_margin_at400=4),1);e,m=terminal_observations(self.im,self.raw,g,self.seed,self.tip,d)
        self.assertIsNotNone(m['vertex']);self.assertAlmostEqual(m['vertex'][0],259);self.assertTrue(e.any())
    def test_full_diagnostic_payload_serializes_without_changing_observation(self):
        import json
        from tools.fiber_local_dark_evidence import public
        p=np.c_[np.arange(20,150,3),np.full(44,185.)];a=edge_observations(self.im,self.raw,p,np.array([1.,0.]),-1,self.seed,self.tip,self.cfg['edge'],1)
        q=json.loads(json.dumps(public(dict(RGB_observations=[a])),allow_nan=False));self.assertEqual(len(q['RGB_observations'][0]['points']),44);self.assertIsInstance(a['points'],np.ndarray)
if __name__=='__main__':unittest.main()
