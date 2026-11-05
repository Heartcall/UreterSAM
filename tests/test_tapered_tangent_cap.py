import copy,math,unittest
from unittest.mock import patch
import numpy as np
from tools.fiber_reference_workspace import ROOT,read
from tools.fiber_tapered_tangent_cap import TaperedTangentCap,fit_geometry,raster_geometry

def analytic_geometry(alpha=.12,V=None):
    V=np.array([300.,200.]) if V is None else np.asarray(V,float);u=np.array([1.,0.]);n=np.array([0.,1.]);o=V-150*u
    lines=[dict(point=V.copy(),direction=np.array([math.cos(alpha),sign*math.sin(alpha)])) for sign in [1,-1]]
    return dict(origin=o,axis=u,normal=n,intersection=V,half_angle_radians=alpha,independent_lines=lines)

def scene():
    g=analytic_geometry();T=np.array([270.,200.]);body,*_=raster_geometry((400,400),g,T)
    y,x=np.indices(body.shape);frame=np.full((400,400,3),(45,50,80),np.uint8);frame[body>0]=(210,100,15)
    raw=np.zeros(body.shape,np.uint8);raw[(body>0)&(x<130)]=2
    p=np.zeros((3,400,400),np.float32);p[0]=.99;p[2]=.01;p[0,raw==2]=.005;p[2,raw==2]=.99
    return frame,raw,p,np.where(body>0,2,0).astype(np.uint8)

class TaperedTangentTests(unittest.TestCase):
    def setUp(self):self.cfg=read(ROOT/'configs/tapered_tangent_cap_no_fallback_20261104.json')
    def test_exact_original_line_tangencies_nonsemicircle(self):
        g=analytic_geometry(.2);T=np.array([270.,200.]);mask,rod,cap,c,fail=raster_geometry((400,400),g,T)
        self.assertIsNone(fail);self.assertAlmostEqual(c['radius'],30*math.sin(.2)/(1-math.sin(.2)))
        self.assertAlmostEqual(c['arc_span_degrees'],180-math.degrees(.4));self.assertFalse(c['parallel_projection'])
        for line,q in zip(g['independent_lines'],c['join_points']):
            v=line['direction'];normal=np.array([-v[1],v[0]])
            self.assertAlmostEqual(float((q-line['point'])@normal),0,places=10)
            self.assertAlmostEqual(float((q-c['center'])@v),0,places=10)
            self.assertAlmostEqual(float(abs((c['center']-line['point'])@normal)),c['radius'],places=10)
        np.testing.assert_array_equal(mask,rod|cap);self.assertLess(max(c['analytic_join_angle_degrees']),1e-5)
        self.assertGreater(int(mask[:,50].sum()),int(mask[:,200].sum()))
    def test_tip_and_arc_have_same_axis_vertex(self):
        g=analytic_geometry(.25);T=np.array([280.,200.]);_,_,_,c,_=raster_geometry((400,400),g,T)
        np.testing.assert_allclose(c['curve'][64],T,atol=1e-10)
        self.assertAlmostEqual(c['axis_distance'],0);self.assertLess(np.max(abs(np.linalg.norm(c['curve']-c['center'],axis=1)-c['radius'])),1e-10)
    def test_fit_preserves_convergence_and_exact_bisector(self):
        im,raw,p,old=scene();g,e=fit_geometry(old==2,np.array([1.,0.]),self.cfg['fit'],1)
        self.assertFalse(e,e);self.assertGreater(g['side_angle_degrees'],10)
        self.assertGreater(g['independent_lines'][0]['direction'][1],0);self.assertLess(g['independent_lines'][1]['direction'][1],0)
        self.assertAlmostEqual(float((g['intersection']-g['origin'])@g['normal']),0,places=10)
        self.assertNotIn('sides_not_parallel',e);self.assertNotIn('width_unstable',e)
    def test_actual_tapered_replacement_applies(self):
        im,raw,p,old=scene();g=TaperedTangentCap(self.cfg);g.run(im,raw,p,old)
        self.assertTrue(g.info['applied'],g.info);np.testing.assert_array_equal(g.final_mask==2,g.layers['geometric_proposal']>0)
        self.assertLess(g.info['cap']['arc_span_degrees'],180);self.assertNotAlmostEqual(g.info['cap']['radius'],g.info['geometry']['width']/2,delta=1)
    def test_direct_geometry_fills_internal_gap_removes_old_outlier(self):
        im,raw,p,old=scene();old[199:202,170:174]=0;im[199:202,170:174]=0;old[300:304,300:304]=2
        g=TaperedTangentCap(self.cfg);g.run(im,raw,p,old);self.assertTrue(g.info['applied'],g.info)
        self.assertTrue(np.all(g.final_mask[199:202,170:174]==2));self.assertFalse((g.final_mask[300:304,300:304]==2).any());self.assertGreater(g.info['removed_old_pixels'],0);self.assertGreater(g.info['geometry_without_blue_pixels'],0)
    def test_dark_low_probability_tip_not_pixel_growth(self):
        im,raw,p,old=scene();y,x=np.indices(old.shape);im[(old==2)&(x>=200)]=(65,24,8);old[:,200:]=0
        with patch('tools.fiber_weak_blue_evidence.GrowthPipeline.expand',side_effect=AssertionError('weak growth')),patch('tools.fiber_weak_stage2.GrowthPipeline.expand',side_effect=AssertionError('S2 growth')):
            g=TaperedTangentCap(self.cfg);g.run(im,raw,p,old)
        self.assertTrue(g.info['applied'],g.info);self.assertAlmostEqual(g.info['tip']['vertex'][0],270,delta=1);self.assertLess(p[2,200,270],.1)
    def test_farthest_blue_projection_axis(self):
        im,raw,p,old=scene();g=TaperedTangentCap(self.cfg);g.run(im,raw,p,old);q=g.info['geometry'];tip=g.info['tip'];T=np.array(tip['vertex']);O=np.array(q['origin']);u=np.array(q['axis']);n=np.array(q['normal']);source=np.array(tip['source_pixel'])
        self.assertAlmostEqual(float((T-O)@n),0,places=10);self.assertAlmostEqual(float((source-O)@u),tip['tip_t'])
        y,x=np.nonzero(g.layers['tip_eligible']);self.assertAlmostEqual(float(((np.c_[x,y]-O)@u).max()),tip['tip_t'])
    def test_impossible_tip_not_moved_to_satisfy_geometry(self):
        g=analytic_geometry();T=g['intersection']+3*g['axis'];*_,reason=raster_geometry((400,400),g,T)
        self.assertEqual(reason,'tip_at_or_beyond_side_intersection')
    def test_no_trusted_identity_is_unavailable_not_fallback(self):
        im,raw,p,old=scene();raw[:]=0;g=TaperedTangentCap(self.cfg);g.run(im,raw,p,old);self.assertFalse(g.info['applied']);self.assertEqual(g.info['status'],'unavailable');np.testing.assert_array_equal(g.final_mask,old)
    def test_stone_overlap_is_clipped_without_fallback(self):
        im,raw,p,old=scene();raw[198:202,220:224]=1;old[198:202,220:224]=1;g=TaperedTangentCap(self.cfg);g.run(im,raw,p,old);self.assertTrue(g.info['applied']);self.assertGreater(g.info['stone_clipped_pixels'],0);self.assertIn('geometric_stone_overlap_clipped',g.info['nonblocking_diagnostics']);self.assertTrue(np.all(g.final_mask[raw==1]==1))
    def test_frozen_legacy_config_keeps_historical_fallback(self):
        cfg=read(ROOT/'configs/tapered_tangent_cap_control_20261104.json');im,raw,p,old=scene();raw[198:202,220:224]=1;old[198:202,220:224]=1
        g=TaperedTangentCap(cfg);g.run(im,raw,p,old);self.assertFalse(g.info['applied']);self.assertEqual(g.info['status'],'fallback');self.assertIn('geometric_stone_conflict',g.info['rejections']);np.testing.assert_array_equal(g.final_mask,old)
    def test_forward_stone_does_not_reject_tip_candidate(self):
        im,raw,p,old=scene();im[194:207,272:279]=(80,80,80);raw[194:207,272:279]=1;old[194:207,272:279]=1
        g=TaperedTangentCap(self.cfg);g.run(im,raw,p,old);self.assertTrue(g.info['applied'],g.info);self.assertFalse(g.info['tip']['forward_stone_rejection_enabled']);self.assertGreater(g.info['tip']['forward_stone_candidates'],0);self.assertTrue(g.info['tip']['selected_forward_stone'])
    def test_quality_and_area_checks_are_nonblocking(self):
        cfg=copy.deepcopy(self.cfg);cfg['fit']['minimum_observed_aspect']=999.;cfg['geometry']['maximum_frame_area_fraction']=0.
        im,raw,p,old=scene();g=TaperedTangentCap(cfg);g.run(im,raw,p,old);self.assertTrue(g.info['applied'],g.info);self.assertIn('subject_not_slender',g.info['nonblocking_diagnostics']);self.assertIn('geometric_area_budget',g.info['nonblocking_diagnostics'])
    def test_inputs_unchanged_and_reset(self):
        im,raw,p,old=scene();inputs=[z.copy() for z in [im,raw,p,old]];g=TaperedTangentCap(self.cfg);g.run(im,raw,p,old)
        for a,b in zip(inputs,[im,raw,p,old]):np.testing.assert_array_equal(a,b)
        g.reset();self.assertEqual(g.layers,{});self.assertIsNone(g.final_mask)
    def test_no_parallel_width_variation_or_seed_retention_gate(self):
        self.assertNotIn('maximum_side_angle_degrees',self.cfg['fit']);self.assertNotIn('maximum_width_robust_CV',self.cfg['fit']);self.assertNotIn('maximum_seed_loss_fraction',self.cfg['geometry'])

if __name__=='__main__':unittest.main()
