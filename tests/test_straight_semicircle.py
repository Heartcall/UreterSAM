import copy,unittest
import cv2,numpy as np
from tools.fiber_reference_workspace import ROOT,read
from tools.fiber_straight_semicircle import StraightSemicircle,robust_line,fit_geometry,raster_geometry,locate_tip

def scene():
    y,x=np.indices((400,400));body=((x<=250)&(abs(y-200)<=10))|(((x-250)**2+(y-200)**2<=100)&(x>=250))
    frame=np.full((400,400,3),(45,50,80),np.uint8);frame[body]=(210,100,15)
    raw=np.zeros((400,400),np.uint8);raw[body&(x<130)]=2
    p=np.zeros((3,400,400),np.float32);p[0]=.99;p[2]=.01;p[0,raw==2]=.005;p[2,raw==2]=.99
    return frame,raw,p,np.where(body,2,0).astype(np.uint8)

class StraightSemicircleTests(unittest.TestCase):
    def setUp(self):self.cfg=read(ROOT/'configs/straight_semicircle_control_20261104.json')
    def test_robust_straight_line_ignores_sparse_outliers(self):
        x=np.arange(100.);pts=np.c_[x,.1*x+30];pts[::17,1]+=12;g=robust_line(pts,np.array([1.,0.]),1.5,15);self.assertAlmostEqual(g['direction'][1]/g['direction'][0],.1,delta=.01)
    def test_exact_circle_tangency_and_axis_tip(self):
        u=np.array([.8,.6]);o=np.array([20.,80.]);T=o+200*u;mask,rod,cap,g=raster_geometry((400,400),o,u,20,T)
        self.assertAlmostEqual(np.linalg.norm(g['center']-T),10);self.assertAlmostEqual(g['axis_distance'],0,places=10)
        for join in g['join_points']:self.assertAlmostEqual(float((join-g['center'])@u),0,places=10);self.assertAlmostEqual(np.linalg.norm(join-g['center']),10)
        np.testing.assert_array_equal(mask,rod|cap);self.assertTrue(all(z==0 for z in g['analytic_join_angle_degrees']))
    def test_actual_direct_reconstruction_applies(self):
        im,raw,p,old=scene();g=StraightSemicircle(self.cfg);g.run(im,raw,p,old);self.assertTrue(g.info['applied'],g.info);self.assertAlmostEqual(g.info['geometry']['width'],20,delta=.05);self.assertAlmostEqual(g.info['tip']['vertex'][0],260,delta=.1)
    def test_old_irregular_union_can_be_removed(self):
        im,raw,p,old=scene();old[300:304,300:304]=2;old[188:191,80:84]=2;g=StraightSemicircle(self.cfg);g.run(im,raw,p,old);self.assertTrue(g.info['applied']);self.assertFalse((g.final_mask[300:304,300:304]==2).any());self.assertGreater(g.info['removed_old_pixels'],0)
    def test_no_per_pixel_blue_intersection(self):
        im,raw,p,old=scene();old[199:202,170:174]=0;im[199:202,170:174]=0;g=StraightSemicircle(self.cfg);g.run(im,raw,p,old);self.assertTrue(g.info['applied']);self.assertTrue(np.all(g.final_mask[199:202,170:174]==2));self.assertGreater(g.info['geometry_without_blue_pixels'],0)
    def test_low_probability_dark_tip_is_found_without_growth(self):
        im,raw,p,old=scene();im[(old==2)&(np.indices(old.shape)[1]>=200)]=(65,24,8);old[:,200:]=0;g=StraightSemicircle(self.cfg);g.run(im,raw,p,old);self.assertTrue(g.info['applied'],g.info);self.assertAlmostEqual(g.info['tip']['vertex'][0],260,delta=.1);self.assertLess(p[2,200,260],.1)
    def test_farthest_observed_pixel_projected_to_axis(self):
        im,raw,p,old=scene();im[201:204,260:263]=(210,100,15);g=StraightSemicircle(self.cfg);g.run(im,raw,p,old);self.assertTrue(g.info['applied']);T=np.array(g.info['tip']['vertex']);o=np.array(g.info['geometry']['origin']);u=np.array(g.info['geometry']['axis']);source=np.array(g.info['tip']['source_pixel']);self.assertAlmostEqual(float((T-o)@u),float((source-o)@u));self.assertAlmostEqual(T[1],200,delta=.1);self.assertNotAlmostEqual(source[1],T[1],delta=.1)
    def test_nonparallel_lines_identified(self):
        y,x=np.indices((400,400));b=((x<250)&(abs(y-200)<60-.2*x)).astype(np.uint8);g,errors=fit_geometry(b,np.array([1.,0.]),self.cfg['fit'],1);self.assertIn('sides_not_parallel',errors);self.assertGreater(g['side_angle_degrees'],15)
    def test_failed_model_retains_old_mask_exactly(self):
        im,raw,p,old=scene();cfg=copy.deepcopy(self.cfg);cfg['fit']['maximum_side_angle_degrees']=-1;g=StraightSemicircle(cfg);g.run(im,raw,p,old);self.assertFalse(g.info['applied']);self.assertEqual(g.info['reason'],'sides_not_parallel');np.testing.assert_array_equal(g.final_mask,old);self.assertTrue(g.layers['geometric_proposal'].any())
    def test_stone_conflict_reverts_whole_shape(self):
        im,raw,p,old=scene();raw[198:202,220:224]=1;old[198:202,220:224]=1;g=StraightSemicircle(self.cfg);g.run(im,raw,p,old);self.assertFalse(g.info['applied']);self.assertIn('geometric_stone_conflict',g.info['rejections']);np.testing.assert_array_equal(g.final_mask,old)
    def test_missing_subject_fallback(self):
        im,raw,p,old=scene();raw[:]=0;g=StraightSemicircle(self.cfg);g.run(im,raw,p,old);self.assertFalse(g.info['applied']);self.assertEqual(g.info['reason'],'no_trusted_subject');np.testing.assert_array_equal(g.final_mask,old)
    def test_inputs_and_reset(self):
        im,raw,p,old=scene();before=[a.copy() for a in [im,raw,p,old]];g=StraightSemicircle(self.cfg);g.run(im,raw,p,old)
        for a,b in zip(before,[im,raw,p,old]):np.testing.assert_array_equal(a,b)
        g.reset();self.assertEqual(g.layers,{});self.assertIsNone(g.final_mask)
if __name__=='__main__':unittest.main()
