import copy,unittest
import cv2,numpy as np
from tools.fiber_reference_workspace import ROOT,read
from tools.fiber_blue_extent import BlueExtent,GrowthPipeline,compatible_history

def configuration():return read(ROOT/'configs/fiber_blue_extent_control_20261101.json')
def scene(straight=True):
    image=np.full((400,400,3),(45,50,80),np.uint8);truth=np.zeros((400,400),np.uint8)
    vertices=[[0,180],[0,220],[300,220],[300,180]] if straight else [[0,165],[0,235],[300,202],[300,198]]
    cv2.fillConvexPoly(truth,np.array(vertices,np.int32),1)
    for x in range(301):image[:,x][truth[:,x]>0]=np.rint(np.array([210,100,15])*max(.22,1-x/380)).astype(np.uint8)
    raw=np.zeros_like(truth);raw[(truth>0)&(np.indices(truth.shape)[1]<135)]=2
    prob=np.zeros((3,400,400),np.float32);prob[0]=.98;prob[2]=.01;prob[2][raw==2]=.99;prob[0][raw==2]=.005
    return image,raw,prob,truth

class ExtentTests(unittest.TestCase):
    def setUp(self):self.cfg=configuration();self.g=BlueExtent(self.cfg['extent'])
    def test_blue_brightness_invariance_above_noise(self):
        seed=dict(mu=np.array([15,100,210])/325,v=np.array([15,100,210])/325-1/3,bg=np.array([.5,.3,.2]),gap=.5)
        colors=np.array([[210,100,15],[105,50,8],[42,20,3]],np.uint8)
        self.assertTrue(self.g.color(colors,seed).all())
    def test_black_is_unknown_not_normalized_blue(self):
        seed=dict(mu=np.array([.05,.3,.65]),v=np.array([.05,.3,.65])-1/3,bg=np.array([.5,.3,.2]),gap=.5)
        self.assertFalse(self.g.color(np.zeros((1,3),np.uint8),seed)[0])
    def test_dark_extension_with_original_probability_below_point_one(self):
        im,raw,p,t=scene();c,proposal,conf,m=self.g.grow(im,raw,p)
        self.assertGreater(int(c.sum()),0,m);self.assertTrue((p[2][c>0]<.1).all());self.assertFalse(np.any(c[raw!=0]));self.assertFalse(np.any(c[t==0]))
    def test_raster_taper_rejection_is_preserved_not_tuned_away(self):
        im,raw,p,_=scene(False);c,_,_,m=self.g.grow(im,raw,p)
        self.assertFalse(c.any());self.assertEqual(m['reason'],'direction_or_curvature_jump')
    def test_true_black_gap_blocks_later_blue(self):
        im,raw,p,t=scene();im[:,170:178]=0;c,_,_,m=self.g.grow(im,raw,p)
        self.assertFalse(c[:,178:].any(),m)
    def test_independent_blue_island_never_joins(self):
        im,raw,p,t=scene();im[40:65,260:290]=(180,90,15);c,_,_,_=self.g.grow(im,raw,p)
        self.assertFalse(c[40:65,260:290].any())
    def test_stone_is_connectivity_barrier_and_preserved(self):
        im,raw,p,t=scene();raw[:,175:182]=1;c,_,conf,m=self.g.grow(im,raw,p)
        self.assertFalse(c[raw==1].any());self.assertFalse(c[:,182:].any());self.assertGreater(conf.sum(),0,m)
    def test_inputs_immutable(self):
        im,raw,p,_=scene();old=[x.copy() for x in [im,raw,p]];self.g.grow(im,raw,p)
        for a,b in zip([im,raw,p],old):np.testing.assert_array_equal(a,b)
    def test_history_has_angle_width_distance_and_age_gates(self):
        cfg=self.cfg['extent']['history'];prior=[0,200,0,50];u=np.array([1.,0.]);center=np.array([100.,200.])
        self.assertTrue(compatible_history(prior,0,center,u,40,(400,400),cfg))
        for z,age in [([0,200,1.5,50],0),([0,20,0,50],0),([0,200,0,2],0),(prior,4),(None,0)]:self.assertFalse(compatible_history(z,age,center,u,40,(400,400),cfg))
    def test_real_single_weak_side_needs_history_and_has_finite_run(self):
        im,raw,p,t=scene();seed=self.g.seeds(im,raw,p,None,99)[0][0]
        im[:,150:][t[:,150:]>0]=(18,8,2);im[:180,150:]=(10,8,6)
        sec,reason=self.g.section(im,np.array([210.,200.]),np.array([1.,0.]),seed['width'],seed)
        self.assertIsNotNone(sec,reason);self.assertTrue(sec['weak']);self.assertEqual(sec['states'].count('observed_strong'),1)
        seed=copy.copy(seed);seed['start_sections']=[sec,sec]
        rejected=self.g.walk(im,seed,1,None,99);accepted=self.g.walk(im,seed,1,[0,200,0,40],0)
        self.assertEqual(len(rejected['polygons']),0);self.assertEqual(rejected['stop_reason'],'weak_requires_compatible_history')
        self.assertGreater(len(accepted['polygons']),0);self.assertLessEqual(len(accepted['polygons']),2)
        self.assertIn(accepted['stop_reason'],['weak_run_limit','direction_or_curvature_jump'])
    def test_both_weak_sides_not_accepted_by_color_alone(self):
        im,raw,p,t=scene();seed=self.g.seeds(im,raw,p,None,99)[0][0]
        im[:,150:]=(10,8,6);im[:,150:][t[:,150:]>0]=(18,8,2)
        sec,reason=self.g.section(im,np.array([210.,200.]),np.array([1.,0.]),seed['width'],seed)
        self.assertIsNone(sec);self.assertEqual(reason,'both_sides_not_strong')
    def test_history_without_current_seed_cannot_make_pixels(self):
        im,raw,p,_=scene();raw[:]=0;p[2]=0;c,_,_,m=self.g.grow(im,raw,p,[0,200,0,50],0);self.assertFalse(c.any());self.assertEqual(m['reason'],'no_trusted_seed')
    def test_strong_current_does_not_depend_on_old_prior(self):
        im,raw,p,_=scene();a=self.g.grow(im,raw,p)[0];b=self.g.grow(im,raw,p,[200,0,1.5,1],0)[0];np.testing.assert_array_equal(a,b)
    def test_tip_attempted_even_when_no_new_pixels(self):
        im,raw,p,t=scene();raw[t>0]=2;c,_,_,m=self.g.grow(im,raw,p)
        self.assertFalse(c.any());self.assertGreater(len(m['tips']),0,m)
    def test_search_stop_not_asserted_tip(self):
        im,raw,p,_=scene();seeds,_=self.g.seeds(im,raw,p,None,99);seed=seeds[0];branch=self.g.walk(im,seed,1,None,99);branch['stop_reason']='search_limit';z=self.g.tip(im,raw,seed,branch,1);self.assertIsNone(z['xy']);self.assertEqual(z['state'],'unknown')
    def test_static_has_no_temporal_state(self):
        pipe=GrowthPipeline(self.cfg,'blue_extent',read(ROOT/self.cfg['t1']['config']),False);self.assertIsNone(pipe.tracker)
    def test_video_context_reset_isolated(self):
        pipe=GrowthPipeline(self.cfg,'blue_extent',read(ROOT/self.cfg['t1']['config']),True);im,raw,p,_=scene();pipe.run(im,raw,p,0.,'first');_,_,_,m=pipe.run(im,raw,p,0.,'second');self.assertEqual(m['t1_advice']['context_reset_reason'],'source_changed');pipe.reset();self.assertIsNone(pipe.tracker.source)
    def test_unique_identity_not_forced(self):
        im,raw,p,t=scene();raw[30:55,40:100]=2;p[2][30:55,40:100]=.99;im[30:55,40:100]=(210,100,15)
        raw[330:350,10:100]=2;p[2][330:350,10:100]=.99;im[330:350,10:100]=(210,100,15)
        c,_,_,m=self.g.grow(im,raw,p);self.assertFalse(c.any());self.assertEqual(m['seeds'][0]['reason'],'excessive_seed_hypotheses')

if __name__=='__main__':unittest.main()
