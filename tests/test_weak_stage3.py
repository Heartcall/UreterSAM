import unittest
import numpy as np
from tools.fiber_reference_workspace import ROOT,read
from tools.fiber_weak_stage3 import GrowthPipeline,prune_new_weak,frontier,terminal,native_gradient
from tests.test_fiber_blue_outline import scene

class Stage3Tests(unittest.TestCase):
    def setUp(self):self.cfg=read(ROOT/'configs/weak_blue_evidence_control_20261103.json')
    def fixture(self):
        b=np.zeros((80,100),np.uint8);b[30:50,:60]=1;protect=b>0;b[40,60:63]=1;weak=(b>0)&~protect;return b,protect,np.zeros_like(weak),weak,np.full(b.shape,.5),np.zeros(b.shape),protect
    def test_only_unsupported_new_leaves_pruned(self):
        b,p,s,w,q,e,a=self.fixture();r,removed,info=prune_new_weak(b,p,s,w,q,e,a);self.assertGreater(removed.sum(),0);self.assertTrue(np.all(r[p]==1));self.assertFalse((removed[p]>0).any());self.assertTrue(np.all(w[removed>0]));self.assertTrue(np.all(r[removed>0]==0))
    def test_strong_and_gradient_protect(self):
        for support in ['strong','edge']:
            b,p,s,w,q,e,a=self.fixture()
            if support=='strong':s[w]=True
            else:e[w]=.02
            r,removed,_=prune_new_weak(b,p,s,w,q,e,a);np.testing.assert_array_equal(r,b);self.assertFalse(removed.any())
    def test_bridge_not_severed(self):
        b=np.zeros((80,100),np.uint8);b[35:45,:30]=1;b[35:45,60:70]=1;b[40,30:60]=1;p=b>0;p[40,30:60]=False;a=np.zeros_like(p);a[40,20]=True;w=(b>0)&~p;r,removed,info=prune_new_weak(b,p,np.zeros_like(p),w,np.full(b.shape,.5),np.zeros(b.shape),a);np.testing.assert_array_equal(r,b);self.assertEqual(info['connectivity_rollbacks'],1)
    def test_singleton_distal_spur_does_not_confirm(self):
        b=np.zeros((80,100),np.uint8);b[30:50,:60]=1;b[40,60:64]=1;c,f=frontier(b,np.array([0.,40.]),np.array([1.,0.]),self.cfg['weak_blue']['tip'],1.);self.assertGreater(f['unresolved_farther_visible_pixels'],0);self.assertFalse(f['strong_cluster']);self.assertLess(f['blue_front_t'],f['raw_extreme_t'])
    def test_dense_front_is_selected(self):
        b=np.zeros((80,100),np.uint8);b[30:50,:60]=1;c,f=frontier(b,np.array([0.,40.]),np.array([1.,0.]),self.cfg['weak_blue']['tip'],1.);self.assertTrue(f['strong_cluster']);self.assertEqual(f['unresolved_farther_visible_pixels'],0);self.assertEqual(f['blue_front_t'],59.)
    def test_black_gap_stone_union_inputs(self):
        im,raw,p,_=scene();im[:,190:196]=0;raw[:,250:255]=1;copies=[z.copy() for z in [im,raw,p]];g=GrowthPipeline(self.cfg,'weak_blue_extension_tip');g.run(im,raw,p);self.assertFalse(g.layers['reachable_blue'][:,196:].any());self.assertTrue(np.all(g.final_mask[g.layers['union_fiber']>0]==2));self.assertTrue(np.all(g.final_mask[raw==1]==1))
        for a,b in zip(copies,[im,raw,p]):np.testing.assert_array_equal(a,b)
    def test_inside_points_adapt_to_measured_width(self):
        im,raw,p,_=scene();g=GrowthPipeline(self.cfg,'weak_blue_extension_tip');g.run(im,raw,p);obs=g.outline.get('tip_observation',{});self.assertIn('inside_sections',obs)
        for sec in obs['inside_sections']:self.assertGreaterEqual(sec['normal_offset_px'],.5);self.assertLessEqual(sec['normal_offset_px'],1.)
    def test_absent_seed_no_growth(self):
        im,raw,p,_=scene();raw[:]=0;g=GrowthPipeline(self.cfg,'weak_blue_extension_tip');g.run(im,raw,p);self.assertFalse(g.outline['eligible']);self.assertFalse(g.final_mask.any())
if __name__=='__main__':unittest.main()
