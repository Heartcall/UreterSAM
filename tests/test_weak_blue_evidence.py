import unittest,copy
import cv2,numpy as np
from tools.fiber_reference_workspace import ROOT,read
from tools.fiber_weak_blue_evidence import ARMS,LAYERS,GrowthPipeline,evidence,connect,frontier,terminal
from tools.fiber_bisector_blue_tip import GrowthPipeline as Frozen
from tests.test_fiber_blue_outline import scene

class WeakEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg=read(ROOT/'configs/weak_blue_evidence_control_20261103.json');self.q=self.cfg['weak_blue'];self.seed=dict(mu=np.array([.05,.30,.65]),v=np.array([.05,.30,.65])-1/3,bg=np.array([.5,.3,.2]),gap=.6)
    def run_scene(self,arm=ARMS[-1],image=None):
        x=list(scene());x[0]=x[0] if image is None else image;g=GrowthPipeline(self.cfg,arm);g.run(*x[:3]);return g,x
    def test_control_union_exact(self):
        g,x=self.run_scene(ARMS[0]);f=Frozen(read(ROOT/'configs/bisector_blue_tip_control_20261103.json'),ARMS[0]);f.run(*x[:3]);np.testing.assert_array_equal(g.final_mask,f.final_mask)
    def test_control_bisector_exact(self):
        g,x=self.run_scene(ARMS[1]);f=Frozen(read(ROOT/'configs/bisector_blue_tip_control_20261103.json'),ARMS[1]);f.run(*x[:3]);np.testing.assert_array_equal(g.final_mask,f.final_mask)
    def test_layers(self):g,x=self.run_scene();self.assertEqual(set(g.layers),set(LAYERS))
    def test_union_protected(self):g,x=self.run_scene();self.assertTrue(np.all(g.final_mask[g.inner.final_mask==2]==2))
    def test_raw_stone_protected(self):
        im,raw,p,b=scene();raw[:,210:218]=1;g=GrowthPipeline(self.cfg,ARMS[-1]);g.run(im,raw,p);self.assertTrue(np.all(g.final_mask[raw==1]==1))
    def test_inputs_immutable(self):
        im,raw,p,b=scene();before=[z.copy() for z in [im,raw,p]];GrowthPipeline(self.cfg,ARMS[-1]).run(im,raw,p)
        for a,z in zip(before,[im,raw,p]):np.testing.assert_array_equal(a,z)
    def test_dark_scale_color(self):
        for intensity in [1.,.5,.2]:
            im=np.full((12,12,3),np.array([200,90,15])*intensity,np.uint8);a=evidence(im,self.seed,self.q['evidence']);self.assertTrue((a[1]|a[2]).all())
    def test_black_not_evidence(self):a=evidence(np.zeros((9,9,3),np.uint8),self.seed,self.q['evidence']);self.assertFalse((a[1]|a[2]).any())
    def test_neutral_not_evidence(self):a=evidence(np.full((9,9,3),50,np.uint8),self.seed,self.q['evidence']);self.assertFalse((a[1]|a[2]).any())
    def test_red_not_evidence(self):a=evidence(np.full((9,9,3),[10,40,200],np.uint8),self.seed,self.q['evidence']);self.assertFalse((a[1]|a[2]).any())
    def test_evidence_continuous_finite(self):
        im=np.random.default_rng(7).integers(0,255,(20,20,3),dtype=np.uint8);a=evidence(im,self.seed,self.q['evidence']);self.assertTrue(np.isfinite(a[0]).all());self.assertGreater(len(np.unique(a[0])),10)
    def test_grades_disjoint(self):
        im=np.random.default_rng(8).integers(0,255,(20,20,3),dtype=np.uint8);_,s,w,u,*_=evidence(im,self.seed,self.q['evidence']);self.assertFalse((s&w|s&u|w&u).any())
    def test_diagonal_evidence_connects(self):
        b=np.eye(8,dtype=np.uint8);a=np.zeros_like(b,dtype=bool);a[0,0]=True;r=connect(b,a);np.testing.assert_array_equal(r,b)
    def test_no_black_gap_bridge(self):
        b=np.ones((8,12),np.uint8);b[:,5]=0;a=np.zeros_like(b,dtype=bool);a[:,0]=True;r=connect(b,a);self.assertFalse(r[:,6:].any());self.assertFalse(r[:,5].any())
    def test_disconnected_blue_not_added(self):
        b=np.zeros((20,30),np.uint8);b[4:7,2:6]=1;b[4:7,12:19]=1;a=np.zeros_like(b,dtype=bool);a[5,3]=True;r=connect(b,a);self.assertFalse(r[:,12:].any())
    def test_no_seed_no_change(self):
        im,raw,p,b=scene();raw[:]=0;g=GrowthPipeline(self.cfg,ARMS[-1]);g.run(im,raw,p);self.assertFalse(g.final_mask.any());self.assertFalse(g.layers['reachable_blue'].any())
    def test_probability_not_color_gate(self):
        im,raw,p,b=scene();p[:,:,200:]=0;p[0,:,200:]=1;g=GrowthPipeline(self.cfg,ARMS[2]);g.run(im,raw,p);self.assertTrue(g.layers['reachable_blue'][:,205:230].any())
    def test_frontier_cluster_not_extreme_point(self):
        b=np.zeros((40,80),np.uint8);b[15:26,:60]=1;c,f=frontier(b,np.array([0.,0.]),np.array([1.,0.]),self.q['tip'],1);self.assertTrue(f['strong_cluster']);self.assertGreater(f['area'],6)
    def test_spike_weak_not_confirmed(self):
        b=np.zeros((40,80),np.uint8);b[15:26,:40]=1;b[20,40:70]=1;c,f=frontier(b,np.array([0.,0.]),np.array([1.,0.]),self.q['tip'],1);self.assertFalse(f['strong_cluster'])
    def test_search_limited_unknown(self):
        im=np.full((40,80,3),[200,90,15],np.uint8);raw=np.zeros((40,80),np.uint8);r=terminal(im,raw,np.array([40,20]),np.array([1.,0.]),self.seed,self.q['evidence'],self.q['tip'],1,dict(strong_cluster=True),True);self.assertEqual(r['state'],'unknown')
    def test_black_background_unknown(self):
        im=np.zeros((40,80,3),np.uint8);im[:,:41]=[200,90,15];raw=np.zeros((40,80),np.uint8);r=terminal(im,raw,np.array([40,20]),np.array([1.,0.]),self.seed,self.q['evidence'],self.q['tip'],1,dict(strong_cluster=True),False);self.assertEqual(r['reason'],'terminal_background_unknown')
    def test_cap_no_extension_past_front(self):
        g,x=self.run_scene();cap=g.outline['cap']
        if cap.get('accepted'):
            q=g.outline['cap_coordinate_context'];self.assertLessEqual(((np.asarray(cap['curve'])-q['origin'])@q['axis']).max(),g.outline['distal']['blue_front_t']+1e-6)
    def test_reset(self):g,x=self.run_scene();g.reset();self.assertIsNone(g.final_mask);self.assertEqual(g.layers,{})
    def test_same_recovery_across_new_arms(self):
        a,x=self.run_scene(ARMS[2]);b,_=self.run_scene(ARMS[3]);np.testing.assert_array_equal(a.layers['reachable_blue'],b.layers['reachable_blue']);self.assertEqual(a.outline.get('tip_observation'),b.outline.get('tip_observation'))

if __name__=='__main__':unittest.main()
