import unittest,copy
import cv2,numpy as np
from tools.fiber_reference_workspace import ROOT,read
from tools.fiber_connected_blue_union import ConnectedBlue,GrowthPipeline,rooted,trim,LAYERS
from tools.fiber_aggressive_blue import GrowthPipeline as OldPipeline
from tests.test_fiber_blue_extent import scene

class ConnectedUnionTests(unittest.TestCase):
    def setUp(self):self.cfg=read(ROOT/'configs/connected_blue_union_control_20261102.json')
    def runscene(self,regularized=False,values=None):
        im,raw,p,t=values or scene();g=ConnectedBlue(self.cfg,regularized);return g,g.grow(im,raw,p)
    def test_base_parameters_frozen(self):self.assertEqual(self.cfg['extent'],read(ROOT/self.cfg['reference_aggressive_config'])['extent'])
    def test_aggressive_exact_baseline(self):
        im,raw,p,_=scene();old=OldPipeline(read(ROOT/self.cfg['reference_aggressive_config']),'aggressive',None,False).run(im,raw,p);new=GrowthPipeline(self.cfg,'aggressive',None,False).run(im,raw,p)
        for a,b in zip(old[:3],new[:3]):np.testing.assert_array_equal(a,b)
        for k in ['reason','seeds','branches','tips','tip_visibility_audit']:self.assertEqual(old[3][k],new[3][k])
    def test_lowP_dark_reaches_full_visible_extent(self):
        im,raw,p,t=scene();g,(c,_,_,m)=self.runscene(values=(im,raw,p,t));self.assertTrue(c[:,270:].any(),m);self.assertTrue((p[2][c>0]<.1).all());self.assertFalse(c[t==0].any())
    def test_taper_not_cleared_by_local_curvature(self):
        g,(c,_,_,m)=self.runscene(values=scene(False));self.assertTrue(c[:,250:].any(),m);self.assertFalse(m['abnormal'])
    def test_black_gap_no_bridge(self):
        im,raw,p,t=scene();im[:,170:178]=0;g,(c,_,_,m)=self.runscene(values=(im,raw,p,t));self.assertFalse(c[:,178:].any(),m)
    def test_disconnected_blue_excluded(self):
        im,raw,p,t=scene();im[100:120,260:280]=(210,100,15);g,(c,_,_,_)=self.runscene(values=(im,raw,p,t));self.assertFalse(c[100:120,260:280].any())
    def test_stone_barrier_before_connectivity(self):
        im,raw,p,t=scene();raw[:,175:182]=1;g,(c,_,conf,m)=self.runscene(values=(im,raw,p,t));self.assertFalse(c[raw==1].any());self.assertFalse(c[:,182:].any());self.assertGreater(conf.sum(),0,m)
    def test_no_seed_history_cannot_fill(self):
        im,raw,p,_=scene();raw[:]=0;c,_,_,_=ConnectedBlue(self.cfg,False).grow(im,raw,p,[0,200,0,40],0);self.assertFalse(c.any())
    def test_inputs_immutable(self):
        im,raw,p,_=scene();old=[x.copy() for x in [im,raw,p]];ConnectedBlue(self.cfg,True).grow(im,raw,p)
        for a,b in zip(old,[im,raw,p]):np.testing.assert_array_equal(a,b)
    def test_stage_subset_and_original_protection(self):
        im,raw,p,t=scene(False);a=ConnectedBlue(self.cfg,False);b=ConnectedBlue(self.cfg,True);u=a.grow(im,raw,p);r=b.grow(im,raw,p);self.assertFalse(np.any((r[0]>0)&(u[0]==0)));np.testing.assert_array_equal(a.layers['blue_body'],b.layers['blue_body']);self.assertFalse(r[0][raw!=0].any());self.assertEqual(set(b.layers),set(LAYERS))
    def test_smoothing_subset_and_protected_core(self):
        b=np.zeros((30,30),np.uint8);b[10:20,3:25]=1;b[8:10,15]=1;core=np.zeros_like(b,bool);core[12:16,5:8]=True;r=trim(b,core,self.cfg['connected_blue']['regularization'],1.);self.assertFalse(r[b==0].any());self.assertTrue(r[core].all());self.assertLess(r.sum(),b.sum())
    def test_hole_not_filled(self):
        b=np.ones((20,20),np.uint8);b[10,10]=0;core=np.zeros_like(b,bool);core[3:6,3:6]=True;r=trim(b,core,self.cfg['connected_blue']['regularization'],1.);self.assertEqual(r[10,10],0)
    def test_multiple_color_roots_rejected(self):
        b=np.zeros((20,20),np.uint8);b[1:4,1:4]=1;b[10:14,10:14]=1;r,n=rooted(b,b>0,3);self.assertEqual(n,2);self.assertFalse(r.any())
    def test_proximal_new_pixels_excluded(self):
        im,raw,p,t=scene();g,(c,_,_,_)=self.runscene(values=(im,raw,p,t));self.assertFalse(c[:,:1].any())
    def test_static_no_video_state(self):self.assertIsNone(GrowthPipeline(self.cfg,'connected_blue_union',None,False).tracker)
    def test_video_source_reset(self):
        pipe=GrowthPipeline(self.cfg,'connected_blue_union',read(ROOT/self.cfg['t1']['config']),True);im,raw,p,_=scene();pipe.run(im,raw,p,0.,'one');_,_,_,m=pipe.run(im,raw,p,0.,'two');self.assertEqual(m['t1_advice']['context_reset_reason'],'source_changed');pipe.reset();self.assertIsNone(pipe.tracker.source)
    def tipcase(self,mode):
        im,raw,p,t=scene(False);g=ConnectedBlue(self.cfg,False);seed=g.observer.seeds(im,raw,p,None,99)[0][0];body=t.copy();domain=np.ones_like(t,bool)
        if mode=='continued':body[:,250:]=0
        if mode=='stone':raw[190:210,301:310]=1
        if mode=='black':im[:,301:]=0
        return g.end(im,raw,body,seed,domain,None)
    def test_tip_continued_blue_is_internal(self):
        t=self.tipcase('continued');self.assertEqual(t['state'],'internal_transition');self.assertIsNone(t['xy'])
    def test_tip_stone_contact_is_occlusion(self):
        t=self.tipcase('stone');self.assertEqual(t['state'],'occlusion_suspected');self.assertIsNone(t['xy'])
    def test_unknown_terminal_background_no_tip(self):
        t=self.tipcase('black');self.assertEqual(t['state'],'unknown');self.assertIsNone(t['xy'])
    def test_absolute_expansion_safety_keeps_proposal(self):
        im,raw,p,t=scene();g=ConnectedBlue(self.cfg,False);seeds,records=g.observer.seeds(im,raw,p,None,99);im[80:320,135:390]=(210,100,15);g.observer.seeds=lambda *a:(seeds,records);c,proposal,_,m=g.grow(im,raw,p);self.assertTrue(m['abnormal']);self.assertFalse(c.any());self.assertTrue(proposal.any());self.assertTrue(g.layers['blue_body'].any())
    def test_supported_tip_candidate_is_on_current_cap_only(self):
        t=self.tipcase('normal');self.assertEqual(t['state'],'visible_end_candidate',t);self.assertLessEqual(t['xy'][0],300.);self.assertGreaterEqual(t['xy'][0],299.)

if __name__=='__main__':unittest.main()
