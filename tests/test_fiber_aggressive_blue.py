import copy,unittest
import cv2,numpy as np
from tools.fiber_reference_workspace import ROOT,read
from tools.fiber_aggressive_blue import AggressiveVariant,GrowthPipeline,ARMS,fixed_axis,tip_visibility
from tools.fiber_distal_only import GrowthPipeline as OldPipeline
from tools.fiber_blue_extent import BlueExtent
from tests.test_fiber_blue_extent import scene

def config():return read(ROOT/'configs/fiber_aggressive_blue_control_20261102.json')
def seed():
    mu=np.array([15,100,210])/325
    return dict(mu=mu,v=mu-1/3,bg=np.array([.5,.3,.2]),gap=.5)

class AggressiveBlueTests(unittest.TestCase):
    def setUp(self):self.cfg=config()
    def test_frozen_base_identical(self):self.assertEqual(self.cfg['extent'],read(ROOT/self.cfg['reference_distal_config'])['extent'])
    def test_stages_only_registered_numeric_families(self):
        variants={a:AggressiveVariant(self.cfg,a) for a in ARMS[2:]}
        self.assertEqual(variants['visible_bg'].cfg,variants['fixed_axis'].cfg)
        edge=copy.deepcopy(variants['fixed_axis'].cfg);edge['section']['strong_RGB_contrast']=0.;edge['seed']['normal_min_sections']=2;self.assertEqual(edge,variants['color_edges'].cfg)
        broad=copy.deepcopy(edge);broad['color'].update(self.cfg['aggressive_parameters']['broad_color']);self.assertEqual(broad,variants['broad_blue'].cfg)
        broad['geometry'].update(self.cfg['aggressive_parameters']['geometry']);self.assertEqual(broad,variants['aggressive'].cfg)
    def test_fixed_anchor_points_into_center(self):
        pts=np.array([[0,330],[1,335],[2,340],[30,320],[60,290],[90,270]],float);anchor,u=fixed_axis(pts,(400,400),self.cfg['aggressive_parameters']['fixed_direction'])
        self.assertEqual(min(anchor[0],anchor[1],399-anchor[0],399-anchor[1]),0);self.assertGreater(np.dot(u,np.array([199.5,199.5])-anchor),0)
    def test_current_blue_brightness_family(self):
        g=AggressiveVariant(self.cfg,'broad_blue');colors=np.array([[210,100,15],[105,50,8],[42,20,3],[9,4,1]],np.uint8);self.assertTrue(g.color(colors,seed()).all())
    def test_dark_noise_relaxation_is_separate_factor(self):
        colors=np.array([[9,4,1]],np.uint8);self.assertFalse(AggressiveVariant(self.cfg,'color_edges').color(colors,seed())[0]);self.assertTrue(AggressiveVariant(self.cfg,'broad_blue').color(colors,seed())[0])
    def test_true_black_and_neutral_not_blue(self):
        g=AggressiveVariant(self.cfg,'aggressive');self.assertFalse(g.color(np.array([[0,0,0],[20,20,20]],np.uint8),seed()).any())
    def test_relaxed_edges_use_visible_color_not_RGB_strong(self):
        im,raw,p,t=scene();base=BlueExtent(self.cfg['extent']);s=base.seeds(im,raw,p,None,99)[0][0];im[:,150:]=(10,8,6);im[:,150:][t[:,150:]>0]=(18,8,2)
        q,why=base.section(im,np.array([210.,200.]),np.array([1.,0.]),s['width'],s);self.assertIsNone(q)
        q,why=AggressiveVariant(self.cfg,'color_edges').section(im,np.array([210.,200.]),np.array([1.,0.]),s['width'],s);self.assertIsNotNone(q,why);self.assertEqual(q['states'],['color_supported','color_supported']);self.assertFalse(q['weak'])
    def test_visible_bg_records_22_sites_without_lowering_six(self):
        im,raw,p,_=scene();g=AggressiveVariant(self.cfg,'visible_bg');_,records=g.seeds(im,raw,p,None,99);self.assertGreaterEqual(records[0]['background_valid_points'],6);self.assertEqual(sum(records[0]['background_side_counts']),22);self.assertGreaterEqual(records[0]['background_axial_coverage'],.4)
    def test_visible_bg_uses_only_inbounds_samples(self):
        im,raw,p,_=scene();im=np.roll(im,185,axis=0);raw=np.roll(raw,185,axis=0);p=np.roll(p,185,axis=1);_,records=AggressiveVariant(self.cfg,'visible_bg').seeds(im,raw,p,None,99)
        self.assertTrue(all(r.get('background_valid_points',0)<=22 for r in records))
    def test_low_network_probability_can_be_candidate(self):
        im,raw,p,_=scene();c,_,_,m=AggressiveVariant(self.cfg,'aggressive').grow(im,raw,p);self.assertGreater(c.sum(),0,m);self.assertTrue((p[2][c>0]<.1).all())
    def test_all_variants_stop_at_black_gap(self):
        for arm in ARMS[2:]:
            im,raw,p,_=scene();im[:,170:178]=0;c,_,_,_=AggressiveVariant(self.cfg,arm).grow(im,raw,p);self.assertFalse(c[:,178:].any(),arm)
    def test_disconnected_blue_not_absorbed(self):
        im,raw,p,_=scene();im[30:55,300:330]=(200,100,15);c,_,_,_=AggressiveVariant(self.cfg,'aggressive').grow(im,raw,p);self.assertFalse(c[30:55,300:330].any())
    def test_stone_barrier_before_connectivity(self):
        im,raw,p,_=scene();raw[:,175:182]=1;c,_,conf,m=AggressiveVariant(self.cfg,'aggressive').grow(im,raw,p);self.assertFalse(c[raw==1].any());self.assertFalse(c[:,182:].any());self.assertGreater(conf.sum(),0,m)
    def test_no_seed_no_pixels_even_with_history(self):
        im,raw,p,_=scene();raw[:]=0;c,_,_,_=AggressiveVariant(self.cfg,'aggressive').grow(im,raw,p,[0,200,0,40],0);self.assertFalse(c.any())
    def test_fixed_variants_search_only_positive(self):
        im,raw,p,_=scene()
        for arm in ARMS[3:]:
            _,_,_,m=AggressiveVariant(self.cfg,arm).grow(im,raw,p);self.assertEqual(m['direction_control']['walk_calls'],[1],arm)
    def test_inputs_not_changed(self):
        im,raw,p,_=scene();copies=[v.copy() for v in [im,raw,p]];AggressiveVariant(self.cfg,'aggressive').grow(im,raw,p)
        for a,b in zip([im,raw,p],copies):np.testing.assert_array_equal(a,b)
    def test_legacy_baseline_pixels_and_tips_unchanged(self):
        im,raw,p,_=scene();t1=read(ROOT/self.cfg['t1']['config']);new=GrowthPipeline(self.cfg,'distal_only',t1,False).run(im,raw,p);old=OldPipeline(read(ROOT/self.cfg['reference_distal_config']),'distal_only',t1,False).run(im,raw,p)
        for i in range(3):np.testing.assert_array_equal(new[i],old[i])
        self.assertEqual(new[3]['tips'],old[3]['tips'])
    def test_tip_without_root_visible_support_rejected(self):
        im,raw,p,_=scene();meta=dict(seeds=[dict(reason='qualified',label=1)],tips=[dict(state='visible_end_candidate',xy=[350,350])]);a=tip_visibility(raw,p,np.zeros_like(raw),meta,2.)[0];self.assertFalse(a['valid_current_mask_support']);self.assertIsNone(a['xy'])
    def test_tip_on_root_visible_body_only_audit_not_GT(self):
        _,raw,p,_=scene();meta=dict(seeds=[dict(reason='qualified',label=1)],tips=[dict(state='visible_end_candidate',xy=[100,200])]);a=tip_visibility(raw,p,np.zeros_like(raw),meta,2.)[0];self.assertTrue(a['valid_current_mask_support'])
    def test_static_has_no_video_state(self):
        self.assertIsNone(GrowthPipeline(self.cfg,'aggressive',read(ROOT/self.cfg['t1']['config']),False).tracker)
    def test_video_state_resets(self):
        pipe=GrowthPipeline(self.cfg,'aggressive',read(ROOT/self.cfg['t1']['config']),True);im,raw,p,_=scene();pipe.run(im,raw,p,0.,'one');_,_,_,m=pipe.run(im,raw,p,0.,'two');self.assertEqual(m['t1_advice']['context_reset_reason'],'source_changed');pipe.reset();self.assertIsNone(pipe.tracker.source)

if __name__=='__main__':unittest.main()
