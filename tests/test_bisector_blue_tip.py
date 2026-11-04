import copy,json,unittest
import cv2,numpy as np
from tools.fiber_reference_workspace import ROOT,read
from tools.fiber_bisector_blue_tip import GrowthPipeline,direction,distal_cluster,visibility,ARMS,LAYERS
from tools.fiber_blue_outline_repaired import GrowthPipeline as Old
from tests.test_fiber_blue_outline import scene

class BisectorTests(unittest.TestCase):
    def setUp(self):self.cfg=read(ROOT/'configs/bisector_blue_tip_control_20261103.json')
    def pipe(self,arm=ARMS[-1],curved=False):
        x=scene(curved);g=GrowthPipeline(self.cfg,arm,None,False);result=g.run(*x[:3]);return g,x,result
    def test_union_exact(self):
        g,x,r=self.pipe(ARMS[0]);o=Old(read(ROOT/'configs/blue_outline_control_20261102_v2.json'),ARMS[0],None,False);rr=o.run(*x[:3]);np.testing.assert_array_equal(g.final_mask,o.final_mask)
        for a,b in zip(r[:3],rr[:3]):np.testing.assert_array_equal(a,b)
    def test_previous_cap_exact(self):
        g,x,_=self.pipe(ARMS[1]);o=Old(read(ROOT/'configs/blue_outline_control_20261102_v2.json'),ARMS[1],None,False);o.run(*x[:3]);np.testing.assert_array_equal(g.final_mask,o.final_mask)
    def test_union_all_pixels_preserved(self):
        g,x,_=self.pipe();o=Old(read(ROOT/'configs/blue_outline_control_20261102_v2.json'),ARMS[0],None,False);o.run(*x[:3]);self.assertTrue(np.all(g.final_mask[o.final_mask==2]==2));self.assertFalse(g.layers['edit_remove'].any())
    def test_blue_preserved(self):
        g,_,_=self.pipe();self.assertTrue(np.all(g.final_mask[g.layers['blue_body']>0]==2))
    def test_layer_protocol(self):g,_,_=self.pipe();self.assertEqual(set(g.layers),set(LAYERS))
    def test_currentRGB_candidate(self):
        g,_,_=self.pipe();self.assertEqual(g.outline['distal']['reason'],'supported_farthest_blue_cluster');self.assertEqual(g.outline['tip_observation']['state'],'current_RGB_confirmed');self.assertIsNotNone(g.outline['final_tip_xy'])
    def test_cluster_rejects_single_spike(self):
        b=np.zeros((50,100),np.uint8);b[20:31,:60]=1;b[25,60:90]=1;c,m=distal_cluster(b,np.array([0.,0.]),np.array([1.,0.]),self.cfg['bisector_tip']['cluster'],1.);self.assertIsNone(m['candidate_xy'])
    def test_cluster_not_single_max(self):
        b=np.zeros((50,100),np.uint8);b[20:31,:70]=1;c,m=distal_cluster(b,np.array([0.,0.]),np.array([1.,0.]),self.cfg['bisector_tip']['cluster'],1.);self.assertGreater(m['area'],12);self.assertEqual(m['frontier_points'],22)
    def test_no_seed_no_tip(self):
        im,raw,p,b=scene();raw[:]=0;g=GrowthPipeline(self.cfg,ARMS[-1]);g.run(im,raw,p);self.assertFalse(g.outline['eligible']);self.assertFalse(g.final_mask.any())
    def test_occlusion_no_fill(self):
        im,raw,p,b=scene();raw[:,282:289]=1;g=GrowthPipeline(self.cfg,ARMS[-1]);g.run(im,raw,p);self.assertIsNone(g.outline.get('final_tip_xy'));self.assertTrue(np.all(g.final_mask[raw==1]==1))
    def test_black_gap(self):
        im,raw,p,b=scene();im[:,190:196]=0;g=GrowthPipeline(self.cfg,ARMS[-1]);g.run(im,raw,p);self.assertFalse(g.layers['blue_body'][:,196:].any());self.assertFalse(g.layers['cap_fill'][:,196:].any())
    def test_unknown_exact_union(self):
        im,raw,p,b=scene();im[:,281:]=0;a=GrowthPipeline(self.cfg,ARMS[0]);g=GrowthPipeline(self.cfg,ARMS[-1]);a.run(im,raw,p);g.run(im,raw,p);np.testing.assert_array_equal(g.final_mask,a.final_mask)
    def test_curved_local_centerline(self):
        g,_,_=self.pipe(curved=True);self.assertEqual(g.outline['bisector']['mode'],'local_centerline_curved_or_parallel')
    def test_tip_not_line_intersection(self):
        g,_,_=self.pipe();self.assertIn('distal',g.outline);xy=g.outline['candidate_tip_xy'];x,y=np.rint(xy).astype(int);self.assertEqual(g.layers['blue_body'][y,x],1)
    def test_explicit_cap_anchored(self):
        g,_,_=self.pipe();c=g.outline['cap'];self.assertGreater(c.get('candidate_count',0),0);self.assertEqual(len(c['curve']),65);self.assertEqual(c['tip_t'],g.outline['distal']['blue_front_t'])
    def test_synthetic_visible_round_cap_accepted(self):
        g,_,_=self.pipe();self.assertTrue(g.outline['cap']['accepted']);self.assertGreater(g.outline['cap']['feasible_count'],0)
    def test_six_grid_original_panel_unmarked(self):
        from tools.bisector_blue_tip_media import grid
        g,x,_=self.pipe();im,raw,_,_=x;rows={a:dict(outline=g.outline) for a in ARMS};masks={a:g.final_mask for a in ARMS};v=grid(im,raw,masks,rows,g.layers['distal_cluster'],'v',1208,40.2);self.assertEqual(v.shape,(960,1200,3));np.testing.assert_array_equal(v[48:448,:400],im)
    def test_media_rgb_tip_unknown_safe(self):
        from tools.bisector_blue_tip_media import analysis
        im,raw,p,b=scene();row=dict(outline=dict(cap=dict(accepted=False,curve=[])));np.testing.assert_array_equal(analysis(im,row,np.zeros_like(raw)),im)
    def test_no_extra_pixel_without_rgb_evidence(self):
        g,_,_=self.pipe();new=(g.layers['side_fill']>0)|(g.layers['cap_fill']>0);self.assertFalse(new[g.layers['rgb_evidence']==0].any())
    def test_no_cap_extrapolation(self):
        g,_,_=self.pipe();q=g.outline['cap'];c=g.outline['cap_coordinate_context'];t=(np.asarray(q['curve'])-c['origin'])@c['axis'];self.assertLessEqual(t.max(),g.outline['distal']['blue_front_t']+1e-8)
    def test_inputs_immutable(self):
        im,raw,p,b=scene();before=[z.copy() for z in [im,raw,p]];GrowthPipeline(self.cfg,ARMS[-1]).run(im,raw,p)
        for a,b in zip(before,[im,raw,p]):np.testing.assert_array_equal(a,b)
    def test_JSON_finite(self):g,_,_=self.pipe();json.dumps(g.outline,allow_nan=False)
    def test_static_no_history(self):self.assertIsNone(GrowthPipeline(self.cfg,ARMS[-1]).tracker)
    def test_state_reset(self):
        t=read(ROOT/self.cfg['t1']['config']);g=GrowthPipeline(self.cfg,ARMS[-1],t,True);im,raw,p,b=scene();g.run(im,raw,p,0.,'a');g.reset();self.assertIsNone(g.tracker.source)
    def test_criteria_diagnostic(self):self.assertFalse(self.cfg['geometry_quality']['reference_metrics_are_gates'])
if __name__=='__main__':unittest.main()
