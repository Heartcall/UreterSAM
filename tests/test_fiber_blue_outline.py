import copy,json,unittest
import cv2,numpy as np
from tools.fiber_reference_workspace import ROOT,read
from tools.fiber_blue_outline import GrowthPipeline,Outline,natural_values,robust_model,root_instance,retained,LAYERS
from tools.fiber_connected_blue_union import GrowthPipeline as OldPipeline
from tools.fiber_connected_blue_growth import chroma

def scene(curved=False):
    h=w=400;yy,xx=np.indices((h,w));center=200+(8*np.sin(xx/85) if curved else 0);width=20-13*np.minimum(xx,260)/260;width=np.where(xx>=260,7*np.sqrt(np.clip(1-((xx-260)/20)**2,0,1)),width);body=((xx<=280)&(np.abs(yy-center)<=width)).astype(np.uint8);image=np.full((h,w,3),(45,50,80),np.uint8);image[body>0]=(210,100,15);raw=np.zeros_like(body);raw[(body>0)&(xx<135)]=2;p=np.zeros((3,h,w),np.float32);p[0]=.98;p[2]=.01;p[2][raw==2]=.99;p[0][raw==2]=.005;return image,raw,p,body

class BlueOutlineTests(unittest.TestCase):
    def setUp(self):self.cfg=read(ROOT/'configs/blue_outline_control_20261102.json')
    def pipe(self,arm,curved=False):
        im,raw,p,b=scene(curved);g=GrowthPipeline(self.cfg,arm,None,False);out=g.run(im,raw,p);return g,(im,raw,p,b),out
    def test_reference_precision_not_gate(self):self.assertNotIn('quality',self.cfg);self.assertIn('diagnostic only',self.cfg['geometry_quality']['reference_FP_Precision_Dice'])
    def test_union_exact_original(self):
        g,(im,raw,p,b),out=self.pipe('connected_blue_union');old=OldPipeline(read(ROOT/self.cfg['reference_union_config']),'connected_blue_union',None,False).run(im,raw,p)
        for a,z in zip(out[:3],old[:3]):np.testing.assert_array_equal(a,z)
        for k in ['reason','seeds','branches','tips']:self.assertEqual(out[3][k],old[3][k])
    def test_natural_spline_interpolates_nodes(self):
        x=np.arange(7.);y=np.sin(x);np.testing.assert_allclose(natural_values(x,y,x),y,atol=1e-10)
    def test_natural_line_derivative(self):self.assertTrue(np.allclose(natural_values([0,10],[2,7],np.arange(11),True),.5))
    def test_robust_line_ignores_sparse_burrs(self):
        t=np.arange(100.);y=.1*t;y[::13]+=8;m=robust_model(t,y,np.ones(100),self.cfg['outline'],1.);self.assertEqual(m['mode'],'line');self.assertAlmostEqual(float(natural_values(m['knots'],m['values'],80)),8,delta=.4)
    def test_true_bend_uses_spline(self):
        t=np.arange(120.);y=8*np.sin(t/35);m=robust_model(t,y,np.ones(120),self.cfg['outline'],1.);self.assertEqual(m['mode'],'spline');self.assertGreater(m['curve_deviation'],2)
    def test_sides_from_complete_RGB_body(self):
        g,(im,raw,p,b),out=self.pipe('contour_sides');self.assertTrue(g.outline['eligible']);self.assertTrue(all(m['valid'] for m in g.outline['side_models']));self.assertEqual(len(g.outline['side_models']),2)
        if not g.outline['sides_applied']:self.assertEqual(g.outline['reason'],'side_visibility_retention_fallback');np.testing.assert_array_equal(g.layers['processed_body'],g.layers['blue_body'])
    def test_curved_scene_not_forced_straight(self):
        g,_,_=self.pipe('contour_sides',True);self.assertTrue(any(m.get('mode')=='spline' for m in g.outline['side_models']))
    def test_blue_body_identity_replacement(self):
        g,(im,raw,p,b),out=self.pipe('blue_body_only');root=g.layers['root_instance']>0;np.testing.assert_array_equal(g.final_mask[root],np.where(g.layers['blue_body'][root]>0,2,0))
    def test_all_actual_inputs_immutable(self):
        im,raw,p,b=scene();old=[x.copy() for x in [im,raw,p]];GrowthPipeline(self.cfg,'contour_sides_cap',None,False).run(im,raw,p)
        for a,z in zip(old,[im,raw,p]):np.testing.assert_array_equal(a,z)
    def test_original_stone_preserved(self):
        im,raw,p,b=scene();raw[:,220:225]=1;g=GrowthPipeline(self.cfg,'contour_sides_cap',None,False);g.run(im,raw,p);self.assertTrue(np.all(g.final_mask[raw==1]==1));self.assertFalse(g.layers['processed_body'][:,225:].any())
    def test_black_gap_not_filled(self):
        im,raw,p,b=scene();im[:,190:196]=0;g=GrowthPipeline(self.cfg,'contour_sides_cap',None,False);g.run(im,raw,p);self.assertFalse(g.layers['processed_body'][:,196:].any())
    def test_no_seed_no_geometry(self):
        im,raw,p,b=scene();raw[:]=0;g=GrowthPipeline(self.cfg,'contour_sides_cap',None,False);g.run(im,raw,p);self.assertFalse(g.outline['eligible']);self.assertFalse(g.final_mask.any())
    def test_other_neural_visible_component_retained(self):
        im,raw,p,b=scene();raw[350:352,350:352]=2;p[2,350:352,350:352]=.1;g=GrowthPipeline(self.cfg,'contour_sides',None,False);g.run(im,raw,p);self.assertTrue(np.all(g.final_mask[350:352,350:352]==2))
    def test_raster_layers_and_original_class_delta(self):
        g,(im,raw,p,b),_=self.pipe('contour_sides_cap');self.assertEqual(set(g.layers),set(LAYERS));np.testing.assert_array_equal(g.layers['edit_remove']>0,(raw==2)&(g.final_mask!=2));self.assertFalse(g.layers['processed_body'][g.layers['rgb_evidence']==0].any())
    def test_retention_rejects_but_keeps_previous_reference(self):
        _,raw,p,b=scene();before=b.copy();core=raw==2;empty=np.zeros_like(b);r,f,l,ok=retained(empty,b,core,np.array([1.,0.]),self.cfg['outline']['retention'],1.);self.assertFalse(ok);self.assertTrue(np.isfinite(l));np.testing.assert_array_equal(b,before)
    def test_observed_end_not_extrapolated(self):
        g,_,_=self.pipe('contour_sides_cap');c=g.outline['cap']
        if c.get('curve'):self.assertLessEqual(c['tip_t'],g.outline['context']['max_t']+1e-9)
    def test_explicit_ellipse_not_gaussian_cap(self):
        g,_,_=self.pipe('contour_sides_cap');c=g.outline['cap'];self.assertTrue(c['attempted'],c);self.assertGreater(c['candidate_count'],1);self.assertEqual(len(c['curve']),65);self.assertIn('a',c);self.assertIn('b',c)
    def test_supported_round_tip_can_be_accepted(self):
        g,_,_=self.pipe('contour_sides_cap');self.assertTrue(g.outline['cap']['accepted'],g.outline['cap'])
    def test_cap_failure_retains_sides(self):
        im,raw,p,b=scene();im[:,281:]=0;a=GrowthPipeline(self.cfg,'contour_sides',None,False);z=GrowthPipeline(self.cfg,'contour_sides_cap',None,False);a.run(im,raw,p);z.run(im,raw,p);self.assertFalse(z.outline['cap']['accepted']);np.testing.assert_array_equal(a.final_mask,z.final_mask)
    def test_network_width_not_used_for_side_fit(self):
        g,(im,raw,p,b),out=self.pipe('contour_sides');core,_=root_instance(raw,p,out[3]);m=copy.deepcopy(out[3]);m['seed_width']*=10;a=Outline(self.cfg).run(im,raw,p,g.layers['blue_body'],core,out[3],'contour_sides');z=Outline(self.cfg).run(im,raw,p,g.layers['blue_body'],core,m,'contour_sides');np.testing.assert_array_equal(a[0],z[0]);self.assertEqual(a[4]['side_models'],z[4]['side_models'])
    def test_geometry_is_finite_json(self):
        g,_,_=self.pipe('contour_sides_cap');json.dumps(g.outline,allow_nan=False)
    def test_static_has_no_state(self):self.assertIsNone(GrowthPipeline(self.cfg,'contour_sides_cap',None,False).tracker)
    def test_source_state_reset(self):
        t=read(ROOT/self.cfg['t1']['config']);g=GrowthPipeline(self.cfg,'contour_sides_cap',t,True);im,raw,p,b=scene();g.run(im,raw,p,0.,'a');_,_,_,m=g.run(im,raw,p,0.,'b');self.assertEqual(m['t1_advice']['context_reset_reason'],'source_changed');g.reset();self.assertIsNone(g.tracker.source)
    def test_root_control_not_initial_boundary_lock(self):
        im,raw,p,b=scene();raw[182,60]=2;p[2,182,60]=.99;a=GrowthPipeline(self.cfg,'connected_blue_union',None,False);z=GrowthPipeline(self.cfg,'blue_body_only',None,False);a.run(im,raw,p);z.run(im,raw,p);self.assertEqual(a.final_mask[182,60],2);self.assertEqual(z.final_mask[182,60],0)

if __name__=='__main__':unittest.main()
