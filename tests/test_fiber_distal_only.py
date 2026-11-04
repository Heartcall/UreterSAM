import copy,unittest
from unittest.mock import patch
import numpy as np
from tools.fiber_reference_workspace import ROOT,read
from tools.fiber_blue_extent import BlueExtent,GrowthPipeline as Previous
from tools.fiber_distal_only import DistalOnlyExtent,GrowthPipeline,direction
from tests.test_fiber_blue_extent import scene,configuration

class DistalOnlyTests(unittest.TestCase):
    def setUp(self):self.cfg=read(ROOT/'configs/fiber_distal_only_control_20261102.json');self.g=DistalOnlyExtent(self.cfg['extent'])
    def test_inherited_numeric_settings_exact(self):self.assertEqual(self.cfg['extent'],configuration()['extent'])
    def test_only_one_real_walk(self):
        im,raw,p,_=scene();c,_,_,m=self.g.grow(im,raw,p)
        self.assertGreater(c.sum(),0);self.assertEqual(m['direction_control']['walk_calls'],[1]);self.assertTrue(any(b['stop_reason']=='proximal_not_searched' for b in m['branches']))
    def test_negative_distal_also_only_one_walk(self):
        im,raw,p,_=scene();im=im[:,::-1].copy();raw=raw[:,::-1].copy();p=p[:,:,::-1].copy();_,_,_,m=self.g.grow(im,raw,p);self.assertEqual(m['direction_control']['walk_calls'],[-1])
    def test_no_direction_no_search_or_tip(self):
        im,raw,p,_=scene()
        with patch('tools.fiber_distal_only.direction',return_value=(0,'ambiguous')):
            # Frozen post-walk direction must match: remove its border preference too.
            cfg=copy.deepcopy(self.cfg['extent']);cfg['history']['orientation_border_distance_difference_diagonal_fraction']=1.
            g=DistalOnlyExtent(cfg);c,proposal,_,m=g.grow(im,raw,p)
        self.assertFalse(c.any());self.assertFalse(proposal.any());self.assertEqual(m['direction_control']['walk_calls'],[]);self.assertTrue(all(t['xy'] is None for t in m['tips']))
    def test_raster_proximal_failure_no_longer_deletes_other_end(self):
        im,raw,p,_=scene(False);before=BlueExtent(self.cfg['extent']).grow(im,raw,p);after=self.g.grow(im,raw,p)
        self.assertFalse(before[0].any());self.assertGreater(after[0].sum(),0);self.assertFalse(after[3]['abnormal'])
    def test_selected_end_abnormal_still_rejects(self):
        im,raw,p,_=scene();original=BlueExtent.walk
        def broken(obj,*args,**kwargs):
            branch=original(obj,*args,**kwargs);branch.update(abnormal='width_expansion_or_jump',stop_reason='width_expansion_or_jump');return branch
        with patch.object(BlueExtent,'walk',new=broken):c,_,_,m=self.g.grow(im,raw,p)
        self.assertFalse(c.any());self.assertTrue(m['abnormal'])
    def test_black_gap_stone_and_island_remain_protected(self):
        im,raw,p,t=scene();im[:,170:178]=0;im[30:50,300:330]=(210,100,15);raw[:,190:195]=1;c,_,_,_=self.g.grow(im,raw,p)
        self.assertFalse(c[:,178:].any());self.assertFalse(c[raw!=0].any());self.assertFalse(c[30:50,300:330].any())
    def test_baseline_pipeline_exact(self):
        im,raw,p,_=scene();t1=read(ROOT/self.cfg['t1']['config']);a=GrowthPipeline(self.cfg,'blue_extent',t1,False).run(im,raw,p);b=Previous(configuration(),'blue_extent',t1,False).run(im,raw,p)
        for i in range(3):np.testing.assert_array_equal(a[i],b[i])
        for key in ['reason','seeds','branches','tips']:self.assertEqual(a[3][key],b[3][key])
    def test_seed_failure_identical(self):
        im,raw,p,_=scene();raw[:]=0;a=BlueExtent(self.cfg['extent']).grow(im,raw,p);b=self.g.grow(im,raw,p);self.assertEqual(a[3]['seeds'],b[3]['seeds']);self.assertFalse(b[0].any())
    def test_threshold_mutation_is_rejected(self):
        cfg=copy.deepcopy(self.cfg);cfg['extent']['color']['angle_normal_degrees']+=1
        with self.assertRaises(ValueError):GrowthPipeline(cfg,'distal_only')

if __name__=='__main__':unittest.main()
