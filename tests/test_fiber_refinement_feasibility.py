import unittest
import numpy as np
from tools.fiber_refinement_feasibility import fiber_rois, stone_protected_upper_bound,region_counts

SETTINGS={'native_crop_side':12,'margin_native_pixels':2,'max_tile_stride':10,'minimum_component_pixels':4,'soft_probability_threshold':.2}
class FiberRefinementFeasibilityTests(unittest.TestCase):
    def test_no_gt_no_seed_no_roi_and_stone_does_not_seed(self):
        a=np.ones((30,40),np.uint8);p=np.zeros(a.shape,np.float32)
        self.assertEqual(fiber_rois(a,p,SETTINGS,'hard2'),[])
        a[:]=0;t=np.full(a.shape,2,np.uint8)
        self.assertEqual(region_counts(t,a,a,[])['regions']['fn'],{'total':1200,'inside':0,'outside':1200})
    def test_disconnected_components_cap_deterministic_bounds_and_soft_seed(self):
        a=np.zeros((30,40),np.uint8);a[1:5,1:5]=2;a[25:29,35:39]=2
        original=a.copy();p=(a==2).astype(np.float32)*.21
        hard=fiber_rois(a,p,SETTINGS,'hard2');self.assertEqual(len(hard),2)
        self.assertEqual(hard,fiber_rois(a,p,SETTINGS,'soft02_2'))
        self.assertEqual(hard,fiber_rois(a,p,SETTINGS,'hard2'))
        self.assertTrue(np.array_equal(a,original))
        self.assertEqual(len(fiber_rois(a,p,SETTINGS,'hard2',cap=1)),1)
        self.assertTrue(all(0<=x0<x1<=40 and 0<=y0<y1<=30 for x0,y0,x1,y1 in hard))
        a[:]=0;self.assertEqual(fiber_rois(a,p,SETTINGS,'hard2'),[]);self.assertTrue(fiber_rois(a,p,SETTINGS,'soft02_2'))
    def test_oracle_protects_stone_unknown_and_outside_counts(self):
        a=np.array([[1,2,0,2],[0,2,0,0]],np.uint8);t=np.array([[2,1,2,255],[2,0,2,0]],np.uint8)
        result=stone_protected_upper_bound(a,t,[(0,0,3,1)])
        np.testing.assert_array_equal(result,[[1,0,2,2],[0,2,0,0]])
        counts=region_counts(t,a,a,[(0,0,3,1)])['regions']
        self.assertEqual(counts['gt_fiber_blocked_by_stone']['inside'],1)
        self.assertEqual(counts['fp']['total'],2)
    def test_bad_geometry_and_probabilities_rejected(self):
        a=np.zeros((30,40),np.uint8)
        for p in [np.ones((2,2)),np.full(a.shape,np.nan),np.full(a.shape,1.1)]:
            with self.assertRaises(ValueError):fiber_rois(a,p,SETTINGS,'soft02_2')
        with self.assertRaises(ValueError):fiber_rois(a,None,SETTINGS,'hard2',cap=-1)

if __name__=='__main__':unittest.main()
