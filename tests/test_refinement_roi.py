import unittest
import json
import numpy as np
from tools.refinement_roi import propose_rois, roi_union, coverage_counts


SETTINGS = {"native_crop_side":16,"margin_native_pixels":2,"max_tile_stride":12,
            "minimum_component_pixels":4,"soft_probability_threshold":0.2}


class RefinementRoiTests(unittest.TestCase):
    def test_empty_does_not_oracle_rescue_missed_target(self):
        p=np.zeros((3,32,32)); p[0]=1
        target=np.zeros((32,32),np.uint8);target[20:24,20:24]=1
        boxes=propose_rois(p,SETTINGS)
        self.assertEqual(boxes,[])
        counts=coverage_counts(p.argmax(0),target,boxes)
        self.assertEqual(counts['stone']['covered_pixels'],0)
        self.assertEqual(counts['stone']['coarse_fn'],16)

    def test_border_and_large_extent_covered_without_out_of_image_boxes(self):
        p=np.zeros((3,35,43));p[1]=1
        boxes=propose_rois(p,SETTINGS)
        self.assertTrue(roi_union(boxes,(35,43)).all())
        self.assertEqual(boxes,propose_rois(p,SETTINGS))
        self.assertEqual(len(boxes),len(set(boxes)))
        self.assertEqual(json.loads(json.dumps(boxes)),[list(box) for box in boxes])

    def test_cap_and_class_balancing_cover_both_separated_classes(self):
        p=np.zeros((3,64,64));p[0]=1
        p[0,0:20,0:20]=0;p[1,0:20,0:20]=1
        p[0,56:60,56:60]=0;p[2,56:60,56:60]=1
        boxes=propose_rois(p,SETTINGS,max_rois=2)
        self.assertEqual(len(boxes),2)
        union=roi_union(boxes,(64,64))
        self.assertTrue(union[56:60,56:60].all())
        self.assertTrue(union[:20,:20].any())

    def test_soft_seed_can_propose_region_without_argmax_class(self):
        p=np.zeros((3,40,40));p[0]=1
        p[0,25:29,25:29]=0.7;p[1,25:29,25:29]=0.3
        self.assertEqual(propose_rois(p,SETTINGS,seed='argmax'),[])
        boxes=propose_rois(p,SETTINGS,seed='probability')
        self.assertTrue(roi_union(boxes,(40,40))[25:29,25:29].all())

    def test_unknown_pixels_ignored_and_overlap_not_double_counted(self):
        target=np.array([[1,1,255],[0,2,255],[0,0,0]],dtype=np.uint8)
        pred=np.ones_like(target)
        c=coverage_counts(pred,target,[(0,0,2,2),(1,0,3,2)])
        self.assertEqual(c['roi_area_sum'],8)
        self.assertEqual(c['roi_area_union'],6)
        self.assertEqual(c['valid_pixels_in_roi'],4)
        self.assertEqual(c['stone']['coarse_fp'],5)
        self.assertEqual(c['stone']['coarse_fp_covered'],2)
        self.assertEqual(c['fiber']['coarse_fn_covered'],1)

    def test_small_image_and_rejected_geometry(self):
        p=np.zeros((3,8,10));p[2]=1
        self.assertEqual(propose_rois(p,SETTINGS),[(0,0,10,8)])
        with self.assertRaises(ValueError):roi_union([(0,0,11,8)],(8,10))
        with self.assertRaises(ValueError):propose_rois(p,{**SETTINGS,'max_tile_stride':17})


if __name__ == '__main__':unittest.main()
