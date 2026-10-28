import unittest
import numpy as np
from PIL import Image
from tools.diagnose_joint_failures import class_error_counts,paired_changes,frame_descriptors


class JointFailureDiagnosisTests(unittest.TestCase):
    def test_unknowns_and_far_background_errors_not_counted_as_boundary(self):
        target=np.zeros((12,12),dtype=np.uint8);target[1:3,1:3]=2;target[10,10]=255
        pred=np.zeros_like(target);pred[1:3,1:3]=2;pred[9,9]=2;pred[10,10]=2
        r=class_error_counts(target,pred,2,3)
        self.assertEqual(r['tp'],4);self.assertEqual(r['fp'],1);self.assertEqual(r['fp_far'],1)
        self.assertEqual(r['fp_over_background'],1);self.assertEqual(r['fp_near'],0)
        empty=class_error_counts(np.zeros((3,3),np.uint8),np.full((3,3),2,np.uint8),2,3)
        self.assertEqual(empty['fp_far'],9)

    def test_paired_changes_decompose_net_fn_and_fp_even_with_unknown(self):
        target=np.array([[1,1,0],[0,2,255]],dtype=np.uint8)
        old=np.array([[1,0,1],[0,2,1]],dtype=np.uint8)
        new=np.array([[0,1,0],[1,2,0]],dtype=np.uint8)
        r=paired_changes(target,old,new,1)
        self.assertEqual(r,{'lost_tp':1,'gained_tp':1,'added_fp':1,'removed_fp':1})
        a=class_error_counts(target,old,1);b=class_error_counts(target,new,1)
        self.assertEqual(b['fn']-a['fn'],r['lost_tp']-r['gained_tp'])
        self.assertEqual(b['fp']-a['fp'],r['added_fp']-r['removed_fp'])

    def test_common_descriptor_grid_keeps_full_class_and_ignore_semantics(self):
        image=Image.fromarray(np.full((64,64,3),128,np.uint8))
        mask=np.ones((64,64),dtype=np.uint8);mask[:32]=255
        d=frame_descriptors(image,mask)
        self.assertEqual(d['annotated_stone_fraction'],1)
        self.assertEqual(d['annotated_fiber_fraction'],0)
        self.assertEqual(d['laplacian_variance'],0)
        self.assertFalse(d['has_fiber_annotation'])


if __name__=='__main__':unittest.main()
