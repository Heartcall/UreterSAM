import unittest
import numpy as np
from tools.fiber_tip_diagnostics import loss_partition, reference_axis, appearance_fields, distributions

SETTINGS={'minimum_reference_pixels':16,'minimum_axis_eigenvalue_ratio':3,'minimum_end_distance_difference_fraction':.01}

class FiberTipDiagnosticTests(unittest.TestCase):
    def test_first_exclusion_stages_are_disjoint_and_conserve_lost_tp(self):
        target=np.full((1,5),2,np.uint8);raw=target.copy();new=np.array([[0,0,0,2,2]],np.uint8)
        hyp=np.array([[False,True,True,True,True]]);field=np.array([[.9,.2,.8,.8,.8]]);value=np.array([[100,100,0,100,100]])
        result=loss_partition(target,raw,new,True,hyp,field,.5,value)
        self.assertEqual([result[k] for k in ('outside_hypothesis','foreground_field_rejected','darkness_rejected','unexplained')],[1,1,1,0])
        self.assertEqual(result['lost_tp'],3)
        rejected=loss_partition(target,raw,np.zeros_like(raw),False,hyp,field,.5,value)
        self.assertEqual(rejected['fit_rejected'],5);self.assertEqual(rejected['unexplained'],0)

    def test_unknown_pixels_are_not_false_positives(self):
        t=np.array([[255,0,2]],np.uint8);raw=np.array([[2,2,2]],np.uint8);new=np.zeros_like(raw)
        r=loss_partition(t,raw,new,False,np.zeros_like(raw,bool),np.zeros_like(raw),.5,np.ones_like(raw)*100)
        self.assertEqual(r['removed_fp'],1);self.assertEqual(r['lost_tp'],1)

    def test_reference_axis_uses_disconnected_parts_without_generating_tip_truth(self):
        target=np.zeros((100,120),np.uint8);target[45:55,:45]=2;target[47:53,55:80]=2
        record,t=reference_axis(np.full((100,120,3),100,np.uint8),target,SETTINGS)
        self.assertTrue(record['valid']);self.assertGreater(t[50,79],t[50,0]);self.assertIn('NOT_anatomical_tip',record['role'])
        r,t=reference_axis(np.zeros((10,10,3),np.uint8),np.zeros((10,10),np.uint8),SETTINGS)
        self.assertFalse(r['valid']);self.assertIsNone(t)

    def test_color_channels_white_hue_reliability_and_input_immutability(self):
        frame=np.array([[[255,0,0],[255,255,255],[0,0,0]]],np.uint8);before=frame.copy()
        p=np.full((3,1,3),1/3,np.float32);arrays={'a0_native':p,'a0_member1_native':p,'a0_member2_native':p}
        f=appearance_fields(frame,arrays,{'valid':False})
        self.assertEqual(float(f['blue'][0,0]),1);self.assertEqual(float(f['red'][0,0]),0)
        self.assertEqual(float(f['saturation'][0,1]),0);self.assertTrue(all(np.isfinite(x).all() for x in f.values()))
        white=np.array([[False,True,False]]);self.assertIsNone(distributions(f,white,[5,50,95])['saturation_weighted_hue_degrees'])
        np.testing.assert_array_equal(frame,before)

if __name__=='__main__':unittest.main()
