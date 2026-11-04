import json
from pathlib import Path
import unittest
import cv2
import numpy as np

from tools.direct_fiber_template import DirectTracker,locate,visible

CONFIG=json.loads((Path(__file__).resolve().parents[1]/'configs/direct_fiber_template_20261029.json').read_text(encoding='utf-8'))


def synthetic():
    mask=np.zeros((160,200),np.uint8)
    cv2.fillConvexPoly(mask,np.array([[0,58],[92,72],[92,88],[0,102]],np.int32),1)
    cv2.ellipse(mask,(92,80),(8,8),0,-90,90,1,-1)
    frame=np.full((160,200,3),100,np.uint8);frame[mask>0]=(200,70,20)
    color=np.full((16,16),.05);hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV);color[int(hsv[80,40,0])*16//180,int(hsv[80,40,1])*16//256]=.95
    prior={'cap_radius_fraction':.05,'halfwidth_slope':.15,'histogram_bins':16,'appearance_probability':color.tolist()}
    return frame,mask>0,prior


class DirectTemplateTests(unittest.TestCase):
    def test_direct_image_search_localizes_known_fixed_shape(self):
        frame,truth,prior=synthetic();outputs,_=locate(frame,prior,CONFIG,requested=['image_template'])
        result=outputs['image_template'];self.assertTrue(result['record']['accepted'],result['record'])
        self.assertGreater((result['fiber']&truth).sum()/(result['fiber']|truth).sum(),.70)
        x,y,angle,length=result['record']['pose'];self.assertLess(np.linalg.norm([x+length*np.cos(angle)-100,y+length*np.sin(angle)-80]),12)
        self.assertFalse((result['fiber']&~result['hypothesis']).any())

    def test_image_branch_independent_of_neural_and_history(self):
        frame,_,prior=synthetic();tracker=DirectTracker(CONFIG['tracking']);tracker.update([150,150,2,100],True)
        a,_=locate(frame,prior,CONFIG,np.zeros(frame.shape[:2],np.float32),tracker)
        b,_=locate(frame,prior,CONFIG,np.ones(frame.shape[:2],np.float32),DirectTracker(CONFIG['tracking']))
        self.assertEqual(a['image_template']['record'],b['image_template']['record'])
        self.assertTrue(np.array_equal(a['image_template']['fiber'],b['image_template']['fiber']))

    def test_absence_and_prediction_do_not_force_visible_fiber(self):
        frame,_,prior=synthetic();flat=np.full_like(frame,100)
        output,_=locate(flat,prior,CONFIG,requested=['image_template'])
        self.assertFalse(output['image_template']['record']['accepted']);self.assertFalse(output['image_template']['fiber'].any())
        a=DirectTracker(CONFIG['tracking']);b=DirectTracker(CONFIG['tracking'])
        seq=[[0,80,0,100],[0,81,0,102],[0,80,0,101]]
        first=[a.update(p,True) for p in seq];prefix=[b.update(p,True) for p in seq[:2]];self.assertEqual(first[:2],prefix)
        for _ in range(3):self.assertFalse(a.update(None,False)['applied'])
        self.assertEqual(a.update(None,False)['status'],'expired');a.reset();self.assertIsNone(a.pose)
        mask,_=visible(frame,[0,80,0,100],False,prior,CONFIG);self.assertFalse(mask.any())
        b.update([0,80,0,100],True)
        black,_=locate(np.zeros_like(frame),prior,CONFIG,np.zeros(frame.shape[:2],np.float32),b)
        self.assertFalse(black['causal_template']['fiber'].any());self.assertEqual(black['causal_template']['record']['tracking']['status'],'predicted_not_visible')


if __name__=='__main__':unittest.main()
