import unittest
from types import SimpleNamespace

import numpy as np
import torch

from tools.fiber_photometric_input import ARMS, infer, normalized_tensor, prepare_inputs, rgb_grid, transform_rgb
from tools.replay_classwise_experts import pil_input


class CounterModel(torch.nn.Module):
    def __init__(self, cls):
        super().__init__()
        self.cls = cls
        self.calls = 0
        self.last = None

    def forward(self, x):
        self.calls += 1
        self.last = x.clone()
        logits = torch.zeros(x.shape[0], 3, *x.shape[2:])
        logits[:, self.cls] = 3
        return logits


class PhotometricTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.random.default_rng(3).integers(0,256,(37,51,3),dtype=np.uint8)
        self.rgb = rgb_grid(self.frame)

    def test_original_pil_normalization_exact(self):
        self.assertTrue(torch.equal(normalized_tensor(self.rgb,'cpu'),pil_input(self.frame,256,'cpu')))

    def test_raw_shares_tensor_and_transformed_separates(self):
        for arm in ARMS:
            a,b,_ = prepare_inputs(self.frame,arm,256,'cpu',lambda:None)
            self.assertEqual(a.data_ptr()==b.data_ptr(),arm=='raw2')
            self.assertTrue(torch.equal(a,pil_input(self.frame,256,'cpu')))

    def test_endpoints_gray_and_pure_blue(self):
        rgb = np.array([[[0,0,0],[1,1,1],[0,0,.25],[.25,.25,.25]]],np.float32)
        for arm in ARMS[1:]:
            y = transform_rgb(rgb,arm)
            np.testing.assert_array_equal(y[0,:2],rgb[0,:2])
            self.assertTrue(np.all(y[0,2,:2]==0))
            np.testing.assert_allclose(y[0,3],.25**.7,rtol=1e-6)

    def test_gamma_one_analytic_identity(self):
        for arm in ARMS[1:]:
            np.testing.assert_allclose(transform_rgb(self.rgb,arm,1),self.rgb,atol=6e-8,rtol=1e-7)

    def test_no_inplace_input_and_float_range(self):
        rgb_before = self.rgb.copy(); frame_before = self.frame.copy()
        for arm in ARMS:
            y=transform_rgb(self.rgb,arm)
            self.assertEqual(y.dtype,np.float32)
            self.assertTrue(np.isfinite(y).all() and y.min()>=0 and y.max()<=1)
        np.testing.assert_array_equal(self.rgb,rgb_before)
        np.testing.assert_array_equal(self.frame,frame_before)

    def test_common_scale_preserves_chromaticity(self):
        y=transform_rgb(self.rgb,'fiber_vlift')
        np.testing.assert_allclose(y/y.sum(2,keepdims=True),self.rgb/self.rgb.sum(2,keepdims=True),atol=1e-7)

    def test_same_value_different_channel_ratios(self):
        a=transform_rgb(self.rgb,'fiber_vlift'); b=transform_rgb(self.rgb,'fiber_channel_gamma')
        np.testing.assert_allclose(a.max(2),b.max(2),atol=1e-7)
        self.assertGreater(float(np.abs(a-b).max()),.01)

    def test_exact_two_forwards_native_shape_and_fiber_priority(self):
        for arm in ARMS:
            stone=CounterModel(1); fiber=CounterModel(2)
            experts=SimpleNamespace(stone=SimpleNamespace(models=[stone]),fiber_model=fiber,size=256,device='cpu',sync=lambda:None)
            mask,sp,fp,t = infer(experts,self.frame,arm)
            self.assertEqual((stone.calls,fiber.calls),(1,1))
            self.assertEqual(mask.shape,self.frame.shape[:2]); self.assertTrue((mask==2).all())
            self.assertEqual(sp.shape,(3,*mask.shape)); self.assertEqual(fp.dtype,np.float32)
            self.assertTrue(torch.equal(stone.last,pil_input(self.frame,256,'cpu')))
            self.assertTrue(all(v>=0 for v in t.values()))

    def test_stateless_order_and_no_annotation_argument(self):
        a=transform_rgb(self.rgb,'fiber_vlift').copy()
        transform_rgb(np.zeros_like(self.rgb),'fiber_channel_gamma')
        np.testing.assert_array_equal(a,transform_rgb(self.rgb,'fiber_vlift'))
        with self.assertRaises(TypeError): transform_rgb(self.rgb,'fiber_vlift',target=np.zeros((2,2)))

    def test_reject_unknown_arm_and_invalid_gamma(self):
        with self.assertRaises(ValueError): transform_rgb(self.rgb,'adaptive')
        for g in (0,-1,float('nan')):
            with self.assertRaises(ValueError): transform_rgb(self.rgb,'fiber_vlift',g)


if __name__=='__main__': unittest.main()
