import json
from pathlib import Path
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import torch
from types import SimpleNamespace

from tools.fiber_component_gate import gate_fiber, nominate, ComponentGatePredictor

CONFIG=json.loads((Path(__file__).resolve().parents[1]/'configs/fiber_geometry_prior_20261028.json').read_text(encoding='utf-8'))['fit']


class ComponentGateTests(unittest.TestCase):
    def test_single_network_arm_never_loads_a0_or_three_experts(self):
        class Constant(torch.nn.Module):
            def forward(self,x):
                result=torch.zeros(len(x),3,*x.shape[-2:]);result[:,0]=5;return result
        expert=SimpleNamespace(models=(Constant(),),device=torch.device('cpu'),size=32)
        with patch('tools.fiber_component_gate.SegmentationPredictor',return_value=expert) as load, patch('tools.fiber_component_gate.ClasswiseExperts',side_effect=AssertionError('A0 must not load')):
            p=ComponentGatePredictor('old190_raw_1net',{'old190':{'path':'old.pt'},'A0':{'path':'forbidden.pt'}},CONFIG,'cpu')
            result=p.predict(np.zeros((40,50,3),np.uint8))
            self.assertEqual(p.model_count,1);load.assert_called_once_with('old.pt','cpu')
            self.assertEqual(result['mask'].shape,(40,50));self.assertFalse(result['mask'].any())

    def test_preserve_entire_supported_raw_component_not_ideal_template(self):
        truth=np.zeros((160,200),np.uint8)
        cv2.rectangle(truth,(0,68),(98,92),1,-1);cv2.ellipse(truth,(98,80),(12,12),0,-90,90,1,-1)
        raw=truth.copy();cv2.circle(raw,(165,35),8,1,-1)
        frame=np.full((160,200,3),100,np.uint8);frame[truth>0]=(200,70,20)
        probability=np.where(raw,.95,.02).astype(np.float32)
        output,record=gate_fiber(frame,probability,raw>0,CONFIG)
        self.assertTrue(record['accepted'],record)
        self.assertTrue(np.array_equal(output,truth>0))
        self.assertEqual(record['added_pixels'],0)
        self.assertEqual(int(output.sum()),record['selected_pixels'])

    def test_unsupported_fit_returns_complete_raw_mask(self):
        raw=np.zeros((100,120),bool);raw[40:60,:90]=True;raw[80:85,100:105]=True
        before=raw.copy()
        output,record=gate_fiber(np.full((100,120,3),100,np.uint8),raw.astype(np.float32),raw,CONFIG)
        self.assertFalse(record['accepted']);self.assertTrue(np.array_equal(output,before))
        self.assertTrue(np.array_equal(raw,before));self.assertEqual(record['removed_pixels'],0)

    def test_detached_true_tip_is_explicitly_a_possible_loss(self):
        raw=np.zeros((100,120),bool);raw[40:60,:70]=True;raw[45:55,74:80]=True
        candidate,_=nominate(raw,np.ones(raw.shape,bool),CONFIG)
        self.assertIsNotNone(candidate);self.assertFalse(candidate[50,77])
        with patch('tools.fiber_component_gate.fit_shape',return_value={'accepted':True,'reason':'image_supported_fit','state':[0,50,0,70,20,20,1]}):
            result,record=gate_fiber(np.zeros((100,120,3),np.uint8),raw.astype(np.float32),raw,CONFIG)
        self.assertEqual(record['removed_pixels'],60);self.assertTrue(np.array_equal(result,candidate))
        self.assertEqual(record['added_pixels'],0)


if __name__=='__main__':unittest.main()
