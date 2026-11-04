import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from tools.classwise_experts import fuse_masks,native_outputs,ClasswiseExperts


class Constant(torch.nn.Module):
    def __init__(self,cls):super().__init__();self.cls=cls
    def forward(self,x):
        output=torch.zeros(len(x),3,*x.shape[-2:],device=x.device);output[:,self.cls]=5;return output


class ClasswiseExpertTests(unittest.TestCase):
    def test_fiber_priority_and_class_delegation_without_input_mutation(self):
        stone=torch.tensor([[1,1,2],[0,2,0]])
        fiber=torch.tensor([[0,2,0],[1,2,0]])
        result=fuse_masks(stone,fiber)
        self.assertTrue(torch.equal(result,torch.tensor([[1,2,0],[0,2,0]],dtype=torch.uint8)))
        self.assertEqual(int(stone[0,1]),1)
        self.assertTrue(torch.equal(result==2,fiber==2))
        self.assertTrue(bool(((result==1)&(stone!=1)).sum()==0))
        with self.assertRaises(ValueError):fuse_masks(stone,fiber[:1])

    def test_probability_resize_precedes_each_argmax(self):
        fg=torch.tensor([[.51,.01],[.01,.01]])
        stone=torch.stack([1-fg,fg,torch.zeros_like(fg)])
        fiber=torch.zeros_like(stone);fiber[0]=1
        result=native_outputs(stone,fiber,(1,1))
        self.assertEqual(int(result['mask'][0,0]),0)
        self.assertEqual(int(stone.argmax(0)[0,0]),1)
        self.assertEqual(result['stone_probability'].shape,(3,1,1))
        self.assertEqual(result['fiber_probability'].shape,(3,1,1))

    def test_three_network_eval_only_batch_and_native_shapes(self):
        stone=SimpleNamespace(size=32,model_count=1,device=torch.device('cpu'),models=(Constant(1),))
        fiber=SimpleNamespace(size=32,model_count=2,device=torch.device('cpu'),models=(Constant(2),Constant(2)))
        with patch('tools.classwise_experts.SegmentationPredictor',side_effect=[stone,fiber]):
            experts=ClasswiseExperts('stone.pt','fiber.pt',device='cpu')
            self.assertEqual(experts.model_count,3)
            result=experts.predict_native(torch.zeros(2,3,32,32),[(40,60),(64,64)])
            self.assertEqual(result[0]['mask'].shape,(40,60));self.assertEqual(result[1]['mask'].shape,(64,64))
            self.assertTrue(bool((result[0]['mask']==2).all()))
            self.assertFalse(result[0]['fiber_probability'].requires_grad)
            self.assertTrue(all(not model.training for model in experts.models))


if __name__=='__main__':unittest.main()
