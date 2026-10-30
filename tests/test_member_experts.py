from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from tools.member_experts import MemberExperts


class Tiny(torch.nn.Module):
    def __init__(self, **_):
        super().__init__(); self.bias = torch.nn.Parameter(torch.zeros(3))
    def forward(self, x):
        return self.bias[None, :, None, None].expand(x.shape[0], 3, x.shape[2], x.shape[3])


class MemberExpertTests(unittest.TestCase):
    def test_only_one_fiber_model_is_constructed_and_selected_state_is_used(self):
        for member, bias in [('model1', [1., 2., 3.]), ('model2', [3., 2., 1.])]:
            stone = SimpleNamespace(models=(Tiny(),), model_count=1, size=4, device=torch.device('cpu'))
            state = {'size': 4, 'model_count': 2, 'model1': {'bias': torch.tensor([1., 2., 3.])},
                     'model2': {'bias': torch.tensor([3., 2., 1.])}}
            with patch('tools.member_experts.SegmentationPredictor', return_value=stone), patch('tools.member_experts.torch.load', return_value=state), patch('tools.member_experts.build_model', side_effect=Tiny) as build:
                experts = MemberExperts('stone', 'fiber', member, 'cpu')
                self.assertEqual(build.call_count, 1); self.assertEqual(experts.model_count, 2)
                self.assertEqual(len(experts.models), 2); self.assertEqual(experts.parameter_count, 6)
                self.assertTrue(all(not m.training for m in experts.models))
                torch.testing.assert_close(experts.fiber_model.bias, torch.tensor(bias))
                p = experts.probabilities(torch.zeros(1, 3, 4, 4))[1]
                torch.testing.assert_close(p[0, :, 0, 0], torch.tensor(bias).softmax(0))

    def test_native_probability_mapping_fiber_priority_and_no_input_mutation(self):
        stone = SimpleNamespace(models=(Tiny(),), model_count=1, size=4, device=torch.device('cpu'))
        stone.models[0].bias.data[:] = torch.tensor([0., 4., 0.])
        state = {'size': 4, 'model_count': 2, 'model1': {'bias': torch.tensor([0., 0., 5.])}}
        with patch('tools.member_experts.SegmentationPredictor', return_value=stone), patch('tools.member_experts.torch.load', return_value=state), patch('tools.member_experts.build_model', side_effect=Tiny):
            experts = MemberExperts('s', 'f', 'model1', 'cpu'); image = torch.zeros(1, 3, 4, 4)
            result = experts.predict_native(image, [(7, 9)])[0]
            self.assertEqual(tuple(result['mask'].shape), (7, 9)); self.assertTrue((result['mask'] == 2).all())
            self.assertTrue((result['stone_mask'] == 1).all()); self.assertEqual(result['fiber_probability'].dtype, torch.float32)
            self.assertTrue((image == 0).all())

    def test_invalid_member_and_mismatched_metadata_stop(self):
        with self.assertRaises(ValueError): MemberExperts('s', 'f', 'model3', 'cpu')
        stone = SimpleNamespace(model_count=1, size=4, device=torch.device('cpu'))
        with patch('tools.member_experts.SegmentationPredictor', return_value=stone), patch('tools.member_experts.torch.load', return_value={'size': 5}):
            with self.assertRaises(ValueError): MemberExperts('s', 'f', 'model1', 'cpu')


if __name__ == '__main__': unittest.main()
