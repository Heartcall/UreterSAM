"""Existing A0 single-member ablation: only the selected fiber model is constructed."""
from __future__ import annotations

import torch

from ureter_semis.inference import SegmentationPredictor
from ureter_semis.model import build_model
from .classwise_experts import native_outputs


class MemberExperts:
    def __init__(self, stone_checkpoint, fiber_checkpoint, member, device='cuda:0'):
        if member not in {'model1', 'model2'}:
            raise ValueError('Only the two predefined A0 members may be selected')
        self.stone = SegmentationPredictor(stone_checkpoint, device)
        if self.stone.model_count != 1:
            raise ValueError('Frozen stone expert must have one network')
        state = torch.load(fiber_checkpoint, map_location='cpu', weights_only=True)
        classes = state.get('classes', ['background', 'stone', 'fiber'])
        if classes != ['background', 'stone', 'fiber'] or int(state.get('model_count', 2)) != 2:
            raise ValueError('Expected the original3-class A0 dual checkpoint')
        if int(state.get('size', 320)) != self.stone.size:
            raise ValueError('Shared preprocessing size differs')
        architecture = state.get('architecture', 'unet_resnet34')
        kwargs = {} if architecture == 'unet_resnet34' else {'architecture': architecture}
        self.device = self.stone.device; self.size = self.stone.size; self.member = member
        self.fiber_model = build_model(pretrained=False, **kwargs).to(self.device).eval()
        self.fiber_model.load_state_dict(state[member])
        del state  # The unused CPU state is not retained; its network was never built.
        self.models = (*self.stone.models, self.fiber_model)
        self.model_count = len(self.models)
        self.parameter_count = sum(p.numel() for model in self.models for p in model.parameters())
        for model in self.models: model.eval()

    def sync(self):
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def probabilities(self, images):
        return self.stone.models[0](images).softmax(1), self.fiber_model(images).softmax(1)

    @torch.inference_mode()
    def predict_native(self, images, shapes):
        if len(images) != len(shapes): raise ValueError('Need one native shape per image')
        sp, fp = self.probabilities(images.to(self.device))
        return [native_outputs(sp[i], fp[i], shape) for i, shape in enumerate(shapes)]
