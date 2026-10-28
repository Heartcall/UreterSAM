"""Inference-only category delegation; expert probabilities stay separate."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ureter_semis.inference import SegmentationPredictor
from ureter_semis.model import ensemble_logits


def fuse_masks(stone_mask: torch.Tensor,fiber_mask: torch.Tensor) -> torch.Tensor:
    if stone_mask.shape!=fiber_mask.shape or stone_mask.ndim!=2:
        raise ValueError('Expert masks must share native2D geometry')
    result=torch.zeros_like(stone_mask,dtype=torch.uint8)
    result[stone_mask==1]=1
    result[fiber_mask==2]=2
    return result


def native_outputs(stone_probability,fiber_probability,shape):
    if any(p.ndim!=3 or p.shape[0]!=3 for p in (stone_probability,fiber_probability)):
        raise ValueError('Expected separate3class expert probabilities')
    def resize(p):
        return F.interpolate(p.unsqueeze(0).float(),size=shape,mode='bilinear',align_corners=False)[0]
    stone_probability,fiber_probability=resize(stone_probability),resize(fiber_probability)
    stone_mask,fiber_mask=stone_probability.argmax(0),fiber_probability.argmax(0)
    return {'mask':fuse_masks(stone_mask,fiber_mask),'stone_mask':stone_mask.byte(),'fiber_mask':fiber_mask.byte(),
        'stone_probability':stone_probability,'fiber_probability':fiber_probability}


class ClasswiseExperts:
    def __init__(self,stone_checkpoint,fiber_checkpoint,device='cuda:0'):
        self.stone=SegmentationPredictor(stone_checkpoint,device)
        self.fiber=SegmentationPredictor(fiber_checkpoint,device)
        if self.stone.size!=self.fiber.size or self.stone.model_count!=1 or self.fiber.model_count!=2:
            raise ValueError('Frozen experiment requires same-input single stone and dual fiber experts')
        self.size,self.device=self.stone.size,self.stone.device
        self.models=(*self.stone.models,*self.fiber.models)
        self.model_count=len(self.models)
        self.parameter_count=sum(p.numel() for model in self.models for p in model.parameters())
        for model in self.models:model.eval()

    def sync(self):
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def probabilities(self,images):
        return (ensemble_logits(self.stone.models,images).softmax(1),
                ensemble_logits(self.fiber.models,images).softmax(1))

    @torch.inference_mode()
    def predict_native(self,images,shapes):
        if len(images)!=len(shapes):raise ValueError('Need one native shape per image')
        stone,fiber=self.probabilities(images.to(self.device))
        return [native_outputs(stone[i],fiber[i],shape) for i,shape in enumerate(shapes)]
