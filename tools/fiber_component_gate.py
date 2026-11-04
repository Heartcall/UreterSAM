"""Preserve a supported raw connected component; never rasterize ideal geometry."""
from __future__ import annotations

import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ureter_semis.inference import SegmentationPredictor
from ureter_semis.model import ensemble_logits
from ureter_semis.video_metrics import estimated_fov
from .classwise_experts import ClasswiseExperts
from .fiber_shape_prior import fit_shape, combine
from .replay_classwise_experts import pil_input

ARMS = ('classwise_raw_3net', 'classwise_component_gate_3net', 'old190_raw_1net', 'old190_component_gate_1net')


def nominate(raw_fiber, fov, config):
    """Same area, entrance criterion and tie breaking as preceding frozen fit_shape."""
    distance = cv2.distanceTransform(np.pad(fov.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    count, labels, sizes, _ = cv2.connectedComponentsWithStats(raw_fiber.astype(np.uint8), 8)
    candidates = []
    for label in range(1, count):
        component = labels == label
        if sizes[label, cv2.CC_STAT_AREA] >= config['minimum_candidate_pixels'] and int((component & (distance <= config['entry_distance_fraction'] * min(raw_fiber.shape))).sum()) >= 3:
            candidates.append((int(component.sum()), label))
    if not candidates:
        return None, {'components': count - 1, 'entry_candidates': 0, 'selected_label': None, 'selected_pixels': 0}
    area, label = max(candidates)
    return labels == label, {'components': count - 1, 'entry_candidates': len(candidates), 'selected_label': label, 'selected_pixels': area}


def gate_fiber(frame, probability, raw_fiber, config):
    fov = estimated_fov(frame, raw_fiber.shape)
    fitted = fit_shape(frame, probability, raw_fiber, fov, config)
    candidate, details = nominate(raw_fiber, fov, config)
    if fitted['accepted']:
        if candidate is None: raise RuntimeError('Frozen fit and component nomination disagree')
        result = candidate.copy()
    else:
        result = raw_fiber.copy()
    return result, {'accepted': fitted['accepted'], 'reason': fitted['reason'], 'fit': fitted,
                    **details, 'removed_pixels': int((raw_fiber & ~result).sum()),
                    'added_pixels': int((result & ~raw_fiber).sum()), 'preserves_selected_component': True}


class ComponentGatePredictor:
    def __init__(self, arm, checkpoint, geometry_config, device='cuda:0'):
        if arm not in ARMS: raise ValueError(arm)
        self.arm = arm; self.gated = '_component_gate_' in arm; self.config = geometry_config
        self.classwise = arm.startswith('classwise_')
        if self.classwise:
            self.expert = ClasswiseExperts(checkpoint['old190']['path'], checkpoint['A0']['path'], device)
        else:
            self.expert = SegmentationPredictor(checkpoint['old190']['path'], device)
        self.models = self.expert.models; self.device = self.expert.device; self.size = self.expert.size
        self.model_count = len(self.models)
        self.parameter_count = sum(p.numel() for m in self.models for p in m.parameters())
        if self.model_count != (3 if self.classwise else 1): raise ValueError('Required-model count differs')
        for model in self.models: model.eval()

    def sync(self):
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def predict(self, frame):
        self.sync(); start = time.perf_counter()
        image = pil_input(frame, self.size, self.device)
        self.sync(); preprocessed = time.perf_counter()
        if self.classwise:
            result = self.expert.predict_native(image, [frame.shape[:2]])[0]
            raw_gpu, stone_gpu = result['mask'], result['stone_mask']
            probability_gpu = result['fiber_probability'][2]
        else:
            probabilities = F.interpolate(ensemble_logits(self.models, image).softmax(1), size=frame.shape[:2], mode='bilinear', align_corners=False)[0]
            raw_gpu = probabilities.argmax(0).byte(); stone_gpu = raw_gpu
            probability_gpu = probabilities[2]
        self.sync(); forwarded = time.perf_counter()
        raw = raw_gpu.cpu().numpy().copy(); stone = stone_gpu.cpu().numpy().copy()
        probability = probability_gpu.cpu().numpy().copy()
        self.sync(); transferred = time.perf_counter()
        if self.gated:
            fiber, record = gate_fiber(frame, probability, raw == 2, self.config)
        else:
            fiber = raw == 2
            record = {'accepted': None, 'reason': 'gate_disabled', 'removed_pixels': 0, 'added_pixels': 0}
        fitted = time.perf_counter()
        mask = combine(stone, fiber) if self.gated else raw.copy()
        finished = time.perf_counter()
        timing = {'pipeline_ms': (finished-start)*1000, 'preprocess_ms': (preprocessed-start)*1000,
                  'forward_and_native_gpu_ms': (forwarded-preprocessed)*1000, 'cpu_transfer_ms': (transferred-forwarded)*1000,
                  'gate_ms': (fitted-transferred)*1000, 'fusion_ms': (finished-fitted)*1000}
        return {'mask': mask, 'raw_mask': raw, 'stone_mask': stone, 'gate': record, 'timing': timing}
