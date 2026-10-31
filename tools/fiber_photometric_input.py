"""Fixed stateless fiber-input controls; no labels, history or model fusion."""
from __future__ import annotations

import time

import cv2
import numpy as np
from PIL import Image
import torch

from ureter_semis.data import MEAN, STD
from .classwise_experts import native_outputs

ARMS = ('raw2', 'fiber_vlift', 'fiber_channel_gamma')
GAMMA = 0.7


def rgb_grid(frame, size=256):
    """The frozen PIL resize, before normalization, with no uint8 roundtrip."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / np.float32(255.)


def transform_rgb(rgb, arm, gamma=GAMMA):
    if arm not in ARMS:
        raise ValueError('Unknown fixed input arm')
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError('Gamma must be finite and positive')
    if arm == 'raw2':
        return rgb
    if arm == 'fiber_channel_gamma':
        return np.clip(np.power(rgb, np.float32(gamma)), 0, 1).astype(np.float32, copy=False)
    value = rgb.max(axis=2, keepdims=True)
    ratios = np.divide(rgb, value, out=np.zeros_like(rgb), where=value > 0)
    return np.clip(ratios * np.power(value, np.float32(gamma)), 0, 1).astype(np.float32, copy=False)


def normalized_tensor(rgb, device):
    return torch.from_numpy((rgb.transpose(2, 0, 1) - MEAN) / STD).unsqueeze(0).to(device)


def prepare_inputs(frame, arm, size, device, sync):
    if arm not in ARMS:
        raise ValueError('Unknown fixed input arm')
    start = time.perf_counter()
    original = rgb_grid(frame, size)
    resized = time.perf_counter()
    transformed = transform_rgb(original, arm)
    changed = time.perf_counter()
    stone = normalized_tensor(original, device)
    fiber = stone if arm == 'raw2' else normalized_tensor(transformed, device)
    sync()
    prepared = time.perf_counter()
    return stone, fiber, {
        'rgb_pil_grid_ms': (resized-start)*1000,
        'fixed_transform_ms': (changed-resized)*1000,
        'normalize_transfer_ms': (prepared-changed)*1000,
    }


@torch.inference_mode()
def infer(experts, frame, arm):
    """Exactly two forwards. Return both native three-class FP32 maps and mask."""
    experts.sync()
    start = time.perf_counter()
    stone_input, fiber_input, timings = prepare_inputs(frame, arm, experts.size, experts.device, experts.sync)
    prepared = time.perf_counter()
    stone = experts.stone.models[0](stone_input).softmax(1)
    experts.sync()
    stone_end = time.perf_counter()
    fiber = experts.fiber_model(fiber_input).softmax(1)
    experts.sync()
    fiber_end = time.perf_counter()
    result = native_outputs(stone[0], fiber[0], frame.shape[:2])
    mask = result['mask'].cpu().numpy().copy()
    sp = result['stone_probability'].cpu().numpy().copy()
    fp = result['fiber_probability'].cpu().numpy().copy()
    experts.sync()
    end = time.perf_counter()
    timings.update(stone_forward_ms=(stone_end-prepared)*1000,
                   fiber_forward_ms=(fiber_end-stone_end)*1000,
                   native_mapping_transfer_ms=(end-fiber_end)*1000,
                   pipeline_total_ms=(end-start)*1000)
    return mask, sp, fp, timings
