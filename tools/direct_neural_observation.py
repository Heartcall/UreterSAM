"""Single-factor foreground-field control of the frozen fixed shape experiment."""
from __future__ import annotations

import time
import cv2
import numpy as np
from . import direct_fiber_template as frozen

IMAGE = 'fixed_image_observation_template'
NEURAL = 'fixed_neural_observation_template'


def probability_field(probability, frame):
    value = np.asarray(probability)
    if value.shape != frame.shape[:2] or not np.isfinite(value).all():
        raise ValueError('Expected finite native HxW fiber probability')
    if value.min() < 0 or value.max() > 1:
        raise ValueError('Fiber probability must be in [0,1]')
    return value.astype(np.float32)


def neural_observation(frame, prior, config, probability):
    probability = probability_field(probability, frame)
    obs = frozen.observation(frame, prior, config)
    if obs is None:
        return None
    field = cv2.resize(probability, (obs['shape'][1], obs['shape'][0]), interpolation=cv2.INTER_LINEAR)
    value = cv2.cvtColor(obs['image'], cv2.COLOR_BGR2HSV)[:, :, 2]
    # Preserve the predecessor's dark-pixel exclusion and all RGB/proposal fields.
    obs['appearance'] = np.where(value > 10, field, 0).astype(np.float32)
    return obs


def neural_visible(frame, pose, accepted, prior, config, probability):
    probability = probability_field(probability, frame)
    # The predecessor is the sole definition of the complete fixed shape.
    _, template = frozen.visible(frame, pose, accepted, prior, config)
    value = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
    return template & (probability >= config['acceptance']['visible_appearance_min']) & (value > 10), template


def locate(frame, prior, config, network_probability=None, tracker=None, requested=None):
    names = set(requested if requested is not None else (IMAGE, NEURAL))
    if names - {IMAGE, NEURAL, 'causal_template'}:
        raise ValueError('Unknown observation-control arm')
    out = {}; timing = {}
    if IMAGE in names:
        result, times = frozen.locate(frame, prior, config, requested=['image_template'])
        out[IMAGE] = result['image_template']
        timing['image_total_ms'] = sum(times.values())
    if not names.intersection({NEURAL, 'causal_template'}):
        return out, timing
    started = time.perf_counter()
    if network_probability is None:
        raise ValueError('Neural observation requires native fiber probability')
    obs = neural_observation(frame, prior, config, network_probability)
    coarse = frozen.score_poses(obs, obs['poses'], prior, config) if obs is not None else None
    timing['neural_shared_ms'] = (time.perf_counter() - started) * 1000
    for name in (NEURAL, 'causal_template'):
        if name not in names:
            continue
        if name == 'causal_template' and tracker is None:
            raise ValueError('Causal arm requires a per-video tracker')
        started = time.perf_counter()
        if obs is None:
            record = {'accepted': False, 'checks': {'fov': False}, 'pose': None,
                      'objective': None, 'features': {}, 'role': 'no_observation'}
            if name == 'causal_template':
                record['tracking'] = tracker.update(None, False)
            mask = hypothesis = np.zeros(frame.shape[:2], bool)
        else:
            poses = frozen.fine_poses(obs['poses'], coarse['score'], config, obs['shape'], obs['boundary'])
            history = None
            if name == 'causal_template':
                prediction = tracker.prediction()
                if prediction is not None:
                    history = prediction.copy(); history[[0, 1, 3]] *= obs['ratio']
                    local = {**config, 'search': {**config['search'], 'topk': 1}}
                    poses = np.vstack([poses, frozen.fine_poses(history[None, :], np.array([0.]), local, obs['shape'], obs['boundary'])])
            # No network_soft bonus: the neural field is already the foreground evidence.
            scores = frozen.score_poses(obs, poses, prior, config)
            record = frozen.pick(obs, poses, scores, 'score', config, history)
            record['pose'] = frozen.native_pose(record, obs['ratio'])
            if name == 'causal_template':
                record['tracking'] = tracker.update(record['pose'], record['accepted'])
            mask, hypothesis = neural_visible(frame, record['pose'], record['accepted'], prior, config, network_probability)
        out[name] = {'record': record, 'fiber': mask, 'hypothesis': hypothesis}
        timing[name + '_ms'] = (time.perf_counter() - started) * 1000
    return out, timing
