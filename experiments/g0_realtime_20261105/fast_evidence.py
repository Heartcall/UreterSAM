"""Reject impossible colors before expensive distance/angle/score evaluation.

All mask predicates are the original predicates. Score values outside the cheap
necessary conditions are display-only for full images and are omitted. Small
terminal/curve probes keep the original continuous scores, including non-blue
pixels, because those scores affect S2's terminal decision.
"""
from __future__ import annotations

import numpy as np

from tools.fiber_connected_blue_growth import chroma
from tools.fiber_weak_stage2 import evidence as original_evidence, native_gradient
from tools.fiber_straight_semicircle import blue_identity as original_blue_identity


def fast_evidence(frame, seed, cfg, gradient=None):
    if frame.shape[1] < 32 or frame.shape[0]*frame.shape[1] < 4096:
        return original_evidence(frame, seed, cfg, gradient)
    color, noise = chroma(frame)
    blue = color[..., 2]-.5*(color[..., 0]+color[..., 1])
    bg = seed['bg']; gap = blue-(bg[2]-.5*(bg[0]+bg[1]))
    necessary = ((blue > cfg['minimum_blue_excess'])
                 & (gap > max(cfg['minimum_blue_excess'], cfg['blue_seed_gap_fraction']*seed['gap']))
                 & (noise < cfg['max_chroma_noise']))
    edge = native_gradient(frame) if gradient is None else np.asarray(gradient, np.float32)
    if edge.shape != frame.shape[:-1]:
        raise ValueError('Gradient/sample shape mismatch')
    score = np.zeros(frame.shape[:2], np.float32)
    strong = np.zeros_like(necessary); weak = strong.copy(); uncertain = strong.copy()
    if necessary.any():
        result = original_evidence(frame[necessary][:, None, :], seed, cfg, edge[necessary][:, None])
        score[necessary] = result[0].ravel()
        strong[necessary] = result[1].ravel()
        weak[necessary] = result[2].ravel()
        uncertain[necessary] = result[3].ravel()
    return score, strong, weak, uncertain, edge, noise


def fast_color(observer, bgr, seed, weak=False):
    if int(np.prod(bgr.shape[:-1])) < 4096:
        return observer.color_score(bgr, seed, weak) > 0
    color, noise = chroma(bgr)
    cfg = observer.cfg['color']; suffix = 'weak' if weak else 'normal'
    blue = color[..., 2]-.5*(color[..., 0]+color[..., 1])
    bg = seed['bg']; bgblue = bg[..., 2]-.5*(bg[..., 0]+bg[..., 1])
    necessary = ((cfg['max_noise']-noise > 0)
                 & (blue-bgblue-cfg['blue_gap_'+suffix+'_fraction']*seed['gap'] > 0))
    result = np.zeros_like(necessary)
    if necessary.any():
        result[necessary] = observer.color_score(bgr[necessary], seed, weak) > 0
    return result


def fast_blue_identity(frame, seed, cfg):
    if frame.shape[1] < 32 or frame.shape[0]*frame.shape[1] < 4096:
        return original_blue_identity(frame, seed, cfg)
    color, noise = chroma(frame)
    blue = color[..., 2]-.5*(color[..., 0]+color[..., 1])
    bg = seed['bg']; gap = blue-(bg[2]-.5*(bg[0]+bg[1]))
    necessary = ((blue > cfg['minimum_blue_advantage'])
                 & (gap > max(cfg['minimum_blue_advantage'], cfg['minimum_seed_gap_fraction']*seed['gap']))
                 & (noise < cfg['maximum_chroma_noise']))
    result = np.zeros_like(necessary)
    if necessary.any():
        result[necessary] = original_blue_identity(frame[necessary][:, None, :], seed, cfg)[0].ravel()
    return result, color, noise
