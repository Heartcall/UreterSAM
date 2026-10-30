"""Bounded positive color evidence; original visible labels are immutable anchors."""
from __future__ import annotations

import cv2
import numpy as np


def blue_evidence(frame, settings):
    """Analytic evidence, not a fitted or calibrated probability of fiber identity."""
    bgr = frame.astype(np.float32) / 255
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    total = b + g + r
    excess = (b - np.maximum(r, g)) / np.maximum(total, 1e-6)
    z = (excess - settings['blue_excess_center']) / settings['blue_excess_scale']
    score = (1 / (1 + np.exp(-np.clip(z, -30, 30)))).astype(np.float32)
    high, low = bgr.max(2), bgr.min(2)
    saturation = (high - low) / np.maximum(high, 1e-6)
    reliability = np.minimum(saturation / settings['saturation_reliability_scale'], 1) * np.minimum(high / settings['value_reliability_scale'], 1)
    boost = settings['maximum_log_odds_boost'] * reliability * np.maximum(2 * score - 1, 0)
    return {'blue_score_uncalibrated': score, 'color_reliability': reliability,
            'blue_excess': excess, 'log_odds_boost': boost.astype(np.float32)}


def refine(frame, raw, log_margin, settings, mode='blue_aplus'):
    if raw.ndim != 2 or frame.shape != (*raw.shape, 3) or log_margin.shape != raw.shape:
        raise ValueError('Native RGB/mask/probability geometry mismatch')
    if mode not in {'blue_aplus', 'neural_relaxation_control'}:
        raise ValueError('Unknown fixed comparison scheme')
    record = {'scheme': mode, 'proposed_pixels': 0, 'added_pixels': 0, 'applied': False,
              'fallback': None, 'fitted_geometry': False, 'tip_metrics': None}
    result = raw.copy()
    anchor = raw == 2; count = int(anchor.sum())
    if count < settings['minimum_anchor_pixels']:
        record['fallback'] = 'insufficient_network_fiber_anchor'
        return result, record
    if not np.isfinite(log_margin).all():
        record['fallback'] = 'invalid_neural_margin'
        return result, record
    radius = max(1, int(round(settings['adjacent_radius_at400'] * min(raw.shape) / 400)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    nearby = cv2.dilate(anchor.astype(np.uint8), kernel).astype(bool) & (raw == 0)
    boost = blue_evidence(frame, settings)['log_odds_boost'] if mode == 'blue_aplus' else settings['maximum_log_odds_boost']
    proposed = nearby & (log_margin + boost > 0)
    # A tie/invalid raw inconsistency cannot add a pixel with no positive evidence.
    proposed &= (boost > 0)
    additions = int(proposed.sum())
    budget = min(int(np.ceil(count * settings['maximum_addition_fraction_of_anchor'])),
                 int(np.ceil(raw.size * settings['maximum_addition_fraction_of_image'])))
    record.update(proposed_pixels=additions, budget_pixels=budget, radius_pixels=radius)
    if additions > budget:
        record['fallback'] = 'addition_budget'
        return result, record
    if not additions:
        record['fallback'] = 'no_supported_uncertain_background'
        return result, record
    result[proposed] = 2
    record.update(added_pixels=additions, applied=True)
    if np.any((raw == 2) & (result != 2)) or np.any((raw == 1) != (result == 1)):
        raise AssertionError('Protected visible labels changed')
    return result, record
