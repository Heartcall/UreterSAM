"""Image-supported tube geometry and causal parameter filtering, no learning."""
from __future__ import annotations

import math
import cv2
import numpy as np


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def robust_line(x, y, config):
    design = np.column_stack([np.ones(len(x)), x])
    weights = np.ones(len(x))
    for _ in range(config['robust_iterations']):
        coef = np.linalg.lstsq(design * np.sqrt(weights[:, None]), y * np.sqrt(weights), rcond=None)[0]
        residual = y - design @ coef
        scale = max(.25, 1.4826 * np.median(np.abs(residual - np.median(residual))))
        weights = np.minimum(1, config['huber_mad_multiplier'] * scale / np.maximum(np.abs(residual), 1e-6))
    return coef, float(np.sqrt(np.average(residual ** 2, weights=weights)))


def _sample(array, points):
    return cv2.remap(array, points[:, 0].astype(np.float32)[None, :], points[:, 1].astype(np.float32)[None, :],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[0]


def tip_of(state):
    return np.asarray(state[:2]) + np.array([np.cos(state[2]), np.sin(state[2])]) * state[3]


def rasterize(state, shape):
    """Return full geometric hypothesis, not an observed visible mask."""
    x, y, angle, length, width0, width1, ratio = state
    u = np.array([np.cos(angle), np.sin(angle)]); v = np.array([-u[1], u[0]])
    yy, xx = np.indices(shape, dtype=np.float32)
    along = (xx - x) * u[0] + (yy - y) * u[1]
    normal = (xx - x) * v[0] + (yy - y) * v[1]
    rx = max(.5, width1 / 2 * ratio); center = max(.5, length - rx)
    halfwidth = (width0 + (width1 - width0) * np.clip(along / center, 0, 1)) / 2
    shaft = (along >= 0) & (along <= center) & (np.abs(normal) <= halfwidth)
    cap = (along >= center) & (((along - center) / rx) ** 2 + (normal / max(.5, width1 / 2)) ** 2 <= 1)
    return shaft | cap


def visible_fiber(probability, raw_fiber, state, threshold):
    if state is None:
        return raw_fiber.copy()
    return rasterize(state, raw_fiber.shape) & ((probability >= threshold) | raw_fiber)


def combine(stone_mask, fiber):
    return np.where(fiber, 2, np.where(stone_mask == 1, 1, 0)).astype(np.uint8)


def fit_shape(frame, probability, raw_fiber, fov, config):
    """Fit only current RGB edges and network candidates; never access GT/history."""
    h, w = raw_fiber.shape; size = min(h, w); scale = size / 400
    def reject(reason, **extra):
        return {'accepted': False, 'reason': reason, 'state': None, **extra}
    if frame.shape[:2] != (h, w) or probability.shape != (h, w):
        raise ValueError('Shape prior inputs must share native geometry')
    dist = cv2.distanceTransform(np.pad(fov.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw_fiber.astype(np.uint8), 8)
    candidates = []
    for i in range(1, count):
        selected = labels == i
        if stats[i, cv2.CC_STAT_AREA] >= config['minimum_candidate_pixels'] and int((selected & (dist <= config['entry_distance_fraction'] * size)).sum()) >= 3:
            candidates.append((int(selected.sum()), i))
    if not candidates:
        return reject('no_edge_attached_candidate')
    selected = labels == max(candidates)[1]
    yy, xx = np.where(selected); points = np.column_stack([xx, yy]).astype(np.float32)
    origin = points.mean(0); _, vectors = cv2.PCACompute(points, mean=None, maxComponents=1)
    u = vectors[0]; s = (points - origin) @ u
    ends = [np.percentile(s, q) for q in (5, 95)]
    scores = [float(np.median(dist[yy[np.abs(s - e) <= max(2 * scale, np.std(s) * .1)], xx[np.abs(s - e) <= max(2 * scale, np.std(s) * .1)]])) for e in ends]
    if scores[0] > scores[1]: u = -u
    v = np.array([-u[1], u[0]]); s = (points - origin) @ u; n = (points - origin) @ v
    start, front = np.percentile(s, [.2, 99.8]); length = float(front - start)
    if length < size * config['minimum_length_fraction']:
        return reject('shaft_too_short')
    # Color edge magnitude supports chromatic boundaries without a fixed blue-color rule.
    image = cv2.GaussianBlur(frame.astype(np.float32) / 255, (3, 3), .8)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(np.sum((gx * v[0] + gy * v[1]) ** 2, axis=2))
    ts = np.linspace(start + config['shaft_fit_fraction'][0] * length, start + config['shaft_fit_fraction'][1] * length, config['shaft_bins'])
    pairs = []; supported = 0
    for t in ts:
        section = np.abs(s - t) < max(1.5 * scale, length / config['shaft_bins'] * .7)
        if section.sum() < 4: continue
        low, high = np.percentile(n[section], [2, 98]); width = high - low
        if width < max(2 * scale, 1) or width > .35 * size: continue
        search = min(config['edge_search_max_fraction'] * size, max(config['edge_search_min_pixels_at400'] * scale, config['edge_search_fraction_of_width'] * width))
        bounds = []; contrasts = []
        for anchor in (low, high):
            ns = np.linspace(anchor - search, anchor + search, max(7, int(2 * search) + 1))
            coords = origin + u * t + ns[:, None] * v
            strengths = _sample(edge, coords)
            # Position prior prevents strong distant tissue boundaries stealing the fit.
            score = strengths * np.exp(-.5 * ((ns - anchor) / max(search * .65, 1)) ** 2)
            best = int(np.argmax(score)); bounds.append(float(ns[best])); contrasts.append(float(strengths[best]))
        if min(contrasts) >= config['edge_contrast_min']:
            pairs.append([t, *bounds]); supported += 1
    support = supported / len(ts)
    if supported < config['minimum_edge_pairs'] or support < config['minimum_shaft_support_fraction']:
        return reject('insufficient_two_sided_image_edges', shaft_support=support)
    pairs = np.asarray(pairs)
    lower, residual0 = robust_line(pairs[:, 0], pairs[:, 1], config)
    upper, residual1 = robust_line(pairs[:, 0], pairs[:, 2], config)
    def widths(t): return float((upper - lower) @ np.array([1, t]))
    def middle(t): return float((upper + lower) @ np.array([1, t]) / 2)
    w0, w1 = widths(start), widths(front)
    if min(w0, w1) < max(2 * scale, 1): return reject('nonpositive_or_unresolved_width')
    if not config['distal_to_proximal_width_ratio'][0] <= w1 / w0 <= config['distal_to_proximal_width_ratio'][1]:
        return reject('width_profile_out_of_family')
    residual = max(residual0, residual1)
    if residual > max(config['max_line_residual_pixels_at400'] * scale, config['max_line_residual_fraction_of_width'] * np.median(pairs[:, 2] - pairs[:, 1])):
        return reject('shaft_not_straight_enough', line_residual_px=residual)
    if abs(np.arctan((upper[1] + lower[1]) / 2)) > np.deg2rad(config['max_centerline_rotation_degrees']):
        return reject('axis_inconsistent_with_sides')
    best = None; angles = np.linspace(-np.pi / 2, np.pi / 2, 33)
    for ratio in config['cap_axis_ratios']:
        for offset in config['cap_front_offsets_in_radius']:
            tip = float(front + offset * w1 / 2); center = tip - ratio * w1 / 2
            ry = widths(center) / 2; rx = ry * ratio
            if min(ry, rx) <= .5: continue
            coords = origin + (center + rx * np.cos(angles))[:, None] * u + (middle(center) + ry * np.sin(angles))[:, None] * v
            nx = np.cos(angles)[:, None] * u / ratio + np.sin(angles)[:, None] * v
            nx /= np.linalg.norm(nx, axis=1)[:, None]
            dx = _sample(gx, coords); dy = _sample(gy, coords)
            strengths = np.sqrt(np.sum((dx * nx[:, 0, None] + dy * nx[:, 1, None]) ** 2, axis=1))
            cap_support = float((strengths >= config['edge_contrast_min']).mean())
            inside = _sample(probability, coords - nx * max(1, 2 * scale)); outside = _sample(probability, coords + nx * max(1, 2 * scale))
            score = float(np.minimum(strengths / .2, 1).mean() + .3 * (inside - outside).mean() - .1 * abs(offset))
            if best is None or score > best[0]: best = (score, tip, center, ry * 2, ratio, cap_support)
    if best is None or best[-1] < config['minimum_cap_support_fraction']:
        return reject('tip_arc_not_supported', shaft_support=support, line_residual_px=residual)
    _, tip, center, capwidth, ratio, cap_support = best
    proximal = origin + u * start + v * middle(start)
    distal = origin + u * tip + v * middle(tip)
    direction = distal - proximal; new_length = float(np.linalg.norm(direction))
    state = [float(proximal[0]), float(proximal[1]), float(np.arctan2(direction[1], direction[0])), new_length, w0, capwidth, ratio]
    tip_pixel = np.rint(distal).astype(int)
    if not 0 <= tip_pixel[0] < w or not 0 <= tip_pixel[1] < h or dist[tip_pixel[1], tip_pixel[0]] <= config['entry_distance_fraction'] * size:
        return reject('tip_near_or_outside_view_boundary')
    return {'accepted': True, 'reason': 'image_supported_fit', 'state': state, 'shaft_support': support,
            'cap_support': cap_support, 'line_residual_px': residual, 'tip_xy': tip_of(state).tolist(),
            'hypothesis_role': 'uncalibrated_2D_shape_not_visible_truth'}


class CausalShapeTracker:
    def __init__(self, config):
        self.config = config; self.reset()

    def reset(self):
        self.state = None; self.velocity = None; self.age = 0

    def step(self, measurement, shape, dt):
        if not dt > 0: raise ValueError('Positive causal time step required')
        if measurement is None:
            if self.state is None: return {'state': None, 'apply': False, 'status': 'no_state', 'age': 0}
            self.age += 1
            if self.age > self.config['maximum_prediction_only_frames']:
                self.reset(); return {'state': None, 'apply': False, 'status': 'expired_reset', 'age': 0}
            self.state = self.state + self.velocity * dt; self.state[2] = wrap(self.state[2])
            return {'state': self.state.tolist(), 'apply': False, 'status': 'predicted_not_applied', 'age': self.age}
        measured = np.asarray(measurement, dtype=float)
        if self.state is None:
            self.state = measured.copy(); self.velocity = np.zeros_like(measured); self.age = 0
            return {'state': self.state.tolist(), 'apply': True, 'status': 'initialized', 'age': 0}
        predicted = self.state + self.velocity * dt; residual = measured - predicted; residual[2] = wrap(residual[2])
        diagonal = math.hypot(*shape); ratios = measured[4:6] / np.maximum(predicted[4:6], 1e-6)
        reset = (np.linalg.norm(tip_of(measured) - tip_of(predicted)) > diagonal * self.config['reset_tip_jump_diagonal_fraction']
                 or abs(residual[3]) > diagonal * self.config['reset_length_jump_diagonal_fraction']
                 or abs(residual[2]) > np.deg2rad(self.config['reset_angle_jump_degrees'])
                 or np.any(ratios < self.config['reset_width_ratio'][0]) or np.any(ratios > self.config['reset_width_ratio'][1]))
        if reset:
            self.state = measured.copy(); self.velocity = np.zeros_like(measured); status = 'measurement_jump_reset'
        else:
            self.state = predicted + self.config['alpha'] * residual
            self.velocity = self.velocity + self.config['beta'] * residual / dt; status = 'updated'
        self.state[2] = wrap(self.state[2]); self.age = 0
        return {'state': self.state.tolist(), 'apply': True, 'status': status, 'age': 0}
