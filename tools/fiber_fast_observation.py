"""T1 bounded RGB/probability observations; never reads or modifies semantic masks.

All coordinates and lengths exported in native image pixels. Front means a visible
transition candidate, not an anatomical tip or recovered occluded surface.
"""
from __future__ import annotations
import hashlib
import math
import time
import cv2
import numpy as np


def wrap(x):
    return float((x + np.pi) % (2 * np.pi) - np.pi)


def grid_to_native(points, shape, side=160):
    return (np.asarray(points, float) + .5) * np.array([shape[1], shape[0]]) / side - .5


def sample(image, points):
    """Vectorized bilinear sampling; invalid samples are zero AND explicitly invalid."""
    points = np.asarray(points, float)
    h, w = image.shape[:2]
    good = np.isfinite(points).all(-1) & (points[..., 0] >= 0) & (points[..., 0] <= w-1) & (points[..., 1] >= 0) & (points[..., 1] <= h-1)
    q = np.nan_to_num(points)
    x = np.clip(q[..., 0], 0, w-1); y = np.clip(q[..., 1], 0, h-1)
    x0 = x.astype(int); y0 = y.astype(int); x1 = np.minimum(x0+1, w-1); y1 = np.minimum(y0+1, h-1)
    dx = x-x0; dy = y-y0
    if image.ndim == 3:
        dx = dx[..., None]; dy = dy[..., None]
    value = (image[y0, x0]*(1-dx)*(1-dy) + image[y0, x1]*dx*(1-dy) + image[y1, x0]*(1-dx)*dy + image[y1, x1]*dx*dy)
    return value, good


def robust_line(x, y, iterations, multiplier):
    a = np.stack([x, np.ones_like(x)], axis=1)
    weights = np.ones_like(x)
    coeff = np.linalg.lstsq(a, y, rcond=None)[0]
    for _ in range(iterations):
        residual = y-a@coeff
        mad = float(np.median(np.abs(residual-np.median(residual))))
        cutoff = multiplier*max(mad, 1e-6)
        weights = np.minimum(1., cutoff/np.maximum(np.abs(residual), 1e-12))
        coeff = np.linalg.lstsq(a*np.sqrt(weights[:, None]), y*np.sqrt(weights), rcond=None)[0]
    return coeff, y-a@coeff


def select_current(candidates, config):
    """Deduplicate metadata, then evaluate current-only identity ambiguity."""
    q = config['quality']; kept = []; duplicates = []
    for c in sorted(candidates, key=lambda c: (-c['score'], c['label'])):
        duplicate = None
        for prior in kept:
            x, y = set(c['_support_ids']), set(prior['_support_ids'])
            iou = len(x & y)/max(len(x | y), 1)
            angle = abs(wrap(c['axis_theta']-prior['axis_theta']))
            angle = min(angle, np.pi-angle)
            if iou >= q['current_duplicate_support_iou_min'] and angle <= np.deg2rad(q['current_duplicate_axis_difference_degrees_max']):
                duplicate = prior['label']; break
        c['duplicate_of'] = duplicate
        if duplicate is None:
            kept.append(c)
        else:
            duplicates.append({'label': c['label'], 'kept_label': duplicate})
    qualified = [c for c in kept if c['level'] in ('strong', 'weak')]
    ambiguous = len(qualified) > 1 and qualified[0]['score']-qualified[1]['score'] < q['current_winner_score_margin_min']
    winner = qualified[0] if qualified and not ambiguous else None
    strong = winner if winner is not None and winner['level'] == 'strong' and winner['oriented'] else None
    return {'candidates': candidates, 'duplicates': duplicates, 'ambiguous': ambiguous,
            'winner_label': winner['label'] if winner else None, 'strong_label': strong['label'] if strong else None,
            'qualified_labels': [c['label'] for c in qualified]}


class FastFiberObserver:
    def __init__(self, config):
        self.config = config

    def proposals(self, frame, probability):
        cfg = self.config['proposal']; side = cfg['grid_side']; h, w = frame.shape[:2]
        p = cv2.resize(probability[2], (side, side), interpolation=cv2.INTER_AREA)
        rgb = cv2.resize(frame, (side, side), interpolation=cv2.INTER_AREA)
        fov = (rgb.max(2) > 10).astype(np.uint8)
        contours, _ = cv2.findContours(fov, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fov[:] = 0
        if contours:
            cv2.drawContours(fov, [max(contours, key=cv2.contourArea)], -1, 1, -1)
        distance = cv2.distanceTransform(np.pad(fov, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]*min(h, w)/side
        n, labels = cv2.connectedComponents((p >= cfg['seed_probability']).astype(np.uint8), connectivity=cfg['connectivity'])
        counts = np.bincount(labels.ravel(), minlength=n)
        mass = np.bincount(labels.ravel(), weights=p.ravel(), minlength=n)
        eligible = [k for k in range(1, n) if counts[k] >= cfg['minimum_component_grid_pixels']]
        eligible.sort(key=lambda k: (-mass[k], k))
        gy, gx = np.nonzero(labels)
        ids = gy*side+gx; all_points = grid_to_native(np.stack([gx, gy], -1), (h, w), side)
        all_weights = p[gy, gx]
        result = []
        for label in eligible[:cfg['maximum_current_axes']]:
            index = np.flatnonzero(labels[gy, gx] == label)
            if len(index) > cfg['maximum_support_points_per_axis']:
                index = index[np.linspace(0, len(index)-1, cfg['maximum_support_points_per_axis']).astype(int)]
            points, weights = all_points[index], all_weights[index]
            center = np.average(points, axis=0, weights=weights)
            centered = points-center
            _, vectors = np.linalg.eigh((centered*weights[:, None]).T@centered/weights.sum())
            u = vectors[:, -1]
            # Deterministic sign before FOV orientation, including an exactly vertical axis.
            if u[0] < 0 or (u[0] == 0 and u[1] < 0): u = -u
            normal = np.array([-u[1], u[0]])
            a = centered@u; b = centered@normal
            ends = np.percentile(a, [2, 98]); bounds = np.percentile(b, [2, 98]); width = max(float(bounds[1]-bounds[0]), min(h, w)/side)
            endpoints = center+ends[:, None]*u
            grid_ends = (endpoints+.5)*side/np.array([w, h])-.5
            ds, _ = sample(distance, grid_ends)
            oriented = bool(abs(ds[0]-ds[1]) >= cfg['orientation_distance_difference_diagonal_fraction']*math.hypot(h, w))
            if ds[0] > ds[1]:
                u = -u; normal = -normal; ends = np.array([-ends[1], -ends[0]])
            along = (all_points-center)@u; across = (all_points-center)@normal
            halfwidth = cfg['corridor_halfwidth_of_initial_component_width']*width + cfg['corridor_additional_halfwidth_at400']*min(h, w)/400
            extension = cfg['axial_extension_diagonal_fraction']*math.hypot(h, w)
            support = (np.abs(across) <= halfwidth) & (along >= ends[0]-extension) & (along <= ends[1]+extension)
            # Preserve all support IDs for duplicate testing; cap only points used in fitting.
            selected = np.flatnonzero(support); full_ids = ids[selected].tolist()
            if len(selected) > cfg['maximum_support_points_per_axis']:
                selected = selected[np.linspace(0, len(selected)-1, cfg['maximum_support_points_per_axis']).astype(int)]
            result.append({'label': int(label), 'mass': float(mass[label]), 'origin': center, 'u': u, 'normal': normal,
                           'ends': ends, 'width': width, 'oriented': oriented, 'fov_end_distances': ds.tolist(),
                           'support': all_points[selected], '_support_ids': full_ids, 'provenance': 'independent_current'})
        return result, {'components': n-1, 'eligible_components': len(eligible), 'axes_proposed': len(result), 'grid_shape': [side, side]}

    def measure(self, frame, probability, proposal):
        cfg = self.config['observation']; q = self.config['quality']; h, w = frame.shape[:2]; scale = min(h, w)/400
        p = proposal; origin, u, normal = p['origin'], p['u'], p['normal']; start, end = p['ends']; length = float(end-start)
        ids = p.get('_support_ids', [])
        result = {'label': p['label'], 'provenance': p['provenance'], 'score': 0., 'level': 'rejected',
                  'reason': 'insufficient_edge_pairs', 'axis_theta': float(math.atan2(u[1], u[0])), 'oriented': p['oriented'],
                  'front_observed': False, 'shaft_observed': False, 'pose': None, '_support_ids': ids,
                  'support_count': len(ids), 'support_ids_sha256': hashlib.sha256(np.asarray(ids, np.int32).tobytes()).hexdigest(),
                  'seed_length_px': length, 'seed_mass': p.get('mass'), 'fov_end_distances': p.get('fov_end_distances'),
                  'qN': None, 'qE': 0., 'qR': None, 'edge_pairs': 0}
        stations = np.linspace(*cfg['shaft_axial_fraction'], cfg['shaft_stations'])*length+start
        support = p.get('support', np.empty((0, 2)))
        a = (support-origin)@u; b = (support-origin)@normal
        boundary = np.zeros((len(stations), 2)); evidence = np.zeros(len(stations), bool); station_counts = []
        for i, t in enumerate(stations):
            active = np.abs(a-t) <= length/8; station_counts.append(int(active.sum()))
            if p['provenance'] == 'independent_current':
                if active.sum() < 4: continue
                boundary[i] = np.percentile(b[active], [2, 98])
            else:
                fraction = (t-start)/max(length, 1e-6)
                width = p['width']+(p.get('distal_width', p['width'])-p['width'])*fraction
                boundary[i] = [-width/2, width/2]
            evidence[i] = True
        widths = boundary[:, 1]-boundary[:, 0]
        half = np.clip(cfg['search_halfwidth_fraction_of_local_width']*widths, cfg['search_halfwidth_min_at400']*scale, cfg['search_halfwidth_max_at400']*scale)
        offsets = np.linspace(-1, 1, cfg['transverse_samples_per_side'])
        cross = boundary[:, :, None]+half[:, None, None]*offsets
        xy = origin+stations[:, None, None, None]*u+cross[..., None]*normal
        delta = cfg['rgb_difference_offset_at400']*scale
        inside, valid1 = sample(frame, xy-delta*normal); outside, valid2 = sample(frame, xy+delta*normal)
        contrast = np.linalg.norm(inside-outside, axis=-1)/(255*math.sqrt(3))
        valid = valid1 & valid2 & evidence[:, None, None]
        # np.argmax picks first tied coordinate; explicitly tie-break distance then coordinate.
        order = sorted(range(len(offsets)), key=lambda k: (abs(offsets[k]), offsets[k]))
        ordered = np.where(valid, contrast, -1)[..., order]
        best = np.asarray(order)[ordered.argmax(-1)]
        best_contrast = np.take_along_axis(contrast, best[..., None], axis=-1)[..., 0]
        best_valid = np.take_along_axis(valid, best[..., None], axis=-1)[..., 0]
        chosen = np.take_along_axis(cross, best[..., None], axis=-1)[..., 0]
        paired = (best_valid & (best_contrast >= cfg['rgb_edge_contrast_min'])).all(1) & (chosen[:, 1] > chosen[:, 0])
        count = int(paired.sum()); result.update(edge_pairs=count, qE=count/len(stations), station_support_counts=station_counts,
                                                valid_edge_samples=int(valid.sum()), selected_edge_contrasts=best_contrast.tolist(),
                                                selected_edge_pairs=paired.tolist())
        if count < cfg['minimum_supported_edge_pairs']: return result
        coeffs, residuals = zip(*(robust_line(stations[paired], chosen[paired, k], cfg['robust_line_iterations'], cfg['huber_mad_multiplier']) for k in range(2)))
        coeff = np.asarray(coeffs); rms = float(np.sqrt(np.mean(np.square(residuals))))
        fitted = stations[:, None]*coeff[:, 0]+coeff[:, 1]
        fw = fitted[:, 1]-fitted[:, 0]; median_width = float(np.median(fw))
        limit = max(cfg['line_residual_max_at400']*scale, cfg['line_residual_max_fraction_width']*median_width)
        result.update(line_coefficients=coeff.tolist(), line_rms_px=rms, residual_limit_px=limit, qR=float(np.exp(-rms/max(limit, 1e-9))))
        centerline = fitted.mean(1)
        fractions = np.array(cfg['interior_probability_normal_fractions'])
        interior = origin+stations[:, None, None]*u+(centerline[:, None]+fw[:, None]*fractions)[..., None]*normal
        probs, ingrid = sample(probability.transpose(1, 2, 0), interior)
        triplets = ingrid.all(1)
        result['interior_valid_triplets'] = int(triplets.sum()); result['interior_probability_samples'] = probs[..., 2].tolist()
        result['interior_valid_mask'] = ingrid.tolist()
        if triplets.sum() < cfg['minimum_supported_edge_pairs']:
            result['reason'] = 'insufficient_inbounds_interior'; return result
        neural = float(probs[triplets, :, 2].mean()); winning = float((probs[triplets].argmax(-1) == 2).mean())
        result.update(qN=neural, fiber_argmax_fraction=winning)
        mid = coeff.mean(0); corrected_u = u+mid[0]*normal; corrected_u /= np.linalg.norm(corrected_u)
        entry = origin+start*u+(mid[0]*start+mid[1])*normal
        far = origin+end*u+(mid[0]*end+mid[1])*normal
        observed_length = float(np.linalg.norm(far-entry))
        w0 = float((coeff[1]-coeff[0])@[start, 1]); wf = float((coeff[1]-coeff[0])@[end, 1])
        ratio = wf/max(w0, 1e-9)
        result.update(axis_theta=float(math.atan2(corrected_u[1], corrected_u[0])), entry_xy=entry.tolist(), far_seed_xy=far.tolist(),
                      width0_px=w0, far_seed_width_px=wf, observed_shaft_length_px=observed_length)
        result['score'] = float(np.dot(q['score_weights_neural_edges_residual'], [neural, result['qE'], result['qR']]))
        if length < cfg['minimum_visible_length_fraction_min_shape']*min(h, w): result['reason'] = 'shaft_too_short'; return result
        if np.any(fw <= 0) or min(w0, wf) <= 0 or not cfg['distal_proximal_width_ratio'][0] <= ratio <= cfg['distal_proximal_width_ratio'][1]: result['reason'] = 'width_profile'; return result
        if rms > limit: result['reason'] = 'line_residual'; return result
        if neural < q['weak_mean_fiber_probability_min']: result['reason'] = 'weak_neural_evidence'; return result
        result.update(shaft_observed=True, reason='qualified_shaft', level='weak')
        if neural >= q['strong_mean_fiber_probability_min'] and winning >= q['strong_interior_fiber_argmax_fraction_min'] and p['provenance'] == 'independent_current': result['level'] = 'strong'
        result['shaft_lines'] = [[(origin+start*u+(c[0]*start+c[1])*normal).tolist(), (origin+end*u+(c[0]*end+c[1])*normal).tolist()] for c in coeff]
        # A shaft without a directed front still carries a partial group observation.
        result['pose'] = [*entry.tolist(), result['axis_theta'], None, w0, None]
        if not p['oriented'] or not p.get('front_search_available', True):
            result['front_reason'] = 'orientation_ambiguous' if not p['oriented'] else 'predicted_front_unavailable'; return result
        fn = np.array([-corrected_u[1], corrected_u[0]])
        axial = np.linspace(-cfg['front_axial_half_range_at400']*scale, cfg['front_axial_half_range_at400']*scale, cfg['front_samples_per_profile'])
        positions = far+np.array(cfg['front_profile_normal_fractions_width'])[:, None, None]*wf*fn+axial[None, :, None]*corrected_u
        before, v1 = sample(frame, positions-delta*corrected_u); after, v2 = sample(frame, positions+delta*corrected_u)
        pin, v3 = sample(probability[2], positions-delta*corrected_u); pout, v4 = sample(probability[2], positions+delta*corrected_u)
        color = np.linalg.norm(before-after, axis=-1)/(255*math.sqrt(3)); drop = np.maximum(pin-pout, 0)
        supported = v1 & v2 & v3 & v4 & ((color >= cfg['rgb_edge_contrast_min']) | (drop >= cfg['front_probability_drop_min']))
        scores = np.where(supported, color+cfg['front_probability_drop_score_weight']*drop, -1)
        idx = scores.argmax(1); exists = supported[np.arange(3), idx]
        found = axial[idx]; result.update(front_profile_support=exists.tolist(), front_profile_offsets_px=found.tolist(), front_profile_scores=scores[np.arange(3), idx].tolist())
        if not exists[1] or exists.sum() < cfg['front_support_minimum']:
            result['front_reason'] = 'insufficient_current_front_support'; return result
        if float(np.ptp(found[exists])) > cfg['front_transition_spread_max_widths']*wf:
            result['front_reason'] = 'front_transition_spread'; return result
        front_length = observed_length+float(found[1]); endpoint_width = float((coeff[1]-coeff[0])@[start+front_length/np.linalg.norm(u+mid[0]*normal), 1])
        if front_length <= 0 or endpoint_width <= 0:
            result['front_reason'] = 'invalid_front_geometry'; return result
        result['pose'][3] = front_length; result['pose'][5] = endpoint_width
        result.update(front_observed=True, front_reason='current_visible_transition_unverified_tip', visible_front_candidate=(entry+front_length*corrected_u).tolist())
        return result

    def current(self, frame, probability):
        start = time.perf_counter(); proposals, info = self.proposals(frame, probability); proposed = time.perf_counter()
        candidates = [self.measure(frame, probability, p) for p in proposals]; measured = time.perf_counter()
        result = select_current(candidates, self.config); ended = time.perf_counter()
        result.update(proposal_info=info, timings={'grid_fov_proposals_ms': (proposed-start)*1000, 'current_sampling_fit_ms': (measured-proposed)*1000, 'quality_select_ms': (ended-measured)*1000})
        return result

    def history(self, frame, probability, prior):
        if prior is None or prior.get('shaft') is None: return None
        shaft, endpoint = prior['shaft'], prior.get('endpoint'); ex, ey, theta, width = shaft
        if endpoint is None:
            # Last observed shaft extent is search support, never an asserted tip.
            if prior.get('shaft_extent') is None: return None
            length = float(prior['shaft_extent'])
            distal_width = width
        else:
            length, distal_width = endpoint
        u = np.array([np.cos(theta), np.sin(theta)])
        p = {'label': -1, 'origin': np.array([ex, ey]), 'u': u, 'normal': np.array([-u[1], u[0]]),
             'ends': np.array([0., length]), 'width': width, 'distal_width': distal_width, 'oriented': True,
             'front_search_available': endpoint is not None, 'provenance': 'history_conditioned', '_support_ids': []}
        return self.measure(frame, probability, p)


def public_observation(value):
    """Remove temporary support IDs/timing only where explicitly requested by caller."""
    if isinstance(value, dict): return {k: public_observation(v) for k, v in value.items() if not k.startswith('_')}
    if isinstance(value, list): return [public_observation(v) for v in value]
    return value
