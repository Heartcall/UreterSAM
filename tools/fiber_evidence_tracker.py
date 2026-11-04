"""T1 causal partial geometry state. No mask argument and no rasterization."""
from __future__ import annotations
import math
import numpy as np
from .fiber_fast_observation import wrap


def valid_group(group, value):
    if value is None: return False
    z = np.asarray(value, float)
    return bool(z.shape == ((4,) if group == 'shaft' else (2,)) and np.isfinite(z).all() and (z[3] > 0 if group == 'shaft' else np.all(z > 0)))


def candidate_groups(candidate, prior=None):
    if candidate is None or not candidate.get('shaft_observed'): return {}, False
    pose = candidate['pose']; resolved = False
    if candidate['oriented']:
        shaft = np.array([pose[0], pose[1], pose[2], pose[4]], float)
    elif prior is not None and prior.get('shaft') is not None:
        shaft = np.array([pose[0], pose[1], pose[2], pose[4]], float)
        if abs(wrap(shaft[2]-prior['shaft'][2])) > np.pi/2:
            shaft[:2] = candidate['far_seed_xy']; shaft[2] = wrap(shaft[2]+np.pi); shaft[3] = candidate['far_seed_width_px']
        resolved = True
    else:
        return {}, False
    result = {'shaft': shaft}
    if candidate['oriented'] and candidate['front_observed']:
        result['endpoint'] = np.array([pose[3], pose[5]], float)
    return result, resolved


def compatibility(groups, prior, shape, settings):
    """Only compare mutually available fields; absence is explicitly logged."""
    checks = {}; skipped = []; d = math.hypot(*shape)
    if 'shaft' in groups and valid_group('shaft', prior.get('shaft')):
        a, b = groups['shaft'], np.asarray(prior['shaft'])
        checks['entry'] = bool(np.linalg.norm(a[:2]-b[:2]) <= d*settings['weak_entry_tip_jump_diagonal_fraction'])
        checks['angle'] = bool(abs(wrap(a[2]-b[2])) <= np.deg2rad(settings['weak_angle_jump_degrees']))
        checks['width0'] = bool(settings['weak_width_ratio'][0] <= a[3]/b[3] <= settings['weak_width_ratio'][1])
    else: skipped.extend(['entry', 'angle', 'width0'])
    if 'endpoint' in groups and valid_group('endpoint', prior.get('endpoint')):
        a, b = groups['endpoint'], np.asarray(prior['endpoint'])
        checks['length'] = bool(abs(a[0]-b[0]) <= d*settings['weak_length_jump_diagonal_fraction'])
        checks['width1'] = bool(settings['weak_width_ratio'][0] <= a[1]/b[1] <= settings['weak_width_ratio'][1])
    else: skipped.extend(['length', 'width1'])
    if all(g in groups and valid_group(g, prior.get(g)) for g in ('shaft', 'endpoint')):
        a, b = groups['shaft'], np.asarray(prior['shaft'])
        ta = a[:2]+groups['endpoint'][0]*np.array([np.cos(a[2]), np.sin(a[2])])
        tb = b[:2]+prior['endpoint'][0]*np.array([np.cos(b[2]), np.sin(b[2])])
        checks['front'] = bool(np.linalg.norm(ta-tb) <= d*settings['weak_entry_tip_jump_diagonal_fraction'])
    else: skipped.append('front')
    return {'passed': bool(checks) and all(checks.values()), 'checks': checks, 'skipped': skipped}


class EvidenceTracker:
    def __init__(self, config, nominal_dt=1/30):
        self.config = config; self.settings = config['state']; self.nominal_dt = nominal_dt; self.reset()

    def reset(self):
        self.groups = {g: None for g in ('shaft', 'endpoint')}
        self.velocity = {g: None for g in self.groups}; self.age = {g: 0 for g in self.groups}
        self.source = self.shape = self.timestamp = None; self.extent = None

    def prepare(self, shape, timestamp, source):
        if not np.isfinite(timestamp): raise ValueError('Finite source time required')
        shape = tuple(shape); reason = None; dt = self.nominal_dt
        if self.source is not None:
            if self.source != source: reason = 'source_changed'
            elif self.shape != shape: reason = 'shape_changed'
            elif timestamp <= self.timestamp: reason = 'nonmonotonic_timestamp'
            elif timestamp-self.timestamp > self.settings['timestamp_gap_reset_seconds']: reason = 'timestamp_gap'
            else: dt = timestamp-self.timestamp
        if reason: self.reset()
        self.source = source; self.shape = shape; self.timestamp = float(timestamp)
        prior = {'shaft': None, 'endpoint': None, 'shaft_extent': self.extent}; invalid = []
        for group, value in self.groups.items():
            if value is None: continue
            predicted = value+self.velocity[group]*dt
            if group == 'shaft': predicted[2] = wrap(predicted[2])
            if valid_group(group, predicted): prior[group] = predicted.tolist()
            else: invalid.append(group)
        # A detached length is not a usable search pose, but retains its independent diagnostic validity.
        return {'prior': prior, 'invalid_predictions': invalid, 'context_reset_reason': reason, 'dt_seconds': float(dt)}

    def update(self, current, history, prepared):
        prior = prepared['prior']; dt = prepared['dt_seconds']; candidates = current['candidates']
        strong = next((c for c in candidates if c['label'] == current['strong_label']), None)
        selected = strong; independent_strong = strong is not None; diagnostics = []
        groups, resolved = candidate_groups(selected, prior)
        if selected is None:
            pool = [c for c in candidates if c['level'] != 'rejected' and c.get('duplicate_of') is None]
            if history is not None and history['level'] != 'rejected': pool.append(history)
            # Current-only scores order compatible weak candidates; no prior proximity reward.
            pool.sort(key=lambda c: (-c['score'], c['label']))
            for c in pool:
                cg, cr = candidate_groups(c, prior)
                gate = compatibility(cg, prior, self.shape, self.settings)
                diagnostics.append({'label': c['label'], 'provenance': c['provenance'], **gate})
                if gate['passed'] and valid_group('shaft', prior.get('shaft')):
                    selected, groups, resolved = c, cg, cr; break
        gate = compatibility(groups, prior, self.shape, self.settings)
        states = {}; used = []; overrides = []; invalid_updates = []
        for group in self.groups:
            measured = groups.get(group)
            predicted = np.asarray(prior[group], float) if prior[group] is not None else None
            can_update = valid_group(group, measured) and (independent_strong or (selected is not None and gate['passed'] and predicted is not None))
            if can_update:
                # Strong measurements are accepted before prediction rejection, including invalid priors.
                if predicted is None or (independent_strong and not gate['passed']):
                    self.groups[group] = measured.copy(); self.velocity[group] = np.zeros_like(measured)
                    status = 'strong_initialized'
                    if predicted is not None: overrides.append(group); status = 'strong_current_override'
                else:
                    residual = measured-predicted
                    if group == 'shaft': residual[2] = wrap(residual[2])
                    self.groups[group] = predicted+self.settings['alpha']*residual
                    self.velocity[group] = self.velocity[group]+self.settings['beta']*residual/dt
                    if group == 'shaft': self.groups[group][2] = wrap(self.groups[group][2])
                    status = 'strong_updated' if independent_strong else 'weak_updated'
                if not valid_group(group, self.groups[group]):
                    self.groups[group] = self.velocity[group] = None; status = 'invalid_update_reset'; invalid_updates.append(group)
                else:
                    self.age[group] = 0; used.append(group)
            else:
                if predicted is not None:
                    self.age[group] += 1
                    if self.age[group] <= self.settings['maximum_prediction_only_frames_per_group']:
                        self.groups[group] = predicted; status = 'predicted'
                    else:
                        self.groups[group] = self.velocity[group] = None; status = 'expired'
                else:
                    self.groups[group] = self.velocity[group] = None
                    status = 'invalid_prediction_reset' if group in prepared['invalid_predictions'] else 'unavailable'
            states[group] = {'value': self.groups[group].tolist() if self.groups[group] is not None else None,
                             'status': status, 'age': self.age[group], 'observed': group in used,
                             'predicted': self.groups[group] is not None and group not in used}
        if selected is not None and 'shaft' in used:
            self.extent = selected.get('observed_shaft_length_px', self.extent)
        if self.groups['shaft'] is None: self.extent = None
        shaft, endpoint = self.groups['shaft'], self.groups['endpoint']
        front = None
        if shaft is not None and endpoint is not None:
            front = (shaft[:2]+endpoint[0]*np.array([np.cos(shaft[2]), np.sin(shaft[2])])).tolist()
        return {'groups': states, 'visible_front_candidate': front,
                'front_is_prediction': front is not None and ('shaft' not in used or 'endpoint' not in used),
                'selected_label': selected['label'] if selected else None,
                'used_provenance': selected['provenance'] if used else None,
                'orientation_resolved_by_history': bool(resolved and used), 'independent_strong_used': bool(independent_strong and used),
                'strong_override_groups': overrides, 'weak_candidate_gates': diagnostics,
                'selected_gate': gate, 'invalid_updates': invalid_updates, **prepared,
                'apply_to_mask': False, 'tip_metrics': None}

def empty_current():
    return {'candidates': [], 'duplicates': [], 'ambiguous': False, 'winner_label': None, 'strong_label': None, 'qualified_labels': []}


def stateless_result(current):
    c = next((c for c in current['candidates'] if c['label'] == current['winner_label']), None)
    groups, _ = candidate_groups(c)
    return {'groups': {g: {'value': groups[g].tolist() if g in groups else None, 'status': 'current_observation' if g in groups else 'unavailable', 'age': 0,
                            'observed': g in groups, 'predicted': False} for g in ('shaft', 'endpoint')},
            'visible_front_candidate': c.get('visible_front_candidate') if c else None, 'front_is_prediction': False,
            'selected_label': c['label'] if c else None, 'used_provenance': 'independent_current' if groups else None,
            'apply_to_mask': False, 'tip_metrics': None}
