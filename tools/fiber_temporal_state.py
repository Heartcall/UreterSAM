"""Causal geometry metadata only. This module never accepts or creates masks."""
from __future__ import annotations
import math
import numpy as np
from .fiber_shape_prior import CausalShapeTracker, tip_of, wrap


def valid_geometry(state):
    if state is None: return False
    a = np.asarray(state, dtype=float)
    return bool(a.shape == (7,) and np.isfinite(a).all() and np.all(a[3:] > 0) and a[3] - a[6]*a[5]/2 > 0)


def geometry_features(state):
    if not valid_geometry(state): return None
    e = np.asarray(state[:2], dtype=float); angle, length, w0, w1, ratio = state[2:]
    u = np.array([np.cos(angle), np.sin(angle)]); v = np.array([-u[1], u[0]])
    lc = length - ratio*w1/2; half = math.atan2(w0-w1, 2*lc)
    return {'tip_xy': tip_of(state).tolist(), 'side_angles_rad': [wrap(angle-half), wrap(angle+half)],
        'signed_taper_angle_rad': 2*half, 'cap_center_xy': (e+u*lc).tolist(),
        'side_segments': [[(e+v*w0/2).tolist(), (e+u*lc+v*w1/2).tolist()],
                          [(e-v*w0/2).tolist(), (e+u*lc-v*w1/2).tolist()]],
        'role': 'uncalibrated_image_geometry_not_physical_tip_truth'}


def innovation(measured, predicted, shape, limits):
    residual = np.asarray(measured, float)-np.asarray(predicted, float); residual[2] = wrap(residual[2])
    d = math.hypot(*shape); ratios = np.asarray(measured)[4:6]/np.asarray(predicted)[4:6]
    tip_jump = float(np.linalg.norm(tip_of(measured)-tip_of(predicted)))
    checks = {'tip': tip_jump <= d*limits['tip_jump_diagonal_fraction'],
        'length': abs(residual[3]) <= d*limits['length_jump_diagonal_fraction'],
        'angle': abs(residual[2]) <= np.deg2rad(limits['angle_jump_degrees']),
        'width': bool(np.all(ratios >= limits['width_ratio'][0]) and np.all(ratios <= limits['width_ratio'][1]))}
    checks = {key: bool(value) for key, value in checks.items()}
    return {'passed': all(checks.values()), 'checks': checks, 'residual': residual.tolist(),
            'tip_jump_px': tip_jump, 'width_ratios': ratios.tolist()}


class GatedGeometryTracker:
    def __init__(self, settings, limits):
        self.settings = settings; self.limits = limits; self.reset()

    def reset(self):
        self.state = None; self.velocity = None; self.age = 0; self.pending = []; self.clock = 0.

    def _result(self, status, used=False, diagnostic=None):
        state = self.state.tolist() if valid_geometry(self.state) else None
        return {'state': state, 'status': status, 'age': self.age, 'pending_count': len(self.pending),
            'pending_state': self.pending[-1][0].tolist() if self.pending else None,
            'uses_current_observation': bool(used), 'prediction_only': state is not None and not used,
            'innovation': diagnostic, 'apply_to_mask': False}

    def _predict_only(self, predicted, status, diagnostic=None):
        if self.state is not None:
            self.age += 1
            if self.age > self.settings['maximum_prediction_only_frames']:
                self.state = None; self.velocity = None
                status = 'expired_reset_with_pending' if self.pending else 'expired_reset'
            else: self.state = predicted.copy()
        return self._result(status, diagnostic=diagnostic)

    def step(self, measurement, shape, dt):
        if not np.isfinite(dt) or dt <= 0: raise ValueError('Positive finite video dt required')
        self.clock += dt
        measured = np.asarray(measurement, float).copy() if valid_geometry(measurement) else None
        predicted = None if self.state is None else self.state+self.velocity*dt
        if predicted is not None: predicted[2] = wrap(predicted[2])
        if predicted is not None and not valid_geometry(predicted):
            self.state = None; self.velocity = None; self.age = 0; self.pending = []
            return self._result('invalid_prediction_reset')
        if measured is None:
            self.pending = []
            return self._predict_only(predicted, 'predicted_missing' if self.state is not None else 'no_state')
        measured[2] = wrap(measured[2])
        if self.state is None and not self.pending:
            self.state = measured; self.velocity = np.zeros(7); self.age = 0
            return self._result('initialized', used=True)
        active_innovation = None if predicted is None else innovation(measured, predicted, shape, self.limits)
        if active_innovation is not None and active_innovation['passed']:
            residual = np.asarray(active_innovation['residual'])
            self.state = predicted+self.settings['alpha']*residual; self.state[2] = wrap(self.state[2])
            self.velocity += self.settings['beta']*residual/dt; self.age = 0; self.pending = []
            if not valid_geometry(self.state):
                self.state = None; self.velocity = None
                return self._result('invalid_update_reset', diagnostic=active_innovation)
            return self._result('updated', used=True, diagnostic=active_innovation)
        pending_checks = None
        if self.pending:
            last, ts = self.pending[-1]; pending_predicted = last.copy()
            if len(self.pending) >= 2:
                previous, prev_ts = self.pending[-2]; delta = last-previous; delta[2] = wrap(delta[2])
                pending_predicted += delta/(ts-prev_ts)*(self.clock-ts); pending_predicted[2] = wrap(pending_predicted[2])
            pair = innovation(measured, last, shape, self.limits)
            extrapolated = innovation(measured, pending_predicted, shape, self.limits) if valid_geometry(pending_predicted) else None
            consistent = pair['passed'] and extrapolated is not None and extrapolated['passed']
            pending_checks = {'pairwise': pair, 'extrapolated': extrapolated, 'passed': bool(consistent)}
            if not consistent: self.pending = []
        self.pending.append((measured, self.clock))
        diagnostic = {'active': active_innovation, 'pending': pending_checks}
        if len(self.pending) >= self.settings['confirmation_observations']:
            self.state = measured; self.velocity = np.zeros(7); self.age = 0; self.pending = []
            return self._result('confirmed_relocation', used=True, diagnostic=diagnostic)
        return self._predict_only(predicted, 'rejected_jump_pending', diagnostic)


class GeometryStateControl:
    """Context isolation shared across controls; no future data or image state inputs."""
    def __init__(self, arm, config, legacy, nominal_dt=1/30):
        if arm not in config['arms']: raise ValueError('Unknown frozen arm')
        self.arm = arm; self.config = config; self.nominal_dt = nominal_dt
        self.tracker = GatedGeometryTracker(config['state'], config['innovation_limits']) if arm == 'gated_state_only' else CausalShapeTracker(legacy['tracking']) if arm == 'legacy_state_only' else None
        self.reset()

    def reset(self):
        if self.tracker is not None: self.tracker.reset()
        self.source = None; self.shape = None; self.timestamp = None

    def step(self, measurement, shape, timestamp, source):
        if not np.isfinite(timestamp): raise ValueError('Finite video timestamp required')
        shape = tuple(shape); reason = None; dt = self.nominal_dt
        if self.source is not None:
            if source != self.source: reason = 'source_changed'
            elif shape != self.shape: reason = 'shape_changed'
            elif timestamp <= self.timestamp: reason = 'nonmonotonic_timestamp'
            elif timestamp-self.timestamp > self.config['state']['timestamp_gap_reset_seconds']: reason = 'timestamp_gap'
            else: dt = timestamp-self.timestamp
        if reason and self.tracker is not None: self.tracker.reset()
        self.source = source; self.shape = shape; self.timestamp = float(timestamp)
        measurement_valid = valid_geometry(measurement)
        measured = np.asarray(measurement, float).copy() if measurement_valid else None
        if self.arm == 'gated_state_only': result = self.tracker.step(measured, shape, dt)
        elif self.arm == 'legacy_state_only':
            old = self.tracker.step(measured, shape, dt); valid = valid_geometry(old['state'])
            result = {'state': old['state'] if valid else None, 'status': old['status'] if valid or old['state'] is None else 'legacy_invalid_geometry',
                'age': old['age'], 'pending_count': 0, 'pending_state': None,
                'uses_current_observation': bool(old['apply'] and valid), 'prediction_only': bool(valid and not old['apply']), 'innovation': None, 'apply_to_mask': False}
        else:
            state = measured.tolist() if measured is not None and self.arm != 'raw2' else None
            result = {'state': state, 'status': 'current_observation' if state is not None else 'no_observation' if self.arm != 'raw2' else 'raw_no_geometry',
                'age': 0, 'pending_count': 0, 'pending_state': None, 'uses_current_observation': state is not None, 'prediction_only': False, 'innovation': None, 'apply_to_mask': False}
        result.update(context_reset_reason=reason, dt_seconds=float(dt), current_measurement_valid=measurement_valid,
            geometry=geometry_features(result['state']), tip_metrics=None)
        return result
