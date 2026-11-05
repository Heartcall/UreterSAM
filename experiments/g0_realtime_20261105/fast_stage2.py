"""Defer expensive S2 cap raster evaluation until earlier gates have passed.

The original enumerates 16 cap candidates and rasterizes/validates all of them,
including candidates already rejected by RMS, tangent or RGB gates. These
operations cannot change acceptance of an already rejected candidate. This
specialized runtime keeps accepted edits and candidate ordering, without
producing every rejected candidate's display-only diagnostic.
"""
from __future__ import annotations

import math
import importlib.util

import numpy as np

from tools.fiber_weak_stage2 import connect
from tools.fiber_reference_workspace import ROOT
from tools.fiber_blue_outline import profiles, boundary_distance
from tools.fiber_fast_observation import sample
from tools.fiber_blue_outline import robust_model
from tools.fiber_blue_outline_repaired import RepairedOutline
from tools.fiber_connected_blue_growth import _pca
from tools.fiber_connected_blue_union import ConnectedBlue
from .fast_evidence import fast_evidence as evidence, fast_color


_spec = importlib.util.spec_from_file_location(
    'tools._s2_realtime_20261105_private', ROOT/'tools/fiber_weak_stage2.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
_module.evidence = evidence
GrowthPipeline = _module.GrowthPipeline


class DirectionOnlyOutline(RepairedOutline):
    """Keep exactly the side models/context consumed by S2's direction().

    The original also creates RGB evidence, fitted side masks and retention
    diagnostics here. S2 discards those four return values and only uses info.
    """
    def run(self, frame, raw, probability, body, core, meta, arm):
        assert arm == 'contour_sides'
        scale = min(raw.shape)/400
        yy, xx = np.nonzero(body); points = np.stack([xx, yy], -1).astype(float)
        origin, axis, _, _ = _pca(points)
        if np.dot(axis, np.asarray(meta['seed_axis'])) < 0:
            axis = -axis
        normal = np.array([-axis[1], axis[0]])
        profile, tmin, tmax = profiles(body, origin, axis, self.cfg['profile_step_at400']*scale,
                                       self.cfg['side_frame_margin_at400']*scale)
        length = tmax-tmin
        region = profile[(profile[:, 0] >= tmin+.65*length) & (profile[:, 0] <= tmin+.80*length)]
        width = (float(np.median(region[:, 2]-region[:, 1]+1)) if len(region)
                 else float(np.median(profile[:, 2]-profile[:, 1]+1)))
        caplength = min(max(self.cfg['cap_length_width_fraction']*width,
                            self.cfg['cap_length_min_at400']*scale),
                        self.cfg['cap_length_max_length_fraction']*length)
        target = tmax-caplength
        joint = profile[(profile[:, 0] <= target) & (profile[:, 3] > 0) & (profile[:, 4] > 0)]
        join = float(joint[-1, 0]) if len(joint) else target
        context = dict(origin=origin, axis=axis, profile=profile, min_t=tmin, max_t=tmax,
                       join_t=join, scale=scale, visible_length=length, cap_length=tmax-join)
        models = []
        for side in range(2):
            rows = profile[(profile[:, 0] <= join) & (profile[:, 3+side] > 0)]
            if len(rows) < self.cfg['minimum_side_points'] or np.ptp(rows[:, 0]) < self.cfg['minimum_side_span_fraction']*length:
                models.append(dict(valid=False, mode='unchanged', reason='insufficient_visible_side'))
                continue
            xy = origin+rows[:, 0, None]*axis+rows[:, 1+side, None]*normal
            a, av = sample(frame, xy-scale*normal); b, bv = sample(frame, xy+scale*normal)
            contrast = np.where(av & bv, np.linalg.norm(a-b, axis=-1)/(255*math.sqrt(3)), 0)
            weights = self.cfg['minimum_edge_weight']+np.clip(contrast/self.cfg['edge_weight_scale'], 0, 1)
            models.append(robust_model(rows[:, 0], rows[:, 1+side], weights, self.cfg, scale))
        return None, None, None, None, dict(eligible=True, context=context, side_models=models,
                                          reason='direction_only_runtime')


class FastStage2(GrowthPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.side = DirectionOnlyOutline(self.side.config)
        node = self.inner
        while hasattr(node, 'inner'):
            node = node.inner
        # ConnectedBlue.end only supplies diagnostic tips; the RGB growth and
        # S2's own terminal/cap checks are retained unchanged.
        if hasattr(node, 'growth') and isinstance(node.growth, ConnectedBlue):
            node.growth.end = lambda *args: dict(state='unknown', reason='runtime_diagnostic_omitted', xy=None)
            observer = node.growth.observer
            observer.color = lambda bgr, seed, weak=False: fast_color(observer, bgr, seed, weak)
        else:
            raise RuntimeError('Unexpected upstream S2 graph')

    def fit_cap(self, frame, raw, p, B, core, meta, E, o, u, xy, seed):
        scale = min(raw.shape) / 400
        cfg = self.q['cap']
        normal = np.array([-u[1], u[0]])
        by, bx = np.nonzero(B)
        end = float(((np.c_[bx, by] - o) @ u).max())
        profile, _, _ = profiles(B, o, u, scale, 1.5 * scale)
        candidates = []
        # Full-frame evidence and grids are needed only for geometrically
        # plausible candidates. Construct lazily and reuse for all candidates.
        shared = None

        def prepare_shared():
            yy, xx = np.indices(raw.shape)
            tt = (xx-o[0])*u[0] + (yy-o[1])*u[1]
            vv = (xx-o[0])*normal[0] + (yy-o[1])*normal[1]
            bd = boundary_distance(B)
            ev = evidence(frame, seed, self.q['evidence'], self.gradient)
            fringe = (ev[3] & (ev[4] >= self.q['evidence']['minimum_gradient'])
                      & (bd <= cfg['maximum_boundary_distance_at400'] * scale)
                      & (raw != 1))
            cap_evidence = connect(E | fringe | (B > 0), B > 0) > 0
            self.layers['cap_boundary_evidence'] = cap_evidence.astype(np.uint8)
            return tt, vv, bd, cap_evidence

        for retreat in cfg['retreat_at400']:
            join = end - retreat * scale
            near = profile[(profile[:, 0] >= join-10*scale)
                           & (profile[:, 0] <= join)
                           & (profile[:, 3] > 0) & (profile[:, 4] > 0)]
            tail = profile[profile[:, 0] >= join]
            if len(near) < 4 or len(tail) < 3:
                continue
            slopes, sides = [], []
            for k in (1, 2):
                A = np.c_[near[:, 0]-join, np.ones(len(near))]
                beta = np.linalg.lstsq(A, near[:, k], rcond=None)[0]
                for _ in range(12):
                    residual = near[:, k] - A @ beta
                    weight = np.minimum(1, 1.5*scale / np.maximum(abs(residual), 1e-9))
                    beta = np.linalg.lstsq(A*weight[:, None]**.5,
                                           near[:, k]*weight**.5, rcond=None)[0]
                slopes.append(beta[0]); sides.append(beta[1])
            half = (sides[1]-sides[0])/2
            center = sum(sides)/2
            length = end-join
            if half <= 0:
                continue
            slope = (float((np.asarray(xy)-o) @ normal)-center)/length
            for r in cfg['r_candidates']:
                a = length/(1-r)
                ct = join-a*r
                b = half/math.sqrt(1-r*r)
                wp = -half*r/(length*(1+r))
                jumps = np.degrees(abs(np.arctan(slopes)-np.arctan([slope-wp, slope+wp])))
                width = b*np.sqrt(np.clip(1-((tail[:, 0]-ct)/a)**2, 0, 1))
                cy = center+slope*(tail[:, 0]-join)
                residual = np.r_[tail[:, 1]-(cy-width), tail[:, 2]-(cy+width)]
                rms = float(np.sqrt(np.mean(residual**2)))
                row = dict(accepted=False, reason='ellipse_fit_residual', RMS=rms,
                           score=float(np.median(abs(residual))), r=r, join=join,
                           a=a, b=b, center_t=ct, center_n=center, slope=slope,
                           candidate_xy=xy, tangent_jumps=jumps, addition=None,
                           tip_t=end)
                candidates.append(row)
                if rms > cfg['maximum_RMS_at400']*scale:
                    continue
                row['reason'] = 'model_join_tangent_mismatch'
                if max(jumps) > cfg['maximum_model_join_degrees']:
                    continue
                row['reason'] = 'requires_RGB_evaluation'

        # The original chooses the minimum-ranked accepted candidate. Evaluating
        # in that exact stable rank order permits stopping at the first accepted
        # one, without computing lower-ranked candidates' diagnostic rasters.
        ranked = sorted(candidates, key=lambda row: (row['score'], row['RMS'], row['r']))
        selected = None
        for row in ranked:
                if row['reason'] != 'requires_RGB_evaluation':
                    continue
                a, b, ct, center, slope, join, r = [row[k] for k in
                    ('a', 'b', 'center_t', 'center_n', 'slope', 'join', 'r')]
                if shared is None:
                    shared = prepare_shared()
                tt, vv, bd, cap_evidence = shared
                theta = np.linspace(-math.acos(r), math.acos(r), cfg['samples'])
                t = ct+a*np.cos(theta)
                v = center+slope*(t-join)+b*np.sin(theta)
                curve = o+t[:, None]*u+v[:, None]*normal
                relative = v-center-slope*(t-join)
                normals = (((t-ct)/a**2-slope*relative/b**2)[:, None]*u
                           + (relative/b**2)[:, None]*normal)
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
                inside, iv = sample(frame, curve-1.5*scale*normals)
                outside, ov = sample(frame, curve+1.5*scale*normals)
                distance, dv = sample(bd, curve)
                ag, agv = sample(self.gradient, curve-1.5*scale*normals)
                bg, bgv = sample(self.gradient, curve+1.5*scale*normals)
                ai = evidence(inside[:, None, :], seed, self.q['evidence'], ag[:, None])
                bo = evidence(outside[:, None, :], seed, self.q['evidence'], bg[:, None])
                contrast = np.linalg.norm(inside-outside, axis=-1)/(255*math.sqrt(3))
                supported = (iv & ov & dv & agv & bgv & ((ai[1] | ai[2]).ravel())
                             & ((~(bo[1] | bo[2]).ravel())
                                | (contrast >= self.q['tip']['minimum_RGB_contrast']))
                             & (distance <= cfg['maximum_boundary_distance_at400']*scale))
                row.update(curve=curve, RGB_supported_fraction=float(supported.mean()),
                           reason='insufficient_current_RGB_arc')
                if supported.mean() < cfg['minimum_RGB_support']:
                    continue
                # Pointwise arithmetic is unchanged, but irrelevant pixels
                # outside the axial/evidence/background domain need no sqrt.
                relevant = (tt >= join) & (tt <= end) & cap_evidence & (self.final_mask == 0)
                addition = np.zeros_like(relevant)
                rt, rv = tt[relevant], vv[relevant]
                addition[relevant] = (abs(rv-center-slope*(rt-join))
                                     <= b*np.sqrt(np.clip(1-((rt-ct)/a)**2, 0, 1)))
                result = ((B > 0) | addition).astype(np.uint8)
                actual, av = sample(boundary_distance(result), curve)
                match = float((av & (actual <= cfg['maximum_boundary_distance_at400']*scale)).mean())
                row.update(addition=addition, actual_arc_match_fraction=match,
                           reason='protected_raster_not_following_arc')
                if match < cfg['minimum_actual_arc_match']:
                    continue
                row.update(accepted=True, reason='accepted_current_RGB_cap')
                selected = row
                break
        if candidates:
            best = selected or ranked[0]
            addition = best.pop('addition')
            best.update(candidate_count=len(candidates),
                        feasible_count=None, diagnostic_policy='stop_at_first_ranked_feasible')
            self.outline['cap'] = best
            if best['accepted']:
                self.final_mask[addition] = 2
                self.layers['cap_fill'] = addition.astype(np.uint8)
                self.layers['processed_body'] |= addition.astype(np.uint8)
                self.outline['cap_coordinate_context'] = dict(
                    origin=o, axis=u, join_t=best['join'], scale=scale,
                    min_t=float(shared[0][B > 0].min()), max_t=end)
                self.outline['reason'] = 'recovered_body_with_observed_cap'
        else:
            self.outline['cap'].update(attempted=True, reason='insufficient_recovered_side_join')
