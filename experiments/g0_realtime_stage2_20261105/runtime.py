"""Current-frame reuse, mask-only bookkeeping, and lazy original-order tip search."""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import types
import time

import cv2
import numpy as np

from tools.fiber_reference_workspace import ROOT, read
from tools.fiber_connected_blue_growth import chroma
from tools.fiber_fast_observation import sample
from tools.fiber_local_dark_evidence import nearest, public
from tools.fiber_weak_stage2 import native_gradient, connect, frontier, terminal
from experiments.g0_realtime_20261105.fast_geometry import (
    grouped_observed_sides, cached_raster_geometry)


def private_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grouped_profiles(body, origin, u, step, frame_margin=0):
    yy, xx = np.nonzero(body)
    points = np.stack([xx, yy], -1).astype(float)
    n = np.array([-u[1], u[0]])
    t, v = (points-origin) @ u, (points-origin) @ n
    lo = float(t.min()); bins = np.floor((t-lo)/step).astype(int)
    order = np.argsort(bins, kind='stable')
    cut = np.flatnonzero(np.diff(bins[order]))+1
    rows = []
    for ids in np.split(order, cut):
        z, w = ids[np.argmin(v[ids])], ids[np.argmax(v[ids])]
        a, b = points[z], points[w]
        def valid(p):
            return min(p[0], p[1], body.shape[1]-1-p[0], body.shape[0]-1-p[1]) >= frame_margin
        rows.append([float(np.mean(t[ids])), float(v[z]), float(v[w]), float(valid(a)), float(valid(b))])
    return np.asarray(rows), float(t.min()), float(t.max())


def lazy_tip(frame, raw, g, seed, cfg, scale, blue_identity, counters):
    """First passing point in the exact original lexicographic order wins.

    Qualified-pixel totals and the complete eligibility display layer are not
    generated. No lower-ranked point can replace an already qualified winner.
    """
    o, u, n, V = [g[k] for k in ['origin', 'axis', 'normal', 'intersection']]
    blue, color, _ = blue_identity(frame, seed, cfg)
    yy, xx = np.indices(raw.shape)
    t = (xx-o[0])*u[0]+(yy-o[1])*u[1]
    v = (xx-o[0])*n[0]+(yy-o[1])*n[1]
    lo, old_end = g['body_extent']
    begin = lo+cfg['minimum_search_axial_fraction']*(old_end-lo)
    limit = old_end+cfg['maximum_forward_at400']*scale
    halfwidth = np.maximum(0., (float((V-o) @ u)-t)*math.tan(g['half_angle_radians']))
    domain = ((t >= begin) & (t <= limit)
              & (abs(v) <= halfwidth+cfg['search_side_margin_at400']*scale))
    patch = cv2.filter2D(blue.astype(np.float32), -1, np.ones((3, 3), np.float32),
                        borderType=cv2.BORDER_CONSTANT)
    candidate = domain & blue & (patch >= cfg['minimum_local_blue_pixels_3x3']) & (raw != 1)
    y, x = np.nonzero(candidate); eligible = np.zeros_like(raw, np.uint8)
    meta = dict(reason='no_blue_external_terminal', source_pixel=None, vertex=None,
                qualified_pixels=None, search_limit_t=float(limit), pixel_candidates=len(x),
                network_probability_gate=False, diagnostic_policy='lazy_original_rank_order')
    if not len(x):
        return blue, domain, eligible, meta
    points = np.c_[x, y].astype(float)
    axial, across = (points-o) @ u, abs((points-o) @ n)
    order = np.lexsort((across, -axial))
    counters['tip_candidate_pixels'] += len(x)
    offset, chunk = 0, 32
    while offset < len(order):
        ids = order[offset:offset+chunk]
        pts = points[ids]
        future = pts[:, None, :]+np.asarray(cfg['forward_offsets_at400'])[None, :, None]*scale*u
        bg, valid = sample(frame, future); labels, lv = nearest(raw, future)
        forward_blue, bc, bnoise = blue_identity(bg, seed, cfg)
        med = np.median(bg, axis=1)
        contrast = np.linalg.norm(frame[y[ids], x[ids]].astype(float)-med, axis=1)/(255*math.sqrt(3))
        chromatic = np.linalg.norm(color[y[ids], x[ids]]-np.median(bc, axis=1), axis=1)
        margin = np.minimum.reduce([pts[:, 0], pts[:, 1], raw.shape[1]-1-pts[:, 0], raw.shape[0]-1-pts[:, 1]])
        forward_stone = np.any(labels == 1, axis=1)
        reject_stone = cfg.get('reject_forward_stone', True)
        stone_ok = ~forward_stone if reject_stone else np.ones(len(pts), bool)
        good = (valid.all(1) & lv.all(1) & stone_ok
                & (bnoise < cfg['maximum_chroma_noise']).all(1)
                & (forward_blue.mean(1) <= cfg['maximum_forward_blue_fraction'])
                & ((contrast >= cfg['minimum_RGB_contrast']) | (chromatic >= cfg['minimum_chroma_contrast']))
                & (margin >= cfg['minimum_frame_margin_at400']*scale))
        counters['tip_evaluated_pixels'] += len(ids)
        eligible[y[ids[good]], x[ids[good]]] = 1
        valid_ids = np.flatnonzero(good)
        if len(valid_ids):
            j = int(valid_ids[0]); chosen = int(ids[j]); end = float(axial[chosen]); T = o+end*u
            meta.update(source_pixel=points[chosen], vertex=T, tip_t=end,
                        source_normal_distance=float((points[chosen]-o) @ n),
                        RGB_contrast=float(contrast[j]), chroma_contrast=float(chromatic[j]),
                        forward_blue_fraction=float(forward_blue[j].mean()),
                        reason='farthest_current_blue_external_contrast')
            if not reject_stone:
                meta.update(forward_stone_rejection_enabled=False, selected_forward_stone=bool(forward_stone[j]))
            if end >= limit-cfg['search_limit_margin_at400']*scale:
                meta['reason'] = 'tip_reaches_search_limit'
            return blue, domain, eligible, meta
        offset += len(ids); chunk = min(chunk*4, 4096)
    return blue, domain, eligible, meta


class Runtime:
    def __init__(self):
        self.counters = dict(seed_reuse=0, seed_fallback=0, evidence_reuse=0,
                             chroma_reuse=0, tip_candidate_pixels=0, tip_evaluated_pixels=0)
        self.frame = self.raw = self.probability = None
        self._chroma = self._evidence = self._captured = None
        path = ROOT/'experiments/g0_realtime_20261105'
        colors = private_module('experiments.g0_realtime_20261105._v2_colors', path/'fast_evidence.py')
        colors.chroma = self.chroma
        self.colors = colors
        module = private_module('experiments.g0_realtime_20261105._v2_stage', path/'fast_stage2.py')
        module.evidence = self.evidence
        module._module.evidence = self.evidence
        module.profiles = grouped_profiles
        self.stage = module.FastStage2(
            read(ROOT/'configs/weak_blue_evidence_control_20261103.json'), 'weak_blue_extension_tip',
            read(ROOT/'configs/fiber_temporal_observation_search_20261030.json'), True)
        # Use the same functional expansion with display-only bookkeeping removed.
        self.stage.expand = types.MethodType(lambda stage, *args: self.expand(*args), self.stage)
        node = self.stage.inner
        while hasattr(node, 'inner'):
            node = node.inner
        observer = node.growth.observer
        self.observer = observer
        original_seeds = observer.seeds
        def capture(frame, raw, probability, prior, age):
            result = original_seeds(frame, raw, probability, prior, age)
            self._captured = (frame, raw, probability, result)
            return result
        observer.seeds = capture
        geometry = private_module('tools._g0_realtime_v2_private', ROOT/'tools/fiber_tapered_tangent_cap.py')
        geometry.observed_sides = grouped_observed_sides
        geometry.raster_geometry = cached_raster_geometry
        geometry.locate_tip = lambda *args: lazy_tip(*args, self.colors.fast_blue_identity, self.counters)
        self.g0 = geometry.TaperedTangentCap(read(ROOT/'configs/tapered_tangent_cap_no_fallback_20261104.json'))
        assert self.g0.seed.cfg == observer.cfg and self.g0.seed.full == observer.full
        original_g0_seeds = self.g0.seed.seeds
        def reuse(frame, raw, probability, prior, age):
            if self._captured is not None:
                f, r, p, (seeds, records) = self._captured
                weak_history = any(z.get('mode') == 'weak_history' for z in records)
                if f is frame and r is raw and p is probability and not weak_history:
                    self.counters['seed_reuse'] += 1
                    return seeds, records
            self.counters['seed_fallback'] += 1
            return original_g0_seeds(frame, raw, probability, prior, age)
        self.g0.seed.seeds = reuse

    def chroma(self, image):
        if image is self.frame:
            if self._chroma is None:
                self._chroma = chroma(image)
            else:
                self.counters['chroma_reuse'] += 1
            return self._chroma
        return chroma(image)

    def evidence(self, image, seed, cfg, gradient=None):
        if image is self.frame and self._evidence is not None:
            old_seed, old_cfg, old_gradient, value = self._evidence
            if (old_cfg == cfg and old_gradient is gradient
                    and all(np.array_equal(seed[k], old_seed[k]) for k in ('mu', 'bg', 'v', 'gap'))):
                self.counters['evidence_reuse'] += 1
                return value
        value = self.colors.fast_evidence(image, seed, cfg, gradient)
        if image is self.frame:
            self._evidence = (seed, cfg, gradient, value)
        return value

    def reset(self):
        self.stage.reset(); self.g0.reset()
        self.frame = self.raw = self.probability = None
        self._chroma = self._evidence = self._captured = None
        for key in self.counters:
            self.counters[key] = 0

    def run(self, frame, raw, probability, timestamp, video):
        self.frame, self.raw, self.probability = frame, raw, probability
        self._chroma = self._evidence = self._captured = None
        start = time.perf_counter()
        self.stage.run(frame, raw, probability, timestamp, video)
        mid = time.perf_counter()
        self.g0.run(frame, raw, probability, self.stage.final_mask)
        end = time.perf_counter()
        self.timings = dict(s2_ms=(mid-start)*1000, g0_ms=(end-mid)*1000)
        return self.g0.final_mask

    def expand(self, frame, raw, probability, meta, B, core, u):
        stage = self.stage
        q = next(z for z in meta['seeds'] if z['reason'] == 'qualified')
        mu = np.asarray(q['mu'], np.float32)
        seed = dict(mu=mu, v=mu-1/3, bg=np.asarray(q['background_chroma'], np.float32), gap=q['relative_blue_advantage'])
        stage.gradient = native_gradient(frame)
        score, strong, weak, uncertain, edge, noise = self.evidence(frame, seed, stage.q['evidence'], stage.gradient)
        scale = min(raw.shape)/400; cfg = stage.q['search']
        origin = np.asarray(stage.outline['context']['origin']); normal = np.array([-u[1], u[0]])
        yy, xx = np.indices(raw.shape)
        t = (xx-origin[0])*u[0]+(yy-origin[1])*u[1]
        v = (xx-origin[0])*normal[0]+(yy-origin[1])*normal[1]
        bt, bv = t[B > 0], v[B > 0]; front = float(bt.max())
        limit = min(cfg['maximum_forward_at400']*scale, cfg['maximum_forward_diagonal_fraction']*np.hypot(*raw.shape))
        lateral = (np.quantile(bv, .98)-np.quantile(bv, .02))*cfg['lateral_width_multiplier']/2+cfg['lateral_margin_at400']*scale
        center = float(np.median(bv))
        domain = (t >= bt.min()-2*scale) & (t <= front+limit) & (abs(v-center) <= lateral)
        eligible = (strong | weak) & domain & (raw != 1)
        reachable = connect(eligible | (B > 0), (B > 0) & core)
        new = (reachable > 0) & (B == 0) & (raw == 0)
        blocked = (int(reachable.sum()) > cfg['maximum_component_fraction']*raw.size
                   or int(new.sum()) > cfg['maximum_new_fraction']*raw.size)
        recovered = B.copy() if blocked else reachable.copy()
        new = (recovered > 0) & (B == 0) & (raw == 0)
        final = stage.final_mask.copy(); final[new] = 2; stage.final_mask = final
        cluster, distal = frontier(recovered, origin, u, stage.q['tip'], scale)
        info = stage.outline; info['distal'] = distal; xy = distal['candidate_xy']
        limited = bool(np.any((recovered > 0) & (t >= front+limit-1.5*scale)))
        reason = 'area_budget_fallback' if blocked else 'evidence_graph_recovered' if new.any() else 'no_additional_reachable_evidence'
        info.update(recovery=dict(reason=reason, diagnostic_policy='functional_runtime_only'),
                    candidate_tip_xy=xy, reason=reason)
        stage.layers['processed_body'] = ((stage.layers['root_instance'] > 0) | (recovered > 0)).astype(np.uint8)
        stage.layers['reachable_blue'] = recovered
        if xy is None:
            info['tip_observation'] = dict(state='unknown', reason=distal['reason'])
            stage.outline = public(info)
            return
        observation = terminal(frame, raw, np.asarray(xy), u, seed, stage.q['evidence'],
                               stage.q['tip'], scale, distal, limited or blocked, stage.gradient)
        info['tip_observation'] = observation
        info['cap'].update(candidate_xy=xy, state=observation['state'], reason=observation['reason'],
                           observable=observation['state'] == 'current_RGB_confirmed')
        if observation['state'] == 'current_RGB_confirmed':
            info['final_tip_xy'] = xy
            stage.fit_cap(frame, raw, probability, recovered, core, meta, eligible, origin, u, xy, seed)
        stage.outline = public(stage.outline)
