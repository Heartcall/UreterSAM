"""Immutable absolute-coordinate grids and conservative G0 search bounds."""
from __future__ import annotations

from collections import OrderedDict
import math

import cv2
import numpy as np

from tools.fiber_fast_observation import sample
from tools.fiber_local_dark_evidence import nearest


class GridCache:
    def __init__(self, maximum=4):
        self._capacity = maximum
        self.cache = OrderedDict()
        self.hits = self.misses = 0

    def __getattr__(self, key):
        return getattr(np, key)

    def indices(self, dimensions, dtype=int, sparse=False):
        shape = tuple(dimensions)
        if len(shape) != 2 or sparse:
            return np.indices(shape, dtype=dtype, sparse=sparse)
        key = shape, np.dtype(dtype).str
        if key not in self.cache:
            self.misses += 1
            result = np.indices(shape, dtype=dtype)
            result.setflags(write=False)
            self.cache[key] = result
            if len(self.cache) > self._capacity:
                self.cache.popitem(last=False)
        else:
            self.hits += 1
            self.cache.move_to_end(key)
        return self.cache[key]

    def reset_stats(self):
        self.hits = self.misses = 0


def search_box(shape, geometry, cfg, scale):
    """Bound the complete axial interval and the widest taper strip.

    Invert the actual axis/normal matrix rather than assuming exact unit
    vectors. Two extra pixels guard rounding. No truncation at intersection.
    Return None for ill-conditioned/nonfinite inputs, requesting full fallback.
    """
    o, u, n, vertex = [np.asarray(geometry[k], float) for k in
                        ('origin', 'axis', 'normal', 'intersection')]
    lo, end = geometry['body_extent']
    begin = lo+cfg['minimum_search_axial_fraction']*(end-lo)
    limit = end+cfg['maximum_forward_at400']*scale
    if begin > limit:
        return (0, 0, 0, 0)
    tangent = math.tan(geometry['half_angle_radians'])
    intersection_t = float((vertex-o) @ u)
    half = max(0., (intersection_t-begin)*tangent,
               (intersection_t-limit)*tangent)+cfg['search_side_margin_at400']*scale
    matrix = np.stack([u, n])
    if (not np.isfinite([begin, limit, half, *o, *matrix.ravel()]).all()
            or half < 0 or np.linalg.cond(matrix) > 1e8):
        return None
    corners = np.asarray([[begin, -half], [begin, half], [limit, -half], [limit, half]])
    try:
        points = np.linalg.solve(matrix, corners.T).T+o
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(points).all():
        return None
    height, width = shape
    x0 = max(0, min(width, math.floor(float(points[:, 0].min()))-2))
    x1 = max(0, min(width, math.ceil(float(points[:, 0].max()))+3))
    y0 = max(0, min(height, math.floor(float(points[:, 1].min()))-2))
    y1 = max(0, min(height, math.ceil(float(points[:, 1].max()))+3))
    return y0, y1, x0, x1


def roi_tip(frame, raw, g, seed, cfg, scale, blue_identity, counters,
            grid, spatial_counters, fallback):
    box = search_box(raw.shape, g, cfg, scale)
    if box is None:
        spatial_counters['roi_fallback'] += 1
        return fallback(frame, raw, g, seed, cfg, scale)
    o, u, n, vertex = [g[k] for k in ('origin', 'axis', 'normal', 'intersection')]
    blue, color, _ = blue_identity(frame, seed, cfg)
    lo, old_end = g['body_extent']
    begin = lo+cfg['minimum_search_axial_fraction']*(old_end-lo)
    limit = old_end+cfg['maximum_forward_at400']*scale
    y0, y1, x0, x1 = box
    domain = np.zeros_like(raw, bool)
    eligible = np.zeros_like(raw, np.uint8)
    spatial_counters['tip_full_pixels'] += raw.size
    spatial_counters['tip_roi_pixels'] += (y1-y0)*(x1-x0)
    if y0 < y1 and x0 < x1:
        yy, xx = grid.indices(raw.shape)[:, y0:y1, x0:x1]
        # Same absolute-coordinate arithmetic as the full-frame reference.
        t = (xx-o[0])*u[0]+(yy-o[1])*u[1]
        v = (xx-o[0])*n[0]+(yy-o[1])*n[1]
        halfwidth = np.maximum(0., (float((vertex-o) @ u)-t)*math.tan(g['half_angle_radians']))
        local = ((t >= begin) & (t <= limit)
                 & (abs(v) <= halfwidth+cfg['search_side_margin_at400']*scale))
        domain[y0:y1, x0:x1] = local
        # One pixel of REAL image halo is sufficient for the original 3x3
        # count. At an actual image edge, retain BORDER_CONSTANT zero padding.
        hy0, hy1 = max(0, y0-1), min(raw.shape[0], y1+1)
        hx0, hx1 = max(0, x0-1), min(raw.shape[1], x1+1)
        patch = cv2.filter2D(blue[hy0:hy1, hx0:hx1].astype(np.float32), -1,
                            np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT)
        patch = patch[y0-hy0:y1-hy0, x0-hx0:x1-hx0]
        candidate = (local & blue[y0:y1, x0:x1]
                     & (patch >= cfg['minimum_local_blue_pixels_3x3'])
                     & (raw[y0:y1, x0:x1] != 1))
        y, x = np.nonzero(candidate); y += y0; x += x0
    else:
        y = x = np.empty(0, int)
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
            j = int(valid_ids[0]); chosen = int(ids[j]); end = float(axial[chosen]); target = o+end*u
            meta.update(source_pixel=points[chosen], vertex=target, tip_t=end,
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
