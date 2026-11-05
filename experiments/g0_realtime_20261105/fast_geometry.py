"""Identical G0 arithmetic with grouped side samples and reusable pixel grids.

Load the trusted implementation into a private module namespace to substitute
two functions without patching the repository modules used by the reference.
"""
from __future__ import annotations

from functools import lru_cache
import importlib.util
import math

import numpy as np

from tools.fiber_reference_workspace import ROOT
from tools.fiber_straight_semicircle import unit
from .fast_evidence import fast_blue_identity


@lru_cache(maxsize=4)
def pixel_grid(shape):
    yy, xx = np.indices(shape)
    points = np.stack([xx, yy], axis=-1).astype(float)
    points.setflags(write=False)
    return points


def grouped_observed_sides(body, origin, u, cfg, scale):
    y, x = np.nonzero(body)
    points = np.c_[x, y].astype(float)
    normal = np.array([-u[1], u[0]])
    t, v = (points-origin) @ u, (points-origin) @ normal
    lo, hi = float(t.min()), float(t.max())
    bins = np.floor((t-lo)/(cfg['profile_step_at400']*scale)).astype(int)
    # Stable grouping retains original raster-order tie breaking and the exact
    # order of floating-point values supplied to each per-bin mean.
    order = np.argsort(bins, kind='stable')
    cut = np.flatnonzero(np.diff(bins[order])) + 1
    left, right = [], []
    for ix in np.split(order, cut):
        if not lo+cfg['axial_fraction'][0]*(hi-lo) <= np.mean(t[ix]) <= lo+cfg['axial_fraction'][1]*(hi-lo):
            continue
        a, b = points[ix[np.argmin(v[ix])]], points[ix[np.argmax(v[ix])]]
        margin = min(*a, body.shape[1]-1-a[0], body.shape[0]-1-a[1],
                     *b, body.shape[1]-1-b[0], body.shape[0]-1-b[1])
        if margin < cfg['frame_margin_at400']*scale:
            continue
        left.append(a); right.append(b)
    return np.asarray(left).reshape(-1, 2), np.asarray(right).reshape(-1, 2), (lo, hi)


def raster_bounds(shape, g, center, radius, join_t):
    """Conservative bbox of clipped rod polygon plus the complete cap circle."""
    h, w = shape
    polygon = [np.array(p, float) for p in [(0, 0), (w-1, 0), (w-1, h-1), (0, h-1)]]
    planes = []
    origin = np.asarray(g['origin'])
    for line in g['independent_lines']:
        p = np.asarray(line['point']); v = np.asarray(line['direction'])
        n = np.array([-v[1], v[0]])
        sign = 1. if float((origin-p) @ n) > 0 else -1.
        planes.append((p, n*sign))
    axis = np.asarray(g['axis'])
    planes.append((center+join_t*axis, -axis))
    for p, normal in planes:
        clipped = []
        if not polygon:
            break
        previous = polygon[-1]; pv = float((previous-p) @ normal)
        for current in polygon:
            cv = float((current-p) @ normal)
            if (cv >= 0) != (pv >= 0):
                clipped.append(previous+(pv/(pv-cv))*(current-previous))
            if cv >= 0:
                clipped.append(current)
            previous, pv = current, cv
        polygon = clipped
    bounds = polygon + [center-radius, center+radius]
    lo = np.floor(np.min(bounds, axis=0)).astype(int)-2
    hi = np.ceil(np.max(bounds, axis=0)).astype(int)+3
    return slice(max(0, lo[1]), min(h, max(0, hi[1]))), slice(max(0, lo[0]), min(w, max(0, hi[0])))


def cached_raster_geometry(shape, g, T, samples=129):
    u, n, V, o = [np.asarray(g[k], float) for k in ['axis', 'normal', 'intersection', 'origin']]
    T = np.asarray(T, float)
    d = float((V-T) @ u); sine = math.sin(g['half_angle_radians'])
    if d <= 1e-6 or not 0 < sine < 1:
        return None, None, None, None, 'tip_at_or_beyond_side_intersection'
    radius = d*sine/(1-sine); center = T-radius*u; joins = []
    # Join positions are tiny scalar geometry; determine a conservative bounded
    # raster domain before allocating pointwise full-image intermediates.
    for line in g['independent_lines']:
        p = np.asarray(line['point']); direction = np.asarray(line['direction'])
        normal = np.array([-direction[1], direction[0]])
        joins.append(center+float((p-center) @ normal)*normal)
    join_t = float(np.mean([(q-center) @ u for q in joins]))
    region = raster_bounds(shape, g, center, radius, join_t)
    points = pixel_grid(tuple(shape))[region]; inside = np.ones(points.shape[:2], bool)
    for line in g['independent_lines']:
        p = np.asarray(line['point']); direction = np.asarray(line['direction'])
        normal = np.array([-direction[1], direction[0]])
        sign = 1. if float((o-p) @ normal) > 0 else -1.
        inside &= ((points-p) @ normal)*sign >= -1e-9
    along = (points-center) @ u
    rod, cap = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    rod[region] = inside & (along <= join_t+1e-9)
    cap[region] = ((along >= join_t-1e-9)
                  & (np.sum((points-center)**2, axis=-1) <= radius*radius+1e-9))
    bound = math.pi/2-g['half_angle_radians']; theta = np.linspace(-bound, bound, samples)
    arc = center+radius*np.cos(theta)[:, None]*u+radius*np.sin(theta)[:, None]*n
    errors = []
    for line, q in zip(g['independent_lines'], joins):
        direction = np.asarray(line['direction']); radial = unit(q-center)
        tangent = np.array([-radial[1], radial[0]])
        errors.append(float(np.degrees(np.arccos(np.clip(abs(tangent @ direction), -1, 1)))))
    info = dict(center=center, vertex=T, radius=radius, curve=arc, join_points=joins,
                join_t_relative_center=join_t, analytic_join_angle_degrees=errors,
                axis_distance=float(abs((T-o) @ n)), arc_span_degrees=float(np.degrees(2*bound)),
                circle_line_distance_errors=[float(abs(np.linalg.norm(q-center)-radius)) for q in joins],
                construction='original_converging_lines_tangent_circle_through_RGB_projected_vertex',
                parallel_projection=False)
    return rod | cap, rod, cap, info, None


_spec = importlib.util.spec_from_file_location(
    'tools._g0_realtime_20261105_private', ROOT/'tools/fiber_tapered_tangent_cap.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
_module.observed_sides = grouped_observed_sides
_module.raster_geometry = cached_raster_geometry
_module.blue_identity = fast_blue_identity
FastGeometry = _module.TaperedTangentCap
