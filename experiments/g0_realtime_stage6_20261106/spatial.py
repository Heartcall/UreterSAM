"""Exact absolute-coordinate S2 domain, bounded before NumPy broadcasting."""
import math
import numpy as np


def rectangle(shape, origin, u, normal, lo, hi, center, lateral):
    full = (0, shape[0], 0, shape[1])
    if not np.isfinite([*origin, *u, *normal, lo, hi, center, lateral]).all():
        return full, True
    matrix = np.stack([u, normal])
    if np.linalg.cond(matrix) > 1e8:
        return full, True
    corners = np.asarray([[lo, center-lateral], [lo, center+lateral],
                          [hi, center-lateral], [hi, center+lateral]])
    try:
        points = np.linalg.solve(matrix, corners.T).T+origin
    except np.linalg.LinAlgError:
        return full, True
    if not np.isfinite(points).all():
        return full, True
    h, w = shape
    return (max(0, min(h, math.floor(float(points[:, 1].min()))-2)),
            max(0, min(h, math.ceil(float(points[:, 1].max()))+3)),
            max(0, min(w, math.floor(float(points[:, 0].min()))-2)),
            max(0, min(w, math.ceil(float(points[:, 0].max()))+3))), False


def domain_from_body(body, origin, u, cfg, scale):
    normal = np.array([-u[1], u[0]])
    by, bx = np.nonzero(body > 0)
    # Keep row-major quantile inputs and the reference's elementwise order.
    bt = (bx-origin[0])*u[0]+(by-origin[1])*u[1]
    bv = (bx-origin[0])*normal[0]+(by-origin[1])*normal[1]
    front = float(bt.max())
    limit = min(cfg['maximum_forward_at400']*scale,
                cfg['maximum_forward_diagonal_fraction']*np.hypot(*body.shape))
    lateral = (np.quantile(bv, .98)-np.quantile(bv, .02))*cfg['lateral_width_multiplier']/2+cfg['lateral_margin_at400']*scale
    center = float(np.median(bv)); lo = bt.min()-2*scale; hi = front+limit
    (y0, y1, x0, x1), fallback = rectangle(body.shape, origin, u, normal, lo, hi, center, lateral)
    domain = np.zeros(body.shape, bool)
    if y0 < y1 and x0 < x1:
        # Same absolute integer values as np.indices; fewer subtraction arrays.
        yy = np.arange(y0, y1, dtype=int)[:, None]
        xx = np.arange(x0, x1, dtype=int)[None, :]
        t = (xx-origin[0])*u[0]+(yy-origin[1])*u[1]
        v = (xx-origin[0])*normal[0]+(yy-origin[1])*normal[1]
        domain[y0:y1, x0:x1] = (t >= lo) & (t <= hi) & (abs(v-center) <= lateral)
    return domain, front, limit, dict(full_pixels=body.size,
        roi_pixels=(y1-y0)*(x1-x0), fallback=int(fallback))
