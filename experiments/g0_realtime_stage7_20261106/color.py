"""Original large-array color gate with precomputed chroma and noise."""
import math
import numpy as np
from experiments.g0_realtime_stage3_20261105.pixels import sum3, norm3


def from_chroma(observer, color, noise, seed, weak=False):
    cfg = observer.cfg['color']; suffix = 'weak' if weak else 'normal'
    blue = color[..., 2]-.5*(color[..., 0]+color[..., 1])
    bg = seed['bg']; bgblue = bg[..., 2]-.5*(bg[..., 0]+bg[..., 1])
    necessary = ((cfg['max_noise']-noise > 0)
                 & (blue-bgblue-cfg['blue_gap_'+suffix+'_fraction']*seed['gap'] > 0))
    result = np.zeros_like(necessary)
    if necessary.any():
        c, ns = color[necessary], noise[necessary]
        v = c-1/3; norm = norm3(v)
        cosine = sum3(v*seed['v'])/np.maximum(norm*np.linalg.norm(seed['v']), 1e-9)
        separation = norm3(c-bg)-norm3(c-seed['mu'])-cfg['background_distance_margin_'+suffix]
        result[necessary] = ((norm-ns > 0)
                             & (cosine-math.cos(math.radians(cfg['angle_'+suffix+'_degrees'])) > 0)
                             & (separation > 0))
    return result
