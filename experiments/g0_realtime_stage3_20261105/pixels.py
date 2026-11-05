"""Three-channel arithmetic and label lookup, with original precision/order."""
from __future__ import annotations

import math

import cv2
import numpy as np

from tools.fiber_connected_blue_growth import chroma as original_chroma
from tools.fiber_weak_stage2 import evidence as original_evidence
from tools.fiber_straight_semicircle import blue_identity as original_blue_identity


def sum3(a):
    return (a[..., 0]+a[..., 1])+a[..., 2]


def norm3(a):
    return np.sqrt(sum3(a*a))


def chroma(bgr):
    rgb = np.asarray(bgr, np.float32)[..., ::-1]
    total = sum3(rgb)
    return rgb/(total[..., None]+1e-6), 3./(total+3.)


def native_gradient(frame):
    image = frame.astype(np.float32)/255.
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)/8
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)/8
    np.multiply(gx, gx, out=gx)
    np.multiply(gy, gy, out=gy)
    np.add(gx, gy, out=gx)
    result = sum3(gx)
    result /= 3
    np.sqrt(result, out=result)
    return result


def connect(binary, anchor):
    count, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, np.uint8)
    keep[labels[anchor & (binary > 0)]] = 1
    keep[0] = 0
    return keep[labels]


def rooted(binary, core, minimum):
    count, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=4)
    counts = np.bincount(labels[core], minlength=count)
    ids = np.flatnonzero((counts >= minimum) & (counts > 0))
    ids = ids[ids > 0]
    return ((labels == ids[0]).astype(np.uint8) if len(ids) == 1
            else np.zeros_like(binary, np.uint8)), int(len(ids))


class PixelMath:
    def __init__(self, get_chroma):
        self.chroma = get_chroma

    def evidence(self, frame, seed, cfg, gradient=None):
        # Continuous probe scores participate in terminal decisions. Keep those
        # small probes on the original implementation, including its dtype.
        if frame.shape[1] < 32 or frame.shape[0]*frame.shape[1] < 4096:
            return original_evidence(frame, seed, cfg, gradient)
        color, noise = self.chroma(frame)
        blue = color[..., 2]-.5*(color[..., 0]+color[..., 1])
        bg = seed['bg']
        gap = blue-(bg[2]-.5*(bg[0]+bg[1]))
        necessary = ((blue > cfg['minimum_blue_excess'])
                     & (gap > max(cfg['minimum_blue_excess'], cfg['blue_seed_gap_fraction']*seed['gap']))
                     & (noise < cfg['max_chroma_noise']))
        edge = native_gradient(frame) if gradient is None else np.asarray(gradient, np.float32)
        if edge.shape != frame.shape[:-1]:
            raise ValueError('Gradient/sample shape mismatch')
        score = np.zeros(frame.shape[:2], np.float32)
        strong = np.zeros_like(necessary); weak = strong.copy(); uncertain = strong.copy()
        if necessary.any():
            # Reuse exactly the already normalized RGB values. The original
            # three-term reductions have the same left-to-right grouping.
            c, ns, e = color[necessary], noise[necessary], edge[necessary]
            v = c-1/3; norm = norm3(v)
            cosine = sum3(v*seed['v'])/np.maximum(norm*np.linalg.norm(seed['v']), 1e-9)
            separation = norm3(c-bg)-norm3(c-seed['mu'])
            a = np.clip((cosine-cfg['cosine_floor'])/(1-cfg['cosine_floor']), 0, 1)
            b = np.clip(gap[necessary]/max(seed['gap'], cfg['minimum_blue_excess']), 0, 1)
            d = np.clip((separation+cfg['distance_margin'])/(2*cfg['distance_margin']), 0, 1)
            reliability = np.clip(1-ns/cfg['max_chroma_noise'], 0, 1)
            value = (.4*a+.35*b+.25*d)*np.sqrt(reliability)
            identity = (cosine > cfg['cosine_floor']) & (norm > cfg['minimum_chroma_norm'])
            support = (separation > -cfg['distance_margin']) | (e >= cfg['minimum_gradient'])
            w = identity & support & (value >= cfg['weak_score'])
            st = w & (value >= cfg['strong_score'])
            score[necessary] = value
            strong[necessary] = st; weak[necessary] = w & ~st; uncertain[necessary] = identity & ~w
        return score, strong, weak, uncertain, edge, noise

    def color(self, observer, bgr, seed, weak=False):
        if int(np.prod(bgr.shape[:-1])) < 4096:
            return observer.color_score(bgr, seed, weak) > 0
        color, noise = self.chroma(bgr)
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

    def blue_identity(self, frame, seed, cfg):
        if frame.shape[1] < 32 or frame.shape[0]*frame.shape[1] < 4096:
            return original_blue_identity(frame, seed, cfg)
        color, noise = self.chroma(frame)
        blue = color[..., 2]-.5*(color[..., 0]+color[..., 1])
        bg = seed['bg']; gap = blue-(bg[2]-.5*(bg[0]+bg[1]))
        necessary = ((blue > cfg['minimum_blue_advantage'])
                     & (gap > max(cfg['minimum_blue_advantage'], cfg['minimum_seed_gap_fraction']*seed['gap']))
                     & (noise < cfg['maximum_chroma_noise']))
        result = np.zeros_like(necessary)
        if necessary.any():
            # Keep matmul here: the original tip uses it instead of sum(v*r).
            c = color[necessary][:, None, :]; v = c-1/3
            norm = np.linalg.norm(v, axis=-1); reference = seed['mu']-1/3
            cosine = (v@reference)/np.maximum(norm*np.linalg.norm(reference), 1e-12)
            result[necessary] = ((norm > cfg['minimum_chroma_norm'])
                                 & (cosine > cfg['minimum_seed_cosine'])).ravel()
        return result, color, noise
