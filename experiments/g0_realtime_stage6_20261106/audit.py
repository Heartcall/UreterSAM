"""Lossless two-bit label storage for deferred comparison exports."""
import numpy as np


def pack_masks(s2, final):
    if s2.ndim != 2 or final.shape != s2.shape:
        raise ValueError('Expected equal two-dimensional audit masks')
    for mask in (s2, final):
        if mask.dtype != np.uint8 or np.any(mask > 2):
            raise ValueError('Audit labels must be uint8 values 0, 1 or 2')
    return np.packbits(np.stack((s2 == 1, s2 == 2, final == 1, final == 2)))


def unpack_masks(packed, shape):
    layers = np.unpackbits(packed, count=4*int(np.prod(shape))).reshape(4, *shape)
    return layers[0]+2*layers[1], layers[2]+2*layers[3]
