"""Privately bind optimized pixel primitives into the frozen v2 pipeline."""
from __future__ import annotations

from tools.fiber_reference_workspace import ROOT
from experiments.g0_realtime_stage2_20261105.runtime import private_module
from . import pixels


_base = private_module('experiments.g0_realtime_stage2_20261105._v3_base',
                       ROOT/'experiments/g0_realtime_stage2_20261105/runtime.py')
_base.chroma = pixels.chroma
_base.native_gradient = pixels.native_gradient
_base.connect = pixels.connect


class Runtime(_base.Runtime):
    def __init__(self):
        super().__init__()
        self.pixels = pixels.PixelMath(self.chroma)
        self.colors.fast_evidence = self.pixels.evidence
        self.colors.fast_blue_identity = self.pixels.blue_identity
        self.observer.color = lambda bgr, seed, weak=False: self.pixels.color(self.observer, bgr, seed, weak)
        self.stage.fit_cap.__func__.__globals__['connect'] = pixels.connect
