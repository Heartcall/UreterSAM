"""v6 plus current-frame region color reuse and private sparse grid views."""
import types
import numpy as np

from tools.fiber_reference_workspace import ROOT
from experiments.g0_realtime_stage2_20261105.runtime import private_module
from experiments.g0_realtime_stage6_20261106.runtime import Runtime as V6
from .color import from_chroma


class BroadcastNumpy:
    """Private globals only: two callers unpack indices and then broadcast."""
    def __init__(self, grid, counts, tag):
        self.grid, self.counts, self.tag = grid, counts, tag

    def __getattr__(self, name):
        return getattr(self.grid, name)

    def indices(self, dimensions, dtype=int, sparse=False):
        if not sparse and len(dimensions) == 2:
            grid = self.grid.indices(dimensions, dtype=dtype)
            self.counts[self.tag] += 1
            return grid[0, :, :1], grid[1, :1, :]
        return self.grid.indices(dimensions, dtype=dtype, sparse=sparse)


class Runtime(V6):
    def __init__(self, reuse=True, broadcast=True):
        super().__init__()
        self.extra_counts = dict(region_reused=0, region_fallback=0, grow_broadcast=0, cap_broadcast=0)
        growth = self.stage.inner
        while hasattr(growth, 'inner'):
            growth = growth.inner
        growth = growth.growth
        self.growth = growth
        module = private_module('experiments.g0_realtime_stage7_20261106._grow',
                                ROOT/'experiments/g0_realtime_stage7_20261106/grow.py')
        module.np = BroadcastNumpy(self.grid, self.extra_counts, 'grow_broadcast') if broadcast else self.grid
        module.region_color = self.region_color if reuse else lambda frame, region, seed: self.observer.color(frame[region], seed)
        growth.grow = types.MethodType(module.grow, growth)
        if broadcast:
            original = self.stage.fit_cap.__func__
            private_globals = dict(original.__globals__)
            private_globals['np'] = BroadcastNumpy(self.grid, self.extra_counts, 'cap_broadcast')
            function = types.FunctionType(original.__code__, private_globals, original.__name__,
                                          original.__defaults__, original.__closure__)
            self.stage.fit_cap = types.MethodType(function, self.stage)

    def region_color(self, frame, region, seed):
        n = int(np.count_nonzero(region))
        if frame is not self.frame or frame.dtype != np.uint8 or n < 4096:
            self.extra_counts['region_fallback'] += 1
            return self.observer.color(frame[region], seed)
        color, noise = self.chroma(frame)
        self.extra_counts['region_reused'] += 1
        return from_chroma(self.observer, color[region], noise[region], seed)

    def reset(self):
        super().reset()
        for name in self.extra_counts:
            self.extra_counts[name] = 0
