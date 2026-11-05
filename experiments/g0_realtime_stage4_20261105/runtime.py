"""Private instance bindings for immutable grids and complete bounded tip ROI."""
import types

from tools.fiber_reference_workspace import ROOT
from experiments.g0_realtime_stage2_20261105.runtime import private_module
from experiments.g0_realtime_stage3_20261105.runtime import Runtime as V3
from experiments.g0_realtime_stage3_20261105 import pixels
from .spatial import GridCache, roi_tip


class Runtime(V3):
    def __init__(self, roi=True):
        self.grid = GridCache()
        self.spatial_counters = dict(tip_full_pixels=0, tip_roi_pixels=0, roi_fallback=0)
        spatial = private_module('experiments.g0_realtime_stage2_20261105._v4_spatial',
                                 ROOT/'experiments/g0_realtime_stage2_20261105/runtime.py')
        spatial.np = self.grid
        spatial.native_gradient = pixels.native_gradient
        spatial.connect = pixels.connect
        self.expand = types.MethodType(spatial.Runtime.expand, self)
        super().__init__()
        self.stage.fit_cap.__func__.__globals__['np'] = self.grid
        node = self.stage.inner
        while hasattr(node, 'inner'):
            node = node.inner
        union = private_module('tools._g0_v4_union', ROOT/'tools/fiber_connected_blue_union.py')
        union.np = self.grid
        node.growth.grow = types.MethodType(union.ConnectedBlue.grow, node.growth)
        geometry = self.g0.run.__func__.__globals__
        original = geometry['locate_tip']
        if roi:
            geometry['locate_tip'] = lambda *args: roi_tip(
                *args, self.colors.fast_blue_identity, self.counters,
                self.grid, self.spatial_counters, original)
        else:
            # Isolated grid-only ablation: the old lazy tip has private globals.
            spatial.chroma = pixels.chroma
            geometry['locate_tip'] = lambda *args: spatial.lazy_tip(
                *args, self.colors.fast_blue_identity, self.counters)

    def reset(self):
        super().reset()
        self.grid.reset_stats()
        for key in self.spatial_counters:
            self.spatial_counters[key] = 0
