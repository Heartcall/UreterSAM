"""Screening candidate only: one C-order region index instead of two masks."""
import numpy as np
from experiments.g0_realtime_stage7_20261106.runtime import Runtime as V7
from experiments.g0_realtime_stage7_20261106.color import from_chroma


def indexed_color(observer,color,noise,region,seed):
    ids=np.flatnonzero(region)
    return from_chroma(observer,color.reshape(-1,3)[ids],noise.reshape(-1)[ids],seed)


class Runtime(V7):
    def region_color(self,frame,region,seed):
        if frame is not self.frame or frame.dtype != np.uint8:
            self.extra_counts['region_fallback']+=1
            return self.observer.color(frame[region],seed)
        ids=np.flatnonzero(region)
        if len(ids)<4096:
            self.extra_counts['region_fallback']+=1
            return self.observer.color(frame[region],seed)
        color,noise=self.chroma(frame)
        self.extra_counts['region_reused']+=1
        return from_chroma(self.observer,color.reshape(-1,3)[ids],noise.reshape(-1)[ids],seed)
