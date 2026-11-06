"""v7 plus exact seed component reuse; no stage9 region candidate."""
from experiments.g0_realtime_stage7_20261106.runtime import Runtime as V7
from .seeds import qualified_seeds


class Runtime(V7):
    def __init__(self,roi=True):
        super().__init__()
        self.seed_counts=dict(outer_component_passes=0,inner_component_passes_saved=0,
                              coordinate_full_pixels=0,coordinate_evaluated_pixels=0)
        def capture(frame,raw,probability,prior,age):
            result=qualified_seeds(self.observer,frame,raw,probability,prior,age,roi,self.seed_counts)
            self._captured=(frame,raw,probability,result)
            return result
        self.observer.seeds=capture

    def reset(self):
        super().reset()
        for key in self.seed_counts:self.seed_counts[key]=0
