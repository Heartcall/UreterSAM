"""Independent profile-only and component-plus-profile ablations."""
from experiments.g0_realtime_stage7_20261106.runtime import Runtime as V7
from .runtime import Runtime as ComponentRuntime
from .profiles import grouped_profiles


def install(runtime):
    runtime.stage.fit_cap.__func__.__globals__['profiles']=grouped_profiles
    runtime.stage.side.run.__func__.__globals__['profiles']=grouped_profiles


class ProfileOnlyRuntime(V7):
    def __init__(self):
        super().__init__();install(self)


class Runtime(ComponentRuntime):
    def __init__(self):
        super().__init__();install(self)
