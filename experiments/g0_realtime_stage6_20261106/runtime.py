"""v4 NumPy with bounded S2 coordinates; full evidence and graph retained."""
import types
import numpy as np
from tools.fiber_local_dark_evidence import public
from tools.fiber_weak_stage2 import frontier, terminal
from experiments.g0_realtime_stage3_20261105.pixels import native_gradient, connect
from experiments.g0_realtime_stage4_20261105.runtime import Runtime as V4
from .spatial import domain_from_body

class Runtime(V4):
    def __init__(self):
        super().__init__()
        self.domain_counters = dict(full_pixels=0, roi_pixels=0, fallback=0)
        self.expand = types.MethodType(expand, self)
        self.last_domain = None

    def run(self, *args, **kwargs):
        self.last_domain = None
        return super().run(*args, **kwargs)

    def reset(self):
        super().reset()
        self.last_domain = None
        for key in self.domain_counters:
            self.domain_counters[key] = 0

def expand(self, frame, raw, probability, meta, B, core, u):
    stage = self.stage
    q = next(z for z in meta['seeds'] if z['reason'] == 'qualified')
    mu = np.asarray(q['mu'], np.float32)
    seed = dict(mu=mu, v=mu-1/3, bg=np.asarray(q['background_chroma'], np.float32), gap=q['relative_blue_advantage'])
    stage.gradient = native_gradient(frame)
    score, strong, weak, uncertain, edge, noise = self.evidence(frame, seed, stage.q['evidence'], stage.gradient)
    scale = min(raw.shape)/400; cfg = stage.q['search']
    origin = np.asarray(stage.outline['context']['origin']); normal = np.array([-u[1], u[0]])
    domain, front, limit, counters = domain_from_body(B, origin, u, cfg, scale)
    self.last_domain = domain
    for key, value in counters.items():
        self.domain_counters[key] += value
    eligible = (strong | weak) & domain & (raw != 1)
    reachable = connect(eligible | (B > 0), (B > 0) & core)
    new = (reachable > 0) & (B == 0) & (raw == 0)
    blocked = (int(reachable.sum()) > cfg['maximum_component_fraction']*raw.size
               or int(new.sum()) > cfg['maximum_new_fraction']*raw.size)
    recovered = B.copy() if blocked else reachable.copy()
    new = (recovered > 0) & (B == 0) & (raw == 0)
    final = stage.final_mask.copy(); final[new] = 2; stage.final_mask = final
    cluster, distal = frontier(recovered, origin, u, stage.q['tip'], scale)
    info = stage.outline; info['distal'] = distal; xy = distal['candidate_xy']
    ry, rx = np.nonzero(recovered > 0)
    rt = (rx-origin[0])*u[0]+(ry-origin[1])*u[1]
    limited = bool(np.any(rt >= front+limit-1.5*scale))
    reason = 'area_budget_fallback' if blocked else 'evidence_graph_recovered' if new.any() else 'no_additional_reachable_evidence'
    info.update(recovery=dict(reason=reason, diagnostic_policy='functional_runtime_only'),
                candidate_tip_xy=xy, reason=reason)
    stage.layers['processed_body'] = ((stage.layers['root_instance'] > 0) | (recovered > 0)).astype(np.uint8)
    stage.layers['reachable_blue'] = recovered
    if xy is None:
        info['tip_observation'] = dict(state='unknown', reason=distal['reason'])
        stage.outline = public(info)
        return
    observation = terminal(frame, raw, np.asarray(xy), u, seed, stage.q['evidence'],
                           stage.q['tip'], scale, distal, limited or blocked, stage.gradient)
    info['tip_observation'] = observation
    info['cap'].update(candidate_xy=xy, state=observation['state'], reason=observation['reason'],
                       observable=observation['state'] == 'current_RGB_confirmed')
    if observation['state'] == 'current_RGB_confirmed':
        info['final_tip_xy'] = xy
        stage.fit_cap(frame, raw, probability, recovered, core, meta, eligible, origin, u, xy, seed)
    stage.outline = public(stage.outline)
