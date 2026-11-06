"""Hypothetical CPU sensitivity and actual existing-state TTL probe."""
from pathlib import Path
import hashlib,json
import numpy as np
from tools.fiber_evidence_tracker import EvidenceTracker
from .simulation import simulate,summarize

ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    base=ROOT/'runs/g0_realtime_stage9_20261106';sensitivity={}
    for video in ('video_1','video_2'):
        records=[json.loads(line) for line in (ROOT/f'runs/g0_realtime_stage8_20261106/full/v7-pipeline_{video}/frames.jsonl').read_text().splitlines()]
        cases=[]
        for factor in (1.,.8,.6,.5,.4):
            services=[]
            for r in records:
                w=r['wall_intervals'];q=r['timings']['queue_wait_ms']/1000
                n=w['network'][1]-w['admitted']+(w['post'][0]-w['network'][1]-q)
                p=(w['post'][1]-w['post'][0])*factor
                services.append(dict(network=n,post=p,release=p+w['released']-w['post'][1]))
            for policy in ('fifo_pipeline2','latest_ready'):
                rows,counts=simulate(services,30.,policy)
                s=summarize(rows,counts,len(records),30.,100.,10.,60.)
                cases.append(dict(cpu_service_factor=factor,policy=policy,summary=s))
        sensitivity[video]=cases
    config=json.loads((ROOT/'configs/fiber_temporal_observation_search_20261030.json').read_text())
    probes=[]
    for dt in (1/30,.1):
        tracker=EvidenceTracker(config);tracker.prepare((400,400),0.,'synthetic')
        tracker.groups=dict(shaft=np.array([40.,30.,0.,8.]),endpoint=np.array([30.,4.]))
        tracker.velocity={k:np.zeros_like(v) for k,v in tracker.groups.items()}
        rows=[]
        for step in range(1,5):
            prepared=tracker.prepare((400,400),step*dt,'synthetic')
            output=tracker.update(dict(candidates=[],strong_label=None),None,prepared)
            rows.append(dict(elapsed_ms=step*dt*1000,dt=prepared['dt_seconds'],
                             age=output['groups']['shaft']['age'],status=output['groups']['shaft']['status']))
        assert [r['age'] for r in rows]==[1,2,3,4]
        assert [r['status'] for r in rows]==['predicted']*3+['expired']
        probes.append(dict(processed_interval_seconds=dt,steps=rows))
    trace=json.loads((base/'trace_analysis/results.json').read_text())
    assert all(sha(ROOT/name)==value for name,value in trace['trace_sha256'].items())
    result=dict(sensitivity_scope='Hypothetical multiplicative CPU reduction using fixed historical NN/audit services; no implementation or speedup measurement. Assumes external budget 10 ms and 60 Hz presentation.',
                sensitivities=sensitivity,
                temporal_probe_scope='Executed existing EvidenceTracker on synthetic missing observations; no algorithm or threshold changed; physical lifetime changes when processed-frame count differs.',
                temporal_probes=probes,
                driver_sha256=sha(Path(__file__)),
                state_source_sha256=sha(ROOT/'tools/fiber_evidence_tracker.py'),
                state_config_sha256=sha(ROOT/'configs/fiber_temporal_observation_search_20261030.json'))
    (base/'followup_analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for video,cases in sensitivity.items():
        for c in cases:
            s=c['summary'];print(json.dumps(dict(video=video,factor=c['cpu_service_factor'],policy=c['policy'],
                p95=s['latency_ms']['p95'],maximum=s['latency_ms']['maximum'],
                miss_pct=100*s['deadline_miss_fraction_of_processed'],timely_hz=s['timely_distinct_results_per_source_second'])))
    print(json.dumps(probes,indent=2))


if __name__=='__main__':main()
