"""Read only explicit stage8 traces. Simulation does not run segmentation."""
from pathlib import Path
import hashlib,json
from .simulation import simulate,summarize

ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    output=ROOT/'runs/g0_realtime_stage9_20261106/trace_analysis';output.mkdir(parents=True,exist_ok=False)
    base=ROOT/'runs/g0_realtime_stage8_20261106/full';summaries={};identities={}
    for video in ('video_1','video_2'):
        for arm in ('v7-serial','v7-pipeline'):
            path=base/f'{arm}_{video}'/'frames.jsonl';identities[str(path.relative_to(ROOT))]=sha(path)
            records=[json.loads(line) for line in path.read_text().splitlines()]
            services=[]
            for r in records:
                w=r['wall_intervals'];q=r['timings']['queue_wait_ms']/1000
                # Include observed decode/input hashing and output/audit overhead.
                # Remove only measured ready-queue wait from the NN stage.
                n=w['network'][1]-w['admitted']+(w['post'][0]-w['network'][1]-q)
                p=w['post'][1]-w['post'][0];release=w['released']-w['post'][0]
                assert n>=0 and release>=p>=0
                services.append(dict(network=n,post=p,release=release))
            policies=('fifo_serial','latest_serial') if arm=='v7-serial' else ('fifo_pipeline2','latest_pipeline2','latest_ready')
            for policy in policies:
                rows,counts=simulate(services,30.,policy)
                assert [r['frame_index'] for r in rows]==sorted(set(r['frame_index'] for r in rows))
                key=f'{video}_{policy}'
                scenarios=[summarize(rows,counts,len(records),30.,100.,e,display)
                           for e,display in ((0.,None),(10.,30.),(10.,60.),(20.,60.))]
                summaries[key]=dict(trace_arm=arm,scenarios=scenarios)
                (output/f'{key}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    frozen=json.loads((base/'v7-pipeline_video_1/identity.json').read_text())['sources']
    assert all(sha(ROOT/name)==value for name,value in frozen.items())
    sources={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}
    result=dict(scope='DETERMINISTIC FIXED-SERVICE SIMULATION ONLY; no camera, display, dropped-frame segmentation or speedup validated',
        input_fps=30,deadline_ms=100,original_stage8_sources_unchanged=len(frozen),
        trace_sha256=identities,sources=sources,assumptions=[
            'Capture arrivals are exactly index/30; no sensor/backend buffering or jitter modeled.',
            'Each retained frame borrows its measured stage8 service time although state and load may differ after dropping.',
            'CPU/NN service times are fixed; resource contention changes are not predicted.',
            'No preemption, stale-mask reprojection or cross-frame overlay. Paired RGB/result only.',
            'Latest-ready has one overwriteable ready slot and one NN + one post worker; can waste NN work.',
            'External delay and refresh rates are hypothetical sensitivity budgets, not measurements.'],
        results=summaries)
    (output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for key,value in summaries.items():
        a=value['scenarios'][0]
        print(json.dumps(dict(case=key,processed=a['processed_frames'],skipped=a['unprocessed_frames'],
            p95=a['latency_ms']['p95'],maximum=a['latency_ms']['maximum'],
            deadline_miss_fraction=a['deadline_miss_fraction_of_processed'],timely_hz=a['timely_distinct_results_per_source_second'])))


if __name__=='__main__':main()
