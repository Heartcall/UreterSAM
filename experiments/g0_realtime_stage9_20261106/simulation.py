"""Deterministic service-trace simulation, not camera or algorithm validation."""
from bisect import bisect_right
import math


def simulate(services, fps, policy):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Positive finite source FPS required')
    if policy not in ('fifo_serial', 'fifo_pipeline2', 'latest_serial', 'latest_pipeline2', 'latest_ready'):
        raise ValueError('Unknown policy')
    for s in services:
        n, p, r = (s[k] for k in ('network', 'post', 'release'))
        if not all(math.isfinite(x) for x in (n,p,r)) or n < 0 or not 0 <= p <= r or n+r <= 0:
            raise ValueError('Invalid service durations')
    if not services:
        return [], dict(network_executions=0, input_skipped=0, ready_replaced=0)
    count=len(services); outputs=[]
    def row(i,n0,n1,p0):
        return dict(frame_index=i,captured=i/fps,network_start=n0,network_end=n1,
                    post_start=p0,result=p0+services[i]['post'],released=p0+services[i]['release'])
    def newest(t, minimum):
        return min(count-1,max(minimum,math.floor(t*fps+1e-10)))
    if policy == 'latest_ready':
        # One NN worker takes the latest captured frame. Completed NN results
        # replace a single ready slot; post consumes the newest ready result.
        net=[]; i=0; free=0.
        while i < count:
            i=newest(free,i); start=max(free,i/fps); free=start+services[i]['network']
            net.append((i,start,free)); i+=1
        ends=[x[2] for x in net]; k=0; free=0.
        while k < len(net):
            free=max(free,ends[k]); j=max(k,bisect_right(ends,free+1e-12)-1)
            i,n0,n1=net[j]; output=row(i,n0,n1,free);outputs.append(output)
            free=output['released'];k=j+1
        return outputs,dict(network_executions=len(net),input_skipped=count-len(net),ready_replaced=len(net)-len(outputs))
    latest=policy.startswith('latest'); pipeline=policy.endswith('pipeline2')
    i=0; network_free=post_free=0.
    while i < count:
        gate=outputs[-2]['released'] if pipeline and len(outputs)>=2 else post_free if not pipeline else 0.
        available=max(network_free,gate)
        if latest: i=newest(available,i)
        n0=max(available,i/fps);n1=n0+services[i]['network'];p0=max(n1,post_free)
        output=row(i,n0,n1,p0);outputs.append(output)
        network_free=n1;post_free=output['released'];i+=1
    return outputs,dict(network_executions=len(outputs),input_skipped=count-len(outputs),ready_replaced=0)


def quantile(values,q):
    values=sorted(values)
    if not values:return None
    x=(len(values)-1)*q;i=math.floor(x);j=math.ceil(x)
    return values[i]+(values[j]-values[i])*(x-i)


def distribution(values):
    if not values:return dict(mean=None,p50=None,p95=None,p99=None,maximum=None)
    return dict(mean=sum(values)/len(values),p50=quantile(values,.5),p95=quantile(values,.95),
                p99=quantile(values,.99),maximum=max(values))


def summarize(rows, counts, total_frames, fps, deadline_ms, external_ms=0., display_fps=None):
    # The external budget and refresh phase are hypothetical sensitivity inputs.
    if deadline_ms<=0 or external_ms<0 or (display_fps is not None and display_fps<=0):
        raise ValueError('Invalid latency/presentation parameters')
    present=[]
    for r in rows:
        t=r['result']+external_ms/1000
        if display_fps is not None:t=math.ceil(t*display_fps-1e-10)/display_fps
        present.append(t)
    latency=[(t-r['captured'])*1000 for r,t in zip(rows,present)]
    timely=sum(x<=deadline_ms+1e-8 for x in latency)
    source_seconds=total_frames/fps
    inter=[(b-a)*1000 for a,b in zip(present,present[1:])]
    # Age of the most recently completed, correctly paired frame on 30 Hz
    # input-time ticks. No extrapolation, no old mask on a different RGB frame.
    held=[];absent=0;j=-1
    for k in range(total_frames):
        tick=k/fps
        while j+1<len(present) and present[j+1]<=tick+1e-12:j+=1
        if j<0:absent+=1
        else:held.append((tick-rows[j]['captured'])*1000)
    return dict(**counts,input_frames=total_frames,processed_frames=len(rows),
        unprocessed_frames=total_frames-len(rows),deadline_ms=deadline_ms,
        external_budget_ms=external_ms,display_fps=display_fps,
        latency_ms=distribution(latency),deadline_misses=len(rows)-timely,
        deadline_miss_fraction_of_processed=(len(rows)-timely)/len(rows) if rows else None,
        timely_fraction_of_input=timely/total_frames if total_frames else None,
        timely_distinct_results_per_source_second=timely/source_seconds if source_seconds else None,
        processed_results_per_source_second=len(rows)/source_seconds if source_seconds else None,
        all_work_fps=len(rows)/max(r['released'] for r in rows) if rows else None,
        presentation_interval_ms=distribution(inter),
        held_paired_frame_age_ms_at_input_ticks=distribution(held),
        held_age_over_deadline_ticks=sum(x>deadline_ms+1e-8 for x in held),
        absent_result_ticks=absent)
