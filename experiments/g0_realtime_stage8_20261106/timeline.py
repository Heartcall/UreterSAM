"""Validate admitted-frame bounds and measure observed service overlap."""
import math


def analyze(records, capacity):
    if capacity < 1:
        raise ValueError('Capacity must be positive')
    indices = [r['frame_index'] for r in records]
    if indices != sorted(set(indices)):
        raise ValueError('Frame records must be unique and ordered')
    network, post, events = [], [], []
    for row in records:
        w = row['wall_intervals']
        a = w['admitted']; n0, n1 = w['network']; p0, p1 = w['post']; release = w['released']
        if not all(math.isfinite(x) for x in (a, n0, n1, p0, p1, release)):
            raise ValueError('Non-finite wall timestamps')
        if not a <= n0 <= n1 <= p0 <= p1 <= release:
            raise ValueError('Invalid per-frame execution order')
        network.append((n0, n1)); post.append((p0, p1))
        events.extend(((a, 1), (release, -1)))
    for intervals in (network, post):
        if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
            raise ValueError('A sequential stage overlapped itself')
    active = peak = 0
    for _, change in sorted(events):
        active += change; peak = max(peak, active)
        if active < 0 or active > capacity:
            raise ValueError('Admitted-frame capacity violated')
    if active:
        raise ValueError('Unreleased admitted frames')
    overlap = 0.; i = j = 0
    while i < len(network) and j < len(post):
        n, p = network[i], post[j]
        overlap += max(0., min(n[1], p[1])-max(n[0], p[0]))
        if n[1] <= p[1]: i += 1
        else: j += 1
    nt = sum(b-a for a, b in network); pt = sum(b-a for a, b in post)
    return dict(frames=len(records),capacity=capacity,observed_in_flight_peak=peak,
        network_service_seconds=nt,post_service_seconds=pt,overlap_seconds=overlap,
        fraction_network_service_overlapped=overlap/nt if nt else 0.,
        fraction_post_service_overlapped=overlap/pt if pt else 0.,
        post_serial_order_verified=True,network_serial_order_verified=True,
        scope='Host perf_counter intervals: synchronized network service vs CPU postprocessing; not pure GPU kernel occupancy')
