"""Comparable serial/pipelined wall throughput and latency, with deferred export."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import os
from pathlib import Path
from queue import Queue, Empty, Full
from threading import Semaphore, Lock, Event
import time
import zipfile

import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_limits, threadpool_info

from tools.fiber_reference_workspace import ROOT, read, sha256
from tools.fiber_photometric_input import infer
from tools.run_fiber_temporal_state_control import construct
from tools.replay_classwise_experts import model_state_hash
from ureter_semis.experiment import seed_training
from experiments.g0_realtime_20261105.fast_stage2 import FastStage2
from experiments.g0_realtime_20261105.fast_geometry import FastGeometry
from experiments.g0_realtime_20261105.benchmark import array_hash, geometry_record, stats, DIAGNOSTIC
from experiments.g0_realtime_stage4_20261105.runtime import Runtime as V4
from .runtime import Runtime


def host_memory():
    class Status(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ('total_phys', 'available_phys', 'total_page', 'available_page', 'total_virtual',
             'available_virtual', 'available_extended_virtual')]
    status = Status(); status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError('Cannot inspect host memory')
    return dict(total_physical_bytes=status.total_phys, available_physical_bytes=status.available_phys)


class V1:
    def __init__(self):
        self.stage = FastStage2(read(ROOT/'configs/weak_blue_evidence_control_20261103.json'),
                               'weak_blue_extension_tip',
                               read(ROOT/'configs/fiber_temporal_observation_search_20261030.json'), True)
        self.g0 = FastGeometry(read(ROOT/'configs/tapered_tangent_cap_no_fallback_20261104.json'))
        self.counters = {}
    def reset(self):
        self.stage.reset(); self.g0.reset()
    def run(self, frame, raw, probability, timestamp, video):
        a = time.perf_counter(); self.stage.run(frame, raw, probability, timestamp, video)
        b = time.perf_counter(); self.g0.run(frame, raw, probability, self.stage.final_mask)
        c = time.perf_counter(); self.timings = dict(s2_ms=(b-a)*1000, g0_ms=(c-b)*1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=('v4-serial', 'v5-serial'), required=True)
    ap.add_argument('--video', choices=('video_1', 'video_2'), required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--diagnostic', action='store_true')
    args = ap.parse_args(); output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    limit = threadpool_limits(1); torch.set_num_threads(1); cv2.setNumThreads(1)
    pre = read(ROOT/'runs/a1_weak_blue_evidence_20261103/preflight.json')
    video = next(x for x in pre['videos'] if x['video_id'] == args.video)
    assert sha256(video['path']) == video['sha256']
    assert all(sha256(x['path']) == x['sha256'] for x in pre['checkpoints'].values())
    source = read(ROOT/f'runs/a1_weak_blue_evidence_20261103/source_{args.video}.json')['frames']
    history = read(ROOT/f'runs/a1_tapered_tangent_cap_20261104/comparison/{args.video}_sync.json')
    selected = DIAGNOSTIC[args.video] if args.diagnostic else set(range(video['frames']))
    memory = host_memory(); expected_bytes = len(selected)*400*400*2
    if memory['available_physical_bytes'] < 2*expected_bytes+512*1024**2:
        raise MemoryError('Insufficient free host memory for deferred audit masks')
    seed_training(42); experts = construct(pre)
    model_before = model_state_hash(experts)
    runtime = V4() if args.arm == 'v4-serial' else Runtime()
    source_paths = sorted({p for folder in ('tools', 'ureter_semis', 'experiments/g0_realtime_20261105',
                                            'experiments/g0_realtime_stage2_20261105',
                                            'experiments/g0_realtime_stage3_20261105',
                                            'experiments/g0_realtime_stage4_20261105',
                                            'experiments/g0_realtime_stage5_20261106')
                           for p in (ROOT/folder).glob('*.py')}
                          | set((ROOT/'third_party/Endo-SemiS/src/models').glob('*.py'))
                          | set((ROOT/'configs').glob('*.json'))
                          | set((ROOT/'experiments/g0_realtime_stage5_20261106').glob('*.cpp'))
                          | set((ROOT/'experiments/g0_realtime_stage5_20261106').glob('*.ps1')))
    identities = {p.relative_to(ROOT).as_posix():sha256(p) for p in source_paths}
    counts = [0, 0]; events = {}; handles = []
    # Events cover the actual two top-level model forwards. Synchronization in
    # infer() completes the events before their elapsed times are read.
    for index, model in enumerate((experts.stone.models[0], experts.fiber_model)):
        def before(module, inputs, i=index):
            if i not in events:
                events[i] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            pair = events[i]
            pair[0].record(); counts[i] += 1
        def after(module, inputs, result, i=index):
            events[i][1].record()
        handles += [model.register_forward_pre_hook(before), model.register_forward_hook(after)]

    def network(frame):
        with torch.cuda.device(experts.device):
            a = time.perf_counter(); raw, _, fp, _ = infer(experts, frame, 'raw2')
            b = time.perf_counter()
            timing = dict(network_ms=(b-a)*1000,
                          stone_cuda_forward_ms=events[0][0].elapsed_time(events[0][1]),
                          fiber_cuda_forward_ms=events[1][0].elapsed_time(events[1][1]))
            return raw, fp, timing

    cap = cv2.VideoCapture(video['path']); ok, first = cap.read(); cap.release(); assert ok
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='g0-network')
    for j in range(12):
        raw, fp, _ = executor.submit(network, first).result()
        runtime.run(first, raw, fp, j/video['fps'], 'warmup')
    runtime.reset(); counts[:] = [0, 0]
    torch.cuda.reset_peak_memory_stats()
    identity = dict(arm=args.arm, video=video, diagnostic=args.diagnostic, pid=os.getpid(),
                    sources=identities, checkpoints=pre['checkpoints'], model_state_sha256=model_before,
                    warmup=12, precision='FP32', batch_size=1, max_in_flight_frames=2 if args.arm.endswith('pipeline') else 1,
                    hardware=dict(gpu=torch.cuda.get_device_name(), cpu=os.environ.get('PROCESSOR_IDENTIFIER')),
                    environment=dict(torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__,
                                     torch_threads=torch.get_num_threads(), opencv_threads=cv2.getNumThreads(), pools=threadpool_info()),
                    host_memory=memory, expected_mask_buffer_bytes=expected_bytes,
                    timing_scope='wall throughput includes streaming decode, synchronized network, sequential postprocessing, bounded queue, lightweight audit hashes and RAM mask snapshots; PNG/log export deferred; no display/encode/camera capture')
    dll_path=ROOT/'runs/g0_realtime_stage5_20261106/native_final/g0_kernels.dll'
    build_path=dll_path.parent/'build_identity.json'
    dll_hash=sha256(dll_path)
    identity['native_artifact']=dict(path=str(dll_path),sha256=dll_hash,build=json.loads(build_path.read_text(encoding='utf-8-sig')))
    identity['native_active']=hasattr(runtime, 'backend')
    (output/'identity.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
    with zipfile.ZipFile(output/'source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in identities:
            archive.write(ROOT/name, name)
        archive.write(dll_path, 'native/g0_kernels.dll')
        archive.write(build_path, 'native/build_identity.json')
    queue = Queue(maxsize=2); credits = Semaphore(2)
    lock = Lock(); stop = Event(); in_flight = 0; maximum_in_flight = 0

    def admitted():
        nonlocal in_flight, maximum_in_flight
        with lock:
            in_flight += 1; maximum_in_flight = max(maximum_in_flight, in_flight)
    def completed():
        nonlocal in_flight
        with lock:
            in_flight -= 1
        if producer:
            credits.release()

    def send(item):
        while not stop.is_set():
            try:
                queue.put(item, timeout=.1)
                return True
            except Full:
                pass
        return False

    def produce():
        cap = cv2.VideoCapture(video['path'])
        index = 0
        try:
            while not stop.is_set():
                if not credits.acquire(timeout=.1):
                    continue
                if stop.is_set():
                    credits.release(); break
                captured = time.perf_counter(); ok, frame = cap.read(); decoded = time.perf_counter()
                if not ok:
                    credits.release(); break
                if index not in selected:
                    credits.release(); index += 1; continue
                admitted()
                rgb_hash = array_hash(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                assert rgb_hash == source[index]['rgb_raw_sha256']
                raw, fp, timing = network(frame)
                hashes = dict(source_rgb_sha256=rgb_hash, raw_sha256=array_hash(raw),
                              fiber_probability_sha256=array_hash(fp))
                if not send((index, frame, raw, fp, timing, captured, decoded, time.perf_counter(), hashes)):
                    break
                index += 1
            if not stop.is_set():
                assert index == video['frames']
        except BaseException as error:
            send(error)
        finally:
            cap.release(); send(None)

    records, mask_buffer = [], []
    reasons = Counter(); accepted_caps = 0; history_mismatches = stone_changes = 0
    start_wall = time.perf_counter()
    producer = executor.submit(produce) if args.arm.endswith('pipeline') else None
    cap = None if producer else cv2.VideoCapture(video['path'])
    index = 0
    try:
        while True:
            if producer:
                item = queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                i, frame, raw, fp, timing, captured, decoded, ready, hashes = item
            else:
                captured = time.perf_counter(); ok, frame = cap.read(); decoded = time.perf_counter()
                if not ok:
                    break
                i = index; index += 1
                if i not in selected:
                    continue
                admitted()
                rgb_hash = array_hash(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                assert rgb_hash == source[i]['rgb_raw_sha256']
                raw, fp, timing = executor.submit(network, frame).result()
                hashes = dict(source_rgb_sha256=rgb_hash, raw_sha256=array_hash(raw),
                              fiber_probability_sha256=array_hash(fp))
                ready = time.perf_counter()
            post_start = time.perf_counter()
            runtime.run(frame, raw, fp, source[i]['timestamp_seconds'], args.video)
            done = time.perf_counter()
            timing.update(runtime.timings, post_ms=(done-post_start)*1000,
                          decode_ms=(decoded-captured)*1000, queue_wait_ms=(post_start-ready)*1000,
                          admission_to_result_ms=(done-captured)*1000,
                          service_sum_ms=timing['network_ms']+(done-post_start)*1000)
            s2 = runtime.stage.final_mask.copy(); final = runtime.g0.final_mask.copy()
            s2hash = array_hash(s2); history_mismatches += s2hash != history[i]['old_mask_sha256']
            stone_changes += int(np.count_nonzero((final == 1) != (raw == 1)))
            records.append(dict(frame_index=i, **hashes, s2_sha256=s2hash,
                                final_sha256=array_hash(final), g0=geometry_record(runtime.g0.info), timings=timing))
            mask_buffer.append((i, s2, final))
            reasons[runtime.g0.info['reason']] += 1
            accepted_caps += bool(runtime.stage.outline.get('cap', {}).get('accepted'))
            completed()
            if len(records) % 100 == 0:
                print(json.dumps(dict(arm=args.arm, video=args.video, frames=len(records),
                                      output_fps=len(records)/(time.perf_counter()-start_wall))), flush=True)
    except BaseException:
        stop.set()
        credits.release(); credits.release()
        while True:
            try:
                queue.get_nowait()
            except Empty:
                break
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    finally:
        if cap is not None:
            cap.release()
    end_wall = time.perf_counter()
    if producer:
        producer.result(timeout=30)
    executor.shutdown()
    for handle in handles:
        handle.remove()
    assert len(records) == len(selected) and counts == [len(selected)]*2
    assert in_flight == 0 and maximum_in_flight <= (2 if producer else 1)
    assert model_state_hash(experts) == model_before
    assert all(sha256(ROOT/name) == value for name, value in identities.items())
    assert sha256(dll_path)==dll_hash
    assert stone_changes == 0
    if not args.diagnostic:
        assert history_mismatches == 0
    assert [r['frame_index'] for r in records] == sorted(selected)
    summary = dict(arm=args.arm, video=args.video, frames=len(records), diagnostic=args.diagnostic,
                   output_fps=len(records)/(end_wall-start_wall), timed_wall_seconds=end_wall-start_wall,
                   timings={k:stats([r['timings'][k] for r in records]) for k in records[0]['timings']},
                   network_forwards=counts, source_mismatches=0, historical_s2_mismatches=history_mismatches,
                   stone_pixel_changes=stone_changes, model_state_unchanged=True, source_identity_unchanged=True,
                   maximum_in_flight_frames=maximum_in_flight, reasons=reasons, accepted_s2_caps=accepted_caps,
                   optimization_counters=runtime.counters, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                   audit_mask_bytes=sum(a.nbytes+b.nbytes for _, a, b in mask_buffer))
    summary['native_counts']=getattr(getattr(runtime,'backend',None),'counts',{})
    summary['s2_spatial']=getattr(runtime,'s2_spatial',{})
    summary['spatial_counters'] = getattr(runtime, 'spatial_counters', {})
    grid = getattr(runtime, 'grid', None)
    summary['grid_cache'] = dict(hits=grid.hits, misses=grid.misses, retained_bytes=sum(x.nbytes for x in grid.cache.values())) if grid else None
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    # Audit export happens only after all measurements and state checks.
    (output/'masks').mkdir(); (output/'s2_masks').mkdir()
    for i, s2, final in mask_buffer:
        for folder, mask in [('s2_masks', s2), ('masks', final)]:
            assert cv2.imwrite(str(output/folder/f'{i:06d}.png'), mask, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    (output/'frames.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in records), encoding='utf-8')
    runtime.reset(); limit.restore_original_limits()


if __name__ == '__main__':
    main()
