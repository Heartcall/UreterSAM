"""Profile/screen/full-video benchmark, with exports outside measured inference."""
from __future__ import annotations

import argparse
from collections import Counter
import cProfile
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import pstats
import time

import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from tools.fiber_reference_workspace import ROOT, read, sha256
from tools.fiber_photometric_input import infer
from tools.fiber_tapered_tangent_cap import TaperedTangentCap
from tools.fiber_weak_stage2 import GrowthPipeline
from tools.run_fiber_temporal_state_control import construct
from tools.replay_classwise_experts import model_state_hash
from ureter_semis.experiment import seed_training
from .fast_stage2 import FastStage2
from .fast_geometry import FastGeometry

CONFIG = ROOT / 'configs/tapered_tangent_cap_no_fallback_20261104.json'
DIAGNOSTIC = {
    'video_1': {0, 1, 100, 200, 316, 366, 367, 402, 500, 600, 662, 698, 720, 721},
    'video_2': {0, 1, 100, 316, 500, 769, 1000, 1203, 1208, 1232, 1249, 1265,
                1290, 1400, 1500, 1683, 1702, 1715, 1759, 1777, 1822, 1900, 1971},
}


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def stats(values):
    a = np.asarray(values, float)
    return dict(mean=float(a.mean()), p50=float(np.median(a)),
                p95=float(np.quantile(a, .95)), p99=float(np.quantile(a, .99)),
                maximum=float(a.max()))


def geometry_record(info):
    g, t, c = (info.get(k) for k in ('geometry', 'tip', 'cap'))
    return dict(applied=info['applied'], status=info['status'], reason=info['reason'],
                tip_source=t.get('source_pixel') if t else None,
                T=t.get('vertex') if t else None,
                axis=g['axis'] if g else None,
                lines=[dict(point=z['point'], direction=z['direction'])
                       for z in g['independent_lines']] if g else None,
                C=c['center'] if c else None, radius=c['radius'] if c else None,
                stone_clipped_pixels=info.get('stone_clipped_pixels', 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=('reference', 'fast-cap', 'fast-cap-single', 'fast-runtime'), required=True)
    ap.add_argument('--video', choices=('video_1', 'video_2'), required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--diagnostic', action='store_true')
    ap.add_argument('--profile', action='store_true')
    args = ap.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output/'masks').mkdir(); (output/'s2_masks').mkdir()
    limiter = None
    if args.variant.endswith('single') or args.variant == 'fast-runtime':
        limiter = threadpool_limits(limits=1)
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
    preflight = read(ROOT/'runs/a1_weak_blue_evidence_20261103/preflight.json')
    video = next(v for v in preflight['videos'] if v['video_id'] == args.video)
    assert sha256(video['path']) == video['sha256']
    for cp in preflight['checkpoints'].values():
        assert sha256(cp['path']) == cp['sha256']
    source = read(ROOT/f'runs/a1_weak_blue_evidence_20261103/source_{args.video}.json')['frames']
    expected_s2 = read(ROOT/f'runs/a1_tapered_tangent_cap_20261104/comparison/{args.video}_sync.json')
    seed_training(42)
    experts = construct(preflight)
    model_identity = model_state_hash(experts)
    stage_cls = GrowthPipeline if args.variant == 'reference' else FastStage2
    stage = stage_cls(read(ROOT/'configs/weak_blue_evidence_control_20261103.json'),
                      'weak_blue_extension_tip',
                      read(ROOT/'configs/fiber_temporal_observation_search_20261030.json'), True)
    geometry_cls = FastGeometry if args.variant == 'fast-runtime' else TaperedTangentCap
    geometry = geometry_cls(read(CONFIG))
    source_paths = sorted({p for folder in ('tools', 'ureter_semis', 'experiments/g0_realtime_20261105')
                           for p in (ROOT/folder).glob('*.py')}
                          | set((ROOT/'third_party/Endo-SemiS/src/models').glob('*.py'))
                          | set((ROOT/'configs').glob('*.json')))
    source_identities = {p.relative_to(ROOT).as_posix():sha256(p) for p in source_paths}
    identity = dict(variant=args.variant, diagnostic=args.diagnostic, profiled=args.profile, pid=os.getpid(),
                    video=video, config_sha256=sha256(CONFIG), checkpoints=preflight['checkpoints'],
                    hardware=dict(cpu=os.environ.get('PROCESSOR_IDENTIFIER'),
                                  gpu=torch.cuda.get_device_name(), cpu_count=os.cpu_count()),
                    environment=dict(python=platform.python_version(), numpy=np.__version__,
                                     opencv=cv2.__version__, torch=torch.__version__,
                                     torch_threads=torch.get_num_threads(),
                                     opencv_threads=cv2.getNumThreads(), pools=threadpool_info()),
                    sources=source_identities, model_state_sha256=model_identity,
                    warmup=12, precision='FP32', batch_size=1,
                    scope='synchronized complete Raw2 + sequential S2 + G0; no render, encode, file export or audit hashing in core/decode timings')
    (output/'identity.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
    cap = cv2.VideoCapture(video['path']); ok, first = cap.read(); cap.release()
    assert ok
    for j in range(12):
        raw, _, fp, _ = infer(experts, first, 'raw2')
        stage.run(first, raw, fp, j/video['fps'], 'warmup')
        geometry.run(first, raw, fp, stage.final_mask)
    stage.reset(); geometry.reset()
    torch.cuda.reset_peak_memory_stats()
    cap = cv2.VideoCapture(video['path'])
    timings, reasons, accepted_caps = [], Counter(), 0
    historical_s2_mismatches, stone_changes, source_mismatches = 0, 0, 0
    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()
    wall = time.perf_counter()
    index = 0
    with (output/'frames.jsonl').open('w', encoding='utf-8') as log:
        while True:
            begin = time.perf_counter(); ok, frame = cap.read(); decoded = time.perf_counter()
            if not ok:
                break
            if args.diagnostic and index not in DIAGNOSTIC[args.video]:
                index += 1
                continue
            # Audit outside timing. In diagnostic mode only selected frames update
            # history; this is explicitly not a complete sequential accuracy run.
            rgb_hash = array_hash(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            source_mismatches += rgb_hash != source[index]['rgb_raw_sha256']
            torch.cuda.synchronize(); start = time.perf_counter()
            raw, _, fp, _ = infer(experts, frame, 'raw2'); after_net = time.perf_counter()
            stage.run(frame, raw, fp, source[index]['timestamp_seconds'], args.video)
            after_stage = time.perf_counter()
            geometry.run(frame, raw, fp, stage.final_mask)
            torch.cuda.synchronize(); end = time.perf_counter()
            timing = dict(network_ms=(after_net-start)*1000,
                          s2_ms=(after_stage-after_net)*1000,
                          g0_ms=(end-after_stage)*1000,
                          core_ms=(end-start)*1000,
                          decode_ms=(decoded-begin)*1000,
                          decode_core_ms=((decoded-begin)+(end-start))*1000)
            timings.append(timing)
            s2hash = array_hash(stage.final_mask)
            historical_s2_mismatches += s2hash != expected_s2[index]['old_mask_sha256']
            stone_changes += int(np.count_nonzero((geometry.final_mask == 1) != (raw == 1)))
            reasons[geometry.info['reason']] += 1
            accepted_caps += bool(stage.outline.get('cap', {}).get('accepted'))
            for folder, mask in [('masks', geometry.final_mask), ('s2_masks', stage.final_mask)]:
                assert cv2.imwrite(str(output/folder/f'{index:06d}.png'), mask,
                                   [cv2.IMWRITE_PNG_COMPRESSION, 1])
            log.write(json.dumps(dict(frame_index=index, source_rgb_sha256=rgb_hash,
                                      raw_sha256=array_hash(raw), fiber_probability_sha256=array_hash(fp),
                                      s2_sha256=s2hash, final_sha256=array_hash(geometry.final_mask),
                                      g0=geometry_record(geometry.info), timings=timing), allow_nan=False)+'\n')
            index += 1
            if len(timings) % 100 == 0:
                print(json.dumps(dict(variant=args.variant, video=args.video, frames=len(timings),
                                      core_fps=1000/np.mean([x['core_ms'] for x in timings]))), flush=True)
    cap.release()
    if profiler:
        profiler.disable(); profiler.dump_stats(str(output/'cpu_profile.pstats'))
        buf = io.StringIO(); pstats.Stats(profiler, stream=buf).sort_stats('cumtime').print_stats(60)
        (output/'cpu_profile.txt').write_text(buf.getvalue(), encoding='utf-8')
    expected = len(DIAGNOSTIC[args.video]) if args.diagnostic else video['frames']
    assert index == video['frames'] and len(timings) == expected
    wall_seconds = time.perf_counter()-wall
    assert model_state_hash(experts) == model_identity, 'Model state changed'
    assert all(sha256(ROOT/k) == h for k, h in source_identities.items()), 'Source changed during benchmark'
    summary = dict(variant=args.variant, video=args.video, diagnostic=args.diagnostic, profiled=args.profile,
                   frames=len(timings), decoded_frames=index,
                   timings={k:stats([x[k] for x in timings]) for k in timings[0]},
                   core_fps=1000/np.mean([x['core_ms'] for x in timings]),
                   decode_core_fps=1000/np.mean([x['decode_core_ms'] for x in timings]),
                   wall_seconds=wall_seconds,
                   export_inclusive_fps=len(timings)/wall_seconds,
                   source_mismatches=source_mismatches,
                   historical_s2_mismatches=historical_s2_mismatches,
                   stone_pixel_changes=stone_changes, accepted_s2_caps=accepted_caps,
                   model_state_unchanged=True, source_identity_unchanged=True,
                   reasons=reasons, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    assert source_mismatches == stone_changes == 0
    if not args.diagnostic:
        assert historical_s2_mismatches == 0, 'Historical sequential S2 output changed'
    if limiter:
        limiter.restore_original_limits()


if __name__ == '__main__':
    main()
