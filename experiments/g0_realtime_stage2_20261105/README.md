# G0 second runtime experiment: contract frozen before full runs

`final/` is excluded from all reading, execution and modification. Original
sources and the first runtime experiment remain unchanged.

This experiment examines current residual bottlenecks, then compares three
prespecified arms on both complete videos: previous optimized runtime (v1),
memoized/lazy runtime (v2), and the same v2 with at most one frame ahead of the
stateful CPU postprocessor. Models, FP32, input preprocessing, thresholds,
geometry and state update order are fixed. No frames may be dropped.

Functional comparisons require exact Raw2/fiber probability hashes, exact S2
and final G0 pixels, identical availability/reasons/tip source, and geometry
within absolute 1e-9 px. Display-only rejected-candidate counters/layers may be
omitted and must be identified as such. Reuse of S2 seeds in G0 is disabled for
history-qualified weak seeds. No result from another frame may be reused.

Warm up each actual execution thread 12 times and reset state. Run each arm
and video in a fresh process, sequentially. Measure actual completed-output
throughput and per-frame admission/decode-to-result latency separately from
network/postprocessing service time and queue wait. Buffer output masks and
audit records in bounded-by-video memory; write PNGs only after timing, so
PNG/log exports do not create gaps inside either serial or pipelined timing.
Check host memory before choosing this audit mechanism. Camera capture,
display and encoding remain outside the benchmark.

Diagnostic frame subsets and profilers are for mechanism screening only.
Full results must include all 722/1972 frames, including unavailable G0 frames.
Do not compare these new wall-throughput values directly with the first
experiment's sum-of-service-time FPS as if their timing scopes were identical.

The current CPU profile is saved under
`runs/g0_realtime_stage2_20261105/profile_current/`.

Implementation: `runtime.py`; common benchmark: `benchmark.py`; full replay and
independent PNG comparisons: `run_full.ps1`; differential checks:
`test_runtime.py`. The Chinese analysis and limitations are in `report.md`.

From the repository root, reproduce in a fresh output directory:

```powershell
& ./experiments/g0_realtime_stage2_20261105/run_full.ps1 -OutputRoot runs/g0_stage2_repeat
```

The benchmark intentionally includes all decoded frames and both network
forwards. Its output is a full replay and audit artifact, not a camera or
display application. No existing inference entry point is changed.
