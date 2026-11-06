# Fifth G0 experiment: S2 ROI and native arithmetic (2026-11-06)

**WITHDRAWN by user request on 2026-11-06.** `runtime.Runtime` is now an exact
alias of the v4 NumPy Runtime. The compatibility benchmark redirects the former
v5 arm to v4 and records the real v4 identity. Native loading, compilation,
microbenchmarks and the old full-run entry are disabled. No custom DLL is used.
The text below and the existing run artifacts describe the historical experiment.
Original sources were preserved before withdrawal in
`runs/g0_realtime_stage6_20261106/withdrawn_v5_sources.zip`.
Continue optimization in `experiments/g0_realtime_stage6_20261106/` from v4.

`final/` is excluded from reading, execution and changes. Existing production
and four previous experiments remain frozen. No package or SDK is installed.

Prespecified formal comparison: v4 serial vs v5 serial on both complete videos
(722/1972 frames), fresh process per case, reverse within-pair order for video
2, original two FP32 forwards, 12 warmups, single CPU library threads, no frame
dropping, and deferred PNG/log export. No pipeline benefit is inferred.

Candidate: conservative S2 coordinate-domain ROI from exact body pixel order
and quantiles; strict native fusion of large color/evidence/projection loops.
Full gradient, full 8-neighbor propagation graph, full cap fringe, original
small probes and all fitter/seed/history rules remain. Keep whole-body pixels
outside the new search domain; do not crop connectivity or distance transforms.

Native arithmetic uses the existing MSVC x64 compiler, /fp:strict, scalar SSE2
sqrt, no CRT/SDK dependency, no DLL entry point or external imports. The module
is experimental Windows x64 code; the v4 reference remains available. Standard
STL compilation failed because Windows SDK/UCRT headers are absent; an initial
noncanonical intrinsic type also failed. Logs are retained in native/ and
native_sse/. The final native_final/ binary and identity are the tested artifact.

Build from repository root with `./experiments/g0_realtime_stage5_20261106/build.ps1`.
Run full replay using `./experiments/g0_realtime_stage5_20261106/run_full.ps1`.
Both reject existing output directories. A new binary requires rerunning
numerical tests; the prebuilt experimental binary is archived with each run.

Require exact Raw2/fiber probability hashes and S2/G0 pixels, identical recorded
G0 decisions and geometry within 1e-9; check historical S2, stone protection,
source/model/binary identities, source frame order and exact forward counts.
Full evidence score arrays, domains, native projections and large color gates
are also checked exactly on deterministic fixtures. Float64 seed parameters,
small or floating RGB probes use the existing reference functions.

FPS uses actual completed-output wall time, including decode, synchronized
network, postprocessing, lightweight audits and RAM mask copies; excludes
bulk export, camera/display/encoding and pre-admission waiting. This is a serial
file replay, not a live camera application's total latency or accuracy study.
