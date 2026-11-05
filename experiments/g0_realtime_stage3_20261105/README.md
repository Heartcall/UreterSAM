# G0 third runtime experiment: frozen comparison contract

Date: 2026-11-05. `final/` is excluded from reading, execution and changes.
Existing tools/configs/models and the first two experiments remain unchanged.

Candidate: explicit three-term float arithmetic, reused normalized color,
in-place gradient temporaries and exact 8-neighbor anchor reachability via
label lookup. A 4-neighbor bincount alternative was screened but is not bound
into the runtime because its microbenchmark benefit was negligible (~1%).
No new package is installed.
No network, input size, precision, thresholds or state order is changed.

Before full runs, prespecify four arms: v2 serial, v3 serial, v2 pipeline and
v3 pipeline, all on both complete videos (722/1972 frames). All cases run in
fresh processes, sequentially, with the existing benchmark's 12 warmups,
single CPU library threads, synchronized original two-network inference,
deferred audit export and max 2 admitted frames in pipeline mode. Exact
Raw2/fiber probabilities, S2/G0 pixels, recorded G0 decisions, and geometry
within 1e-9 remain mandatory. The previous diagnostic omission policy remains.

Metrics: wall completed-output throughput, admission/decode-to-result mean
and P95 latency, queue wait, network/CPU service times, source/model identity,
forward counts and pixel differences. These are file-replay metrics, excluding
camera/display/encoding and pre-admission backpressure waiting. They do not
demonstrate a live camera application's total delay or accuracy on new data.

Diagnostic subsets and per-primitive microbenchmarks screen mechanisms only;
keep the candidate only if it improves representative full replay while
passing all equivalence checks. Full results use all frames without dropping.

Implementation: `pixels.py`, private bindings `runtime.py`, differential checks
`test_runtime.py`, common benchmark `benchmark.py`. Source identity archives
are captured per run; measurements cannot overlap other experiments.

Completed: 44 tests; 10776 formal frames across eight cases; ten full saved-PNG
comparisons passed, plus a 23-frame post-only profile comparison. See
`report.md` for the Chinese analysis and `results.json` for measured values.
The pipeline gain was inconsistent across the two videos; do not present it
as a guaranteed improvement.

Reproduce from the repository root in a fresh directory:

```powershell
& ./experiments/g0_realtime_stage3_20261105/run_full.ps1 -OutputRoot runs/g0_stage3_repeat
```
