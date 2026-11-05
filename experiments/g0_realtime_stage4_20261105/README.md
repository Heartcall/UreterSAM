# G0 fourth runtime experiment: immutable grids and bounded tip search

`final/` is excluded from reads, execution and changes. Original sources and
the first three experiments stay frozen. No new dependency, model, precision,
threshold, input resolution, state update order or frame dropping is allowed.

Candidate: instance-owned read-only absolute coordinate cache (max four shapes
or dtypes) used in S2 grow/expand/cap and G0 tip; complete conservative G0 tip
ROI from the full axial interval and maximum taper half-width, with actual
basis inversion, two-pixel guard, and real one-pixel halo for the 3x3 count.
Do not truncate at the line intersection. Ill-conditioned/nonfinite bounds
fall back to the original full search. Full global blue identity is retained.
No gradient or connectivity cropping is introduced.

Prespecified full comparison: v3 serial vs v4 serial on all 722/1972 source
frames, fresh process per case, single-thread CPU libraries, original two FP32
network forwards, 12 warmups then state/statistics reset, deferred audit PNG
export. Reverse new/old run order for video 2. Grid-only ablation is a mechanism
screen, not a formal throughput claim. Pipeline benefit is not inferred from
this serial experiment.

Require exact Raw2/fiber probability hashes and S2/G0 pixels, identical recorded
G0 decisions and geometry within 1e-9, exact historical S2, unchanged models,
sources and stone pixels, complete frames and forward counts. Keep the previous
diagnostic policy. Validate synthetic full domain/partial-eligibility maps,
image borders, no winner, beyond-intersection and bad-basis fallback cases.

Throughput uses actual wall time with decode, synchronized network, CPU post,
lightweight audit and RAM snapshots. Latency starts at admission/decode and
ends at available post result. Bulk PNG/log exports, camera/display/encoding,
and pre-admission backpressure waiting remain excluded. No 30 FPS or live-camera
latency claim is implied by the mechanism microbenchmarks.

Completed: 48 tests, 5388 formal frames across four serial cases, four full
saved-PNG comparisons passed, and one 23-frame post-only profile comparison.
See `report.md` and `results.json`. Final S2/G0 masks are unchanged; v4 pipeline
throughput has not been measured. No original entry point is changed.

Reproduce from the repository root in a fresh output directory:

```powershell
& ./experiments/g0_realtime_stage4_20261105/run_full.ps1 -OutputRoot runs/g0_stage4_repeat
```
