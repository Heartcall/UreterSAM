# Independent G0 latency experiment

This experiment excludes `final/` completely. It imports the existing repository
implementation and uses the current no-fallback G0 configuration. Existing source,
weights, thresholds, inputs, historical results and default entry points remain unchanged.

## Contract and measurements

- Compare the complete Raw2 -> sequential S2 -> latest G0 path, FP32, batch 1.
- Profile before optimizing. Diagnostic frame subsets are not full-video results.
- Optimize unnecessary CPU work first. Do not remove the S2 stage or its accepted
  cap edits, change input size, reduce precision, skip frames, or alter thresholds.
- Retain a candidate only after checking every output pixel, G0 availability/reason,
  selected tip source, T, circle and side geometry against the reference.
- Run reference and candidate in separate processes, sequentially, on both full
  videos (722 and 1972 frames), with 12 warmup calls and state reset afterwards.
- Report synchronized network/S2/G0/core and decode-inclusive latency separately;
  diagnostic serialization and mask export are outside core timings. No video
  rendering is included in the inference benchmark.
- Save frame identities, timings, masks, mismatch counts, source/config/model
  identities and environment details under a new `runs/g0_realtime_20261105/` tree.
- These are engineering equivalence/speed experiments, not new accuracy claims.

Optimization workflow informed by the installed `optimize-for-gpu` skill:
`C:/Users/1/.codex/skills/optimize-for-gpu/SKILL.md`.

Implementation, commands and measured results will be documented in `report.md`.
