# G0 exact component and profile optimization experiment

Target: real endoscope input at 30 FPS and capture-to-presentation latency below
100 ms. This experiment measures offline service cost and preserves all frames;
it does not validate camera or display latency.

The reference is the verified v7 NumPy runtime. Two independent changes:

- Reuse the qualified seed's outer 4-connected label instead of constructing
  an isolated full mask and labeling it again. Extract row-major coordinates
  inside the exact component bounding box, restoring absolute coordinates.
- Batch the first per-bin minimum/maximum selection in grouped_profiles.
  Keep stable bin ordering, original first-tie selection and the original
  per-bin np.mean input order and reduction.

Models, FP32, batch 1, thresholds, fits, evidence graphs, temporal state,
rejection rules, timestamps and G0 stay unchanged. The withdrawn DLL and the
stage9 region-index candidate are not enabled. Existing sources, restored v4
entry and all files under final remain isolated and unchanged.

screen.py compares baseline/component-only/component-ROI. screen_profiles.py
compares baseline/profile-only/combined. Each uses the same 23 neural inputs
and four rotating/reversing CPU rounds. Full profiling uses a separate process
after the timed comparisons. All benchmark source files are frozen and archived.

Run run_full_profiles.ps1 with a new OutputRoot to compare fresh v7-pipeline
and combined v10-pipeline processes on both complete videos. Each performs
12 warmups then resets, one CPU library thread and at most two admitted frames.
Independent PNG comparisons check the candidate against the same-round baseline
and the baseline against stage8. Read report.md for results and limitations.
