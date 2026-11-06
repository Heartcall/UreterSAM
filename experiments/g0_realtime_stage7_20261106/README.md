# NumPy experiment 7: region chroma reuse and sparse coordinate views

Starts from the verified v6 NumPy candidate. No custom DLL, compilation, package
installation or SDK. The restored v4 entry and all previous experiments remain
unchanged. `final/` is never read, run or modified.

Two switches permit separate ablations: current-frame region chroma reuse,
and broadcasting in the initial grow / cap coordinate preparation. Coordinate
views use the same immutable absolute integer values as v4's grid. Only private
instance function globals receive the sparse-grid proxy. Connectivity, distance
transforms, cap fringe, gradients, seed/history rules, small probes, FP32 model
inputs, every-frame dual forwards and G0 fitting are unchanged.

Validate exact intermediate colors and growth layers, numerical domains, masks,
cap decisions and state sequences before complete-video paired measurements.
Screen each switch separately on identical resident diagnostic inputs; retain
only when synchronized complete replay shows a useful improvement.
