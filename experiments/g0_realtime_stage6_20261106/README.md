# Sixth experiment: restore v4, then evaluate NumPy S2 ROI

The Windows x64 DLL experiment is withdrawn at the user's request. The v5
Runtime is an exact v4 alias; custom DLL loading/build entries are disabled.
Old sources and binary run artifacts remain historical evidence only.

This experiment starts from v4 NumPy. It changes only S2 coordinate-domain
construction and the final forward-limit check to absolute-coordinate NumPy
operations. Full RGB evidence, gradient, connectivity, cap fitting and G0
remain v4. Model precision/inputs/thresholds/history/forward counts are frozen.
No custom compiled module, new package or SDK is used. Standard existing
NumPy/OpenCV/PyTorch binaries are unchanged.

`final/` is excluded from reading, execution and modification. First four
experiments remain frozen. Run identities explicitly record NumPy-only status.
Validate intermediate domains, boundary/fallback fixtures, sequential state,
complete video output equivalence and same-round wall throughput before
recommending adoption. A smaller ROI alone does not prove a useful speedup.
