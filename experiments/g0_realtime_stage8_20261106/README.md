# v7 bounded serial/pipeline scheduling experiment

Both arms import the exact verified v7 Runtime; only scheduling changes. No
custom DLL, package/SDK install, model/input/precision/threshold or segmentation
change. Previous experiments and the restored v4 entry stay frozen. `final/`
is excluded from all reads, execution and modification.

One worker runs decoding and the original synchronized dual-network inference.
The main thread runs S2/G0 in exact frame/timestamp order. A two-credit gate
bounds admitted frames, including any frame being processed or waiting. There
is no frame dropping, no parallel state updates and no unbounded backlog.

Compare fresh v7-serial and v7-pipeline processes on both complete videos, with
reverse arm order for video2, twelve warmups, identical audit packing and one
CPU library thread. Report wall throughput, admitted-to-result latency and
queue waits. Waiting before admission, camera capture, display and encoding
are outside this offline replay's latency scope.

Host service intervals verify actual overlap and the admission bound. They
are not GPU-only occupancy or end-to-end camera latency. Validate Raw2,
probability, S2/G0 pixels, geometry, model/source identities and every forward
before interpreting performance.
