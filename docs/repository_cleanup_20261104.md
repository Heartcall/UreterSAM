# Repository cleanup and remote repair

Date: 2026-11-04

The repository previously tracked 1,146,700 files. Of those, 1,134,578 were
under `runs/`, representing about 6.2 GiB in the current tree and about
5.93 GiB of packed Git objects. The local commit that had not reached the
remote added another 9,226 files, mostly per-frame masks and logs.

The repaired repository treats the following as source-controlled material:

- application, training, evaluation and experiment source code;
- tests and small deterministic fixtures;
- frozen configurations, annotations and dataset manifests;
- research plans, progress reports and compact result summaries;
- the existing small source datasets and vendored reference implementation.

The following are local or reproducible artifacts and are no longer tracked:

- `runs/` experiment trees, checkpoints, masks, layers, videos and caches;
- `downloads/` archives that can be downloaded again;
- intermediate checkpoints and training-resume smoke-test outputs.

Compact evidence for the latest decision was copied to `docs/results/`.
The only test that depended directly on large `runs/` frame logs now uses a
small fixed fixture in `tests/fixtures/weak_blue_stage2/`.

Pre-cleanup identities, retained for audit in this document:

- local head: `fa783f6755a86c86625bbbd88e123f2a28a556d4`
- remote main: `b9a404f4d177b10b9a23d38ef421e69b9152c51a`

The remote `main` branch is intentionally replaced by the clean baseline.
This makes new clones small and prevents generated artifacts from re-entering
normal pushes. Historical experimental conclusions remain in the progress
records even though the bulky generated objects are removed from active Git
history.
