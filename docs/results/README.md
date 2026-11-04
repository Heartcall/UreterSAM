# Versioned result summaries

This directory contains compact, durable result records that are useful for
reviewing decisions without checking generated experiment trees into Git.

Full-resolution masks, frame logs, rendered videos, model checkpoints and
other reproducible outputs belong in the ignored local `runs/` workspace.
Narrative conclusions remain in `docs/progress/`; frozen configurations and
implementations remain in `configs/` and `tools/`.

`2026-11-04-latest/` preserves the last classwise baseline summary, the
converging tangent-cap summaries, and the final RGB-control acceptance and
engineering checks used by the current repository decision.
