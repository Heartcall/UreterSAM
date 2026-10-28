"""Read exact decode-order indices without timestamp-dependent random seeking."""
from __future__ import annotations


def read_forward(capture, next_index: int, requested_index: int):
    if requested_index < next_index:
        raise ValueError("Sequential video reader cannot revisit or overlap earlier frames")
    while next_index <= requested_index:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Unexpected end of video at decode-order frame {next_index}")
        next_index += 1
    return frame, next_index
