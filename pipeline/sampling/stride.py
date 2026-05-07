"""Stride-based keyframe sampling.

Take every Nth frame, with a `min_keyframes` floor — if the requested
stride would yield fewer than `min_keyframes` frames, the effective
stride drops to `len(frames) // min_keyframes`. Reproducible (no
pose-threshold sensitivity), used for smoke tests and as the COSMIC
base sampler.
"""

from __future__ import annotations


def select_keyframes_stride(
    all_frames: list[str],
    *,
    frame_stride: int = 50,
    min_keyframes: int = 30,
) -> tuple[list[str], int]:
    """Return `(sampled_ids, effective_stride)`.

    The caller logs the effective stride (for the floored case) so the
    helper stays free of logging side-effects.
    """
    if min_keyframes > 0 and len(all_frames) > 0:
        implied_stride = max(1, len(all_frames) // min_keyframes)
        effective_stride = min(frame_stride, implied_stride)
    else:
        effective_stride = frame_stride
    sampled = all_frames[::effective_stride]
    return sampled, effective_stride
