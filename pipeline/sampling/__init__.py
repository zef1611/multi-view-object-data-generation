"""Frame-sampling strategies — pluggable via `--sampling <name>`.

Each strategy file under this package owns one algorithm
(`adaptive.py`, `stride.py`, `cosmic.py`). The dispatcher in
`sample_keyframes()` picks the right one and returns
`(sampled_frame_ids, mode_description)`.

Adding a new strategy = drop a `<name>.py` here, add it to `SAMPLERS`,
and `--sampling <name>` works automatically.
"""

from __future__ import annotations

import logging
from typing import Optional

from .adaptive import select_keyframes_adaptive
from .stride import select_keyframes_stride

logger = logging.getLogger(__name__)


SAMPLERS = {
    "adaptive": select_keyframes_adaptive,
    "stride": select_keyframes_stride,
}


def sample_keyframes(
    adapter,
    *,
    sampling: str = "adaptive",
    frame_stride: int = 50,
    min_keyframes: int = 30,
    min_translation_m: float = 0.40,
    min_rotation_deg: float = 25.0,
    limit_frames: Optional[int] = None,
    cosmic_base_sampling: str = "stride",
    cosmic_union_coverage_min: float = 0.6,
    cosmic_yaw_diff_min_deg: float = 30.0,
    log: bool = True,
) -> tuple[list[str], str]:
    """Pure-sampling step extracted from `select_pairs`.

    Used both by `select_pairs` and by `cli/generate.py` for pre-flight
    cache checks (so we know which frames need filter/labeler verdicts
    before deciding whether to launch a vLLM server).
    """
    all_frames = adapter.list_frames()
    base_sampling = sampling if sampling != "cosmic" else cosmic_base_sampling
    if base_sampling == "adaptive":
        sampled = select_keyframes_adaptive(
            adapter, min_translation_m=min_translation_m,
            min_rotation_deg=min_rotation_deg,
        )
        mode = f"adaptive(d>={min_translation_m}m or r>={min_rotation_deg}°)"
    elif base_sampling == "stride":
        sampled, effective_stride = select_keyframes_stride(
            all_frames, frame_stride=frame_stride, min_keyframes=min_keyframes,
        )
        mode = (f"stride={effective_stride}" if effective_stride == frame_stride
                else f"stride={effective_stride} (floored from {frame_stride}, "
                     f"min_keyframes={min_keyframes})")
    else:
        raise ValueError(f"unknown sampling '{sampling}' (base={base_sampling})")
    if sampling == "cosmic":
        mode = (f"cosmic(base={mode}, alpha>={cosmic_union_coverage_min}, "
                f"yaw>={cosmic_yaw_diff_min_deg}°)")
    if limit_frames is not None and limit_frames > 0:
        sampled = sampled[:limit_frames]
    if log:
        logger.info("[%s] %d frames -> %d after %s (avg 1:%.1f)",
                    adapter.scene_id, len(all_frames), len(sampled), mode,
                    len(all_frames) / max(len(sampled), 1))
    return sampled, mode


__all__ = [
    "sample_keyframes",
    "select_keyframes_adaptive",
    "select_keyframes_stride",
    "SAMPLERS",
]
