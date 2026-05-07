"""COSMIC visibility-set sampling — see ``pipeline/cosmic.py``.

The full module (visibility-set computation + per-pair gate) lives at
the pipeline top level because it's used by both the sampling stage
and downstream skill-gate code (`visualize_pairs.py`, `viz/layer2/gt.py`).
This file re-exports the sampling-relevant bits so a `--sampling cosmic`
caller can stay inside `pipeline.sampling.*`.
"""

from __future__ import annotations

from ..cosmic import (
    COSMIC_SKILLS,
    compute_visibility_set,
    floor_plane_yaw_deg,
    _yaw_diff_deg,
)

__all__ = [
    "COSMIC_SKILLS",
    "compute_visibility_set",
    "floor_plane_yaw_deg",
    "_yaw_diff_deg",
]
