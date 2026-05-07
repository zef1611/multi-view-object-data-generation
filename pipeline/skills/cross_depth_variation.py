"""Pose-stage extractor: same object at noticeably different depths
between views. Pose-stage means the per-pair gate has already fired;
this extractor picks the qualifying matches.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from datasets.base import Frame
from models.base import ObjectMask
from ..match import Match
from ..pairs import ViewPair
from .base import SkillEvidence


def evidence_cross_depth_variation(
    pair: ViewPair, f_src: Frame, masks_src: list[ObjectMask],
    f_tgt: Frame, masks_tgt: list[ObjectMask], matches: list[Match],
    min_point_delta_m: float = 0.5,
) -> Optional[SkillEvidence]:
    qualifying: list[int] = []
    details: list[dict] = []
    deltas: list[float] = []
    for mi, m in enumerate(matches):
        if not m.visible:
            continue
        dz = float(m.depth_src - m.depth_obs_tgt)
        deltas.append(abs(dz))
        if abs(dz) >= min_point_delta_m:
            qualifying.append(mi)
            details.append({
                "match_idx": mi,
                "label": masks_src[m.src_mask_idx].label,
                "depth_src": round(m.depth_src, 4),
                "depth_tgt": round(m.depth_obs_tgt, 4),
                "delta_m": round(dz, 4),
            })
    if not qualifying:
        return None
    return SkillEvidence(
        skill="cross_depth_variation",
        qualifying_matches=qualifying,
        meta={
            "pair_median_delta_m": round(float(np.median(deltas)), 4)
                if deltas else 0.0,
            "varying_objects": details,
        },
    )
