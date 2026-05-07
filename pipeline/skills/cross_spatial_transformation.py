"""Pose-stage extractor: at least one matched object whose 2D footprint
really changed. Pose-stage means the per-pair gate already fired in
`pipeline/pairs.py::_assign_tasks`; this extractor only collects the
qualifying matches and per-skill metadata.
"""

from __future__ import annotations

from typing import Optional

from datasets.base import Frame
from models.base import ObjectMask
from ..match import Match
from ..pairs import ViewPair
from .base import SkillEvidence, _bbox_area, _pair_pose_deltas


def evidence_cross_spatial_transformation(
    pair: ViewPair, f_src: Frame, masks_src: list[ObjectMask],
    f_tgt: Frame, masks_tgt: list[ObjectMask], matches: list[Match],
    scale_ratio_excl: tuple[float, float] = (0.6, 1.67),
) -> Optional[SkillEvidence]:
    lo, hi = scale_ratio_excl
    qualifying: list[int] = []
    details: list[dict] = []
    for mi, m in enumerate(matches):
        if not (m.visible and m.tgt_mask_idx >= 0):
            continue
        s = _bbox_area(masks_src[m.src_mask_idx].bbox)
        ta = _bbox_area(masks_tgt[m.tgt_mask_idx].bbox)
        if s <= 0.0 or ta <= 0.0:
            continue
        ratio = ta / s
        if ratio < lo or ratio > hi:
            qualifying.append(mi)
            details.append({
                "match_idx": mi,
                "label": masks_src[m.src_mask_idx].label,
                "scale_ratio": float(ratio),
            })
    if not qualifying:
        return None
    t, r = _pair_pose_deltas(f_src, f_tgt)
    return SkillEvidence(
        skill="cross_spatial_transformation",
        qualifying_matches=qualifying,
        meta={
            "pair_rotation_deg": r,
            "pair_translation_m": t,
            "transformed_objects": details,
        },
    )
