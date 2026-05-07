"""Relative distance — reference object + ordered candidates with margin.

Tries every visible match (with depth-reliable src mask) as the reference
and returns the first reference whose farthest candidate beats the
runner-up by `min_margin_m`. The clean-margin requirement is what
prevents the question from being ambiguous in 3D.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from datasets.base import Frame
from models.base import ObjectMask
from ..match import Match
from ..pairs import ViewPair
from .base import (
    ContentSkillConfig,
    SkillEvidence,
    _mask_depth_coverage,
    _overlap_ok,
)


def gate_relative_distance(
    pair: ViewPair, f_src: Frame, masks_src: list[ObjectMask],
    f_tgt: Frame, masks_tgt: list[ObjectMask], matches: list[Match],
    cfg: ContentSkillConfig,
) -> Optional[SkillEvidence]:
    if not _overlap_ok(pair, cfg):
        return None
    min_objects = int(cfg.params.get("min_objects", 3))
    min_margin = float(cfg.params.get("min_margin_m", 0.5))
    min_cov = float(cfg.params.get("mask_depth_coverage_min", 0.6))

    candidates: list[tuple[int, np.ndarray]] = []
    for mi, m in enumerate(matches):
        if not m.visible:
            continue
        mask = masks_src[m.src_mask_idx].mask
        cov = _mask_depth_coverage(mask, f_src.depth,
                                   f_src.image_size, f_src.depth_size)
        if cov < min_cov:
            continue
        candidates.append((mi, np.asarray(m.X_world, dtype=float)))

    if len(candidates) < min_objects:
        return None

    for ref_pos, (ref_mi, ref_X) in enumerate(candidates):
        rows = []
        for j, (mi, X) in enumerate(candidates):
            if j == ref_pos:
                continue
            rows.append((mi, float(np.linalg.norm(X - ref_X))))
        rows.sort(key=lambda t: t[1], reverse=True)
        if len(rows) < 2:
            continue
        margin = rows[0][1] - rows[1][1]
        if margin < min_margin:
            continue
        qualifying = [ref_mi] + [mi for mi, _ in rows]
        return SkillEvidence(
            skill="relative_distance",
            qualifying_matches=qualifying,
            meta={
                "reference_match_idx": ref_mi,
                "reference_label": masks_src[matches[ref_mi].src_mask_idx].label,
                "candidates": [
                    {
                        "match_idx": mi,
                        "label": masks_src[matches[mi].src_mask_idx].label,
                        "distance_m": round(d, 4),
                    } for mi, d in rows
                ],
                "farthest_match_idx": rows[0][0],
                "margin_m": round(margin, 4),
            },
        )
    return None
