"""Pose-stage extractor: pair contains BOTH visible and occluded objects.

Trains "is X visible from view 2?" Q&A. The per-pair gate already
fired upstream; this extractor returns evidence iff the pair really
has both classes of match (gating on `require_both_classes`).
"""

from __future__ import annotations

from typing import Optional

from datasets.base import Frame
from models.base import ObjectMask
from ..match import Match
from ..pairs import ViewPair
from .base import SkillEvidence


def evidence_cross_occlusion_visibility(
    pair: ViewPair, f_src: Frame, masks_src: list[ObjectMask],
    f_tgt: Frame, masks_tgt: list[ObjectMask], matches: list[Match],
    require_both_classes: bool = True,
) -> Optional[SkillEvidence]:
    visible: list[int] = []
    occluded: list[int] = []
    for mi, m in enumerate(matches):
        (visible if m.visible else occluded).append(mi)
    if require_both_classes and (not visible or not occluded):
        return None
    if not (visible or occluded):
        return None
    return SkillEvidence(
        skill="cross_occlusion_visibility",
        qualifying_matches=visible + occluded,
        meta={
            "visible_match_idx": visible,
            "occluded_match_idx": occluded,
            "n_visible": len(visible),
            "n_occluded": len(occluded),
        },
    )
