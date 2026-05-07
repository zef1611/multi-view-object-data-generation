"""Anchor skill — at least one shared object with a non-trivial scale change.

Trains "this object in image 1 is the same physical thing as that object
in image 2, despite changing apparent size." Gates require both views
to be reasonably overlapping, the camera to have moved meaningfully,
and at least one matched object whose 2D footprint ratio is outside
`scale_ratio_excl`.
"""

from __future__ import annotations

from typing import Optional

from datasets.base import Frame
from models.base import ObjectMask
from ..match import Match
from ..pairs import ViewPair
from .base import (
    ContentSkillConfig,
    SkillEvidence,
    _bbox_area,
    _overlap_ok,
    _pair_pose_deltas,
)


def gate_anchor(
    pair: ViewPair, f_src: Frame, masks_src: list[ObjectMask],
    f_tgt: Frame, masks_tgt: list[ObjectMask], matches: list[Match],
    cfg: ContentSkillConfig,
) -> Optional[SkillEvidence]:
    if not _overlap_ok(pair, cfg):
        return None
    t, r = _pair_pose_deltas(f_src, f_tgt)
    if (t < float(cfg.params.get("min_trans_m", 0.0))
            and r < float(cfg.params.get("min_rot_deg", 0.0))):
        return None

    lo, hi = cfg.params.get("scale_ratio_excl", [0.0, float("inf")])
    min_n = int(cfg.params.get("min_visible_matches", 1))
    shared: list[int] = []
    details: list[dict] = []
    for mi, m in enumerate(matches):
        if not (m.visible and m.tgt_mask_idx >= 0):
            continue
        s = _bbox_area(masks_src[m.src_mask_idx].bbox)
        ta = _bbox_area(masks_tgt[m.tgt_mask_idx].bbox)
        if s <= 0.0 or ta <= 0.0:
            continue
        ratio = ta / s
        non_trivial = ratio < lo or ratio > hi
        shared.append(mi)
        details.append({
            "match_idx": mi,
            "src_label": masks_src[m.src_mask_idx].label,
            "tgt_label": masks_tgt[m.tgt_mask_idx].label,
            "scale_ratio": float(ratio),
            "non_trivial": bool(non_trivial),
        })
    if len(shared) < min_n:
        return None
    if not any(d["non_trivial"] for d in details):
        return None

    return SkillEvidence(
        skill="anchor",
        qualifying_matches=shared,
        meta={
            "n_shared": len(shared),
            "pair_rotation_deg": r,
            "pair_translation_m": t,
            "shared_objects": details,
        },
    )
