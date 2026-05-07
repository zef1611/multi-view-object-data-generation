"""Point-level cross-view correspondence.

CrossPoint-378K's original `cross_correspondence` skill: given a marked
point in image 1, find the same physical spot in image 2.

Gates:
  - overlap within configured window;
  - viewpoint shift (rotation OR/AND translation) above per-pair floor;
  - max rotation cap (reject near-opposite views);
  - >= min_visible_matches matches that are visible, on labeled src
    objects with detector score >= min_label_score, with src-mask depth
    coverage >= mask_depth_coverage_min (needed for reliable point-level
    GT).

Occluded matches that pass label/score/depth-coverage filters are kept
in a parallel list so phase 2 can build negative ("which view sees X?")
Q&A against the same src query point.
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
    _mask_depth_coverage,
    _overlap_ok,
    _pair_pose_deltas,
)


def gate_cross_point_correspondence(
    pair: ViewPair, f_src: Frame, masks_src: list[ObjectMask],
    f_tgt: Frame, masks_tgt: list[ObjectMask], matches: list[Match],
    cfg: ContentSkillConfig,
) -> Optional[SkillEvidence]:
    if not _overlap_ok(pair, cfg):
        return None

    t, r = _pair_pose_deltas(f_src, f_tgt)
    min_rot = float(cfg.params.get("min_rot_deg", 0.0))
    min_trans = float(cfg.params.get("min_trans_m", 0.0))
    max_rot = float(cfg.params.get("max_rot_deg", float("inf")))
    mode = str(cfg.params.get("viewpoint_shift_mode", "or")).lower()
    if r > max_rot:
        return None
    if mode == "and":
        if not (r >= min_rot and t >= min_trans):
            return None
    else:
        if not (r >= min_rot or t >= min_trans):
            return None

    min_n = int(cfg.params.get("min_visible_matches", 1))
    min_score = float(cfg.params.get("min_label_score", 0.0))
    min_cov = float(cfg.params.get("mask_depth_coverage_min", 0.0))

    qualifying: list[int] = []
    details: list[dict] = []
    occluded: list[int] = []
    occluded_details: list[dict] = []
    for mi, m in enumerate(matches):
        src_mask = masks_src[m.src_mask_idx]
        if not (src_mask.label or "").strip():
            continue
        score = float(getattr(src_mask, "score", 1.0))
        if score < min_score:
            continue
        if min_cov > 0.0:
            cov = _mask_depth_coverage(src_mask.mask, f_src.depth,
                                       f_src.image_size, f_src.depth_size)
            if cov < min_cov:
                continue
        row = {
            "match_idx": mi,
            "src_label": src_mask.label,
            "tgt_label": (masks_tgt[m.tgt_mask_idx].label
                          if m.tgt_mask_idx >= 0 else ""),
            "score": score,
        }
        target_idx, target_rows = ((qualifying, details) if m.visible
                                    else (occluded, occluded_details))
        target_idx.append(mi)
        target_rows.append(row)
    if len(qualifying) < min_n:
        return None

    return SkillEvidence(
        skill="cross_point_correspondence",
        qualifying_matches=qualifying,
        meta={
            "pair_rotation_deg": r,
            "pair_translation_m": t,
            "n_visible_labeled": len(qualifying),
            "n_occluded_candidates": len(occluded),
            "matches": details,
            "occluded_candidates": occluded,
            "occluded_matches": occluded_details,
        },
    )
