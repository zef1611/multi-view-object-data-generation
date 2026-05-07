"""Counting — total unique instances of a category across both views.

Picks the category with shared + private instances that lands in the
configured `[min_cat_count, max_cat_count]` range. Identity is the
labeler-supplied canonical (set on each mask after SAM in
`PerceptionCache.get`); falls back to the specific label string when
canonical is missing.
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
    _canonical_label,
    _overlap_ok,
)


def gate_counting(
    pair: ViewPair, f_src: Frame, masks_src: list[ObjectMask],
    f_tgt: Frame, masks_tgt: list[ObjectMask], matches: list[Match],
    cfg: ContentSkillConfig,
) -> Optional[SkillEvidence]:
    if not _overlap_ok(pair, cfg):
        return None
    lo = int(cfg.params.get("min_cat_count", 3))
    hi = int(cfg.params.get("max_cat_count", 15))
    require_shared = bool(cfg.params.get("require_shared", True))
    require_private = bool(cfg.params.get("require_private", True))

    def _idfor(m) -> str:
        return _canonical_label(getattr(m, "canonical", "") or m.label)

    src_lbl = [_idfor(m) for m in masks_src]
    tgt_lbl = [_idfor(m) for m in masks_tgt]
    matched_src = {m.src_mask_idx for m in matches if m.visible}
    matched_tgt = {m.tgt_mask_idx for m in matches
                   if m.visible and m.tgt_mask_idx >= 0}

    categories = {c for c in set(src_lbl) | set(tgt_lbl) if c}
    best: Optional[dict] = None
    for cat in categories:
        src_idx = [i for i, lbl in enumerate(src_lbl) if lbl == cat]
        tgt_idx = [j for j, lbl in enumerate(tgt_lbl) if lbl == cat]
        shared_match_idx = [
            mi for mi, m in enumerate(matches)
            if m.visible and m.tgt_mask_idx >= 0
            and src_lbl[m.src_mask_idx] == cat and tgt_lbl[m.tgt_mask_idx] == cat
        ]
        n_shared = len(shared_match_idx)
        private_src = [i for i in src_idx if i not in matched_src]
        private_tgt = [j for j in tgt_idx if j not in matched_tgt]
        unique_total = n_shared + len(private_src) + len(private_tgt)
        if not (lo <= unique_total <= hi):
            continue
        if require_shared and n_shared == 0:
            continue
        if require_private and not private_src and not private_tgt:
            continue
        # Prefer counts close to lo (easier to annotate); tiebreak on shared.
        score = (-abs(unique_total - lo), n_shared)
        if best is None or score > best["_score"]:
            best = {
                "_score": score,
                "category": cat,
                "unique_total": unique_total,
                "shared_match_idx": shared_match_idx,
                "private_src_idx": private_src,
                "private_tgt_idx": private_tgt,
            }
    if best is None:
        return None
    best.pop("_score")
    return SkillEvidence(
        skill="counting",
        qualifying_matches=best["shared_match_idx"],
        meta=best,
    )
