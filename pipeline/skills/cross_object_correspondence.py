"""Frame-level lift of `cross_point_correspondence`.

Task: point to a target-side object that also appears in the src view.
No query point is marked in view 1 — the model must identify a shared
object on its own. Shares the viewpoint-shift / overlap gates with the
point-level task, drops src-side depth coverage (no pixel-exact GT
needed), and adds a tgt-mask-area floor so the answer point isn't
ambiguous.
"""

from __future__ import annotations

from typing import Optional

from datasets.base import Frame
from models.base import ObjectMask
from ..match import Match
from ..pairs import ViewPair
from .base import ContentSkillConfig, SkillEvidence, _overlap_ok, _pair_pose_deltas


def gate_cross_object_correspondence(
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
    min_area_frac = float(cfg.params.get("min_tgt_mask_area_frac", 0.0))

    W_t, H_t = f_tgt.image_size
    img_area = float(W_t * H_t) if W_t and H_t else 0.0

    qualifying: list[int] = []
    details: list[dict] = []
    for mi, m in enumerate(matches):
        if not (m.visible and m.tgt_mask_idx >= 0):
            continue
        tgt_mask = masks_tgt[m.tgt_mask_idx]
        label = (tgt_mask.label or "").strip()
        if not label:
            continue
        if float(getattr(tgt_mask, "score", 1.0)) < min_score:
            continue
        if min_area_frac > 0.0 and img_area > 0.0:
            area = float(tgt_mask.mask.sum())
            if area / img_area < min_area_frac:
                continue
        u, v = m.p_tgt
        if not (0 <= u < W_t and 0 <= v < H_t):
            continue
        qualifying.append(mi)
        details.append({
            "match_idx": mi,
            "tgt_label": label,
            "src_label": masks_src[m.src_mask_idx].label,
            "point_tgt": [int(round(u)), int(round(v))],
            "tgt_mask_area_frac": round(
                float(tgt_mask.mask.sum()) / img_area, 5
            ) if img_area > 0.0 else 0.0,
            "score": float(getattr(tgt_mask, "score", 1.0)),
        })
    if len(qualifying) < min_n:
        return None

    return SkillEvidence(
        skill="cross_object_correspondence",
        qualifying_matches=qualifying,
        meta={
            "pair_rotation_deg": r,
            "pair_translation_m": t,
            "n_shared_objects": len(qualifying),
            "shared_objects": details,
        },
    )
