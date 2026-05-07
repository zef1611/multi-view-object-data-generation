"""Shared types and helpers for content / pose skill gates.

Public types:
* `SkillEvidence` — dataclass written into the per-pair manifest.
* `ContentSkillConfig` — per-skill parameter bundle from `tasks.json`.
* `CONTENT_SKILLS` / `POSE_SKILLS` — name tuples used for stage routing.
* `load_content_skills` — parse `tasks.json::content_skills` into configs.

Private helpers (underscore-prefixed) live here so each `<skill>.py`
file can `from .base import _bbox_area, _overlap_ok, ...` without
re-implementing them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from datasets.base import Frame
from ..geometry import pair_pose_deltas
from ..pairs import ViewPair

logger = logging.getLogger(__name__)


CONTENT_SKILLS = (
    "cross_point_correspondence",
    "cross_object_correspondence",
    "anchor",
    "counting",
    "relative_distance",
    "relative_direction",
)
POSE_SKILLS = (
    "cross_spatial_transformation",
    "cross_depth_variation",
    "cross_occlusion_visibility",
)


@dataclass(frozen=True)
class SkillEvidence:
    """Structured qualification payload written to the pair manifest.

    `qualifying_matches`: indices into the pair's `matches` list that the
    skill considers relevant (e.g. for counting, every matched instance
    of the winning category; for rel_distance, the reference + candidates).

    `meta`: skill-specific fields — see each gate for the schema.
    """
    skill: str
    qualifying_matches: list[int] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContentSkillConfig:
    name: str
    overlap: tuple[float, float]
    params: dict[str, Any]


def load_content_skills(cfg: dict) -> dict[str, ContentSkillConfig]:
    out: dict[str, ContentSkillConfig] = {}
    for name, spec in cfg.get("content_skills", {}).items():
        ov = spec.get("overlap", [0.0, 1.0])
        params = {k: v for k, v in spec.items() if k != "overlap"}
        out[name] = ContentSkillConfig(
            name=name,
            overlap=(float(ov[0]), float(ov[1])),
            params=params,
        )
    return out


# ── helpers ────────────────────────────────────────────────────────────

def _bbox_area(b: tuple[float, float, float, float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _pair_pose_deltas(f_src: Frame, f_tgt: Frame) -> tuple[float, float]:
    return pair_pose_deltas(f_src.pose_c2w, f_tgt.pose_c2w)


def _overlap_ok(pair: ViewPair, cfg: ContentSkillConfig) -> bool:
    lo, hi = cfg.overlap
    return lo <= pair.overlap <= hi


_LABEL_MATCHER = None


def _label_matcher():
    """Lazy global CLIP-text matcher for paraphrase-tolerant label
    clustering. Returns None if CLIP can't be loaded — callers fall
    back to string equality."""
    global _LABEL_MATCHER
    if _LABEL_MATCHER is False:
        return None
    if _LABEL_MATCHER is not None:
        return _LABEL_MATCHER
    try:
        from ..label_matcher import LabelMatcher
        _LABEL_MATCHER = LabelMatcher()
    except Exception as e:
        logger.warning("CLIP-text label matcher unavailable (%s); "
                       "falling back to string equality", e)
        _LABEL_MATCHER = False
        return None
    return _LABEL_MATCHER


def _canonical_label(label: str) -> str:
    """Cheap label normalization for counting. Lowercase + strip only —
    'Chair' and 'chair ' collapse but 'chair' and 'armchair' stay
    distinct. Upgrade via a synonym map when counting falls short."""
    return (label or "").strip().lower()


def _mask_depth_coverage(mask: np.ndarray, depth: np.ndarray,
                         image_size: tuple[int, int],
                         depth_size: tuple[int, int]) -> float:
    """Fraction of a mask's footprint with valid (>0) depth samples.

    Color-pixel mask → downsample to depth resolution → count nonzero
    depth inside. Used by relative_distance to reject noisy candidates.
    """
    W, H = image_size
    Wd, Hd = depth_size
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0.0
    uds = np.clip((xs.astype(np.float32) * Wd / W).astype(np.int32), 0, Wd - 1)
    vds = np.clip((ys.astype(np.float32) * Hd / H).astype(np.int32), 0, Hd - 1)
    samples = depth[vds, uds]
    valid = int(np.count_nonzero(samples > 0.0))
    return valid / int(xs.size)
