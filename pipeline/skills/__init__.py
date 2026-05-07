"""Skill gates and pose-stage evidence extractors.

Each skill is one file under this package:

* Content-stage gates: `cross_point_correspondence`,
  `cross_object_correspondence`, `anchor`, `counting`,
  `relative_distance`, `relative_direction`. Each exposes a
  `gate_<name>` function and is registered in `SKILL_GATES`.
* Pose-stage extractors: `cross_spatial_transformation`,
  `cross_depth_variation`, `cross_occlusion_visibility`. The per-pair
  gate already fired in `pipeline/pairs.py::_assign_tasks`; the
  extractor here just collects qualifying matches and per-skill
  metadata. Registered in `POSE_EVIDENCE`.

Adding a new skill:
  1. Drop a new `<name>.py` here exposing `gate_<name>` (or
     `evidence_<name>` for pose-stage).
  2. Register it in `SKILL_GATES` / `POSE_EVIDENCE` below.
  3. Add the skill's gate config to `configs/tasks.json`.

Evidence dispatchers `extract_all_evidence` and `assign_content_skills`
live here so they can iterate over the registries.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from datasets.base import Frame
from models.base import ObjectMask
from ..match import Match
from ..pairs import ViewPair

from .anchor import gate_anchor
from .base import (
    CONTENT_SKILLS,
    POSE_SKILLS,
    ContentSkillConfig,
    SkillEvidence,
    load_content_skills,
)
from .counting import gate_counting
from .cross_depth_variation import evidence_cross_depth_variation
from .cross_object_correspondence import gate_cross_object_correspondence
from .cross_occlusion_visibility import evidence_cross_occlusion_visibility
from .cross_point_correspondence import gate_cross_point_correspondence
from .cross_spatial_transformation import evidence_cross_spatial_transformation
from .relative_direction import gate_relative_direction
from .relative_distance import gate_relative_distance

logger = logging.getLogger(__name__)


SKILL_GATES: dict[str, Callable[..., Optional[SkillEvidence]]] = {
    "cross_point_correspondence": gate_cross_point_correspondence,
    "cross_object_correspondence": gate_cross_object_correspondence,
    "anchor": gate_anchor,
    "counting": gate_counting,
    "relative_distance": gate_relative_distance,
    "relative_direction": gate_relative_direction,
}

# Back-compat alias used by pre-split callers.
CONTENT_GATES = SKILL_GATES

POSE_EVIDENCE: dict[str, Callable[..., Optional[SkillEvidence]]] = {
    "cross_spatial_transformation": evidence_cross_spatial_transformation,
    "cross_depth_variation": evidence_cross_depth_variation,
    "cross_occlusion_visibility": evidence_cross_occlusion_visibility,
}


def extract_all_evidence(
    pair: ViewPair, f_src: Frame, masks_src: list[ObjectMask],
    f_tgt: Frame, masks_tgt: list[ObjectMask], matches: list[Match],
    content_skills: dict[str, ContentSkillConfig],
) -> dict[str, SkillEvidence]:
    """Run every gate / extractor applicable to this pair.

    Pose-stage extractors run only for skills already in `pair.tasks`
    (assigned by `pipeline/pairs.py::_assign_tasks`). Content-stage gates
    run for every configured content skill regardless.
    """
    out: dict[str, SkillEvidence] = {}
    for name in pair.tasks:
        extractor = POSE_EVIDENCE.get(name)
        if extractor is None:
            continue
        try:
            ev = extractor(pair, f_src, masks_src, f_tgt, masks_tgt, matches)
        except Exception as e:
            logger.debug("pose evidence '%s' failed on %s->%s: %s",
                         name, pair.src_id, pair.tgt_id, e)
            ev = None
        if ev is not None:
            out[name] = ev
    for name, cfg in content_skills.items():
        gate = SKILL_GATES.get(name)
        if gate is None:
            continue
        try:
            ev = gate(pair, f_src, masks_src, f_tgt, masks_tgt, matches, cfg)
        except Exception as e:
            logger.debug("content gate '%s' failed on %s->%s: %s",
                         name, pair.src_id, pair.tgt_id, e)
            ev = None
        if ev is not None:
            out[name] = ev
    return out


def assign_content_skills(
    pair: ViewPair, f_src: Frame, masks_src: list[ObjectMask],
    f_tgt: Frame, masks_tgt: list[ObjectMask], matches: list[Match],
    skills: dict[str, ContentSkillConfig],
) -> frozenset[str]:
    """Bool dispatcher kept for back-compat; returns the set of skill
    names whose gate fires (without their evidence payload)."""
    out: set[str] = set()
    for name, cfg in skills.items():
        gate = SKILL_GATES.get(name)
        if gate is None:
            continue
        try:
            ev = gate(pair, f_src, masks_src, f_tgt, masks_tgt, matches, cfg)
        except Exception as e:
            logger.debug("content gate '%s' failed on %s->%s: %s",
                         name, pair.src_id, pair.tgt_id, e)
            ev = None
        if ev is not None:
            out.add(name)
    return frozenset(out)


__all__ = [
    "CONTENT_SKILLS",
    "POSE_SKILLS",
    "SKILL_GATES",
    "POSE_EVIDENCE",
    "CONTENT_GATES",
    "ContentSkillConfig",
    "SkillEvidence",
    "assign_content_skills",
    "evidence_cross_depth_variation",
    "evidence_cross_occlusion_visibility",
    "evidence_cross_spatial_transformation",
    "extract_all_evidence",
    "gate_anchor",
    "gate_counting",
    "gate_cross_object_correspondence",
    "gate_cross_point_correspondence",
    "gate_relative_direction",
    "gate_relative_distance",
    "load_content_skills",
]
