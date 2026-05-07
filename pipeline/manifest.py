"""Pair-level manifest writer for stage 1.

Per (pair, qualifying skill) the pipeline emits one line into
`stage_1/<skill>/pairs.jsonl`. Each line bundles everything phase 2
needs to build a question for this skill without revisiting the scene
— pair identity, full pose + image metadata, the list of matched
objects in the pair, and the skill-specific evidence payload
returned by `core/skill_gates.py`.

The existing per-match `correspondences.jsonl` shard is still written
by `core/emit.py::TaskRouter` — the manifest is additive, not a
replacement. Phase 2 generators for counting / distance / direction
read `pairs.jsonl`; anchor / cross_correspondence / cross_* can
operate from either shard.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Iterable, Optional

import numpy as np

from datasets.base import Frame
from models.base import ObjectMask
from .match import Match
from .pairs import ViewPair
from .skills import SkillEvidence


@dataclass
class PairObject:
    """One matched-object row inside a PairManifest."""
    match_idx: int
    src_mask_id: int
    tgt_mask_id: int                  # -1 when visible=False
    src_label: str
    tgt_label: str
    src_bbox: tuple[float, float, float, float]
    tgt_bbox: tuple[float, float, float, float]
    src_centroid: tuple[float, float]
    tgt_centroid: tuple[float, float]
    point_src: tuple[int, int]        # color-pixel int
    point_tgt: tuple[int, int]
    X_world: tuple[float, float, float]
    depth_src: float
    depth_pred_tgt: float
    depth_obs_tgt: float
    iou_src_to_tgt: float
    visible: bool


@dataclass
class PairManifest:
    skill: str
    scene_id: str
    dataset_source: str
    frame_src: str
    frame_tgt: str
    image_src: str
    image_tgt: str
    image_src_size: tuple[int, int]
    image_tgt_size: tuple[int, int]
    K_src: list[list[float]]
    K_tgt: list[list[float]]
    pose_src_c2w: list[list[float]]
    pose_tgt_c2w: list[list[float]]
    pair_overlap: float
    pair_occluded_frac: float
    pair_angle_deg: float
    pair_distance_m: float
    objects: list[PairObject] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        # Trim floats for smaller JSONL.
        for k in ("pair_overlap", "pair_occluded_frac",
                  "pair_angle_deg", "pair_distance_m"):
            d[k] = round(float(d[k]), 4)
        for o in d["objects"]:
            for k in ("depth_src", "depth_pred_tgt", "depth_obs_tgt",
                      "iou_src_to_tgt"):
                o[k] = round(float(o[k]), 4)
        return d


# ---- builders ----------------------------------------------------------

def _build_objects(
    src_frame: Frame, src_masks: list[ObjectMask],
    tgt_frame: Frame, tgt_masks: list[ObjectMask],
    matches: list[Match],
) -> list[PairObject]:
    W, H = src_frame.image_size
    Wt, Ht = tgt_frame.image_size
    out: list[PairObject] = []
    for mi, m in enumerate(matches):
        src_m = src_masks[m.src_mask_idx]
        if m.tgt_mask_idx >= 0:
            tgt_m = tgt_masks[m.tgt_mask_idx]
            tgt_label = tgt_m.label
            tgt_bbox = tgt_m.bbox
            tgt_centroid = tgt_m.centroid
        else:
            tgt_label = ""
            tgt_bbox = (-1.0, -1.0, -1.0, -1.0)
            tgt_centroid = (-1.0, -1.0)
        out.append(PairObject(
            match_idx=mi,
            src_mask_id=m.src_mask_idx,
            tgt_mask_id=m.tgt_mask_idx,
            src_label=src_m.label,
            tgt_label=tgt_label,
            src_bbox=tuple(float(x) for x in src_m.bbox),
            tgt_bbox=tuple(float(x) for x in tgt_bbox),
            src_centroid=tuple(float(x) for x in src_m.centroid),
            tgt_centroid=tuple(float(x) for x in tgt_centroid),
            point_src=(
                max(0, min(W - 1, int(round(m.p_src[0])))),
                max(0, min(H - 1, int(round(m.p_src[1])))),
            ),
            point_tgt=(
                max(0, min(Wt - 1, int(round(m.p_tgt[0])))),
                max(0, min(Ht - 1, int(round(m.p_tgt[1])))),
            ),
            X_world=tuple(float(x) for x in m.X_world),
            depth_src=float(m.depth_src),
            depth_pred_tgt=float(m.depth_pred_tgt),
            depth_obs_tgt=float(m.depth_obs_tgt),
            iou_src_to_tgt=float(m.iou),
            visible=bool(m.visible),
        ))
    return out


def build_manifest(
    skill: str, evidence: SkillEvidence,
    pair: ViewPair,
    scene_id: str, dataset_source: str,
    src_frame: Frame, src_masks: list[ObjectMask],
    tgt_frame: Frame, tgt_masks: list[ObjectMask],
    matches: list[Match],
) -> PairManifest:
    return PairManifest(
        skill=skill,
        scene_id=scene_id,
        dataset_source=dataset_source,
        frame_src=pair.src_id,
        frame_tgt=pair.tgt_id,
        image_src=str(src_frame.image_path),
        image_tgt=str(tgt_frame.image_path),
        image_src_size=tuple(int(x) for x in src_frame.image_size),
        image_tgt_size=tuple(int(x) for x in tgt_frame.image_size),
        K_src=np.asarray(src_frame.K_color, dtype=float).tolist(),
        K_tgt=np.asarray(tgt_frame.K_color, dtype=float).tolist(),
        pose_src_c2w=np.asarray(src_frame.pose_c2w, dtype=float).tolist(),
        pose_tgt_c2w=np.asarray(tgt_frame.pose_c2w, dtype=float).tolist(),
        pair_overlap=float(pair.overlap),
        pair_occluded_frac=float(pair.occluded_frac),
        pair_angle_deg=float(pair.angle_deg),
        pair_distance_m=float(pair.distance_m),
        objects=_build_objects(src_frame, src_masks,
                               tgt_frame, tgt_masks, matches),
        evidence={
            "qualifying_matches": list(evidence.qualifying_matches),
            **evidence.meta,
        },
    )


# ---- writer ------------------------------------------------------------

class PairManifestWriter:
    """Appends one pairs.jsonl per skill under `stage_1/<skill>/`."""

    FILENAME = "pairs.jsonl"

    def __init__(self, stage_root: Path, skills: Iterable[str],
                 resume: bool = False):
        self.stage_root = Path(stage_root)
        self._fps: dict[str, IO] = {}
        self._counts: dict[str, int] = {}
        mode = "a" if resume else "w"
        for s in skills:
            d = self.stage_root / s
            d.mkdir(parents=True, exist_ok=True)
            self._fps[s] = open(d / self.FILENAME, mode)
            self._counts[s] = 0

    def emit(self, manifest: PairManifest) -> None:
        fp = self._fps.get(manifest.skill)
        if fp is None:  # skill not registered with this writer
            return
        fp.write(json.dumps(manifest.to_json()) + "\n")
        fp.flush()
        self._counts[manifest.skill] += 1

    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def close(self) -> None:
        for fp in self._fps.values():
            fp.close()

    def __enter__(self): return self
    def __exit__(self, *a): self.close()
