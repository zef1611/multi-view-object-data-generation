"""JSONL writer for Phase 1 verified correspondences.

One JSONL line per (scene, view-pair, src mask -> tgt mask) point pair.
Schema is defined in the plan (Phase 1 intermediate handoff to Phase 2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Optional


@dataclass
class CorrespondenceRecord:
    scene_id: str
    frame_src: str
    frame_tgt: str
    image_src: str
    image_tgt: str
    point_src: tuple[int, int]            # color-resolution int (u, v)
    point_tgt: tuple[int, int]            # geometric reproj; valid even when visible=False
    X_world: tuple[float, float, float]   # meters
    src_mask_id: int
    tgt_mask_id: int                      # -1 when visible=False
    src_bbox: tuple[float, float, float, float]
    tgt_bbox: tuple[float, float, float, float]
    src_label: str
    tgt_label: str
    depth_src: float
    depth_pred_tgt: float                 # geometric (z in tgt camera)
    depth_obs_tgt: float                  # observed (z from tgt depth map at point_tgt)
    iou_src_to_tgt: float                 # 0.0 when visible=False
    pair_overlap: float
    seed_retry: int
    visible: bool = True                  # False = occlusion-visibility negative
    dataset_source: str = "unknown"       # adapter.source_name (scannet / scannet++ / matterport)
    src_canonical: str = ""               # cross-frame stable category (Gemini-derived); falls back to src_label when empty
    tgt_canonical: str = ""

    def to_json(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "dataset_source": self.dataset_source,
            "frame_src": self.frame_src,
            "frame_tgt": self.frame_tgt,
            "image_src": self.image_src,
            "image_tgt": self.image_tgt,
            "point_src": list(self.point_src),
            "point_tgt": list(self.point_tgt),
            "X_world": list(self.X_world),
            "src_mask_id": self.src_mask_id,
            "tgt_mask_id": self.tgt_mask_id,
            "src_bbox": list(self.src_bbox),
            "tgt_bbox": list(self.tgt_bbox),
            "src_label": self.src_label,
            "tgt_label": self.tgt_label,
            "src_canonical": self.src_canonical,
            "tgt_canonical": self.tgt_canonical,
            "depth_src": round(float(self.depth_src), 4),
            "depth_pred_tgt": round(float(self.depth_pred_tgt), 4),
            "depth_obs_tgt": round(float(self.depth_obs_tgt), 4),
            "iou_src_to_tgt": round(float(self.iou_src_to_tgt), 4),
            "pair_overlap": round(float(self.pair_overlap), 4),
            "seed_retry": int(self.seed_retry),
            "visible": bool(self.visible),
        }


def round_clip_pixel(u: float, v: float, W: int, H: int,
                     tol: float = 2.0) -> Optional[tuple[int, int]]:
    """Round to int and clip to image bounds. Reject if pre-clip values
    were >`tol` pixels outside (means the projection actually missed)."""
    if u < -tol or u > W - 1 + tol or v < -tol or v > H - 1 + tol:
        return None
    ui = int(round(u)); vi = int(round(v))
    ui = max(0, min(W - 1, ui))
    vi = max(0, min(H - 1, vi))
    return ui, vi


class CorrespondenceWriter:
    """Append-only JSONL writer with sidecar rejection log."""

    def __init__(self, out_path: Path, rejections_path: Optional[Path] = None,
                 resume: bool = False):
        self.out_path = Path(out_path)
        self.rejections_path = (
            Path(rejections_path)
            if rejections_path is not None
            else self.out_path.with_name(self.out_path.stem + ".rejections.jsonl")
        )
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if resume else "w"
        self._fp: IO = open(self.out_path, mode)
        self._fp_rej: IO = open(self.rejections_path, mode)
        self._counts: dict[str, int] = {"emitted": 0}

    def emit(self, record: CorrespondenceRecord) -> None:
        self._fp.write(json.dumps(record.to_json()) + "\n")
        self._fp.flush()
        self._counts["emitted"] = self._counts.get("emitted", 0) + 1

    def reject(self, scene_id: str, frame_src: str, frame_tgt: str,
               src_mask_id: int, reason: str) -> None:
        self._fp_rej.write(json.dumps({
            "scene_id": scene_id, "frame_src": frame_src, "frame_tgt": frame_tgt,
            "src_mask_id": src_mask_id, "reason": reason,
        }) + "\n")
        self._fp_rej.flush()
        self._counts[reason] = self._counts.get(reason, 0) + 1

    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def close(self) -> None:
        self._fp.close()
        self._fp_rej.close()

    def __enter__(self): return self
    def __exit__(self, *a): self.close()


# ---------- task-routed writer (one folder per CrossPoint task) ----------

# Predicates: which task does a record support?
# A record may satisfy several — same data can feed multiple QA categories.
TASK_PREDICATES = {
    "cross_depth_variation":        lambda r: r.visible,
    "cross_occlusion_visibility":   lambda r: True,    # both pos and neg
    "cross_spatial_transformation": lambda r: r.visible,  # paired in Phase 2
    # Content-stage skills. Eligibility is decided by core/skill_gates.py
    # at pair level; the record predicate only enforces visibility where
    # the QA task itself requires it.
    "cross_point_correspondence":   lambda r: r.visible,
    "cross_object_correspondence":  lambda r: r.visible,
    "anchor":                       lambda r: r.visible,
    "counting":                     lambda r: True,
    "relative_distance":            lambda r: r.visible,
    "relative_direction":           lambda r: r.visible,
}


class TaskRouter:
    """Routes each CorrespondenceRecord into one JSONL per applicable task.

    Layout under `out_root/stage_1/`:
      <task>/correspondences.pos.jsonl          (visible=True records)
      <task>/correspondences.neg.jsonl          (visible=False, occluded)
      _all/correspondences.jsonl                (everything; for QC / viz)
      _all/correspondences.rejections.jsonl
      perception/<scene>.png                    (auto-viz: GD+SAM per scene)
      pairs/<scene>_<src>_<tgt>.png             (auto-viz: per-pair match)

    Phase 2 will land under `out_root/stage_2/` alongside.
    """

    STAGE = "stage_1"

    def __init__(self, out_root: Path, resume: bool = False,
                 task_predicates: dict = TASK_PREDICATES):
        self.out_root = Path(out_root)
        self.stage_root = self.out_root / self.STAGE
        self.stage_root.mkdir(parents=True, exist_ok=True)
        # Per task we keep two writers — pos (visible=True) and neg
        # (visible=False) — so phase-2 / QC can grab one slice without
        # filtering. _all collects everything for legacy QC tooling.
        self._pos_writers: dict[str, CorrespondenceWriter] = {}
        self._neg_writers: dict[str, CorrespondenceWriter] = {}
        for task in task_predicates:
            self._pos_writers[task] = CorrespondenceWriter(
                self.stage_root / task / "correspondences.pos.jsonl",
                resume=resume,
            )
            self._neg_writers[task] = CorrespondenceWriter(
                self.stage_root / task / "correspondences.neg.jsonl",
                resume=resume,
            )
        self._all_writer = CorrespondenceWriter(
            self.stage_root / "_all" / "correspondences.jsonl", resume=resume,
        )
        self._predicates = dict(task_predicates)
        self._task_counts: dict[str, int] = {"_all": 0}
        for t in task_predicates:
            self._task_counts[f"{t}.pos"] = 0
            self._task_counts[f"{t}.neg"] = 0

    def emit(self, record: CorrespondenceRecord,
             eligible_tasks: Optional[set[str]] = None) -> None:
        """Write to _all and to every task whose predicate is satisfied,
        routing into pos.jsonl or neg.jsonl by record.visible.

        If `eligible_tasks` is given, routing is further restricted to that
        set — used when pair selection has already decided which tasks a
        pair qualifies for, so records from a pair never leak into tasks
        that pair didn't pass the gates for.
        """
        self._all_writer.emit(record)
        self._task_counts["_all"] += 1
        for task, pred in self._predicates.items():
            if eligible_tasks is not None and task not in eligible_tasks:
                continue
            if not pred(record):
                continue
            if record.visible:
                self._pos_writers[task].emit(record)
                self._task_counts[f"{task}.pos"] += 1
            else:
                self._neg_writers[task].emit(record)
                self._task_counts[f"{task}.neg"] += 1

    def reject(self, scene_id: str, frame_src: str, frame_tgt: str,
               src_mask_id: int, reason: str) -> None:
        # Rejections only go to the _all stream.
        self._all_writer.reject(scene_id, frame_src, frame_tgt,
                                src_mask_id, reason)

    def counts(self) -> dict[str, int]:
        out = self._all_writer.counts()
        out["per_task"] = dict(self._task_counts)
        return out

    def close(self) -> None:
        for w in self._pos_writers.values():
            w.close()
        for w in self._neg_writers.values():
            w.close()
        self._all_writer.close()

    def __enter__(self): return self
    def __exit__(self, *a): self.close()
