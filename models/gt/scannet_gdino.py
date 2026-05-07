"""GT-labeled, GDino-localized detector for ScanNet.

Hybrid of `ScanNetGTDetector` and `GDinoDetector`:

  1. Per frame, read the GT instance map and extract the unique set of
     GT class **labels** present (after the structural blocklist).
  2. Feed those labels to GDino as the prompt vocabulary. GDino
     re-grounds each label against the actual color image and returns
     image-aligned bboxes — sidestepping the mesh-projection
     misalignment in GT bboxes.
  3. Each GDino box is matched back to a specific GT instance by IoU
     against same-label GT masks. The Detection inherits the GT label
     (so cross-frame identity stays anchored on GT) and the GDino bbox
     (so the downstream SAM box prompt is well-aligned).

When a GDino box has no same-label GT mask with IoU >= `gt_match_iou`,
the box is dropped — this enforces "every emitted detection corresponds
to a real GT instance".
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from datasets.base import BaseSceneAdapter
from ..base import Detection
from ..detectors.gdino import GDinoDetector
from .base import GTDetectorBase

logger = logging.getLogger(__name__)


def _mask_box_iou(mask: np.ndarray, bbox: tuple) -> float:
    """IoU between a binary mask and a bbox (axis-aligned, in pixels)."""
    H, W = mask.shape
    x0, y0, x1, y1 = bbox
    x0 = max(0, int(round(x0))); y0 = max(0, int(round(y0)))
    x1 = min(W, int(round(x1))); y1 = min(H, int(round(y1)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    bbox_area = (x1 - x0) * (y1 - y0)
    inter = int(mask[y0:y1, x0:x1].sum())
    mask_area = int(mask.sum())
    union = mask_area + bbox_area - inter
    return inter / union if union > 0 else 0.0


class ScanNetGTLabelGDinoDetector(GTDetectorBase):
    def __init__(
        self,
        adapter: Optional[BaseSceneAdapter] = None,
        gdino: Optional[GDinoDetector] = None,
        min_area_frac: float = 0.005,
        label_blocklist: Optional[frozenset[str]] = None,
        gt_match_iou: float = 0.10,
        max_dets: int = 50,
    ):
        super().__init__(
            adapter=adapter, min_area_frac=min_area_frac,
            label_blocklist=label_blocklist, max_dets=max_dets,
        )
        # Label-agnostic GDino — the vocabulary is overridden per call.
        self.gdino = gdino or GDinoDetector(classes=["object"])
        self.gt_match_iou = float(gt_match_iou)

    def config(self) -> dict:
        return {
            "kind": "scannet_gt_label_gdino",
            "min_area_frac": self.min_area_frac,
            "label_blocklist": sorted(self.label_blocklist),
            "gt_match_iou": self.gt_match_iou,
            "max_dets": self.max_dets,
            "gdino": {k: v for k, v in self.gdino.config().items()
                      if k not in ("classes",)},
        }

    def detect(self, frame) -> list[Detection]:
        image_path = frame.image_path
        out = self._extract_gt_instances(image_path)
        if out is None:
            return []
        _, per_instance = out
        if not per_instance:
            return []

        unique_labels = sorted({lab for _, lab, _ in per_instance})
        prompts = [f"{lab} ." for lab in unique_labels]
        per_prompt = self.gdino.detect_batched_prompts(
            image_path, prompts,
            chunk_size=getattr(self.gdino, "_default_chunk_size", 8),
        )

        claimed: set[int] = set()
        out_dets: list[Detection] = []
        for label, dets in zip(unique_labels, per_prompt):
            same_label_instances = [(iid, m) for iid, lab, m in per_instance
                                    if lab == label]
            if not dets or not same_label_instances:
                continue
            dets_sorted = sorted(dets, key=lambda d: -d.score)
            for d in dets_sorted:
                best_iid = -1; best_iou = self.gt_match_iou
                for iid, gt_m in same_label_instances:
                    if iid in claimed:
                        continue
                    iou = _mask_box_iou(gt_m, d.bbox)
                    if iou > best_iou:
                        best_iou = iou; best_iid = iid
                if best_iid >= 0:
                    claimed.add(best_iid)
                    out_dets.append(Detection(
                        bbox=d.bbox, score=float(d.score),
                        label=label, canonical=label,
                    ))
                    if len(out_dets) >= self.max_dets:
                        return out_dets
        return out_dets
