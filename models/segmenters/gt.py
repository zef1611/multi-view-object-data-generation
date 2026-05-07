"""ScanNet GT mask segmenter — passthrough, no SAM.

Pairs with `ScanNetGTDetector`: given a Detection (bbox + label from
`instance-filt/{i}.png`), returns the matching GT instance mask as an
`ObjectMask`. Skips SAM entirely. The output mask is the raw mesh-
projected GT — noisy at the pixel level but identity-perfect.

Use this when you want **deterministic GT masks** (no detector or
segmenter cost, no edge-aligned refinement). Pair with
`scannet-gt-label+gdino` if you want clean SAM-refined masks while
keeping GT identity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from datasets.base import BaseSceneAdapter
from ..base import Detection, ObjectMask, Segmenter

logger = logging.getLogger(__name__)


class GTMaskSegmenter(Segmenter):
    def __init__(
        self,
        adapter: Optional[BaseSceneAdapter] = None,
        min_area_frac: float = 0.005,
    ):
        self.adapter = adapter
        self.min_area_frac = float(min_area_frac)

    def config(self) -> dict:
        return {
            "kind": "scannet_gt_mask",
            "min_area_frac": self.min_area_frac,
        }

    def set_adapter(self, adapter: BaseSceneAdapter) -> None:
        self.adapter = adapter

    def segment(self, image_path: Path,
                detections: list[Detection]) -> list[ObjectMask]:
        if self.adapter is None or not detections:
            return []
        fid = Path(image_path).stem
        out = self.adapter.qc_instance_mask(fid)
        if out is None:
            return []
        inst_mask, label_map = out
        H, W = inst_mask.shape[:2]
        img_area = float(H * W)
        if img_area <= 0.0:
            return []
        min_area = int(self.min_area_frac * img_area)

        results: list[ObjectMask] = []
        used: set[int] = set()
        for det in detections:
            x0, y0, x1, y1 = (int(round(v)) for v in det.bbox)
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(W, x1); y1 = min(H, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            # Find the dominant non-zero, unused instance ID inside the
            # detection bbox. `ScanNetGTDetector` computed each bbox from a
            # specific instance's mask, so matching by max-vote is
            # deterministic in practice.
            crop = inst_mask[y0:y1, x0:x1]
            ids, counts = np.unique(crop, return_counts=True)
            best_iid = -1; best_count = 0
            for iid, cnt in zip(ids, counts):
                iid = int(iid)
                if iid == 0 or iid in used:
                    continue
                # The detection's label must match this GT instance's label
                # (so two same-bbox different-label edge cases don't cross
                # over).
                gt_lab = (label_map.get(iid, "") or "").strip().lower()
                if det.label and gt_lab and gt_lab != det.label.strip().lower():
                    continue
                if cnt > best_count:
                    best_count = int(cnt); best_iid = iid
            if best_iid < 0:
                continue
            mask = (inst_mask == best_iid)
            area = int(mask.sum())
            if area < min_area:
                continue
            ys, xs = np.where(mask)
            cx = float(xs.mean()); cy = float(ys.mean())
            label = (label_map.get(best_iid, "") or "").strip()
            results.append(ObjectMask(
                mask=mask,
                bbox=det.bbox,
                score=float(getattr(det, "score", 1.0)),
                label=label,
                centroid=(cx, cy),
                area=area,
                canonical=label.lower(),
            ))
            used.add(best_iid)
        return results
