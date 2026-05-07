"""GT-prompted SAM detector for ScanNet.

Uses ScanNet's per-frame GT instance masks (`instance-filt/{i}.png`) as
the source of identity + bbox, then lets the segmenter (SAM2.1 or the
GT-mask passthrough) refine each bbox into a clean image-aligned mask.

Why this exists:
- Mesh-projected GT 2D masks are noisy (TSDF holes, voxel-aligned edges,
  resolution mismatch with color frames). Using them as detection
  *prompts* (not segmentation outputs) gives clean SAM masks while
  preserving GT instance identity.
- No API call, no GDino — runs ~50× faster than `labeled-gdino` on
  ScanNet, and identity is deterministic.

Pipeline integration:
- `Detection.label` and `.canonical` are both set to the ScanNet GT
  label (e.g. `"chair"`). Vocabulary is the ScanNet aggregation set.
- `Detection.score = 1.0`.
- `PerceptionCache` automatically backfills `mask.canonical` via
  `canonicalize_mask_label()`. For GT this is a no-op.

Limitations:
- ScanNet-only (depends on `qc_instance_mask`).
- Coarse label vocab.
- Tiny GT instances drop below `min_area_frac`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..base import Detection
from .base import GTDetectorBase

logger = logging.getLogger(__name__)


class ScanNetGTDetector(GTDetectorBase):
    def config(self) -> dict:
        return {
            "kind": "scannet_gt",
            "min_area_frac": self.min_area_frac,
            "label_blocklist": sorted(self.label_blocklist),
            "max_dets": self.max_dets,
        }

    def detect(self, frame) -> list[Detection]:
        image_path = frame.image_path
        out = self._extract_gt_instances(image_path)
        if out is None:
            return []
        _, instances = out

        detections: list[Detection] = []
        for iid, label, mask in instances:
            ys, xs = np.where(mask)
            if xs.size == 0:
                continue
            x0, y0 = float(xs.min()), float(ys.min())
            x1, y1 = float(xs.max() + 1), float(ys.max() + 1)
            detections.append(Detection(
                bbox=(x0, y0, x1, y1),
                score=1.0,
                label=label,
                canonical=label,
            ))
            if len(detections) >= self.max_dets:
                break
        return detections
