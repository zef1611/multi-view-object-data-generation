"""Shared logic for GT-driven detectors.

Both `ScanNetGTDetector` (GT bbox passthrough) and
`ScanNetGTLabelGDinoDetector` (GT labels → GDino re-ground) need the
same up-front filtering: per-frame instance map → unique IDs → area gate
→ blocklist gate. This module owns that pipeline so the two detectors
share one implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from datasets.base import BaseSceneAdapter
from pipeline.label_blocklist import DEFAULT_LABEL_BLOCKLIST
from ..base import Detector


class GTDetectorBase(Detector):
    """Common knobs + GT-instance extraction for ScanNet-style detectors."""

    def __init__(
        self,
        adapter: Optional[BaseSceneAdapter] = None,
        min_area_frac: float = 0.005,
        label_blocklist: Optional[frozenset[str]] = None,
        max_dets: int = 50,
    ):
        self.adapter = adapter
        self.min_area_frac = float(min_area_frac)
        self.label_blocklist = (
            frozenset(label_blocklist)
            if label_blocklist is not None else DEFAULT_LABEL_BLOCKLIST
        )
        self.max_dets = int(max_dets)

    # Per-scene hook — `process_scene` calls this so a single detector
    # instance can be reused across scenes.
    def set_adapter(self, adapter: BaseSceneAdapter) -> None:
        self.adapter = adapter

    def canonicalize_mask_label(self, label: str) -> str:
        """No-op for GT detectors — labels are already canonical."""
        return (label or "").strip().lower()

    @staticmethod
    def _frame_id_for(image_path: Path) -> Optional[str]:
        try:
            return Path(image_path).stem
        except Exception:
            return None

    def _extract_gt_instances(
        self, image_path: Path,
    ) -> Optional[tuple[np.ndarray, list[tuple[int, str, np.ndarray]]]]:
        """Load the GT instance map for `image_path` and return
        `(inst_mask, [(iid, label_lower, bool_mask), ...])` after the
        area + blocklist gates. Returns None if no GT is available.

        Sorted by area descending so callers that truncate at `max_dets`
        keep the largest objects.
        """
        if self.adapter is None:
            return None
        fid = self._frame_id_for(image_path)
        if fid is None:
            return None
        out = self.adapter.qc_instance_mask(fid)
        if out is None:
            return None
        inst_mask, label_map = out
        if inst_mask.size == 0:
            return None
        H, W = inst_mask.shape[:2]
        img_area = float(H * W)
        if img_area <= 0.0:
            return None

        ids, counts = np.unique(inst_mask, return_counts=True)
        keep = ids != 0
        ids = ids[keep]; counts = counts[keep]
        order = np.argsort(-counts)
        ids = ids[order]; counts = counts[order]

        kept: list[tuple[int, str, np.ndarray]] = []
        for iid, cnt in zip(ids, counts):
            if cnt / img_area < self.min_area_frac:
                continue
            label = (label_map.get(int(iid), "") or "").strip()
            if not label or label.lower() in self.label_blocklist:
                continue
            kept.append((int(iid), label.lower(), inst_mask == iid))
        return inst_mask, kept
