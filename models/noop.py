"""No-op model wrappers for unit/smoke tests (no GPU required).

NoopDetector emits a deterministic 3x3 grid of bboxes per frame.
NoopSegmenter turns each bbox into a filled-rectangle mask.

This lets us validate the full Phase 1 plumbing without loading
Grounding-DINO or SAM 2.1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ._frame_ref import FrameRef
from .base import Detection, Detector, ObjectMask, Segmenter


class NoopDetector(Detector):
    def __init__(self, grid: int = 3, label: str = "obj"):
        self.grid = grid
        self.label = label

    def detect(self, frame: FrameRef) -> list[Detection]:
        image_path = frame.image_path
        with Image.open(image_path) as im:
            W, H = im.size
        out: list[Detection] = []
        cell_w = W / (self.grid + 1)
        cell_h = H / (self.grid + 1)
        for i in range(self.grid):
            for j in range(self.grid):
                cx = (i + 1) * cell_w
                cy = (j + 1) * cell_h
                w = cell_w * 0.6
                h = cell_h * 0.6
                out.append(Detection(
                    bbox=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                    score=1.0 - 0.01 * (i * self.grid + j),
                    label=f"{self.label}{i * self.grid + j}",
                ))
        return out


class NoopSegmenter(Segmenter):
    def segment(self, image_path: Path,
                detections: list[Detection]) -> list[ObjectMask]:
        with Image.open(image_path) as im:
            W, H = im.size
        out: list[ObjectMask] = []
        for det in detections:
            x0, y0, x1, y1 = (int(round(v)) for v in det.bbox)
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(W, x1); y1 = min(H, y1)
            mask = np.zeros((H, W), dtype=bool)
            mask[y0:y1, x0:x1] = True
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            out.append(ObjectMask(
                mask=mask, bbox=det.bbox, score=det.score, label=det.label,
                centroid=(cx, cy), area=int(mask.sum()),
            ))
        return out
