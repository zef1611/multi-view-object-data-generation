"""Model ABCs for the perception layer — `Detector` and `Segmenter`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._frame_ref import FrameRef


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]   # x0, y0, x1, y1 in image pixels
    score: float
    label: str                                # GD's class name (open-vocab)
    # Stable cross-frame category (e.g. "office chair" → canonical "chair").
    # Empty when the labeler doesn't expose canonicals; consumers should
    # fall back to `label` in that case.
    canonical: str = ""


@dataclass
class ObjectMask:
    """A SAM mask for one detection."""
    mask: np.ndarray                           # HxW bool, color-image resolution
    bbox: tuple[float, float, float, float]
    score: float
    label: str
    centroid: tuple[float, float]              # (u, v) in color pixels
    area: int
    canonical: str = ""


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: FrameRef) -> list[Detection]: ...


class Segmenter(ABC):
    @abstractmethod
    def segment(self, image_path: Path,
                detections: list[Detection]) -> list[ObjectMask]: ...
