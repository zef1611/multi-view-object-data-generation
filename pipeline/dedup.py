"""Per-scene voxel dedup of emitted 3D world points."""

from __future__ import annotations

from typing import Iterable

import numpy as np


class VoxelSet:
    def __init__(self, voxel_size_m: float = 0.05):
        self.voxel = float(voxel_size_m)
        self._set: set[tuple[int, int, int]] = set()

    def _key(self, X: Iterable[float]) -> tuple[int, int, int]:
        x, y, z = X
        v = self.voxel
        return (int(np.floor(x / v)), int(np.floor(y / v)), int(np.floor(z / v)))

    def __contains__(self, X) -> bool:
        return self._key(X) in self._set

    def add(self, X) -> bool:
        """Return True if newly added, False if already present."""
        k = self._key(X)
        if k in self._set:
            return False
        self._set.add(k)
        return True

    def __len__(self) -> int:
        return len(self._set)
