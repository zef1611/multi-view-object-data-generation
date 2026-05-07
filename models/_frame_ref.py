"""FrameRef — the canonical handle for one cached frame.

Every cache in this repo (perception, labels, filter) is keyed by the
triple `(adapter, scene_id, frame_id)` plus a model tag in the parent
directory. `FrameRef` bundles those three with the actual image path
so downstream callers don't have to reconstruct them from filesystem
conventions that differ per adapter.

Cache layout consequence:
    cache/<namespace>/<model_tag>/<adapter>/<scene_id>/<frame_id>.<ext>
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrameRef:
    image_path: Path
    adapter: str
    scene_id: str
    frame_id: str

    def __post_init__(self) -> None:
        for name in ("adapter", "scene_id", "frame_id"):
            v = getattr(self, name)
            if not v or "/" in v or "\\" in v:
                raise ValueError(
                    f"FrameRef.{name}={v!r} must be non-empty and free of "
                    f"path separators (used as a cache subdir)."
                )

    @property
    def cache_subpath(self) -> str:
        """`<adapter>/<scene_id>/<frame_id>` — the per-frame cache key
        appended under any model-tagged cache root."""
        return f"{self.adapter}/{self.scene_id}/{self.frame_id}"
