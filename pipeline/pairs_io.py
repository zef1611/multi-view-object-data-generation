"""Read/write ``pairs.scored.jsonl`` — the Phase 3 (pair-gate) artifact.

Distinct from ``<out_root>/<skill>/pairs.jsonl`` (the per-skill
``PairManifest`` written by Phase 5). This module serializes the
**pre-emission scored pair list** so:

  * ``cli pair_gate`` can dump its output and stop
  * ``cli label`` / ``cli perceive`` / ``cli match`` can read it back
    and operate only on frames-in-pairs

The line schema is fully self-describing — every field needed to
reconstruct ``FrameRef`` for src/tgt without rebuilding an adapter::

    {"adapter":"scannet","scene_id":"scene0093_00",
     "src_id":"000050","tgt_id":"000200",
     "image_src":"/abs/000050.jpg","image_tgt":"/abs/000200.jpg",
     "overlap":0.45,"occluded_frac":0.12,"angle_deg":28.3,
     "distance_m":1.42,"quality":0.38,
     "median_depth_src":2.1,"median_depth_tgt":2.3,
     "tasks":["cross_depth_variation","cross_spatial_transformation"]}

``ViewPair.tasks`` is a ``frozenset`` — JSON-incompatible. We serialize
to a sorted list and reconstruct on read so cross-process ordering is
deterministic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from models._frame_ref import FrameRef
from .pairs import ViewPair


@dataclass
class ScoredPair:
    """A ``ViewPair`` annotated with adapter + scene + image paths so a
    standalone CLI can rebuild ``FrameRef`` for src/tgt without an
    adapter."""
    adapter: str
    scene_id: str
    src_id: str
    tgt_id: str
    image_src: str
    image_tgt: str
    overlap: float
    occluded_frac: float
    angle_deg: float
    distance_m: float
    quality: float
    median_depth_src: float
    median_depth_tgt: float
    tasks: frozenset = field(default_factory=frozenset)

    def to_view_pair(self) -> ViewPair:
        return ViewPair(
            src_id=self.src_id, tgt_id=self.tgt_id,
            overlap=self.overlap, occluded_frac=self.occluded_frac,
            angle_deg=self.angle_deg, distance_m=self.distance_m,
            quality=self.quality,
            median_depth_src=self.median_depth_src,
            median_depth_tgt=self.median_depth_tgt,
            tasks=self.tasks,
        )

    def src_frame_ref(self) -> FrameRef:
        return FrameRef(
            image_path=Path(self.image_src),
            adapter=self.adapter, scene_id=self.scene_id,
            frame_id=self.src_id,
        )

    def tgt_frame_ref(self) -> FrameRef:
        return FrameRef(
            image_path=Path(self.image_tgt),
            adapter=self.adapter, scene_id=self.scene_id,
            frame_id=self.tgt_id,
        )


def _row(p: ScoredPair) -> dict:
    return {
        "adapter": p.adapter,
        "scene_id": p.scene_id,
        "src_id": p.src_id,
        "tgt_id": p.tgt_id,
        "image_src": p.image_src,
        "image_tgt": p.image_tgt,
        "overlap": float(p.overlap),
        "occluded_frac": float(p.occluded_frac),
        "angle_deg": float(p.angle_deg),
        "distance_m": float(p.distance_m),
        "quality": float(p.quality),
        "median_depth_src": float(p.median_depth_src),
        "median_depth_tgt": float(p.median_depth_tgt),
        "tasks": sorted(p.tasks),
    }


def write_scored_pairs(pairs: Iterable[ScoredPair], path: Path,
                       *, append: bool = False) -> int:
    """Write one JSON object per pair, one per line. Returns the number
    of pairs written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    n = 0
    with open(path, mode) as f:
        for p in pairs:
            f.write(json.dumps(_row(p)) + "\n")
            n += 1
    return n


def read_scored_pairs(path: Path) -> list[ScoredPair]:
    out: list[ScoredPair] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(ScoredPair(
                adapter=d["adapter"],
                scene_id=d["scene_id"],
                src_id=d["src_id"],
                tgt_id=d["tgt_id"],
                image_src=d["image_src"],
                image_tgt=d["image_tgt"],
                overlap=float(d["overlap"]),
                occluded_frac=float(d["occluded_frac"]),
                angle_deg=float(d["angle_deg"]),
                distance_m=float(d["distance_m"]),
                quality=float(d["quality"]),
                median_depth_src=float(d["median_depth_src"]),
                median_depth_tgt=float(d["median_depth_tgt"]),
                tasks=frozenset(d.get("tasks", [])),
            ))
    return out


def view_pairs_to_scored(
    pairs: list[ViewPair],
    *,
    adapter: str,
    scene_id: str,
    image_path_for: dict[str, str],
) -> list[ScoredPair]:
    """Convert in-memory ``ViewPair`` list to ``ScoredPair`` ready for
    serialization. ``image_path_for`` maps frame_id → absolute image path
    (the caller already has these via ``adapter.frame_ref``)."""
    return [
        ScoredPair(
            adapter=adapter, scene_id=scene_id,
            src_id=p.src_id, tgt_id=p.tgt_id,
            image_src=image_path_for[p.src_id],
            image_tgt=image_path_for[p.tgt_id],
            overlap=p.overlap, occluded_frac=p.occluded_frac,
            angle_deg=p.angle_deg, distance_m=p.distance_m,
            quality=p.quality,
            median_depth_src=p.median_depth_src,
            median_depth_tgt=p.median_depth_tgt,
            tasks=p.tasks,
        )
        for p in pairs
    ]
