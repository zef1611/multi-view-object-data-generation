"""Helpers for the per-stage CLIs to read/write a ``frames.json`` file —
a list of FrameRef dicts that can be reconstructed without rerunning
the sampler.

Format (one JSON document, list of dicts):

    [
      {"adapter": "scannet", "scene_id": "scene0093_00",
       "frame_id": "000050",
       "image_path": "/abs/path/to/000050.jpg"},
      ...
    ]

This is the contract between ``cli/sample.py`` and the downstream VLM
stage runners (``cli/filter.py``, ``cli/label.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

from models._frame_ref import FrameRef


def write_frames(frames: list[FrameRef], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "adapter": f.adapter,
            "scene_id": f.scene_id,
            "frame_id": f.frame_id,
            "image_path": str(f.image_path),
        }
        for f in frames
    ]
    path.write_text(json.dumps(payload, indent=2))


def read_frames(path: Path) -> list[FrameRef]:
    raw = json.loads(path.read_text())
    return [
        FrameRef(
            image_path=Path(d["image_path"]),
            adapter=d["adapter"],
            scene_id=d["scene_id"],
            frame_id=d["frame_id"],
        )
        for d in raw
    ]
