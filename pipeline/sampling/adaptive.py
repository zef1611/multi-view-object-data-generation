"""Adaptive (pose-thresholded) keyframe sampling.

Greedy redundancy-based downsampling: keep a frame if its pose moved
>= `min_translation_m` OR optical axis rotated >= `min_rotation_deg`
relative to the most recently kept frame. This is the paper-faithful
default sampler — produces sparser, more diverse keyframes than stride
on slow-trajectory scenes.
"""

from __future__ import annotations

import numpy as np

from datasets.base import BaseSceneAdapter
from ..geometry import angle_between, camera_center, optical_axis


def select_keyframes_adaptive(
    adapter: BaseSceneAdapter,
    min_translation_m: float = 0.40,
    min_rotation_deg: float = 25.0,
) -> list[str]:
    """Return the keyframe IDs surviving the pose-delta gate."""
    all_ids = adapter.list_frames()
    kept: list[str] = []
    last_pose = None
    load_pose = getattr(adapter, "load_pose", None)
    for fid in all_ids:
        try:
            if load_pose is not None:
                pose = load_pose(fid)
            else:
                pose = adapter.load_frame(fid).pose_c2w
        except (FileNotFoundError, OSError, ValueError):
            continue
        if pose is None or not np.all(np.isfinite(pose)):
            continue
        if last_pose is None:
            kept.append(fid); last_pose = pose; continue
        dt = float(np.linalg.norm(camera_center(pose)
                                  - camera_center(last_pose)))
        dr = angle_between(optical_axis(pose), optical_axis(last_pose))
        if dt >= min_translation_m or dr >= min_rotation_deg:
            kept.append(fid); last_pose = pose
    return kept
