"""Pose / camera geometry helpers shared by pair selection and skill gates."""

from __future__ import annotations

import numpy as np


def camera_center(pose_c2w: np.ndarray) -> np.ndarray:
    return pose_c2w[:3, 3]


def optical_axis(pose_c2w: np.ndarray) -> np.ndarray:
    """+Z in the camera frame, expressed in world coordinates (OpenCV convention)."""
    return pose_c2w[:3, :3] @ np.array([0.0, 0.0, 1.0])


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    cos = np.clip(float(np.dot(a, b) / (na * nb)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def pair_pose_deltas(pose_a_c2w: np.ndarray,
                     pose_b_c2w: np.ndarray) -> tuple[float, float]:
    """Return (translation_m, optical_axis_rotation_deg) between two camera poses."""
    t = float(np.linalg.norm(camera_center(pose_a_c2w)
                             - camera_center(pose_b_c2w)))
    r = angle_between(optical_axis(pose_a_c2w), optical_axis(pose_b_c2w))
    return t, r
