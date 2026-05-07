"""Relative direction — assign a compass bucket to each visible target.

Bucketing is 8-way (front, front-right, right, back-right, back,
back-left, left, front-left) computed in the *target* camera's frame,
with hysteresis near bucket edges to avoid label flips. A pair only
qualifies when at least two targets differ in azimuth by
`min_azimuth_sep_deg` so phase 2 has non-trivial distractors.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from datasets.base import Frame
from models.base import ObjectMask
from ..geometry import camera_center, optical_axis
from ..match import Match
from ..pairs import ViewPair
from .base import ContentSkillConfig, SkillEvidence, _overlap_ok, _pair_pose_deltas


# Azimuth bucket layout (8-way compass, tgt camera frame). Centers at
# 0, ±45, ±90, ±135, 180 deg; each bucket is 45 deg wide.
_AZIMUTH_BUCKETS = [
    (-22.5, 22.5,   "front"),
    (22.5,  67.5,   "front-right"),
    (67.5,  112.5,  "right"),
    (112.5, 157.5,  "back-right"),
    (-67.5, -22.5,  "front-left"),
    (-112.5, -67.5, "left"),
    (-157.5, -112.5,"back-left"),
]


def _azimuth_bucket(az_deg: float, hysteresis: float) -> Optional[str]:
    """Assign azimuth to a compass bucket with hysteresis near edges.

    Returns None if `az_deg` is within `hysteresis` degrees of any bucket
    edge — those pairs are too close to a label flip to be safe.
    """
    a = ((az_deg + 180.0) % 360.0) - 180.0
    if abs(abs(a) - 180.0) <= 22.5:
        dist_to_edge = 22.5 - abs(abs(a) - 180.0)
        if dist_to_edge < hysteresis:
            return None
        return "back"
    for lo, hi, name in _AZIMUTH_BUCKETS:
        if lo <= a < hi:
            if (a - lo) < hysteresis or (hi - a) < hysteresis:
                return None
            return name
    return None


def gate_relative_direction(
    pair: ViewPair, f_src: Frame, masks_src: list[ObjectMask],
    f_tgt: Frame, masks_tgt: list[ObjectMask], matches: list[Match],
    cfg: ContentSkillConfig,
) -> Optional[SkillEvidence]:
    if not _overlap_ok(pair, cfg):
        return None
    t, r = _pair_pose_deltas(f_src, f_tgt)
    if t < float(cfg.params.get("min_trans_m", 0.0)):
        return None
    if r < float(cfg.params.get("min_rot_deg", 0.0)):
        return None

    max_elev = float(cfg.params.get("max_elev_deg", 90.0))
    min_sep = float(cfg.params.get("min_azimuth_sep_deg", 0.0))
    hysteresis = float(cfg.params.get("bucket_hysteresis_deg", 10.0))

    C_tgt = camera_center(f_tgt.pose_c2w)
    axis_tgt = optical_axis(f_tgt.pose_c2w)
    up = np.array([0.0, 0.0, 1.0])

    axis_flat = axis_tgt - np.dot(axis_tgt, up) * up
    if np.linalg.norm(axis_flat) < 1e-6:
        return None
    axis_flat /= np.linalg.norm(axis_flat)

    entries: list[dict] = []
    for mi, m in enumerate(matches):
        if not m.visible:
            continue
        X = np.asarray(m.X_world, dtype=float)
        d = X - C_tgt
        nd = float(np.linalg.norm(d))
        if nd < 1e-6:
            continue
        d_unit = d / nd
        elev = float(np.degrees(np.arcsin(np.clip(float(np.dot(d_unit, up)), -1.0, 1.0))))
        if abs(elev) > max_elev:
            continue
        flat = d_unit - float(np.dot(d_unit, up)) * up
        n_flat = float(np.linalg.norm(flat))
        if n_flat < 1e-6:
            continue
        flat /= n_flat
        cos = float(np.clip(np.dot(flat, axis_flat), -1.0, 1.0))
        cross = float(np.cross(axis_flat, flat) @ up)
        az = float(np.degrees(np.arctan2(cross, cos)))
        bucket = _azimuth_bucket(az, hysteresis)
        if bucket is None:
            continue
        entries.append({
            "match_idx": mi,
            "label": masks_src[m.src_mask_idx].label,
            "azimuth_deg": round(az, 2),
            "elevation_deg": round(elev, 2),
            "bucket": bucket,
            "distance_m": round(nd, 4),
        })

    if not entries:
        return None

    azimuths = sorted(e["azimuth_deg"] for e in entries)
    spread_ok = (
        min_sep <= 0.0
        or (len(azimuths) >= 2 and max(
            azimuths[i + 1] - azimuths[i] for i in range(len(azimuths) - 1)
        ) >= min_sep)
    )
    if not spread_ok:
        return None

    return SkillEvidence(
        skill="relative_direction",
        qualifying_matches=[e["match_idx"] for e in entries],
        meta={
            "pair_rotation_deg": r,
            "pair_translation_m": t,
            "targets": entries,
        },
    )
