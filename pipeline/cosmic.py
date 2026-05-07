"""COSMIC-style pair gate (arXiv 2603.27183).

Layered on top of the regular pose+probe pair pipeline. Adds three
visibility-set predicates so the surviving pairs satisfy the COSMIC
benchmark's "shared anchor + near-complete coverage" property:

    1. O_src ∩ O_tgt ≠ ∅                       (a shared anchor exists)
    2. |O_src ∪ O_tgt| / |O_scene| >= alpha    (joint near-complete coverage)
    3. |yaw_src - yaw_tgt| >= beta             (genuinely two viewpoints)

`O_i` (per-frame visibility set) is computed from the adapter's GT
instance mask: an instance enters O_i iff its mask area fraction is at
least `area_min_frac` AND it has at least `depth_pix_min` valid depth
pixels (the mask-space analog of COSMIC's "≥3 of 8 corners inside
frustum AND unoccluded" rule).

`O_scene` is the union of O_i over all kept frames in the scene
(achievable analog of "all objects in the environment" — a real ScanNet
trajectory rarely covers every aggregation-file instance).

Pairs surviving the COSMIC gate are tagged with `tasks` restricted to
`COSMIC_SKILLS` (object-level skills only). Pixel-precise skills
(`cross_point_correspondence`) and pose-stage skills
(`cross_spatial_transformation`, etc.) are out of scope for this gate.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

from datasets.base import BaseSceneAdapter, Frame
from .geometry import optical_axis

logger = logging.getLogger(__name__)


COSMIC_SKILLS = frozenset({
    "cross_object_correspondence",
    "anchor",
    "counting",
    "relative_distance",
    "relative_direction",
})

# DEFAULT_LABEL_BLOCKLIST moved to pipeline/label_blocklist.py so detectors
# in `models/gt/` can import it without crossing into sampling code.
from .label_blocklist import DEFAULT_LABEL_BLOCKLIST  # noqa: F401


def floor_plane_yaw_deg(pose_c2w: np.ndarray) -> float:
    """Heading angle of the optical axis projected onto the floor plane.

    Returns yaw in degrees in [-180, 180]. Used to reject pairs whose
    cameras face nearly the same direction (independent-yaw analog of
    COSMIC's `Yaw ∼ U(-180°, 180°)` placement).
    """
    axis = optical_axis(pose_c2w)
    # ScanNet world is gravity-aligned; we use XY as the floor plane.
    # Empty floor projection (pure up/down look) → yaw=0 fallback.
    if abs(axis[0]) < 1e-9 and abs(axis[1]) < 1e-9:
        return 0.0
    return float(np.degrees(np.arctan2(axis[1], axis[0])))


def _yaw_diff_deg(a: float, b: float) -> float:
    """Smallest absolute angular distance between two yaws in degrees."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def compute_visibility_set(
    adapter: BaseSceneAdapter,
    frame: Frame,
    *,
    area_min_frac: float = 0.005,
    depth_pix_min: int = 50,
    label_blocklist: Optional[frozenset[str]] = None,
) -> Optional[frozenset[int]]:
    """Compute O_i: the set of instance IDs "visible enough" in this frame.

    Returns None if the adapter doesn't expose instance masks for this
    frame (we cannot apply COSMIC gates without GT segmentation, so
    callers should fall back or skip).

    `label_blocklist`: instance IDs whose label (lowercased) is in this
    set are dropped from O_i. Defaults to `DEFAULT_LABEL_BLOCKLIST`
    (walls, floors, ceilings) — structural surfaces that aren't
    object-level anchors.
    """
    if label_blocklist is None:
        label_blocklist = DEFAULT_LABEL_BLOCKLIST
    qc = getattr(adapter, "qc_instance_mask", None)
    if qc is None:
        return None
    out = qc(frame.frame_id)
    if out is None:
        return None
    inst_mask, label_map = out
    if inst_mask.size == 0:
        return frozenset()

    def _is_blocked(iid: int) -> bool:
        lab = label_map.get(int(iid), "")
        return (lab or "").strip().lower() in label_blocklist

    inst_mask = np.asarray(inst_mask)
    H, W = inst_mask.shape[:2]
    img_area = float(H * W)

    # Per-instance pixel counts (skip 0 = background / unannotated).
    ids, counts = np.unique(inst_mask, return_counts=True)
    keep = ids != 0
    ids = ids[keep]
    counts = counts[keep]
    if ids.size == 0:
        return frozenset()

    # Drop structural-surface labels first, then apply area gate to what
    # remains.
    label_pass = np.array([not _is_blocked(int(i)) for i in ids])
    area_pass = (counts / img_area >= area_min_frac) & label_pass
    if not np.any(area_pass):
        return frozenset()

    visible: set[int] = set()
    depth = frame.depth
    Wd, Hd = frame.depth_size
    # Pre-resample mask to depth resolution once so we count valid-depth
    # pixels per instance in the depth grid (avoids per-instance loops
    # over color-resolution coordinates).
    if depth is None:
        # No depth → can't enforce depth gate; accept area-only.
        for i, ok in zip(ids, area_pass):
            if ok:
                visible.add(int(i))
        return frozenset(visible)

    ys = (np.arange(Hd, dtype=np.float32) * H / Hd).astype(np.int32)
    xs = (np.arange(Wd, dtype=np.float32) * W / Wd).astype(np.int32)
    inst_at_depth = inst_mask[ys[:, None], xs[None, :]]
    valid_depth = depth > 0.0
    for i, ok in zip(ids, area_pass):
        if not ok:
            continue
        m = (inst_at_depth == i) & valid_depth
        if int(m.sum()) >= depth_pix_min:
            visible.add(int(i))
    return frozenset(visible)


def cosmic_filter(
    pairs: list,
    frames: dict[str, Frame],
    visibility: dict[str, frozenset[int]],
    scene_objects: frozenset[int],
    *,
    union_coverage_min: float,
    yaw_diff_min_deg: float,
    on_reject=None,
) -> list:
    """Reject pairs that don't satisfy the COSMIC visibility predicates.

    `pairs`: list[ViewPair]; mutated in `tasks` field for survivors so they
    only carry COSMIC_SKILLS. `on_reject(pair, reason)` is called per drop
    if provided (used by the CLI to log per-pair rejections).
    """
    if not scene_objects:
        # No GT instances at all in the scene's kept frames; the union
        # coverage ratio is undefined. Drop everything rather than
        # silently divide by zero.
        if on_reject is not None:
            for p in pairs:
                on_reject(p, "cosmic_no_scene_objects")
        return []

    n_scene = len(scene_objects)
    survivors = []
    n_no_anchor = n_low_cov = n_low_yaw = 0
    for p in pairs:
        O_src = visibility.get(p.src_id)
        O_tgt = visibility.get(p.tgt_id)
        if O_src is None or O_tgt is None:
            if on_reject is not None:
                on_reject(p, "cosmic_no_visibility")
            continue
        inter = O_src & O_tgt
        if not inter:
            n_no_anchor += 1
            if on_reject is not None:
                on_reject(p, "cosmic_no_shared_anchor")
            continue
        union = O_src | O_tgt
        cov = len(union) / n_scene
        if cov < union_coverage_min:
            n_low_cov += 1
            if on_reject is not None:
                on_reject(p, f"cosmic_low_union_coverage:{cov:.2f}")
            continue
        yaw_s = floor_plane_yaw_deg(frames[p.src_id].pose_c2w)
        yaw_t = floor_plane_yaw_deg(frames[p.tgt_id].pose_c2w)
        if _yaw_diff_deg(yaw_s, yaw_t) < yaw_diff_min_deg:
            n_low_yaw += 1
            if on_reject is not None:
                on_reject(p, "cosmic_low_yaw_diff")
            continue
        # Restrict tasks to object-level skills (COSMIC scope).
        p.tasks = frozenset(p.tasks & COSMIC_SKILLS) if p.tasks else frozenset()
        # Stash the visibility set sizes in case downstream code wants them.
        p.cosmic_meta = {  # type: ignore[attr-defined]
            "n_src": len(O_src), "n_tgt": len(O_tgt),
            "n_intersection": len(inter), "n_union": len(union),
            "n_scene": n_scene, "coverage": cov,
            "yaw_src": yaw_s, "yaw_tgt": yaw_t,
            "yaw_diff": _yaw_diff_deg(yaw_s, yaw_t),
        }
        survivors.append(p)
    logger.info(
        "cosmic gate: %d/%d pairs kept (drops: no_anchor=%d, low_coverage=%d, low_yaw=%d)",
        len(survivors), len(pairs), n_no_anchor, n_low_cov, n_low_yaw,
    )
    return survivors
