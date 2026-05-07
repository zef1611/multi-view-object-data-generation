"""View-pair sampling (threshold-driven, no hard caps).

Per scene:
  1. Frame keepset: greedy pose-delta gate (translation OR rotation).
  2. Pose pre-filter on C(N,2) candidates: drop pairs outside
     [angle_min, angle_max] or distance > max_distance.
  3. Cheap corner probe: 5x5 lattice reprojection → (in_bounds_frac,
     occluded_frac). Drop anything below `corner_overlap_min`.
  4. Quality score = overlap * angle_weight (triangular, peak at 30°).
     Drop below `pair_quality_min`.
  5. Diversity prune: quality-sorted greedy; reject candidates whose
     6-D pose signature [c_src, c_tgt] is within `pair_diversity_min_m`
     of any already-accepted pair.
  6. Per-task assignment (pose-stage only). Surviving pairs are tagged
     with any of the following that their probe statistics satisfy:
        - `cross_spatial_transformation` if angle_deg >= spatial_angle_min
        - `cross_depth_variation`        if max/min median-depth ratio
                                          >= depth_ratio_min
        - `cross_occlusion_visibility`   if overlap >= occlusion_overlap_min
                                          AND occluded_frac >= occlusion_fraction_min
     `cross_point_correspondence` (CrossPoint-378K's original
     `cross_correspondence`) and `cross_object_correspondence` are now
     CONTENT-stage skills (see `core/skill_gates.py`) — they require a
     valid labeled match, not just a surviving pair.

No per-scene / per-task pair caps: densely observed scenes naturally
yield more pairs, matching the heavy-tailed per-scene distribution of
the published CrossPoint-378K dataset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from datasets.base import BaseSceneAdapter, Frame, default_reproject_with_depth
from models._frame_ref import FrameRef
from .config import PairConfig, TaskGate
from .geometry import angle_between, camera_center, optical_axis
from .sampling import sample_keyframes, select_keyframes_adaptive

QualityFilter = Callable[[FrameRef], tuple[bool, str]]

logger = logging.getLogger(__name__)

CROSS_TASKS = (
    "cross_depth_variation",
    "cross_occlusion_visibility",
    "cross_spatial_transformation",
)


@dataclass
class ViewPair:
    src_id: str
    tgt_id: str
    overlap: float            # mean in-bounds fraction over the 5x5 probe
    occluded_frac: float      # fraction of in-bounds corners blocked by a closer tgt surface
    angle_deg: float
    distance_m: float
    quality: float            # overlap * angle_weight
    median_depth_src: float
    median_depth_tgt: float
    tasks: frozenset = field(default_factory=frozenset)


# ---- geometry helpers --------------------------------------------------

# Re-exported from .geometry for in-module use; external callers should
# import from pipeline.geometry directly.
_camera_center = camera_center
_optical_axis = optical_axis
_angle_between = angle_between


def _frame_gap(a: str, b: str) -> Optional[int]:
    """Temporal gap in frames, if both IDs parse as ints. Silently returns
    None for adapters with non-integer frame IDs (ScanNet++ DSLR timestamps),
    so the caller can skip the filter without branching on dataset type."""
    try:
        return abs(int(a) - int(b))
    except (TypeError, ValueError):
        return None


def _angle_weight(angle_deg: float, peak: float = 30.0,
                  lo: float = 10.0, hi: float = 80.0) -> float:
    """Triangular: 0 at `lo`/`hi`, 1 at `peak`, linear in between.

    Peak at 30° matches the CrossPoint-378K p50 (27.6°). The `hi=80°` and
    `lo=10°` bounds align with the global pose pre-filter."""
    if angle_deg <= lo or angle_deg >= hi:
        return 0.0
    if angle_deg <= peak:
        return (angle_deg - lo) / (peak - lo)
    return (hi - angle_deg) / (hi - peak)


def _sample_tgt_depth(tgt: Frame, u: float, v: float) -> float:
    W, H = tgt.image_size
    Wd, Hd = tgt.depth_size
    ud = int(round(u * Wd / W))
    vd = int(round(v * Hd / H))
    if not (0 <= ud < Wd and 0 <= vd < Hd):
        return 0.0
    return float(tgt.depth[vd, ud])


def _probe_pair(src: Frame, tgt: Frame, grid: int = 5,
                depth_tol_m: float = 0.10) -> tuple[float, float]:
    """5x5 lattice reprojection. Returns (in_bounds_frac, occluded_frac).

    `occluded_frac` counts corners that land in-bounds AND whose observed
    tgt depth is strictly closer than the predicted depth (i.e. a real
    occluder, not just missing depth).
    """
    W, H = src.image_size
    us = np.linspace(0, W - 1, grid)
    vs = np.linspace(0, H - 1, grid)
    total = 0
    in_bounds = 0
    occluded = 0
    for u in us:
        for v in vs:
            total += 1
            rep = default_reproject_with_depth(src, (float(u), float(v)), tgt)
            if rep is None or not rep.in_bounds:
                continue
            in_bounds += 1
            z_obs = _sample_tgt_depth(tgt, rep.u, rep.v)
            if z_obs > 0.0 and (rep.depth_pred - z_obs) > depth_tol_m:
                occluded += 1
    return in_bounds / total, occluded / total


def _median_depth(frame: Frame) -> float:
    d = frame.depth
    if d is None:
        return 0.0
    valid = d[d > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))



def _assign_tasks(pair: ViewPair, task_gates: dict[str, TaskGate]) -> frozenset:
    """Apply every per-task gate. A task is included iff all its gates pass.

    Gates with value 0.0 are treated as disabled. The frame-gap gate uses
    the parsed |f_src - f_tgt| when available (None → skipped, matching
    adapters with non-integer frame IDs).
    """
    gap = _frame_gap(pair.src_id, pair.tgt_id)
    d1, d2 = pair.median_depth_src, pair.median_depth_tgt
    depth_ratio = (max(d1, d2) / min(d1, d2)) if (d1 > 0.0 and d2 > 0.0) else 0.0

    tasks = set()
    for name, g in task_gates.items():
        if gap is not None and gap < g.min_frame_gap:
            continue
        if g.angle_deg_min > 0.0 and pair.angle_deg < g.angle_deg_min:
            continue
        if g.median_depth_ratio_min > 0.0 and depth_ratio < g.median_depth_ratio_min:
            continue
        if g.overlap_min > 0.0 and pair.overlap < g.overlap_min:
            continue
        if g.occluded_fraction_min > 0.0 and pair.occluded_frac < g.occluded_fraction_min:
            continue
        tasks.add(name)
    return frozenset(tasks)


# ---- pair selection entry point ---------------------------------------

def select_pairs(
    adapter: BaseSceneAdapter,
    config: PairConfig,
    *,
    adapter_name: Optional[str] = None,
    sampling: str = "adaptive",
    frame_stride: int = 50,
    min_keyframes: int = 30,
    min_translation_m: float = 0.40,
    min_rotation_deg: float = 25.0,
    limit_frames: Optional[int] = None,
    quality_filter: Optional[QualityFilter] = None,
    quality_filter_concurrency: int = 1,
    on_filter_drop=lambda fid, reason: None,
    cosmic_base_sampling: str = "stride",
    cosmic_union_coverage_min: float = 0.6,
    cosmic_yaw_diff_min_deg: float = 30.0,
    cosmic_obj_vis_area_min: float = 0.005,
    cosmic_obj_vis_depth_pix_min: int = 50,
    on_pair_reject=None,
) -> list[ViewPair]:
    """Select view-pairs for a scene, each annotated with its qualifying tasks.

    All quality / task / diversity thresholds come from `config`
    (`PairConfig`, typically built via `config.load_config` + `.resolve`).
    No hard cap: density is shaped by `pair_quality_min` + `pair_diversity_min_m`;
    per-task composition is shaped by per-task gates.

    `sampling="cosmic"` layers the COSMIC visibility-set gate on top of the
    base sampler (chosen via `cosmic_base_sampling`). Surviving pairs are
    tagged with object-level skills only (see `core.cosmic.COSMIC_SKILLS`).
    """
    angle_min = config.angle_min_deg
    angle_max = config.angle_max_deg
    max_distance = config.max_distance_m
    corner_overlap_min = config.corner_overlap_min
    pair_quality_min = config.pair_quality_min
    pair_diversity_min_m = config.pair_diversity_min_m
    min_yaw_diff_deg = config.min_yaw_diff_deg
    min_frame_gap = config.min_frame_gap_pre
    all_frames = adapter.list_frames()
    sampled, mode = sample_keyframes(
        adapter,
        sampling=sampling,
        frame_stride=frame_stride,
        min_keyframes=min_keyframes,
        min_translation_m=min_translation_m,
        min_rotation_deg=min_rotation_deg,
        limit_frames=limit_frames,
        cosmic_base_sampling=cosmic_base_sampling,
        cosmic_union_coverage_min=cosmic_union_coverage_min,
        cosmic_yaw_diff_min_deg=cosmic_yaw_diff_min_deg,
        log=False,
    )
    logger.info("[%s] %d frames -> %d after %s (avg 1:%.1f)",
                adapter.scene_id, len(all_frames), len(sampled), mode,
                len(all_frames) / max(len(sampled), 1))

    if quality_filter is not None:
        before = len(sampled)
        # Build refs once, single-threaded — `adapter.image_path` may
        # call `load_frame` (depth+pose I/O) on default adapters; running
        # that under ThreadPoolExecutor would multiply the disk traffic.
        refs_by_fid = {fid: adapter.frame_ref(fid, adapter_name)
                       for fid in sampled}

        def _check(fid):
            usable, reason = quality_filter(refs_by_fid[fid])
            return fid, usable, reason

        if quality_filter_concurrency > 1 and len(sampled) > 1:
            from concurrent.futures import ThreadPoolExecutor
            workers = min(quality_filter_concurrency, len(sampled))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                # Preserve input order so kept_q matches the sampling order.
                verdicts = list(ex.map(_check, sampled))
        else:
            verdicts = [_check(fid) for fid in sampled]

        kept_q: list[str] = []
        for fid, usable, reason in verdicts:
            if usable:
                kept_q.append(fid)
            else:
                on_filter_drop(fid, reason)
        sampled = kept_q
        logger.info("[%s] %d frames -> %d after Qwen quality filter (concurrency=%d)",
                    adapter.scene_id, before, len(sampled),
                    quality_filter_concurrency)

    # Cheap pose-only pass for the pose pre-filter.
    poses, valid_ids = {}, []
    for fid in sampled:
        f = adapter.load_frame(fid)
        if not np.all(np.isfinite(f.pose_c2w)):
            continue
        poses[fid] = f.pose_c2w
        valid_ids.append(fid)

    # Pre-compute floor-plane yaw per kept frame (cheap, O(N)).
    if min_yaw_diff_deg > 0.0:
        from .cosmic import _yaw_diff_deg, floor_plane_yaw_deg
        yaw_by_id = {fid: floor_plane_yaw_deg(poses[fid])
                     for fid in valid_ids}
    else:
        yaw_by_id = None

    pose_ok: list[tuple[str, str, float, float]] = []
    n_gap_dropped = 0
    n_yaw_dropped = 0
    for i, j in combinations(valid_ids, 2):
        if min_frame_gap > 0:
            gap = _frame_gap(i, j)
            if gap is not None and gap < min_frame_gap:
                n_gap_dropped += 1
                continue
        dist = float(np.linalg.norm(_camera_center(poses[i])
                                    - _camera_center(poses[j])))
        if dist > max_distance:
            continue
        angle = _angle_between(_optical_axis(poses[i]),
                               _optical_axis(poses[j]))
        if not (angle_min <= angle <= angle_max):
            continue
        if yaw_by_id is not None:
            if _yaw_diff_deg(yaw_by_id[i], yaw_by_id[j]) < min_yaw_diff_deg:
                n_yaw_dropped += 1
                continue
        pose_ok.append((i, j, dist, angle))
    logger.info(
        "[%s] %d pairs after pose pre-filter "
        "(%d dropped by frame-gap<%d, %d dropped by yaw<%.1f°)",
        adapter.scene_id, len(pose_ok), n_gap_dropped, min_frame_gap,
        n_yaw_dropped, min_yaw_diff_deg)

    # Full-frame (depth) loads only for survivors.
    needed = {fid for i, j, *_ in pose_ok for fid in (i, j)}
    frames = {fid: adapter.load_frame(fid) for fid in needed}
    median_depth = {fid: _median_depth(frames[fid]) for fid in needed}

    scored: list[ViewPair] = []
    for i, j, dist, angle in pose_ok:
        ov_ij, occ_ij = _probe_pair(frames[i], frames[j])
        ov_ji, occ_ji = _probe_pair(frames[j], frames[i])
        ov = (ov_ij + ov_ji) / 2.0
        occ = (occ_ij + occ_ji) / 2.0
        if ov < corner_overlap_min:
            continue
        q = ov * _angle_weight(angle)
        if q < pair_quality_min:
            continue
        scored.append(ViewPair(
            src_id=i, tgt_id=j,
            overlap=ov, occluded_frac=occ,
            angle_deg=angle, distance_m=dist, quality=q,
            median_depth_src=median_depth[i],
            median_depth_tgt=median_depth[j],
        ))
    logger.info("[%s] %d pairs after quality gate (q>=%.2f, overlap>=%.2f)",
                adapter.scene_id, len(scored), pair_quality_min, corner_overlap_min)

    accepted = _diversity_prune(scored, poses, pair_diversity_min_m)
    logger.info("[%s] %d pairs after diversity prune (d>=%.2fm)",
                adapter.scene_id, len(accepted), pair_diversity_min_m)

    # Per-task assignment (uses resolved per-source gates from the config).
    task_hist = {t: 0 for t in CROSS_TASKS}
    for p in accepted:
        p.tasks = _assign_tasks(p, config.tasks)
        for t in p.tasks:
            task_hist[t] = task_hist.get(t, 0) + 1
    logger.info("[%s] per-task pair counts (source=%s): %s",
                adapter.scene_id, config.source, task_hist)

    if sampling == "cosmic":
        # Lazy import: COSMIC gate uses GT instance masks and is only
        # active when explicitly requested.
        from .cosmic import (
            COSMIC_SKILLS, compute_visibility_set, cosmic_filter,
        )
        # Cache visibility sets; only frames referenced by surviving pairs
        # need to be checked.
        ref_ids = {fid for p in accepted for fid in (p.src_id, p.tgt_id)}
        vis: dict[str, frozenset[int]] = {}
        for fid in ref_ids:
            v = compute_visibility_set(
                adapter, frames[fid],
                area_min_frac=cosmic_obj_vis_area_min,
                depth_pix_min=cosmic_obj_vis_depth_pix_min,
            )
            if v is not None:
                vis[fid] = v
        scene_objects = frozenset().union(*vis.values()) if vis else frozenset()
        logger.info("[%s] cosmic visibility: %d frames, |O_scene|=%d",
                    adapter.scene_id, len(vis), len(scene_objects))
        accepted = cosmic_filter(
            accepted, frames, vis, scene_objects,
            union_coverage_min=cosmic_union_coverage_min,
            yaw_diff_min_deg=cosmic_yaw_diff_min_deg,
            on_reject=on_pair_reject,
        )
        # The pre-cosmic _assign_tasks may have tagged cross_* pose skills
        # the user doesn't want for this sampler; re-restrict here.
        for p in accepted:
            p.tasks = frozenset(p.tasks & COSMIC_SKILLS)
        logger.info("[%s] %d pairs after cosmic gate (alpha>=%.2f, yaw>=%.1f°)",
                    adapter.scene_id, len(accepted),
                    cosmic_union_coverage_min, cosmic_yaw_diff_min_deg)
    return accepted


def _pair_signature(pair: ViewPair, poses: dict) -> np.ndarray:
    return np.concatenate([_camera_center(poses[pair.src_id]),
                           _camera_center(poses[pair.tgt_id])])


def _diversity_prune(pairs: list[ViewPair], poses: dict,
                     diversity_min_m: float) -> list[ViewPair]:
    """Quality-sorted greedy: accept a pair iff its 6-D pose signature is
    >= `diversity_min_m` from every already-accepted pair.

    No cap — we keep as many pairs as remain distinct at the threshold.
    """
    if not pairs or diversity_min_m <= 0.0:
        return list(pairs)
    pairs = sorted(pairs, key=lambda p: p.quality, reverse=True)
    sigs = [_pair_signature(p, poses) for p in pairs]
    accepted_idx: list[int] = []
    for k, s in enumerate(sigs):
        if all(np.linalg.norm(s - sigs[a]) >= diversity_min_m
               for a in accepted_idx):
            accepted_idx.append(k)
    return [pairs[k] for k in accepted_idx]
