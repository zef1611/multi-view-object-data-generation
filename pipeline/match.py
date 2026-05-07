"""Cross-view matching: visibility + nearest-mask + IoU.

Given two frames with their SAM masks, for each src mask:
  1. Sample up to K random pixels in the mask.
  2. Reproject each via depth.
  3. Reject if out-of-bounds, depth-inconsistent, no tgt mask under it.
  4. Pick the nearest-centroid tgt mask M_tgt under p_tgt.
  5. Reproject the entire src mask into tgt; require IoU(M_src->tgt, M_tgt) >= 0.3.
  6. Accept the first surviving seed.

Returns one Match per accepted (src mask, tgt mask) pair.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from datasets.base import (
    BaseSceneAdapter, Frame, Reprojection, world_point_from_pixel,
)
from models.base import ObjectMask
from .project import mask_iou, reproject_mask
from .rng import make_rng

logger = logging.getLogger(__name__)


@dataclass
class Match:
    src_mask_idx: int
    tgt_mask_idx: int                # -1 when visible=False (occluder may not be a SAM mask)
    p_src: tuple[float, float]
    p_tgt: tuple[float, float]
    X_world: tuple[float, float, float]
    depth_src: float
    depth_pred_tgt: float
    depth_obs_tgt: float
    iou: float                       # 0.0 when visible=False
    seed_retry: int
    visible: bool = True             # False = occluded negative for visibility QA


def _erode_for_seeds(mask: np.ndarray, iterations: int = 5) -> np.ndarray:
    """Shrink a binary mask inward by `iterations` pixels via 3x3 erosion.

    Returns the eroded mask; caller must handle the fully-eroded case (tiny
    masks become empty). scipy is already a transitive dep of SAM/skimage
    so this is effectively free.
    """
    try:
        from scipy.ndimage import binary_erosion
    except ImportError:
        return mask
    return binary_erosion(mask, iterations=iterations)


def _depth_at_color_pixel(frame: Frame, u: float, v: float) -> float:
    Wd, Hd = frame.depth_size
    W, H = frame.image_size
    ud = int(round(u * Wd / W))
    vd = int(round(v * Hd / H))
    if not (0 <= ud < Wd and 0 <= vd < Hd):
        return 0.0
    return float(frame.depth[vd, ud])


def _nearest_mask_under(p: tuple[float, float],
                        masks: list[ObjectMask]) -> Optional[int]:
    """Return index of the tgt mask that contains p AND has the closest
    centroid to p. None if no mask contains p."""
    u, v = p
    ui, vi = int(round(u)), int(round(v))
    best_idx, best_dist = None, float("inf")
    for i, m in enumerate(masks):
        h, w = m.mask.shape
        if not (0 <= ui < w and 0 <= vi < h):
            continue
        if not bool(m.mask[vi, ui]):
            continue
        cx, cy = m.centroid
        d = (cx - u) ** 2 + (cy - v) ** 2
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def match_pair(
    adapter: BaseSceneAdapter,
    src_frame: Frame, src_masks: list[ObjectMask],
    tgt_frame: Frame, tgt_masks: list[ObjectMask],
    *,
    seed: int,
    seed_retries: int = 5,
    depth_tol_m: float = 0.15,
    iou_min: float = 0.30,
    emit_occlusion_negatives: bool = False,
    on_reject=lambda src_idx, reason: None,
) -> list[Match]:
    """Run the matching protocol over all src masks; return surviving Matches.

    `on_reject(src_mask_idx, reason)` is called once per src mask that fails
    after all retries.
    """
    rng = make_rng(seed, f"{adapter.scene_id}:{src_frame.frame_id}->{tgt_frame.frame_id}")

    # Pre-compute reprojected-mask cache key: for each (src_idx, tgt_idx)
    # the IoU only needs to be computed once per call.
    iou_cache: dict[tuple[int, int], float] = {}

    matches: list[Match] = []
    for s_idx, m_src in enumerate(src_masks):
        # Sample seeds from an eroded mask so they sit on object interior,
        # not on SAM's boundary halo where depth often reads the background
        # (a common cause of spurious "occluded" / low-IoU matches). Fall
        # back to the original mask when erosion empties it (tiny objects).
        seed_mask = _erode_for_seeds(m_src.mask)
        ys, xs = np.where(seed_mask)
        if xs.size == 0:
            ys, xs = np.where(m_src.mask)
        if xs.size == 0:
            on_reject(s_idx, "empty_mask")
            continue

        accepted: Optional[Match] = None
        # Track the first valid occluded reprojection (in-bounds + valid tgt
        # depth, but blocked by something closer) for visibility-negative emit.
        occluded_neg: Optional[Match] = None
        last_reason = "retries_exhausted"
        for retry in range(seed_retries):
            k = rng.randrange(xs.size)
            u_src, v_src = float(xs[k]), float(ys[k])

            z_src = _depth_at_color_pixel(src_frame, u_src, v_src)
            if z_src <= 0.0:
                last_reason = "bad_depth"; continue

            rep: Optional[Reprojection] = adapter.reproject(
                src_frame, (u_src, v_src), tgt_frame
            )
            if rep is None:
                last_reason = "bad_depth"; continue
            if not rep.in_bounds:
                last_reason = "out_of_bounds"; continue

            z_obs = _depth_at_color_pixel(tgt_frame, rep.u, rep.v)
            if z_obs <= 0.0:
                last_reason = "no_tgt_depth"; continue
            if abs(rep.depth_pred - z_obs) > depth_tol_m:
                last_reason = "occluded"
                # Capture the first occluded candidate as a visibility-negative
                # fallback. Only "true" occluders count: tgt observed depth
                # must be SMALLER (something closer is blocking the view).
                # Additionally require mask-level support: the src mask must
                # have non-zero IoU with SOME tgt mask. Without it, the
                # "occlusion" is almost always a mesh-edge leak rather than
                # a real occluder.
                if (emit_occlusion_negatives and occluded_neg is None
                        and z_obs < rep.depth_pred):
                    occluder_idx = _nearest_mask_under((rep.u, rep.v), tgt_masks)
                    # Compute / cache the projected-src→tgt-mask IoU. We
                    # check against the occluder if any, else against every
                    # tgt mask (some may not have an occluder under the
                    # reprojected pixel but still partially visible elsewhere).
                    proj_iou = 0.0
                    if occluder_idx is not None:
                        key = (s_idx, occluder_idx)
                        if key not in iou_cache:
                            proj = reproject_mask(src_frame, m_src.mask,
                                                  tgt_frame, subsample=4)
                            iou_cache[key] = mask_iou(
                                proj, tgt_masks[occluder_idx].mask)
                        proj_iou = iou_cache[key]
                    else:
                        # No occluder mask under the reprojected point; check
                        # if the projected src mask still overlaps any tgt
                        # mask (object visible elsewhere in tgt).
                        proj = reproject_mask(src_frame, m_src.mask,
                                              tgt_frame, subsample=4)
                        for ti, tm in enumerate(tgt_masks):
                            iv = mask_iou(proj, tm.mask)
                            if iv > proj_iou:
                                proj_iou = iv
                    if proj_iou > 0.0:
                        X_neg = world_point_from_pixel(src_frame, u_src, v_src)
                        if X_neg is not None:
                            occluded_neg = Match(
                                src_mask_idx=s_idx,
                                tgt_mask_idx=occluder_idx if occluder_idx is not None else -1,
                                p_src=(u_src, v_src), p_tgt=(rep.u, rep.v),
                                X_world=tuple(float(c) for c in X_neg),
                                depth_src=z_src, depth_pred_tgt=rep.depth_pred,
                                depth_obs_tgt=z_obs, iou=proj_iou,
                                seed_retry=retry, visible=False,
                            )
                    else:
                        last_reason = "occluded_no_mask_support"
                continue

            t_idx = _nearest_mask_under((rep.u, rep.v), tgt_masks)
            if t_idx is None:
                last_reason = "no_tgt_mask"; continue

            key = (s_idx, t_idx)
            if key not in iou_cache:
                proj = reproject_mask(src_frame, m_src.mask, tgt_frame, subsample=4)
                iou_cache[key] = mask_iou(proj, tgt_masks[t_idx].mask)
            iou = iou_cache[key]
            if iou < iou_min:
                last_reason = "low_iou"; continue

            X = world_point_from_pixel(src_frame, u_src, v_src)
            assert X is not None
            accepted = Match(
                src_mask_idx=s_idx, tgt_mask_idx=t_idx,
                p_src=(u_src, v_src), p_tgt=(rep.u, rep.v),
                X_world=tuple(float(c) for c in X),
                depth_src=z_src, depth_pred_tgt=rep.depth_pred,
                depth_obs_tgt=z_obs, iou=iou, seed_retry=retry,
                visible=True,
            )
            break

        if accepted is not None:
            matches.append(accepted)
        elif occluded_neg is not None:
            matches.append(occluded_neg)
        else:
            on_reject(s_idx, last_reason)
    return matches
