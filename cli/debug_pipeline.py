"""Skill-agnostic per-stage pipeline tracer.

For one (scene, skill), runs every pipeline stage and captures what's
kept vs dropped at each one. Emits:

    {out}/<scene>/dashboard.png        # one-page visual summary
    {out}/<scene>/trace.jsonl          # per-pair / per-frame event log

Stages traced (numbering matches README):

    1. Frame sampling
    2. Per-frame Qwen3-VL quality filter
    3. Pose pre-filter        (frame_gap, distance, angle bounds)
    4. Quality gate           (corner_overlap, pair_quality)
    5. Diversity prune        (6-D pose signature)
    6. Per-task pose-stage assignment
    7. Perception             (detector + segmenter, cached)
    8. Geometric matching     (per src mask)
    9. Per-skill content gate (the chosen --skill)

Usage:

    python pipeline_debug.py \\
        --scene scene0012_00 \\
        --skill cross_point_correspondence \\
        --out outputs/pipeline_debug
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from datasets.scannet import ScanNetAdapter
from pipeline import pairs as pairs_mod
from pipeline.config import load_config, resolve
from pipeline.match import match_pair
from pipeline.pairs import (
    ViewPair, _angle_between, _angle_weight, _assign_tasks, _camera_center,
    _frame_gap, _median_depth, _optical_axis, _pair_signature, _probe_pair,
    select_keyframes_adaptive,
)
from pipeline.skills import (
    CONTENT_GATES, POSE_EVIDENCE, load_content_skills,
)
from models.gt.scannet import ScanNetGTDetector
from models.segmenters.gt import GTMaskSegmenter
from viz import color_for
from cli import generate as _gen  # for make_detector / make_segmenter


# ---- pipeline tracer ---------------------------------------------------

_QWEN_FILTER_ROOT = Path("cache/filter")


def _qwen_filter_subdirs() -> list[Path]:
    """Memoized list of model-tagged dirs under ``cache/filter/``."""
    if not _QWEN_FILTER_ROOT.exists():
        return []
    return sorted(d for d in _QWEN_FILTER_ROOT.iterdir() if d.is_dir())


def _qwen_lookup(adapter_name: str, scene_id: str, frame_id: str,
                 spec_dirs: Optional[list[Path]] = None):
    """Read any cached filter verdict for a frame. Returns the first hit
    among model-tagged subdirs. Debug-only; the main pipeline uses
    ``QwenFilter`` directly."""
    if spec_dirs is None:
        spec_dirs = _qwen_filter_subdirs()
    rel = f"{adapter_name}/{scene_id}/{frame_id}.json"
    for sub in spec_dirs:
        cp = sub / rel
        if not cp.exists():
            continue
        try:
            d = json.loads(cp.read_text())
            return bool(d["usable"]), str(d["reason"])
        except Exception:
            continue
    return None


def _load_qwen_filter():
    """Debug helper: this script no longer drives the live filter — it
    only reads from `cache/filter/<model>/`. The main pipeline does the
    live filtering. Returning None here forces cache-only lookup
    (`_qwen_lookup`)."""
    return None


def trace_pipeline(scene_id: str, skill: str, *,
                    sampling: str = "stride",
                    frame_stride: int = 50,
                    min_keyframes: int = 30,
                    limit_frames: int = 0,
                    use_qwen: bool = True,
                    min_frame_masks: int = 3,
                    detector_name: str = "scannet-gt",
                    segmenter_name: str = "gt-mask",
                    scenes_root: Path = Path("/home/mila/l/leh/scratch/dataset/scannet_data/scans")):
    adapter = ScanNetAdapter(scenes_root / scene_id)
    cfg_all = load_config()
    pcfg = resolve(cfg_all, "scannet")
    content_skills = load_content_skills(cfg_all)

    skill_cfg = content_skills.get(skill)  # None for pose-stage skills
    is_pose_skill = skill in POSE_EVIDENCE

    # ---- stage 1: sampling -------------------------------------------------
    all_frames = adapter.list_frames()
    if sampling == "stride":
        if min_keyframes > 0 and len(all_frames) > 0:
            implied_stride = max(1, len(all_frames) // min_keyframes)
            effective_stride = min(frame_stride, implied_stride)
        else:
            effective_stride = frame_stride
        sampled = all_frames[::effective_stride]
        s1_mode = (f"stride={effective_stride}" if effective_stride == frame_stride
                   else f"stride={effective_stride} (floored from {frame_stride})")
    elif sampling == "adaptive":
        sampled = select_keyframes_adaptive(adapter)
        s1_mode = "adaptive"
    else:
        raise ValueError(sampling)
    if limit_frames is not None and limit_frames > 0:
        sampled = sampled[:limit_frames]
        s1_mode += f" (limit={limit_frames})"

    # ---- stage 2: qwen quality filter --------------------------------------
    qwen_events = []
    kept_q = []
    qf = _load_qwen_filter() if use_qwen else None
    qf_subdirs = _qwen_filter_subdirs()
    for fid in sampled:
        ref = adapter.frame_ref(fid)
        ip = ref.image_path
        if qf is not None:
            usable, reason = qf.is_usable(ref)
        else:
            cached = _qwen_lookup(ref.adapter, scene_id, fid,
                                  spec_dirs=qf_subdirs)
            if cached is None:
                usable, reason = True, "qwen_unavailable"
            else:
                usable, reason = cached
        qwen_events.append({"fid": fid, "usable": usable,
                            "reason": reason, "image": str(ip)})
        if usable:
            kept_q.append(fid)

    # ---- stage 2b: per-frame GT-mask-count filter --------------------------
    # Drop "empty content" frames where the detector found < min_frame_masks.
    detector = _gen.make_detector(detector_name)
    if hasattr(detector, "set_adapter"):
        detector.set_adapter(adapter)
    segmenter = _gen.make_segmenter(segmenter_name)
    if hasattr(segmenter, "set_adapter"):
        segmenter.set_adapter(adapter)
    perc_cache: dict[str, list] = {}

    def perc(fid: str):
        if fid in perc_cache:
            return perc_cache[fid]
        ref = adapter.frame_ref(fid)
        dets = detector.detect(ref)
        masks = segmenter.segment(ref.image_path, dets)
        canon = getattr(detector, "canonicalize_mask_label", None)
        if callable(canon):
            for m in masks:
                m.canonical = canon(m.label)
        perc_cache[fid] = masks
        return masks

    mask_count_events = []
    kept_mc: list[str] = []
    for fid in kept_q:
        n = len(perc(fid))
        passed = n >= min_frame_masks
        mask_count_events.append({"fid": fid, "n_masks": n, "kept": passed})
        if passed:
            kept_mc.append(fid)

    # ---- stage 3: pose pre-filter ------------------------------------------
    poses, valid_ids = {}, []
    for fid in kept_mc:
        try:
            f = adapter.load_frame(fid)
        except Exception:
            continue
        if not np.all(np.isfinite(f.pose_c2w)):
            continue
        poses[fid] = f.pose_c2w
        valid_ids.append(fid)

    if pcfg.min_yaw_diff_deg > 0.0:
        from pipeline.cosmic import (
            _yaw_diff_deg, floor_plane_yaw_deg,
        )
        yaw_by_id = {fid: floor_plane_yaw_deg(poses[fid]) for fid in valid_ids}
    else:
        yaw_by_id = None

    pose_events = []
    pose_ok = []
    for i, j in combinations(valid_ids, 2):
        gap = _frame_gap(i, j)
        dist = float(np.linalg.norm(_camera_center(poses[i])
                                    - _camera_center(poses[j])))
        angle = float(_angle_between(_optical_axis(poses[i]),
                                      _optical_axis(poses[j])))
        yaw_diff = (_yaw_diff_deg(yaw_by_id[i], yaw_by_id[j])
                    if yaw_by_id is not None else None)
        drop = None
        if pcfg.min_frame_gap_pre > 0 and gap is not None and gap < pcfg.min_frame_gap_pre:
            drop = f"frame_gap<{pcfg.min_frame_gap_pre}"
        elif dist > pcfg.max_distance_m:
            drop = f"distance>{pcfg.max_distance_m}"
        elif not (pcfg.angle_min_deg <= angle <= pcfg.angle_max_deg):
            drop = (f"angle_oob<{pcfg.angle_min_deg}"
                    if angle < pcfg.angle_min_deg
                    else f"angle_oob>{pcfg.angle_max_deg}")
        elif yaw_diff is not None and yaw_diff < pcfg.min_yaw_diff_deg:
            drop = f"yaw_diff<{pcfg.min_yaw_diff_deg}"
        pose_events.append({"i": i, "j": j, "gap": gap,
                            "dist": dist, "angle": angle,
                            "yaw_diff": yaw_diff, "drop": drop})
        if drop is None:
            pose_ok.append((i, j, dist, angle))

    # ---- stage 4: quality gate ---------------------------------------------
    needed = {fid for i, j, *_ in pose_ok for fid in (i, j)}
    frames = {fid: adapter.load_frame(fid) for fid in needed}
    median_depth = {fid: _median_depth(frames[fid]) for fid in needed}

    quality_events = []
    quality_ok = []
    for i, j, dist, angle in pose_ok:
        ov_ij, occ_ij = _probe_pair(frames[i], frames[j])
        ov_ji, occ_ji = _probe_pair(frames[j], frames[i])
        ov = (ov_ij + ov_ji) / 2.0
        occ = (occ_ij + occ_ji) / 2.0
        q = ov * _angle_weight(angle)
        drop = None
        if ov < pcfg.corner_overlap_min:
            drop = f"overlap<{pcfg.corner_overlap_min}"
        elif q < pcfg.pair_quality_min:
            drop = f"quality<{pcfg.pair_quality_min}"
        quality_events.append({"i": i, "j": j, "overlap": ov, "occluded_frac": occ,
                                "angle": angle, "dist": dist, "quality": q,
                                "drop": drop})
        if drop is None:
            vp = ViewPair(
                src_id=i, tgt_id=j, overlap=ov, occluded_frac=occ,
                angle_deg=angle, distance_m=dist, quality=q,
                median_depth_src=median_depth[i],
                median_depth_tgt=median_depth[j],
            )
            quality_ok.append(vp)

    # ---- stage 5: diversity prune ------------------------------------------
    sorted_pairs = sorted(quality_ok, key=lambda p: p.quality, reverse=True)
    sigs = [_pair_signature(p, poses) for p in sorted_pairs]
    accepted_idx = []
    diversity_events = []
    for k, s in enumerate(sigs):
        dominator = None
        for a in accepted_idx:
            d = float(np.linalg.norm(s - sigs[a]))
            if d < pcfg.pair_diversity_min_m:
                dominator = (a, d)
                break
        kept = dominator is None
        diversity_events.append({
            "i": sorted_pairs[k].src_id, "j": sorted_pairs[k].tgt_id,
            "quality": float(sorted_pairs[k].quality),
            "kept": kept,
            "dominator": (sorted_pairs[dominator[0]].src_id + "->"
                          + sorted_pairs[dominator[0]].tgt_id) if dominator else None,
            "dominator_dist_m": dominator[1] if dominator else None,
        })
        if kept:
            accepted_idx.append(k)
    accepted = [sorted_pairs[k] for k in accepted_idx]

    # ---- stage 6: per-task pose-stage assignment ---------------------------
    task_events = []
    for p in accepted:
        p.tasks = _assign_tasks(p, pcfg.tasks)
        task_events.append({"i": p.src_id, "j": p.tgt_id,
                            "tasks": sorted(p.tasks),
                            "in_skill": skill in p.tasks if is_pose_skill else None})

    # ---- stages 7+8: matching (perception already done in stage 2b) --------
    perception_events = {fid: {"n_masks": len(perc(fid))} for fid in needed}

    match_events = []
    match_results = {}  # (i,j) -> matches list (used in stage 9)
    for p in accepted:
        ms_src = perc(p.src_id)
        ms_tgt = perc(p.tgt_id)
        rejects: list[str] = []
        matches = match_pair(
            adapter, frames[p.src_id], ms_src,
            frames[p.tgt_id], ms_tgt,
            seed=42, seed_retries=5,
            depth_tol_m=0.15, iou_min=0.20,
            emit_occlusion_negatives=True,
            on_reject=lambda _idx, reason: rejects.append(reason),
        )
        n_vis = sum(1 for m in matches if m.visible)
        n_occ = sum(1 for m in matches if not m.visible)
        match_events.append({
            "i": p.src_id, "j": p.tgt_id,
            "n_visible": n_vis, "n_occluded": n_occ,
            "rejects": dict(zip(*np.unique(rejects, return_counts=True)))
                if rejects else {},
        })
        match_results[(p.src_id, p.tgt_id)] = matches

    # ---- stage 9: skill-specific content gate ------------------------------
    skill_events = []
    emit_payloads: dict[tuple[str, str], dict] = {}
    if is_pose_skill:
        extractor = POSE_EVIDENCE[skill]
        for p in accepted:
            tagged = skill in p.tasks
            ev = None
            if tagged:
                ev = extractor(p, frames[p.src_id], perc(p.src_id),
                                frames[p.tgt_id], perc(p.tgt_id),
                                match_results[(p.src_id, p.tgt_id)])
            skill_events.append({
                "i": p.src_id, "j": p.tgt_id,
                "tagged_in_step6": tagged,
                "evidence": ev is not None,
                "n_qualifying": len(ev.qualifying_matches) if ev else 0,
                "drop_reason": (None if ev is not None
                                else "not_tagged" if not tagged
                                else "extractor_returned_None"),
            })
            if ev is not None:
                emit_payloads[(p.src_id, p.tgt_id)] = {
                    "matches": match_results[(p.src_id, p.tgt_id)],
                    "evidence": ev,
                    "src_size": frames[p.src_id].image_size,
                    "tgt_size": frames[p.tgt_id].image_size,
                    "src_masks": perc(p.src_id),
                    "tgt_masks": perc(p.tgt_id),
                }
    elif skill_cfg is not None:
        gate = CONTENT_GATES[skill]
        for p in accepted:
            ev = gate(p, frames[p.src_id], perc(p.src_id),
                       frames[p.tgt_id], perc(p.tgt_id),
                       match_results[(p.src_id, p.tgt_id)], skill_cfg)
            skill_events.append({
                "i": p.src_id, "j": p.tgt_id,
                "evidence": ev is not None,
                "n_qualifying": len(ev.qualifying_matches) if ev else 0,
                "drop_reason": None if ev is not None else "content_gate_failed",
            })
            if ev is not None:
                emit_payloads[(p.src_id, p.tgt_id)] = {
                    "matches": match_results[(p.src_id, p.tgt_id)],
                    "evidence": ev,
                    "src_size": frames[p.src_id].image_size,
                    "tgt_size": frames[p.tgt_id].image_size,
                    "src_masks": perc(p.src_id),
                    "tgt_masks": perc(p.tgt_id),
                }
    else:
        raise SystemExit(f"unknown skill: {skill}")

    return {
        "scene_id": scene_id, "skill": skill,
        "sampling": s1_mode,
        "_detector": detector_name,
        "_segmenter": segmenter_name,
        "_perception_cache": dict(perc_cache),     # not JSON-serialized
        "stage_1_sampling": {
            "n_total": len(all_frames), "n_kept": len(sampled),
            "kept_ids": sampled,
        },
        "stage_2_qwen": {
            "n_in": len(sampled), "n_kept": len(kept_q),
            "events": qwen_events,
        },
        "stage_2b_mask_count": {
            "min_frame_masks": min_frame_masks,
            "n_in": len(kept_q), "n_kept": len(kept_mc),
            "events": mask_count_events,
        },
        "stage_3_pose_prefilter": {
            "n_in": len(valid_ids) * (len(valid_ids) - 1) // 2,
            "n_kept": len(pose_ok), "events": pose_events,
        },
        "stage_4_quality_gate": {
            "n_in": len(pose_ok), "n_kept": len(quality_ok),
            "events": quality_events,
        },
        "stage_5_diversity_prune": {
            "n_in": len(quality_ok), "n_kept": len(accepted),
            "events": diversity_events,
        },
        "stage_6_task_assignment": {
            "n_in": len(accepted),
            "events": task_events,
        },
        "stage_7_perception": perception_events,
        "stage_8_match": match_events,
        "stage_9_skill_gate": {
            "skill": skill, "is_pose_skill": is_pose_skill,
            "n_in": len(accepted),
            "n_kept": sum(1 for e in skill_events if e["evidence"]),
            "events": skill_events,
        },
        "_emit_payloads": emit_payloads,    # not JSON-serialized
    }


# ---- visualization -----------------------------------------------------

def _flow_summary(trace: dict) -> list[tuple[str, int]]:
    return [
        ("raw frames",         trace["stage_1_sampling"]["n_total"]),
        (f"sampled ({trace['sampling']})", trace["stage_1_sampling"]["n_kept"]),
        ("after Qwen3-VL",     trace["stage_2_qwen"]["n_kept"]),
        (f"after mask-count ≥ {trace['stage_2b_mask_count']['min_frame_masks']}",
         trace["stage_2b_mask_count"]["n_kept"]),
        ("pairs after pose pre-filter", trace["stage_3_pose_prefilter"]["n_kept"]),
        ("pairs after quality gate",    trace["stage_4_quality_gate"]["n_kept"]),
        ("pairs after diversity prune", trace["stage_5_diversity_prune"]["n_kept"]),
        (f"pairs surviving '{trace['skill']}' gate",
         trace["stage_9_skill_gate"]["n_kept"]),
    ]


_THUMB_W, _THUMB_H = 220, 165   # bigger for the stage-9 emit panel
_GAP = 6


def _draw_dot(draw, x, y, color, r=5):
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline="black")


def _draw_circle(draw, x, y, color, r=6):
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)


def _draw_x(draw, x, y, color, r=6):
    draw.line((x - r, y - r, x + r, y + r), fill=color, width=2)
    draw.line((x - r, y + r, x + r, y - r), fill=color, width=2)


def _overlay_mask(half_rgb: np.ndarray, mask_full: np.ndarray,
                   color: tuple[int, int, int], alpha: float = 0.40
                   ) -> np.ndarray:
    """Resize a full-resolution binary mask to thumbnail size and alpha-blend
    `color` onto `half_rgb` wherever the mask is True. Returns a new array."""
    if mask_full is None or mask_full.size == 0:
        return half_rgb
    from PIL import Image
    pim = Image.fromarray((mask_full.astype(np.uint8) * 255), mode="L")
    pim = pim.resize((half_rgb.shape[1], half_rgb.shape[0]), Image.NEAREST)
    m = np.array(pim) > 127
    out = half_rgb.copy()
    cvec = np.array(color, dtype=np.float32)
    out[m] = (out[m].astype(np.float32) * (1 - alpha)
              + cvec * alpha).clip(0, 255).astype(np.uint8)
    return out


def _annotate_cpc(pair_img: np.ndarray, matches, qualifying: list[int],
                   occluded: list[int], src_size: tuple[int, int],
                   tgt_size: tuple[int, int],
                   src_masks: list, tgt_masks: list) -> np.ndarray:
    """Overlay GT masks + POS/NEG points on a pair thumbnail.

    POS (visible) → green-tinted GT mask + green dot/circle.
    NEG (occluded) → red-tinted GT mask + red dot/✗.
    """
    from PIL import Image, ImageDraw
    src_half = pair_img[:, :_THUMB_W].copy()
    tgt_half = pair_img[:, _THUMB_W + _GAP:].copy()

    GREEN = (0, 200, 0)
    RED = (220, 30, 30)

    for mi in qualifying:
        m = matches[mi]
        if 0 <= m.src_mask_idx < len(src_masks):
            src_half = _overlay_mask(src_half,
                                      src_masks[m.src_mask_idx].mask, GREEN)
        if 0 <= m.tgt_mask_idx < len(tgt_masks):
            tgt_half = _overlay_mask(tgt_half,
                                      tgt_masks[m.tgt_mask_idx].mask, GREEN)
    for mi in occluded:
        m = matches[mi]
        if 0 <= m.src_mask_idx < len(src_masks):
            src_half = _overlay_mask(src_half,
                                      src_masks[m.src_mask_idx].mask, RED)
        if 0 <= m.tgt_mask_idx < len(tgt_masks):
            tgt_half = _overlay_mask(tgt_half,
                                      tgt_masks[m.tgt_mask_idx].mask, RED,
                                      alpha=0.30)

    gap = np.full((_THUMB_H, _GAP, 3), 255, dtype=np.uint8)
    new_img = np.concatenate([src_half, gap, tgt_half], axis=1)

    im = Image.fromarray(new_img)
    draw = ImageDraw.Draw(im)
    src_W, src_H = src_size
    tgt_W, tgt_H = tgt_size
    sx = _THUMB_W / max(1, src_W)
    sy = _THUMB_H / max(1, src_H)
    tx = _THUMB_W / max(1, tgt_W)
    ty = _THUMB_H / max(1, tgt_H)
    tgt_offset = _THUMB_W + _GAP
    for mi in qualifying:
        m = matches[mi]
        u, v = m.p_src
        _draw_dot(draw, int(u * sx), int(v * sy), "lime")
        u, v = m.p_tgt
        _draw_circle(draw, int(u * tx) + tgt_offset, int(v * ty), "lime")
    for mi in occluded:
        m = matches[mi]
        u, v = m.p_src
        _draw_dot(draw, int(u * sx), int(v * sy), "red")
        u, v = m.p_tgt
        _draw_x(draw, int(u * tx) + tgt_offset, int(v * ty), "red")
    return np.array(im)


def _frame_thumb(image_path: str):
    try:
        return np.array(Image.open(image_path).resize((_THUMB_W, _THUMB_H)))
    except Exception:
        return None


def _pair_image(adapter_image_dir: Path, fa: str, fb: str):
    """Return a single H × 2W RGB array (src | gap | tgt)."""
    a = _frame_thumb(adapter_image_dir / f"{fa}.jpg")
    b = _frame_thumb(adapter_image_dir / f"{fb}.jpg")
    if a is None or b is None:
        return None
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if b.ndim == 2:
        b = np.stack([b] * 3, axis=-1)
    a = a[..., :3]; b = b[..., :3]
    gap = np.full((_THUMB_H, _GAP, 3), 255, dtype=np.uint8)
    return np.concatenate([a, gap, b], axis=1)


def _grid_panel(ax, title: str, pair_imgs: list[tuple[np.ndarray, str]],
                cols: int = 6, pad: float = 0.012):
    """Render a grid of pair thumbnails on `ax` with padding between cells.

    `pad` is a fraction of the panel's width/height left blank between cells.
    """
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    if not pair_imgs:
        ax.text(0.5, 0.5, "(no pairs at this stage)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, family="monospace")
        return
    n = len(pair_imgs)
    rows = int(np.ceil(n / cols))
    cell_w = 1.0 / cols
    cell_h = 1.0 / rows
    inner_w = cell_w - 2 * pad
    inner_h = cell_h - 2 * pad
    cap_h = 0.18 * inner_h         # caption strip at the bottom of each cell
    img_h = inner_h - cap_h
    for k, (img, caption) in enumerate(pair_imgs):
        r, c = divmod(k, cols)
        x0 = c * cell_w + pad
        y0 = 1.0 - (r + 1) * cell_h + pad
        ax.imshow(img, extent=(x0, x0 + inner_w,
                                y0 + cap_h, y0 + cap_h + img_h),
                  transform=ax.transAxes, aspect="auto")
        ax.text(x0 + inner_w / 2, y0 + cap_h / 2, caption,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7, family="monospace",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="black", lw=0.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def _sample(seq, k):
    """Sample up to k items uniformly across seq, preserving order."""
    if len(seq) <= k:
        return list(seq)
    idx = np.linspace(0, len(seq) - 1, k).round().astype(int)
    return [seq[i] for i in idx]


def _full_pair_image(img_dir: Path, fa: str, fb: str, gap_px: int = 16):
    """Load src+tgt at FULL resolution and concatenate horizontally."""
    try:
        a = np.array(Image.open(img_dir / f"{fa}.jpg").convert("RGB"))
        b = np.array(Image.open(img_dir / f"{fb}.jpg").convert("RGB"))
    except Exception:
        return None
    h = max(a.shape[0], b.shape[0])
    if a.shape[0] != h:
        a = np.array(Image.fromarray(a).resize(
            (a.shape[1], h), Image.BILINEAR))
    if b.shape[0] != h:
        b = np.array(Image.fromarray(b).resize(
            (b.shape[1], h), Image.BILINEAR))
    gap = np.full((h, gap_px, 3), 255, dtype=np.uint8)
    return np.concatenate([a, gap, b], axis=1), a.shape, b.shape, gap_px


def _annotate_cpc_full(pair_img, src_shape, tgt_shape, gap_px,
                        matches, qualifying, occluded,
                        src_masks, tgt_masks):
    """Full-resolution version of _annotate_cpc — no scaling, marker sizes
    in absolute pixels."""
    src_H, src_W = src_shape[:2]
    tgt_H, tgt_W = tgt_shape[:2]
    src_half = pair_img[:, :src_W].copy()
    tgt_half = pair_img[:, src_W + gap_px:].copy()

    GREEN = (0, 200, 0)
    RED = (220, 30, 30)

    for mi in qualifying:
        m = matches[mi]
        if 0 <= m.src_mask_idx < len(src_masks):
            src_half = _overlay_mask(src_half,
                                      src_masks[m.src_mask_idx].mask, GREEN)
        if 0 <= m.tgt_mask_idx < len(tgt_masks):
            tgt_half = _overlay_mask(tgt_half,
                                      tgt_masks[m.tgt_mask_idx].mask, GREEN)
    for mi in occluded:
        m = matches[mi]
        if 0 <= m.src_mask_idx < len(src_masks):
            src_half = _overlay_mask(src_half,
                                      src_masks[m.src_mask_idx].mask, RED)
        if 0 <= m.tgt_mask_idx < len(tgt_masks):
            tgt_half = _overlay_mask(tgt_half,
                                      tgt_masks[m.tgt_mask_idx].mask, RED,
                                      alpha=0.30)

    gap = np.full((src_H, gap_px, 3), 255, dtype=np.uint8)
    new_img = np.concatenate([src_half, gap, tgt_half], axis=1)
    from PIL import Image, ImageDraw
    im = Image.fromarray(new_img)
    draw = ImageDraw.Draw(im)
    tgt_offset = src_W + gap_px
    for mi in qualifying:
        m = matches[mi]
        u, v = m.p_src
        _draw_dot(draw, int(u), int(v), "lime", r=10)
        u, v = m.p_tgt
        _draw_circle(draw, int(u) + tgt_offset, int(v), "lime", r=14)
    for mi in occluded:
        m = matches[mi]
        u, v = m.p_src
        _draw_dot(draw, int(u), int(v), "red", r=10)
        u, v = m.p_tgt
        _draw_x(draw, int(u) + tgt_offset, int(v), "red", r=14)
    return np.array(im)


def _save_pair_png(arr: np.ndarray, out_path: Path,
                    caption: str | None = None) -> None:
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(arr)
    if caption:
        draw = ImageDraw.Draw(im)
        # White strip at the top, 28 px tall.
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
        # Translucent black bar at top-left.
        bbox = draw.textbbox((0, 0), caption, font=font)
        pad = 6
        draw.rectangle((0, 0, bbox[2] + 2 * pad, bbox[3] + 2 * pad),
                        fill=(0, 0, 0))
        draw.text((pad, pad), caption, fill="white", font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, optimize=True)


def save_per_pair_files(trace: dict, out_dir: Path,
                         scenes_root: Path = Path("/home/mila/l/leh/scratch/dataset/scannet_data/scans"),
                         stages: tuple[int, ...] = (5,)) -> None:
    """Write one PNG per kept pair at every stage so the user can scroll
    through the full set rather than the dashboard's 18-pair sample.

    Layout under `out_dir`:
        stage_3_pose_prefilter/<src>_<tgt>.png        (1231 pairs on s0012)
        stage_4_quality_gate/<src>_<tgt>.png          (134 pairs)
        stage_5_diversity_prune/<src>_<tgt>.png       (56 pairs)
        stage_6_task_assignment/<src>_<tgt>.png       (only kept + tagged)
        stage_7_8_match/<src>_<tgt>.png               (with vis/occ counts)
        stage_9_emit/<src>_<tgt>.png                  (full GT mask + markers)

    Stage 9 PNGs are full-resolution annotated; earlier stages are full
    resolution but unannotated (just the raw RGB pair + caption).
    """
    img_dir = scenes_root / trace["scene_id"] / "color"
    skill = trace["skill"]
    payloads = trace.get("_emit_payloads", {})

    def _save_unannotated(stage_dir, e, caption_fn):
        res = _full_pair_image(img_dir, e["i"], e["j"])
        if res is None:
            return
        pair_img, _src_sh, _tgt_sh, _gp = res
        _save_pair_png(pair_img, stage_dir / f"{e['i']}_{e['j']}.png",
                        caption=caption_fn(e))

    div_kept_keys = {(e["i"], e["j"])
                     for e in trace["stage_5_diversity_prune"]["events"]
                     if e["kept"]}
    is_pose = trace["stage_9_skill_gate"]["is_pose_skill"]

    if 3 in stages:
        s3_dir = out_dir / "stage_3_pose_prefilter"
        for e in trace["stage_3_pose_prefilter"]["events"]:
            if e["drop"] is None:
                _save_unannotated(
                    s3_dir, e,
                    lambda e: f"[3] {e['i']}->{e['j']}  "
                              f"r={e['angle']:.1f}°  d={e['dist']:.2f}m  "
                              f"gap={e['gap']}")

    if 4 in stages:
        s4_dir = out_dir / "stage_4_quality_gate"
        for e in trace["stage_4_quality_gate"]["events"]:
            if e["drop"] is None:
                _save_unannotated(
                    s4_dir, e,
                    lambda e: f"[4] {e['i']}->{e['j']}  "
                              f"ov={e['overlap']:.3f}  q={e['quality']:.3f}")

    if 5 in stages:
        s5_dir = out_dir / "stage_5_diversity_prune"
        for e in trace["stage_5_diversity_prune"]["events"]:
            if e["kept"]:
                _save_unannotated(
                    s5_dir, e,
                    lambda e: f"[5] {e['i']}->{e['j']}  q={e['quality']:.3f}")

    if 6 in stages:
        s6_dir = out_dir / "stage_6_task_assignment"
        for e in trace["stage_6_task_assignment"]["events"]:
            if (e["i"], e["j"]) not in div_kept_keys:
                continue
            if is_pose and skill not in e["tasks"]:
                continue
            _save_unannotated(
                s6_dir, e,
                lambda e: f"[6] {e['i']}->{e['j']}  "
                          f"tags={','.join(e['tasks']) or '-'}")

    if 8 in stages:
        s78_dir = out_dir / "stage_7_8_match"
        for e in trace["stage_8_match"]:
            if (e["i"], e["j"]) not in div_kept_keys:
                continue
            _save_unannotated(
                s78_dir, e,
                lambda e: f"[7+8] {e['i']}->{e['j']}  "
                           f"vis={e['n_visible']}  occ={e['n_occluded']}")

    if 9 in stages:
        s9_dir = out_dir / "stage_9_emit"
        for e in trace["stage_9_skill_gate"]["events"]:
            if not e["evidence"]:
                continue
            pl = payloads.get((e["i"], e["j"]))
            if pl is None:
                continue
            res = _full_pair_image(img_dir, e["i"], e["j"])
            if res is None:
                continue
            pair_img, src_sh, tgt_sh, gp = res
            if skill == "cross_point_correspondence":
                ev = pl["evidence"]
                qual = list(ev.qualifying_matches)
                occ = list(ev.meta.get("occluded_candidates", []))
                pair_img = _annotate_cpc_full(
                    pair_img, src_sh, tgt_sh, gp,
                    pl["matches"], qual, occ,
                    pl["src_masks"], pl["tgt_masks"])
                cap = (f"[9 EMIT] {e['i']}->{e['j']}  "
                       f"POS={len(qual)}  NEG={len(occ)}")
            else:
                cap = (f"[9 EMIT] {e['i']}->{e['j']}  "
                       f"n_q={e['n_qualifying']}")
            _save_pair_png(pair_img, s9_dir / f"{e['i']}_{e['j']}.png",
                            caption=cap)


def _annotate_frame_with_masks(image_path: str, masks: list,
                                target_w: int = _THUMB_W,
                                target_h: int = _THUMB_H) -> np.ndarray | None:
    """Resize a single frame to (target_w x target_h) and overlay every
    mask in `masks` with a per-label color, plus a label tag at centroid."""
    try:
        im = Image.open(image_path).resize((target_w, target_h), Image.BILINEAR)
        arr = np.array(im.convert("RGB"))
    except Exception:
        return None
    if not masks:
        return arr
    out = arr.astype(np.float32)
    for m in masks:
        try:
            label = (getattr(m, "canonical", "") or m.label or "").strip() or "?"
        except Exception:
            label = "?"
        color = np.array([int(c * 255) for c in color_for(label or "")],
                          dtype=np.float32)
        # Resize mask to thumbnail size
        try:
            pil_m = Image.fromarray((m.mask.astype(np.uint8) * 255), mode="L")
            pil_m = pil_m.resize((target_w, target_h), Image.NEAREST)
            mb = np.array(pil_m) > 127
        except Exception:
            continue
        out[mb] = out[mb] * 0.55 + color * 0.45
    out_im = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    # Draw label text at each mask's centroid
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(out_im)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 10)
    except Exception:
        font = ImageFont.load_default()
    src_h, src_w = next(
        (m.mask.shape for m in masks if hasattr(m, "mask")),
        (target_h, target_w),
    )
    sx = target_w / max(1, src_w)
    sy = target_h / max(1, src_h)
    for m in masks:
        try:
            label = (getattr(m, "canonical", "") or m.label or "").strip() or "?"
            cx, cy = m.centroid
            x = int(cx * sx); y = int(cy * sy)
            bbox = draw.textbbox((x, y), label, font=font)
            pad = 2
            draw.rectangle((bbox[0] - pad, bbox[1] - pad,
                             bbox[2] + pad, bbox[3] + pad),
                            fill=(0, 0, 0))
            draw.text((x, y), label, fill="white", font=font)
        except Exception:
            continue
    return np.array(out_im)


def render_dashboard(trace: dict, out_png: Path,
                      scenes_root: Path = Path("/home/mila/l/leh/scratch/dataset/scannet_data/scans"),
                      max_pairs_per_stage: int = 18,
                      max_perception_frames: int = 24) -> None:
    img_dir = scenes_root / trace["scene_id"] / "color"

    # Build pair grids for stages 3..9. Each stage shows the SURVIVING pairs
    # at that stage (sampled uniformly if there are too many).
    def _build(events_keys, fmt_caption):
        out = []
        for e in events_keys:
            img = _pair_image(img_dir, e["i"], e["j"])
            if img is None:
                continue
            out.append((img, fmt_caption(e)))
        return out

    pose_kept = [e for e in trace["stage_3_pose_prefilter"]["events"]
                  if e["drop"] is None]
    qual_kept = [e for e in trace["stage_4_quality_gate"]["events"]
                  if e["drop"] is None]
    div_kept = [e for e in trace["stage_5_diversity_prune"]["events"]
                 if e["kept"]]

    task_map = {(e["i"], e["j"]): e
                for e in trace["stage_6_task_assignment"]["events"]}
    match_map = {(e["i"], e["j"]): e for e in trace["stage_8_match"]}
    skill_map = {(e["i"], e["j"]): e
                  for e in trace["stage_9_skill_gate"]["events"]}

    # Stage 6: pairs that retain the chosen skill among their tags
    # (or all kept pairs for content-stage skills).
    is_pose = trace["stage_9_skill_gate"]["is_pose_skill"]
    skill = trace["skill"]
    stage6_kept = []
    for e in div_kept:
        tasks = task_map[(e["i"], e["j"])]["tasks"]
        if (not is_pose) or (skill in tasks):
            stage6_kept.append({**e, "tasks": tasks})

    # Stage 7+8: kept pairs with match counts
    stage8_kept = []
    for e in stage6_kept:
        m = match_map[(e["i"], e["j"])]
        stage8_kept.append({**e, "n_visible": m["n_visible"],
                              "n_occluded": m["n_occluded"]})

    # Stage 9: only pairs that emitted
    stage9_emit = []
    for e in stage8_kept:
        s = skill_map.get((e["i"], e["j"]))
        if s and s["evidence"]:
            stage9_emit.append({**e, "n_qualifying": s["n_qualifying"]})

    # Layout: 9 rows — flow strip / sampling+qwen / perception (new) /
    # stages 3..9 pair grids / drop-reasons.
    fig = plt.figure(figsize=(18, 44))
    gs = fig.add_gridspec(9, 1,
                           height_ratios=[1.0, 2.4, 3.2, 2.4, 2.4, 2.4, 2.4, 5.5, 0.5],
                           hspace=0.40)
    fig.suptitle(
        f"{trace['scene_id']}  —  skill='{trace['skill']}'  "
        f"sampling='{trace['sampling']}'", fontsize=14, y=0.995)

    # Row 0: stage 1+2 — frame strip with green/red border
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ev = trace["stage_2_qwen"]["events"]
    n = len(ev)
    if n:
        cols = min(20, n)
        rows = int(np.ceil(n / cols))
        cw, ch = 1.0 / cols, 1.0 / rows
        for k, e in enumerate(ev):
            r, c = divmod(k, cols)
            t = _frame_thumb(e["image"])
            if t is None:
                continue
            x0 = c * cw; y0 = 1.0 - (r + 1) * ch
            ax.imshow(t, extent=(x0, x0 + cw, y0, y0 + ch),
                      transform=ax.transAxes, aspect="auto")
            ax.add_patch(plt.Rectangle(
                (x0, y0), cw, ch, fill=False,
                edgecolor=("lime" if e["usable"] else "red"),
                lw=1.6, transform=ax.transAxes))
            ax.text(x0 + cw / 2, y0 + 0.005, e["fid"],
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=6, color="white",
                    bbox=dict(boxstyle="round,pad=0.1", fc="black",
                              alpha=0.7))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(
        f"[stage 1+2] Frames after stride sampling, qwen filter "
        f"(green=kept, red=dropped). Showing all {n} sampled.",
        fontsize=11)

    # Row 1 (NEW): perception — every kept frame with its mask+label overlay.
    ax = fig.add_subplot(gs[1, 0]); ax.axis("off")
    perc_cache = trace.get("_perception_cache", {})
    detector_name = trace.get("_detector", "?")
    segmenter_name = trace.get("_segmenter", "?")
    # Pull frames that survived stage 2b (mask-count); fall back to qwen kept.
    kept_for_perc = [e["fid"] for e in
                      trace.get("stage_2b_mask_count", {}).get("events", [])
                      if e.get("kept")]
    if not kept_for_perc:
        kept_for_perc = [e["fid"] for e in trace["stage_2_qwen"]["events"]
                          if e["usable"]]
    fid_to_image = {e["fid"]: e["image"]
                    for e in trace["stage_2_qwen"]["events"]}
    sample = _sample(kept_for_perc, max_perception_frames)
    n_show = len(sample)
    if n_show:
        cols = min(8, n_show)
        rows = int(np.ceil(n_show / cols))
        cw, ch = 1.0 / cols, 1.0 / rows
        for k, fid in enumerate(sample):
            r, c = divmod(k, cols)
            masks = perc_cache.get(fid, [])
            img = _annotate_frame_with_masks(
                fid_to_image[fid], masks,
                target_w=_THUMB_W * 2, target_h=_THUMB_H * 2)
            if img is None:
                continue
            x0 = c * cw; y0 = 1.0 - (r + 1) * ch
            ax.imshow(img,
                       extent=(x0 + 0.005, x0 + cw - 0.005,
                                y0 + 0.10 * ch, y0 + ch),
                       transform=ax.transAxes, aspect="auto")
            ax.text(x0 + cw / 2, y0 + 0.05 * ch,
                     f"frame {fid}  masks={len(masks)}",
                     transform=ax.transAxes, ha="center", va="center",
                     fontsize=7, family="monospace",
                     bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                ec="black", lw=0.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(
        f"[stages 7+8 perception] {detector_name} + {segmenter_name} masks "
        f"(each color = one detected instance, label drawn at centroid). "
        f"Showing {n_show} of {len(kept_for_perc)} kept frames.",
        fontsize=11)

    # Row 2: stage 3 (pose pre-filter) — surviving pairs
    ax = fig.add_subplot(gs[2, 0])
    sample = _sample(pose_kept, max_pairs_per_stage)
    grid = _build(sample, lambda e: f"{e['i']}→{e['j']}\n"
                                      f"r={e['angle']:.0f}° d={e['dist']:.2f}m")
    _grid_panel(ax,
                 f"[stage 3] After pose pre-filter — surviving pairs "
                 f"(showing {len(sample)} of {len(pose_kept)})", grid)

    # Row 3: stage 4 (quality gate)
    ax = fig.add_subplot(gs[3, 0])
    sample = _sample(qual_kept, max_pairs_per_stage)
    grid = _build(sample, lambda e: f"{e['i']}→{e['j']}\n"
                                      f"ov={e['overlap']:.2f} q={e['quality']:.2f}")
    _grid_panel(ax,
                 f"[stage 4] After quality gate — surviving pairs "
                 f"(showing {len(sample)} of {len(qual_kept)})", grid)

    # Row 4: stage 5 (diversity prune) — pair thumbnails with all detected
    # masks overlaid on both src and tgt halves.
    ax = fig.add_subplot(gs[4, 0])
    sample = _sample(div_kept, max_pairs_per_stage)
    perc_cache = trace.get("_perception_cache", {})
    fid_to_image = {e["fid"]: e["image"]
                    for e in trace["stage_2_qwen"]["events"]}
    grid = []
    for e in sample:
        src_img = _annotate_frame_with_masks(
            fid_to_image.get(e["i"], ""), perc_cache.get(e["i"], []),
            target_w=_THUMB_W, target_h=_THUMB_H)
        tgt_img = _annotate_frame_with_masks(
            fid_to_image.get(e["j"], ""), perc_cache.get(e["j"], []),
            target_w=_THUMB_W, target_h=_THUMB_H)
        if src_img is None or tgt_img is None:
            continue
        gap = np.full((_THUMB_H, _GAP, 3), 255, dtype=np.uint8)
        pair_img = np.concatenate([src_img, gap, tgt_img], axis=1)
        n_src = len(perc_cache.get(e["i"], []))
        n_tgt = len(perc_cache.get(e["j"], []))
        cap = (f"{e['i']}→{e['j']}\nq={e['quality']:.2f}  "
               f"masks {n_src}|{n_tgt}")
        grid.append((pair_img, cap))
    _grid_panel(ax,
                 f"[stage 5] After diversity prune — surviving pairs "
                 f"with detected masks (per-mask color = unique label) "
                 f"(showing {len(sample)} of {len(div_kept)})", grid)

    # Row 5: stage 6 (task assignment) — pairs that carry the chosen skill tag
    ax = fig.add_subplot(gs[5, 0])
    sample = _sample(stage6_kept, max_pairs_per_stage)
    grid = _build(sample, lambda e: f"{e['i']}→{e['j']}\n"
                                      f"tags={','.join(e['tasks']) or '—'}")
    _grid_panel(ax,
                 f"[stage 6] After task assignment — pairs eligible for "
                 f"'{skill}' "
                 f"(showing {len(sample)} of {len(stage6_kept)})", grid)

    # Row 6: stage 7+8 (perception + match)
    ax = fig.add_subplot(gs[6, 0])
    sample = _sample(stage8_kept, max_pairs_per_stage)
    grid = _build(sample, lambda e: f"{e['i']}→{e['j']}\n"
                                      f"vis={e['n_visible']} occ={e['n_occluded']}")
    _grid_panel(ax,
                 f"[stage 7+8] After perception + matching — pairs "
                 f"with at least one match "
                 f"(showing {len(sample)} of {len(stage8_kept)})", grid)

    # Row 7: stage 9 — emitted pairs with skill-specific overlays
    ax = fig.add_subplot(gs[7, 0])
    payloads = trace.get("_emit_payloads", {})
    sample = _sample(stage9_emit, max_pairs_per_stage)
    grid = []
    for e in sample:
        img = _pair_image(img_dir, e["i"], e["j"])
        if img is None:
            continue
        pl = payloads.get((e["i"], e["j"]))
        if pl is not None and skill == "cross_point_correspondence":
            ev = pl["evidence"]
            qual = list(ev.qualifying_matches)
            occ = list(ev.meta.get("occluded_candidates", []))
            img = _annotate_cpc(img, pl["matches"], qual, occ,
                                 pl["src_size"], pl["tgt_size"],
                                 pl["src_masks"], pl["tgt_masks"])
            cap = (f"{e['i']}→{e['j']}\n"
                   f"POS={len(qual)}  NEG={len(occ)}")
        else:
            cap = f"{e['i']}→{e['j']}\nn_q={e['n_qualifying']}"
        grid.append((img, cap))
    _grid_panel(ax,
                 f"[stage 9] Pairs surviving '{skill}' content gate "
                 f"— green-tinted GT mask + dot/circle = visible POS • "
                 f"red-tinted GT mask + dot/✗ = occluded NEG "
                 f"(showing {len(sample)} of {len(stage9_emit)})",
                 grid, cols=3)

    # Row 8: drop-reason breakdown (compact, just for the chosen skill's
    # final stage — answers WHY pairs failed). Kept as a small text strip.
    ax = fig.add_subplot(gs[8, 0]); ax.axis("off")
    drop_counts: dict[str, int] = {}
    for e in trace["stage_9_skill_gate"]["events"]:
        if not e["evidence"]:
            r = e.get("drop_reason") or "(unspecified)"
            drop_counts[r] = drop_counts.get(r, 0) + 1
    if drop_counts:
        s = "  ".join(f"{r}: {n}" for r, n in
                       sorted(drop_counts.items(), key=lambda x: -x[1]))
    else:
        s = "(no drops at stage 9 — all eligible pairs emitted)"
    ax.set_title(f"[stage 9] drop reasons (for the '{skill}' content gate)",
                  fontsize=11)
    ax.text(0.0, 0.7, s, transform=ax.transAxes, fontsize=10,
             family="monospace")

    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---- driver ------------------------------------------------------------

def _to_jsonable(o):
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, action="append",
                   help="repeat for multiple scenes")
    p.add_argument("--skill", default="cross_point_correspondence",
                   help="any of the 9 skills (content_skills + pose_skills)")
    p.add_argument("--sampling", default="stride",
                   choices=["stride", "adaptive"])
    p.add_argument("--frame-stride", type=int, default=50,
                   help="max stride for stride sampling (default 50)")
    p.add_argument("--min-keyframes", type=int, default=30,
                   help="floor on the keyframe count (default 30); short "
                        "scenes get a smaller effective stride to retain "
                        ">= N keyframes; 0 to disable the floor")
    p.add_argument("--limit-frames", type=int, default=0,
                   help="cap kept keyframes (default 0 = no cap)")
    p.add_argument("--save-stages", default="5",
                   help="comma-separated stages (3,4,5,6,8,9) to write "
                        "per-pair PNG files for; default '5' = only "
                        "post-diversity-prune pairs. Use 'all' for every stage.")
    p.add_argument("--min-frame-masks", type=int, default=3,
                   help="drop frames whose GT detector finds < N instance "
                        "masks after blocklist removal (default 3 — filters "
                        "empty-walls / blank-closet pairs)")
    p.add_argument("--detector", default="scannet-gt",
                   choices=["noop", "gdino", "gdino+scannet200",
                            "labeled-gdino", "gemini+gdino",
                            "scannet-gt", "scannet-gt-label+gdino"])
    p.add_argument("--segmenter", default="gt-mask",
                   choices=["noop", "sam2.1", "sam3", "gt-mask"])
    p.add_argument("--no-qwen", action="store_true",
                   help="skip qwen filter (use cache lookups only)")
    p.add_argument("--scenes-root", type=Path,
                   default=Path("/home/mila/l/leh/scratch/dataset/scannet_data/scans"))
    p.add_argument("--out", type=Path,
                   default=Path("outputs/pipeline_debug"))
    args = p.parse_args()

    if args.save_stages.strip().lower() == "all":
        save_stages = (3, 4, 5, 6, 8, 9)
    else:
        save_stages = tuple(int(s) for s in args.save_stages.split(",") if s.strip())

    args.out.mkdir(parents=True, exist_ok=True)
    for sc in args.scene:
        out_dir = args.out / sc / args.skill
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== {sc} / {args.skill} ===")
        trace = trace_pipeline(
            sc, args.skill,
            sampling=args.sampling, frame_stride=args.frame_stride,
            min_keyframes=args.min_keyframes,
            limit_frames=args.limit_frames,
            min_frame_masks=args.min_frame_masks,
            detector_name=args.detector,
            segmenter_name=args.segmenter,
            use_qwen=not args.no_qwen, scenes_root=args.scenes_root,
        )
        # JSONL trace (one event-list per stage, one line per stage)
        with open(out_dir / "trace.jsonl", "w") as f:
            for stage_key in sorted(trace.keys()):
                if stage_key.startswith("_"):
                    continue   # non-JSON helper data (e.g. _emit_payloads)
                f.write(json.dumps({"stage": stage_key,
                                     "data": _to_jsonable(trace[stage_key])})
                        + "\n")
        # Dashboard PNG (quick look) + per-pair PNGs (full inspection)
        out_png = out_dir / "dashboard.png"
        render_dashboard(trace, out_png, scenes_root=args.scenes_root)
        save_per_pair_files(trace, out_dir, scenes_root=args.scenes_root,
                              stages=save_stages)
        print(f"  -> {out_png}")
        for name, n in _flow_summary(trace):
            print(f"     {name:<40}  {n}")


if __name__ == "__main__":
    main()
