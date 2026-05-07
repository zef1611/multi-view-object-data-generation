"""Per-skill, per-pair debug visualizer for the Phase-1 generation pipeline.

Purpose: for each pair that a given skill qualified on, render a single
multi-panel figure that shows EVERY stage of the pipeline — so you can
see not only the final emit but also what the detector, segmenter, and
matcher handed the gate, and which gate conditions passed with what
margins.

Reads from:
    <out-root>/stage_1/<skill>/pairs.jsonl          (one per qualifying pair)
    cache/perception/<adapter>/<scene>/<model_tag>/<frame>.pkl   (per-frame masks)
    configs/tasks.json               (gate thresholds)

Writes to:
    <out-root>/stage_1/<skill>/_debug/<scene>_<fsrc>_<ftgt>.png

Panels (top-to-bottom):
    (1) POSE STAGE        — raw src/tgt frames with pose metrics
                            (rotation, translation, frame gap,
                            overlap, occluded-frac).
    (2) PERCEPTION STAGE  — src/tgt with every detector+SAM mask
                            overlaid (bbox + label + score + mask).
    (3) MATCH STAGE       — src/tgt with ONLY the qualifying matches
                            highlighted, color-linked src<->tgt, and
                            per-match metadata (IoU, depth,
                            visibility, label).
    (4) CONTENT GATE      — table panel: one row per gate condition,
                            showing threshold, measured value, and
                            pass/fail. Shows margins even for
                            surviving pairs.
    (5) FINAL EMIT        — skill-specific overlay of what actually
                            gets emitted. cross_point_correspondence:
                            src point + tgt point (green=visible,
                            red=occluded) with answer line. Other
                            skills: falls back to match panel for now.

Usage:
    python dryrun_inspect.py \
        --out-root outputs/dryrun_3scenes \
        --skill cross_point_correspondence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from viz import (
    add_cache_args,
    color_for,
    draw_bbox as _draw_bbox,
    load_frame_masks,
)


_CFG_PATH_DEFAULT = Path("configs/tasks.json")


# ---- panel 1: pose stage ------------------------------------------------

def _panel_pose(ax_s, ax_t, img_src, img_tgt, manifest: dict,
                frame_gap: int):
    ax_s.imshow(img_src); ax_t.imshow(img_tgt)
    for ax in (ax_s, ax_t):
        ax.set_xticks([]); ax.set_yticks([])
    m = manifest
    stitle_s = (
        f"[1] POSE STAGE  src frame {m['frame_src']}\n"
        f"Δrot={m['pair_angle_deg']:.1f}°  Δtrans={m['pair_distance_m']:.2f}m  "
        f"frame_gap={frame_gap}"
    )
    stitle_t = (
        f"[1] POSE STAGE  tgt frame {m['frame_tgt']}\n"
        f"overlap={m['pair_overlap']:.3f}  "
        f"occluded_frac={m['pair_occluded_frac']:.3f}"
    )
    ax_s.set_title(stitle_s, fontsize=9)
    ax_t.set_title(stitle_t, fontsize=9)


# ---- panel 2: perception stage ------------------------------------------

def _panel_perception(ax_s, ax_t, img_src, img_tgt, masks_s, masks_t):
    ax_s.imshow(img_src); ax_t.imshow(img_tgt)
    for ax, masks, img in ((ax_s, masks_s, img_src),
                            (ax_t, masks_t, img_tgt)):
        ax.set_xticks([]); ax.set_yticks([])
        if not masks:
            continue
        ov = np.zeros(img.shape[:2] + (4,), dtype=np.float32)
        for i, mk in enumerate(masks):
            color = color_for(f"label|{(mk.label or '').lower()}")
            ov[mk.mask] = (*color, 0.30)
        ax.imshow(ov)
        for i, mk in enumerate(masks):
            color = color_for(f"label|{(mk.label or '').lower()}")
            _draw_bbox(ax, mk.bbox, color=color, lw=1.3)
            x0, y0, _, _ = mk.bbox
            ax.annotate(
                f"[{i}] {mk.label} {getattr(mk, 'score', 0.0):.2f}",
                (x0, y0), xytext=(2, -2), textcoords="offset points",
                fontsize=6, color="white",
                bbox=dict(boxstyle="round,pad=0.15",
                          fc=tuple(c * 0.6 for c in color), alpha=0.8),
            )
    n_s = len(masks_s) if masks_s else 0
    n_t = len(masks_t) if masks_t else 0
    ax_s.set_title(f"[2] PERCEPTION  detections={n_s}", fontsize=9)
    ax_t.set_title(f"[2] PERCEPTION  detections={n_t}", fontsize=9)


# ---- panel 3: match stage -----------------------------------------------

def _panel_match(ax_s, ax_t, img_src, img_tgt, manifest: dict,
                 masks_s, masks_t, qualifying_idxs: list[int],
                 occluded_idxs: list[int]):
    ax_s.imshow(img_src); ax_t.imshow(img_tgt)
    for ax in (ax_s, ax_t):
        ax.set_xticks([]); ax.set_yticks([])

    # Concatenate visible + occluded; render visible first so occluded
    # markers (red ✗) draw on top of the same color overlay.
    all_idxs = list(qualifying_idxs) + list(occluded_idxs)
    caption_lines = []
    for rank, mi in enumerate(all_idxs):
        obj = manifest["objects"][mi]
        key = (f"{manifest['scene_id']}|{obj['X_world'][0]:.2f}|"
               f"{obj['X_world'][1]:.2f}|{obj['X_world'][2]:.2f}")
        color = color_for(key)
        # src side
        sid = obj["src_mask_id"]
        if masks_s and 0 <= sid < len(masks_s):
            ov = np.zeros(img_src.shape[:2] + (4,), dtype=np.float32)
            ov[masks_s[sid].mask] = (*color, 0.45)
            ax_s.imshow(ov)
        _draw_bbox(ax_s, obj["src_bbox"], color=color, lw=2.4)
        u, v = obj["point_src"]
        ax_s.plot(u, v, "o", ms=10, mec="black", mfc=color, mew=1.5)
        ax_s.annotate(f"[{rank}]", (u, v), xytext=(8, -8),
                      textcoords="offset points", fontsize=8,
                      color="white",
                      bbox=dict(boxstyle="round,pad=0.2",
                                fc="black", alpha=0.8))
        # tgt side
        tid = obj["tgt_mask_id"]
        if masks_t and 0 <= tid < len(masks_t):
            ov = np.zeros(img_tgt.shape[:2] + (4,), dtype=np.float32)
            ov[masks_t[tid].mask] = (*color, 0.45)
            ax_t.imshow(ov)
        _draw_bbox(ax_t, obj["tgt_bbox"], color=color, lw=2.4)
        u2, v2 = obj["point_tgt"]
        if obj["visible"]:
            ax_t.plot(u2, v2, "o", ms=12, mec="lime", mfc="none", mew=2.5)
        else:
            ax_t.plot(u2, v2, "x", ms=14, mec="red", mew=3.0)
        ax_t.annotate(f"[{rank}]", (u2, v2), xytext=(8, -8),
                      textcoords="offset points", fontsize=8,
                      color="white",
                      bbox=dict(boxstyle="round,pad=0.2",
                                fc=("darkgreen" if obj["visible"]
                                    else "darkred"),
                                alpha=0.85))
        ddepth = obj["depth_pred_tgt"] - obj["depth_obs_tgt"]
        vis = "✓" if obj["visible"] else "✗OCC"
        caption_lines.append(
            f"[{rank}] '{obj['src_label']}' → '{obj['tgt_label']}'  "
            f"iou={obj['iou_src_to_tgt']:.2f}  "
            f"d_src={obj['depth_src']:.2f}m  Δd={ddepth:+.2f}m  {vis}"
        )
    ax_s.set_title(
        f"[3] MATCH  qualifying={len(qualifying_idxs)}  "
        f"occluded={len(occluded_idxs)}", fontsize=9)
    ax_t.set_title("[3] MATCH", fontsize=9)
    ax_s.annotate(
        "\n".join(caption_lines) or "(none)",
        xy=(0, 0), xycoords="axes fraction",
        xytext=(0, -14), textcoords="offset points",
        ha="left", va="top", fontsize=7, family="monospace",
        annotation_clip=False,
    )


# ---- panel 4: content gate ----------------------------------------------

def _gate_rows_cross_point_correspondence(manifest: dict, cfg_all: dict,
                                           n_vis_labeled: int):
    cfg = cfg_all.get("content_skills", {}).get(
        "cross_point_correspondence", {})
    ev = manifest["evidence"]
    ov_lo, ov_hi = cfg["overlap"]
    rows = [
        ("overlap", f"[{ov_lo:.2f}, {ov_hi:.2f}]",
         f"{manifest['pair_overlap']:.3f}",
         ov_lo <= manifest["pair_overlap"] <= ov_hi),
        ("rotation ≤ max_rot_deg",
         f"≤ {cfg['max_rot_deg']:.1f}°",
         f"{manifest['pair_angle_deg']:.2f}°",
         manifest["pair_angle_deg"] <= cfg["max_rot_deg"]),
        (f"rot ≥ {cfg['min_rot_deg']}° {cfg.get('viewpoint_shift_mode', 'or').upper()} trans ≥ {cfg['min_trans_m']}m",
         f"{cfg.get('viewpoint_shift_mode', 'or').upper()} of (rot, trans)",
         (f"rot={manifest['pair_angle_deg']:.2f}°  "
          f"trans={manifest['pair_distance_m']:.3f}m"),
         ((manifest["pair_angle_deg"] >= cfg["min_rot_deg"]
           and manifest["pair_distance_m"] >= cfg["min_trans_m"])
          if str(cfg.get("viewpoint_shift_mode", "or")).lower() == "and"
          else (manifest["pair_angle_deg"] >= cfg["min_rot_deg"]
                or manifest["pair_distance_m"] >= cfg["min_trans_m"]))),
        ("min_visible_matches",
         f"≥ {cfg['min_visible_matches']}",
         f"{n_vis_labeled}",
         n_vis_labeled >= cfg["min_visible_matches"]),
    ]
    # Per-match score/coverage summary
    min_score = cfg.get("min_label_score", 0.0)
    min_cov = cfg.get("mask_depth_coverage_min", 0.0)
    rows.append(
        ("min_label_score (per qualifying match)",
         f"≥ {min_score}",
         ", ".join(f"{m['score']:.2f}" for m in ev.get("matches", []))
         or "-", True),
    )
    rows.append(
        ("mask_depth_coverage_min",
         f"≥ {min_cov}  (checked in gate, not reported)",
         "(qualifying count implies all passed)", True),
    )
    return rows


def _gate_rows_cross_object_correspondence(manifest: dict, cfg_all: dict,
                                             _n_vis: int):
    cfg = cfg_all.get("content_skills", {}).get(
        "cross_object_correspondence", {})
    ev = manifest["evidence"]
    ov_lo, ov_hi = cfg["overlap"]
    n_shared = ev.get("n_shared_objects",
                      len(ev.get("qualifying_matches", [])))
    shared = ev.get("shared_objects", [])
    rows = [
        ("overlap", f"[{ov_lo:.2f}, {ov_hi:.2f}]",
         f"{manifest['pair_overlap']:.3f}",
         ov_lo <= manifest["pair_overlap"] <= ov_hi),
        ("rotation ≤ max_rot_deg",
         f"≤ {cfg['max_rot_deg']:.1f}°",
         f"{manifest['pair_angle_deg']:.2f}°",
         manifest["pair_angle_deg"] <= cfg["max_rot_deg"]),
        (f"rot ≥ {cfg['min_rot_deg']}° {cfg.get('viewpoint_shift_mode', 'or').upper()} trans ≥ {cfg['min_trans_m']}m",
         f"{cfg.get('viewpoint_shift_mode', 'or').upper()} of (rot, trans)",
         (f"rot={manifest['pair_angle_deg']:.2f}°  "
          f"trans={manifest['pair_distance_m']:.3f}m"),
         ((manifest["pair_angle_deg"] >= cfg["min_rot_deg"]
           and manifest["pair_distance_m"] >= cfg["min_trans_m"])
          if str(cfg.get("viewpoint_shift_mode", "or")).lower() == "and"
          else (manifest["pair_angle_deg"] >= cfg["min_rot_deg"]
                or manifest["pair_distance_m"] >= cfg["min_trans_m"]))),
        ("min_visible_matches (with labeled tgt mask)",
         f"≥ {cfg['min_visible_matches']}",
         f"{n_shared}",
         n_shared >= cfg["min_visible_matches"]),
        ("min_label_score (per shared object)",
         f"≥ {cfg.get('min_label_score', 0.0)}",
         ", ".join(f"{s['score']:.2f}" for s in shared) or "-", True),
        ("min_tgt_mask_area_frac (per shared object)",
         f"≥ {cfg.get('min_tgt_mask_area_frac', 0.0)}",
         ", ".join(f"{s['tgt_mask_area_frac']:.4f}"
                   for s in shared) or "-", True),
    ]
    return rows


def _gate_rows_anchor(manifest: dict, cfg_all: dict, _n: int):
    cfg = cfg_all.get("content_skills", {}).get("anchor", {})
    ev = manifest["evidence"]
    ov_lo, ov_hi = cfg.get("overlap", [0.0, 1.0])
    lo, hi = cfg.get("scale_ratio_excl", [0.5, 2.0])
    min_n = int(cfg.get("min_visible_matches", 1))
    min_rot = float(cfg.get("min_rot_deg", 0.0))
    min_trans = float(cfg.get("min_trans_m", 0.0))
    shared = ev.get("shared_objects", [])
    n_shared = ev.get("n_shared", len(ev.get("qualifying_matches", [])))
    n_nontriv = sum(1 for s in shared if s.get("non_trivial"))
    return [
        ("overlap", f"[{ov_lo:.2f}, {ov_hi:.2f}]",
         f"{manifest['pair_overlap']:.3f}",
         ov_lo <= manifest["pair_overlap"] <= ov_hi),
        (f"rot ≥ {min_rot}° OR trans ≥ {min_trans}m",
         "one must pass",
         f"rot={manifest['pair_angle_deg']:.2f}°  "
         f"trans={manifest['pair_distance_m']:.3f}m",
         (manifest["pair_angle_deg"] >= min_rot
          or manifest["pair_distance_m"] >= min_trans)),
        ("min_visible_matches", f"≥ {min_n}", f"{n_shared}", n_shared >= min_n),
        (f"scale_ratio outside [{lo}, {hi}]",
         "≥ 1 non-trivial",
         f"{n_nontriv}/{n_shared} non-trivial",
         n_nontriv >= 1),
        ("per-match scale ratios", "—",
         ", ".join(f"{s['scale_ratio']:.2f}"
                   f"{'*' if s.get('non_trivial') else ''}"
                   for s in shared) or "-",
         True),
    ]


def _gate_rows_counting(manifest: dict, cfg_all: dict, _n: int):
    cfg = cfg_all.get("content_skills", {}).get("counting", {})
    ev = manifest["evidence"]
    ov_lo, ov_hi = cfg.get("overlap", [0.0, 1.0])
    lo = int(cfg.get("min_cat_count", 3))
    hi = int(cfg.get("max_cat_count", 15))
    req_s = bool(cfg.get("require_shared", True))
    req_p = bool(cfg.get("require_private", True))
    cat = ev.get("category", "?")
    total = int(ev.get("unique_total", 0))
    n_sh = len(ev.get("shared_match_idx", []))
    n_ps = len(ev.get("private_src_idx", []))
    n_pt = len(ev.get("private_tgt_idx", []))
    return [
        ("overlap", f"[{ov_lo:.2f}, {ov_hi:.2f}]",
         f"{manifest['pair_overlap']:.3f}",
         ov_lo <= manifest["pair_overlap"] <= ov_hi),
        ("winning category (canonical)", "—", f"'{cat}'", True),
        (f"unique_total in [{lo}, {hi}]",
         f"[{lo}, {hi}]", f"{total}", lo <= total <= hi),
        ("shared instances", "≥1" if req_s else "—",
         f"{n_sh}", (n_sh >= 1) if req_s else True),
        ("private instances (src or tgt)",
         "≥1" if req_p else "—",
         f"src={n_ps}  tgt={n_pt}",
         ((n_ps + n_pt) >= 1) if req_p else True),
    ]


def _gate_rows_relative_distance(manifest: dict, cfg_all: dict, _n: int):
    cfg = cfg_all.get("content_skills", {}).get("relative_distance", {})
    ev = manifest["evidence"]
    ov_lo, ov_hi = cfg.get("overlap", [0.0, 1.0])
    min_obj = int(cfg.get("min_objects", 3))
    min_marg = float(cfg.get("min_margin_m", 0.5))
    min_cov = float(cfg.get("mask_depth_coverage_min", 0.6))
    cands = ev.get("candidates", [])
    n_cand = len(cands) + 1  # +1 for the reference itself
    margin = float(ev.get("margin_m", 0.0))
    far_d = cands[0]["distance_m"] if cands else 0.0
    run_d = cands[1]["distance_m"] if len(cands) >= 2 else 0.0
    return [
        ("overlap", f"[{ov_lo:.2f}, {ov_hi:.2f}]",
         f"{manifest['pair_overlap']:.3f}",
         ov_lo <= manifest["pair_overlap"] <= ov_hi),
        (f"reference + candidates ≥ {min_obj}",
         f"≥ {min_obj}", f"{n_cand}", n_cand >= min_obj),
        ("mask_depth_coverage_min (per src mask)",
         f"≥ {min_cov}", "(applied in gate)", True),
        ("reference label", "—",
         f"'{ev.get('reference_label', '?')}'", True),
        (f"margin (farthest − runner-up) ≥ {min_marg}m",
         f"≥ {min_marg}m",
         f"far={far_d:.2f}m  next={run_d:.2f}m  Δ={margin:.2f}m",
         margin >= min_marg),
        ("ordered distances (m)", "—",
         ", ".join(f"{c['label']}:{c['distance_m']:.2f}"
                   for c in cands) or "-",
         True),
    ]


def _gate_rows_cross_spatial_transformation(manifest: dict, cfg_all: dict, _n: int):
    cfg = cfg_all.get("tasks", {}).get("cross_spatial_transformation", {})
    sel = cfg_all.get("selection", {})
    ev = manifest["evidence"]
    angle_min = float(cfg.get("angle_deg_min", 30.0))
    fg_bonus = int(cfg.get("min_frame_gap_bonus_by_source", {}).get("scannet", 0))
    fg_floor = int(cfg_all.get("min_frame_gap_by_source", {}).get("scannet", 0))
    fg = abs(int(manifest["frame_tgt"]) - int(manifest["frame_src"]))
    objs = ev.get("transformed_objects", [])
    return [
        (f"angle ≥ {angle_min}°", f"≥ {angle_min}°",
         f"{manifest['pair_angle_deg']:.2f}°",
         manifest["pair_angle_deg"] >= angle_min),
        ("frame_gap ≥ source floor + bonus",
         f"≥ {fg_floor + fg_bonus}",
         f"{fg}",
         fg >= (fg_floor + fg_bonus)),
        (f"distance ≤ {sel.get('max_distance_m', 5.0)}m",
         f"≤ {sel.get('max_distance_m', 5.0)}m",
         f"{manifest['pair_distance_m']:.3f}m",
         manifest["pair_distance_m"] <= float(sel.get("max_distance_m", 5.0))),
        ("transformed objects (scale ratio outside [0.6,1.67])",
         "≥ 1", f"{len(objs)}", len(objs) >= 1),
        ("per-obj scale ratios", "—",
         ", ".join(f"{o['label']}:{o['scale_ratio']:.2f}"
                   for o in objs) or "-",
         True),
    ]


def _gate_rows_cross_depth_variation(manifest: dict, cfg_all: dict, _n: int):
    cfg = cfg_all.get("tasks", {}).get("cross_depth_variation", {})
    ev = manifest["evidence"]
    ratio_min = float(cfg.get("median_depth_ratio_min", 1.3))
    objs = ev.get("varying_objects", [])
    median = float(ev.get("pair_median_delta_m", 0.0))
    return [
        (f"median |Δdepth| (per-match)", "informational",
         f"{median:.3f}m", True),
        (f"median_depth_ratio_min ≥ {ratio_min}",
         f"≥ {ratio_min}",
         "(applied at pair gate, not re-checked here)", True),
        ("matches with |Δdepth| ≥ 0.5m",
         "≥ 1", f"{len(objs)}", len(objs) >= 1),
        ("per-match Δdepth", "—",
         ", ".join(f"{o['label']}:{o['delta_m']:+.2f}m"
                   for o in objs) or "-",
         True),
    ]


def _gate_rows_cross_occlusion_visibility(manifest: dict, cfg_all: dict, _n: int):
    cfg = cfg_all.get("tasks", {}).get("cross_occlusion_visibility", {})
    ev = manifest["evidence"]
    ov_min = float(cfg.get("overlap_min", 0.40))
    occ_min = float(cfg.get("occluded_fraction_min", 0.15))
    fg_bonus = int(cfg.get("min_frame_gap_bonus_by_source", {}).get("scannet", 0))
    fg_floor = int(cfg_all.get("min_frame_gap_by_source", {}).get("scannet", 0))
    fg = abs(int(manifest["frame_tgt"]) - int(manifest["frame_src"]))
    n_v = int(ev.get("n_visible", 0))
    n_o = int(ev.get("n_occluded", 0))
    return [
        (f"overlap ≥ {ov_min}",
         f"≥ {ov_min}", f"{manifest['pair_overlap']:.3f}",
         manifest["pair_overlap"] >= ov_min),
        (f"occluded_fraction ≥ {occ_min}",
         f"≥ {occ_min}", f"{manifest['pair_occluded_frac']:.3f}",
         manifest["pair_occluded_frac"] >= occ_min),
        ("frame_gap ≥ source floor + occlusion bonus",
         f"≥ {fg_floor + fg_bonus}",
         f"{fg}",
         fg >= (fg_floor + fg_bonus)),
        ("visible matches", "≥ 1", f"{n_v}", n_v >= 1),
        ("occluded matches", "≥ 1", f"{n_o}", n_o >= 1),
    ]


def _gate_rows_relative_direction(manifest: dict, cfg_all: dict, _n: int):
    cfg = cfg_all.get("content_skills", {}).get("relative_direction", {})
    ev = manifest["evidence"]
    ov_lo, ov_hi = cfg.get("overlap", [0.0, 1.0])
    min_t = float(cfg.get("min_trans_m", 0.0))
    min_r = float(cfg.get("min_rot_deg", 0.0))
    max_e = float(cfg.get("max_elev_deg", 60.0))
    min_sep = float(cfg.get("min_azimuth_sep_deg", 0.0))
    hyst = float(cfg.get("bucket_hysteresis_deg", 10.0))
    targets = ev.get("targets", [])
    azims = sorted(t["azimuth_deg"] for t in targets)
    if len(azims) >= 2:
        max_gap = max(azims[i + 1] - azims[i]
                      for i in range(len(azims) - 1))
    else:
        max_gap = 0.0
    return [
        ("overlap", f"[{ov_lo:.2f}, {ov_hi:.2f}]",
         f"{manifest['pair_overlap']:.3f}",
         ov_lo <= manifest["pair_overlap"] <= ov_hi),
        (f"translation ≥ {min_t}m",
         f"≥ {min_t}m", f"{manifest['pair_distance_m']:.3f}m",
         manifest["pair_distance_m"] >= min_t),
        (f"rotation ≥ {min_r}°",
         f"≥ {min_r}°", f"{manifest['pair_angle_deg']:.2f}°",
         manifest["pair_angle_deg"] >= min_r),
        (f"|elev| ≤ {max_e}°  AND  bucket-edge ≥ {hyst}° (hysteresis)",
         "applied per target",
         f"{len(targets)} survivor(s)", len(targets) >= 1),
        (f"max azimuth gap ≥ {min_sep}° (discriminative spread)",
         f"≥ {min_sep}°", f"{max_gap:.1f}°",
         (min_sep <= 0) or (max_gap >= min_sep)),
        ("per-target buckets", "—",
         ", ".join(f"{t['label']}→{t['bucket']}({t['azimuth_deg']:+.0f}°)"
                   for t in targets) or "-",
         True),
    ]


GATE_DISPATCH = {
    "cross_point_correspondence": _gate_rows_cross_point_correspondence,
    "cross_object_correspondence": _gate_rows_cross_object_correspondence,
    "anchor": _gate_rows_anchor,
    "counting": _gate_rows_counting,
    "relative_distance": _gate_rows_relative_distance,
    "cross_spatial_transformation": _gate_rows_cross_spatial_transformation,
    "cross_depth_variation": _gate_rows_cross_depth_variation,
    "cross_occlusion_visibility": _gate_rows_cross_occlusion_visibility,
    "relative_direction": _gate_rows_relative_direction,
}


def _panel_gate(ax, skill: str, manifest: dict, cfg: dict):
    ax.axis("off")
    ev = manifest["evidence"]
    n_q = len(ev.get("qualifying_matches", []))
    n_vis_labeled = ev.get("n_visible_labeled", n_q)
    fn = GATE_DISPATCH.get(skill)
    if fn is None:
        ax.set_title(f"[4] CONTENT GATE  ({skill}: no table yet)",
                     fontsize=9)
        ax.text(0.0, 0.9,
                f"qualifying_matches = {n_q}\n"
                f"(per-skill table not wired yet; extend "
                f"GATE_DISPATCH in dryrun_inspect.py to add one)",
                fontsize=9, family="monospace",
                transform=ax.transAxes, va="top")
        return
    rows = fn(manifest, cfg, n_vis_labeled)
    headers = ("gate", "threshold", "measured", "pass")
    col_w = [0.30, 0.22, 0.38, 0.10]
    y = 0.95
    ax.set_title(f"[4] CONTENT GATE — {skill}", fontsize=10)
    # header
    x = 0.0
    for h, w in zip(headers, col_w):
        ax.text(x, y, h, fontsize=9, fontweight="bold",
                family="monospace", transform=ax.transAxes, va="top")
        x += w
    y -= 0.08
    for name, thr, meas, ok in rows:
        x = 0.0
        color = "#0a7a0a" if ok else "#b00020"
        for val, w in zip((name, str(thr), str(meas),
                           "PASS" if ok else "FAIL"),
                          col_w):
            ax.text(x, y, val, fontsize=8, family="monospace",
                    color=color if w == col_w[-1] else "black",
                    transform=ax.transAxes, va="top")
            x += w
        y -= 0.07
        if y < 0.05:
            break


# ---- panel 5: final emit ------------------------------------------------

def _draw_emit_qa(ax_s, ax_t, img_src, img_tgt, obj: dict, label: str):
    ax_s.imshow(img_src); ax_t.imshow(img_tgt)
    for ax in (ax_s, ax_t):
        ax.set_xticks([]); ax.set_yticks([])
    u, v = obj["point_src"]
    ax_s.plot(u, v, "+", ms=30, mec="cyan", mew=3.0)
    ax_s.plot(u, v, "o", ms=10, mec="black", mfc="cyan", mew=1.5)
    ax_s.annotate(
        f"query: {obj['src_label']}", (u, v),
        xytext=(12, -10), textcoords="offset points", fontsize=9,
        color="white",
        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.85),
    )
    u2, v2 = obj["point_tgt"]
    visible = obj["visible"]
    if visible:
        ax_t.plot(u2, v2, "o", ms=16, mec="lime", mfc="none", mew=3.0)
        tcol = "darkgreen"; tag = "answer (visible)"
    else:
        ax_t.plot(u2, v2, "x", ms=18, mec="red", mew=3.5)
        tcol = "darkred"; tag = "answer (occluded)"
    ax_t.annotate(
        f"{tag}\nΔd={obj['depth_pred_tgt']-obj['depth_obs_tgt']:+.2f}m",
        (u2, v2), xytext=(12, -10), textcoords="offset points",
        fontsize=9, color="white",
        bbox=dict(boxstyle="round,pad=0.2", fc=tcol, alpha=0.9),
    )
    ax_s.set_title(f"[5] EMIT {label}: src", fontsize=9)
    ax_t.set_title(f"[5] EMIT {label}: tgt", fontsize=9)


def _panel_emit_cross_point_correspondence(ax_pos_s, ax_pos_t,
                                            ax_neg_s, ax_neg_t,
                                            img_src, img_tgt, manifest: dict):
    qual = manifest["evidence"].get("qualifying_matches", [])
    occ = manifest["evidence"].get("occluded_candidates", [])
    if qual:
        _draw_emit_qa(ax_pos_s, ax_pos_t, img_src, img_tgt,
                      manifest["objects"][qual[0]], label="POSITIVE")
    else:
        for ax in (ax_pos_s, ax_pos_t):
            ax.imshow(img_src if ax is ax_pos_s else img_tgt)
            ax.set_xticks([]); ax.set_yticks([])
        ax_pos_s.set_title("[5] EMIT POSITIVE: (none)", fontsize=9)
        ax_pos_t.set_title("[5] EMIT POSITIVE: (none)", fontsize=9)
    if occ:
        _draw_emit_qa(ax_neg_s, ax_neg_t, img_src, img_tgt,
                      manifest["objects"][occ[0]], label="NEGATIVE")
    else:
        for ax in (ax_neg_s, ax_neg_t):
            ax.imshow(img_src if ax is ax_neg_s else img_tgt)
            ax.set_xticks([]); ax.set_yticks([])
        ax_neg_s.set_title("[5] EMIT NEGATIVE: (none)", fontsize=9)
        ax_neg_t.set_title("[5] EMIT NEGATIVE: (none)", fontsize=9)


def _panel_emit_cross_object_correspondence(ax_pos_s, ax_pos_t,
                                              ax_neg_s, ax_neg_t,
                                              img_src, img_tgt, manifest: dict):
    """Frame-level lift: model picks a shared object itself and points
    at it in tgt. POSITIVE = a shared (src∩tgt) object's tgt point.
    NEGATIVE = a tgt-only mask (private), built from any object in
    `manifest['objects']` whose tgt mask exists but isn't in
    `shared_objects`. Falls back to (none) for either side if missing.
    """
    ev = manifest["evidence"]
    shared = ev.get("shared_objects", [])
    qual_idxs = set(ev.get("qualifying_matches", []))

    def _show(ax_s, ax_t):
        ax_s.imshow(img_src); ax_t.imshow(img_tgt)
        for ax in (ax_s, ax_t):
            ax.set_xticks([]); ax.set_yticks([])

    # POSITIVE: a shared-object tgt point (no src query).
    _show(ax_pos_s, ax_pos_t)
    if shared:
        s = shared[0]
        u2, v2 = s["point_tgt"]
        ax_pos_t.plot(u2, v2, "o", ms=18, mec="lime",
                       mfc="none", mew=3.0)
        ax_pos_t.annotate(
            f"answer: {s['tgt_label']}\n"
            f"area={s['tgt_mask_area_frac']*100:.1f}%  "
            f"score={s['score']:.2f}",
            (u2, v2), xytext=(12, -10), textcoords="offset points",
            fontsize=9, color="white",
            bbox=dict(boxstyle="round,pad=0.2",
                      fc="darkgreen", alpha=0.9),
        )
        ax_pos_s.set_title("[5] EMIT POSITIVE: src (context only)",
                            fontsize=9)
        ax_pos_t.set_title(
            f"[5] EMIT POSITIVE: tgt — point on shared '{s['tgt_label']}'",
            fontsize=9)
    else:
        ax_pos_s.set_title("[5] EMIT POSITIVE: (none)", fontsize=9)
        ax_pos_t.set_title("[5] EMIT POSITIVE: (none)", fontsize=9)

    # NEGATIVE: a tgt mask that wasn't matched to any src mask
    # ("private to tgt"). Phase 2 uses this for "point to an object that
    # ISN'T in the other view" complementary task.
    _show(ax_neg_s, ax_neg_t)
    private = [o for o in manifest["objects"]
               if o["match_idx"] not in qual_idxs and o["tgt_mask_id"] >= 0]
    if private:
        o = private[0]
        u2, v2 = o["tgt_centroid"]
        ax_neg_t.plot(u2, v2, "x", ms=18, mec="red", mew=3.5)
        ax_neg_t.annotate(
            f"private answer: {o['tgt_label']}",
            (u2, v2), xytext=(12, -10), textcoords="offset points",
            fontsize=9, color="white",
            bbox=dict(boxstyle="round,pad=0.2",
                      fc="darkred", alpha=0.9),
        )
        ax_neg_s.set_title("[5] EMIT NEGATIVE: src (context only)",
                            fontsize=9)
        ax_neg_t.set_title(
            f"[5] EMIT NEGATIVE: tgt — '{o['tgt_label']}' "
            f"NOT in src view", fontsize=9)
    else:
        ax_neg_s.set_title("[5] EMIT NEGATIVE: (none)", fontsize=9)
        ax_neg_t.set_title("[5] EMIT NEGATIVE: (none)", fontsize=9)


def _show_pair(ax_s, ax_t, img_src, img_tgt):
    ax_s.imshow(img_src); ax_t.imshow(img_tgt)
    for ax in (ax_s, ax_t):
        ax.set_xticks([]); ax.set_yticks([])


def _id_color(obj: dict, scene_id: str):
    key = (f"{scene_id}|{obj['X_world'][0]:.2f}|"
           f"{obj['X_world'][1]:.2f}|{obj['X_world'][2]:.2f}")
    return color_for(key)


def _annotate_obj(ax, point, text, fc="black"):
    u, v = point
    ax.annotate(
        text, (u, v), xytext=(8, -8), textcoords="offset points",
        fontsize=8, color="white",
        bbox=dict(boxstyle="round,pad=0.2", fc=fc, alpha=0.85),
    )


def _placeholder(ax_s, ax_t, img_src, img_tgt, label):
    _show_pair(ax_s, ax_t, img_src, img_tgt)
    ax_s.set_title(f"[5] EMIT {label}: (none)", fontsize=9)
    ax_t.set_title(f"[5] EMIT {label}: (none)", fontsize=9)


def _panel_emit_anchor(ax_pos_s, ax_pos_t, ax_neg_s, ax_neg_t,
                        img_src, img_tgt, manifest: dict):
    ev = manifest["evidence"]
    objs_all = manifest["objects"]
    shared = ev.get("shared_objects", [])
    nontriv = [s for s in shared if s.get("non_trivial")]
    triv = [s for s in shared if not s.get("non_trivial")]

    def _draw(ax_s, ax_t, rows, label):
        _show_pair(ax_s, ax_t, img_src, img_tgt)
        for s in rows:
            obj = objs_all[s["match_idx"]]
            color = _id_color(obj, manifest["scene_id"])
            _draw_bbox(ax_s, obj["src_bbox"], color=color, lw=2.4)
            _draw_bbox(ax_t, obj["tgt_bbox"], color=color, lw=2.4)
            _annotate_obj(ax_s, obj["src_centroid"],
                          f"{obj['src_label']}\nratio={s['scale_ratio']:.2f}")
            _annotate_obj(ax_t, obj["tgt_centroid"],
                          f"{obj['tgt_label']}\nratio={s['scale_ratio']:.2f}")
        ax_s.set_title(f"[5] EMIT {label}: src ({len(rows)})", fontsize=9)
        ax_t.set_title(f"[5] EMIT {label}: tgt", fontsize=9)

    if nontriv:
        _draw(ax_pos_s, ax_pos_t, nontriv, "POSITIVE (non-trivial scale)")
    else:
        _placeholder(ax_pos_s, ax_pos_t, img_src, img_tgt, "POSITIVE")
    if triv:
        _draw(ax_neg_s, ax_neg_t, triv, "NEGATIVE (trivial scale, dropped)")
    else:
        _placeholder(ax_neg_s, ax_neg_t, img_src, img_tgt, "NEGATIVE")


def _panel_emit_counting(ax_pos_s, ax_pos_t, ax_neg_s, ax_neg_t,
                          img_src, img_tgt, manifest: dict):
    ev = manifest["evidence"]
    objs_all = manifest["objects"]
    cat = ev.get("category", "?")
    color = color_for(f"cat|{cat}")

    # POSITIVE: shared + private instances of the winning category.
    _show_pair(ax_pos_s, ax_pos_t, img_src, img_tgt)
    shared_idx = ev.get("shared_match_idx", [])
    for mi in shared_idx:
        obj = objs_all[mi]
        _draw_bbox(ax_pos_s, obj["src_bbox"], color=color, lw=2.4)
        _draw_bbox(ax_pos_t, obj["tgt_bbox"], color=color, lw=2.4)
        _annotate_obj(ax_pos_s, obj["src_centroid"], "shared",
                      fc="darkgreen")
        _annotate_obj(ax_pos_t, obj["tgt_centroid"], "shared",
                      fc="darkgreen")
    ax_pos_s.set_title(
        f"[5] EMIT POSITIVE: src '{cat}' "
        f"shared={len(shared_idx)}  unique_total={ev.get('unique_total', 0)}",
        fontsize=9)
    ax_pos_t.set_title(
        f"[5] EMIT POSITIVE: tgt '{cat}'", fontsize=9)

    # NEGATIVE: private instances marked on whichever side they're in.
    # We don't have direct mask-id-only annotations w/o re-deriving from
    # the perception cache, so just call out counts in the title.
    _show_pair(ax_neg_s, ax_neg_t, img_src, img_tgt)
    n_ps = len(ev.get("private_src_idx", []))
    n_pt = len(ev.get("private_tgt_idx", []))
    ax_neg_s.set_title(
        f"[5] EMIT NEGATIVE: src private '{cat}' = {n_ps} (mask-ids only)",
        fontsize=9)
    ax_neg_t.set_title(
        f"[5] EMIT NEGATIVE: tgt private '{cat}' = {n_pt}",
        fontsize=9)


def _panel_emit_relative_distance(ax_pos_s, ax_pos_t, ax_neg_s, ax_neg_t,
                                    img_src, img_tgt, manifest: dict):
    ev = manifest["evidence"]
    objs_all = manifest["objects"]
    ref_mi = ev.get("reference_match_idx")
    cands = ev.get("candidates", [])

    _show_pair(ax_pos_s, ax_pos_t, img_src, img_tgt)
    if ref_mi is not None and 0 <= ref_mi < len(objs_all):
        obj = objs_all[ref_mi]
        u, v = obj["src_centroid"]
        ax_pos_s.plot(u, v, "*", ms=22, mec="black", mfc="gold", mew=1.5)
        _annotate_obj(ax_pos_s, (u, v),
                      f"REF: {ev.get('reference_label', '?')}",
                      fc="goldenrod")
        ax_pos_t.plot(*obj["tgt_centroid"], "*", ms=22,
                      mec="black", mfc="gold", mew=1.5)
    for rank, c in enumerate(cands):
        obj = objs_all[c["match_idx"]]
        color = _id_color(obj, manifest["scene_id"])
        _draw_bbox(ax_pos_s, obj["src_bbox"], color=color, lw=2.0)
        _draw_bbox(ax_pos_t, obj["tgt_bbox"], color=color, lw=2.0)
        tag = ("FAR" if rank == 0
               else f"#{rank + 1}")
        _annotate_obj(ax_pos_s, obj["src_centroid"],
                      f"[{tag}] {c['label']}\n{c['distance_m']:.2f}m",
                      fc=("darkred" if rank == 0 else "black"))
    margin = float(ev.get("margin_m", 0.0))
    ax_pos_s.set_title(
        f"[5] EMIT POSITIVE: ref + ordered candidates "
        f"(margin={margin:.2f}m)", fontsize=9)
    ax_pos_t.set_title("[5] EMIT POSITIVE: tgt context", fontsize=9)

    _placeholder(ax_neg_s, ax_neg_t, img_src, img_tgt, "NEGATIVE")


def _panel_emit_cross_spatial_transformation(ax_pos_s, ax_pos_t,
                                               ax_neg_s, ax_neg_t,
                                               img_src, img_tgt, manifest: dict):
    ev = manifest["evidence"]
    objs_all = manifest["objects"]
    objs = ev.get("transformed_objects", [])
    _show_pair(ax_pos_s, ax_pos_t, img_src, img_tgt)
    for o in objs:
        obj = objs_all[o["match_idx"]]
        color = _id_color(obj, manifest["scene_id"])
        _draw_bbox(ax_pos_s, obj["src_bbox"], color=color, lw=2.4)
        _draw_bbox(ax_pos_t, obj["tgt_bbox"], color=color, lw=2.4)
        _annotate_obj(ax_pos_s, obj["src_centroid"],
                      f"{o['label']}\nratio={o['scale_ratio']:.2f}")
        _annotate_obj(ax_pos_t, obj["tgt_centroid"],
                      f"{o['label']}\nratio={o['scale_ratio']:.2f}")
    ax_pos_s.set_title(
        f"[5] EMIT POSITIVE: src — transformed objects "
        f"(angle={manifest['pair_angle_deg']:.1f}°)", fontsize=9)
    ax_pos_t.set_title("[5] EMIT POSITIVE: tgt", fontsize=9)
    _placeholder(ax_neg_s, ax_neg_t, img_src, img_tgt, "NEGATIVE")


def _panel_emit_cross_depth_variation(ax_pos_s, ax_pos_t,
                                        ax_neg_s, ax_neg_t,
                                        img_src, img_tgt, manifest: dict):
    ev = manifest["evidence"]
    objs_all = manifest["objects"]
    objs = ev.get("varying_objects", [])
    _show_pair(ax_pos_s, ax_pos_t, img_src, img_tgt)
    for o in objs:
        obj = objs_all[o["match_idx"]]
        color = _id_color(obj, manifest["scene_id"])
        _draw_bbox(ax_pos_s, obj["src_bbox"], color=color, lw=2.4)
        _draw_bbox(ax_pos_t, obj["tgt_bbox"], color=color, lw=2.4)
        _annotate_obj(ax_pos_s, obj["src_centroid"],
                      f"{o['label']}\nd={o['depth_src']:.2f}m")
        _annotate_obj(ax_pos_t, obj["tgt_centroid"],
                      f"{o['label']}\nd={o['depth_tgt']:.2f}m  "
                      f"Δ={o['delta_m']:+.2f}m")
    ax_pos_s.set_title(
        f"[5] EMIT POSITIVE: src depths "
        f"(median |Δ|={ev.get('pair_median_delta_m', 0):.2f}m)",
        fontsize=9)
    ax_pos_t.set_title("[5] EMIT POSITIVE: tgt depths", fontsize=9)
    _placeholder(ax_neg_s, ax_neg_t, img_src, img_tgt, "NEGATIVE")


def _panel_emit_cross_occlusion_visibility(ax_pos_s, ax_pos_t,
                                             ax_neg_s, ax_neg_t,
                                             img_src, img_tgt, manifest: dict):
    ev = manifest["evidence"]
    objs_all = manifest["objects"]
    vis = ev.get("visible_match_idx", [])
    occ = ev.get("occluded_match_idx", [])

    _show_pair(ax_pos_s, ax_pos_t, img_src, img_tgt)
    for mi in vis:
        obj = objs_all[mi]
        color = _id_color(obj, manifest["scene_id"])
        _draw_bbox(ax_pos_s, obj["src_bbox"], color=color, lw=2.0)
        _draw_bbox(ax_pos_t, obj["tgt_bbox"], color=color, lw=2.0)
        ax_pos_t.plot(*obj["tgt_centroid"], "o", ms=14,
                      mec="lime", mfc="none", mew=2.5)
        _annotate_obj(ax_pos_t, obj["tgt_centroid"],
                      f"{obj['tgt_label']}", fc="darkgreen")
    ax_pos_s.set_title(
        f"[5] EMIT POSITIVE (visible): src n={len(vis)}", fontsize=9)
    ax_pos_t.set_title("[5] EMIT POSITIVE (visible): tgt", fontsize=9)

    _show_pair(ax_neg_s, ax_neg_t, img_src, img_tgt)
    for mi in occ:
        obj = objs_all[mi]
        color = _id_color(obj, manifest["scene_id"])
        _draw_bbox(ax_neg_s, obj["src_bbox"], color=color, lw=2.0)
        u, v = obj["point_tgt"]
        ax_neg_t.plot(u, v, "x", ms=16, mec="red", mew=3.0)
        _annotate_obj(ax_neg_s, obj["src_centroid"],
                      f"{obj['src_label']}", fc="darkred")
        _annotate_obj(ax_neg_t, (u, v),
                      f"OCC reproj\nΔd={obj['depth_pred_tgt']-obj['depth_obs_tgt']:+.2f}m",
                      fc="darkred")
    ax_neg_s.set_title(
        f"[5] EMIT NEGATIVE (occluded): src n={len(occ)}", fontsize=9)
    ax_neg_t.set_title(
        "[5] EMIT NEGATIVE (occluded): tgt — reprojection ✗",
        fontsize=9)


def _panel_emit_relative_direction(ax_pos_s, ax_pos_t,
                                     ax_neg_s, ax_neg_t,
                                     img_src, img_tgt, manifest: dict):
    ev = manifest["evidence"]
    objs_all = manifest["objects"]
    targets = ev.get("targets", [])
    _show_pair(ax_pos_s, ax_pos_t, img_src, img_tgt)
    for t in targets:
        obj = objs_all[t["match_idx"]]
        color = _id_color(obj, manifest["scene_id"])
        _draw_bbox(ax_pos_t, obj["tgt_bbox"], color=color, lw=2.4)
        _annotate_obj(
            ax_pos_t, obj["tgt_centroid"],
            f"{t['label']}\n{t['bucket']} ({t['azimuth_deg']:+.0f}°)\n"
            f"d={t['distance_m']:.2f}m",
            fc="navy",
        )
        _draw_bbox(ax_pos_s, obj["src_bbox"], color=color, lw=1.5, ls="--")
    ax_pos_s.set_title(
        "[5] EMIT POSITIVE: src (context) — dashed = tgt-side targets",
        fontsize=9)
    ax_pos_t.set_title(
        f"[5] EMIT POSITIVE: tgt — 8-way compass labels (n={len(targets)})",
        fontsize=9)
    _placeholder(ax_neg_s, ax_neg_t, img_src, img_tgt, "NEGATIVE")


EMIT_DISPATCH = {
    "cross_point_correspondence": _panel_emit_cross_point_correspondence,
    "cross_object_correspondence": _panel_emit_cross_object_correspondence,
    "anchor": _panel_emit_anchor,
    "counting": _panel_emit_counting,
    "relative_distance": _panel_emit_relative_distance,
    "cross_spatial_transformation": _panel_emit_cross_spatial_transformation,
    "cross_depth_variation": _panel_emit_cross_depth_variation,
    "cross_occlusion_visibility": _panel_emit_cross_occlusion_visibility,
    "relative_direction": _panel_emit_relative_direction,
}


# ---- driver -------------------------------------------------------------

def _render_pair(manifest: dict, skill: str, cfg: dict,
                  cache_root: Path, adapter: str, out_path: Path):
    img_src = np.array(Image.open(manifest["image_src"]))
    img_tgt = np.array(Image.open(manifest["image_tgt"]))
    masks_s = load_frame_masks(cache_root, adapter, manifest["scene_id"],
                                manifest["frame_src"])
    masks_t = load_frame_masks(cache_root, adapter, manifest["scene_id"],
                                manifest["frame_tgt"])
    qualifying = list(manifest["evidence"].get("qualifying_matches", []))
    occluded = list(manifest["evidence"].get("occluded_candidates", []))

    fig = plt.figure(figsize=(16, 26))
    gs = fig.add_gridspec(
        6, 2,
        height_ratios=[1.0, 1.4, 1.2, 0.7, 1.0, 1.0],
        hspace=0.40, wspace=0.05,
    )
    # (1)
    _panel_pose(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                img_src, img_tgt, manifest,
                frame_gap=abs(int(manifest["frame_tgt"]) - int(manifest["frame_src"])))
    # (2)
    _panel_perception(fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
                      img_src, img_tgt, masks_s, masks_t)
    # (3)
    _panel_match(fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]),
                 img_src, img_tgt, manifest, masks_s, masks_t,
                 qualifying, occluded)
    # (4)
    _panel_gate(fig.add_subplot(gs[3, :]), skill, manifest, cfg)
    # (5) two-row final emit: positive then negative
    emit_fn = EMIT_DISPATCH.get(skill)
    ax_pos_s = fig.add_subplot(gs[4, 0]); ax_pos_t = fig.add_subplot(gs[4, 1])
    ax_neg_s = fig.add_subplot(gs[5, 0]); ax_neg_t = fig.add_subplot(gs[5, 1])
    if emit_fn is not None:
        emit_fn(ax_pos_s, ax_pos_t, ax_neg_s, ax_neg_t,
                img_src, img_tgt, manifest)
    else:
        for ax in (ax_pos_s, ax_pos_t, ax_neg_s, ax_neg_t):
            ax.axis("off")
        ax_pos_s.text(0.5, 0.5,
                      f"final-emit viz for '{skill}' not wired yet",
                      transform=ax_pos_s.transAxes, ha="center",
                      fontsize=10, family="monospace")

    fig.suptitle(
        f"{manifest['scene_id']}  {manifest['frame_src']} → {manifest['frame_tgt']}  "
        f"[skill={skill}]",
        fontsize=12, y=0.995,
    )
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _dedupe_pairs(manifests: list[dict], frame_thresh: int) -> list[dict]:
    """Keep the first-seen manifest for each scene; drop later manifests
    whose src or tgt frame is within `frame_thresh` of any already-kept
    pair's frames in the same scene. Set frame_thresh<=0 to disable."""
    if frame_thresh <= 0:
        return list(manifests)
    kept: list[dict] = []
    kept_frames: dict[str, list[int]] = {}
    for m in manifests:
        scene = m["scene_id"]
        try:
            fs = int(m["frame_src"]); ft = int(m["frame_tgt"])
        except (ValueError, TypeError):
            kept.append(m)
            continue
        seen = kept_frames.get(scene, [])
        too_close = any(abs(fs - x) <= frame_thresh
                        or abs(ft - x) <= frame_thresh
                        for x in seen)
        if too_close:
            continue
        kept.append(m)
        kept_frames.setdefault(scene, []).extend([fs, ft])
    return kept


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--skill", required=True)
    p.add_argument("--task-config", type=Path, default=_CFG_PATH_DEFAULT)
    add_cache_args(p, include_model_tag=False)
    p.add_argument("--limit", type=int, default=None,
                   help="cap number of pairs rendered")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--dedupe-frame-thresh", type=int, default=0,
                   help="drop a pair if either of its frames is within "
                        "this many frames of an already-kept pair in the "
                        "same scene; 0 disables (default)")
    args = p.parse_args()
    perception_root = args.cache_root / "perception"

    pairs_path = args.out_root / "stage_1" / args.skill / "pairs.jsonl"
    if not pairs_path.exists():
        raise SystemExit(f"no pairs.jsonl at {pairs_path}")

    cfg_all = json.loads(args.task_config.read_text())

    manifests = [json.loads(l) for l in pairs_path.read_text().splitlines()
                 if l.strip()]
    n_raw = len(manifests)
    manifests = _dedupe_pairs(manifests, args.dedupe_frame_thresh)
    print(f"deduped {n_raw} -> {len(manifests)} (thresh={args.dedupe_frame_thresh})")
    if args.limit is not None:
        manifests = manifests[: args.limit]

    out_dir = args.out_dir or (args.out_root / "stage_1" / args.skill / "_debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    for m in manifests:
        key = f"{m['scene_id']}_{m['frame_src']}_{m['frame_tgt']}"
        out = out_dir / f"{key}.png"
        print(f"-> {out}")
        _render_pair(m, args.skill, cfg_all, perception_root,
                     args.adapter, out)

    print(f"done. {len(manifests)} pair(s) rendered to {out_dir}")


if __name__ == "__main__":
    main()
