"""Debug-oriented visualizer for Phase-1 correspondence records.

Output layout (default):
    <jsonl_dir>/inspect/
        pair_<scene>_<fsrc>_<ftgt>.png   # one file per frame pair
        all_pairs.png                    # stacked grid of all pairs

Per-pair rendering shows both views side by side with:
  * every detected object in the frame (faint gray bbox + small label tag)
    — lets you see what the detector found that did NOT become a
    correspondence
  * per-correspondence bbox (bold, color-coded — same color on both sides
    = same physical object)
  * src point: filled ○ in object color
  * tgt point: green hollow ○ (visible) or red ✗ (occluded)
  * caption listing src_label → tgt_label, IoU, Δdepth, visibility
  * optional rejection overlay (--show-rejections) for dropped candidates

Flags:
    --no-masks           skip SAM mask overlays (on by default)
    --no-all-detections  skip faint "all frame detections" overlay
    --show-rejections    draw dropped candidates from *.rejections.jsonl
    --out-dir DIR        override output directory (defaults to
                         <jsonl_dir>/inspect)
    --only <pattern>     only render pairs whose key matches substring

Usage:
    python visualize_correspondences.py \
        --jsonl outputs/debug_3scenes/stage_1/cross_point_correspondence/correspondences.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from viz import (
    PALETTE,
    add_cache_args,
    color_for,
    draw_mask_outline,
    draw_src_point,
    draw_tgt_point,
    load_frame_masks,
    mask_centroid,
)


_MATCHER_CACHE = None


def _get_label_matcher():
    """Lazy CLIP-text matcher; same instance reused across pairs."""
    global _MATCHER_CACHE
    if _MATCHER_CACHE is False:
        return None
    if _MATCHER_CACHE is not None:
        return _MATCHER_CACHE
    try:
        from pipeline.label_matcher import LabelMatcher
        _MATCHER_CACHE = LabelMatcher()
    except Exception:
        _MATCHER_CACHE = False
        return None
    return _MATCHER_CACHE


def _labels_match(matcher, a: str, b: str) -> bool:
    if (a or "").lower() == (b or "").lower():
        return True
    if matcher is None:
        return False
    return matcher.match(a, b)


def _identity_for(rec: dict, side: str) -> str:
    """Cross-frame identity for a record: canonical when present, else the
    specific label. `side` is 'src' or 'tgt'."""
    canon = (rec.get(f"{side}_canonical") or "").strip()
    if canon:
        return canon
    return (rec.get(f"{side}_label") or "").strip()


def _caption_line(idx: int, r: dict) -> str:
    src_l = r["src_label"]
    tgt_l = r.get("tgt_label", src_l)
    src_id = _identity_for(r, "src")
    tgt_id = _identity_for(r, "tgt")
    match = "=" if src_id.lower() == tgt_id.lower() else "≠"
    iou = r.get("iou_src_to_tgt", 0.0)
    d_obs = r.get("depth_obs_tgt")
    d_pred = r.get("depth_pred_tgt")
    if d_pred is not None and d_obs is not None:
        ddepth = f"Δd={d_pred - d_obs:+.2f}m (pred {d_pred:.2f} vs obs {d_obs:.2f})"
    else:
        d_src = r.get("depth_src")
        ddepth = f"d_src={d_src:.2f}m" if d_src is not None else "d=?"
    vis = "✓" if r.get("visible", True) else "✗OCC"
    # Show specific labels and the canonical identity used for matching.
    src_show = f"{src_l}/{src_id}" if src_id and src_id != src_l.lower() else src_l
    tgt_show = f"{tgt_l}/{tgt_id}" if tgt_id and tgt_id != tgt_l.lower() else tgt_l
    return f"[{idx}] '{src_show}' {match} '{tgt_show}' | iou={iou:.2f} {ddepth} {vis}"


def _draw_all_detections(ax, masks_list, used_ids: set,
                         show_masks: bool, img_shape):
    """Color every un-matched detection with a label-stable hue so you can
    eyeball whether the same object kind shows up consistently across
    frames. Correspondence-matched masks are drawn separately elsewhere
    (bolder outline + point marker)."""
    if not masks_list:
        return
    H, W = img_shape[:2]
    if show_masks:
        ov = np.zeros((H, W, 4), dtype=np.float32)
        for i, m in enumerate(masks_list):
            if i in used_ids:
                continue
            rc, gc, bc = color_for(f"label|{m.label.lower()}")
            ov[m.mask] = (rc, gc, bc, 0.30)
        ax.imshow(ov)
    for i, m in enumerate(masks_list):
        if i in used_ids:
            continue
        canon = (getattr(m, "canonical", "") or "").strip()
        color_key = canon.lower() if canon else m.label.lower()
        color = color_for(f"label|{color_key}")
        draw_mask_outline(ax, m.mask, color=color, lw=1.2, ls=":")
        cen = mask_centroid(m.mask)
        if cen is None:
            continue
        cx, cy = cen
        tag = (f"{m.label} → {canon}" if canon and canon.lower() != m.label.lower()
               else m.label)
        ax.annotate(f"{tag} {m.score:.2f}", (cx, cy),
                    ha="center", va="center",
                    fontsize=6, color="white",
                    bbox=dict(boxstyle="round,pad=0.12",
                              fc=tuple(ch * 0.6 for ch in color), alpha=0.75))


def _render_pair(ax_s, ax_t, recs, rejs, scene, fsrc, ftgt,
                 masks_s, masks_t, show_masks: bool,
                 show_all_detections: bool, show_rejections: bool,
                 show_points: bool = True):
    img_src = np.array(Image.open(recs[0]["image_src"]))
    img_tgt = np.array(Image.open(recs[0]["image_tgt"]))
    ax_s.imshow(img_src); ax_t.imshow(img_tgt)
    for ax in (ax_s, ax_t):
        ax.set_xticks([]); ax.set_yticks([])

    overlap = recs[0].get("pair_overlap", 0.0)
    n_occ = sum(1 for r in recs if not r.get("visible", True))
    matcher = _get_label_matcher()
    n_lbl_mis = sum(
        1 for r in recs
        if not _labels_match(matcher, _identity_for(r, "src"),
                             _identity_for(r, "tgt"))
    )
    n_det_s = len(masks_s) if masks_s else 0
    n_det_t = len(masks_t) if masks_t else 0
    ax_s.set_title(
        f"{scene}  src={fsrc}  overlap={overlap:.2f}  "
        f"{len(recs)} corr | {n_occ} occluded | {n_lbl_mis} label≠ | "
        f"{len(rejs)} rejected  (det src={n_det_s})",
        fontsize=9,
    )
    ax_t.set_title(f"{scene}  tgt={ftgt}  (det tgt={n_det_t})", fontsize=9)

    used_s = {r.get("src_mask_id") for r in recs if r.get("src_mask_id") is not None}
    used_t = {r.get("tgt_mask_id") for r in recs if r.get("tgt_mask_id") is not None}

    # Highlighted correspondence masks on top of the base image.
    if show_masks and masks_s:
        ov_s = np.zeros(img_src.shape[:2] + (4,), dtype=np.float32)
        for r in recs:
            mid = r.get("src_mask_id")
            if mid is not None and mid < len(masks_s):
                ck = f"{scene}|{r['X_world'][0]:.2f}|{r['X_world'][1]:.2f}|{r['X_world'][2]:.2f}"
                rc, gc, bc = color_for(ck)
                ov_s[masks_s[mid].mask] = (rc, gc, bc, 0.40)
        ax_s.imshow(ov_s)
    if show_masks and masks_t:
        ov_t = np.zeros(img_tgt.shape[:2] + (4,), dtype=np.float32)
        for r in recs:
            mid = r.get("tgt_mask_id")
            if mid is not None and mid < len(masks_t):
                ck = f"{scene}|{r['X_world'][0]:.2f}|{r['X_world'][1]:.2f}|{r['X_world'][2]:.2f}"
                rc, gc, bc = color_for(ck)
                ov_t[masks_t[mid].mask] = (rc, gc, bc, 0.40)
        ax_t.imshow(ov_t)

    # Faint "everything else the detector found" overlay.
    if show_all_detections:
        _draw_all_detections(ax_s, masks_s, used_s, show_masks, img_src.shape)
        _draw_all_detections(ax_t, masks_t, used_t, show_masks, img_tgt.shape)

    # Rejections — record-level rejections only carry bbox, no mask
    # array, so per the masks-only viz policy we skip drawing them.
    # The rejection counts still appear in the title.

    # Correspondences on top.
    caption_lines = []
    for idx, r in enumerate(recs, 1):
        ck = f"{scene}|{r['X_world'][0]:.2f}|{r['X_world'][1]:.2f}|{r['X_world'][2]:.2f}"
        color = color_for(ck)
        sid = r.get("src_mask_id")
        tid = r.get("tgt_mask_id")
        if masks_s and sid is not None and sid < len(masks_s):
            draw_mask_outline(ax_s, masks_s[sid].mask, color=color, lw=2.0)
        if masks_t and tid is not None and tid < len(masks_t):
            draw_mask_outline(ax_t, masks_t[tid].mask, color=color, lw=2.0)
        if show_points:
            draw_src_point(ax_s, *r["point_src"], color=color, text=f"[{idx}]")
            draw_tgt_point(ax_t, *r["point_tgt"],
                            visible=r.get("visible", True), text=f"[{idx}]")
        caption_lines.append(_caption_line(idx, r))

    ax_s.annotate("\n".join(caption_lines), xy=(0, 0),
                  xycoords="axes fraction",
                  xytext=(0, -18), textcoords="offset points",
                  ha="left", va="top", fontsize=8, family="monospace",
                  annotation_clip=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", type=Path, required=True)
    p.add_argument("--scene", help="filter by scene_id")
    p.add_argument("--only", default=None,
                   help="substring match on pair key 'scene_fsrc_ftgt'")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="directory for PNGs; defaults to <jsonl_dir>/inspect")
    p.add_argument("--no-masks", action="store_true")
    p.add_argument("--no-points", action="store_true",
                   help="skip per-correspondence point markers (mask-only view)")
    p.add_argument("--no-all-detections", action="store_true")
    p.add_argument("--show-rejections", action="store_true")
    add_cache_args(p)
    p.add_argument("--limit", type=int, default=None,
                   help="cap the number of pairs rendered")
    args = p.parse_args()
    perception_root = args.cache_root / "perception"

    records = [json.loads(l) for l in args.jsonl.read_text().splitlines() if l]
    if args.scene:
        records = [r for r in records if r["scene_id"] == args.scene]
    if not records:
        print("No records to visualize.")
        return

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["scene_id"], r["frame_src"], r["frame_tgt"])].append(r)

    rej_path = args.jsonl.parent / (args.jsonl.stem + ".rejections.jsonl")
    rej_groups: dict[tuple, list[dict]] = defaultdict(list)
    if args.show_rejections and rej_path.exists():
        for ln in rej_path.read_text().splitlines():
            if not ln:
                continue
            rr = json.loads(ln)
            rej_groups[(rr.get("scene_id"), rr.get("frame_src"),
                        rr.get("frame_tgt"))].append(rr)

    out_dir = args.out_dir or (args.jsonl.parent / "inspect")
    out_dir.mkdir(parents=True, exist_ok=True)

    show_masks = not args.no_masks
    show_all = not args.no_all_detections

    pair_keys = list(groups.keys())
    if args.only:
        pair_keys = [k for k in pair_keys
                     if args.only in f"{k[0]}_{k[1]}_{k[2]}"]
    if args.limit:
        pair_keys = pair_keys[: args.limit]

    summary = Counter()
    # Cache per-scene cfg_dir lookup once, then per-(scene,frame) masks.
    from viz.cache_io import discover_cfg_dir
    cfg_dir_cache: dict[str, object] = {}
    mask_cache: dict[tuple, object] = {}

    def _masks(scene, fid):
        key = (scene, fid)
        if key in mask_cache:
            return mask_cache[key]
        if scene not in cfg_dir_cache:
            cfg_dir_cache[scene] = discover_cfg_dir(
                perception_root, args.adapter, scene,
                model_tag=args.model_tag)
        mask_cache[key] = load_frame_masks(
            perception_root, args.adapter, scene, fid,
            cfg_dir=cfg_dir_cache[scene])
        return mask_cache[key]

    # --- Per-pair PNGs ---
    for key in pair_keys:
        scene, fsrc, ftgt = key
        recs = groups[key]
        rejs = rej_groups.get(key, [])
        fig, (ax_s, ax_t) = plt.subplots(1, 2, figsize=(14, 5.5))
        _render_pair(ax_s, ax_t, recs, rejs, scene, fsrc, ftgt,
                     _masks(scene, fsrc), _masks(scene, ftgt),
                     show_masks=show_masks,
                     show_all_detections=show_all,
                     show_rejections=args.show_rejections,
                     show_points=not args.no_points)
        out = out_dir / f"pair_{scene}_{fsrc}_{ftgt}.png"
        plt.tight_layout(rect=(0, 0.02, 1, 1))
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        summary["pairs"] += 1
        summary["emitted"] += len(recs)
        summary["occluded"] += sum(1 for r in recs if not r.get("visible", True))
        _matcher = _get_label_matcher()
        summary["label_mismatch"] += sum(
            1 for r in recs
            if not _labels_match(_matcher, _identity_for(r, "src"),
                                 _identity_for(r, "tgt"))
        )
        summary["rejected"] += len(rejs)
    print(f"wrote {summary['pairs']} per-pair PNGs to {out_dir}")

    # --- Combined grid ---
    n = len(pair_keys)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(14, 5.2 * n), squeeze=False)
    for row, key in enumerate(pair_keys):
        scene, fsrc, ftgt = key
        recs = groups[key]
        rejs = rej_groups.get(key, [])
        _render_pair(axes[row, 0], axes[row, 1], recs, rejs, scene, fsrc, ftgt,
                     _masks(scene, fsrc), _masks(scene, ftgt),
                     show_masks=show_masks,
                     show_all_detections=show_all,
                     show_rejections=args.show_rejections,
                     show_points=not args.no_points)
    banner = (
        f"{args.jsonl.name}  |  {n} pairs shown / {len(groups)} total  |  "
        f"emitted={summary['emitted']}  occluded={summary['occluded']}  "
        f"label≠={summary['label_mismatch']}"
    )
    if args.show_rejections:
        banner += f"  rejected={summary['rejected']}"
    fig.suptitle(banner, fontsize=10)
    combined = out_dir / "all_pairs.png"
    plt.tight_layout(rect=(0, 0, 1, 0.995))
    fig.savefig(combined, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote combined grid -> {combined}")
    print(f"summary: {dict(summary)}")


if __name__ == "__main__":
    main()
