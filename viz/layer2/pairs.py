"""Visualize the view-pairs selected by `pipeline.pairs.select_pairs`.

Renders pairs as side-by-side image grid with overlap/angle/distance
annotations. Optionally overlays the corner-grid projection so you can see
which corners of the src image actually land inside the tgt frame.

Usage:
    python visualize_pairs.py --scene scene0000_00 --num 6 --save pairs.png
    python visualize_pairs.py --scene scene0000_00 --num 12 \
        --frame-stride 50 --max-pairs 100 --save pairs_full.png
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from datasets.base import default_reproject_with_depth
from datasets.scannet import ScanNetAdapter
from pipeline.pairs import select_pairs
from viz import (
    add_scene_args,
    add_scenes_root_arg,
    draw_mask_outline,
    mask_centroid,
)


def _overlay_instances(ax_s, ax_t, adapter, pair, f_src, f_tgt,
                       label_map, *, area_min_frac, depth_pix_min):
    """Outline visible instances per frame; green = in O_src ∩ O_tgt,
    red = view-exclusive. Annotates each contour with its label."""
    from pipeline.cosmic import compute_visibility_set

    O_src = compute_visibility_set(
        adapter, f_src,
        area_min_frac=area_min_frac, depth_pix_min=depth_pix_min,
    ) or frozenset()
    O_tgt = compute_visibility_set(
        adapter, f_tgt,
        area_min_frac=area_min_frac, depth_pix_min=depth_pix_min,
    ) or frozenset()
    shared = O_src & O_tgt

    def _draw(ax, frame, visible_ids):
        out = adapter.qc_instance_mask(frame.frame_id)
        if out is None:
            return
        inst_mask, _ = out
        for iid in sorted(visible_ids):
            mask = (inst_mask == iid)
            if not mask.any():
                continue
            color = "lime" if iid in shared else "red"
            draw_mask_outline(ax, mask, color, lw=1.4)
            cen = mask_centroid(mask)
            if cen is None:
                continue
            cx, cy = cen
            label = label_map.get(int(iid), str(iid))
            ax.text(cx, cy, f"{iid}:{label}",
                    fontsize=7, color="white", ha="center", va="center",
                    bbox=dict(facecolor=color, edgecolor="none",
                              alpha=0.75, pad=1))

    _draw(ax_s, f_src, O_src)
    _draw(ax_t, f_tgt, O_tgt)


def _project_grid(adapter, src_frame, tgt_frame, grid: int = 5):
    """Return list of (u, v, hit) tuples for the corner grid."""
    W, H = src_frame.image_size
    out_src, out_tgt = [], []
    for u in np.linspace(0, W - 1, grid):
        for v in np.linspace(0, H - 1, grid):
            rep = default_reproject_with_depth(src_frame, (float(u), float(v)), tgt_frame)
            hit = rep is not None and rep.in_bounds
            out_src.append((u, v, hit))
            if hit:
                out_tgt.append((rep.u, rep.v))
    return out_src, out_tgt


def main() -> None:
    p = argparse.ArgumentParser()
    add_scene_args(p)
    add_scenes_root_arg(p)
    p.add_argument("--num", type=int, default=6, help="number of pairs to draw")
    p.add_argument("--sampling", choices=["adaptive", "stride", "cosmic"],
                   default="adaptive")
    p.add_argument("--frame-stride", type=int, default=50)
    p.add_argument("--min-translation-m", type=float, default=0.40)
    p.add_argument("--min-rotation-deg", type=float, default=25.0)
    p.add_argument("--limit-frames", type=int, default=None)
    p.add_argument("--cosmic-base-sampling", choices=["adaptive", "stride"],
                   default="stride")
    p.add_argument("--cosmic-union-coverage-min", type=float, default=0.3)
    p.add_argument("--cosmic-yaw-diff-min-deg", type=float, default=30.0)
    p.add_argument("--cosmic-obj-vis-area-min", type=float, default=0.005)
    p.add_argument("--cosmic-obj-vis-depth-pix-min", type=int, default=50)
    p.add_argument("--save", type=Path, default=None)
    p.add_argument("--no-grid", action="store_true",
                   help="hide the corner-grid projection overlay")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    adapter = ScanNetAdapter(args.scenes_root / args.scene)
    from pipeline.config import load_config, resolve
    cfg = resolve(load_config(), getattr(adapter, "source_name", "unknown"))
    pairs = select_pairs(
        adapter, cfg,
        sampling=args.sampling,
        frame_stride=args.frame_stride,
        min_translation_m=args.min_translation_m,
        min_rotation_deg=args.min_rotation_deg,
        limit_frames=args.limit_frames,
        cosmic_base_sampling=args.cosmic_base_sampling,
        cosmic_union_coverage_min=args.cosmic_union_coverage_min,
        cosmic_yaw_diff_min_deg=args.cosmic_yaw_diff_min_deg,
        cosmic_obj_vis_area_min=args.cosmic_obj_vis_area_min,
        cosmic_obj_vis_depth_pix_min=args.cosmic_obj_vis_depth_pix_min,
    )
    if not pairs:
        print("No pairs survived the filters.")
        return

    pairs = pairs[: args.num]
    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n), squeeze=False)

    # In cosmic mode, also overlay the per-frame instance masks colored by
    # shared (green) vs view-exclusive (red) so the gate's predicate is
    # visible at a glance.
    label_map = adapter._aggregation() if args.sampling == "cosmic" else {}

    for row, pair in enumerate(pairs):
        f_src = adapter.load_frame(pair.src_id)
        f_tgt = adapter.load_frame(pair.tgt_id)
        img_src = np.array(Image.open(f_src.image_path))
        img_tgt = np.array(Image.open(f_tgt.image_path))

        ax_s, ax_t = axes[row, 0], axes[row, 1]
        ax_s.imshow(img_src)
        ax_t.imshow(img_tgt)

        meta = getattr(pair, "cosmic_meta", None)
        if meta is not None:
            header = (f"yaw_diff={meta['yaw_diff']:.0f}°  "
                      f"shared={meta['n_intersection']}  "
                      f"cov={meta['coverage']:.2f}  "
                      f"|O_src|={meta['n_src']} |O_tgt|={meta['n_tgt']}")
        else:
            header = (f"overlap={pair.overlap:.2f}  "
                      f"angle={pair.angle_deg:.0f}°  "
                      f"dist={pair.distance_m:.2f} m")
        ax_s.set_title(f"src {pair.src_id}")
        ax_t.set_title(f"tgt {pair.tgt_id}    {header}")
        for ax in (ax_s, ax_t):
            ax.set_xticks([]); ax.set_yticks([])

        if args.sampling == "cosmic":
            _overlay_instances(
                ax_s, ax_t, adapter, pair, f_src, f_tgt,
                label_map,
                area_min_frac=args.cosmic_obj_vis_area_min,
                depth_pix_min=args.cosmic_obj_vis_depth_pix_min,
            )
        elif not args.no_grid:
            src_pts, tgt_pts = _project_grid(adapter, f_src, f_tgt, grid=5)
            for u, v, hit in src_pts:
                color = "lime" if hit else "red"
                ax_s.plot(u, v, "o", ms=6, mec="black", mfc=color, mew=1)
            for u, v in tgt_pts:
                ax_t.plot(u, v, "o", ms=6, mec="black", mfc="lime", mew=1)

    plt.tight_layout()
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=120, bbox_inches="tight")
        print(f"saved -> {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
