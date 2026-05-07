"""Visualize what happens for a single (frame_src, frame_tgt) pair:
  - all GD+SAM detections in src and tgt with bbox + label
  - which src masks became accepted matches (lines connecting src→tgt)
  - which src masks were rejected, with reason

Reads cached perception (cache/perception/...) and re-runs match_pair to
see why coverage is sparse on a specific pair.

Usage:
    python visualize_pair_match.py --scene scene0000_00 \
        --src 500 --tgt 550 --save outputs/_diag/pair_500_550.png
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from datasets.scannet import ScanNetAdapter
from pipeline.match import match_pair
from viz import (
    add_cache_args,
    add_scene_args,
    add_scenes_root_arg,
    color_for,
    draw_mask_outline,
    mask_centroid,
)


def main() -> None:
    p = argparse.ArgumentParser()
    add_scene_args(p)
    add_cache_args(p, include_model_tag=False)
    add_scenes_root_arg(p)
    p.add_argument("--src", required=True)
    p.add_argument("--tgt", required=True)
    p.add_argument("--iou-min", type=float, default=0.20)
    p.add_argument("--depth-tol", type=float, default=0.10)
    p.add_argument("--seed-retries", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--emit-occlusion-negatives", action="store_true", default=True)
    p.add_argument("--save", type=Path, default=None)
    args = p.parse_args()

    perception_root = args.cache_root / "perception"
    cache_dir = perception_root / args.adapter / args.scene
    cfg_dirs = [d for d in cache_dir.iterdir() if d.is_dir()]
    if not cfg_dirs:
        raise SystemExit(f"No perception cache under {cache_dir}")
    # Prefer dirs that actually contain BOTH frames; among those, pick the
    # most recent. Falls back to "most recent overall" if none qualify.
    src_name = f"{args.src}.pkl"; tgt_name = f"{args.tgt}.pkl"
    candidates = [d for d in cfg_dirs
                  if (d / src_name).exists() and (d / tgt_name).exists()]
    if not candidates:
        raise SystemExit(
            f"No cache dir under {cache_dir} contains both {src_name} and {tgt_name}. "
            f"Available dirs: {[d.name for d in cfg_dirs]}"
        )
    cfg_dir = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    src_pkl = cfg_dir / src_name
    tgt_pkl = cfg_dir / tgt_name

    masks_src = pickle.load(open(src_pkl, "rb"))
    masks_tgt = pickle.load(open(tgt_pkl, "rb"))

    adapter = ScanNetAdapter(args.scenes_root / args.scene)
    f_src = adapter.load_frame(args.src)
    f_tgt = adapter.load_frame(args.tgt)

    # Run match with rejection logging.
    rejections = []
    matches = match_pair(
        adapter, f_src, masks_src, f_tgt, masks_tgt,
        seed=args.seed, seed_retries=args.seed_retries,
        depth_tol_m=args.depth_tol, iou_min=args.iou_min,
        emit_occlusion_negatives=args.emit_occlusion_negatives,
        on_reject=lambda s_idx, reason: rejections.append((s_idx, reason)),
    )
    matched_src_idx = {m.src_mask_idx: m for m in matches}

    img_src = np.array(Image.open(f_src.image_path))
    img_tgt = np.array(Image.open(f_tgt.image_path))

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, img, masks, side in zip(
        axes, [img_src, img_tgt], [masks_src, masks_tgt], ["src", "tgt"]
    ):
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])

        H, W = img.shape[:2]
        overlay = np.zeros((H, W, 4), dtype=np.float32)
        for i, m in enumerate(masks):
            color = color_for(f"{side}|{i}")
            r, g, b = color
            overlay[m.mask] = (r, g, b, 0.40)
        ax.imshow(overlay)
        # Mask silhouette outlines + centroid-anchored labels (no bboxes).
        for i, m in enumerate(masks):
            color = color_for(f"{side}|{i}")
            draw_mask_outline(ax, m.mask, color)
            cen = mask_centroid(m.mask)
            if cen is None:
                continue
            cx, cy = cen
            canon = (getattr(m, "canonical", "") or "").strip()
            label_show = (f"{m.label} → {canon}"
                          if canon and canon.lower() != m.label.lower()
                          else m.label)
            tag = f"{side}#{i} {label_show} {m.score:.2f}"
            ax.annotate(tag, (cx, cy), ha="center", va="center",
                        fontsize=8, color="white",
                        bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.7))

    # Overlay accepted matches with src→tgt arrows / labels
    for s_idx, m in matched_src_idx.items():
        u_s, v_s = m.p_src
        u_t, v_t = m.p_tgt
        marker = "x" if not m.visible else "o"
        edge = "red" if not m.visible else "lime"
        axes[0].plot(u_s, v_s, marker, ms=14, mec=edge, mfc="none", mew=3)
        axes[1].plot(u_t, v_t, marker, ms=14, mec=edge, mfc="none", mew=3)
        status = "OCCLUDED" if not m.visible else f"iou={m.iou:.2f}"
        axes[1].annotate(f"src#{s_idx}→tgt#{m.tgt_mask_idx} {status}",
                         (u_t, v_t), xytext=(10, 10), textcoords="offset points",
                         fontsize=8, color="white",
                         bbox=dict(boxstyle="round,pad=0.2",
                                   fc="darkred" if not m.visible else "darkgreen",
                                   alpha=0.85))

    # Title with rejection summary
    n_total = len(masks_src)
    n_acc = sum(1 for m in matches if m.visible)
    n_occ = sum(1 for m in matches if not m.visible)
    n_rej = len(rejections)
    from collections import Counter
    rej_breakdown = Counter(r for _, r in rejections)
    axes[0].set_title(
        f"{args.scene} src={args.src}  src_masks={n_total}  "
        f"accepted={n_acc} occluded={n_occ} rejected={n_rej}\n"
        f"rejections: {dict(rej_breakdown)}",
        fontsize=10,
    )
    axes[1].set_title(
        f"tgt={args.tgt}  tgt_masks={len(masks_tgt)}",
        fontsize=10,
    )

    plt.tight_layout()
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=120, bbox_inches="tight")
        print(f"saved -> {args.save}")
    else:
        plt.show()
    print(f"src_masks={n_total} accepted={n_acc} occluded={n_occ} rejected={n_rej}")
    print(f"rejection breakdown: {dict(rej_breakdown)}")


if __name__ == "__main__":
    main()
