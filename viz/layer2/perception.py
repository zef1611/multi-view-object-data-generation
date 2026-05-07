"""Visualize GD+SAM perception output per frame.

Reads cached per-frame masks from
``cache/perception/<adapter>/<scene>/<model_tag>/<fid>.pkl``
(produced by ``python -m cli generate``). Renders each frame with GD
bboxes + SAM mask overlays + label tags.

Usage:
    python visualize_perception.py --scene scene0000_00 --num 12 \
        --save outputs/2scenes_qwen/_diag_perception_scene0000_00.png
"""

import argparse
import pickle
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from viz import (
    add_cache_args,
    add_scene_args,
    add_scenes_root_arg,
    color_for,
    discover_cfg_dir,
    draw_mask_outline,
    mask_centroid,
)


def main() -> None:
    p = argparse.ArgumentParser()
    add_scene_args(p)
    add_cache_args(p)
    add_scenes_root_arg(p)
    p.add_argument("--num", type=int, default=12, help="number of frames to render")
    p.add_argument("--cols", type=int, default=2)
    p.add_argument("--save", type=Path, default=None)
    args = p.parse_args()

    perception_root = args.cache_root / "perception"
    cache_dir = perception_root / args.adapter / args.scene
    if not cache_dir.exists():
        raise SystemExit(
            f"No perception cache at {cache_dir}. Run `python -m cli "
            f"generate` with the same adapter+scene first."
        )
    if args.model_tag is not None:
        cfg_dir = cache_dir / args.model_tag
        if not cfg_dir.is_dir():
            available = sorted(d.name for d in cache_dir.iterdir() if d.is_dir())
            raise SystemExit(
                f"--model-tag {args.model_tag!r} not found under {cache_dir}. "
                f"Available: {available}"
            )
    else:
        cfg_dir = discover_cfg_dir(perception_root, args.adapter, args.scene)
        if cfg_dir is None:
            raise SystemExit(f"No model-tagged subdirs under {cache_dir}")

    pkls = sorted(cfg_dir.glob("*.pkl"), key=lambda p: int(p.stem))[: args.num]
    if not pkls:
        raise SystemExit(f"No .pkl files under {cfg_dir}")

    # Class counts across the rendered subset, keyed by canonical when
    # available (collapses paraphrases like "office chair"/"folding chair"
    # → "chair") so the header reflects scene-wide categories.
    label_counter: Counter = Counter()
    for pkl in pkls:
        for m in pickle.load(open(pkl, "rb")):
            key = (getattr(m, "canonical", "") or "").strip() or m.label
            label_counter[key] += 1

    n = len(pkls)
    cols = args.cols
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), squeeze=False)

    color_dir = args.scenes_root / args.scene / "color"
    for k, pkl in enumerate(pkls):
        fid = pkl.stem
        ax = axes[k // cols, k % cols]
        img = np.array(Image.open(color_dir / f"{fid}.jpg"))
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])

        masks = pickle.load(open(pkl, "rb"))
        ax.set_title(f"frame {fid}   {len(masks)} detections")

        H, W = img.shape[:2]
        overlay = np.zeros((H, W, 4), dtype=np.float32)
        for i, m in enumerate(masks):
            color = color_for(f"{m.label}|{i}")
            r, g, b = color
            overlay[m.mask] = (r, g, b, 0.45)
        ax.imshow(overlay)
        # Outline + label per mask (no detector bbox).
        for i, m in enumerate(masks):
            color = color_for(f"{m.label}|{i}")
            draw_mask_outline(ax, m.mask, color)
            cen = mask_centroid(m.mask)
            if cen is None:
                continue
            cx, cy = cen
            canon = (getattr(m, "canonical", "") or "").strip()
            tag = (f"{m.label} → {canon}" if canon and canon.lower() != m.label.lower()
                   else m.label)
            ax.annotate(f"{tag} {m.score:.2f}", (cx, cy),
                        ha="center", va="center",
                        fontsize=7, color="white",
                        bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.7))

    for k in range(n, rows * cols):
        axes[k // cols, k % cols].axis("off")

    fig.suptitle(
        f"{args.scene} perception (top-{n} cached frames, cfg={cfg_dir.name})  "
        f"|  classes: " + ", ".join(f"{l}:{c}" for l, c in label_counter.most_common(8)),
        fontsize=10,
    )
    plt.tight_layout()
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=110, bbox_inches="tight")
        print(f"saved -> {args.save}")
        print(f"label counts: {dict(label_counter.most_common())}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
