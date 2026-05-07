"""Visualize ScanNet GT instance masks per frame.

Counterpart to `visualize_perception.py` — same layout, but mask source
is `ScanNetAdapter.qc_instance_mask(frame_id)` (the per-frame GT
`instance-filt/{i}.png` + scene-wide `objectId → label` map). Useful
for eyeball-comparing perception output (GD + SAM + Gemini canonical)
against ground-truth annotations.

Usage:
    python visualize_gt.py --scene scene0000_00 --num 6 \
        --save outputs/dryrun_test/scene0000_00/gt_instances.png
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from datasets.scannet import ScanNetAdapter
from pipeline.label_blocklist import DEFAULT_LABEL_BLOCKLIST
from viz import (
    add_scene_args,
    add_scenes_root_arg,
    color_for,
    draw_mask_outline,
    mask_centroid,
)


def _pick_frames(scene_dir: Path, n: int) -> list[str]:
    color_dir = scene_dir / "color"
    fids = sorted(int(p.stem) for p in color_dir.glob("*.jpg"))
    if not fids:
        raise SystemExit(f"No color frames under {color_dir}")
    # Even-spaced sampling so the rendered frames span the full trajectory.
    if len(fids) <= n:
        sel = fids
    else:
        step = len(fids) / n
        sel = [fids[int(i * step)] for i in range(n)]
    return [str(f) for f in sel]


def main() -> None:
    p = argparse.ArgumentParser()
    add_scene_args(p)
    add_scenes_root_arg(p)
    p.add_argument("--num", type=int, default=6)
    p.add_argument("--cols", type=int, default=2)
    p.add_argument("--frames", type=int, nargs="+", default=None,
                   help="explicit frame ids; default = even-spaced sample")
    p.add_argument("--save", type=Path, default=None)
    p.add_argument("--include-structural", action="store_true",
                   help="include walls / floors / ceilings (default off — "
                        "matches the cosmic visibility-set blocklist)")
    p.add_argument("--min-area-frac", type=float, default=0.005,
                   help="drop instances whose mask covers < this fraction "
                        "of the frame (default 0.5%%)")
    args = p.parse_args()

    adapter = ScanNetAdapter(args.scenes_root / args.scene)
    if args.frames:
        fids = [str(f) for f in args.frames][: args.num]
    else:
        fids = _pick_frames(args.scenes_root / args.scene, args.num)

    blocklist = frozenset() if args.include_structural else DEFAULT_LABEL_BLOCKLIST
    label_counter: Counter = Counter()

    n = len(fids)
    cols = args.cols
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), squeeze=False)

    color_dir = args.scenes_root / args.scene / "color"
    for k, fid in enumerate(fids):
        ax = axes[k // cols, k % cols]
        img = np.array(Image.open(color_dir / f"{fid}.jpg"))
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])

        out = adapter.qc_instance_mask(fid)
        if out is None:
            ax.set_title(f"frame {fid}   (no GT mask)")
            continue
        inst_mask, label_map = out
        H, W = inst_mask.shape[:2]
        img_area = float(H * W)

        ids, counts = np.unique(inst_mask, return_counts=True)
        keep = ids != 0
        ids = ids[keep]; counts = counts[keep]

        kept = []
        overlay = np.zeros((H, W, 4), dtype=np.float32)
        for iid, cnt in zip(ids, counts):
            if cnt / img_area < args.min_area_frac:
                continue
            label = label_map.get(int(iid), str(iid))
            if (label or "").strip().lower() in blocklist:
                continue
            mask = (inst_mask == iid)
            color = color_for(f"{label}|{int(iid)}")
            r, g, b = color
            overlay[mask] = (r, g, b, 0.45)
            kept.append((int(iid), label, mask, color))
            label_counter[label] += 1

        ax.imshow(overlay)
        for iid, label, mask, color in kept:
            draw_mask_outline(ax, mask, color)
            cen = mask_centroid(mask)
            if cen is None:
                continue
            cx, cy = cen
            ax.annotate(f"{iid}: {label}", (cx, cy),
                        ha="center", va="center", fontsize=7, color="white",
                        bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.7))
        ax.set_title(f"frame {fid}   {len(kept)} GT instances")

    for k in range(n, rows * cols):
        axes[k // cols, k % cols].axis("off")

    fig.suptitle(
        f"{args.scene} GT instances (top-{n} sampled frames)  "
        f"|  classes: " + ", ".join(f"{l}:{c}" for l, c
                                    in label_counter.most_common(10)),
        fontsize=10,
    )
    plt.tight_layout()
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=120, bbox_inches="tight")
        print(f"saved -> {args.save}")
        print(f"label counts: {dict(label_counter.most_common())}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
