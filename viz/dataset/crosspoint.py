"""Visualize CrossPoint-378K samples.

Usage:
    python visualize_crosspoint.py                       # random sample
    python visualize_crosspoint.py --type cross_correspondence --num 6
    python visualize_crosspoint.py --index 42 --save out.png
    python visualize_crosspoint.py --type single_fine_grounding --num 4 --save grid.png

Shows the image(s), overlays point1/point2 (or points parsed from the prompt
and assistant JSON for single_* tasks), and prints Q/A text below each panel.
"""

import argparse
import json
import random
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image

DEFAULT_ROOT = Path("/home/mila/l/leh/scratch/dataset/CrossPoint-378k")
DEFAULT_JSON = DEFAULT_ROOT / "CrossPoint-378K" / "CrossPoint-378K.json"

POINT_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")


def parse_points_from_text(text):
    return [(int(x), int(y)) for x, y in POINT_RE.findall(text or "")]


def extract_sample_points(sample):
    """Return a list of points per image (list[list[(x,y)]])."""
    imgs = sample.get("images", [])
    per_image = [[] for _ in imgs]

    if "point1" in sample and len(imgs) >= 1:
        per_image[0].append(tuple(sample["point1"]))
    if "point2" in sample and len(imgs) >= 2:
        per_image[1].append(tuple(sample["point2"]))

    # Fall back to parsing coordinates from text for single_* tasks
    if not any(per_image) and imgs:
        msgs = sample.get("messages", [])
        user_text = next((m["content"] for m in msgs if m["role"] == "user"), "")
        asst_text = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        for pt in parse_points_from_text(user_text) + parse_points_from_text(asst_text):
            per_image[0].append(pt)

    return per_image


def draw_sample(sample, root, axes):
    imgs = sample.get("images", [])
    pts_per_image = extract_sample_points(sample)

    msgs = sample.get("messages", [])
    user_text = next((m["content"] for m in msgs if m["role"] == "user"), "").replace("<image>\n", "")
    asst_text = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    stype = sample.get("type", "?")

    for i, ax in enumerate(axes):
        ax.set_xticks([]); ax.set_yticks([])
        if i >= len(imgs):
            ax.axis("off")
            continue

        img_path = root / imgs[i]
        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"[missing]\n{img_path.name}\n{e}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
            continue

        for (x, y) in pts_per_image[i]:
            ax.add_patch(Circle((x, y), radius=max(img.size) * 0.012,
                                facecolor="red", edgecolor="white", linewidth=1.5, alpha=0.9))
        ax.set_title(f"image{i+1}: {Path(imgs[i]).name}", fontsize=8)

    caption = f"[{stype}]\nQ: {user_text.strip()}\nA: {asst_text.strip()}"
    axes[0].figure.text(
        0.5, 0.02 if len(axes) <= 2 else 0.0,
        "\n".join(textwrap.wrap(caption, width=140)),
        ha="center", va="bottom", fontsize=8, family="monospace",
    )


def pick_samples(data, args):
    if args.index is not None:
        return [data[args.index]]
    pool = [d for d in data if args.type is None or d.get("type") == args.type]
    if not pool:
        raise SystemExit(f"No samples of type {args.type!r}")
    rng = random.Random(args.seed)
    return rng.sample(pool, k=min(args.num, len(pool)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="Dataset root (image paths in the JSON are relative to this).")
    ap.add_argument("--type", choices=[
        "single_spatial_understanding", "single_fine_grounding",
        "cross_correspondence", "cross_spatial_transformation",
        "cross_depth_variation", "cross_occlusion_visibility",
    ], default=None)
    ap.add_argument("--index", type=int, default=None, help="Show a specific sample index.")
    ap.add_argument("--num", type=int, default=1, help="Number of samples to show.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save", type=Path, default=None, help="Save to file instead of showing.")
    args = ap.parse_args()

    print(f"Loading {args.json} ...")
    data = json.loads(args.json.read_text())
    print(f"Loaded {len(data):,} samples.")

    samples = pick_samples(data, args)

    for k, sample in enumerate(samples):
        n_imgs = max(1, len(sample.get("images", [])))
        fig, axes = plt.subplots(1, n_imgs, figsize=(6 * n_imgs, 6))
        if n_imgs == 1:
            axes = [axes]
        draw_sample(sample, args.root, axes)
        fig.tight_layout(rect=(0, 0.12, 1, 1))

        if args.save:
            out = args.save if len(samples) == 1 else args.save.with_stem(f"{args.save.stem}_{k:02d}")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  saved {out}")
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
