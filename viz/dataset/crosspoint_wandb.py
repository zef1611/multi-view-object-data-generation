"""Log CrossPoint-378K cross_* samples to wandb Tables (one table per category).

For each of the four cross-view categories we pick up to N samples with UNIQUE
image pairs (no (img1, img2) tuple is visualized twice) and log them to a
wandb Table. Each row contains the two images side-by-side with point1/point2
overlaid, the Q/A text, and a data-source tag (`scannet` vs `scannet++`).

Source heuristic (verified by inspecting image subdirs of the dataset):
  - scene directories named ``scene0XXX_YY`` come from ScanNet.
  - scene directories named as 10-char hex hashes come from ScanNet++.

Usage:
    python visualize_crosspoint_wandb.py
    python visualize_crosspoint_wandb.py --num 300 --project crosspoint-viz
"""

import argparse
import io
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw
import wandb

DEFAULT_ROOT = Path("/home/mila/l/leh/scratch/dataset/CrossPoint-378k")
DEFAULT_JSON = DEFAULT_ROOT / "CrossPoint-378K" / "CrossPoint-378K.json"

CROSS_TYPES = [
    "cross_correspondence",
    "cross_spatial_transformation",
    "cross_depth_variation",
    "cross_occlusion_visibility",
]

SCANNET_SCENE_RE = re.compile(r"^scene\d{4}_\d{2}$")


def classify_source(image_path: str) -> str:
    """scannet if scene0XXX_YY, else scannet++ (hex hash scene dirs)."""
    parts = Path(image_path).parts
    for p in parts:
        if SCANNET_SCENE_RE.match(p):
            return "scannet"
        if re.fullmatch(r"[0-9a-f]{10}", p):
            return "scannet++"
    return "unknown"


def scene_id(image_path: str) -> str:
    parts = Path(image_path).parts
    for p in parts:
        if SCANNET_SCENE_RE.match(p) or re.fullmatch(r"[0-9a-f]{10}", p):
            return p
    return ""


def draw_point(img: Image.Image, xy, radius_frac=0.012):
    if xy is None:
        return
    x, y = xy
    r = max(3, int(max(img.size) * radius_frac))
    d = ImageDraw.Draw(img)
    d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0), outline=(255, 255, 255), width=2)


def make_pair_viz(root: Path, img_rel_1: str, img_rel_2: str, p1, p2, max_side=512):
    """Return a PIL image (two images concatenated horizontally) with points drawn."""
    im1 = Image.open(root / img_rel_1).convert("RGB")
    im2 = Image.open(root / img_rel_2).convert("RGB")
    draw_point(im1, p1)
    draw_point(im2, p2)

    # Downscale for wandb payload size; scale point coords implicitly by resizing after draw.
    def shrink(im):
        w, h = im.size
        s = min(1.0, max_side / max(w, h))
        if s < 1.0:
            im = im.resize((int(w * s), int(h * s)), Image.BILINEAR)
        return im

    im1, im2 = shrink(im1), shrink(im2)
    h = max(im1.height, im2.height)
    canvas = Image.new("RGB", (im1.width + im2.width + 8, h), (20, 20, 20))
    canvas.paste(im1, (0, (h - im1.height) // 2))
    canvas.paste(im2, (im1.width + 8, (h - im2.height) // 2))
    return canvas


def sample_text(sample):
    msgs = sample.get("messages", [])
    u = next((m["content"] for m in msgs if m["role"] == "user"), "").replace("<image>\n", "").strip()
    a = next((m["content"] for m in msgs if m["role"] == "assistant"), "").strip()
    return u, a


def collect_unique(data, stype: str, n: int):
    """Iterate `data` in order; keep samples of given type whose (img1, img2) pair is new."""
    seen = set()
    out = []
    for s in data:
        if s.get("type") != stype:
            continue
        imgs = s.get("images", [])
        if len(imgs) < 2:
            continue
        key = (imgs[0], imgs[1])
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--num", type=int, default=300, help="Samples per category (unique image pairs).")
    ap.add_argument("--project", type=str, default="crosspoint-378k-viz")
    ap.add_argument("--entity", type=str, default=None)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--max-side", type=int, default=512, help="Downscale images so max(H,W) <= this.")
    args = ap.parse_args()

    print(f"Loading {args.json} ...")
    data = json.loads(args.json.read_text())
    print(f"Loaded {len(data):,} samples.")

    run = wandb.init(project=args.project, entity=args.entity, name=args.run_name,
                     config={"num_per_type": args.num, "types": CROSS_TYPES, "max_side": args.max_side})

    # Also log overall source composition of the cross_* subset for reference.
    source_counts = {"scannet": 0, "scannet++": 0, "unknown": 0}
    type_source_counts = {t: {"scannet": 0, "scannet++": 0, "unknown": 0} for t in CROSS_TYPES}
    for s in data:
        t = s.get("type")
        if t not in type_source_counts:
            continue
        imgs = s.get("images", [])
        if not imgs:
            continue
        src = classify_source(imgs[0])
        source_counts[src] += 1
        type_source_counts[t][src] += 1
    print("Global cross_* source counts:", source_counts)
    for t, c in type_source_counts.items():
        print(f"  {t}: {c}")
    wandb.summary["source_counts_all_cross"] = source_counts
    wandb.summary["source_counts_per_type"] = type_source_counts

    for stype in CROSS_TYPES:
        print(f"\n=== {stype} ===")
        samples = collect_unique(data, stype, args.num)
        print(f"  collected {len(samples)} unique image-pair samples")

        cols = ["idx", "source", "scene_id", "image1", "image2",
                "point1", "point2", "question", "answer", "visualization"]
        table = wandb.Table(columns=cols)

        for i, s in enumerate(samples):
            imgs = s["images"]
            p1 = tuple(s["point1"]) if "point1" in s else None
            p2 = tuple(s["point2"]) if "point2" in s else None
            src = classify_source(imgs[0])
            sid = scene_id(imgs[0])
            q, a = sample_text(s)

            try:
                viz = make_pair_viz(args.root, imgs[0], imgs[1], p1, p2, max_side=args.max_side)
                wimg = wandb.Image(viz, caption=f"[{stype}] {sid}")
            except Exception as e:
                print(f"  [{i}] failed to render {imgs}: {e}")
                wimg = None

            table.add_data(
                i, src, sid,
                Path(imgs[0]).name, Path(imgs[1]).name,
                str(p1) if p1 else "",
                str(p2) if p2 else "",
                q, a, wimg,
            )

            if (i + 1) % 50 == 0:
                print(f"  added {i + 1}/{len(samples)}")

        wandb.log({f"table/{stype}": table})
        print(f"  logged table/{stype} with {len(samples)} rows")

    run.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
