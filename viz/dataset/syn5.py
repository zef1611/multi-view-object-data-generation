"""Visualize samples from the infinigen syn_5_types dataset.

Usage:
    python visualize_syn5.py                              # random sample
    python visualize_syn5.py --type rotation_proximity_yesno --num 2
    python visualize_syn5.py --sample-id anchor_000000 --save out.png
    python visualize_syn5.py --jsonl cropond_correspondences.jsonl --num 2
        # render scenes from the CroPond output with predicted points overlaid

Mirrors the style of visualize_crosspoint.py: matplotlib Circle overlays,
pathlib paths, argparse CLI.
"""

import argparse
import json
import random
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image

DEFAULT_ROOT = Path(
    "/network/scratch/q/qian.yang/infinigen/training_data_mix_all_rotation/"
    "training_data_mix_all_balance/syn_5_types"
)
DEFAULT_JSON = DEFAULT_ROOT / "training_data.json"

QUESTION_TYPES = [
    "anchor", "counting", "spatial_orientation", "perspective_taking",
    "closest", "farthest",
    "closest_to_camera_1", "closest_to_camera_2",
    "farthest_from_camera_1", "farthest_from_camera_2",
    "rotation_direction_mcq", "rotation_proximity_yesno",
]

ROT_VIEW_KEYS = ("image_1", "image_2", "image_3", "image_4")
ROT_VIEW_LABELS = ("front", "right", "back", "left")
CAM_VIEW_KEYS = ("image_cam0", "image_cam1")
CAM_VIEW_LABELS = ("cam0", "cam1")

VIZ_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff",
    "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1",
]


def sample_views(sample):
    """Return list[(label, path_relative_to_root)] for the sample."""
    if all(k in sample for k in ROT_VIEW_KEYS):
        return [(lbl, sample[k]) for lbl, k in zip(ROT_VIEW_LABELS, ROT_VIEW_KEYS)]
    if all(k in sample for k in CAM_VIEW_KEYS):
        return [(lbl, sample[k]) for lbl, k in zip(CAM_VIEW_LABELS, CAM_VIEW_KEYS)]
    return []


def draw_sample(sample, root, axes, predictions=None):
    """Render one sample across matplotlib axes, optionally overlaying predicted points.

    predictions: dict[view_label -> list[(object, [x, y])]] or None.
    """
    views = sample_views(sample)
    stype = sample.get("question_type", "?")
    instr = sample.get("instruction", "") or ""
    output = sample.get("output", "") or ""

    for ax, (label, rel) in zip(axes, views):
        ax.set_xticks([]); ax.set_yticks([])
        img_path = root / rel
        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"[missing]\n{img_path.name}\n{e}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
            continue

        radius = max(img.size) * 0.012
        if predictions:
            for i, (obj, pt) in enumerate(predictions.get(label, [])):
                color = VIZ_COLORS[i % len(VIZ_COLORS)]
                ax.add_patch(Circle(pt, radius=radius, facecolor=color,
                                    edgecolor="white", linewidth=1.5, alpha=0.9))
                ax.text(pt[0] + radius * 1.2, pt[1] - radius * 0.2, obj,
                        color=color, fontsize=8,
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
        ax.set_title(f"{label}: {Path(rel).name}", fontsize=8)

    # unused axes (if predictions-only from a JSONL with fewer panels than axes)
    for extra in axes[len(views):]:
        extra.axis("off")

    caption = f"[{stype}] id={sample.get('sample_id','?')}  scene={sample.get('scene_id','?')}"
    if instr:
        caption += f"\nQ: {instr.strip()}"
    if output:
        caption += f"\nA: {output.strip()}"
    axes[0].figure.text(
        0.5, 0.02,
        "\n".join(textwrap.wrap(caption, width=140)),
        ha="center", va="bottom", fontsize=8, family="monospace",
    )


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def pick_raw_samples(data, args):
    if args.sample_id:
        hits = [d for d in data if d.get("sample_id") == args.sample_id]
        if not hits:
            raise SystemExit(f"No sample with sample_id={args.sample_id!r}")
        return hits
    if args.scene_id:
        hits = [d for d in data if d.get("scene_id") == args.scene_id]
        if not hits:
            raise SystemExit(f"No sample with scene_id={args.scene_id!r}")
        return hits[: args.num]
    if args.index is not None:
        return [data[args.index]]
    pool = [d for d in data if args.type is None or d.get("question_type") == args.type]
    if not pool:
        raise SystemExit(f"No samples of type {args.type!r}")
    rng = random.Random(args.seed)
    return rng.sample(pool, k=min(args.num, len(pool)))


def _jsonl_record_to_sample(rec):
    """Make a JSONL CroPond record look like a training_data sample for draw_sample."""
    images = rec.get("images", {})
    images_abs = {k: Path(v) for k, v in images.items()}
    # create a sample-shaped dict the view keys of which match the JSONL record
    # We'll hand-build predictions as dict[label -> [(obj, pt), ...]]
    preds = {}
    for c in rec.get("correspondences", []):
        for vk, pt in c.get("points", {}).items():
            if pt is None:
                continue
            preds.setdefault(vk, []).append((c["object"], pt))

    class _Sample(dict):
        pass

    sample = _Sample({
        "sample_id": rec.get("scene_id", "?"),
        "scene_id": rec.get("scene_id", "?"),
        "question_type": rec.get("sample_pattern", "?"),
        "instruction": f"{len(rec.get('enumerated_objects', []))} objects enumerated: "
                       + ", ".join(rec.get("enumerated_objects", [])),
        "output": "",
    })
    # attach the view paths so sample_views returns them in the right order
    view_keys = list(images_abs.keys())
    if view_keys == ["view_a", "view_b"]:
        sample["image_cam0"] = str(images_abs["view_a"])
        sample["image_cam1"] = str(images_abs["view_b"])
    elif view_keys == list(ROT_VIEW_LABELS):
        for lbl, key in zip(ROT_VIEW_LABELS, ROT_VIEW_KEYS):
            sample[key] = str(images_abs[lbl])
    else:
        # unknown ordering; fall back to cam-style with first two
        sample["image_cam0"] = str(images_abs[view_keys[0]])
        if len(view_keys) > 1:
            sample["image_cam1"] = str(images_abs[view_keys[1]])
    return sample, preds


def _map_labels(preds, views):
    """Map prediction keys to the labels draw_sample will use."""
    remap = {"view_a": "cam0", "view_b": "cam1"}
    out = {}
    for vk, lst in preds.items():
        lbl = remap.get(vk, vk)
        out[lbl] = lst
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON,
                    help="training_data.json path")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="dataset root (paths in JSON are relative to this)")
    ap.add_argument("--jsonl", type=Path, default=None,
                    help="CroPond output JSONL; when set, visualize predicted points")
    ap.add_argument("--type", choices=QUESTION_TYPES, default=None)
    ap.add_argument("--sample-id", default=None)
    ap.add_argument("--scene-id", default=None)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--num", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save", type=Path, default=None,
                    help="save to file instead of showing (suffix _NN added for --num>1)")
    args = ap.parse_args()

    if args.jsonl is not None:
        records = [json.loads(l) for l in args.jsonl.read_text().splitlines() if l.strip()]
        print(f"Loaded {len(records)} scenes from {args.jsonl}")
        if args.scene_id:
            records = [r for r in records if r.get("scene_id") == args.scene_id]
        else:
            rng = random.Random(args.seed)
            rng.shuffle(records)
            records = records[: args.num]
    else:
        print(f"Loading {args.json} ...")
        data = json.loads(args.json.read_text())
        print(f"Loaded {len(data):,} samples.")
        samples = pick_raw_samples(data, args)
        records = [(s, None) for s in samples]
        # normalize to (sample, preds) tuples
        records = [{"_sample": s, "_preds": None} for s in samples]

    for k, rec in enumerate(records):
        if args.jsonl is not None:
            sample, preds_raw = _jsonl_record_to_sample(rec)
            preds = _map_labels(preds_raw, sample_views(sample))
        else:
            sample = rec["_sample"]
            preds = None

        views = sample_views(sample)
        n = max(1, len(views))
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
        if n == 1:
            axes = [axes]
        draw_sample(sample, args.root, axes, predictions=preds)
        fig.tight_layout(rect=(0, 0.12, 1, 1))

        if args.save:
            out = args.save if len(records) == 1 else args.save.with_stem(f"{args.save.stem}_{k:02d}")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  saved {out}")
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
