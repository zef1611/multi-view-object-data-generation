"""Tile every kept pair side-by-side for a sampling-strategy comparison.

For one scene, reads `<root>/<scene>/stage_1/<skill>/pairs.jsonl` from
two output roots (e.g. adaptive vs stride), dedupes pairs across skills,
and emits a single PNG per scene with two columns:

    [ adaptive root ]  |  [ stride root ]
       src tgt skills  |     src tgt skills
       ...             |     ...

Each row is one unique (frame_src, frame_tgt) pair; the skill list under
each pair shows which gates it survived.

Usage:
    python -m viz --mode compare_sampling \
        --root outputs/dryrun_3rand_full:adaptive \
        --root outputs/dryrun_3rand_stride:stride \
        --scenes scene0306_00 scene0052_02 scene0012_00 \
        --out-dir outputs/sampling_compare
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _collect_pairs(root: Path, scene: str,
                   only_skill: str | None = None) -> dict:
    """Return {(fsrc, ftgt): {"img_src": p, "img_tgt": p, "skills": [...],
                              "overlap": ..., "angle": ..., "trans": ...}}.

    If `only_skill` is set, only that skill's pairs.jsonl is read.
    """
    stage = root / scene / "stage_1"
    if not stage.exists():
        return {}
    pairs: dict = {}
    for skill_dir in sorted(stage.iterdir()):
        if not skill_dir.is_dir() or skill_dir.name.startswith("_"):
            continue
        if only_skill is not None and skill_dir.name != only_skill:
            continue
        jsonl = skill_dir / "pairs.jsonl"
        if not jsonl.exists():
            continue
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            m = json.loads(line)
            key = (m["frame_src"], m["frame_tgt"])
            entry = pairs.setdefault(key, {
                "img_src": m["image_src"],
                "img_tgt": m["image_tgt"],
                "skills": [],
                "overlap": m["pair_overlap"],
                "angle": m["pair_angle_deg"],
                "trans": m["pair_distance_m"],
            })
            entry["skills"].append(skill_dir.name)
    return pairs


def _thumb(path: str, max_h: int = 160) -> np.ndarray:
    img = Image.open(path)
    w, h = img.size
    new_w = int(round(w * max_h / h))
    return np.array(img.resize((new_w, max_h), Image.BILINEAR))


_SKILL_SHORT = {
    "cross_point_correspondence": "cpc",
    "cross_object_correspondence": "coc",
    "anchor": "anc",
    "counting": "cnt",
    "relative_distance": "rdi",
    "relative_direction": "rdr",
    "cross_spatial_transformation": "cst",
    "cross_depth_variation": "cdv",
    "cross_occlusion_visibility": "cov",
}


def _short(skills: list[str]) -> str:
    return ",".join(_SKILL_SHORT.get(s, s[:3]) for s in sorted(set(skills)))


def _render_column(ax, pairs: dict, label: str, max_h: int):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{label}  ({len(pairs)} unique pairs)", fontsize=12)
    if not pairs:
        ax.text(0.5, 0.5, "(no pairs)", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
        return
    # Build a tall canvas: one row per pair, each row = [src | tgt].
    rows = []
    captions = []
    for (fs, ft), info in sorted(pairs.items(),
                                  key=lambda kv: (int(kv[0][0]), int(kv[0][1]))):
        try:
            s = _thumb(info["img_src"], max_h=max_h)
            t = _thumb(info["img_tgt"], max_h=max_h)
        except Exception:
            continue
        # Match heights, then concat horizontally with a small white gap.
        gap = np.full((max_h, 6, 3), 255, dtype=np.uint8)
        if s.ndim == 2:
            s = np.stack([s] * 3, axis=-1)
        if t.ndim == 2:
            t = np.stack([t] * 3, axis=-1)
        s = s[..., :3]; t = t[..., :3]
        # Pad to common width (max(src_w, tgt_w))
        max_w = max(s.shape[1], t.shape[1])
        def _padw(im, w):
            if im.shape[1] == w:
                return im
            pad = np.full((max_h, w - im.shape[1], 3), 255, dtype=np.uint8)
            return np.concatenate([im, pad], axis=1)
        row = np.concatenate([_padw(s, max_w), gap, _padw(t, max_w)], axis=1)
        rows.append(row)
        captions.append(
            f"{fs}->{ft}  ov={info['overlap']:.2f} "
            f"r={info['angle']:.0f}° t={info['trans']:.2f}m  "
            f"[{_short(info['skills'])}]")

    # Pad all rows to the same width.
    max_row_w = max(r.shape[1] for r in rows)
    rows = [
        np.concatenate(
            [r, np.full((max_h, max_row_w - r.shape[1], 3), 255, dtype=np.uint8)],
            axis=1,
        ) if r.shape[1] < max_row_w else r
        for r in rows
    ]
    sep = np.full((4, max_row_w, 3), 220, dtype=np.uint8)
    canvas_blocks = []
    for r in rows:
        canvas_blocks.append(r)
        canvas_blocks.append(sep)
    canvas = np.concatenate(canvas_blocks, axis=0)
    ax.imshow(canvas)
    # Annotate each row with its caption on the right.
    row_h = max_h + 4
    for i, cap in enumerate(captions):
        y = i * row_h + max_h / 2
        ax.annotate(
            cap, xy=(max_row_w + 8, y), xycoords="data",
            ha="left", va="center", fontsize=7, family="monospace",
            annotation_clip=False,
        )
    # Pad the right side of the axis so captions are visible.
    ax.set_xlim(-2, max_row_w + 380)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", action="append", required=True,
                   help="repeat: --root path[:label]; label defaults to dirname")
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-h", type=int, default=140,
                   help="thumbnail height in pixels")
    p.add_argument("--skill", default=None,
                   help="restrict to a single skill (e.g. "
                        "cross_point_correspondence); default: union "
                        "across all skills")
    args = p.parse_args()

    cols: list[tuple[Path, str]] = []
    for spec in args.root:
        if ":" in spec:
            path_s, label = spec.split(":", 1)
        else:
            path_s, label = spec, Path(spec).name
        cols.append((Path(path_s), label))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for scene in args.scenes:
        per_col = [(_collect_pairs(r, scene, args.skill), lab)
                   for r, lab in cols]
        n_rows = max((len(d) for d, _ in per_col), default=1)
        fig_h = max(4.0, 0.9 * (args.max_h / 100.0) * n_rows + 1.5)
        fig, axes = plt.subplots(
            1, len(cols), figsize=(11 * len(cols), fig_h),
            gridspec_kw=dict(wspace=0.08, left=0.02, right=0.98))
        if len(cols) == 1:
            axes = [axes]
        for ax, (pairs, lab) in zip(axes, per_col):
            _render_column(ax, pairs, lab, max_h=args.max_h)
        scope = (f"skill='{args.skill}'" if args.skill
                 else "union over all 9 skills")
        fig.suptitle(
            f"{scene}  —  " + " vs ".join(lab for _, lab in cols)
            + f"  ({scope})",
            fontsize=13)
        suffix = f"_{args.skill}" if args.skill else ""
        out = args.out_dir / f"{scene}{suffix}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        counts = "  ".join(f"{lab}={len(d)}" for d, lab in per_col)
        print(f"-> {out}  ({counts})")


if __name__ == "__main__":
    main()
