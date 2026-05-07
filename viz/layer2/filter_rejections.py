"""Visualize quality-filter rejections per scene.

Reads per-frame filter verdicts from
``cache/filter/<spec>/<adapter>/<scene>/<frame_id>.json`` (one JSON per
frame with keys ``usable, reason, raw, inference_seconds``, produced
during Phase 2 of ``python -m cli generate``) and renders every
``usable=False`` frame in a grid with the rejection reason as the title
in red.

One PNG per scene, named ``<scene>.png`` under ``--out-dir``.

Usage:
    python -m viz --mode filter_rejections \\
        --scene scene0095_00 --scene scene0132_01 --scene scene0348_00 \\
        --filter-spec qwen3vl-8B \\
        --out-dir outputs/<run>/viz_filter_rejections
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from viz import add_cache_args, add_scene_args, add_scenes_root_arg


def _frame_sort_key(p: Path):
    """Numeric sort when frame_id is an int (ScanNet); fall back to lex."""
    return int(p.stem) if p.stem.isdigit() else p.stem


def collect_rejections(scene: str, filter_spec: str, adapter: str,
                       cache_root: Path) -> list[tuple[str, str]]:
    """Return ``[(frame_id, reason), ...]`` for every ``usable=False`` frame
    in ``cache_root / filter / <spec> / <adapter> / <scene>``."""
    scene_dir = cache_root / "filter" / filter_spec / adapter / scene
    if not scene_dir.is_dir():
        return []
    out: list[tuple[str, str]] = []
    for p in sorted(scene_dir.glob("*.json"), key=_frame_sort_key):
        try:
            verdict = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if not verdict.get("usable", True):
            out.append((p.stem, str(verdict.get("reason", ""))))
    return out


def render_scene(scene: str, scenes_root: Path,
                 items: list[tuple[str, str]],
                 out_path: Path, cols: int = 4) -> None:
    if not items:
        print(f"[{scene}] no rejected frames")
        return
    n = len(items)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.6),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (fid, reason) in enumerate(items):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        img_path = scenes_root / scene / "color" / f"{fid}.jpg"
        if img_path.exists():
            ax.imshow(Image.open(img_path).convert("RGB"))
        else:
            ax.text(0.5, 0.5, f"missing: {img_path.name}",
                    ha="center", va="center", transform=ax.transAxes)
        title = f"frame {fid}\n{reason}" if reason else f"frame {fid}"
        ax.set_title(title, fontsize=9, color="crimson", wrap=True)
    fig.suptitle(f"{scene} — quality-filter rejections ({n} frames)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[{scene}] {n} rejections -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    add_scene_args(p, repeatable=True)
    add_cache_args(p, include_model_tag=False)
    add_scenes_root_arg(p)
    p.add_argument("--filter-spec", default="qwen3vl-8B",
                   help="registry name of the filter model whose cache "
                        "to read (<cache_root>/filter/<spec>/...)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cols", type=int, default=4)
    args = p.parse_args()

    summary: dict[str, int] = {}
    for scene in args.scene:
        rejs = collect_rejections(scene, args.filter_spec, args.adapter,
                                  args.cache_root)
        summary[scene] = len(rejs)
        out_path = args.out_dir / f"{scene}.png"
        render_scene(scene, args.scenes_root, rejs, out_path, cols=args.cols)
    total = sum(summary.values())
    print(f"\ntotal: {total} rejections across {len(summary)} scenes")
    for s, n in summary.items():
        print(f"  {s}: {n}")


if __name__ == "__main__":
    main()
