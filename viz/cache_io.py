"""Perception cache discovery + per-frame mask loading.

Cache layout (post-refactor, no hashes):
    cache/perception/<adapter>/<scene>/<model_tag>/<frame_id>.pkl
where <model_tag> = "<detector>+<segmenter>" (human-readable).

Discovery: callers either pass `model_tag="<detector>+<segmenter>"`
explicitly, or rely on the default `strategy="mtime"` (= "the latest
run" — safe now that model-tagged dirs are uniquely identifiable and
we won't pick up a stray GT-only run instead of the SAM run we wanted).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, Path]


def discover_cfg_dir(cache_root: PathLike, adapter: str, scene: str,
                     *, model_tag: Optional[str] = None,
                     strategy: str = "mtime") -> Optional[Path]:
    """Find the perception-cache model-tagged subdir for a scene.

    Pass `model_tag="<detector>+<segmenter>"` to point at a specific
    run. Otherwise:
      * `strategy="mtime"` (default): pick the most-recently-modified
        subdir — i.e. "the latest run on this scene".
      * `strategy="most_frames"`: pick the dir with the most cached
        `.pkl` files. Kept for back-compat; not recommended now that
        each run writes a uniquely-named subdir.

    Returns None if the requested dir / any dir doesn't exist.
    """
    scene_dir = Path(cache_root) / adapter / scene
    if model_tag is not None:
        d = scene_dir / model_tag
        return d if d.is_dir() else None
    if not scene_dir.exists():
        return None
    cfg_dirs = [d for d in scene_dir.iterdir() if d.is_dir()]
    if not cfg_dirs:
        return None
    if strategy == "most_frames":
        return max(cfg_dirs, key=lambda d: len(list(d.glob("*.pkl"))))
    if strategy == "mtime":
        return max(cfg_dirs, key=lambda d: d.stat().st_mtime)
    raise ValueError(f"Unknown strategy {strategy!r}")


def load_frame_masks(cache_root: PathLike, adapter: str, scene: str, fid,
                     *, model_tag: Optional[str] = None,
                     strategy: str = "mtime",
                     cfg_dir: Optional[Path] = None) -> Optional[list]:
    """Return the list of masks for one frame (`mask_id` = list index).

    Pass `model_tag="<detector>+<segmenter>"` to bind to a specific run.
    Otherwise discovery uses `strategy` (default `"mtime"`). Returns
    None on missing cache.
    """
    if cfg_dir is None:
        cfg_dir = discover_cfg_dir(cache_root, adapter, scene,
                                   model_tag=model_tag, strategy=strategy)
        if cfg_dir is None:
            return None
    pkl = Path(cfg_dir) / f"{fid}.pkl"
    if not pkl.exists():
        return None
    try:
        with open(pkl, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        return None
