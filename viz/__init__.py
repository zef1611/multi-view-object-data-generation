"""Shared visualization primitives.

Single source of truth for color palette, mask outline rendering,
perception-cache discovery, and CLI argument shapes — imported by every
``viz/layer2/*`` and ``viz/dataset/*`` module via ``from viz import ...``.
"""

from pathlib import Path

from .palette import PALETTE, color_for
from .overlays import (
    draw_bbox,
    draw_mask_outline,
    draw_src_point,
    draw_tgt_point,
    mask_centroid,
)
from .cache_io import discover_cfg_dir, load_frame_masks
from ._args import add_scene_args, add_cache_args, add_scenes_root_arg

# Single source of truth for the raw ScanNet dataset root used by viz
# modes that read original images / poses (perception, gt, pairs,
# filter_rejections, ...). Override per-invocation with ``--scenes-root``.
DEFAULT_SCENES_ROOT = Path(
    "/home/mila/l/leh/scratch/dataset/scannet_data/scans"
)

__all__ = [
    "PALETTE",
    "color_for",
    "draw_bbox",
    "draw_mask_outline",
    "draw_src_point",
    "draw_tgt_point",
    "mask_centroid",
    "discover_cfg_dir",
    "load_frame_masks",
    "add_scene_args",
    "add_cache_args",
    "add_scenes_root_arg",
    "DEFAULT_SCENES_ROOT",
]
