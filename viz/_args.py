"""Shared CLI argument helpers for viz modes.

Every `viz/layer2/*.py` and `viz/dataset/*.py` module's ``main()``
calls these helpers so the flag spelling, default, and help text are
identical across modes. Lets users learn one set of flag names.

Conventions enforced here:
- ``--scene`` always means ScanNet scene id (or equivalent for the
  adapter).
- ``--cache-root`` is always the *parent* of ``perception/``,
  ``filter/``, ``labels/``, ``verifier/``. Modes append the namespace
  themselves (e.g. ``cache_root / "perception" / adapter / scene / ...``).
- ``--scenes-root`` defaults to :data:`viz.DEFAULT_SCENES_ROOT`.
- ``--adapter`` defaults to ``scannet``.
- ``--model-tag`` is optional; when omitted, modes resolve the most
  recent model-tagged subdir via :func:`viz.discover_cfg_dir`.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def add_scene_args(p: argparse.ArgumentParser, *,
                   repeatable: bool = False,
                   required: bool = True) -> None:
    """Register ``--scene``. Repeatable for modes that aggregate per-scene
    grids (e.g. filter_rejections, compare_sampling); singular for modes
    that render a single scene at a time."""
    if repeatable:
        p.add_argument("--scene", action="append", required=required,
                       help="scene id (repeatable)")
    else:
        p.add_argument("--scene", required=required,
                       help="scene id")


def add_cache_args(p: argparse.ArgumentParser, *,
                   include_model_tag: bool = True) -> None:
    """Register ``--cache-root``, ``--adapter``, optionally ``--model-tag``.

    ``--cache-root`` is the parent of the per-namespace caches. The
    perception cache lives at ``<cache_root>/perception/<adapter>/...``
    and the filter cache at ``<cache_root>/filter/<spec>/<adapter>/...``.
    """
    p.add_argument("--cache-root", type=Path, default=Path("cache"),
                   help="parent of perception/, filter/, labels/, "
                        "verifier/ (default: cache)")
    p.add_argument("--adapter", default="scannet",
                   help="dataset adapter name (default: scannet)")
    if include_model_tag:
        p.add_argument("--model-tag", type=str, default=None,
                       help="perception cache subdir under "
                            "<cache_root>/perception/<adapter>/<scene>/, "
                            "e.g. 'labeled-gdino+sam2.1'. Default: most "
                            "recently-modified subdir for this scene.")


def add_scenes_root_arg(p: argparse.ArgumentParser) -> None:
    """Register ``--scenes-root`` with the project default."""
    from viz import DEFAULT_SCENES_ROOT
    p.add_argument("--scenes-root", type=Path, default=DEFAULT_SCENES_ROOT,
                   help=f"raw dataset root (default: {DEFAULT_SCENES_ROOT})")
