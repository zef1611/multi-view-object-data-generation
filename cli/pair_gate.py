"""Stand-alone runner for the pair-gate stage (Phase 2).

Reads any frame-bearing artifact (``frames.json`` or any pair file via
``cli/_io.py::load_inputs``), groups frames by ``(adapter, scene_id)``,
runs ``pipeline.stages.stage_pair_gate`` per scene, and writes the
surviving scored pairs to ``pairs.scored.jsonl``.

This stage is **CPU-only** — pure pose/frustum gate, no vLLM server,
no quality filter. The filter runs *after* pair-gate (Phase 3) and
post-culls pairs whose endpoint frames are unusable.

Idempotent: re-runs are byte-deterministic when the inputs and pair
config are unchanged. ``--append`` opts into appending to an existing
``pairs.scored.jsonl`` (e.g. accumulating over multiple invocations
across different scenes).
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from cli._io import load_inputs
from models._frame_ref import FrameRef
from pipeline.config import (
    load_stage_config, merge_cli_with_config, stage_config_path,
)
from pipeline.pairs_io import view_pairs_to_scored, write_scored_pairs

logger = logging.getLogger("pair_gate")


_PAIR_GATE_KEYS = (
    "sampling", "frame_stride", "min_keyframes",
    "min_translation_m", "min_rotation_deg", "limit_frames",
    "cosmic_base_sampling", "cosmic_union_coverage_min",
    "cosmic_yaw_diff_min_deg", "cosmic_obj_vis_area_min",
    "cosmic_obj_vis_depth_pix_min",
)


def _group_frames(frames: list[FrameRef]) -> dict[tuple[str, str],
                                                  list[FrameRef]]:
    out: dict[tuple[str, str], list[FrameRef]] = {}
    for f in frames:
        out.setdefault((f.adapter, f.scene_id), []).append(f)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path,
                   default=stage_config_path("pair_gate"),
                   help="per-stage config JSON (default: "
                        "configs/stages/pair_gate.json). CLI flags override.")
    p.add_argument("--in", dest="in_path", type=Path, required=True,
                   help="frames.json (cli sample) or any frame-bearing "
                        "artifact (cli/_io.py::load_inputs)")
    p.add_argument("--out", type=Path, required=True,
                   help="output pairs.scored.jsonl (overwrites by default; "
                        "see --append)")
    p.add_argument("--append", action="store_true",
                   help="append to --out instead of overwriting "
                        "(useful for multi-invocation accumulation)")
    p.add_argument("--scenes-root", type=Path,
                   default=Path("/home/mila/l/leh/scratch/dataset/scannet_data/scans"),
                   help="needed to instantiate adapters from frame metadata")
    # ── sampler knobs (defaults from configs/stages/pair_gate.json) ---
    p.add_argument("--sampling", choices=["adaptive", "stride", "cosmic"],
                   default=None)
    p.add_argument("--frame-stride", type=int, default=None)
    p.add_argument("--min-keyframes", type=int, default=None)
    p.add_argument("--min-translation-m", type=float, default=None)
    p.add_argument("--min-rotation-deg", type=float, default=None)
    p.add_argument("--limit-frames", type=int, default=None)
    p.add_argument("--cosmic-base-sampling", choices=["adaptive", "stride"],
                   default=None)
    p.add_argument("--cosmic-union-coverage-min", type=float, default=None)
    p.add_argument("--cosmic-yaw-diff-min-deg", type=float, default=None)
    p.add_argument("--cosmic-obj-vis-area-min", type=float, default=None)
    p.add_argument("--cosmic-obj-vis-depth-pix-min", type=int, default=None)
    # ── logging --------------------------------------------------------
    p.add_argument("--logs-dir", type=Path, default=Path("logs"))
    p.add_argument("--run-id", type=str, default=None,
                   help="defaults to pair_gate__<timestamp>")
    args = p.parse_args()

    cfg = load_stage_config("pair_gate", args.config)
    merge_cli_with_config(args, cfg, _PAIR_GATE_KEYS)

    if args.run_id is None:
        args.run_id = f"pair_gate__{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = (args.logs_dir / args.run_id).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_format = "%(asctime)s %(name)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    fh = logging.FileHandler(log_dir / "pipeline.log")
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logger.info("run_id=%s  logs at %s", args.run_id, log_dir)

    bundle = load_inputs(args.in_path)
    if not bundle.frames:
        raise SystemExit(f"--in {args.in_path}: no frames resolved.")

    from pipeline.stages import stage_pair_gate
    from pipeline.config import load_skills_config, resolve as resolve_config

    # Lazy import to avoid pulling in the dataset adapters until needed.
    from cli.generate import make_adapter

    task_config = load_skills_config()

    grouped = _group_frames(bundle.frames)
    logger.info("pair_gate: %d frames across %d (adapter, scene) groups",
                len(bundle.frames), len(grouped))

    all_scored = []
    n_scenes_with_pairs = 0
    for (adapter_name, scene_id), frames in sorted(grouped.items()):
        try:
            adapter = make_adapter(adapter_name,
                                   args.scenes_root / scene_id)
        except FileNotFoundError as e:
            logger.warning("skip (%s, %s): %s", adapter_name, scene_id, e)
            continue
        pair_cfg = resolve_config(task_config,
                                  getattr(adapter, "source_name", "unknown"))
        pairs, _ffp = stage_pair_gate(
            adapter,
            adapter_name=adapter_name,
            pair_config=pair_cfg,
            sampling=args.sampling,
            frame_stride=args.frame_stride,
            min_keyframes=args.min_keyframes,
            min_translation_m=args.min_translation_m,
            min_rotation_deg=args.min_rotation_deg,
            limit_frames=args.limit_frames,
            cosmic_base_sampling=args.cosmic_base_sampling,
            cosmic_union_coverage_min=args.cosmic_union_coverage_min,
            cosmic_yaw_diff_min_deg=args.cosmic_yaw_diff_min_deg,
            cosmic_obj_vis_area_min=args.cosmic_obj_vis_area_min,
            cosmic_obj_vis_depth_pix_min=args.cosmic_obj_vis_depth_pix_min,
            quality_filter=None,
        )
        if not pairs:
            logger.info("[%s] no surviving pairs", scene_id)
            continue
        n_scenes_with_pairs += 1
        # Build image_path lookup for this scene, then convert.
        image_path_for = {f.frame_id: str(f.image_path) for f in frames}
        # Frames referenced by pairs may not all be in `frames` (the input
        # could be a subset like frames-in-prior-pairs); resolve any missing
        # via the adapter so ScoredPair always has full image paths.
        for pair in pairs:
            for fid in (pair.src_id, pair.tgt_id):
                if fid not in image_path_for:
                    image_path_for[fid] = str(
                        adapter.frame_ref(fid, adapter_name).image_path
                    )
        scored = view_pairs_to_scored(
            pairs, adapter=adapter_name, scene_id=scene_id,
            image_path_for=image_path_for,
        )
        all_scored.extend(scored)

    if not all_scored:
        logger.warning("pair_gate: 0 surviving pairs across all scenes")
    n = write_scored_pairs(all_scored, args.out, append=args.append)
    logger.info("pair_gate: wrote %d pairs (%d scenes) → %s%s",
                n, n_scenes_with_pairs, args.out,
                " (appended)" if args.append else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
