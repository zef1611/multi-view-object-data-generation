"""Stand-alone runner for the perception stage (Phase 4.5).

Reads any frame-bearing or pair-bearing artifact, derives the unique
frame list (paper-faithful when fed a pairs file — only frames-in-pairs
are processed), and runs the multi-GPU GDino+SAM pre-pass to populate
``cache/perception/<adapter>/<scene>/<detector>+<segmenter>/<frame>.pkl``.

This stage owns **only the perception model**. It deliberately does
**not** launch the labeler or quality-filter server — those caches must
already exist when ``--detector labeled-gdino`` is used, and the script
will error loudly with the exact upstream ``cli label`` command to run
otherwise.

Auto-skip rules (logged):
  * ``--workers 0``
  * CPU-only detector + segmenter combo (``scannet-gt + gt-mask`` etc.)
  * Cache fully populated (no pending frames after stat)
  * Pending frames < ``--prepass-min-frames``

Idempotent: re-runs short-circuit on the cache stat.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from cli._frames_io import write_frames
from cli._io import load_inputs
from pipeline.config import (
    load_stage_config, merge_cli_with_config, stage_config_path,
)

logger = logging.getLogger("perceive")


_PERCEIVE_KEYS = (
    "detector", "segmenter", "gdino_max_classes",
    "labeler", "prompt_file", "cache_root",
    "workers", "batch_frames", "prepass_min_frames",
    "compile_perception",
    "n_votes", "vote_temperature", "vote_strategy",
)


def _resolve_workers(arg: Optional[int]) -> int:
    if arg is not None:
        return max(0, int(arg))
    try:
        import torch
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path,
                   default=stage_config_path("perceive"),
                   help="per-stage config JSON (default: "
                        "configs/stages/perceive.json). CLI flags override.")
    p.add_argument("--in", dest="in_path", type=Path, required=True,
                   help="frames.json | pairs.scored.jsonl | pairs.jsonl "
                        "(see cli/_io.py::load_inputs)")
    p.add_argument("--out", type=Path, default=None,
                   help="optional passthrough frames.json (the unique frame "
                        "list this stage processed). For chaining with other "
                        "stage CLIs.")
    # ── perception models --------------------------------------------
    p.add_argument("--detector", default=None,
                   choices=["noop", "gdino", "gdino+scannet200",
                            "labeled-gdino", "gemini+gdino",
                            "scannet-gt", "scannet-gt-label+gdino"])
    p.add_argument("--segmenter", default=None,
                   choices=["noop", "sam2.1", "sam3", "gt-mask"])
    p.add_argument("--gdino-max-classes", type=int, default=None)
    from models.registry import MODELS
    p.add_argument("--labeler", default=None,
                   choices=sorted(MODELS),
                   help="Required when --detector=labeled-gdino. The labeler "
                        "cache must already be populated by `cli label`; this "
                        "script will not launch the labeler server.")
    p.add_argument("--prompt-file", dest="prompt_file", type=Path, default=None)
    # ── cache / paths ------------------------------------------------
    p.add_argument("--cache-root", type=Path, default=None)
    p.add_argument("--scenes-root", type=Path,
                   default=Path("/home/mila/l/leh/scratch/dataset/scannet_data/scans"))
    # ── multi-GPU knobs (config keys: workers, batch_frames, ...) ----
    p.add_argument("--workers", type=int, default=None,
                   help="Worker pool size. Default = torch.cuda.device_count(). "
                        "0 disables (serial cache.get fallback inside Phase 5).")
    p.add_argument("--batch-frames", type=int, default=None,
                   help="Frames per batched GDino+SAM call inside each worker.")
    p.add_argument("--prepass-min-frames", type=int, default=None,
                   help="Skip the pre-pass when fewer frames are pending. "
                        "Worker startup is ~30-60s — not worth it for tiny runs.")
    p.add_argument("--compile-perception", dest="compile_perception",
                   action=argparse.BooleanOptionalAction, default=None)
    # ── multi-vote labeler knobs (must match values used at cli label) ─
    p.add_argument("--label-votes", dest="n_votes", type=int, default=None,
                   help="Multi-vote labeler N (default 1). Must match the "
                        "value used at the label stage so the cache_tag "
                        "(cache/labels/<spec>__voteN/) lines up.")
    p.add_argument("--label-vote-temperature", dest="vote_temperature",
                   type=float, default=None)
    p.add_argument("--label-vote-strategy", dest="vote_strategy",
                   default=None,
                   choices=["union", "majority", "per-run-detect"],
                   help="How `--detector labeled-gdino` aggregates the N "
                        "labeler runs (default union).")
    # ── logging ------------------------------------------------------
    p.add_argument("--logs-dir", type=Path, default=Path("logs"))
    p.add_argument("--run-id", type=str, default=None,
                   help="defaults to perceive__<timestamp>")
    args = p.parse_args()

    cfg = load_stage_config("perceive", args.config)
    merge_cli_with_config(args, cfg, _PERCEIVE_KEYS)
    # JSON strings → Path for path-typed knobs.
    if isinstance(args.cache_root, str):
        args.cache_root = Path(args.cache_root)
    if isinstance(args.prompt_file, str):
        args.prompt_file = Path(args.prompt_file)

    if args.run_id is None:
        args.run_id = f"perceive__{datetime.now().strftime('%Y%m%d-%H%M%S')}"
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
    logger.info("perceive: %d unique frames from %s (kind=%s)",
                len(bundle.frames), args.in_path, bundle.source_kind)

    needs_labeler = args.detector in ("labeled-gdino", "gemini+gdino")
    if needs_labeler:
        from models.registry import resolve as resolve_model
        from pipeline.stages import labeler_cache_complete
        labeler_spec = resolve_model(args.labeler)
        if not labeler_cache_complete(
                labeler_spec, bundle.frames,
                prompt_file=args.prompt_file,
                n_votes=int(args.n_votes or 1),
                vote_temperature=float(args.vote_temperature or 0.7)):
            raise SystemExit(
                f"--detector {args.detector!r} requires labeler cache "
                f"({labeler_spec.name!r}, n_votes={args.n_votes or 1}) "
                f"populated for every frame. Run `python -m cli label "
                f"--in {args.in_path} --labeler {labeler_spec.name}"
                f"{' --label-votes ' + str(args.n_votes) if args.n_votes else ''}` "
                f"first."
            )
        labeler_spec_name = labeler_spec.name
    else:
        labeler_spec_name = None

    # Group frames by scene for the per-scene pre-pass map.
    scene_to_frames: dict[str, list] = {}
    for f in bundle.frames:
        scene_to_frames.setdefault(f.scene_id, []).append(f)

    n_workers = _resolve_workers(args.workers)
    try:
        import torch
        gpu_count = torch.cuda.device_count()
    except Exception:
        gpu_count = 0
    if n_workers > 0 and gpu_count > 0:
        n_workers = min(n_workers, gpu_count)

    # All frames must share one adapter for the pre-pass (worker config
    # carries a single adapter_name). Refuse mixed-adapter inputs.
    adapters_in_input = sorted({f.adapter for f in bundle.frames})
    if len(adapters_in_input) > 1:
        raise SystemExit(
            f"--in {args.in_path}: multi-adapter inputs are not supported "
            f"by `cli perceive` (got {adapters_in_input}). Split the input "
            f"per adapter and run perceive once per group."
        )
    adapter_name = adapters_in_input[0]

    from pipeline.stages import stage_perceive

    written = stage_perceive(
        adapter_name=adapter_name,
        scenes_root=args.scenes_root,
        detector_name=args.detector,
        segmenter_name=args.segmenter,
        labeler_spec_name=labeler_spec_name,
        prompt_file=args.prompt_file,
        gdino_max_classes=args.gdino_max_classes,
        cache_root=args.cache_root,
        model_tag=f"{args.detector}+{args.segmenter}",
        compile_perception=bool(args.compile_perception),
        perception_batch_frames=int(args.batch_frames),
        scene_to_frames=scene_to_frames,
        num_workers=n_workers,
        prepass_min_frames=args.prepass_min_frames,
        gpu_ids=list(range(n_workers)) if n_workers > 0 else None,
        log_dir=log_dir,
        n_votes=int(args.n_votes or 1),
        vote_temperature=float(args.vote_temperature or 0.7),
        vote_strategy=str(args.vote_strategy or "union"),
    )
    logger.info("perceive: %d frames written to perception cache", written)

    if args.out is not None:
        write_frames(bundle.frames, args.out)
        logger.info("perceive: passthrough → %s (%d frames)",
                    args.out, len(bundle.frames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
