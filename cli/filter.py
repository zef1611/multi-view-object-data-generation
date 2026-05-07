"""Stand-alone runner for the quality-filter stage.

Reads any frame-bearing artifact (``frames.json`` from ``cli sample``,
or any pair file via ``cli/_io.py::load_inputs``) and populates
``cache/filter/<spec>/<adapter>/<scene>/<frame>.json`` for every unique
frame, using the registry-named vLLM filter spec. When fed a pair file,
only the union of src/tgt frames is filtered.

Idempotent: if every frame already has a cache entry the server is not
launched; the command returns immediately. The same code path is used
inside ``python -m cli generate``, so re-running this CLI between
generate runs is safe and free.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from cli._io import load_inputs
from pipeline.config import (
    load_stage_config, merge_cli_with_config, stage_config_path,
)

logger = logging.getLogger("filter")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=stage_config_path("filter"),
                   help="per-stage config JSON (default: "
                        "configs/stages/filter.json). CLI flags override.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--in", dest="in_path", type=Path,
                   help="any frame-bearing artifact (frames.json | "
                        "pairs.scored.jsonl | pairs.jsonl)")
    g.add_argument("--frames", dest="frames_path", type=Path,
                   help="alias for --in (kept for back-compat)")
    from models.registry import MODELS
    # `--quality-filter` is the legacy name; the resolved value lands on
    # args.model so it matches the config key.
    p.add_argument("--quality-filter", dest="model", default=None,
                   choices=sorted(MODELS),
                   help="registry name of the filter spec (config key: model)")
    p.add_argument("--vllm-concurrency", type=int, default=None,
                   help="ThreadPool size; default = spec.recommended_concurrency")
    p.add_argument("--filter-votes", dest="n_votes", type=int, default=None,
                   help="Number of inference passes per frame (default 1). "
                        ">1 fires N calls and majority-votes on usable. "
                        "Cache lands at cache/filter/<spec>__voteN/.")
    p.add_argument("--filter-vote-temperature", dest="vote_temperature",
                   type=float, default=None,
                   help="Sampling temperature when n_votes>1 (default 0.7).")
    p.add_argument("--logs-dir", type=Path, default=Path("logs"))
    p.add_argument("--run-id", type=str, default=None,
                   help="defaults to filter__<timestamp>")
    args = p.parse_args()
    args.in_path = args.in_path or args.frames_path

    cfg = load_stage_config("filter", args.config)
    merge_cli_with_config(args, cfg, ("model", "vllm_concurrency",
                                      "n_votes", "vote_temperature"))

    if args.run_id is None:
        args.run_id = f"filter__{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = (args.logs_dir / args.run_id).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s %(name)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    fh = logging.FileHandler(log_dir / "pipeline.log")
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logger.info("run_id=%s  logs at %s", args.run_id, log_dir)

    from models.registry import resolve as resolve_model
    from pipeline.stages import stage_filter

    spec = resolve_model(args.model)
    if not spec.is_vllm:
        raise SystemExit(
            f"filter model {args.model!r} is not a vllm spec."
        )

    bundle = load_inputs(args.in_path)
    if not bundle.frames:
        raise SystemExit(f"--in {args.in_path}: no frames resolved.")
    logger.info("filter:%s — %d frames from %s (kind=%s)",
                spec.name, len(bundle.frames), args.in_path,
                bundle.source_kind)
    stage_filter(spec, bundle.frames,
                 concurrency=args.vllm_concurrency, log_dir=log_dir,
                 n_votes=int(args.n_votes or 1),
                 vote_temperature=float(args.vote_temperature or 0.7))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
