"""Stand-alone runner for the labeler stage.

Reads any frame-bearing artifact (``frames.json`` from ``cli sample``,
or a pair file from ``cli pair_gate`` / ``cli match`` via
``cli/_io.py::load_inputs``). When fed a pair file, only the union of
src/tgt frames is labeled (paper-faithful: matches what
``cli generate``'s Phase 4 does).

Idempotent: cache-complete frames trigger no server launch. Same code
path used inside ``python -m cli generate``.
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

logger = logging.getLogger("label")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=stage_config_path("label"),
                   help="per-stage config JSON (default: "
                        "configs/stages/label.json). CLI flags override.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--in", dest="in_path", type=Path,
                   help="any frame-bearing artifact (frames.json | "
                        "pairs.scored.jsonl | pairs.jsonl)")
    g.add_argument("--frames", dest="frames_path", type=Path,
                   help="alias for --in (kept for back-compat)")
    from models.registry import MODELS
    p.add_argument("--labeler", dest="model", default=None,
                   choices=sorted(MODELS),
                   help="registry name of the labeler spec (config key: model)")
    p.add_argument("--prompt-file", type=Path, default=None,
                   help="override label prompt (default: from config / "
                        "configs/label_prompt.txt)")
    p.add_argument("--vllm-concurrency", type=int, default=None,
                   help="ThreadPool size; default = spec.recommended_concurrency")
    p.add_argument("--label-votes", dest="n_votes", type=int, default=None,
                   help="Number of inference passes per frame (default 1). "
                        ">1 fires N calls and persists every parsed run under "
                        "cache/labels/<spec>__voteN/. Aggregation happens "
                        "downstream via the detector's --label-vote-strategy.")
    p.add_argument("--label-vote-temperature", dest="vote_temperature",
                   type=float, default=None,
                   help="Sampling temperature when n_votes>1 (default 0.7). "
                        "Must be >0 or every run returns identical text.")
    p.add_argument("--logs-dir", type=Path, default=Path("logs"))
    p.add_argument("--run-id", type=str, default=None,
                   help="defaults to label__<timestamp>")
    args = p.parse_args()
    args.in_path = args.in_path or args.frames_path

    cfg = load_stage_config("label", args.config)
    merge_cli_with_config(args, cfg,
                          ("model", "prompt_file", "vllm_concurrency",
                           "n_votes", "vote_temperature"))
    if isinstance(args.prompt_file, str):
        args.prompt_file = Path(args.prompt_file)

    if args.run_id is None:
        args.run_id = f"label__{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = (args.logs_dir / args.run_id).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s %(name)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    fh = logging.FileHandler(log_dir / "pipeline.log")
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logger.info("run_id=%s  logs at %s", args.run_id, log_dir)

    from models.registry import resolve as resolve_model
    from pipeline.stages import stage_label

    spec = resolve_model(args.model)
    bundle = load_inputs(args.in_path)
    if not bundle.frames:
        raise SystemExit(f"--in {args.in_path}: no frames resolved.")
    logger.info("labeler:%s — %d frames from %s (kind=%s)",
                spec.name, len(bundle.frames), args.in_path,
                bundle.source_kind)
    stage_label(spec, bundle.frames,
                concurrency=args.vllm_concurrency, log_dir=log_dir,
                prompt_file=args.prompt_file,
                n_votes=int(args.n_votes or 1),
                vote_temperature=float(args.vote_temperature or 0.7))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
