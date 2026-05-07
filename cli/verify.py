"""Stand-alone runner for the pair-verifier stage.

Reads one or more pair-manifest JSONL files (the ``pairs.jsonl`` files
under ``stage_1/<skill>/`` produced by ``python -m cli generate``) and
populates ``cache/verifier/<spec>/<skill>/<scene>/<src>__<tgt>__<sig>.json``
for each row. Optionally writes a per-input ``pairs.verified.jsonl``
sidecar containing only the rows the verifier kept.

Cache-only fast path: if every row already has a cached verdict the
server is not launched.

Fail-closed: a missing cache entry combined with no live endpoint
raises rather than admitting an unverified pair.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("verify")


def _iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> int:
    from pipeline.config import (
        load_stage_config, merge_cli_with_config, stage_config_path,
    )

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=stage_config_path("verify"),
                   help="per-stage config JSON (default: "
                        "configs/stages/verify.json). CLI flags override.")
    p.add_argument("--in", dest="input", action="append", required=True,
                   type=Path,
                   help="path to a pairs.jsonl produced by `python -m cli "
                        "generate`. Repeatable; one verified.jsonl is "
                        "written per input.")
    p.add_argument("--write-verified", dest="write_verified",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="write <input>.verified.jsonl alongside each "
                        "input. Pass --no-write-verified to populate the "
                        "cache only.")
    from models.registry import MODELS
    p.add_argument("--verifier", dest="model", default=None,
                   choices=sorted(MODELS),
                   help="registry name of the verifier spec (must be vllm "
                        "with images_per_prompt>=2). Config key: model.")
    p.add_argument("--verify-concurrency", type=int, default=None,
                   help="ThreadPool size; default = spec.recommended_concurrency")
    p.add_argument("--logs-dir", type=Path, default=Path("logs"))
    p.add_argument("--run-id", type=str, default=None,
                   help="defaults to verify__<timestamp>")
    args = p.parse_args()

    cfg = load_stage_config("verify", args.config)
    merge_cli_with_config(args, cfg,
                          ("model", "verify_concurrency", "write_verified"))

    if args.run_id is None:
        args.run_id = f"verify__{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = (args.logs_dir / args.run_id).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s %(name)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    fh = logging.FileHandler(log_dir / "pipeline.log")
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logger.info("run_id=%s  logs at %s", args.run_id, log_dir)

    from models.registry import resolve as resolve_model
    from pipeline.stages import build_verifier, stage_verify

    spec = resolve_model(args.model)
    if not spec.is_vllm:
        raise SystemExit(
            f"verifier {args.model!r} is not a vllm-backend spec."
        )
    if spec.images_per_prompt < 2:
        raise SystemExit(
            f"verifier {args.model!r} has images_per_prompt="
            f"{spec.images_per_prompt}; need >=2 for pair verification."
        )

    # One server lifetime spans every input file.
    all_rows: list[tuple[Path, int, dict]] = []
    for in_path in args.input:
        if not in_path.exists() or in_path.stat().st_size == 0:
            logger.warning("skip empty/missing %s", in_path)
            continue
        rows = list(_iter_jsonl(in_path))
        for i, r in enumerate(rows):
            if all(k in r for k in
                   ("skill", "scene_id", "frame_src", "frame_tgt",
                    "image_src", "image_tgt")):
                all_rows.append((in_path, i, r))
            else:
                logger.warning("skip malformed row %s:%d (missing keys)",
                               in_path, i)
    logger.info("verifier:%s — %d manifests across %d input file(s)",
                spec.name, len(all_rows), len(args.input))

    manifests = [r for (_, _, r) in all_rows]
    verdicts = stage_verify(spec, manifests,
                            concurrency=args.verify_concurrency,
                            log_dir=log_dir)

    if args.write_verified:
        # Group verdicts back per input file and emit pairs.verified.jsonl.
        from collections import defaultdict
        per_file: dict[Path, list[tuple[int, dict, tuple[bool, str] | None]]] = (
            defaultdict(list)
        )
        for (in_path, i, r), v in zip(all_rows, verdicts):
            per_file[in_path].append((i, r, v))
        for in_path, items in per_file.items():
            out_path = in_path.with_suffix(".verified.jsonl")
            kept = 0
            with open(out_path, "w") as fout:
                for i, r, v in sorted(items, key=lambda t: t[0]):
                    if v is None:
                        continue
                    usable, reason = v
                    if not usable:
                        continue
                    fout.write(json.dumps(r) + "\n")
                    kept += 1
            logger.info("verified %s → %s (%d kept of %d)",
                        in_path, out_path, kept, len(items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
