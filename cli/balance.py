"""Post-stage-1 balancer: per-scene + per-answer-bucket quotas, optional vLLM verifier.

Reads ``stage_1/<skill>/pairs.jsonl`` produced by ``python -m cli generate``
and writes ``stage_1/<skill>/pairs.balanced.jsonl`` containing a subset that:

  1. Passes the pair verifier (optional, ``--verifier <registry-name>``).
  2. Respects a per-scene cap (``--per-scene-cap``).
  3. Respects a per-answer-bucket cap (``--per-bucket-cap``). The bucket is
     skill-specific — see ``_answer_bucket()``.

A rejections sidecar ``pairs.rejected.jsonl`` is written alongside, with
one entry per dropped manifest and the rejection reason.

Verifier execution model — parity with ``cli/generate.py``:

  * One vLLM server is launched for the verifier spec; every skill's
    pending manifests go through that one server lifetime; server is
    killed before this command returns.
  * Per-manifest cache hits are fully reusable; if every selected
    manifest is already cached the server is **not** launched.
  * Verdicts are looked up from cache during the sequential balance
    loop, so quota checks (which depend on the keep/drop decision) stay
    deterministic with respect to the input ordering.

Example:
    python -m cli balance --out-root outputs/run \\
        --skill anchor --skill counting --skill relative_direction \\
        --per-scene-cap 40 --per-bucket-cap 200 \\
        --verifier qwen3vl-8B-pair --verify-concurrency 8

Run ``--verifier none`` (the default) to apply quotas only — no GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("balance_pairs")


ALL_SKILLS = (
    "anchor", "counting", "relative_distance", "relative_direction",
    "cross_correspondence", "cross_spatial_transformation",
    "cross_depth_variation", "cross_occlusion_visibility",
)


# ---- answer-bucket policy per skill ------------------------------------

def _answer_bucket(skill: str, manifest: dict) -> str:
    """One hashable bucket label per manifest, used for per-answer caps.

    Chosen to reflect the *answer* the pair teaches, so quotas prevent
    one answer class from dominating training:

      anchor                         -> winning category (src_label of first match)
      counting                       -> (category, count)
      relative_distance              -> label of the farthest candidate
      relative_direction             -> compass bucket of the first target
      cross_correspondence           -> src_label of first visible match
      cross_spatial_transformation   -> src_label of first transformed obj
      cross_depth_variation          -> sign of first delta (closer/farther)
      cross_occlusion_visibility     -> (n_visible > 0, n_occluded > 0) class
    """
    ev = manifest.get("evidence", {})
    if skill == "anchor":
        objs = ev.get("shared_objects", [])
        return objs[0].get("src_label", "?") if objs else "?"
    if skill == "counting":
        return f"{ev.get('category','?')}/{ev.get('unique_total',0)}"
    if skill == "relative_distance":
        cands = ev.get("candidates", [])
        return cands[0].get("label", "?") if cands else "?"
    if skill == "relative_direction":
        tgts = ev.get("targets", [])
        return tgts[0].get("bucket", "?") if tgts else "?"
    if skill == "cross_correspondence":
        objs = manifest.get("objects", [])
        vis = next((o for o in objs if o.get("visible")), None)
        return vis.get("src_label", "?") if vis else "?"
    if skill == "cross_spatial_transformation":
        objs = ev.get("transformed_objects", [])
        return objs[0].get("label", "?") if objs else "?"
    if skill == "cross_depth_variation":
        objs = ev.get("varying_objects", [])
        if not objs:
            return "?"
        return "closer" if float(objs[0].get("delta_m", 0.0)) > 0 else "farther"
    if skill == "cross_occlusion_visibility":
        return f"vis={ev.get('n_visible',0)>0}/occ={ev.get('n_occluded',0)>0}"
    return "?"


# ---- main loop ---------------------------------------------------------

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


def balance_skill(
    skill: str, stage_root: Path,
    per_scene_cap: Optional[int], per_bucket_cap: Optional[int],
    seed: int, verifier=None,
) -> dict[str, int]:
    in_path = stage_root / skill / "pairs.jsonl"
    if not in_path.exists() or in_path.stat().st_size == 0:
        logger.info("[%s] no pairs.jsonl, skipping", skill)
        return {"in": 0, "kept": 0, "dropped": 0}

    rows = list(_iter_jsonl(in_path))
    random.Random(seed).shuffle(rows)

    out_path = stage_root / skill / "pairs.balanced.jsonl"
    rej_path = stage_root / skill / "pairs.rejected.jsonl"

    scene_counts: Counter = Counter()
    bucket_counts: Counter = Counter()

    kept = 0
    dropped: Counter = Counter()
    with open(out_path, "w") as fout, open(rej_path, "w") as frej:
        for r in rows:
            scene = r.get("scene_id", "?")
            bucket = _answer_bucket(skill, r)

            # Cheap quota check first (no GPU).
            if per_scene_cap is not None and scene_counts[scene] >= per_scene_cap:
                frej.write(json.dumps({**_id(r), "reason": "per_scene_cap"}) + "\n")
                dropped["per_scene_cap"] += 1
                continue
            if per_bucket_cap is not None and bucket_counts[bucket] >= per_bucket_cap:
                frej.write(json.dumps({**_id(r), "reason": "per_bucket_cap",
                                       "bucket": bucket}) + "\n")
                dropped["per_bucket_cap"] += 1
                continue

            # Verifier verdict (cache-only; live pass already populated
            # the cache in main()). Fail-closed: a missing cache entry or
            # a not-usable verdict drops the row.
            if verifier is not None:
                try:
                    usable, reason = verifier.verify(r)
                except Exception as e:
                    usable, reason = False, f"verifier_error:{e}"
                if not usable:
                    frej.write(json.dumps({**_id(r), "reason": f"verifier:{reason}"}) + "\n")
                    dropped["verifier"] += 1
                    continue

            fout.write(json.dumps(r) + "\n")
            scene_counts[scene] += 1
            bucket_counts[bucket] += 1
            kept += 1

    logger.info("[%s] in=%d kept=%d dropped=%s scenes=%d buckets=%d",
                skill, len(rows), kept, dict(dropped),
                len(scene_counts), len(bucket_counts))
    return {"in": len(rows), "kept": kept, **dropped}


def _id(manifest: dict) -> dict:
    return {
        "skill": manifest.get("skill"),
        "scene_id": manifest.get("scene_id"),
        "frame_src": manifest.get("frame_src"),
        "frame_tgt": manifest.get("frame_tgt"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, required=True,
                   help="the --out-root used by python -m cli generate")
    p.add_argument("--skill", action="append", default=None,
                   help="repeatable; default = all skills with a pairs.jsonl")
    p.add_argument("--per-scene-cap", type=int, default=None,
                   help="max accepted manifests per (scene, skill)")
    p.add_argument("--per-bucket-cap", type=int, default=None,
                   help="max accepted manifests per answer bucket per skill")

    from models.registry import MODELS
    _MODEL_CHOICES = sorted(MODELS)
    p.add_argument("--verifier", default="none",
                   choices=["none"] + _MODEL_CHOICES,
                   help="Registry name of the pair-verifier model "
                        "(or 'none' to skip verification). Resolves via "
                        "models.registry.MODELS — same registry used by "
                        "--labeler / --quality-filter. The verifier must "
                        "be a vllm-backend spec with images_per_prompt>=2 "
                        "(qwen3vl-8B-pair is the default operational choice).")
    p.add_argument("--verify-concurrency", type=int, default=None,
                   help="ThreadPool size for concurrent verifier requests "
                        "against the vLLM server. Default: spec's "
                        "recommended_concurrency.")
    p.add_argument("--logs-dir", type=Path, default=Path("logs"),
                   help="Per-run log artifacts land at <logs-dir>/<run-id>/. "
                        "Mirrors `python -m cli generate`.")
    p.add_argument("--run-id", type=str, default=None,
                   help="Subdirectory under --logs-dir. Defaults to "
                        "balance__<timestamp>.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.run_id is None:
        args.run_id = f"balance__{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    args.run_log_dir = (args.logs_dir / args.run_id).resolve()
    args.run_log_dir.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s %(name)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    file_handler = logging.FileHandler(args.run_log_dir / "pipeline.log")
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)
    logger.info("run_id=%s  logs at %s", args.run_id, args.run_log_dir)

    stage_root = args.out_root / "stage_1"
    if not stage_root.exists():
        raise SystemExit(f"{stage_root} not found; run python -m cli generate first")

    skills = args.skill or [
        s for s in ALL_SKILLS
        if (stage_root / s / "pairs.jsonl").exists()
    ]
    if not skills:
        raise SystemExit("no skills to balance (no pairs.jsonl found)")

    # ── Optional verifier stage — one server for all skills ──────────
    verifier_spec = None
    cache_only_verifier = None
    if args.verifier != "none":
        from models.registry import resolve as resolve_model
        from pipeline.stages import (
            build_verifier, collect_pair_manifests, stage_verify,
        )
        verifier_spec = resolve_model(args.verifier)
        if not verifier_spec.is_vllm:
            raise SystemExit(
                f"--verifier {args.verifier!r} is not a vllm-backend spec. "
                f"Pick a vllm spec with images_per_prompt>=2."
            )

        # Collect every manifest that needs a verdict — across all selected
        # skills. The collector applies the same required-key check we used
        # to do inline; row order is deterministic by skill name (sorted
        # iterdir) which keeps verifier cache keys stable across runs.
        all_manifests = collect_pair_manifests(stage_root, skills=skills)
        logger.info("verifier:%s — %d manifests across %d skill(s)",
                    args.verifier, len(all_manifests), len(skills))

        # One server lifetime, fan-out across all manifests.
        stage_verify(verifier_spec, all_manifests,
                     concurrency=args.verify_concurrency,
                     log_dir=args.run_log_dir)

        # Use a cache-only verifier inside the per-skill balance loop so
        # the order of quota decisions stays sequential. Any cache miss
        # at this point is fail-closed (RuntimeError) and the row is
        # rejected with reason verifier_error.
        cache_only_verifier = build_verifier(verifier_spec, endpoint=None)

    totals: dict[str, dict[str, int]] = {}
    for s in skills:
        totals[s] = balance_skill(
            s, stage_root,
            per_scene_cap=args.per_scene_cap,
            per_bucket_cap=args.per_bucket_cap,
            seed=args.seed, verifier=cache_only_verifier,
        )

    logger.info("DONE: %s", {k: v.get("kept", 0) for k, v in totals.items()})


if __name__ == "__main__":
    main()
