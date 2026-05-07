"""Stand-alone runner for the match+emit stage (Phase 5).

Reads a ``pairs.scored.jsonl`` (Phase 3 output), groups pairs by scene,
runs the geometric matcher + skill-evidence extractor + JSONL emitter
per scene, and writes the per-skill ``pairs.jsonl`` /
``correspondences.{pos,neg}.jsonl`` / ``rejections.jsonl`` tree under
``--out-root``. Same layout as ``cli generate``'s Phase 5.

Pre-flights (fail-loud):
  * ``--detector labeled-gdino`` requires a populated labeler cache.
  * Perception cache must be populated for every (scene, frame) referenced
    by the input pairs.

Resume / overwrite:
  * **Default**: overwrite ``<out-root>/<skill>/{pairs,correspondences.{pos,neg},rejections}.jsonl``;
    voxel dedup re-initializes per scene.
  * ``--resume``: append to existing files (mirrors ``CorrespondenceWriter``'s
    historical resume mode); voxel dedup is still per-invocation, so cross-run
    duplicates are possible — clean the output dir if you want strict dedup.

W&B upload is fired here (when ``--wandb-project`` is set), since this is
the stage that writes the final outputs.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from cli._io import load_inputs
from pipeline.config import (
    load_stage_config, merge_cli_with_config, stage_config_path,
)

logger = logging.getLogger("match")


_MATCH_KEYS = (
    "detector", "segmenter", "gdino_max_classes", "labeler",
    "prompt_file", "cache_root",
    "seed", "seed_retries", "depth_tol", "iou_min", "voxel_dedup",
    "emit_occlusion_negatives", "max_samples_per_scene",
    "max_det_per_frame",
    "viz_num", "wandb_project", "wandb_run_name", "wandb_max_rows",
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=stage_config_path("match"),
                   help="per-stage config JSON (default: "
                        "configs/stages/match.json). CLI flags override.")
    p.add_argument("--in", dest="in_path", type=Path, required=True,
                   help="pairs.scored.jsonl from `cli pair_gate`")
    p.add_argument("--out-root", type=Path, required=True,
                   help="root folder; one subfolder per skill is created "
                        "under stage_1/")
    p.add_argument("--scenes-root", type=Path,
                   default=Path("/home/mila/l/leh/scratch/dataset/scannet_data/scans"))
    # ── perception models (must match what populated the cache) ------
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
                   help="Required when --detector=labeled-gdino. Cache must "
                        "already be populated by `cli label`.")
    p.add_argument("--prompt-file", dest="prompt_file", type=Path, default=None)
    p.add_argument("--cache-root", type=Path, default=None)
    # ── match knobs --------------------------------------------------
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seed-retries", type=int, default=None)
    p.add_argument("--depth-tol", type=float, default=None)
    p.add_argument("--iou-min", type=float, default=None)
    p.add_argument("--voxel-dedup", type=float, default=None)
    p.add_argument("--emit-occlusion-negatives", dest="emit_occlusion_negatives",
                   action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--max-samples-per-scene", type=int, default=None)
    p.add_argument("--max-det-per-frame", type=int, default=None)
    # ── output mode --------------------------------------------------
    p.add_argument("--resume", action="store_true",
                   help="append to existing per-skill files instead of "
                        "overwriting (cross-run voxel dedup is NOT enforced)")
    # ── viz / W&B ----------------------------------------------------
    p.add_argument("--viz-num", type=int, default=None,
                   help="render up to N pairs per skill (0 = skip)")
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-max-rows", type=int, default=None)
    # ── logging ------------------------------------------------------
    p.add_argument("--logs-dir", type=Path, default=Path("logs"))
    p.add_argument("--run-id", type=str, default=None,
                   help="defaults to match__<timestamp>")
    args = p.parse_args()

    cfg = load_stage_config("match", args.config)
    merge_cli_with_config(args, cfg, _MATCH_KEYS)
    if isinstance(args.cache_root, str):
        args.cache_root = Path(args.cache_root)
    if isinstance(args.prompt_file, str):
        args.prompt_file = Path(args.prompt_file)

    if args.run_id is None:
        args.run_id = f"match__{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = (args.logs_dir / args.run_id).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    args.run_log_dir = log_dir
    log_format = "%(asctime)s %(name)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    fh = logging.FileHandler(log_dir / "pipeline.log")
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logger.info("run_id=%s  logs at %s", args.run_id, log_dir)

    # Load + validate inputs.
    bundle = load_inputs(args.in_path)
    if bundle.scored_pairs is None:
        raise SystemExit(
            f"--in {args.in_path}: expected pairs.scored.jsonl "
            f"(produced by `cli pair_gate`); got source_kind={bundle.source_kind}"
        )
    pairs_by_scene: dict[tuple[str, str], list] = defaultdict(list)
    for sp in bundle.scored_pairs:
        pairs_by_scene[(sp.adapter, sp.scene_id)].append(sp)
    logger.info("match: %d pairs across %d (adapter, scene) groups",
                len(bundle.scored_pairs), len(pairs_by_scene))

    adapters_in_input = sorted({a for a, _ in pairs_by_scene.keys()})
    if len(adapters_in_input) > 1:
        raise SystemExit(
            f"--in {args.in_path}: multi-adapter inputs are not supported "
            f"by `cli match` (got {adapters_in_input})."
        )
    adapter_name = adapters_in_input[0]
    args.adapter = adapter_name

    # Pre-flight: labeler cache (only if labeled-gdino).
    needs_labeler = args.detector in ("labeled-gdino", "gemini+gdino")
    if needs_labeler:
        from models.registry import resolve as resolve_model
        from pipeline.stages import labeler_cache_complete
        labeler_spec = resolve_model(args.labeler)
        if not labeler_cache_complete(labeler_spec, bundle.frames,
                                      prompt_file=args.prompt_file):
            raise SystemExit(
                f"--detector {args.detector!r} requires labeler cache "
                f"({labeler_spec.name!r}) populated for every frame in pairs. "
                f"Run `cli label --in {args.in_path} --labeler "
                f"{labeler_spec.name}` first."
            )
        cache_only_labeler_spec = labeler_spec
    else:
        cache_only_labeler_spec = None

    # Pre-flight: perception cache (only enforced for GPU-heavy combos;
    # CPU-only combos compute lazily via PerceptionCache.get).
    from pipeline.stages import (
        CPU_ONLY_DETECTORS, CPU_ONLY_SEGMENTERS,
        perception_cache_complete, stage_match,
    )
    model_tag = f"{args.detector}+{args.segmenter}"
    scene_to_frames: dict[str, list] = defaultdict(list)
    for f in bundle.frames:
        scene_to_frames[f.scene_id].append(f)
    is_cpu_only = (args.detector in CPU_ONLY_DETECTORS
                   and args.segmenter in CPU_ONLY_SEGMENTERS)
    if not is_cpu_only:
        for sid, frames in scene_to_frames.items():
            if not perception_cache_complete(args.cache_root, adapter_name,
                                             sid, model_tag, frames):
                raise SystemExit(
                    f"perception cache incomplete for scene={sid!r} "
                    f"(detector={args.detector}, segmenter={args.segmenter}). "
                    f"Run `cli perceive --in {args.in_path} --detector "
                    f"{args.detector} --segmenter {args.segmenter}` first."
                )

    # Build detector + segmenter (cache-only — no live model loads at
    # match time when caches are complete; the local PerceptionCache will
    # only ever read).
    from cli.generate import (
        make_adapter, make_detector, make_segmenter,
    )
    from pipeline.stages import build_labeler
    cache_only_labeler = (
        build_labeler(cache_only_labeler_spec, endpoint=None,
                      prompt_file=args.prompt_file)
        if cache_only_labeler_spec is not None else None
    )
    detector = make_detector(
        args.detector, labeler=cache_only_labeler,
        gdino_max_classes=args.gdino_max_classes,
        labeler_concurrency=1,
    )
    segmenter = make_segmenter(args.segmenter)

    # Build writers.
    from pipeline.config import load_skills_config
    from pipeline.dedup import VoxelSet
    from pipeline.emit import TaskRouter
    from pipeline.manifest import PairManifestWriter
    from pipeline.skills import (
        CONTENT_SKILLS, POSE_SKILLS, load_content_skills,
    )

    task_config = load_skills_config()
    content_skills = load_content_skills(task_config)

    t0 = time.time()
    total = 0
    with TaskRouter(args.out_root, resume=args.resume) as writer:
        manifest_writer = PairManifestWriter(
            args.out_root / TaskRouter.STAGE,
            skills=[*POSE_SKILLS, *CONTENT_SKILLS],
            resume=args.resume,
        )
        try:
            for (adapter_name_, scene_id), scored_pairs in sorted(
                    pairs_by_scene.items()):
                try:
                    adapter = make_adapter(adapter_name_,
                                           args.scenes_root / scene_id)
                except FileNotFoundError as e:
                    logger.warning("skip (%s, %s): %s",
                                   adapter_name_, scene_id, e)
                    continue
                # ScoredPair -> ViewPair so stage_match can use the same
                # in-memory shape generate uses.
                view_pairs = [sp.to_view_pair() for sp in scored_pairs]
                voxels_pos = VoxelSet(args.voxel_dedup)
                voxels_neg = VoxelSet(args.voxel_dedup)
                n = stage_match(
                    adapter, pairs=view_pairs, args=args,
                    segmenter=segmenter, detector=detector,
                    cache_root=args.cache_root, model_tag=model_tag,
                    writer=writer, manifest_writer=manifest_writer,
                    content_skills=content_skills,
                    voxels_pos=voxels_pos, voxels_neg=voxels_neg,
                )
                logger.info("[%s] emitted %d", scene_id, n)
                total += n
            counts = writer.counts()
        finally:
            counts_m = manifest_writer.counts()
            manifest_writer.close()
            logger.info("pair manifests: %s", counts_m)

    dt = time.time() - t0
    logger.info("DONE: %d records, %.1fs, counts=%s", total, dt, counts)

    if args.wandb_project:
        from pipeline.wandb_uploader import upload_run
        cfg = {k: (str(v) if isinstance(v, Path) else v)
               for k, v in vars(args).items()}
        upload_run(
            args.out_root,
            project=args.wandb_project,
            run_name=args.wandb_run_name,
            config=cfg,
            max_table_rows=args.wandb_max_rows,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
