"""Phase 1 CLI: generate verified cross-view object correspondences.

Pipeline (per scene):
  1. Select view-pairs (pose pre-filter + frustum-overlap).
  2. For every frame in the selected pairs, run Detector+Segmenter (cached).
  3. For every pair, match src masks -> tgt masks geometrically.
  4. Voxel-dedup by 3D world point; emit one JSONL line per surviving match.

Examples:
    # Noop smoke (no GPU): synthetic 3x3 grid of "objects" per frame
    python python -m cli generate --scene scene0000_00 \
        --frame-stride 50 --limit-frames 20 \
        --detector noop --segmenter noop \
        --out-root outputs/smoke

    # Real GPU run on 2 scenes with GD + SAM 2.1, occlusion negatives on
    python python -m cli generate --scene scene0000_00 --scene scene0001_00 \
        --detector gdino --segmenter sam2.1 --emit-occlusion-negatives \
        --out-root outputs/2scenes

    # Full dataset, paper-style: 235B labeler + 8B filter (sequential vLLM
    # servers — only one model is loaded at a time, no GPU contention).
    python python -m cli generate --all-scenes \
        --detector labeled-gdino --segmenter sam2.1 \
        --labeler qwen3vl-235B --quality-filter qwen3vl-8B \
        --emit-occlusion-negatives --out-root outputs/full --resume

Output layout under --out-root (one folder per skill, pos / neg split):
    {root}/<skill>/correspondences.pos.jsonl       ← visible matches
    {root}/<skill>/correspondences.neg.jsonl       ← occluded (visible=False)
    {root}/<skill>/pairs.jsonl                     ← per-pair manifest
    {root}/_all/correspondences.jsonl              ← every record (for QC/viz)
    {root}/_all/correspondences.rejections.jsonl   ← per-frame/mask rejection log
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

from datasets.base import BaseSceneAdapter
from datasets.matterport import MatterportAdapter
from datasets.scannet import ScanNetAdapter
from pipeline.dedup import VoxelSet
from pipeline.emit import (
    CorrespondenceRecord, CorrespondenceWriter, TaskRouter, round_clip_pixel,
)
from pipeline.config import (
    load_skills_config, load_run_config, load_stage_config,
    merge_cli_with_config, resolve, RunPreset, STAGE_NAMES,
)
from pipeline.match import match_pair
from pipeline.pairs import select_pairs
from pipeline.skills import (
    CONTENT_SKILLS, POSE_SKILLS,
    extract_all_evidence, load_content_skills,
)
from pipeline.manifest import PairManifestWriter, build_manifest
from pipeline.stages import (
    build_filter, build_labeler,
    filter_cache_complete, labeler_cache_complete,
    run_filter_pass, run_labeler_pass,
    stage_filter, stage_label,
)
from models._frame_ref import FrameRef
from models.base import Detector, Segmenter

logger = logging.getLogger("generate_correspondences")


# ---- adapter / model factories ------------------------------------------

def make_adapter(name: str, scene_dir: Path) -> BaseSceneAdapter:
    if name == "scannet":
        return ScanNetAdapter(scene_dir)
    if name == "matterport":
        return MatterportAdapter(scene_dir)
    raise ValueError(f"Unknown adapter '{name}'")


def make_detector(name: str, labeler=None,
                  gdino_max_classes: int = 80,
                  labeler_concurrency: int = 1,
                  vote_strategy: str = "union") -> Detector:
    """Build a Detector. `labeler` is a pre-built LabelerProtocol instance
    (or None for detectors that don't need one)."""
    if name == "noop":
        from models.noop import NoopDetector
        return NoopDetector()
    if name == "gdino":
        from models.detectors.gdino import GDinoDetector
        return GDinoDetector()
    if name == "gdino+scannet200":
        from models.detectors.gdino import (
            GDinoDetector, SCANNET200_CLASSES_FILE, load_classes_from_file,
        )
        return GDinoDetector(
            classes=load_classes_from_file(SCANNET200_CLASSES_FILE,
                                           max_classes=gdino_max_classes),
        )
    if name in ("labeled-gdino", "gemini+gdino"):
        # `gemini+gdino` is the legacy alias kept for back-compat. The
        # actual labeler is selected via `--labeler <registry name>` and
        # plumbed in by the orchestrator.
        from models.detectors.labeled_gdino import LabeledGDinoDetector
        if labeler is None:
            raise ValueError(
                f"detector {name!r} requires a labeler — pass --labeler "
                f"<registry name>"
            )
        return LabeledGDinoDetector(labeler=labeler,
                                   labeler_concurrency=labeler_concurrency,
                                   vote_strategy=vote_strategy)
    if name == "scannet-gt":
        from models.gt.scannet import ScanNetGTDetector
        return ScanNetGTDetector()
    if name == "scannet-gt-label+gdino":
        from models.gt.scannet_gdino import ScanNetGTLabelGDinoDetector
        return ScanNetGTLabelGDinoDetector()
    raise ValueError(f"Unknown detector '{name}'")


# ---- staged orchestration helpers ---------------------------------------
#
# Builders, cache-completeness probes, and run_*_pass helpers live in
# `pipeline/stages.py` so the per-stage CLIs (cli/filter.py, cli/label.py,
# cli/verify.py) can reuse the exact same code paths. We import them at
# module top.

def _model_tag(args) -> str:
    """Human-readable subdir under <cache>/<adapter>/<scene>/."""
    return f"{args.detector}+{args.segmenter}"


def _frame_refs(adapter_name: str, adapter, frame_ids) -> list[FrameRef]:
    """Build a FrameRef per frame_id under a single adapter+scene."""
    return [adapter.frame_ref(fid, adapter_name) for fid in frame_ids]


def make_segmenter(name: str) -> Segmenter:
    if name == "noop":
        from models.noop import NoopSegmenter
        return NoopSegmenter()
    if name == "sam2.1":
        from models.segmenters.sam21 import SAM21Segmenter
        return SAM21Segmenter()
    if name == "sam3":
        from models.segmenters.sam3 import SAM3Segmenter
        return SAM3Segmenter()
    if name == "gt-mask":
        from models.segmenters.gt import GTMaskSegmenter
        # Adapter is plugged in per-scene by `_perception_emit_scene` via
        # `segmenter.set_adapter(adapter)`.
        return GTMaskSegmenter()
    raise ValueError(f"Unknown segmenter '{name}'")


# ---- per-frame perception cache ----------------------------------------

class PerceptionCache:
    """Disk-cached Detector+Segmenter pipeline.

    Layout: ``<root>/<adapter>/<scene>/<model_tag>/<frame_id>[__<suffix>].pkl``
    where ``model_tag`` is a human-readable detector+segmenter name
    (e.g. ``labeled-gdino+sam2.1``). No hashing.
    """

    def __init__(self, adapter_name: str, scene_id: str, root: Path,
                 detector: Detector, segmenter: Segmenter,
                 model_tag: str):
        self.adapter_name = adapter_name
        self.scene_id = scene_id
        self.detector = detector
        self.segmenter = segmenter
        self.dir = root / adapter_name / scene_id / model_tag
        self.dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, list] = {}

    def get(self, image_path: Path, frame_id: str,
            labels: Optional[list[str]] = None,
            canon_suffix: Optional[str] = None) -> list:
        """Fetch (or compute and cache) masks for a frame.

        If `labels` is supplied, the detector's `detect_with_labels` path is
        used and the cache key is `<frame_id>__<canon_suffix>` so different
        label vocabularies for the same frame don't collide. The double
        underscore avoids ambiguity with frame_ids that themselves
        contain underscores.
        """
        cache_key = (f"{frame_id}__{canon_suffix}" if canon_suffix
                     else frame_id)
        if cache_key in self._mem:
            return self._mem[cache_key]
        p = self.dir / f"{cache_key}.pkl"
        if p.exists():
            try:
                with open(p, "rb") as f:
                    masks = pickle.load(f)
                self._mem[cache_key] = masks
                return masks
            except (EOFError, pickle.UnpicklingError):
                p.unlink()
        frame_ref = FrameRef(
            image_path=image_path, adapter=self.adapter_name,
            scene_id=self.scene_id, frame_id=frame_id,
        )
        if labels is not None and hasattr(self.detector, "detect_with_labels"):
            dets = self.detector.detect_with_labels(frame_ref, labels)
        else:
            dets = self.detector.detect(frame_ref)
        masks = self.segmenter.segment(image_path, dets)
        # Post-SAM step: backfill mask.canonical from the detector's
        # scene-wide object→canonical map (if any). SAM stays
        # label-agnostic; canonical is purely a downstream identity.
        canonicalize = getattr(self.detector, "canonicalize_mask_label", None)
        if callable(canonicalize):
            for m in masks:
                m.canonical = canonicalize(m.label)
        with open(p, "wb") as f:
            pickle.dump(masks, f)
        self._mem[cache_key] = masks
        return masks


# ---- main ---------------------------------------------------------------

# ---- per-scene helpers (called from `process_scenes`) -------------------

def _sample_scene(adapter: BaseSceneAdapter, args) -> list[FrameRef]:
    """Sample keyframes for one scene → list[FrameRef]. No GPU."""
    from pipeline.pairs import sample_keyframes
    sampled_fids, _mode = sample_keyframes(
        adapter,
        sampling=args.sampling,
        frame_stride=args.frame_stride,
        min_keyframes=args.min_keyframes,
        min_translation_m=args.min_translation_m,
        min_rotation_deg=args.min_rotation_deg,
        limit_frames=args.limit_frames,
        cosmic_base_sampling=args.cosmic_base_sampling,
        cosmic_union_coverage_min=args.cosmic_union_coverage_min,
        cosmic_yaw_diff_min_deg=args.cosmic_yaw_diff_min_deg,
        log=False,
    )
    return _frame_refs(args.adapter, adapter, sampled_fids)


def _pair_gate_scene(adapter: BaseSceneAdapter, args,
                     *, pair_config, writer
                     ) -> tuple[list, list[FrameRef]]:
    """Pair-gate one scene → (pairs, frames_for_pairs). Pure
    pose/frustum gate — no vLLM server, no quality filter (filter runs
    *after* pair-gate in the new ordering and culls surviving pairs
    post-hoc).

    Thin wrapper around ``pipeline.stages.stage_pair_gate`` — same
    helper used by the standalone ``cli pair_gate`` runner.
    """
    from pipeline.stages import stage_pair_gate

    def on_pair_reject(pair, reason):
        writer.reject(adapter.scene_id, pair.src_id, pair.tgt_id, -1, reason)

    return stage_pair_gate(
        adapter,
        adapter_name=args.adapter,
        pair_config=pair_config,
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
        on_pair_reject=on_pair_reject,
    )


def _perception_emit_scene(adapter: BaseSceneAdapter, args, writer,
                           segmenter: Segmenter, detector,
                           cache_root: Path, *,
                           pairs, frames_for_pairs: list[FrameRef],
                           content_skills, manifest_writer) -> int:
    """Per-scene perception (GDino+SAM) + geometric matching + emit.

    Thin wrapper around ``pipeline.stages.stage_match`` — same helper
    used by the standalone ``cli match`` runner. Voxel-dedup state is
    fresh per scene (no cross-scene bleed).
    """
    from pipeline.stages import stage_match

    if hasattr(detector, "prepare_scene"):
        # Same prepare_scene log line as before — log here so behavior
        # is identical (stage_match also calls prepare_scene but doesn't
        # log this summary).
        scene_objs = getattr(detector, "_scene_objects", None)
        l2c = getattr(detector, "_label_to_canonical", None)
        if scene_objs and l2c:
            n_canon = len({c for c in l2c.values()})
            logger.info("[%s] scene vocab: %d objects → %d canonicals",
                        adapter.scene_id, len(scene_objs), n_canon)

    voxels_pos = VoxelSet(args.voxel_dedup)
    voxels_neg = VoxelSet(args.voxel_dedup)

    return stage_match(
        adapter, pairs=pairs, args=args,
        segmenter=segmenter, detector=detector,
        cache_root=cache_root, model_tag=_model_tag(args),
        writer=writer, manifest_writer=manifest_writer,
        content_skills=content_skills,
        voxels_pos=voxels_pos, voxels_neg=voxels_neg,
    )


# ---- Phase 4.5: multi-GPU perception pre-pass --------------------------

def _resolve_perception_workers(args) -> int:
    """``--perception-workers`` defaults to ``torch.cuda.device_count()``
    when not explicitly set. Returning 0 disables the pre-pass."""
    if args.perception_workers is not None:
        return max(0, int(args.perception_workers))
    try:
        import torch
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _maybe_run_perception_prepass(args, scene_state, filter_spec,
                                  labeler_spec) -> None:
    """Run the multi-GPU perception pre-pass when the configuration
    supports it; log a one-line skip message otherwise.

    Thin wrapper around ``pipeline.stages.stage_perceive`` (which owns
    the auto-skip rules: ``num_workers<=0``, CPU-only combo, below
    ``prepass_min_frames``).
    """
    from pipeline.stages import stage_perceive

    n_workers = _resolve_perception_workers(args)
    try:
        import torch
        gpu_count = torch.cuda.device_count()
    except Exception:
        gpu_count = 0
    if n_workers > 0 and gpu_count > 0:
        n_workers = min(n_workers, gpu_count)

    scene_to_frames = {
        sid: st.get("frames_for_pairs") or []
        for sid, st in scene_state.items()
    }
    t0 = time.time()
    written = stage_perceive(
        adapter_name=args.adapter,
        scenes_root=args.scenes_root,
        detector_name=args.detector,
        segmenter_name=args.segmenter,
        labeler_spec_name=(labeler_spec.name if labeler_spec is not None
                           else None),
        prompt_file=args.prompt_file,
        gdino_max_classes=args.gdino_max_classes,
        cache_root=args.cache_root,
        model_tag=_model_tag(args),
        compile_perception=bool(args.compile_perception),
        perception_batch_frames=int(args.perception_batch_frames),
        scene_to_frames=scene_to_frames,
        num_workers=n_workers,
        prepass_min_frames=args.perception_prepass_min_frames,
        gpu_ids=list(range(n_workers)) if n_workers > 0 else None,
        log_dir=getattr(args, "run_log_dir", None),
        n_votes=int(getattr(args, "n_votes", None) or 1),
        vote_temperature=float(getattr(args, "vote_temperature", None) or 0.7),
        vote_strategy=str(getattr(args, "vote_strategy", None) or "union"),
    )
    if written:
        logger.info("perception pre-pass: done (%d frames written in %.1fs)",
                    written, time.time() - t0)


# ---- multi-scene orchestrator (one server lifetime per VLM stage) -------

def process_scenes(scene_ids: list[str], args, writer,
                   cache_root: Path, segmenter: Segmenter,
                   task_config,
                   filter_spec=None, labeler_spec=None,
                   verifier_spec=None,
                   content_skills=None, manifest_writer=None) -> int:
    """Process every scene in `scene_ids` with one vLLM server lifetime
    per VLM stage spanning the whole batch.

    Phase 0 (no GPU): build adapters once per scene (lazy, ~10 KB each).
    Phase 1 (no GPU): sample keyframes per scene.
    Phase 2 (no server): pair-gate per scene — pure pose/frustum gate.
    Phase 3 (filter server up — one lifetime, frames-in-pairs only):
       run filter, then drop pairs whose src or tgt frame is unusable.
    Phase 4 (labeler server up — one lifetime, reduced frames-in-pairs).
       Phases 3+4 collapse into one server lifetime when filter_spec ==
       labeler_spec (the same `with launch_server(...)` block does
       filter → post-hoc pair drop → labeler across all scenes).
    Phase 5 (no LLM server, GPUs free for GDino+SAM): perception + match
       + emit per scene. Detector/segmenter built once across scenes;
       per-scene `prepare_scene` / `set_adapter` calls reset state.
    Phase 6 (verifier server up — optional, one lifetime across every
       emitted manifest): runs only when `verifier_spec` is set.

    With same-spec collapse and no verifier, this is **one** vLLM
    server lifetime regardless of N scenes — same shape as
    matterport3d-data-gen's `vllm serve & python loop` pattern.
    """
    from models.registry import launch_server
    from pipeline.stages import (
        apply_filter_to_pairs,
        collect_pair_manifests, write_verified_per_skill,
    )
    concurrency = args.vllm_concurrency

    # Phase 0 — adapters.
    adapters: dict[str, BaseSceneAdapter] = {}
    for sid in scene_ids:
        try:
            adapters[sid] = make_adapter(args.adapter, args.scenes_root / sid)
        except FileNotFoundError as e:
            logger.warning("skip %s: %s", sid, e)
    if not adapters:
        logger.warning("no scenes resolved — exiting")
        return 0

    # Cosmic restricts the content-skill set; do it once for the whole batch.
    if args.sampling == "cosmic" and content_skills:
        from pipeline.cosmic import COSMIC_SKILLS
        content_skills = {k: v for k, v in content_skills.items()
                          if k in COSMIC_SKILLS}

    pair_configs = {sid: resolve(task_config,
                                 getattr(a, "source_name", "unknown"))
                    for sid, a in adapters.items()}

    # Phase 1 — sample.
    scene_state: dict[str, dict] = {}
    for sid, adapter in adapters.items():
        scene_state[sid] = {"sampled_frames": _sample_scene(adapter, args)}

    # Phase 2 — pair-gate (pure pose/frustum, no server, no filter).
    for sid, adapter in adapters.items():
        pairs, ffp = _pair_gate_scene(
            adapter, args,
            pair_config=pair_configs[sid], writer=writer,
        )
        scene_state[sid]["pairs"] = pairs
        scene_state[sid]["frames_for_pairs"] = ffp

    # Frames-in-pairs across all scenes — input set for filter and labeler.
    all_pair_frames = [
        f for st in scene_state.values()
        for f in st.get("frames_for_pairs", [])
    ]

    # Phases 3+4 — filter (frames-in-pairs) → drop unusable pairs → labeler.
    f_votes = int(getattr(args, "filter_n_votes", None) or 1)
    f_vote_temp = float(getattr(args, "filter_vote_temperature", None) or 0.7)
    l_votes = int(args.n_votes or 1)
    l_vote_temp = float(args.vote_temperature or 0.7)
    collapse = (
        filter_spec is not None and labeler_spec is not None
        and filter_spec == labeler_spec
    )
    need_filter_server = (
        filter_spec is not None and all_pair_frames
        and not filter_cache_complete(filter_spec, all_pair_frames,
                                      n_votes=f_votes,
                                      vote_temperature=f_vote_temp)
    )

    def _post_filter_drop_pairs() -> None:
        """After the filter cache is populated, drop pairs whose src or
        tgt frame is unusable. Logs ``qwen_filter:<reason>`` rejects
        and rebuilds each scene's ``frames_for_pairs`` from survivors."""
        if filter_spec is None:
            return
        for sid, adapter in adapters.items():
            pairs = scene_state[sid].get("pairs") or []
            ffp = scene_state[sid].get("frames_for_pairs") or []
            if not pairs:
                continue

            def _on_drop(pair, reason, _sid=sid):
                writer.reject(_sid, pair.src_id, pair.tgt_id, -1, reason)

            kept_pairs, kept_ffp = apply_filter_to_pairs(
                pairs, ffp, filter_spec,
                n_votes=f_votes, vote_temperature=f_vote_temp,
                on_drop=_on_drop,
            )
            n_dropped = len(pairs) - len(kept_pairs)
            if n_dropped:
                logger.info("[%s] filter dropped %d/%d pairs",
                            sid, n_dropped, len(pairs))
            scene_state[sid]["pairs"] = kept_pairs
            scene_state[sid]["frames_for_pairs"] = kept_ffp

    def _all_label_frames() -> list[FrameRef]:
        return [f for st in scene_state.values()
                for f in st.get("frames_for_pairs", [])]

    if collapse and need_filter_server:
        # One server lifetime spans filter (frames-in-pairs) +
        # labeler (reduced frames-in-pairs). Cheapest path.
        with launch_server(filter_spec, log_dir=args.run_log_dir) as endpoint:
            run_filter_pass(filter_spec, endpoint, all_pair_frames,
                            concurrency=concurrency,
                            n_votes=f_votes,
                            vote_temperature=f_vote_temp)
            _post_filter_drop_pairs()
            label_frames = _all_label_frames()
            if label_frames and not labeler_cache_complete(
                    labeler_spec, label_frames,
                    prompt_file=args.prompt_file,
                    n_votes=l_votes,
                    vote_temperature=l_vote_temp):
                run_labeler_pass(labeler_spec, endpoint, label_frames,
                                 concurrency=concurrency,
                                 prompt_file=args.prompt_file,
                                 n_votes=l_votes,
                                 vote_temperature=l_vote_temp)
    else:
        # Filter alone (if needed), then post-hoc pair drop, then labeler.
        if need_filter_server:
            with launch_server(filter_spec, log_dir=args.run_log_dir) as endpoint:
                run_filter_pass(filter_spec, endpoint, all_pair_frames,
                                concurrency=concurrency,
                                n_votes=f_votes,
                                vote_temperature=f_vote_temp)
        _post_filter_drop_pairs()
        if labeler_spec is not None:
            label_frames = _all_label_frames()
            if label_frames:
                stage_label(labeler_spec, label_frames,
                            concurrency=concurrency,
                            log_dir=args.run_log_dir,
                            prompt_file=args.prompt_file,
                            n_votes=l_votes,
                            vote_temperature=l_vote_temp)

    # Phase 4.5 — multi-GPU perception pre-pass (optional). Populates the
    # perception cache for every (scene, frame) tuple downstream Phase 5
    # would compute serially. Phase 5 then becomes a pure cache-read +
    # geometric-match loop.
    _maybe_run_perception_prepass(args, scene_state, filter_spec, labeler_spec)

    # Phase 5 — perception + match + emit per scene (no LLM server).
    cache_only_labeler = build_labeler(
        labeler_spec, endpoint=None, prompt_file=args.prompt_file,
        n_votes=int(args.n_votes or 1),
        vote_temperature=float(args.vote_temperature or 0.7),
    )
    detector = make_detector(
        args.detector, labeler=cache_only_labeler,
        gdino_max_classes=args.gdino_max_classes,
        labeler_concurrency=concurrency,
        vote_strategy=(args.vote_strategy or "union"),
    )

    total = 0
    for sid, adapter in adapters.items():
        pairs = scene_state[sid].get("pairs")
        if not pairs:
            logger.info("[%s] emitted 0 (no surviving pairs)", sid)
            continue
        n = _perception_emit_scene(
            adapter, args, writer, segmenter, detector, cache_root,
            pairs=pairs,
            frames_for_pairs=scene_state[sid]["frames_for_pairs"],
            content_skills=content_skills,
            manifest_writer=manifest_writer,
        )
        logger.info("[%s] emitted %d", sid, n)
        total += n

    # Phase 6 — verifier (optional, one server lifetime across all manifests).
    if verifier_spec is not None and isinstance(writer, TaskRouter):
        # Verifier consumes pair manifests, which are produced by
        # `manifest_writer` only when `out_root` mode is used (TaskRouter).
        from pipeline.stages import stage_verify
        stage_root = args.out_root / TaskRouter.STAGE
        # Flush manifest writer file handles so the collector sees every
        # row; PairManifestWriter flushes per-emit, so this is a no-op
        # safety net.
        if manifest_writer is not None:
            for fp in getattr(manifest_writer, "_fps", {}).values():
                try:
                    fp.flush()
                except Exception:
                    pass
        manifests = collect_pair_manifests(stage_root)
        if manifests:
            logger.info("[verifier:%s] collected %d manifests across %d scenes",
                        verifier_spec.name, len(manifests), len(adapters))
            verdicts = stage_verify(
                verifier_spec, manifests,
                concurrency=args.verify_concurrency,
                log_dir=args.run_log_dir,
            )
            kept = write_verified_per_skill(stage_root, manifests, verdicts)
            logger.info("[verifier:%s] kept per skill: %s",
                        verifier_spec.name, kept)
        else:
            logger.info("[verifier:%s] no manifests to verify",
                        verifier_spec.name)

    return total


# Per-stage config-key → args.<attr> translation tables. Most map 1:1;
# the few renames are preserved here so the CLI flag names users have
# muscle-memory for don't change.
_STAGE_TO_ARGS = {
    "sample": {
        "sampling": "sampling", "frame_stride": "frame_stride",
        "min_keyframes": "min_keyframes",
        "min_translation_m": "min_translation_m",
        "min_rotation_deg": "min_rotation_deg",
        "limit_frames": "limit_frames",
        "cosmic_base_sampling": "cosmic_base_sampling",
        "cosmic_union_coverage_min": "cosmic_union_coverage_min",
        "cosmic_yaw_diff_min_deg": "cosmic_yaw_diff_min_deg",
    },
    "pair_gate": {
        "sampling": "sampling", "frame_stride": "frame_stride",
        "min_keyframes": "min_keyframes",
        "min_translation_m": "min_translation_m",
        "min_rotation_deg": "min_rotation_deg",
        "limit_frames": "limit_frames",
        "cosmic_base_sampling": "cosmic_base_sampling",
        "cosmic_union_coverage_min": "cosmic_union_coverage_min",
        "cosmic_yaw_diff_min_deg": "cosmic_yaw_diff_min_deg",
        "cosmic_obj_vis_area_min": "cosmic_obj_vis_area_min",
        "cosmic_obj_vis_depth_pix_min": "cosmic_obj_vis_depth_pix_min",
    },
    "filter":  {"model": "quality_filter",
                "vllm_concurrency": "vllm_concurrency",
                "n_votes": "filter_n_votes",
                "vote_temperature": "filter_vote_temperature"},
    "label":   {"model": "labeler", "prompt_file": "prompt_file",
                "vllm_concurrency": "vllm_concurrency",
                "n_votes": "n_votes",
                "vote_temperature": "vote_temperature"},
    "perceive": {"detector": "detector", "segmenter": "segmenter",
                 "gdino_max_classes": "gdino_max_classes",
                 "labeler": "labeler", "prompt_file": "prompt_file",
                 "cache_root": "cache_root",
                 "workers": "perception_workers",
                 "batch_frames": "perception_batch_frames",
                 "prepass_min_frames": "perception_prepass_min_frames",
                 "compile_perception": "compile_perception",
                 "n_votes": "n_votes",
                 "vote_temperature": "vote_temperature",
                 "vote_strategy": "vote_strategy"},
    "match": {"detector": "detector", "segmenter": "segmenter",
              "gdino_max_classes": "gdino_max_classes",
              "labeler": "labeler", "prompt_file": "prompt_file",
              "cache_root": "cache_root",
              "seed": "seed", "seed_retries": "seed_retries",
              "depth_tol": "depth_tol", "iou_min": "iou_min",
              "voxel_dedup": "voxel_dedup",
              "emit_occlusion_negatives": "emit_occlusion_negatives",
              "max_samples_per_scene": "max_samples_per_scene",
              "max_det_per_frame": "max_det_per_frame",
              "viz_num": "viz_num", "wandb_project": "wandb_project",
              "wandb_run_name": "wandb_run_name",
              "wandb_max_rows": "wandb_max_rows"},
    # Verifier is intentionally NOT mapped from configs/stages/verify.json:
    # that stage config is for `cli verify` standalone, where the user
    # explicitly invokes verification. For `cli generate`, the verifier is
    # off by default and only enabled via --verifier or a run preset's
    # top-level "verifier" key.
    "verify": {"verify_concurrency": "verify_concurrency"},
}

# Apply order matters when keys collide (last-write wins): perceive ←→
# match overlap on detector/segmenter/labeler/prompt_file/cache_root —
# they should be identical in any sane preset; if not, match's value
# (the downstream consumer) wins.
_STAGE_APPLY_ORDER = ("sample", "pair_gate", "filter", "label",
                       "perceive", "match", "verify")


def _apply_run_config(args: argparse.Namespace) -> None:
    """Resolve per-stage configs and copy onto ``args`` (CLI flags win).

    Preference order:
      1. Existing CLI value on ``args`` (non-None)  — wins absolutely.
      2. ``--run-config`` preset's per-stage dict (deep-merged with
         ``stage_overrides``) when ``--run-config`` is set.
      3. Built-in ``configs/stages/<stage>.json`` defaults otherwise.
    """
    if args.run_config is not None:
        preset = load_run_config(args.run_config)
        per_stage = preset.stages
        # Top-level run knobs.
        if args.adapter is None:
            args.adapter = preset.adapter
        if args.scenes_root is None:
            args.scenes_root = preset.scenes_root
        if args.out_root is None and args.out is None:
            args.out_root = preset.out_root
        if args.logs_dir is None:
            args.logs_dir = preset.logs_dir
        # Top-level overrides for filter/labeler/verifier model picks
        # (preset.extras carries unknown top-level keys).
        for k_top, k_args in (("filter", "quality_filter"),
                              ("labeler", "labeler"),
                              ("verifier", "verifier")):
            if getattr(args, k_args, None) is None and k_top in preset.extras:
                setattr(args, k_args, preset.extras[k_top])
    else:
        per_stage = {s: load_stage_config(s) for s in STAGE_NAMES}

    # Final fallbacks for top-level run knobs (when no --run-config).
    if args.adapter is None:
        args.adapter = "scannet"
    if args.scenes_root is None:
        args.scenes_root = Path(
            "/home/mila/l/leh/scratch/dataset/scannet_data/scans")
    if args.out_root is None and args.out is None:
        args.out_root = Path("outputs/run")
    if args.logs_dir is None:
        args.logs_dir = Path("logs")

    # Per-stage knob copy: only touches args when the CLI didn't supply.
    for stage in _STAGE_APPLY_ORDER:
        cfg = per_stage.get(stage, {})
        for cfg_key, attr in _STAGE_TO_ARGS[stage].items():
            if cfg_key not in cfg:
                continue
            if getattr(args, attr, None) is not None:
                continue  # CLI wins.
            setattr(args, attr, cfg[cfg_key])

    # Path coercion for keys that JSON delivers as str.
    for path_attr in ("scenes_root", "out_root", "logs_dir", "cache_root",
                       "prompt_file"):
        v = getattr(args, path_attr, None)
        if isinstance(v, str):
            setattr(args, path_attr, Path(v))

    # Final back-stops for keys still None: fields with no per-stage entry
    # (verifier choice) or those gated by top-level run knobs.
    if args.verifier is None:
        args.verifier = "none"


def main() -> None:
    p = argparse.ArgumentParser()
    # Top-level run knobs (always have defaults — these don't migrate to
    # per-stage configs because they're orchestrator-level).
    p.add_argument("--run-config", type=Path, default=None,
                   help="Run preset JSON (e.g. configs/runs/qwen3vl_default.json) "
                        "that picks per-stage configs and applies "
                        "stage_overrides. Without --run-config, each stage's "
                        "default configs/stages/<stage>.json supplies its "
                        "knobs. Any explicit CLI flag overrides both.")
    p.add_argument("--adapter", default=None)
    p.add_argument("--scenes-root", type=Path, default=None)
    p.add_argument("--scene", action="append",
                   help="explicit scene id (repeatable). Overrides --all-scenes.")
    p.add_argument("--all-scenes", action="store_true")
    p.add_argument("--limit-scenes", type=int, default=None)
    # All knob flags below default to None: precedence is CLI > run-config /
    # stage config > KeyError (fail-loud) for missing keys.
    p.add_argument("--detector", default=None,
                   choices=["noop", "gdino", "gdino+scannet200",
                            "labeled-gdino", "gemini+gdino",
                            "scannet-gt",
                            "scannet-gt-label+gdino"])
    p.add_argument("--gdino-max-classes", type=int, default=None)
    from models.registry import MODELS
    _MODEL_CHOICES = sorted(MODELS)
    p.add_argument("--labeler", default=None, choices=_MODEL_CHOICES,
                   help="Registry name of the labeler model (used by "
                        "--detector labeled-gdino).")
    p.add_argument("--quality-filter", dest="quality_filter", default=None,
                   choices=["none"] + _MODEL_CHOICES,
                   help="Registry name of the quality-filter model, or 'none' "
                        "to skip filtering.")
    p.add_argument("--prompt-file", dest="prompt_file", type=Path, default=None,
                   help="Path to a .txt or .json file with the labeler prompt. "
                        "Falls back to configs/label_prompt.txt.")
    p.add_argument("--label-votes", dest="n_votes", type=int, default=None,
                   help="Per-frame inference passes for the labeler (default 1). "
                        ">1 enables multi-vote labeling: every run is persisted "
                        "under cache/labels/<spec>__vote{N}/. Aggregation is "
                        "controlled by --label-vote-strategy.")
    p.add_argument("--label-vote-temperature", dest="vote_temperature",
                   type=float, default=None,
                   help="Sampling temperature when n_votes>1 (default 0.7).")
    p.add_argument("--label-vote-strategy", dest="vote_strategy", default=None,
                   choices=["union", "majority", "per-run-detect"],
                   help="How `--detector labeled-gdino` consumes the N runs: "
                        "union of canonicals (default), majority across runs, "
                        "or run GDino once per run and majority-vote on "
                        "detections (heaviest, most faithful to per-run noise).")
    p.add_argument("--filter-votes", dest="filter_n_votes", type=int,
                   default=None,
                   help="Per-frame inference passes for the quality filter "
                        "(default 1). >1 majority-votes on usable; cache "
                        "lands at cache/filter/<spec>__voteN/.")
    p.add_argument("--filter-vote-temperature", dest="filter_vote_temperature",
                   type=float, default=None,
                   help="Sampling temperature for the filter when n_votes>1 "
                        "(default 0.7).")
    p.add_argument("--segmenter", default=None,
                   choices=["noop", "sam2.1", "sam3", "gt-mask"])
    p.add_argument("--vllm-concurrency", type=int, default=None,
                   help="Concurrent HTTP requests to the vLLM server (filter "
                        "and labeler).")
    p.add_argument("--verifier", default=None,
                   choices=["none"] + _MODEL_CHOICES,
                   help="If set, run the pair verifier as a final stage. "
                        "'none' = skip verification.")
    p.add_argument("--verify-concurrency", type=int, default=None,
                   help="ThreadPool size for concurrent verifier requests.")
    p.add_argument("--sampling", choices=["adaptive", "stride", "cosmic"],
                   default=None,
                   help="stride = fixed every-Nth frame (default, less "
                        "near-duplicate clustering than adaptive); "
                        "adaptive = pose-based redundancy filter "
                        "(paper-faithful, avg ~1/50, but pose-signature "
                        "clustering can starve diversity prune); "
                        "cosmic = visibility-set gate from COSMIC "
                        "(arXiv 2603.27183) layered on a base sampler. "
                        "Cosmic restricts pair.tasks to object-level skills "
                        "(cross_object_correspondence, anchor, counting, "
                        "relative_distance, relative_direction).")
    p.add_argument("--frame-stride", type=int, default=None, metavar="RATIO")
    p.add_argument("--min-keyframes", type=int, default=None)
    p.add_argument("--min-translation-m", type=float, default=None)
    p.add_argument("--min-rotation-deg", type=float, default=None)
    p.add_argument("--cosmic-base-sampling", choices=["adaptive", "stride"],
                   default=None)
    p.add_argument("--cosmic-union-coverage-min", type=float, default=None)
    p.add_argument("--cosmic-yaw-diff-min-deg", type=float, default=None)
    p.add_argument("--cosmic-obj-vis-area-min", type=float, default=None)
    p.add_argument("--cosmic-obj-vis-depth-pix-min", type=int, default=None)
    p.add_argument("--max-det-per-frame", type=int, default=None)
    p.add_argument("--seed-retries", type=int, default=None)
    p.add_argument("--depth-tol", type=float, default=None)
    p.add_argument("--iou-min", type=float, default=None)
    p.add_argument("--voxel-dedup", type=float, default=None)
    p.add_argument("--emit-occlusion-negatives",
                   dest="emit_occlusion_negatives",
                   action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--max-samples-per-scene", type=int, default=None)
    p.add_argument("--limit-frames", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-root", type=Path, default=None,
                   help="root folder; pipeline creates one subfolder per skill "
                        "under stage_1/. Default comes from --run-config "
                        "or 'outputs/run'.")
    p.add_argument("--out", type=Path, default=None,
                   help="legacy single-file output. If set, overrides --out-root.")
    p.add_argument("--cache-root", type=Path, default=None)
    p.add_argument("--perception-workers", dest="perception_workers",
                   type=int, default=None)
    p.add_argument("--perception-batch-frames", dest="perception_batch_frames",
                   type=int, default=None)
    p.add_argument("--perception-prepass-min-frames",
                   dest="perception_prepass_min_frames",
                   type=int, default=None)
    p.add_argument("--compile-perception", dest="compile_perception",
                   action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--logs-dir", type=Path, default=None,
                   help="Per-run log artifacts land at <logs-dir>/<run-id>/.")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--viz-num", type=int, default=None)
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-max-rows", type=int, default=None)
    args = p.parse_args()

    # ── resolve config: --run-config (preset) or per-stage defaults ──
    _apply_run_config(args)

    # Per-run log directory: keeps pipeline.log + per-model vllm_*.log
    # together so a run can be monitored / archived as a unit.
    if args.run_id is None:
        from datetime import datetime
        out_basename = (args.out.stem if args.out is not None
                        else args.out_root.name)
        args.run_id = f"{out_basename}__{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    args.run_log_dir = (args.logs_dir / args.run_id).resolve()
    args.run_log_dir.mkdir(parents=True, exist_ok=True)

    # Root logger: stderr (existing behavior) + per-run file. Both share the
    # same format so `tail -f logs/<run-id>/pipeline.log` mirrors what the
    # console shows.
    log_format = "%(asctime)s %(name)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    file_handler = logging.FileHandler(args.run_log_dir / "pipeline.log")
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)
    logger.info("run_id=%s  logs at %s", args.run_id, args.run_log_dir)

    if args.scene:
        scene_ids = args.scene
    elif args.all_scenes:
        scene_ids = sorted(d.name for d in args.scenes_root.iterdir() if d.is_dir())
    else:
        print("Specify --scene <id> (repeatable) or --all-scenes", file=sys.stderr)
        sys.exit(2)
    if args.limit_scenes is not None:
        scene_ids = scene_ids[: args.limit_scenes]

    task_config = load_skills_config()
    content_skills = load_content_skills(task_config)
    logger.info("loaded skills config (content skills: %s)",
                sorted(content_skills))

    from models.registry import resolve as resolve_model

    # Resolve registry names → ModelSpec. The orchestrator inside
    # `process_scenes` decides when (or whether) to launch each model's
    # vLLM server based on what's already cached.
    needs_labeler = args.detector in ("labeled-gdino", "gemini+gdino")
    labeler_spec = resolve_model(args.labeler) if needs_labeler else None
    filter_spec = (resolve_model(args.quality_filter)
                   if args.quality_filter != "none" else None)
    if filter_spec is not None and filter_spec.backend != "vllm":
        raise SystemExit(
            f"--quality-filter {args.quality_filter!r} resolves to backend "
            f"{filter_spec.backend!r}, which has no QwenFilter implementation. "
            f"Pick a vllm-backend model (qwen3vl-*)."
        )
    verifier_spec = (resolve_model(args.verifier)
                     if args.verifier != "none" else None)
    if verifier_spec is not None:
        if not verifier_spec.is_vllm:
            raise SystemExit(
                f"--verifier {args.verifier!r} is not a vllm-backend spec."
            )
        if verifier_spec.images_per_prompt < 2:
            raise SystemExit(
                f"--verifier {args.verifier!r} has images_per_prompt="
                f"{verifier_spec.images_per_prompt}; need >=2 for pair "
                f"verification (qwen3vl-8B-pair is the operational default)."
            )
        if args.out is not None:
            # Verifier consumes per-skill pair manifests, only produced in
            # TaskRouter (--out-root) mode. Refuse loudly rather than
            # silently dropping the verifier stage.
            raise SystemExit(
                "--verifier requires --out-root mode (multi-skill pair "
                "manifests). Drop --out, or run `cli verify` separately "
                "against a pairs.jsonl."
            )

    segmenter = make_segmenter(args.segmenter)

    t0 = time.time()
    total = 0
    if args.out is not None:
        writer_cm = CorrespondenceWriter(args.out, resume=args.resume)
        manifest_writer = None
    else:
        writer_cm = TaskRouter(args.out_root, resume=args.resume)
        manifest_writer = PairManifestWriter(
            args.out_root / TaskRouter.STAGE,
            skills=[*POSE_SKILLS, *CONTENT_SKILLS],
            resume=args.resume,
        )
    with writer_cm as writer:
        try:
            total = process_scenes(
                scene_ids, args, writer, args.cache_root, segmenter,
                task_config=task_config,
                filter_spec=filter_spec,
                labeler_spec=labeler_spec,
                verifier_spec=verifier_spec,
                content_skills=content_skills,
                manifest_writer=manifest_writer,
            )
            counts = writer.counts()
        finally:
            if manifest_writer is not None:
                counts_m = manifest_writer.counts()
                manifest_writer.close()
                logger.info("pair manifests: %s", counts_m)

    dt = time.time() - t0
    logger.info("DONE: %d records, %.1fs, counts=%s", total, dt, counts)

    if args.viz_num > 0 and args.out is None:
        _render_viz(args, args.out_root, args.viz_num)

    if args.wandb_project and args.out is None:
        from pipeline.wandb_uploader import upload_run
        # Path objects don't serialize cleanly into wandb.config — stringify.
        cfg = {k: (str(v) if isinstance(v, Path) else v)
               for k, v in vars(args).items()}
        upload_run(
            args.out_root,
            project=args.wandb_project,
            run_name=args.wandb_run_name,
            config=cfg,
            max_table_rows=args.wandb_max_rows,
        )


def _render_viz(args, out_root: Path, num: int) -> None:
    """Render viz under <out_root>/stage_1/:

       <task>/inspect/                per-pair correspondence PNGs (limit=`num`)
       perception/<scene>.png         every scene's GDino+SAM detections (`num` frames)
       pairs/<scene>_<src>_<tgt>.png  every unique pair's geometric match breakdown

    Calls the project's viz dispatcher (``python -m viz --mode <name>``)
    as a subprocess so any failure in matplotlib / cache loading is logged
    but doesn't abort the run.
    """
    import subprocess

    stage = out_root / TaskRouter.STAGE

    def _run(cmd, target):
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("viz -> %s", target)
        except subprocess.CalledProcessError as e:
            logger.warning("viz failed for %s: %s",
                           target, e.stderr.strip()[-300:])

    model_tag = _model_tag(args)
    # Viz `--cache-root` is the parent of `perception/` (modes append
    # the namespace themselves). `args.cache_root` here points at the
    # perception namespace (legacy default `cache/perception`), so pass
    # its parent to the subprocesses.
    viz_cache_root = args.cache_root.parent

    # 1. Per-task correspondence PNGs.
    for sub in sorted(stage.iterdir()):
        if not sub.is_dir():
            continue
        for kind in ("pos", "neg"):
            jsonl = sub / f"correspondences.{kind}.jsonl"
            if not jsonl.exists() or jsonl.stat().st_size == 0:
                continue
            out_dir = sub / f"inspect.{kind}"
            _run([sys.executable, "-m", "viz", "--mode", "correspondences",
                  "--jsonl", str(jsonl), "--limit", str(num),
                  "--out-dir", str(out_dir),
                  "--cache-root", str(viz_cache_root),
                  "--adapter", args.adapter,
                  "--model-tag", model_tag],
                 target=out_dir)

    # 2. Perception viz (one per scene) + pair-match viz (one per UNIQUE pair).
    all_jsonl = stage / "_all" / "correspondences.jsonl"
    if not all_jsonl.exists() or all_jsonl.stat().st_size == 0:
        return
    scenes_seen, pair_keys = set(), []
    with open(all_jsonl) as f:
        for line in f:
            r = json.loads(line)
            scenes_seen.add(r["scene_id"])
            k = (r["scene_id"], r["frame_src"], r["frame_tgt"])
            if k not in pair_keys:
                pair_keys.append(k)

    perception_dir = stage / "perception"
    pairs_dir = stage / "pairs"
    perception_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    for sid in sorted(scenes_seen):
        png = perception_dir / f"{sid}.png"
        _run([sys.executable, "-m", "viz", "--mode", "perception",
              "--scenes-root", str(args.scenes_root),
              "--cache-root", str(viz_cache_root),
              "--adapter", args.adapter, "--model-tag", model_tag,
              "--scene", sid, "--num", "12", "--save", str(png)],
             target=png)
    for sid, fsrc, ftgt in pair_keys:
        png = pairs_dir / f"{sid}_{fsrc}_{ftgt}.png"
        _run([sys.executable, "-m", "viz", "--mode", "pair_match",
              "--scenes-root", str(args.scenes_root),
              "--cache-root", str(viz_cache_root),
              "--adapter", args.adapter,
              "--scene", sid, "--src", fsrc, "--tgt", ftgt,
              "--save", str(png)],
             target=png)


if __name__ == "__main__":
    main()
