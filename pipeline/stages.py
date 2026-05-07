"""Per-stage runners — pure functions, no argparse.

Each ``stage_*`` either:

- Reads on-disk caches and returns instantly (cache-only fast path), OR
- Launches a vLLM server, fans out concurrent requests, kills the server.

The orchestrator in ``cli/generate.py::process_scenes`` chains the
filter/labeler stages, collapsing onto one server lifetime when the
same spec is used for both. Stand-alone CLIs in ``cli/{filter,label,
verify,sample}.py`` call individual stages.

Concurrency model: ``ThreadPoolExecutor`` with ``as_completed`` for
streaming progress. ``_VLMBase.warmup`` is invoked before fan-out so
the first burst of requests doesn't all stall on the same vision-encoder
JIT compile (which then cascades into retry storms).

Server lifecycle: one model loaded at a time — ``launch_server``
context managers do not nest. The orchestrator can choose to share one
server across two stages (filter + labeler when both reference the
same spec) by calling the lower-level ``run_*_pass`` helpers directly
inside its own ``launch_server`` block.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, TypeVar

from models._frame_ref import FrameRef
from models.registry import ModelSpec, launch_server

logger = logging.getLogger(__name__)


T = TypeVar("T")
R = TypeVar("R")


# ---- builders ----------------------------------------------------------

def build_labeler(spec: Optional[ModelSpec],
                  endpoint: Optional[str] = None,
                  prompt_file: Optional[Path] = None,
                  *,
                  n_votes: int = 1,
                  vote_temperature: float = 0.7):
    """Construct a labeler from a registry spec.

    ``endpoint`` is the live vLLM URL for vllm-backend specs; pass
    ``None`` when only cache reads are expected (after the server has
    been killed). Gemini specs are server-less; ``endpoint`` is ignored.

    ``n_votes>1`` requests N inference passes per frame (vllm only) and
    persists every parsed run under a separate ``cache/labels/<spec>__voteN``
    dir so the perception stage can choose its own aggregation strategy.
    """
    if spec is None:
        return None
    if spec.backend == "vllm":
        from models.labelers.qwen3vl import Qwen3VLLabeler
        return Qwen3VLLabeler(spec, endpoint=endpoint, prompt_file=prompt_file,
                              n_votes=n_votes,
                              vote_temperature=vote_temperature)
    if spec.backend == "gemini":
        if n_votes > 1:
            raise ValueError(
                "Multi-vote labeling is implemented only for the vllm "
                "backend (Qwen3VLLabeler) today. Use --labeler qwen3vl-* "
                "or set n_votes=1."
            )
        from models.labelers.gemini import GeminiLabeler
        return GeminiLabeler(spec, prompt_file=prompt_file)
    raise ValueError(f"Unknown labeler backend {spec.backend!r}")


def build_filter(spec: Optional[ModelSpec],
                 endpoint: Optional[str] = None,
                 *,
                 n_votes: int = 1,
                 vote_temperature: float = 0.7,
                 vote_threshold: Optional[int] = None):
    """Construct a quality filter from a registry spec. vllm-only today.

    ``n_votes>1`` requests N inference passes per frame; the cache stores
    every run plus a precomputed majority verdict
    (``cache/filter/<spec>__voteN/<frame>.json``). Default threshold is
    ``ceil(N/2)``.
    """
    if spec is None:
        return None
    if spec.backend == "vllm":
        from models.filters.qwen import QwenFilter
        return QwenFilter(spec, endpoint=endpoint,
                          n_votes=n_votes,
                          vote_temperature=vote_temperature,
                          vote_threshold=vote_threshold)
    raise ValueError(
        f"Filter backend {spec.backend!r} not supported (only vllm)"
    )


def build_verifier(spec: Optional[ModelSpec],
                   endpoint: Optional[str] = None):
    """Construct a pair verifier from a registry spec. vllm-only."""
    if spec is None:
        return None
    if spec.backend == "vllm":
        from models.verifiers.qwen_pair import QwenPairVerifier
        return QwenPairVerifier(spec, endpoint=endpoint)
    raise ValueError(
        f"Verifier backend {spec.backend!r} not supported (only vllm)"
    )


# ---- cache-completeness probes ---------------------------------------

def filter_cache_complete(spec: Optional[ModelSpec],
                          frames: list[FrameRef],
                          *,
                          n_votes: int = 1,
                          vote_temperature: float = 0.7) -> bool:
    """True iff every FrameRef has a model-tagged filter cache entry.

    Multi-vote shape (``n_votes>1``) is checked under the
    ``cache/filter/<spec>__voteN/`` sibling so a 1-vote cache doesn't
    satisfy a 3-vote query and vice versa.
    """
    if spec is None:
        return True
    flt = build_filter(spec, endpoint=None,
                      n_votes=n_votes,
                      vote_temperature=vote_temperature)
    return all(flt._cache_path(f).exists() for f in frames)


def labeler_cache_complete(spec: Optional[ModelSpec],
                           frames: list[FrameRef],
                           prompt_file: Optional[Path] = None,
                           *,
                           n_votes: int = 1,
                           vote_temperature: float = 0.7) -> bool:
    """True iff every FrameRef has a *valid* labeler cache entry.

    A cache entry exists but is not "valid" when a previous run wrote
    ``valid=False`` after exhausting retries. Treat that as incomplete
    so the orchestrator re-launches the server and tries again.

    Multi-vote shape (``n_votes>1``) is detected automatically — the
    cache is considered complete only when every frame has a valid
    multi-run entry under ``cache/labels/<spec>__voteN/``.
    """
    if spec is None:
        return True
    lab = build_labeler(spec, endpoint=None, prompt_file=prompt_file,
                        n_votes=n_votes, vote_temperature=vote_temperature)
    for f in frames:
        cp = lab._cache_path(f)
        if not cp.exists():
            return False
        try:
            d = json.loads(cp.read_text())
            if not d.get("valid"):
                return False
        except (OSError, json.JSONDecodeError):
            return False
    return True


def verifier_cache_complete(spec: Optional[ModelSpec],
                            manifests: list[dict]) -> bool:
    """True iff every manifest has a verifier cache entry."""
    if spec is None or not manifests:
        return True
    ver = build_verifier(spec, endpoint=None)
    for m in manifests:
        cp = ver._pair_cache_path(
            m["skill"], m["scene_id"],
            m["frame_src"], m["frame_tgt"],
            m.get("evidence", {}),
        )
        if not cp.exists():
            return False
    return True


# ---- fan-out -----------------------------------------------------------

def _fan_out(
    fn: Callable[[T], R],
    items: list[T],
    workers: int,
    *,
    label: str,
    progress_every: int = 50,
) -> list[Optional[R]]:
    """Run ``fn`` over ``items`` with up to ``workers`` threads.

    Uses ``as_completed`` for streaming progress logs every
    ``progress_every`` items, but returns results in **input order**.
    Per-item exceptions are logged at WARNING and recorded as ``None``
    so a single bad input doesn't crash the whole batch — callers that
    need fail-fast behavior should re-raise on ``None``.
    """
    if not items:
        return []
    if workers <= 1 or len(items) == 1:
        return [_safe_call(fn, it, i, label) for i, it in enumerate(items)]
    n = len(items)
    workers = min(workers, n)
    results: dict[int, Optional[R]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fn, it): i for i, it in enumerate(items)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                logger.warning("[%s] item %d raised: %s", label, i, e)
                results[i] = None
            done = len(results)
            if done % progress_every == 0 or done == n:
                logger.info("[%s] progress %d/%d", label, done, n)
    return [results[i] for i in range(n)]


def _safe_call(fn, item, i: int, label: str):
    try:
        return fn(item)
    except Exception as e:
        logger.warning("[%s] item %d raised: %s", label, i, e)
        return None


# ---- low-level pass helpers (callable inside an existing server ctx) ---

def run_filter_pass(spec: ModelSpec, endpoint: Optional[str],
                    frames: list[FrameRef],
                    concurrency: Optional[int] = None,
                    *,
                    n_votes: int = 1,
                    vote_temperature: float = 0.7) -> None:
    """Populate the filter cache for ``frames`` against an existing
    server endpoint. The caller owns the ``launch_server`` lifetime
    (used by ``process_scenes`` when collapsing filter + labeler across
    every scene under one server lifetime)."""
    if not frames:
        return
    flt = build_filter(spec, endpoint=endpoint,
                      n_votes=n_votes,
                      vote_temperature=vote_temperature)
    if endpoint is not None:
        flt.warmup()
    workers = concurrency if concurrency is not None else spec.recommended_concurrency
    _fan_out(flt.is_usable, frames, workers,
             label=f"filter:{spec.name}")


def run_labeler_pass(spec: ModelSpec, endpoint: Optional[str],
                     frames: list[FrameRef],
                     concurrency: Optional[int] = None,
                     prompt_file: Optional[Path] = None,
                     *,
                     n_votes: int = 1,
                     vote_temperature: float = 0.7) -> None:
    """Populate the labeler cache for ``frames`` against an existing
    server endpoint."""
    if not frames:
        return
    lab = build_labeler(spec, endpoint=endpoint, prompt_file=prompt_file,
                        n_votes=n_votes,
                        vote_temperature=vote_temperature)
    if endpoint is not None:
        lab.warmup()
    workers = concurrency if concurrency is not None else spec.recommended_concurrency
    _fan_out(lab.label_with_canonical, frames, workers,
             label=f"labeler:{spec.name}")


def run_verifier_pass(spec: ModelSpec, endpoint: Optional[str],
                      manifests: list[dict],
                      concurrency: Optional[int] = None
                      ) -> list[Optional[tuple[bool, str]]]:
    """Run verifier across ``manifests``. Returns verdicts in input
    order; ``None`` slots correspond to manifests that raised. Cache
    writes happen inside ``verify()`` per-manifest, so partial progress
    is preserved across crashes."""
    if not manifests:
        return []
    ver = build_verifier(spec, endpoint=endpoint)
    if endpoint is not None:
        ver.warmup()
    workers = concurrency if concurrency is not None else spec.recommended_concurrency
    return _fan_out(ver.verify, manifests, workers,
                    label=f"verifier:{spec.name}")


# ---- self-contained stages (own server lifetime) ---------------------

def stage_filter(spec: Optional[ModelSpec], frames: list[FrameRef], *,
                 concurrency: Optional[int] = None,
                 log_dir: Optional[Path] = None,
                 n_votes: int = 1,
                 vote_temperature: float = 0.7) -> None:
    """Run the filter stage end-to-end (server up → fan-out → server down).

    Cache-only fast path: if every FrameRef already has an entry, return
    without launching the server.
    """
    if spec is None or not frames:
        return
    if filter_cache_complete(spec, frames,
                             n_votes=n_votes,
                             vote_temperature=vote_temperature):
        logger.info("[filter:%s] cache complete (%d frames) — skip server",
                    spec.name, len(frames))
        return
    with launch_server(spec, log_dir=log_dir) as endpoint:
        run_filter_pass(spec, endpoint, frames, concurrency=concurrency,
                        n_votes=n_votes,
                        vote_temperature=vote_temperature)


def stage_label(spec: Optional[ModelSpec], frames: list[FrameRef], *,
                concurrency: Optional[int] = None,
                log_dir: Optional[Path] = None,
                prompt_file: Optional[Path] = None,
                n_votes: int = 1,
                vote_temperature: float = 0.7) -> None:
    """Run the labeler stage end-to-end."""
    if spec is None or not frames:
        return
    if labeler_cache_complete(spec, frames, prompt_file=prompt_file,
                              n_votes=n_votes,
                              vote_temperature=vote_temperature):
        logger.info("[labeler:%s] cache complete (%d frames) — skip server",
                    spec.name, len(frames))
        return
    with launch_server(spec, log_dir=log_dir) as endpoint:
        run_labeler_pass(spec, endpoint, frames,
                         concurrency=concurrency, prompt_file=prompt_file,
                         n_votes=n_votes,
                         vote_temperature=vote_temperature)


_VERIFIER_REQUIRED_KEYS = (
    "skill", "scene_id", "frame_src", "frame_tgt",
    "image_src", "image_tgt",
)


def collect_pair_manifests(stage_root: Path,
                           skills: Optional[Iterable[str]] = None
                           ) -> list[dict]:
    """Read every ``<skill>/pairs.jsonl`` under ``stage_root``.

    Returns a flat list of manifest dicts that contain every key
    ``stage_verify`` needs. Malformed lines and rows missing required
    keys are skipped (logged at WARNING). Useful for the verifier stage,
    which runs once across every skill in a single server lifetime.

    ``skills`` (optional): restrict to a subset of skill subdirs. ``None``
    iterates every immediate subdir of ``stage_root``.
    """
    if not stage_root.exists():
        return []
    skills_set = set(skills) if skills is not None else None
    out: list[dict] = []
    for skill_dir in sorted(p for p in stage_root.iterdir() if p.is_dir()):
        if skills_set is not None and skill_dir.name not in skills_set:
            continue
        p = skill_dir / "pairs.jsonl"
        if not p.exists() or p.stat().st_size == 0:
            continue
        skipped_malformed = 0
        skipped_missing_keys = 0
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    skipped_malformed += 1
                    continue
                if not all(k in m for k in _VERIFIER_REQUIRED_KEYS):
                    skipped_missing_keys += 1
                    continue
                out.append(m)
        if skipped_malformed or skipped_missing_keys:
            logger.warning(
                "[verifier-collect] %s: %d malformed, %d missing-key rows skipped",
                p, skipped_malformed, skipped_missing_keys,
            )
    return out


def write_verified_per_skill(
    stage_root: Path,
    manifests: list[dict],
    verdicts: list[Optional[tuple[bool, str]]],
) -> dict[str, int]:
    """Group ``(manifest, verdict)`` by skill and write
    ``<skill>/pairs.verified.jsonl`` containing only the rows where the
    verdict is ``(True, _)``.

    Each output file is **truncated** on every call — verdicts come from
    the (idempotent) verifier cache, so the file is reproducible from
    scratch. Returns ``{skill: kept_count}``. Manifests with ``verdict
    is None`` (verifier raised / failed) are excluded — the verifier's
    own retry-exhausted handling already cached those as ``usable=False``,
    so a None here only happens if the verifier wasn't run for this row.
    """
    from collections import defaultdict
    per_skill: dict[str, list[dict]] = defaultdict(list)
    for m, v in zip(manifests, verdicts):
        if v is None:
            continue
        usable, _reason = v
        if usable:
            per_skill[m["skill"]].append(m)

    counts: dict[str, int] = {}
    for skill, rows in per_skill.items():
        out = stage_root / skill / "pairs.verified.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for m in rows:
                f.write(json.dumps(m) + "\n")
        counts[skill] = len(rows)
    return counts


def stage_verify(spec: Optional[ModelSpec], manifests: list[dict], *,
                 concurrency: Optional[int] = None,
                 log_dir: Optional[Path] = None
                 ) -> list[Optional[tuple[bool, str]]]:
    """Run the verifier stage end-to-end. Returns verdicts in input order.

    Cache-only fast path: if every manifest already has a verdict, the
    server is not launched — verdicts are read from cache.
    """
    if spec is None or not manifests:
        return [None] * len(manifests)
    if verifier_cache_complete(spec, manifests):
        logger.info("[verifier:%s] cache complete (%d manifests) — skip server",
                    spec.name, len(manifests))
        ver = build_verifier(spec, endpoint=None)
        return _fan_out(ver.verify, manifests,
                        concurrency or spec.recommended_concurrency,
                        label=f"verifier:{spec.name}:cache")
    with launch_server(spec, log_dir=log_dir) as endpoint:
        return run_verifier_pass(spec, endpoint, manifests,
                                 concurrency=concurrency)


# ---- non-VLM stages (pair-gate / perceive / match) ----------------------
#
# These are pure orchestration helpers — no vLLM server lifecycle. They
# take pre-built adapter / detector / segmenter / writers so this module
# stays free of cli.generate imports (cli.generate already depends on
# pipeline.stages; the reverse would be a cycle).

def perception_cache_complete(cache_root: Path, adapter_name: str,
                              scene_id: str, model_tag: str,
                              frames: list[FrameRef]) -> bool:
    """True iff every frame already has a perception ``.pkl`` cache entry.

    Uses the same key the serial Phase 5 path uses (``<frame_id>.pkl``,
    no canon-suffix), matching what ``_maybe_run_perception_prepass``
    pre-computes.
    """
    base = Path(cache_root) / adapter_name / scene_id / model_tag
    if not base.exists():
        return False
    return all((base / f"{f.frame_id}.pkl").exists() for f in frames)


def stage_pair_gate(adapter, *, pair_config, sampling: str,
                    frame_stride: int, min_keyframes: int,
                    min_translation_m: float, min_rotation_deg: float,
                    limit_frames: int,
                    cosmic_base_sampling: str,
                    cosmic_union_coverage_min: float,
                    cosmic_yaw_diff_min_deg: float,
                    cosmic_obj_vis_area_min: float,
                    cosmic_obj_vis_depth_pix_min: int,
                    adapter_name: str,
                    quality_filter=None,
                    quality_filter_concurrency: int = 1,
                    on_filter_drop=None,
                    on_pair_reject=None
                    ) -> tuple[list, list[FrameRef]]:
    """Run Phase 3 (pair-gate) for one (adapter, scene). No GPU, no server.

    Returns ``(pairs, frames_for_pairs)`` where ``pairs`` is the list of
    surviving ``ViewPair``s with skill tags assigned, and
    ``frames_for_pairs`` is the unique-frame ``FrameRef`` list referenced
    by those pairs (sorted by frame_id).

    ``quality_filter`` is the same callable shape ``select_pairs``
    expects (``(FrameRef) -> (bool, str)``) — typically built from a
    cache-only ``QwenFilter.is_usable`` so the filter cache populated by
    Phase 2 is consulted without launching a server.

    Callbacks are pass-throughs to ``select_pairs``; pass ``None`` to
    no-op them.
    """
    from .pairs import select_pairs

    pairs = select_pairs(
        adapter, pair_config,
        adapter_name=adapter_name,
        sampling=sampling,
        frame_stride=frame_stride,
        min_keyframes=min_keyframes,
        min_translation_m=min_translation_m,
        min_rotation_deg=min_rotation_deg,
        limit_frames=limit_frames,
        quality_filter=quality_filter,
        quality_filter_concurrency=quality_filter_concurrency,
        on_filter_drop=on_filter_drop or (lambda fid, reason: None),
        cosmic_base_sampling=cosmic_base_sampling,
        cosmic_union_coverage_min=cosmic_union_coverage_min,
        cosmic_yaw_diff_min_deg=cosmic_yaw_diff_min_deg,
        cosmic_obj_vis_area_min=cosmic_obj_vis_area_min,
        cosmic_obj_vis_depth_pix_min=cosmic_obj_vis_depth_pix_min,
        on_pair_reject=on_pair_reject,
    )
    if not pairs:
        return [], []
    frame_ids = sorted({p.src_id for p in pairs}
                       | {p.tgt_id for p in pairs})
    frames_for_pairs = [adapter.frame_ref(fid, adapter_name)
                        for fid in frame_ids]
    return pairs, frames_for_pairs


def apply_filter_to_pairs(
    pairs: list,
    frames_for_pairs: list[FrameRef],
    spec: Optional[ModelSpec],
    *,
    n_votes: int = 1,
    vote_temperature: float = 0.7,
    on_drop: Optional[Callable[[Any, str], None]] = None,
) -> tuple[list, list[FrameRef]]:
    """Drop pairs whose src or tgt frame failed the quality filter.

    Reads filter verdicts from cache only (``endpoint=None``) — assumes
    the filter pass has already populated the cache for every frame in
    ``frames_for_pairs``. A pair is dropped iff ``is_usable`` returns
    ``(False, _)`` for either endpoint; ``on_drop(pair, reason)`` is
    invoked once per dropped pair (with a ``qwen_filter:<reason>``
    string).

    Returns ``(kept_pairs, kept_frames_for_pairs)``. ``kept_frames_for_pairs``
    is rebuilt from the surviving pairs (sorted by frame_id) so downstream
    stages see only frames still referenced.

    No-op (returns inputs unchanged) when ``spec is None`` or
    ``pairs`` is empty.
    """
    if spec is None or not pairs:
        return pairs, frames_for_pairs
    flt = build_filter(spec, endpoint=None,
                      n_votes=n_votes,
                      vote_temperature=vote_temperature)
    verdict: dict[str, tuple[bool, str]] = {}
    refs_by_fid: dict[str, FrameRef] = {f.frame_id: f for f in frames_for_pairs}
    for fid, fr in refs_by_fid.items():
        verdict[fid] = flt.is_usable(fr)

    kept = []
    for pair in pairs:
        v_src = verdict.get(pair.src_id, (True, ""))
        v_tgt = verdict.get(pair.tgt_id, (True, ""))
        if not v_src[0]:
            if on_drop is not None:
                on_drop(pair, f"qwen_filter:src:{v_src[1]}")
            continue
        if not v_tgt[0]:
            if on_drop is not None:
                on_drop(pair, f"qwen_filter:tgt:{v_tgt[1]}")
            continue
        kept.append(pair)

    if not kept:
        return [], []
    surviving_fids = {p.src_id for p in kept} | {p.tgt_id for p in kept}
    kept_frames = [f for f in frames_for_pairs if f.frame_id in surviving_fids]
    return kept, kept_frames


# Detector/segmenter combos that don't need (or can't use) a multi-GPU
# perception pre-pass. Hoisted from ``cli.generate`` so ``stage_perceive``
# can short-circuit without a circular import.
CPU_ONLY_DETECTORS = frozenset({"scannet-gt", "noop"})
CPU_ONLY_SEGMENTERS = frozenset({"gt-mask", "noop"})


def stage_perceive(*, adapter_name: str, scenes_root: Path,
                   detector_name: str, segmenter_name: str,
                   labeler_spec_name: Optional[str],
                   prompt_file: Optional[Path],
                   gdino_max_classes: int,
                   cache_root: Path, model_tag: str,
                   compile_perception: bool,
                   perception_batch_frames: int,
                   scene_to_frames: dict[str, list[FrameRef]],
                   num_workers: int,
                   prepass_min_frames: int = 40,
                   gpu_ids: Optional[list[int]] = None,
                   log_dir: Optional[Path] = None,
                   n_votes: int = 1,
                   vote_temperature: float = 0.7,
                   vote_strategy: str = "union",
                   ) -> int:
    """Run Phase 4.5 (multi-GPU perception pre-pass) for the given scenes.

    Returns the total number of frames written to the perception cache
    (0 when the pre-pass auto-skips: workers<=0, CPU-only combo, no
    pending frames after cache filtering, or below ``prepass_min_frames``).

    ``scene_to_frames`` is the per-scene ``FrameRef`` list — unique frames
    that need perception. Frames whose ``.pkl`` already exists are skipped
    transparently.
    """
    if num_workers <= 0:
        logger.info("perception pre-pass: disabled (num_workers=%d)", num_workers)
        return 0
    if (detector_name in CPU_ONLY_DETECTORS
            and segmenter_name in CPU_ONLY_SEGMENTERS):
        logger.info("perception pre-pass: skipping (CPU-only combo: %s + %s)",
                    detector_name, segmenter_name)
        return 0

    scene_to_work: dict[str, list] = {}
    total_pending = 0
    for sid, frames in scene_to_frames.items():
        if not frames:
            continue
        cache_dir = Path(cache_root) / adapter_name / sid / model_tag
        items = []
        for fr in frames:
            pkl = cache_dir / f"{fr.frame_id}.pkl"
            if pkl.exists():
                continue
            items.append((str(fr.image_path), fr.frame_id))
        if items:
            scene_to_work[sid] = items
            total_pending += len(items)

    if total_pending == 0:
        logger.info("perception pre-pass: 0 frames pending (cache hit)")
        return 0
    if total_pending < prepass_min_frames:
        logger.info("perception pre-pass: %d frames < threshold (%d); "
                    "serial Phase 5 will handle them",
                    total_pending, prepass_min_frames)
        return 0

    from .perception_workers import (
        FrameWork, WorkerConfig, run_perception_prepass,
    )
    cfg = WorkerConfig(
        adapter_name=adapter_name,
        scenes_root=str(scenes_root),
        detector_name=detector_name,
        segmenter_name=segmenter_name,
        labeler_registry_name=labeler_spec_name,
        prompt_file=str(prompt_file) if prompt_file else None,
        gdino_max_classes=gdino_max_classes,
        cache_root=str(cache_root),
        model_tag=model_tag,
        compile_perception=bool(compile_perception),
        log_dir=str(log_dir) if log_dir else None,
        perception_batch_frames=int(perception_batch_frames),
        n_votes=int(n_votes),
        vote_temperature=float(vote_temperature),
        vote_strategy=str(vote_strategy),
    )
    work_by_scene = {
        sid: [
            FrameWork(image_path=ip, frame_id=fid, labels=None,
                      canon_suffix=None)
            for ip, fid in items
        ]
        for sid, items in scene_to_work.items()
    }
    if gpu_ids is None:
        gpu_ids = list(range(num_workers))
    logger.info("perception pre-pass: starting (%d workers, %d frames "
                "across %d scenes, batch=%d)",
                num_workers, total_pending, len(work_by_scene),
                perception_batch_frames)
    return run_perception_prepass(
        cfg, work_by_scene,
        num_workers=num_workers, gpu_ids=gpu_ids,
    )


def stage_match(adapter, *, pairs: list, args, segmenter, detector,
                cache_root: Path, model_tag: str,
                writer, manifest_writer, content_skills,
                voxels_pos, voxels_neg) -> int:
    """Run Phase 5 (per-scene perception read + geometric match + emit).

    Pure CPU/geometry — no vLLM server. ``detector``/``segmenter`` are
    constructed once across all scenes by the caller; this helper invokes
    the per-scene reset hooks. Voxel-dedup state is per-scene (caller
    allocates fresh ``VoxelSet`` instances per call to avoid cross-scene
    bleed).

    The caller (``cli.generate.process_scenes`` or ``cli.match.main``)
    owns the ``writer`` / ``manifest_writer`` lifetimes — this helper
    only emits records and rejects.
    """
    # Lazy imports keep pipeline.stages free of cli.generate dependencies.
    from .match import match_pair
    from .emit import CorrespondenceRecord, TaskRouter, round_clip_pixel
    from .manifest import build_manifest
    from .skills import extract_all_evidence

    # Local PerceptionCache mirrors cli.generate.PerceptionCache. Hoisted
    # to a module-private helper so stage_match has no circular import.
    from cli.generate import PerceptionCache

    if hasattr(segmenter, "set_adapter"):
        segmenter.set_adapter(adapter)
    if hasattr(detector, "set_adapter"):
        detector.set_adapter(adapter)
    if hasattr(detector, "prepare_scene"):
        # frames_for_pairs was the original argument; reconstruct the
        # unique frame list from the surviving pairs.
        frame_ids = sorted({p.src_id for p in pairs}
                           | {p.tgt_id for p in pairs})
        frames_for_pairs = [adapter.frame_ref(fid, args.adapter)
                            for fid in frame_ids]
        detector.prepare_scene(frames_for_pairs)

    cache = PerceptionCache(
        adapter_name=args.adapter, scene_id=adapter.scene_id,
        root=cache_root, detector=detector, segmenter=segmenter,
        model_tag=model_tag,
    )

    n_emitted = 0
    frame_ids = ({pair.src_id for pair in pairs}
                 | {pair.tgt_id for pair in pairs})
    frames = {fid: adapter.load_frame(fid) for fid in frame_ids}

    for pair in pairs:
        f_src = frames[pair.src_id]
        f_tgt = frames[pair.tgt_id]

        masks_src = cache.get(f_src.image_path, pair.src_id)
        masks_tgt = cache.get(f_tgt.image_path, pair.tgt_id)
        if not masks_src or not masks_tgt:
            continue

        def on_reject(s_idx, reason, _pair=pair):
            writer.reject(adapter.scene_id, _pair.src_id, _pair.tgt_id,
                          s_idx, reason)

        matches = match_pair(
            adapter, f_src, masks_src, f_tgt, masks_tgt,
            seed=args.seed, seed_retries=args.seed_retries,
            depth_tol_m=args.depth_tol, iou_min=args.iou_min,
            emit_occlusion_negatives=args.emit_occlusion_negatives,
            on_reject=on_reject,
        )

        evidence_by_skill = extract_all_evidence(
            pair, f_src, masks_src, f_tgt, masks_tgt, matches,
            content_skills or {},
        ) if (content_skills is not None or manifest_writer is not None) else {}
        if evidence_by_skill:
            from .pairs import ViewPair  # noqa: F401  (frozenset already)
            pair.tasks = frozenset(pair.tasks | set(evidence_by_skill.keys()))
            if manifest_writer is not None:
                for skill, ev in evidence_by_skill.items():
                    try:
                        manifest = build_manifest(
                            skill, ev, pair,
                            adapter.scene_id,
                            getattr(adapter, "source_name", "unknown"),
                            f_src, masks_src, f_tgt, masks_tgt, matches,
                        )
                        manifest_writer.emit(manifest)
                    except Exception as e:
                        logger.warning("manifest emit failed (%s): %s", skill, e)

        W, H = f_src.image_size
        Wt, Ht = f_tgt.image_size
        for m in matches:
            voxels = voxels_pos if m.visible else voxels_neg
            if not voxels.add(m.X_world):
                writer.reject(adapter.scene_id, pair.src_id, pair.tgt_id,
                              m.src_mask_idx, "voxel_dup")
                continue
            ps = round_clip_pixel(*m.p_src, W=W, H=H)
            pt = round_clip_pixel(*m.p_tgt, W=Wt, H=Ht)
            if ps is None or pt is None:
                writer.reject(adapter.scene_id, pair.src_id, pair.tgt_id,
                              m.src_mask_idx, "out_of_bounds")
                continue
            tgt_mask = (masks_tgt[m.tgt_mask_idx]
                        if m.tgt_mask_idx >= 0 else None)
            rec = CorrespondenceRecord(
                scene_id=adapter.scene_id,
                frame_src=pair.src_id, frame_tgt=pair.tgt_id,
                image_src=str(f_src.image_path),
                image_tgt=str(f_tgt.image_path),
                point_src=ps, point_tgt=pt,
                X_world=m.X_world,
                src_mask_id=m.src_mask_idx, tgt_mask_id=m.tgt_mask_idx,
                src_bbox=masks_src[m.src_mask_idx].bbox,
                tgt_bbox=tgt_mask.bbox if tgt_mask else (-1.0, -1.0, -1.0, -1.0),
                src_label=masks_src[m.src_mask_idx].label,
                tgt_label=tgt_mask.label if tgt_mask else "",
                src_canonical=getattr(masks_src[m.src_mask_idx],
                                      "canonical", "") or "",
                tgt_canonical=(getattr(tgt_mask, "canonical", "") or "")
                              if tgt_mask else "",
                depth_src=m.depth_src,
                depth_pred_tgt=m.depth_pred_tgt,
                depth_obs_tgt=m.depth_obs_tgt,
                iou_src_to_tgt=m.iou,
                pair_overlap=pair.overlap,
                seed_retry=m.seed_retry,
                visible=m.visible,
                dataset_source=getattr(adapter, "source_name", "unknown"),
            )
            eligible = set(pair.tasks) if isinstance(writer, TaskRouter) else None
            if eligible is not None:
                writer.emit(rec, eligible_tasks=eligible)
            else:
                writer.emit(rec)
            n_emitted += 1
            if (args.max_samples_per_scene is not None
                    and n_emitted >= args.max_samples_per_scene):
                return n_emitted
    return n_emitted
