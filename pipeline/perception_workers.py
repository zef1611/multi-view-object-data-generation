"""Multi-process perception pre-pass: one worker per GPU, GDino+SAM resident.

Spawned by ``cli.generate.process_scenes`` between Phase 4 (labeler) and
Phase 5 (per-pair match+emit) when ``--perception-workers > 0``. Each
worker owns one GPU (via ``CUDA_VISIBLE_DEVICES``), holds its own
detector + segmenter instances, and consumes per-scene chunks of frame
work, populating the same ``cache/perception/<adapter>/<scene>/<model_tag>/<frame_id>.pkl``
layout that the serial Phase 5 reads from. Workers write atomically
(``.pkl.tmp`` → ``os.replace``) so the main process's per-pair loop can
read the on-disk cache safely afterwards.

Module-level imports are kept light — heavy GPU-touching imports
(``torch``, ``transformers``, ``sam2``) happen inside ``_init_worker``
so a CPU-only main process can import this module without pulling in
CUDA.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Worker-local globals populated by `_init_worker`. Spawn-safe: each
# worker process has its own copy.
_WORKER: dict[str, Any] = {}


@dataclass(frozen=True)
class FrameWork:
    """One frame of perception work — what a worker needs to know."""
    image_path: str          # str so it pickles cleanly into the spawn queue
    frame_id: str
    labels: Optional[list[str]]      # None => use detector's scene_objects
    canon_suffix: Optional[str]      # cache-key suffix (matches PerceptionCache.get)


@dataclass(frozen=True)
class WorkerConfig:
    """Inputs needed to (re-)build a detector + segmenter inside a worker.

    Only primitive / picklable fields — no torch tensors, no live
    objects. The worker rebuilds everything fresh from these.
    """
    adapter_name: str
    scenes_root: str
    detector_name: str
    segmenter_name: str
    labeler_registry_name: Optional[str]    # registry key for cache-only labeler
    prompt_file: Optional[str]
    gdino_max_classes: int
    cache_root: str
    model_tag: str
    compile_perception: bool
    log_dir: Optional[str]
    perception_batch_frames: int
    n_votes: int = 1
    vote_temperature: float = 0.7
    vote_strategy: str = "union"


def _init_worker(cfg: WorkerConfig, gpu_ids: list[int]) -> None:
    """Pool initializer. Sets ``CUDA_VISIBLE_DEVICES`` BEFORE importing torch.

    Each worker grabs one GPU id from ``gpu_ids`` based on its
    ``Process._identity`` index (Pool numbers workers from 1).
    """
    proc = mp.current_process()
    # Pool numbers workers via _identity = (i,) starting at 1 per pool.
    worker_idx = (proc._identity[0] - 1) if proc._identity else 0
    if gpu_ids:
        gpu = gpu_ids[worker_idx % len(gpu_ids)]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        gpu = -1

    if cfg.log_dir:
        log_path = Path(cfg.log_dir) / f"perception_worker_{gpu}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(name)s %(message)s"
        ))
        logging.basicConfig(level=logging.INFO, handlers=[handler])
        # Re-attach to root so module-level loggers also flow to the file.
        root = logging.getLogger()
        if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
            root.addHandler(handler)

    # Heavy imports happen now, AFTER CUDA_VISIBLE_DEVICES is set.
    import torch  # noqa: F401 - imported for side effects (CUDA init)
    from cli.generate import make_detector, make_segmenter
    from pipeline.stages import build_labeler
    from models.registry import resolve as resolve_model

    labeler = None
    if cfg.labeler_registry_name is not None:
        spec = resolve_model(cfg.labeler_registry_name)
        labeler = build_labeler(
            spec, endpoint=None,
            prompt_file=Path(cfg.prompt_file) if cfg.prompt_file else None,
            n_votes=cfg.n_votes, vote_temperature=cfg.vote_temperature,
        )

    detector = make_detector(
        cfg.detector_name, labeler=labeler,
        gdino_max_classes=cfg.gdino_max_classes,
        labeler_concurrency=1,
        vote_strategy=cfg.vote_strategy,
    )
    if cfg.segmenter_name == "sam2.1":
        from models.segmenters.sam21 import SAM21Segmenter
        segmenter = SAM21Segmenter(
            compile_image_encoder=cfg.compile_perception,
        )
    else:
        segmenter = make_segmenter(cfg.segmenter_name)

    _WORKER["detector"] = detector
    _WORKER["segmenter"] = segmenter
    _WORKER["cfg"] = cfg
    _WORKER["gpu"] = gpu

    # Determinism check — pinned by the GPU isolation contract.
    try:
        device_count = torch.cuda.device_count()
    except Exception:
        device_count = -1
    logger.info("perception worker init: gpu=%d torch.cuda.device_count=%d "
                "detector=%s segmenter=%s",
                gpu, device_count, cfg.detector_name, cfg.segmenter_name)


def _process_scene_chunk(scene_id: str, work_items: list[FrameWork]) -> int:
    """Pool task — segment every frame in ``work_items`` for ``scene_id``.

    Returns the number of frames written to disk by this call. Frames
    whose pickle already exists on entry are skipped. Pickle writes are
    atomic via ``.pkl.tmp`` + ``os.replace``.
    """
    if not work_items:
        return 0

    cfg: WorkerConfig = _WORKER["cfg"]
    detector = _WORKER["detector"]
    segmenter = _WORKER["segmenter"]

    # Lazy imports keep the spawn pickle small.
    from datasets.scannet import ScanNetAdapter
    from datasets.matterport import MatterportAdapter
    from cli.generate import PerceptionCache, make_adapter
    from models._frame_ref import FrameRef

    # Build a FrameRef per work item up front (used for prepare_scene
    # and for the detector's scene-aware paths).
    frame_refs = [
        FrameRef(
            image_path=Path(w.image_path), adapter=cfg.adapter_name,
            scene_id=scene_id, frame_id=w.frame_id,
        )
        for w in work_items
    ]

    # GT-based detectors / segmenters need the adapter object set per scene
    # so they can access poses / GT instance maps. Build it lazily — most
    # configs that hit the pre-pass (labeled-gdino + sam2.1) skip this.
    adapter = None
    needs_adapter = (
        hasattr(detector, "set_adapter") or hasattr(segmenter, "set_adapter")
    )
    if needs_adapter:
        try:
            adapter = make_adapter(
                cfg.adapter_name, Path(cfg.scenes_root) / scene_id,
            )
        except Exception as e:
            logger.warning("[%s] make_adapter failed in worker: %s", scene_id, e)
        if adapter is not None:
            for hook_owner in (detector, segmenter):
                if hasattr(hook_owner, "set_adapter"):
                    hook_owner.set_adapter(adapter)

    # Per-scene labeler vocab harvest (cache-only — labeler caches were
    # populated upstream by Phase 4). Cheap; just file reads.
    if hasattr(detector, "prepare_scene"):
        try:
            detector.prepare_scene(frame_refs)
        except Exception as e:  # never crash the worker on a single scene's prep
            logger.warning("prepare_scene failed for %s: %s — falling back to "
                           "per-frame labeler calls", scene_id, e)

    cache = PerceptionCache(
        adapter_name=cfg.adapter_name, scene_id=scene_id,
        root=Path(cfg.cache_root), detector=detector, segmenter=segmenter,
        model_tag=cfg.model_tag,
    )

    use_batched = (
        hasattr(detector, "detect_with_labels_multi")
        and hasattr(segmenter, "segment_multi_frame")
        and getattr(detector, "_scene_objects", None)
    )

    written = 0
    if use_batched:
        scene_objects = list(detector._scene_objects)  # type: ignore[attr-defined]
        canon_fn = getattr(detector, "canonicalize_mask_label", None)
        K = max(1, cfg.perception_batch_frames)
        # Filter out items whose pkl already exists (cheap stat).
        pending = [
            (w, fr) for w, fr in zip(work_items, frame_refs)
            if not (cache.dir / f"{w.frame_id}.pkl").exists()
        ]
        for start in range(0, len(pending), K):
            group = pending[start: start + K]
            frames_chunk = [fr for _w, fr in group]
            labels_per_frame = [scene_objects for _ in group]
            dets_per_frame = detector.detect_with_labels_multi(
                frames_chunk, labels_per_frame,
                micro_batch=K,
            )
            seg_items = [
                (Path(w.image_path), dets)
                for (w, _fr), dets in zip(group, dets_per_frame)
            ]
            masks_per_frame = segmenter.segment_multi_frame(seg_items)
            for (w, _fr), masks in zip(group, masks_per_frame):
                if callable(canon_fn):
                    for m in masks:
                        m.canonical = canon_fn(m.label)
                p = cache.dir / f"{w.frame_id}.pkl"
                tmp = p.with_suffix(".pkl.tmp")
                with open(tmp, "wb") as f:
                    pickle.dump(masks, f)
                os.replace(tmp, p)
                written += 1
    else:
        # Fallback: serial cache.get inside the worker. Still benefits
        # from multi-GPU parallelism via the Pool fan-out.
        for w in work_items:
            cache.get(
                Path(w.image_path), w.frame_id,
                labels=w.labels, canon_suffix=w.canon_suffix,
            )
            written += 1

    logger.info("[%s] perception worker: %d frames done", scene_id, written)
    return written


def run_perception_prepass(
    cfg: WorkerConfig,
    scene_to_work: dict[str, list[FrameWork]],
    *,
    num_workers: int,
    gpu_ids: Optional[list[int]] = None,
) -> int:
    """Drive the multi-process pre-pass over every scene.

    ``scene_to_work[scene_id]`` is the list of frames to process for
    that scene. Returns the total number of frames written.

    If ``num_workers <= 1`` the pool is bypassed and tasks run inline
    in the calling process — simpler shape for unit tests and for hosts
    with a single GPU where spawning a Pool just adds overhead.
    """
    if not scene_to_work:
        return 0

    if num_workers <= 1:
        # Inline single-process path. Initialize worker state in this
        # process and run tasks directly.
        _init_worker(cfg, gpu_ids or [0])
        total = 0
        for sid, items in scene_to_work.items():
            total += _process_scene_chunk(sid, items)
        return total

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(cfg, list(gpu_ids or list(range(num_workers)))),
    )
    try:
        # One task per scene — the pool maps scenes onto workers.
        results = [
            pool.apply_async(_process_scene_chunk, (sid, items))
            for sid, items in scene_to_work.items()
        ]
        total = 0
        for r in results:
            total += r.get()
        return total
    finally:
        pool.close()
        pool.join()
