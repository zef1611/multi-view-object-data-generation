"""Gemini-labeled Grounding-DINO detector.

Per scene (when `prepare_scene` is called):
  1. Gemini labels every kept frame, returning [{object, canonical}, ...].
  2. We union the per-frame `object` strings into a scene-wide vocab,
     and build a single `object → canonical` map merged across frames.
  3. Per-frame GDino detection uses `object` (specific) phrases for
     precise grounding.
  4. SAM segments using (object label, GDino box). SAM is label-agnostic.
  5. **After segmentation**, mask labels are mapped object → canonical via
     `canonicalize_mask_label(label)`. The canonical lands on
     `ObjectMask.canonical`; `ObjectMask.label` keeps the specific text.

Downstream code uses `mask.canonical` (when non-empty) for cross-frame
identity / matching, and `mask.label` for descriptive Q&A wording.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import torch
from torchvision.ops import nms

from .._frame_ref import FrameRef
from ..base import Detection, Detector
from .gdino import GDinoDetector


@runtime_checkable
class LabelerProtocol(Protocol):
    """Minimum interface a labeler must expose to plug into this detector.

    Implemented by `GeminiLabeler` and `Qwen3VLLabeler` — both share the
    same prompt + JSON schema so they're interchangeable here.
    `label_runs` is optional; when present (Qwen3VLLabeler with
    ``n_votes>1``) the detector can dispatch on per-run results.
    """
    def label(self, frame: FrameRef) -> list[str]: ...
    def label_with_canonical(self, frame: FrameRef) -> list[dict]: ...
    def config(self) -> dict: ...

logger = logging.getLogger(__name__)


VOTE_STRATEGIES = ("union", "majority", "per-run-detect")


class LabeledGDinoDetector(Detector):
    def __init__(
        self,
        labeler: Optional[LabelerProtocol] = None,
        gdino: Optional[GDinoDetector] = None,
        dedup_iou: float = 0.7,
        batch_chunk_size: int = 8,
        labeler_concurrency: int = 1,
        max_box_frac: float = 0.7,
        vote_strategy: str = "union",
        vote_threshold: Optional[int] = None,
        vote_box_iou: float = 0.5,
    ):
        if labeler is None:
            raise ValueError(
                "LabeledGDinoDetector requires a labeler — pass one in via "
                "the orchestrator (cli/generate.py builds it from --labeler)."
            )
        self.labeler = labeler
        self.gdino = gdino or GDinoDetector()
        self.dedup_iou = dedup_iou
        self.batch_chunk_size = batch_chunk_size
        # Drop any single GDino detection whose bbox covers more than this
        # fraction of the frame. GDino occasionally mis-grounds short
        # generic queries (e.g. "pen") to a near-frame-spanning box on
        # cluttered scenes; SAM then segments the dominant region (a
        # table) and the spurious "pen" label survives. 0.7 = >70% of
        # frame area.
        self.max_box_frac = max_box_frac
        # Per-scene labeler concurrency: 1 = sequential (back-compat, polite
        # to rate-limited APIs); >1 = ThreadPoolExecutor over image_paths.
        # vLLM batches concurrent requests internally, so HTTP-served
        # backends get a near-linear speedup up to ~8 workers.
        self.labeler_concurrency = max(1, int(labeler_concurrency))
        if vote_strategy not in VOTE_STRATEGIES:
            raise ValueError(
                f"vote_strategy={vote_strategy!r} not in {VOTE_STRATEGIES}"
            )
        # `vote_strategy` only affects multi-vote labelers (Qwen3VLLabeler
        # with n_votes>1). For n_votes==1 every strategy collapses to the
        # baseline single-list path, so it's a safe default.
        self.vote_strategy = vote_strategy
        self.vote_threshold = vote_threshold
        self.vote_box_iou = float(vote_box_iou)
        # Scene-wide map: lowercased object phrase → canonical category.
        self._label_to_canonical: dict[str, str] = {}
        # Scene-wide list of `object` phrases used to ground GDino. Sorted
        # for determinism; all frames in the scene share the same vocab.
        self._scene_objects: list[str] = []
        # Per-frame `[run_items_0, run_items_1, ...]` — populated only by
        # the `per-run-detect` strategy in `prepare_scene` so `detect()`
        # can re-use the cached runs without paying for another labeler
        # call (which would have to round-trip through the same cache).
        self._per_frame_runs: dict[str, list[list[dict]]] = {}

    def config(self) -> dict:
        return {
            "labeler": self.labeler.config(),
            "gdino": {k: v for k, v in self.gdino.config().items()
                      if k != "prompt"},
            "per_label_prompting": True,
            "batched": True,
            "dedup_iou": self.dedup_iou,
            "max_box_frac": self.max_box_frac,
            "scene_wide_vocab": True,
            "canonical_from_labeler": True,
            "vote_strategy": self.vote_strategy,
            "vote_threshold": self.vote_threshold,
            "vote_box_iou": self.vote_box_iou,
        }

    def _multi_vote_active(self) -> bool:
        """True iff the labeler exposes multi-run results we should use."""
        return (hasattr(self.labeler, "label_runs")
                and getattr(self.labeler, "n_votes", 1) > 1)

    def prepare_scene(self, frames: list[FrameRef]) -> None:
        """Pre-collect the scene-wide vocab + canonical map.

        Calls the labeler once per frame (cached) to harvest both
        `object` phrases (the GDino prompt) and `canonical` categories
        (the cross-frame identity used downstream).

        Strategy dispatch (only meaningful when the labeler is multi-vote):
          * ``"union"``    — every canonical from any run feeds the vocab.
          * ``"majority"`` — only canonicals seen in >=ceil(N/2) runs of
            *that frame* feed the vocab. The scene-wide canonical map
            still includes any object that survived majority in any
            frame (so post-SAM canonicalization still finds a mapping).
          * ``"per-run-detect"`` — vocab is the union (so GDino has the
            full label space available), but the per-frame run lists are
            cached so :meth:`detect` can run GDino once per run and vote
            on detections after the fact.
        """
        objects: dict[str, str] = {}  # lc(object) → canonical (last writer wins)
        use_runs = self._multi_vote_active()
        self._per_frame_runs = {}

        def _harvest(f: FrameRef) -> tuple[list[dict], list[list[dict]]]:
            """Return (entries_for_vocab, all_runs).

            ``all_runs`` is empty when the labeler is single-vote (or
            doesn't expose label_runs); ``entries_for_vocab`` is the
            list whose canonicals get unioned into the scene vocab.
            """
            if not use_runs:
                return list(self.labeler.label_with_canonical(f)), []
            runs = self.labeler.label_runs(f)  # list[list[dict]]
            if self.vote_strategy == "majority":
                entries = self.labeler.majority_items(
                    f, threshold=self.vote_threshold)
            else:
                # union / per-run-detect both want the full vocab so
                # GDino has every queried phrase available.
                entries = type(self.labeler)._union_items(runs)
            return entries, runs

        if self.labeler_concurrency > 1 and len(frames) > 1:
            workers = min(self.labeler_concurrency, len(frames))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(_harvest, frames))
        else:
            results = [_harvest(f) for f in frames]

        for f, (entries, runs) in zip(frames, results):
            for entry in entries:
                obj = entry.get("object", "").strip()
                canon = entry.get("canonical", obj).strip() or obj
                if obj:
                    objects[obj.lower()] = canon.lower()
            if use_runs and self.vote_strategy == "per-run-detect":
                self._per_frame_runs[f.frame_id] = runs
        self._label_to_canonical = objects
        self._scene_objects = sorted({o for o in objects})
        if use_runs:
            logger.info(
                "scene vocab (vote=%s): %d unique object phrases → %d unique canonicals",
                self.vote_strategy, len(self._scene_objects),
                len({c for c in objects.values()}),
            )
        else:
            logger.info(
                "scene vocab: %d unique object phrases → %d unique canonicals",
                len(self._scene_objects),
                len({c for c in objects.values()}),
            )

    def canonicalize_mask_label(self, label: str) -> str:
        """Map a mask's specific label string to its canonical category.

        Used by the perception layer to backfill `ObjectMask.canonical`
        after SAM segmentation. Falls back to the input label when no
        mapping exists (e.g. detector run without `prepare_scene`).
        """
        if not label:
            return ""
        return self._label_to_canonical.get(label.strip().lower(), label).strip()

    @property
    def label_to_canonical(self) -> dict[str, str]:
        """Read-only access to the scene-wide label→canonical map."""
        return dict(self._label_to_canonical)

    def detect(self, frame: FrameRef) -> list[Detection]:
        # Per-run-detect: run GDino once per labeler run with that run's
        # specific labels, then vote on detections across runs. Falls back
        # to the standard scene-vocab path if `prepare_scene` wasn't
        # called (no per-frame runs cached).
        if (self.vote_strategy == "per-run-detect"
                and self._multi_vote_active()
                and self._per_frame_runs.get(frame.frame_id)):
            return self._detect_per_run_vote(
                frame, self._per_frame_runs[frame.frame_id]
            )
        if self._scene_objects:
            return self.detect_with_labels(frame, self._scene_objects)
        # No scene prep: fall back to per-frame labeler call.
        labels = self.labeler.label(frame)
        if not labels:
            logger.info("labeler: no labels for %s", frame.image_path)
            return []
        return self.detect_with_labels(frame, labels)

    def _detect_per_run_vote(self, frame: FrameRef,
                             runs: list[list[dict]]) -> list[Detection]:
        """Run GDino once per labeler run (with that run's labels), then
        cluster detections across runs and keep clusters that appear in
        ``>=threshold`` runs.

        Clustering rule: same canonical (lowercased) AND box IoU >=
        ``vote_box_iou`` to a cluster representative. Cluster bbox is the
        score-weighted mean of its members; cluster score is the mean
        score; cluster label is the most-common ``object`` wording.
        """
        n = max(1, len(runs))
        thr = (self.vote_threshold
               if self.vote_threshold is not None
               else (n + 1) // 2)

        # Run detection once per run-specific label set. Empty runs
        # (parser failures) contribute zero detections — they still count
        # toward N for the threshold (a run that produced nothing is a
        # negative vote against every detection).
        per_run_dets: list[list[Detection]] = []
        for run_items in runs:
            labels = []
            seen: set[str] = set()
            for it in run_items:
                obj = str(it.get("object", "")).strip()
                key = obj.lower()
                if obj and key not in seen:
                    seen.add(key)
                    labels.append(obj)
            if labels:
                per_run_dets.append(self.detect_with_labels(frame, labels))
            else:
                per_run_dets.append([])

        return self._cluster_and_vote(per_run_dets, threshold=thr)

    def _cluster_and_vote(self, per_run_dets: list[list[Detection]],
                          *, threshold: int) -> list[Detection]:
        """Cluster (canonical, IoU)-equivalent detections across runs;
        keep clusters whose vote count meets ``threshold``."""
        from collections import Counter
        clusters: list[dict] = []  # {canon, members:[Detection], runs:set[int]}

        def _iou(a, b):
            ax0, ay0, ax1, ay1 = a
            bx0, by0, bx1, by1 = b
            ix0, iy0 = max(ax0, bx0), max(ay0, by0)
            ix1, iy1 = min(ax1, bx1), min(ay1, by1)
            iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            au = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
            return inter / max(1e-9, au)

        for run_idx, dets in enumerate(per_run_dets):
            for d in dets:
                canon_key = self._label_to_canonical.get(
                    d.label.strip().lower(), d.label.strip().lower())
                placed = False
                for c in clusters:
                    if c["canon_key"] != canon_key:
                        continue
                    if _iou(c["repr_bbox"], d.bbox) >= self.vote_box_iou:
                        c["members"].append(d)
                        c["runs"].add(run_idx)
                        placed = True
                        break
                if not placed:
                    clusters.append({
                        "canon_key": canon_key,
                        "repr_bbox": d.bbox,
                        "members": [d],
                        "runs": {run_idx},
                    })

        out: list[Detection] = []
        for c in clusters:
            if len(c["runs"]) < threshold:
                continue
            members = c["members"]
            scores = [m.score for m in members]
            ssum = sum(scores) or len(scores)
            ws = [(s / ssum) if ssum else (1.0 / len(members))
                  for s in scores]
            x0 = sum(w * m.bbox[0] for w, m in zip(ws, members))
            y0 = sum(w * m.bbox[1] for w, m in zip(ws, members))
            x1 = sum(w * m.bbox[2] for w, m in zip(ws, members))
            y1 = sum(w * m.bbox[3] for w, m in zip(ws, members))
            label_counts: Counter = Counter(m.label for m in members)
            best_label = label_counts.most_common(1)[0][0]
            out.append(Detection(
                bbox=(x0, y0, x1, y1),
                score=sum(scores) / len(scores),
                label=best_label,
            ))
        if clusters:
            logger.info(
                "[vote] %d clusters → %d kept (threshold=%d / %d runs)",
                len(clusters), len(out), threshold, len(per_run_dets),
            )
        return out

    def detect_with_labels(self, frame: FrameRef,
                           labels: list[str]) -> list[Detection]:
        """Detect using a caller-supplied label list (skips per-frame
        Gemini call). All labels are treated as specific `object`
        phrases — canonical mapping happens post-SAM, not here."""
        unique_labels = self._dedupe_labels(labels)
        if not unique_labels:
            return []
        image_path = frame.image_path
        prompts = [f"{lab} ." for lab in unique_labels]
        per_prompt = self.gdino.detect_batched_prompts(
            image_path, prompts, chunk_size=self.batch_chunk_size,
        )
        return self._postprocess_frame_dets(
            image_path, unique_labels, per_prompt,
        )

    def detect_with_labels_multi(
        self,
        frames: list[FrameRef],
        labels_per_frame: list[list[str]],
        micro_batch: int = 4,
    ) -> list[list[Detection]]:
        """Multi-frame analogue of :meth:`detect_with_labels`.

        Flattens the (frame, label) work into one batched GDino call via
        :meth:`GDinoDetector.detect_multi_frame`, then applies the same
        per-frame area-filter + NMS tail used by the single-frame path.
        ``out[i]`` is the detection list for ``frames[i]`` — identical
        to ``detect_with_labels(frames[i], labels_per_frame[i])`` modulo
        floating-point ordering inside the batched matmul.
        """
        if len(frames) != len(labels_per_frame):
            raise ValueError(
                f"frames ({len(frames)}) and labels_per_frame "
                f"({len(labels_per_frame)}) must be the same length"
            )
        if not frames:
            return []

        per_frame_unique: list[list[str]] = [
            self._dedupe_labels(lst) for lst in labels_per_frame
        ]
        items: list[tuple[Path, list[str]]] = [
            (f.image_path, [f"{lab} ." for lab in u])
            for f, u in zip(frames, per_frame_unique)
        ]
        per_frame_per_prompt = self.gdino.detect_multi_frame(
            items, micro_batch=micro_batch,
        )
        out: list[list[Detection]] = []
        for f, unique_labels, per_prompt in zip(
                frames, per_frame_unique, per_frame_per_prompt):
            if not unique_labels:
                out.append([])
                continue
            out.append(self._postprocess_frame_dets(
                f.image_path, unique_labels, per_prompt,
            ))
        return out

    @staticmethod
    def _dedupe_labels(labels: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for lab in labels:
            s = lab.strip()
            key = s.lower()
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(s)
        return unique

    def _postprocess_frame_dets(
        self,
        image_path: Path,
        unique_labels: list[str],
        per_prompt: list[list[Detection]],
    ) -> list[Detection]:
        """Shared tail for single- and multi-frame detect paths.

        Combines per-prompt detections (re-attaching the original label
        string), drops frame-spanning misgroundings, then runs NMS at
        ``dedup_iou`` across all surviving boxes.
        """
        all_dets: list[Detection] = []
        for lab, dets in zip(unique_labels, per_prompt):
            for d in dets:
                all_dets.append(Detection(
                    bbox=d.bbox, score=d.score, label=lab,
                ))
        if not all_dets:
            return []

        # Frame-area filter: drop GDino mis-groundings (e.g. "pen" → whole
        # table). Apply BEFORE NMS so a spurious huge box doesn't suppress
        # a valid smaller one of a different label.
        from PIL import Image as _PIL
        with _PIL.open(image_path) as _im:
            W, H = _im.size
        frame_area = max(1, W * H)
        kept = []
        dropped: list[tuple[str, float]] = []
        for d in all_dets:
            ba = (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
            if ba / frame_area > self.max_box_frac:
                dropped.append((d.label, ba / frame_area))
            else:
                kept.append(d)
        if dropped:
            logger.info(
                "[%s] dropped %d frame-spanning GDino detections (>%.0f%%): %s",
                image_path.name, len(dropped), self.max_box_frac * 100,
                ", ".join(f"{lab}({frac*100:.0f}%)" for lab, frac in dropped[:5]),
            )
        all_dets = kept
        if not all_dets:
            return []

        boxes = torch.tensor(
            [[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]] for d in all_dets],
            dtype=torch.float32,
        )
        scores = torch.tensor([d.score for d in all_dets], dtype=torch.float32)
        keep = nms(boxes, scores, self.dedup_iou).tolist()
        return [all_dets[k] for k in keep]

