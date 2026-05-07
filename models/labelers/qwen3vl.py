"""Qwen3-VL labeler — talks to a vLLM OpenAI-compatible HTTP server.

Constructed with a `ModelSpec` from the registry plus the live endpoint
URL. Cache lives at
`cache/labels/<spec.name>[__vote{N}]/<adapter>/<scene>/<frame_id>.json` —
model- and frame-tagged, no hashing.

Single-run (``n_votes=1``, default) cache shape::
    {"valid": true, "labels": [...], "canonicals": [...], "items": [...],
     "raw": "...", "attempts": K, "inference_seconds": T}

Multi-run (``n_votes>1``) cache shape — every individual run is kept so
the perception stage can choose its own aggregation strategy
(union / majority / per-run-detect)::
    {"valid": true, "n_votes": N,
     "runs": [
        {"labels": [...], "canonicals": [...], "items": [...],
         "raw": "...", "attempts": K, "inference_seconds": T},
        ...
     ]}

When `endpoint` is None (e.g. a "cache-only" pass after the server has
been killed), `label*()` calls that miss the cache will raise — that's
intentional, the staged pipeline runs the live pass before the
cache-only pass and a miss means we have a bug elsewhere.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from .._frame_ref import FrameRef
from .._vlm_base import _VLMBase
from .gemini import PROMPT, load_prompt, parse_labels
from ..registry import ModelSpec

logger = logging.getLogger(__name__)


def _items_to_labels(items: list[dict], max_objects: int
                     ) -> tuple[list[str], list[str]]:
    """Project parsed VLM JSON to (labels, canonicals) lists, capped."""
    labels: list[str] = []
    canonicals: list[str] = []
    for it in items[: max_objects]:
        name = str(it.get("object", "")).strip()
        canon = str(it.get("canonical", name)).strip() or name
        if name:
            labels.append(name)
            canonicals.append(canon)
    return labels, canonicals


class Qwen3VLLabeler(_VLMBase):
    cache_namespace = "labels"

    def __init__(
        self,
        spec: ModelSpec,
        endpoint: Optional[str] = None,
        *,
        api_key: str = "EMPTY",
        prompt: Optional[str] = None,
        prompt_file: Optional[Path] = None,
        max_objects: int = 20,
        max_retries: int = 3,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        request_timeout: float = 180.0,
        n_votes: int = 1,
        vote_temperature: float = 0.7,
    ):
        if spec.backend != "vllm":
            raise ValueError(
                f"Qwen3VLLabeler expects a vllm-backend spec, got {spec.backend}"
            )
        if n_votes < 1:
            raise ValueError(f"n_votes must be >=1, got {n_votes}")
        # Multi-vote runs land in a sibling cache dir so the canonical
        # single-run cache stays untouched and runs can be A/B-compared.
        cache_tag = f"vote{n_votes}" if n_votes > 1 else None
        super().__init__(spec, endpoint, api_key=api_key,
                         request_timeout=request_timeout,
                         cache_tag=cache_tag)
        if prompt is None:
            prompt = load_prompt(prompt_file) if prompt_file else PROMPT
        self.prompt = prompt
        self.max_objects = max_objects
        self.max_retries = max_retries
        self.max_new_tokens = max_new_tokens
        # n_votes==1 keeps the deterministic temperature; n_votes>1 needs
        # diversity across calls or every run returns the same text.
        self.temperature = vote_temperature if n_votes > 1 else temperature
        self.n_votes = n_votes

    def config(self) -> dict:
        return {
            "model": self.spec.name,
            "model_id": self.spec.model_id,
            "max_objects": self.max_objects,
            "n_votes": self.n_votes,
            "temperature": self.temperature,
        }

    # ── single-call primitive (shared by 1-vote and N-vote paths) ───────

    def _run_once(self, image_url: str) -> tuple[Optional[list[dict]], str,
                                                  int, float]:
        """One end-to-end VLM call: chat → parse → return.

        Returns ``(items, raw_text, attempts_used, dt_seconds)``. ``items``
        is None if every retry attempt failed to produce a parseable
        response.
        """
        client = self._ensure_openai_client()
        last_raw = {"text": ""}

        def _call() -> Optional[list[dict]]:
            response = client.chat.completions.create(
                model=self.spec.model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": image_url}},
                        {"type": "text", "text": self.prompt},
                    ],
                }],
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
            raw_text = (response.choices[0].message.content or "").strip()
            last_raw["text"] = raw_text
            return parse_labels(raw_text)

        t0 = time.monotonic()
        items, _last_err, attempts = self._retry(_call,
                                                 attempts=self.max_retries)
        dt = time.monotonic() - t0
        return items, last_raw["text"], attempts, dt

    # ── public API ──────────────────────────────────────────────────────

    def label(self, frame: FrameRef) -> list[str]:
        """Return per-frame label list. For ``n_votes>1`` this is the
        **union** of canonicals across all runs (de-duped, preserving
        first-seen order). The canonical key matches what every
        downstream gate expects; aggregation strategies that prefer
        majority vote / per-run detection should call ``label_runs()``
        instead."""
        if self.n_votes > 1:
            runs = self.label_runs(frame)
            return self._union_objects(runs)
        return self._label_single(frame)

    def _label_single(self, frame: FrameRef) -> list[str]:
        image_path = frame.image_path
        cp = self._cache_path(frame)
        if cp.exists():
            try:
                d = json.loads(cp.read_text())
                if d.get("valid") and "labels" in d:
                    return list(d["labels"])
                if self.endpoint is None:
                    return []
                cp.unlink()
            except (json.JSONDecodeError, KeyError):
                if self.endpoint is None:
                    return []
                cp.unlink()

        image_url = self._encode_image(image_path)
        items, raw, attempts, dt = self._run_once(image_url)

        if items is None:
            if raw:
                cp.write_text(json.dumps({
                    "valid": False, "labels": [], "raw": raw, "items": None,
                    "attempts": attempts,
                    "inference_seconds": round(dt, 3),
                }))
            logger.error("[labeler:%s] image=%s EXHAUSTED dt=%.2fs",
                         self.spec.name, image_path.name, dt)
            return []

        labels, canonicals = _items_to_labels(items, self.max_objects)
        cp.write_text(json.dumps({
            "valid": True, "labels": labels, "canonicals": canonicals,
            "raw": raw, "items": items, "attempts": attempts,
            "inference_seconds": round(dt, 3),
        }))
        logger.info("[labeler:%s] image=%s labels=%d attempts=%d dt=%.2fs",
                    self.spec.name, image_path.name, len(labels), attempts, dt)
        return labels

    def label_with_canonical(self, frame: FrameRef) -> list[dict]:
        """Return ``[{object, canonical}, ...]``. Multi-vote: union across
        runs (canonical is the dedupe key; the most-common ``object``
        wording wins ties, falling back to first-seen)."""
        if self.n_votes > 1:
            runs = self.label_runs(frame)
            return self._union_items(runs)
        # Single-vote path — re-use existing cache shape.
        self._label_single(frame)
        try:
            d = json.loads(self._cache_path(frame).read_text())
        except (OSError, json.JSONDecodeError):
            return []
        if not d.get("valid"):
            return []
        labels = d.get("labels", [])
        canons = d.get("canonicals") or labels
        return [{"object": o, "canonical": c}
                for o, c in zip(labels, canons)]

    def label_runs(self, frame: FrameRef) -> list[list[dict]]:
        """Return all N parsed runs for ``frame`` (each a list of
        ``{object, canonical}`` dicts).

        ``n_votes==1`` falls back to ``label_with_canonical`` wrapped in
        a single-element list, so callers can always treat the return as
        ``list[list[dict]]``.
        """
        if self.n_votes <= 1:
            return [self.label_with_canonical(frame)]

        cp = self._cache_path(frame)
        if cp.exists():
            try:
                d = json.loads(cp.read_text())
                if d.get("valid") and isinstance(d.get("runs"), list):
                    return [list(r.get("items") or []) for r in d["runs"]]
                if self.endpoint is None:
                    return []
                cp.unlink()
            except (json.JSONDecodeError, KeyError):
                if self.endpoint is None:
                    return []
                cp.unlink()

        image_url = self._encode_image(frame.image_path)
        runs: list[dict] = []
        any_valid = False
        for i in range(self.n_votes):
            items, raw, attempts, dt = self._run_once(image_url)
            if items is None:
                runs.append({
                    "valid": False, "labels": [], "canonicals": [],
                    "items": None, "raw": raw, "attempts": attempts,
                    "inference_seconds": round(dt, 3),
                })
                logger.warning(
                    "[labeler:%s] image=%s vote=%d/%d EXHAUSTED dt=%.2fs",
                    self.spec.name, frame.image_path.name,
                    i + 1, self.n_votes, dt)
                continue
            labels, canonicals = _items_to_labels(items, self.max_objects)
            runs.append({
                "valid": True, "labels": labels, "canonicals": canonicals,
                "items": items, "raw": raw, "attempts": attempts,
                "inference_seconds": round(dt, 3),
            })
            any_valid = True
            logger.info(
                "[labeler:%s] image=%s vote=%d/%d labels=%d attempts=%d dt=%.2fs",
                self.spec.name, frame.image_path.name,
                i + 1, self.n_votes, len(labels), attempts, dt)

        # Cache iff at least one run produced parseable output. If every
        # run failed *and* we never reached the server (raw==""), don't
        # write — let the next invocation retry. Otherwise persist the
        # failure set so we don't loop forever on a model that always
        # produces malformed output for this frame.
        any_raw = any(r.get("raw") for r in runs)
        if any_valid or any_raw:
            cp.write_text(json.dumps({
                "valid": any_valid, "n_votes": self.n_votes, "runs": runs,
            }))

        return [list(r.get("items") or []) for r in runs]

    # ── aggregation helpers (used by union-strategy callers) ────────────

    @staticmethod
    def _union_items(runs: list[list[dict]]) -> list[dict]:
        """Union of ``{object, canonical}`` dicts across runs, keyed by
        lowercase canonical. The most-common ``object`` wording wins
        ties; first-seen order is preserved."""
        from collections import Counter, OrderedDict
        wordings: dict[str, Counter] = {}
        order: OrderedDict[str, str] = OrderedDict()
        for run in runs:
            for it in run:
                obj = str(it.get("object", "")).strip()
                canon = str(it.get("canonical", obj)).strip() or obj
                if not obj:
                    continue
                key = canon.lower()
                wordings.setdefault(key, Counter())[obj] += 1
                if key not in order:
                    order[key] = canon
        out: list[dict] = []
        for key, canon in order.items():
            obj = wordings[key].most_common(1)[0][0]
            out.append({"object": obj, "canonical": canon})
        return out

    @staticmethod
    def _union_objects(runs: list[list[dict]]) -> list[str]:
        return [it["object"] for it in Qwen3VLLabeler._union_items(runs)]

    def majority_items(self, frame: FrameRef,
                       threshold: Optional[int] = None) -> list[dict]:
        """Return ``{object, canonical}`` dicts seen in ``>=threshold``
        runs (default ``ceil(n_votes/2)``). Used by the ``"majority"``
        vote_strategy in ``LabeledGDinoDetector``."""
        runs = self.label_runs(frame)
        if not runs:
            return []
        n = max(1, len(runs))
        thr = threshold if threshold is not None else (n + 1) // 2
        from collections import Counter
        canon_counts: Counter = Counter()
        wordings: dict[str, Counter] = {}
        canon_display: dict[str, str] = {}
        for run in runs:
            seen_in_run: set[str] = set()
            for it in run:
                obj = str(it.get("object", "")).strip()
                canon = str(it.get("canonical", obj)).strip() or obj
                if not obj:
                    continue
                key = canon.lower()
                wordings.setdefault(key, Counter())[obj] += 1
                canon_display.setdefault(key, canon)
                if key not in seen_in_run:
                    seen_in_run.add(key)
                    canon_counts[key] += 1
        out: list[dict] = []
        for key, count in canon_counts.most_common():
            if count >= thr:
                obj = wordings[key].most_common(1)[0][0]
                out.append({"object": obj, "canonical": canon_display[key]})
        return out
