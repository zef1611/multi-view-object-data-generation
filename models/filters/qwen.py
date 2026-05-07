"""Qwen3-VL per-image quality filter via OpenAI-compatible vLLM HTTP server.

Same contract as the labeler: takes a `ModelSpec` from the registry plus
the live endpoint URL. Cache lives at
``cache/filter/<spec.name>[__vote{N}]/<adapter>/<scene>/<frame_id>.json``
— model- and frame-tagged, no hashing.

Drops frames flagged as:
  1. moderately/extremely blurry,
  2. extremely low information density,
  3. containing artificial pink/purple markings.

Single-run shape (``n_votes=1``)::
    {"usable": true, "reason": "...", "raw": "...",
     "inference_seconds": T}

Multi-run shape (``n_votes>1``) — every run is kept and the top-level
``usable`` is the majority verdict (default threshold = ``ceil(N/2)``)::
    {"usable": true, "reason": "majority(2/3): <reason of the
       most-frequent verdict>", "n_votes": 3,
     "runs": [
        {"usable": true,  "reason": "...", "raw": "...",
         "inference_seconds": T},
        ...
     ]}

Each call returns ``(usable: bool, reason: str)``. On HTTP error
(transient server hiccup) the per-call verdict defaults to ``usable=True``
and the run is **not cached**, so the next invocation retries.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from .._frame_ref import FrameRef
from .._vlm_base import _VLMBase
from ..registry import ModelSpec

logger = logging.getLogger(__name__)


# Verbatim from the paper (Fig. 10). Body lives in
# configs/quality_filter_prompt.txt — edit that file to tune the filter
# without touching code. Cache is not auto-invalidated on prompt edits;
# rename the registry spec or `rm -rf cache/filter/<spec>/` after a change.
DEFAULT_PROMPT_FILE = (
    Path(__file__).resolve().parents[2] / "configs" / "quality_filter_prompt.txt"
)


def load_prompt(prompt_file: Optional[Path] = None) -> str:
    p = Path(prompt_file) if prompt_file else DEFAULT_PROMPT_FILE
    return p.read_text()


PROMPT = load_prompt()


_OUTPUT_RE = re.compile(r"^\s*(.*?)\s*\|\s*(yes|no)\s*$",
                        re.IGNORECASE | re.DOTALL)


def parse_output(text: str) -> tuple[bool, str]:
    if not text:
        return True, "empty_response"
    line = text.strip().splitlines()[-1]
    m = _OUTPUT_RE.search(line)
    if m:
        reason = m.group(1).strip()
        return m.group(2).lower() == "yes", reason
    if re.search(r"\bno\b", line, re.IGNORECASE):
        return False, line.strip()
    return True, line.strip()


def _aggregate_runs(runs: list[dict], threshold: int
                    ) -> tuple[bool, str]:
    """Majority vote on ``usable``. Tie-break: ``usable=True`` wins
    (matches the single-vote default of "keep on ambiguity"). Reason
    string is the most-common reason among runs that voted with the
    majority verdict; falls back to first non-empty reason."""
    if not runs:
        return True, "no_runs"
    yes = sum(1 for r in runs if r.get("usable"))
    no = len(runs) - yes
    if yes >= threshold:
        verdict = True
    elif no >= threshold:
        verdict = False
    else:
        # No side reaches threshold (only happens when threshold > N/2).
        # Default keep — matches single-vote behavior on parse failure.
        verdict = True
    matching = [r for r in runs if r.get("usable") == verdict]
    reasons = [str(r.get("reason", "")) for r in matching
               if str(r.get("reason", ""))]
    summary = Counter(reasons).most_common(1)[0][0] if reasons else ""
    tally = f"{yes if verdict else no}/{len(runs)}"
    return verdict, f"majority({tally}): {summary}" if summary \
                                                   else f"majority({tally})"


class QwenFilter(_VLMBase):
    cache_namespace = "filter"

    def __init__(
        self,
        spec: ModelSpec,
        endpoint: Optional[str] = None,
        *,
        api_key: str = "EMPTY",
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        request_timeout: float = 120.0,
        n_votes: int = 1,
        vote_temperature: float = 0.7,
        vote_threshold: Optional[int] = None,
    ):
        if spec.backend != "vllm":
            raise ValueError(
                f"QwenFilter expects a vllm-backend spec, got {spec.backend}"
            )
        if n_votes < 1:
            raise ValueError(f"n_votes must be >=1, got {n_votes}")
        cache_tag = f"vote{n_votes}" if n_votes > 1 else None
        super().__init__(spec, endpoint, api_key=api_key,
                         request_timeout=request_timeout,
                         cache_tag=cache_tag)
        self.max_new_tokens = max_new_tokens
        # n_votes==1 stays deterministic; n_votes>1 needs diversity or
        # every call returns the same verdict.
        self.temperature = vote_temperature if n_votes > 1 else temperature
        self.n_votes = n_votes
        self.vote_threshold = (vote_threshold
                               if vote_threshold is not None
                               else (n_votes + 1) // 2)

    def config(self) -> dict:
        return {
            "model": self.spec.name,
            "model_id": self.spec.model_id,
            "max_new_tokens": self.max_new_tokens,
            "n_votes": self.n_votes,
            "vote_threshold": self.vote_threshold,
            "temperature": self.temperature,
        }

    # ── single-call primitive (shared by 1-vote and N-vote paths) ───────

    def _run_once(self, image_url: str) -> tuple[bool, str, str, float]:
        """One end-to-end VLM call: chat → parse → return.

        Returns ``(usable, reason, raw_text, dt_seconds)``. On transient
        HTTP errors, returns ``(True, 'filter_error:...', '', dt)`` —
        same default-keep semantics as the original single-vote path.
        """
        client = self._ensure_openai_client()
        t0 = time.monotonic()
        try:
            response = client.chat.completions.create(
                model=self.spec.model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": PROMPT},
                    ],
                }],
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
            decoded = (response.choices[0].message.content or "").strip()
        except Exception as e:
            dt = time.monotonic() - t0
            logger.warning("[filter:%s] vote ERROR dt=%.2fs %s",
                           self.spec.name, dt, e)
            return True, f"filter_error:{type(e).__name__}", "", dt
        dt = time.monotonic() - t0
        usable, reason = parse_output(decoded)
        return usable, reason, decoded, dt

    # ── public API ──────────────────────────────────────────────────────

    def is_usable(self, frame: FrameRef) -> tuple[bool, str]:
        image_path = frame.image_path
        cp = self._cache_path(frame)
        if cp.exists():
            try:
                d = json.loads(cp.read_text())
                return bool(d["usable"]), str(d["reason"])
            except (json.JSONDecodeError, KeyError):
                # Corrupted cache. Cache-only mode accepts the failure
                # (defaults usable=True so the frame survives downstream);
                # live mode re-fetches.
                if self.endpoint is None:
                    return True, "filter_cache_corrupt"
                cp.unlink()

        if self.n_votes <= 1:
            return self._run_single(frame, cp)
        return self._run_multi(frame, cp)

    def _run_single(self, frame: FrameRef, cp: Path) -> tuple[bool, str]:
        image_url = self._encode_image(frame.image_path)
        usable, reason, raw, dt = self._run_once(image_url)
        if reason.startswith("filter_error:"):
            # Don't cache transient HTTP errors — let the next run retry.
            return usable, reason
        cp.write_text(json.dumps({
            "usable": usable, "reason": reason, "raw": raw,
            "inference_seconds": round(dt, 3),
        }))
        logger.info("[filter:%s] image=%s usable=%s dt=%.2fs reason=%s",
                    self.spec.name, frame.image_path.name, usable,
                    dt, reason[:60])
        return usable, reason

    def _run_multi(self, frame: FrameRef, cp: Path) -> tuple[bool, str]:
        image_url = self._encode_image(frame.image_path)
        runs: list[dict] = []
        any_transient = False
        for i in range(self.n_votes):
            usable, reason, raw, dt = self._run_once(image_url)
            if reason.startswith("filter_error:"):
                any_transient = True
                continue
            runs.append({
                "usable": usable, "reason": reason, "raw": raw,
                "inference_seconds": round(dt, 3),
            })
            logger.info(
                "[filter:%s] image=%s vote=%d/%d usable=%s dt=%.2fs reason=%s",
                self.spec.name, frame.image_path.name,
                i + 1, self.n_votes, usable, dt, reason[:60])

        # If every vote hit a transient error, return default-keep without
        # caching so the next run retries.
        if not runs:
            return True, "filter_error:all_votes_failed"

        usable, reason = _aggregate_runs(runs, self.vote_threshold)

        # Only cache when at least the threshold's worth of votes
        # actually completed — otherwise the next run should top up.
        if len(runs) >= self.vote_threshold or not any_transient:
            cp.write_text(json.dumps({
                "usable": usable, "reason": reason,
                "n_votes": self.n_votes, "runs": runs,
            }))

        logger.info(
            "[filter:%s] image=%s majority=%s yes=%d/%d reason=%s",
            self.spec.name, frame.image_path.name, usable,
            sum(1 for r in runs if r.get("usable")), len(runs),
            reason[:80])
        return usable, reason
