"""Shared base for VLM clients (labelers, filters).

Eliminates ~80% duplication across `gemini_labeler.py`,
`qwen3_vl_labeler.py`, `qwen_filter.py`. Provides:

* Lazy OpenAI / Gemini client construction.
* Image → base64 data-URL helper.
* Cache directory + path resolution.
* Retry harness with last-error capture.
* Inference-timing helper.

Cache layout (human-readable, model-tagged — no hashes):
    cache/labels/<spec.name>/<adapter>/<scene_id>/<frame_id>.json
    cache/filter/<spec.name>/<adapter>/<scene_id>/<frame_id>.json
    cache/verifier/<spec.name>/<skill>/<scene_id>/<src>__<tgt>__<sig>.json

Bumping the prompt or filter logic does NOT auto-invalidate. Either
rename the registry spec or `rm -rf cache/<labels|filter|verifier>/<spec>/`.

`QwenPairVerifier` subclasses this for the OpenAI client + retry
plumbing but uses a pair-keyed cache path (not the per-frame
``_cache_path`` helper).
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from ._frame_ref import FrameRef
from .registry import (
    ModelSpec, filter_cache_dir, labels_cache_dir, verifier_cache_dir,
)

logger = logging.getLogger(__name__)


CacheNamespace = Literal["labels", "filter", "verifier"]


# 1×1 transparent PNG — used by `warmup()` to force vLLM to compile its
# vision-encoder kernels before fan-out begins.
_WARMUP_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
    "2mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
_WARMUP_DATA_URL = f"data:image/png;base64,{_WARMUP_PNG_B64}"


class _VLMBase:
    """Shared lifecycle + caching for OpenAI/Gemini-backed VLM clients.

    Subclasses set the class var `cache_namespace` (``"labels"`` or
    ``"filter"``). Cache files live at
    ``cache/<namespace>/<spec.name>/<adapter>/<scene>/<frame>.json``.
    """

    cache_namespace: CacheNamespace = "labels"

    def __init__(
        self,
        spec: ModelSpec,
        endpoint: Optional[str] = None,
        *,
        api_key: str = "EMPTY",
        request_timeout: float = 180.0,
        cache_tag: Optional[str] = None,
    ):
        self.spec = spec
        self.endpoint = endpoint
        self.api_key = api_key
        self.request_timeout = request_timeout
        # Optional suffix appended to the per-spec cache directory
        # (``cache/<ns>/<spec.name>__<tag>/...``). Lets a single registry
        # spec host multiple cache layouts side-by-side — used today by
        # Qwen3VLLabeler to isolate ``vote{N}`` (multi-run) caches from
        # the default single-run schema.
        self.cache_tag = cache_tag
        self._client: Any = None
        self._mkdir_seen: set[Path] = set()

    # ── client ────────────────────────────────────────────────────────

    def _ensure_openai_client(self):
        """Lazy `openai.OpenAI` for vLLM / OpenAI-compatible backends."""
        if self._client is not None:
            return self._client
        if self.endpoint is None:
            raise RuntimeError(
                f"{type(self).__name__}({self.spec.name}) called with no live "
                f"endpoint and the cache missed. Was the live stage run "
                f"before the cache-only consumer?"
            )
        from openai import OpenAI
        self._client = OpenAI(
            base_url=self.endpoint, api_key=self.api_key,
            timeout=self.request_timeout,
        )
        return self._client

    # ── image encoding ────────────────────────────────────────────────

    @staticmethod
    def _encode_image(image_path: Path) -> str:
        """Return a `data:image/<mime>;base64,...` URL for OpenAI-style APIs."""
        suffix = Path(image_path).suffix.lower().lstrip(".") or "png"
        mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:image/{mime};base64,{b64}"

    # ── cache paths ───────────────────────────────────────────────────

    def _cache_dir(self) -> Path:
        if self.cache_namespace == "labels":
            base = labels_cache_dir(self.spec)
        elif self.cache_namespace == "filter":
            base = filter_cache_dir(self.spec)
        elif self.cache_namespace == "verifier":
            base = verifier_cache_dir(self.spec)
        else:
            raise ValueError(f"Unknown cache_namespace {self.cache_namespace!r}")
        if not self.cache_tag:
            return base
        tagged = base.parent / f"{base.name}__{self.cache_tag}"
        tagged.mkdir(parents=True, exist_ok=True)
        return tagged

    def _cache_path(self, frame: FrameRef) -> Path:
        """`cache/<ns>/<spec.name>/<adapter>/<scene>/<frame_id>.json`."""
        p = self._cache_dir() / f"{frame.cache_subpath}.json"
        if p.parent not in self._mkdir_seen:
            p.parent.mkdir(parents=True, exist_ok=True)
            self._mkdir_seen.add(p.parent)
        return p

    # ── retry / timing ────────────────────────────────────────────────

    def _retry(
        self,
        fn: Callable[[], Any],
        *,
        attempts: int = 3,
        on_failure_log: str = "",
    ) -> tuple[Optional[Any], str, int]:
        """Run `fn()` with up to `attempts` retries.

        Returns `(result, last_error, attempts_used)`. `result` is None if
        every attempt either raised or returned None.
        """
        last_err = ""
        used = 0
        for i in range(1, attempts + 1):
            used = i
            try:
                result = fn()
            except Exception as e:
                last_err = f"{type(e).__name__}:{e}"
                logger.warning(
                    "[%s] attempt %d/%d failed: %s%s",
                    self.spec.name, i, attempts, last_err,
                    f"  ({on_failure_log})" if on_failure_log else "",
                )
                continue
            if result is not None:
                return result, "", used
            last_err = "parse_failed"
        return None, last_err, used

    @staticmethod
    def _record_inference_seconds(t0: float, cache_obj: dict) -> float:
        dt = time.monotonic() - t0
        cache_obj["inference_seconds"] = round(dt, 3)
        return dt

    # ── warmup ────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """One-shot probe to force vLLM to compile its vision-encoder
        kernels before a fan-out burst hits the server.

        Without this, the first N concurrent requests all stall on the
        same just-in-time compilation step, then time out / retry. After
        warmup the steady-state batching kicks in immediately. No-op for
        non-vLLM backends or when `endpoint is None`.
        """
        if self.endpoint is None or not self.spec.is_vllm:
            return
        client = self._ensure_openai_client()
        try:
            client.chat.completions.create(
                model=self.spec.model_id,
                messages=[{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": _WARMUP_DATA_URL}},
                    {"type": "text", "text": "."},
                ]}],
                max_tokens=1, temperature=0.0,
            )
        except Exception as e:
            logger.warning("[%s] warmup probe failed (non-fatal): %s",
                           self.spec.name, e)
