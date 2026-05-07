"""Single source of truth for VLM models used as labeler / quality filter.

Adding a new model = one line in `MODELS`. The CLI (`--labeler`,
`--quality-filter`) accepts any registry name; everything else (vLLM
launch args, model id sent over the wire, cache dir) is derived from
the spec.

Typical flow inside the pipeline:

    spec = MODELS["qwen3vl-235B"]
    with launch_server(spec) as endpoint:
        labeler = build_labeler(spec, endpoint)
        ...                            # use it; cache writes per-image
    # context manager kills the server, sweeps VLLM::Worker_TP* orphans

Servers run **one at a time** by design — no GPU memory contention.
The pipeline collapses filter and labeler into a single server lifetime
when both reference the same spec; otherwise they run sequentially.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ── Spec --------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """One row of the model registry.

    `name` is the user-facing CLI value (`--labeler qwen3vl-235B`).
    `model_id` is what the backend actually wants (HuggingFace id for
    vLLM, Gemini API id for the gemini backend).

    For vLLM-hosted specs, the launch args (`tp`, `gpu_memory_utilization`,
    `max_model_len`) are baked into the spec — adding a new model size
    means filling these in once.

    `extras` is a free-form bag for backend-specific knobs (e.g. Gemini
    rate-limit hints, future quantization flags); intentionally empty
    today.
    """
    name: str
    backend: str                            # "vllm" | "gemini"
    model_id: str
    tp: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8192
    images_per_prompt: int = 1              # vLLM --limit-mm-per-prompt {"image": N}
    recommended_concurrency: int = 8        # default ThreadPool size when caller passes None
    extras: dict = field(default_factory=dict)

    @property
    def is_vllm(self) -> bool:
        return self.backend == "vllm"


MODELS: dict[str, ModelSpec] = {
    "qwen3vl-8B": ModelSpec(
        name="qwen3vl-8B",
        backend="vllm",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        tp=1,
        # 0.9 keeps a generous KV-cache budget for high concurrency; if a
        # kill→relaunch of this spec races with vLLM's free-memory check,
        # drop to 0.55 (see JOURNAL.md 2026-04-29 cache layout entry).
        recommended_concurrency=8,
    ),
    "qwen3vl-235B": ModelSpec(
        name="qwen3vl-235B",
        backend="vllm",
        model_id="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
        tp=4,
        recommended_concurrency=16,
    ),
    # Verifier-role variants: same Qwen3-VL weights, but vLLM
    # `--limit-mm-per-prompt {"image": 2}` so the pair verifier can send
    # src + tgt in one request. Kept as separate registry entries (rather
    # than bumping the labeler/filter spec) so the filter/labeler cache
    # namespaces stay isolated and their mm budget is unchanged.
    "qwen3vl-8B-pair": ModelSpec(
        name="qwen3vl-8B-pair",
        backend="vllm",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        tp=1,
        images_per_prompt=2,
        recommended_concurrency=8,
    ),
    "qwen3vl-235B-pair": ModelSpec(
        name="qwen3vl-235B-pair",
        backend="vllm",
        model_id="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
        tp=4,
        images_per_prompt=2,
        recommended_concurrency=16,
    ),
    "gemini-2.5-flash": ModelSpec(
        name="gemini-2.5-flash",
        backend="gemini",
        model_id="gemini-2.5-flash",
    ),
    "gemini-2.5-pro": ModelSpec(
        name="gemini-2.5-pro",
        backend="gemini",
        model_id="gemini-2.5-pro",
    ),
}


def resolve(name: str) -> ModelSpec:
    if name not in MODELS:
        raise KeyError(
            f"Unknown model {name!r}. Registered: {sorted(MODELS)}"
        )
    return MODELS[name]


# ── Cache paths -------------------------------------------------------------

CACHE_ROOT = Path("cache")


def filter_cache_dir(spec: ModelSpec) -> Path:
    p = CACHE_ROOT / "filter" / spec.name
    p.mkdir(parents=True, exist_ok=True)
    return p


def labels_cache_dir(spec: ModelSpec) -> Path:
    p = CACHE_ROOT / "labels" / spec.name
    p.mkdir(parents=True, exist_ok=True)
    return p


def verifier_cache_dir(spec: ModelSpec) -> Path:
    p = CACHE_ROOT / "verifier" / spec.name
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Server lifecycle -------------------------------------------------------

VLLM_DEFAULT_PORT = 8000


def _sweep_orphan_workers() -> None:
    """vLLM workers (`VLLM::Worker_TP*`) are children of EngineCore and
    can outlive the api_server when the wrapper dies abruptly. Sweep
    them by name as a safety net after `kill`.
    """
    for pat in ("vllm.entrypoints", "VLLM::"):
        try:
            subprocess.run(["pkill", "-9", "-f", pat],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           check=False)
        except FileNotFoundError:
            return


def _server_ready(endpoint: str, timeout: float = 2.0) -> bool:
    try:
        r = requests.get(f"{endpoint}/models", timeout=timeout)
        return r.ok
    except requests.RequestException:
        return False


@contextlib.contextmanager
def launch_server(
    spec: ModelSpec,
    *,
    port: int = VLLM_DEFAULT_PORT,
    cuda_visible_devices: Optional[str] = None,
    max_wait_s: int = 1500,
    log_path: Optional[Path] = None,
    log_dir: Optional[Path] = None,
):
    """Context manager for a vLLM server hosting `spec`. Yields the
    OpenAI-compatible base URL.

    Gemini specs are server-less — the context manager yields `None` and
    does no work. Calling code should handle the `None` case (Gemini
    labeler talks directly to the API, no endpoint needed).

    On exit (normal or exceptional): `SIGTERM` to the api_server PID,
    sweep `VLLM::Worker_TP*` orphans, then a short wait so GPU memory
    is freed before the next stage tries to claim it.
    """
    if not spec.is_vllm:
        # Server-less backend — nothing to do.
        yield None
        return

    endpoint = f"http://localhost:{port}/v1"

    # Refuse to launch on top of an existing server — partial state would
    # leak between stages and make timing/cache-attribution confusing.
    if _server_ready(endpoint, timeout=1.0):
        raise RuntimeError(
            f"vLLM endpoint {endpoint} already responds — refusing to "
            f"launch a second server. Tear the existing one down first."
        )

    if cuda_visible_devices is None:
        cuda_visible_devices = ",".join(str(i) for i in range(spec.tp))

    if log_path is None:
        # Prefer `log_dir/vllm_<spec.name>.log` so all artifacts of one
        # pipeline run land under a single per-run directory. When the
        # same spec is launched twice (e.g. filter and labeler with the
        # same model after a kill/restart), stdout is appended to the
        # same file so the timeline stays in one place.
        log_path = Path(log_dir or Path("logs")) / f"vllm_{spec.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-u", "-m", "vllm.entrypoints.openai.api_server",
        "--model", spec.model_id,
        "--port", str(port),
        "--tensor-parallel-size", str(spec.tp),
        "--gpu-memory-utilization", str(spec.gpu_memory_utilization),
        "--max-model-len", str(spec.max_model_len),
        "--limit-mm-per-prompt", json.dumps({"image": spec.images_per_prompt}),
        "--disable-log-stats",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env["PYTHONUNBUFFERED"] = "1"

    logger.info("[server:%s] launching on %s, tp=%d, mem_util=%.2f, log=%s",
                spec.name, endpoint, spec.tp, spec.gpu_memory_utilization,
                log_path)

    # Append (don't truncate) so a second launch of the same spec within
    # one pipeline run keeps the full timeline.
    log_f = open(log_path, "ab")
    proc = subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env,
        # New process group so SIGTERM cleanly kills the wrapper + workers
        preexec_fn=os.setsid,
    )

    try:
        # Wait for /v1/models to bind, or the process to die.
        deadline = time.monotonic() + max_wait_s
        while time.monotonic() < deadline:
            if _server_ready(endpoint, timeout=2.0):
                logger.info("[server:%s] ready", spec.name)
                break
            if proc.poll() is not None:
                tail = _read_tail(log_path, 60)
                raise RuntimeError(
                    f"vLLM ({spec.name}) died during startup. "
                    f"Last log lines:\n{tail}"
                )
            time.sleep(5)
        else:
            raise TimeoutError(
                f"vLLM ({spec.name}) did not bind {endpoint} within "
                f"{max_wait_s}s. See {log_path}."
            )

        yield endpoint
    finally:
        logger.info("[server:%s] tearing down", spec.name)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        _sweep_orphan_workers()
        log_f.close()
        # Brief settle — GPU memory release isn't always synchronous with
        # process exit, especially with NCCL teardown.
        time.sleep(3)


def _read_tail(path: Path, n: int) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return "(no log)"
    return "\n".join(lines[-n:])
