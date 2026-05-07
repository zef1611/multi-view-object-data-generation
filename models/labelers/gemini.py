"""Gemini per-image object labeler.

Calls Gemini on a single image and returns short object phrases that
replace GDino's static prompt vocabulary. Whole objects only —
structural surfaces (walls, floors, ceilings) are excluded so detections
stay concentrated on the things downstream geometry needs to anchor.

API key is read from `gemini_api_key.txt` at the repo root (gitignored).
Cache: `cache/labels/<spec.name>/<adapter>/<scene>/<frame_id>.json` —
model- and frame-tagged, no hashing.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from .._frame_ref import FrameRef
from .._json_salvage import find_json_array
from .._vlm_base import _VLMBase
from ..registry import ModelSpec

logger = logging.getLogger(__name__)


DEFAULT_PROMPT_FILE = (
    Path(__file__).resolve().parents[2] / "configs" / "label_prompt.txt"
)


def load_prompt(prompt_file: Optional[Path] = None) -> str:
    """Load the labeler prompt from a file (`.txt` body or `.json` with
    a `"prompt"` field). Editing the file lets you tune the labeler
    without touching code."""
    p = Path(prompt_file) if prompt_file else DEFAULT_PROMPT_FILE
    text = p.read_text()
    if p.suffix == ".json":
        data = json.loads(text)
        if not isinstance(data, dict) or "prompt" not in data:
            raise ValueError(f"{p}: JSON must be an object with a 'prompt' field")
        return str(data["prompt"])
    return text


PROMPT = load_prompt()


# Back-compat alias — older callers imported the underscore name.
_find_json_array = find_json_array


def parse_labels(text: str) -> Optional[list[dict]]:
    """Extract a JSON array from an LLM response, tolerating code fences.

    Returns:
      - list (possibly empty) on a valid array of object-dicts.
      - None when parsing fails — caller should retry.
    """
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    body = fenced.group(1) if fenced else text
    payload = find_json_array(body)
    if payload is None:
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    return [d for d in data if isinstance(d, dict) and "object" in d]


def _load_api_key(key_file: Path) -> str:
    key = key_file.read_text().strip()
    if not key:
        raise RuntimeError(f"{key_file} is empty")
    return key


class GeminiLabeler(_VLMBase):
    cache_namespace = "labels"

    def __init__(
        self,
        spec: ModelSpec,
        endpoint: Optional[str] = None,  # unused; gemini is server-less
        *,
        key_file: Path = Path("gemini_api_key.txt"),
        prompt: Optional[str] = None,
        prompt_file: Optional[Path] = None,
        max_objects: int = 20,
        max_retries: int = 3,
    ):
        if spec.backend != "gemini":
            raise ValueError(
                f"GeminiLabeler expects a gemini-backend spec, got {spec.backend}"
            )
        super().__init__(spec, endpoint)
        if prompt is None:
            prompt = load_prompt(prompt_file)
        self.prompt = prompt
        self.max_objects = max_objects
        self.max_retries = max_retries
        self._key_file = Path(key_file)
        self._model = None  # lazy: configured on first inference call

    def _ensure_gemini_client(self):
        """Defer SDK config + API-key read until first use. Cache-only
        consumers (path checks, validity probes) never pay for it."""
        if self._model is not None:
            return self._model
        import google.generativeai as genai
        key_path = self._key_file
        if not key_path.is_absolute():
            key_path = Path.cwd() / key_path
        genai.configure(api_key=_load_api_key(key_path))
        self._model = genai.GenerativeModel(self.spec.model_id)
        return self._model

    @property
    def model_id(self) -> str:
        # Back-compat: older code reached for `.model_id` directly.
        return self.spec.model_id

    def config(self) -> dict:
        return {
            "model": self.spec.name,
            "model_id": self.spec.model_id,
            "max_objects": self.max_objects,
        }

    def label(self, frame: FrameRef) -> list[str]:
        image_path = frame.image_path
        cp = self._cache_path(frame)
        if cp.exists():
            try:
                d = json.loads(cp.read_text())
                if d.get("valid") and "labels" in d:
                    return list(d["labels"])
                cp.unlink()
            except (json.JSONDecodeError, KeyError):
                cp.unlink()

        from PIL import Image as _PILImage
        image = _PILImage.open(image_path).convert("RGB")

        def _call() -> Optional[list[dict]]:
            response = self._ensure_gemini_client().generate_content(
                [image, self.prompt])
            raw_text = getattr(response, "text", "") or ""
            _call.last_raw = raw_text  # type: ignore[attr-defined]
            return parse_labels(raw_text)

        _call.last_raw = ""  # type: ignore[attr-defined]
        items, last_err, attempts = self._retry(_call, attempts=self.max_retries)
        raw = getattr(_call, "last_raw", "")

        if items is None:
            cp.write_text(json.dumps({
                "valid": False, "labels": [], "raw": raw, "items": None,
                "attempts": attempts, "error": last_err,
            }))
            logger.error("gemini exhausted retries for %s (%s)",
                         image_path, last_err)
            return []

        labels: list[str] = []
        canonicals: list[str] = []
        for it in items[: self.max_objects]:
            name = str(it.get("object", "")).strip()
            canon = str(it.get("canonical", name)).strip() or name
            if name:
                labels.append(name)
                canonicals.append(canon)

        cp.write_text(json.dumps({
            "valid": True, "labels": labels, "canonicals": canonicals,
            "raw": raw, "items": items, "attempts": attempts,
        }))
        return labels

    def label_with_canonical(self, frame: FrameRef) -> list[dict]:
        self.label(frame)
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
