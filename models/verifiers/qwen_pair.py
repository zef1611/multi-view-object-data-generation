"""Pair-level skill verifier — vLLM HTTP backend.

Companion to `models/filters/qwen.py` (which filters individual frames).
This module takes a stage-1 pair manifest (one skill, one pair of
images, and the structured evidence) and asks a Qwen-VL model whether
the pair is a valid training example for that skill. Returns
``(usable, reason)`` with on-disk caching keyed by (scene, src, tgt,
skill, evidence signature) so re-runs are free.

Backend: any vLLM-hosted spec from `models.registry.MODELS` whose
``images_per_prompt >= 2`` (the verifier sends both images in one
request). Default operational spec is ``qwen3vl-8B-pair``; the
``--verifier`` CLI accepts any registry name.

Cache layout (model-tagged, mostly-readable):
    cache/verifier/<spec.name>/<skill>/<scene_id>/<src>__<tgt>__<evsig>.json

Where ``evsig`` is a 10-char sha1 of the canonicalized evidence dict —
unavoidable because evidence payloads are arbitrary nested structures.
The rest of the path is human-readable, so spelunking by skill / scene
/ frame pair is straightforward.

Cache-only mode (``endpoint=None``) is **fail-closed**: a missing cache
entry raises rather than admitting an unverified pair. The orchestrator
in ``cli/balance.py`` catches and excludes the row from the verified
output.

Design choices:

* Each prompt ends with the same ``<reason> | <yes/no>`` format as
  ``models/filters/qwen.py`` so the parser is shared.
* Prompts are plausibility / quality checks, not answer re-derivation.
  Asking the VLM to *solve* the skill and comparing to ground truth is
  noisy and compounds model error; asking "would a careful annotator
  keep this?" is faster and higher-agreement.
* The 9 skills share a `_BUILDERS` dispatcher. Adding a new skill =
  one new branch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

from .._vlm_base import _VLMBase
from ..filters.qwen import parse_output
from ..registry import ModelSpec

logger = logging.getLogger(__name__)


# Prompt template lives in configs/verifier_prompt.txt — edit that to
# tune wording without touching code. The template contains a single
# `{body}` placeholder; the per-skill builders below produce that body
# from the evidence dict (counts, distances, etc.). Cache is not
# auto-invalidated on prompt edits; rename the registry spec or
# `rm -rf cache/verifier/<spec>/` after a change.
DEFAULT_PROMPT_FILE = (
    Path(__file__).resolve().parents[2] / "configs" / "verifier_prompt.txt"
)


def load_prompt(prompt_file: Optional[Path] = None) -> str:
    return Path(prompt_file or DEFAULT_PROMPT_FILE).read_text()


PROMPT_TEMPLATE = load_prompt()


# ---- per-skill prompt builders -----------------------------------------

def _p_anchor(ev: dict) -> str:
    shared = ev.get("shared_objects", [])
    lines = [
        f"  - {o.get('src_label','?')} (scale change {o.get('scale_ratio',0):.2f}x)"
        for o in shared[:5]
    ]
    return (
        f"Skill: ANCHOR (identify the common object across both views).\n"
        f"Annotation: {len(shared)} shared object(s) were matched between "
        f"the two views:\n" + "\n".join(lines) + "\n"
        f"Are these objects genuinely the same physical instances in both "
        f"images, visible under a non-trivial viewpoint change?"
    )


def _p_counting(ev: dict) -> str:
    cat = ev.get("category", "?")
    n = ev.get("unique_total", 0)
    n_shared = len(ev.get("shared_match_idx", []))
    n_priv_s = len(ev.get("private_src_idx", []))
    n_priv_t = len(ev.get("private_tgt_idx", []))
    return (
        f"Skill: COUNTING (total unique instances of a category across "
        f"both views).\n"
        f"Annotation: category '{cat}'; {n} unique instance(s) total "
        f"({n_shared} shared between views, {n_priv_s} only in image 1, "
        f"{n_priv_t} only in image 2).\n"
        f"Do you agree that there are exactly {n} distinct '{cat}' "
        f"instance(s) across the two images, and that the count is "
        f"non-trivial (cannot be fully answered from a single image)?"
    )


def _p_relative_distance(ev: dict) -> str:
    ref = ev.get("reference_label", "?")
    cands = ev.get("candidates", [])
    farthest_idx = ev.get("farthest_match_idx")
    farthest_label = "?"
    for c in cands:
        if c.get("match_idx") == farthest_idx:
            farthest_label = c.get("label", "?")
            break
    lines = [
        f"  - {c.get('label','?')}: {c.get('distance_m',0):.2f} m"
        for c in cands
    ]
    margin = ev.get("margin_m", 0.0)
    return (
        f"Skill: RELATIVE DISTANCE (which candidate is farthest from the "
        f"reference in 3D).\n"
        f"Annotation: reference is '{ref}'. Candidate distances:\n"
        + "\n".join(lines) + "\n"
        f"Farthest = '{farthest_label}' (margin to runner-up = {margin:.2f} m).\n"
        f"Given both images, is this ordering clearly correct and is the "
        f"reference object unambiguous?"
    )


def _p_relative_direction(ev: dict) -> str:
    targets = ev.get("targets", [])
    lines = [
        f"  - {t.get('label','?')}: bucket='{t.get('bucket','?')}' "
        f"(azimuth {t.get('azimuth_deg',0):.0f}°)"
        for t in targets[:5]
    ]
    return (
        f"Skill: RELATIVE DIRECTION (direction of an object as seen from "
        f"the OTHER camera).\n"
        f"Annotation: from image 2's viewpoint, the following object(s) "
        f"lie in these compass directions:\n" + "\n".join(lines) + "\n"
        f"Are these directional labels clearly correct, and is each "
        f"target object unambiguous in image 1?"
    )


def _p_cross_point_correspondence(ev: dict) -> str:
    n = ev.get("n_visible_labeled", len(ev.get("qualifying_matches", [])))
    return (
        f"Skill: CROSS POINT CORRESPONDENCE (given a marked point in image 1, "
        f"find the same real-world spot in image 2).\n"
        f"Annotation: {n} labeled object(s) matched between the two views.\n"
        f"Are at least some of those matches genuinely the same physical "
        f"object in both images, and is the viewpoint change non-trivial?"
    )


def _p_cross_object_correspondence(ev: dict) -> str:
    objs = ev.get("shared_objects", [])
    lines = [
        f"  - {o.get('tgt_label','?')} at point {o.get('point_tgt')}"
        for o in objs[:5]
    ]
    return (
        f"Skill: CROSS OBJECT CORRESPONDENCE (point to a shared object in "
        f"image 2 without being told which one).\n"
        f"Annotation: {len(objs)} shared object(s) available to point at "
        f"in image 2:\n" + "\n".join(lines) + "\n"
        f"Are these objects genuinely present in both images, and is each "
        f"pointable at its labeled position in image 2?"
    )


def _p_cross_spatial_transformation(ev: dict) -> str:
    objs = ev.get("transformed_objects", [])
    lines = [
        f"  - {o.get('label','?')} (scale ratio {o.get('scale_ratio',0):.2f})"
        for o in objs[:5]
    ]
    return (
        f"Skill: CROSS SPATIAL TRANSFORMATION (object appearance changes "
        f"under a large viewpoint shift).\n"
        f"Annotation: {len(objs)} matched object(s) with large 2D footprint "
        f"change:\n" + "\n".join(lines) + "\n"
        f"Do these objects visibly appear under a clearly different viewpoint "
        f"between the two images?"
    )


def _p_cross_depth_variation(ev: dict) -> str:
    objs = ev.get("varying_objects", [])
    lines = [
        f"  - {o.get('label','?')}: image1 {o.get('depth_src',0):.2f} m → "
        f"image2 {o.get('depth_tgt',0):.2f} m (Δ {o.get('delta_m',0):+.2f} m)"
        for o in objs[:5]
    ]
    return (
        f"Skill: CROSS DEPTH VARIATION (same object at noticeably different "
        f"depths between views).\n"
        f"Annotation: {len(objs)} matched object(s) with large depth change:\n"
        + "\n".join(lines) + "\n"
        f"Does the depth change of at least one listed object look visually "
        f"plausible (closer/farther between views)?"
    )


def _p_cross_occlusion_visibility(ev: dict) -> str:
    n_vis = ev.get("n_visible", 0)
    n_occ = ev.get("n_occluded", 0)
    return (
        f"Skill: CROSS OCCLUSION VISIBILITY (decide whether an object from "
        f"image 1 is visible or occluded in image 2).\n"
        f"Annotation: {n_vis} object(s) visible in image 2 and "
        f"{n_occ} occluded (blocked by something closer).\n"
        f"Does this pair genuinely contain both visible and occluded cases "
        f"with clear visual evidence?"
    )


_BUILDERS = {
    "anchor": _p_anchor,
    "counting": _p_counting,
    "relative_distance": _p_relative_distance,
    "relative_direction": _p_relative_direction,
    "cross_point_correspondence": _p_cross_point_correspondence,
    "cross_object_correspondence": _p_cross_object_correspondence,
    "cross_spatial_transformation": _p_cross_spatial_transformation,
    "cross_depth_variation": _p_cross_depth_variation,
    "cross_occlusion_visibility": _p_cross_occlusion_visibility,
}


def build_prompt(skill: str, evidence: dict) -> str:
    builder = _BUILDERS.get(skill)
    if builder is None:
        raise ValueError(f"no verifier prompt for skill '{skill}'")
    return PROMPT_TEMPLATE.replace("{body}", builder(evidence))


# ---- cache key helpers -------------------------------------------------

def _evidence_signature(evidence: dict) -> str:
    """10-char sha1 of canonicalized evidence — keeps the cache file
    name stable across reorderings of the same payload.
    """
    blob = json.dumps(evidence, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


def verifier_cache_subpath(skill: str, scene_id: str, src_id: str,
                           tgt_id: str, evidence: dict) -> str:
    """`<skill>/<scene>/<src>__<tgt>__<evsig>` (no leading dir, no
    extension). Tests pin one example of this against the actual class
    behavior so the layout cannot drift unnoticed.
    """
    sig = _evidence_signature(evidence)
    return f"{skill}/{scene_id}/{src_id}__{tgt_id}__{sig}"


# ---- verifier class ----------------------------------------------------

class QwenPairVerifier(_VLMBase):
    """Yes/no plausibility check on (skill, pair, evidence) — vLLM HTTP."""

    cache_namespace = "verifier"

    def __init__(
        self,
        spec: ModelSpec,
        endpoint: Optional[str] = None,
        *,
        api_key: str = "EMPTY",
        max_new_tokens: int = 48,
        max_retries: int = 3,
        temperature: float = 0.0,
        request_timeout: float = 180.0,
    ):
        if spec.backend != "vllm":
            raise ValueError(
                f"QwenPairVerifier expects a vllm-backend spec, got {spec.backend}"
            )
        if spec.images_per_prompt < 2:
            raise ValueError(
                f"QwenPairVerifier needs spec.images_per_prompt>=2 "
                f"(got {spec.images_per_prompt} on {spec.name}). Set "
                f"images_per_prompt=2 in the registry entry."
            )
        super().__init__(spec, endpoint, api_key=api_key,
                         request_timeout=request_timeout)
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.temperature = temperature

    def config(self) -> dict:
        return {
            "model": self.spec.name,
            "model_id": self.spec.model_id,
            "max_new_tokens": self.max_new_tokens,
        }

    def _pair_cache_path(self, skill: str, scene_id: str,
                         src_id: str, tgt_id: str, evidence: dict) -> Path:
        sub = verifier_cache_subpath(skill, scene_id, src_id, tgt_id, evidence)
        p = self._cache_dir() / f"{sub}.json"
        if p.parent not in self._mkdir_seen:
            p.parent.mkdir(parents=True, exist_ok=True)
            self._mkdir_seen.add(p.parent)
        return p

    def verify(self, manifest: dict) -> tuple[bool, str]:
        skill = manifest["skill"]
        scene_id = manifest["scene_id"]
        src_id = manifest["frame_src"]
        tgt_id = manifest["frame_tgt"]
        evidence = manifest.get("evidence", {})

        cp = self._pair_cache_path(skill, scene_id, src_id, tgt_id, evidence)
        if cp.exists():
            try:
                d = json.loads(cp.read_text())
                return bool(d["usable"]), str(d["reason"])
            except (json.JSONDecodeError, KeyError):
                if self.endpoint is None:
                    raise RuntimeError(
                        f"verifier cache corrupt and no live endpoint: {cp}"
                    )
                cp.unlink()

        # Cache miss. Fail-closed in cache-only mode.
        if self.endpoint is None:
            raise RuntimeError(
                f"verifier cache miss ({skill}/{scene_id}/{src_id}__{tgt_id}) "
                f"and no live endpoint — fail-closed."
            )

        client = self._ensure_openai_client()
        src_path = Path(manifest["image_src"])
        tgt_path = Path(manifest["image_tgt"])
        prompt = build_prompt(skill, evidence)

        last_raw = {"text": ""}

        def _call() -> Optional[tuple[bool, str]]:
            response = client.chat.completions.create(
                model=self.spec.model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": self._encode_image(src_path)}},
                        {"type": "image_url",
                         "image_url": {"url": self._encode_image(tgt_path)}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
            raw = (response.choices[0].message.content or "").strip()
            last_raw["text"] = raw
            return parse_output(raw)

        t0 = time.monotonic()
        result, last_err, attempts = self._retry(
            _call, attempts=self.max_retries
        )
        dt = time.monotonic() - t0
        raw = last_raw["text"]

        if result is None:
            # Exhausted retries — admit on the safe side (False) so the
            # caller can write a reject row. Cache the verdict so we
            # don't re-burn cycles next pass.
            usable, reason = False, f"verify_exhausted:{last_err}"
            cp.write_text(json.dumps({
                "usable": usable, "reason": reason, "raw": raw,
                "attempts": attempts, "inference_seconds": round(dt, 3),
                "exhausted": True,
            }))
            logger.error("[verifier:%s] %s/%s/%s__%s EXHAUSTED dt=%.2fs (%s)",
                         self.spec.name, skill, scene_id, src_id, tgt_id,
                         dt, last_err)
            return usable, reason

        usable, reason = result
        cp.write_text(json.dumps({
            "usable": usable, "reason": reason, "raw": raw,
            "attempts": attempts, "inference_seconds": round(dt, 3),
        }))
        logger.info("[verifier:%s] %s/%s/%s__%s usable=%s dt=%.2fs",
                    self.spec.name, skill, scene_id, src_id, tgt_id,
                    usable, dt)
        return usable, reason
