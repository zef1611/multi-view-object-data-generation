"""Unified input loader for the per-stage CLIs.

Every stage CLI accepts a single ``--in <path>`` and routes it through
``load_inputs``. Shape is detected from file content (and extension as a
hint), not by trusting the user's filename — so a renamed JSONL still
works as long as the row schema fits.

Supported shapes (one ``InputBundle`` returned per call):

  ``frames.json``           — list of FrameRef dicts (cli/_frames_io.py).
                              Output of ``cli sample``.
  ``pairs.scored.jsonl``    — list of ScoredPair rows (pipeline/pairs_io.py).
                              Output of ``cli pair_gate``.
  ``pairs.jsonl``           — list of PairManifest rows (pipeline/manifest.py).
                              Output of ``cli match`` (per-skill subdir).
  ``<directory>``           — looks for ``frames.json`` then
                              ``pairs.scored.jsonl`` then ``pairs.jsonl``
                              in that order.

Frame reconstruction from a pair file is adapter-free — the pair rows
already carry ``image_src`` / ``image_tgt`` / ``adapter`` / ``scene_id``,
so ``FrameRef`` can be built directly without re-walking the dataset
tree. Frames produced from pair files are deduplicated by
``(adapter, scene_id, frame_id)`` and sorted for deterministic ordering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from models._frame_ref import FrameRef
from cli._frames_io import read_frames
from pipeline.pairs_io import ScoredPair, read_scored_pairs


# PairManifest required keys (matches pipeline/stages.py::_VERIFIER_REQUIRED_KEYS).
_PAIR_MANIFEST_KEYS = ("skill", "scene_id", "frame_src", "frame_tgt",
                       "image_src", "image_tgt")


@dataclass
class InputBundle:
    """Normalized view of any prior-stage artifact.

    ``frames`` is always populated (extracted from whichever shape was
    loaded). ``scored_pairs`` and ``pair_manifests`` are populated only
    when the input file carried that shape; otherwise ``None``.
    """
    frames: list[FrameRef]
    scored_pairs: Optional[list[ScoredPair]] = None
    pair_manifests: Optional[list[dict]] = None
    source_path: Optional[Path] = None
    source_kind: str = "unknown"          # "frames" | "scored_pairs" | "pair_manifests"


def _looks_like_pair_manifest(obj: dict) -> bool:
    return all(k in obj for k in _PAIR_MANIFEST_KEYS)


def _looks_like_scored_pair(obj: dict) -> bool:
    needed = ("adapter", "scene_id", "src_id", "tgt_id",
              "image_src", "image_tgt", "quality")
    return all(k in obj for k in needed)


def _looks_like_frames_json(obj) -> bool:
    if not isinstance(obj, list) or not obj:
        return False
    head = obj[0]
    return isinstance(head, dict) and {"adapter", "scene_id",
                                        "frame_id", "image_path"} <= head.keys()


def _frames_from_scored_pairs(pairs: list[ScoredPair]) -> list[FrameRef]:
    seen: dict[tuple[str, str, str], FrameRef] = {}
    for p in pairs:
        for fr in (p.src_frame_ref(), p.tgt_frame_ref()):
            seen.setdefault((fr.adapter, fr.scene_id, fr.frame_id), fr)
    return sorted(seen.values(),
                  key=lambda f: (f.adapter, f.scene_id, f.frame_id))


def _frames_from_pair_manifests(manifests: list[dict]) -> list[FrameRef]:
    seen: dict[tuple[str, str, str], FrameRef] = {}
    for m in manifests:
        # PairManifest doesn't carry "adapter" — derive from dataset_source
        # when present, fall back to a stable placeholder. Per-frame caches
        # need a real adapter, so callers that want full FrameRefs from
        # pair manifests should also pass --adapter explicitly via the
        # CLI; this fallback exists for read-only inspection.
        adapter = m.get("dataset_source", "scannet")
        scene = m["scene_id"]
        for fid, img in (
            (m["frame_src"], m["image_src"]),
            (m["frame_tgt"], m["image_tgt"]),
        ):
            key = (adapter, scene, fid)
            if key in seen:
                continue
            seen[key] = FrameRef(
                image_path=Path(img),
                adapter=adapter, scene_id=scene, frame_id=fid,
            )
    return sorted(seen.values(),
                  key=lambda f: (f.adapter, f.scene_id, f.frame_id))


def _peek_first_jsonl(path: Path) -> Optional[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def _load_file(path: Path) -> InputBundle:
    if path.suffix == ".json":
        try:
            obj = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: not valid JSON ({e})") from e
        if _looks_like_frames_json(obj):
            return InputBundle(frames=read_frames(path),
                               source_path=path, source_kind="frames")
        raise ValueError(
            f"{path}: .json content does not match frames.json schema "
            f"(expected list of dicts with adapter/scene_id/frame_id/image_path)"
        )

    if path.suffix == ".jsonl":
        head = _peek_first_jsonl(path)
        if head is None:
            return InputBundle(frames=[], source_path=path,
                               source_kind="unknown")
        if _looks_like_pair_manifest(head):
            manifests = [json.loads(l) for l in path.read_text().splitlines()
                         if l.strip()]
            return InputBundle(
                frames=_frames_from_pair_manifests(manifests),
                pair_manifests=manifests,
                source_path=path, source_kind="pair_manifests",
            )
        if _looks_like_scored_pair(head):
            pairs = read_scored_pairs(path)
            return InputBundle(
                frames=_frames_from_scored_pairs(pairs),
                scored_pairs=pairs,
                source_path=path, source_kind="scored_pairs",
            )
        raise ValueError(
            f"{path}: .jsonl row shape unrecognized "
            f"(neither PairManifest nor ScoredPair). First line keys: "
            f"{sorted(head.keys()) if isinstance(head, dict) else head!r}"
        )

    raise ValueError(
        f"{path}: unsupported extension {path.suffix!r} "
        f"(expected .json for frames, .jsonl for pairs)"
    )


def load_inputs(path: Path) -> InputBundle:
    """Load any prior-stage artifact from disk; return a normalized view.

    Raises ``FileNotFoundError`` for missing paths and ``ValueError`` for
    unrecognized content. See module docstring for supported shapes.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        for name in ("frames.json", "pairs.scored.jsonl", "pairs.jsonl"):
            cand = path / name
            if cand.exists():
                return _load_file(cand)
        raise FileNotFoundError(
            f"{path}: directory has no frames.json / pairs.scored.jsonl / pairs.jsonl"
        )
    return _load_file(path)
