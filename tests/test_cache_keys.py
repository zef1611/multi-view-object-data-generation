"""Pin the VLM cache-path layout (post-refactor: hashless except for an
unavoidable evidence-payload digest in the verifier).

Cache layout:
    cache/labels/<spec.name>/<adapter>/<scene_id>/<frame_id>.json
    cache/filter/<spec.name>/<adapter>/<scene_id>/<frame_id>.json
    cache/verifier/<spec.name>/<skill>/<scene_id>/<src>__<tgt>__<evsig>.json

These tests guard against accidental reintroduction of hashing in the
per-frame caches or any change to the directory layout that consumers
(viz, debug, balance) depend on.
"""

from __future__ import annotations

from pathlib import Path

from models._frame_ref import FrameRef
from models.filters.qwen import QwenFilter
from models.labelers.qwen3vl import Qwen3VLLabeler
from models.registry import resolve
from models.verifiers.qwen_pair import (
    QwenPairVerifier, verifier_cache_subpath,
)

KNOWN_FRAME = FrameRef(
    image_path=Path(
        "/home/mila/l/leh/scratch/dataset/gsvla/scannet_data/scans/"
        "scene0093_00/color/216.jpg"
    ),
    adapter="scannet",
    scene_id="scene0093_00",
    frame_id="216",
)


def test_qwen3vl_8b_labeler_cache_path():
    spec = resolve("qwen3vl-8B")
    lab = Qwen3VLLabeler(spec, endpoint=None)
    p = lab._cache_path(KNOWN_FRAME)
    assert p.parts[-5:] == (
        "labels", "qwen3vl-8B", "scannet", "scene0093_00", "216.json"
    )


def test_qwen_filter_8b_cache_path():
    spec = resolve("qwen3vl-8B")
    flt = QwenFilter(spec, endpoint=None)
    p = flt._cache_path(KNOWN_FRAME)
    assert p.parts[-5:] == (
        "filter", "qwen3vl-8B", "scannet", "scene0093_00", "216.json"
    )


def test_labels_cache_dir_is_model_tagged():
    s8 = resolve("qwen3vl-8B")
    s235 = resolve("qwen3vl-235B")
    p8 = Qwen3VLLabeler(s8, endpoint=None)._cache_path(KNOWN_FRAME)
    p235 = Qwen3VLLabeler(s235, endpoint=None)._cache_path(KNOWN_FRAME)
    # Different specs must land under different model-tagged roots.
    assert "qwen3vl-8B" in p8.parts
    assert "qwen3vl-235B" in p235.parts
    assert p8 != p235


# ---- verifier cache paths ----------------------------------------------

KNOWN_EVIDENCE = {"shared_objects": [{"src_label": "chair"}]}


def test_verifier_cache_subpath_is_human_readable():
    sub = verifier_cache_subpath(
        skill="anchor", scene_id="scene0093_00",
        src_id="000050", tgt_id="000200", evidence=KNOWN_EVIDENCE,
    )
    head, _, evsig = sub.rpartition("__")
    # `<skill>/<scene>/<src>__<tgt>` must be all human-readable; only the
    # trailing evsig is a digest.
    assert head == "anchor/scene0093_00/000050__000200"
    assert len(evsig) == 10
    assert all(c in "0123456789abcdef" for c in evsig)


def test_verifier_cache_path_layout():
    spec = resolve("qwen3vl-8B-pair")
    ver = QwenPairVerifier(spec, endpoint=None)
    p = ver._pair_cache_path(
        skill="anchor", scene_id="scene0093_00",
        src_id="000050", tgt_id="000200", evidence=KNOWN_EVIDENCE,
    )
    # Path: cache/verifier/qwen3vl-8B-pair/anchor/scene0093_00/000050__000200__<evsig>.json
    assert p.parts[-5:-1] == (
        "verifier", "qwen3vl-8B-pair", "anchor", "scene0093_00",
    )
    assert p.suffix == ".json"
    assert p.stem.startswith("000050__000200__")


def test_verifier_cache_dir_is_model_tagged():
    s = resolve("qwen3vl-8B-pair")
    p = QwenPairVerifier(s, endpoint=None)._pair_cache_path(
        "anchor", "scene0093_00", "1", "2", KNOWN_EVIDENCE,
    )
    assert "qwen3vl-8B-pair" in p.parts


def test_verifier_evsig_stable_across_dict_orderings():
    # Same payload, keys reordered → same evsig.
    a = {"shared_objects": [{"src_label": "chair", "scale_ratio": 1.0}]}
    b = {"shared_objects": [{"scale_ratio": 1.0, "src_label": "chair"}]}
    sa = verifier_cache_subpath("anchor", "s", "1", "2", a)
    sb = verifier_cache_subpath("anchor", "s", "1", "2", b)
    assert sa == sb
