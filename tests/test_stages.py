"""Smoke tests for `pipeline/stages.py` — verify the cache-only fast
path skips server launches and that fan-out preserves input order even
when items finish out of order.

These tests do NOT launch any vLLM server. They rely on:
- Pre-seeded cache JSON files for the cache-complete probe.
- The `endpoint=None` contract on the VLM clients (cache-only mode).
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import pytest

from pipeline.stages import (
    _fan_out, filter_cache_complete, labeler_cache_complete,
    verifier_cache_complete, stage_filter, stage_label, stage_verify,
    collect_pair_manifests, write_verified_per_skill,
)
from models._frame_ref import FrameRef
from models.registry import resolve, MODELS


# ---- _fan_out ---------------------------------------------------------

def test_fan_out_preserves_input_order_under_concurrent_jitter():
    def slow(x: int) -> int:
        time.sleep(random.random() * 0.02)
        return x * 2
    out = _fan_out(slow, list(range(50)), workers=8, label="t")
    assert out == [x * 2 for x in range(50)]


def test_fan_out_records_none_for_raised_items_in_threaded_path():
    def maybe_raise(x: int) -> int:
        if x == 3:
            raise RuntimeError("nope")
        return x
    out = _fan_out(maybe_raise, [0, 1, 2, 3, 4], workers=4, label="t")
    assert out == [0, 1, 2, None, 4]


def test_fan_out_records_none_for_raised_items_in_serial_path():
    def maybe_raise(x: int) -> int:
        if x == 3:
            raise RuntimeError("nope")
        return x
    out = _fan_out(maybe_raise, [0, 1, 2, 3, 4], workers=1, label="t")
    assert out == [0, 1, 2, None, 4]


def test_fan_out_serial_path_for_one_item():
    out = _fan_out(lambda x: x + 100, [7], workers=8, label="t")
    assert out == [107]


# ---- cache-complete probes --------------------------------------------

def test_filter_cache_complete_with_no_spec_returns_true():
    assert filter_cache_complete(None, [])
    assert filter_cache_complete(None, [_dummy_frame()])


def test_labeler_cache_complete_treats_missing_as_incomplete(tmp_path,
                                                              monkeypatch):
    # Redirect cache root so we don't touch the real one.
    import models.registry as reg
    monkeypatch.setattr(reg, "CACHE_ROOT", tmp_path)
    spec = MODELS["qwen3vl-8B"]
    frame = _dummy_frame()
    assert not labeler_cache_complete(spec, [frame])


def test_labeler_cache_complete_treats_invalid_as_incomplete(tmp_path,
                                                              monkeypatch):
    import models.registry as reg
    monkeypatch.setattr(reg, "CACHE_ROOT", tmp_path)
    spec = MODELS["qwen3vl-8B"]
    frame = _dummy_frame()

    # Pre-seed a `valid=False` entry — this counts as "needs re-run".
    cache_path = (tmp_path / "labels" / spec.name
                  / frame.adapter / frame.scene_id / f"{frame.frame_id}.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(
        {"valid": False, "labels": [], "raw": "", "items": None,
         "attempts": 3, "error": "EXHAUSTED"}))

    assert not labeler_cache_complete(spec, [frame])

    # Same path, valid=True, must read as complete.
    cache_path.write_text(json.dumps(
        {"valid": True, "labels": ["chair"], "canonicals": ["chair"],
         "raw": "", "items": [], "attempts": 1}))
    assert labeler_cache_complete(spec, [frame])


def test_verifier_cache_complete_with_empty_manifests():
    spec = MODELS["qwen3vl-8B-pair"]
    assert verifier_cache_complete(spec, [])


def test_verifier_cache_complete_misses_when_unseeded(tmp_path, monkeypatch):
    import models.registry as reg
    monkeypatch.setattr(reg, "CACHE_ROOT", tmp_path)
    spec = MODELS["qwen3vl-8B-pair"]
    manifests = [{
        "skill": "anchor", "scene_id": "s",
        "frame_src": "1", "frame_tgt": "2",
        "evidence": {},
    }]
    assert not verifier_cache_complete(spec, manifests)


# ---- stage_* skip-server fast path -------------------------------------

def test_stage_filter_skips_server_on_empty_input():
    # No exception (no launch_server call).
    stage_filter(MODELS["qwen3vl-8B"], [])


def test_stage_label_skips_server_on_empty_input():
    stage_label(MODELS["qwen3vl-8B"], [])


def test_stage_verify_returns_none_list_on_empty_input():
    out = stage_verify(MODELS["qwen3vl-8B-pair"], [])
    assert out == []


def test_stage_verify_cache_only_path_uses_cached_verdicts(tmp_path,
                                                            monkeypatch):
    """If every manifest has a cached verdict, `stage_verify` reads them
    via a cache-only verifier — never launches a server."""
    import models.registry as reg
    monkeypatch.setattr(reg, "CACHE_ROOT", tmp_path)
    spec = MODELS["qwen3vl-8B-pair"]
    manifest = {
        "skill": "anchor", "scene_id": "s",
        "frame_src": "1", "frame_tgt": "2",
        "image_src": "/no/such/image.jpg", "image_tgt": "/no/such/image.jpg",
        "evidence": {},
    }
    # Pre-seed the cache with a verdict.
    from models.verifiers.qwen_pair import QwenPairVerifier
    ver = QwenPairVerifier(spec, endpoint=None)
    cp = ver._pair_cache_path("anchor", "s", "1", "2", {})
    cp.write_text(json.dumps({"usable": True, "reason": "ok", "raw": ""}))

    # Patch launch_server so this test fails loudly if it ever gets called.
    import pipeline.stages as stages
    def _no_launch(*a, **kw):
        raise AssertionError(
            "stage_verify launched a server even though cache was complete"
        )
    monkeypatch.setattr(stages, "launch_server", _no_launch)

    out = stage_verify(spec, [manifest])
    assert out == [(True, "ok")]


# ---- collect_pair_manifests ------------------------------------------

def _seed_manifest(stage_root: Path, skill: str, scene: str, src: str,
                   tgt: str, **extra) -> dict:
    p = stage_root / skill / "pairs.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    m = {
        "skill": skill, "scene_id": scene,
        "frame_src": src, "frame_tgt": tgt,
        "image_src": f"/tmp/{scene}/{src}.jpg",
        "image_tgt": f"/tmp/{scene}/{tgt}.jpg",
        **extra,
    }
    with open(p, "a") as f:
        f.write(json.dumps(m) + "\n")
    return m


def test_collect_pair_manifests_reads_every_skill(tmp_path):
    stage = tmp_path / "stage_1"
    _seed_manifest(stage, "anchor", "s1", "1", "2", evidence={})
    _seed_manifest(stage, "anchor", "s1", "3", "4", evidence={})
    _seed_manifest(stage, "counting", "s2", "5", "6", evidence={"unique_total": 3})
    out = collect_pair_manifests(stage)
    assert len(out) == 3
    skills = sorted({m["skill"] for m in out})
    assert skills == ["anchor", "counting"]


def test_collect_pair_manifests_filters_by_skills(tmp_path):
    stage = tmp_path / "stage_1"
    _seed_manifest(stage, "anchor", "s1", "1", "2")
    _seed_manifest(stage, "counting", "s2", "5", "6")
    out = collect_pair_manifests(stage, skills={"anchor"})
    assert len(out) == 1
    assert out[0]["skill"] == "anchor"


def test_collect_pair_manifests_skips_malformed_and_missing_keys(tmp_path):
    stage = tmp_path / "stage_1"
    p = stage / "anchor" / "pairs.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write('{"skill":"anchor","scene_id":"s","frame_src":"1","frame_tgt":"2",'
                '"image_src":"/a","image_tgt":"/b"}\n')
        f.write('not-json-at-all\n')
        f.write('{"skill":"anchor","scene_id":"s"}\n')           # missing keys
        f.write('\n')                                            # blank
    out = collect_pair_manifests(stage)
    assert len(out) == 1


def test_collect_pair_manifests_handles_missing_root(tmp_path):
    assert collect_pair_manifests(tmp_path / "does_not_exist") == []
    # Empty stage dir
    (tmp_path / "stage_1").mkdir()
    assert collect_pair_manifests(tmp_path / "stage_1") == []


# ---- write_verified_per_skill -----------------------------------------

def test_write_verified_per_skill_keeps_only_usable(tmp_path):
    stage = tmp_path / "stage_1"
    manifests = [
        _seed_manifest(stage, "anchor", "s1", "1", "2"),
        _seed_manifest(stage, "anchor", "s1", "3", "4"),
        _seed_manifest(stage, "counting", "s2", "5", "6"),
    ]
    verdicts = [
        (True, "ok"),
        (False, "ambiguous"),
        (True, "ok"),
    ]
    counts = write_verified_per_skill(stage, manifests, verdicts)
    assert counts == {"anchor": 1, "counting": 1}

    anchor_lines = (stage / "anchor" / "pairs.verified.jsonl").read_text().splitlines()
    assert len(anchor_lines) == 1
    assert json.loads(anchor_lines[0])["frame_src"] == "1"

    counting_lines = (stage / "counting" / "pairs.verified.jsonl").read_text().splitlines()
    assert len(counting_lines) == 1


def test_write_verified_per_skill_truncates_on_rerun(tmp_path):
    """Verdicts come from the idempotent verifier cache, so the verified
    output is reproducible from scratch — file is rewritten, not appended.
    """
    stage = tmp_path / "stage_1"
    m1 = _seed_manifest(stage, "anchor", "s1", "1", "2")
    m2 = _seed_manifest(stage, "anchor", "s1", "3", "4")

    # First pass — both kept
    write_verified_per_skill(stage, [m1, m2], [(True, "ok"), (True, "ok")])
    assert len((stage / "anchor" / "pairs.verified.jsonl").read_text().splitlines()) == 2

    # Second pass — only one kept; file must be rewritten not appended
    write_verified_per_skill(stage, [m1, m2], [(True, "ok"), (False, "drop")])
    assert len((stage / "anchor" / "pairs.verified.jsonl").read_text().splitlines()) == 1


def test_write_verified_per_skill_drops_none_verdicts(tmp_path):
    stage = tmp_path / "stage_1"
    m1 = _seed_manifest(stage, "anchor", "s1", "1", "2")
    counts = write_verified_per_skill(stage, [m1], [None])
    assert counts == {}
    assert not (stage / "anchor" / "pairs.verified.jsonl").exists()


# ---- helpers ----------------------------------------------------------

def _dummy_frame() -> FrameRef:
    return FrameRef(
        image_path=Path("/tmp/no.jpg"),
        adapter="scannet",
        scene_id="scene_test",
        frame_id="000050",
    )
