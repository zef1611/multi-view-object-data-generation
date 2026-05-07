"""Auto-detection contract for cli/_io.py::load_inputs.

Pins that any prior-stage artifact (frames.json | pairs.scored.jsonl |
pairs.jsonl | a directory containing one of those) returns an
InputBundle with the right shape populated and a deduplicated FrameRef
list.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cli._frames_io import write_frames
from cli._io import load_inputs
from models._frame_ref import FrameRef
from pipeline.pairs_io import ScoredPair, write_scored_pairs


def _frame(adapter, scene, frame_id, img):
    return FrameRef(
        image_path=Path(img), adapter=adapter,
        scene_id=scene, frame_id=frame_id,
    )


def test_loads_frames_json(tmp_path):
    out = tmp_path / "frames.json"
    write_frames([
        _frame("scannet", "scene0", "000050", "/x/000050.jpg"),
        _frame("scannet", "scene0", "000100", "/x/000100.jpg"),
    ], out)
    bundle = load_inputs(out)
    assert bundle.source_kind == "frames"
    assert bundle.scored_pairs is None
    assert bundle.pair_manifests is None
    assert len(bundle.frames) == 2
    assert {f.frame_id for f in bundle.frames} == {"000050", "000100"}


def test_loads_scored_pairs_jsonl_and_extracts_unique_frames(tmp_path):
    out = tmp_path / "pairs.scored.jsonl"
    pairs = [
        ScoredPair(
            adapter="scannet", scene_id="scene0",
            src_id="000050", tgt_id="000100",
            image_src="/x/000050.jpg", image_tgt="/x/000100.jpg",
            overlap=0.4, occluded_frac=0.0, angle_deg=20.0,
            distance_m=1.0, quality=0.3,
            median_depth_src=1.0, median_depth_tgt=1.0,
            tasks=frozenset({"anchor"}),
        ),
        # A second pair sharing src=000050 — frame must dedup.
        ScoredPair(
            adapter="scannet", scene_id="scene0",
            src_id="000050", tgt_id="000200",
            image_src="/x/000050.jpg", image_tgt="/x/000200.jpg",
            overlap=0.5, occluded_frac=0.0, angle_deg=30.0,
            distance_m=2.0, quality=0.4,
            median_depth_src=1.0, median_depth_tgt=1.0,
        ),
    ]
    write_scored_pairs(pairs, out)
    bundle = load_inputs(out)
    assert bundle.source_kind == "scored_pairs"
    assert bundle.scored_pairs is not None
    assert bundle.pair_manifests is None
    assert len(bundle.scored_pairs) == 2
    # Three unique frames (000050 is shared between the two pairs).
    assert sorted(f.frame_id for f in bundle.frames) == \
           ["000050", "000100", "000200"]


def test_loads_pair_manifest_jsonl(tmp_path):
    out = tmp_path / "pairs.jsonl"
    rows = [
        {"skill": "anchor", "scene_id": "scene0",
         "frame_src": "000050", "frame_tgt": "000200",
         "image_src": "/x/000050.jpg", "image_tgt": "/x/000200.jpg",
         "dataset_source": "scannet", "evidence": {}},
    ]
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    bundle = load_inputs(out)
    assert bundle.source_kind == "pair_manifests"
    assert bundle.pair_manifests is not None
    assert bundle.scored_pairs is None
    assert len(bundle.pair_manifests) == 1
    assert {f.frame_id for f in bundle.frames} == {"000050", "000200"}


def test_directory_picks_frames_json_first(tmp_path):
    write_frames([_frame("scannet", "s", "000050", "/x/000050.jpg")],
                 tmp_path / "frames.json")
    write_scored_pairs([
        ScoredPair(
            adapter="scannet", scene_id="s",
            src_id="000050", tgt_id="000200",
            image_src="/x/000050.jpg", image_tgt="/x/000200.jpg",
            overlap=0.0, occluded_frac=0.0, angle_deg=0.0,
            distance_m=0.0, quality=0.0,
            median_depth_src=0.0, median_depth_tgt=0.0,
        ),
    ], tmp_path / "pairs.scored.jsonl")
    bundle = load_inputs(tmp_path)
    assert bundle.source_kind == "frames"


def test_directory_falls_back_to_scored_pairs(tmp_path):
    write_scored_pairs([
        ScoredPair(
            adapter="scannet", scene_id="s",
            src_id="000050", tgt_id="000200",
            image_src="/x/000050.jpg", image_tgt="/x/000200.jpg",
            overlap=0.0, occluded_frac=0.0, angle_deg=0.0,
            distance_m=0.0, quality=0.0,
            median_depth_src=0.0, median_depth_tgt=0.0,
        ),
    ], tmp_path / "pairs.scored.jsonl")
    bundle = load_inputs(tmp_path)
    assert bundle.source_kind == "scored_pairs"


def test_unknown_jsonl_shape_errors(tmp_path):
    bad = tmp_path / "junk.jsonl"
    bad.write_text(json.dumps({"hello": "world"}) + "\n")
    with pytest.raises(ValueError, match="row shape unrecognized"):
        load_inputs(bad)


def test_missing_path_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_inputs(tmp_path / "nope.json")
