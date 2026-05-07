"""Round-trip tests for pipeline.pairs_io.

ViewPair.tasks is a frozenset (JSON-incompatible); the serializer must
convert to/from sorted list. These tests pin that contract and the
self-describing schema (image paths embedded so consumers don't need
an adapter to reconstruct FrameRefs).
"""

from __future__ import annotations

import json

from pipeline.pairs import ViewPair
from pipeline.pairs_io import (
    ScoredPair, read_scored_pairs, view_pairs_to_scored, write_scored_pairs,
)


def _vp(src="000050", tgt="000200", tasks=None):
    return ViewPair(
        src_id=src, tgt_id=tgt,
        overlap=0.45, occluded_frac=0.12,
        angle_deg=28.3, distance_m=1.42,
        quality=0.38,
        median_depth_src=2.1, median_depth_tgt=2.3,
        tasks=frozenset(tasks or []),
    )


def test_view_pairs_to_scored_round_trip(tmp_path):
    vps = [
        _vp("000050", "000200", tasks={"cross_depth_variation",
                                       "cross_spatial_transformation"}),
        _vp("000050", "000300", tasks={"anchor"}),
    ]
    image_path_for = {
        "000050": "/abs/000050.jpg",
        "000200": "/abs/000200.jpg",
        "000300": "/abs/000300.jpg",
    }
    scored = view_pairs_to_scored(
        vps, adapter="scannet", scene_id="scene0093_00",
        image_path_for=image_path_for,
    )
    out = tmp_path / "pairs.scored.jsonl"
    n = write_scored_pairs(scored, out)
    assert n == 2

    loaded = read_scored_pairs(out)
    assert len(loaded) == 2

    # Field-by-field equality (frozenset round-trip is the key concern).
    for original, loaded_sp in zip(scored, loaded):
        assert loaded_sp == original
        # And the ViewPair view round-trips its tasks frozenset:
        rebuilt = loaded_sp.to_view_pair()
        assert isinstance(rebuilt.tasks, frozenset)
        assert rebuilt.tasks == original.tasks


def test_tasks_serialized_as_sorted_list(tmp_path):
    """JSON has no frozenset; ensure on-disk shape is a sorted list so
    cross-process ordering is deterministic."""
    sp = ScoredPair(
        adapter="scannet", scene_id="s",
        src_id="a", tgt_id="b",
        image_src="/x/a.jpg", image_tgt="/x/b.jpg",
        overlap=0.1, occluded_frac=0.0, angle_deg=15.0,
        distance_m=0.5, quality=0.1,
        median_depth_src=1.0, median_depth_tgt=1.0,
        tasks=frozenset({"z_skill", "a_skill", "m_skill"}),
    )
    out = tmp_path / "p.jsonl"
    write_scored_pairs([sp], out)
    raw = json.loads(out.read_text().strip())
    assert raw["tasks"] == ["a_skill", "m_skill", "z_skill"]


def test_image_paths_round_trip_to_frame_refs(tmp_path):
    sp = ScoredPair(
        adapter="scannet", scene_id="scene0093_00",
        src_id="000050", tgt_id="000200",
        image_src="/abs/000050.jpg", image_tgt="/abs/000200.jpg",
        overlap=0.5, occluded_frac=0.0, angle_deg=20.0,
        distance_m=1.0, quality=0.4,
        median_depth_src=2.0, median_depth_tgt=2.0,
    )
    out = tmp_path / "p.jsonl"
    write_scored_pairs([sp], out)
    [loaded] = read_scored_pairs(out)
    fr_src = loaded.src_frame_ref()
    fr_tgt = loaded.tgt_frame_ref()
    assert (fr_src.adapter, fr_src.scene_id, fr_src.frame_id) == \
           ("scannet", "scene0093_00", "000050")
    assert str(fr_src.image_path) == "/abs/000050.jpg"
    assert (fr_tgt.adapter, fr_tgt.scene_id, fr_tgt.frame_id) == \
           ("scannet", "scene0093_00", "000200")
    assert str(fr_tgt.image_path) == "/abs/000200.jpg"


def test_empty_round_trip(tmp_path):
    out = tmp_path / "empty.jsonl"
    n = write_scored_pairs([], out)
    assert n == 0
    assert read_scored_pairs(out) == []


def test_append_mode_accumulates(tmp_path):
    out = tmp_path / "p.jsonl"
    sp = ScoredPair(
        adapter="scannet", scene_id="s",
        src_id="a", tgt_id="b",
        image_src="/x/a.jpg", image_tgt="/x/b.jpg",
        overlap=0.0, occluded_frac=0.0, angle_deg=0.0,
        distance_m=0.0, quality=0.0,
        median_depth_src=0.0, median_depth_tgt=0.0,
    )
    write_scored_pairs([sp], out)
    write_scored_pairs([sp, sp], out, append=True)
    assert len(read_scored_pairs(out)) == 3
