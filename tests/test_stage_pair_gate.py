"""Parity tests for pipeline.stages.stage_pair_gate.

The new helper is the single body shared by `cli generate` (Phase 3) and
the standalone `cli pair_gate` runner. These tests pin that:

  1. `stage_pair_gate(adapter, ...)` returns the same ViewPair list as
     calling `select_pairs(adapter, ...)` directly with the same args.
  2. `frames_for_pairs` is the unique-frame FrameRef list referenced by
     the surviving pairs (sorted by frame_id), reconstructed via
     `adapter.frame_ref`.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.config import resolve as resolve_config
from pipeline.pairs import select_pairs
from pipeline.stages import stage_pair_gate

from tests.test_mock_adapter import MockAdapter, _seed_image_files


def _cfg():
    return resolve_config({
        "selection": {
            "pair_quality_min": 0.0, "pair_diversity_min_m": 0.0,
            "corner_overlap_min": 0.0, "angle_min_deg": 10.0,
            "angle_max_deg": 80.0, "max_distance_m": 100.0,
        },
        "min_frame_gap_by_source": {"unknown": 0},
        "tasks": {
            "cross_spatial_transformation": {
                "min_frame_gap_bonus_by_source": {"unknown": 0},
            },
        },
    }, source="unknown")


def test_stage_pair_gate_matches_select_pairs_directly():
    ad = MockAdapter()
    _seed_image_files(ad)
    cfg = _cfg()

    direct = select_pairs(
        ad, cfg,
        adapter_name="scannet",
        sampling="adaptive",
        frame_stride=50, min_keyframes=30,
        min_translation_m=0.0, min_rotation_deg=10.0,
        limit_frames=0,
        cosmic_base_sampling="stride",
        cosmic_union_coverage_min=0.3,
        cosmic_yaw_diff_min_deg=30.0,
        cosmic_obj_vis_area_min=0.005,
        cosmic_obj_vis_depth_pix_min=50,
    )
    via_stage, ffp = stage_pair_gate(
        ad,
        adapter_name="scannet",
        pair_config=cfg,
        sampling="adaptive",
        frame_stride=50, min_keyframes=30,
        min_translation_m=0.0, min_rotation_deg=10.0,
        limit_frames=0,
        cosmic_base_sampling="stride",
        cosmic_union_coverage_min=0.3,
        cosmic_yaw_diff_min_deg=30.0,
        cosmic_obj_vis_area_min=0.005,
        cosmic_obj_vis_depth_pix_min=50,
    )
    # Same number of survivors with the same identities.
    assert len(direct) == len(via_stage)
    direct_ids = {(p.src_id, p.tgt_id) for p in direct}
    stage_ids = {(p.src_id, p.tgt_id) for p in via_stage}
    assert direct_ids == stage_ids

    # frames_for_pairs is the unique-frame union, sorted by frame_id.
    expected_frames = sorted(
        {p.src_id for p in via_stage} | {p.tgt_id for p in via_stage}
    )
    assert [fr.frame_id for fr in ffp] == expected_frames
    # Adapter name plumbs through to the FrameRef.
    assert all(fr.adapter == "scannet" for fr in ffp)
    assert all(fr.scene_id == "mock" for fr in ffp)


def test_stage_pair_gate_returns_empty_on_no_survivors():
    ad = MockAdapter()
    _seed_image_files(ad)
    # Force quality threshold above any achievable score → 0 survivors.
    cfg = resolve_config({
        "selection": {
            "pair_quality_min": 999.0, "pair_diversity_min_m": 0.0,
            "corner_overlap_min": 0.0, "angle_min_deg": 10.0,
            "angle_max_deg": 80.0, "max_distance_m": 100.0,
        },
        "min_frame_gap_by_source": {"unknown": 0},
        "tasks": {},
    }, source="unknown")
    pairs, ffp = stage_pair_gate(
        ad,
        adapter_name="scannet",
        pair_config=cfg,
        sampling="adaptive",
        frame_stride=50, min_keyframes=30,
        min_translation_m=0.0, min_rotation_deg=10.0,
        limit_frames=0,
        cosmic_base_sampling="stride",
        cosmic_union_coverage_min=0.3,
        cosmic_yaw_diff_min_deg=30.0,
        cosmic_obj_vis_area_min=0.005,
        cosmic_obj_vis_depth_pix_min=50,
    )
    assert pairs == []
    assert ffp == []
