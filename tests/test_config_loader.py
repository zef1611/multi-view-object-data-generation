"""Tests for the unified per-stage / per-skill / per-run config loader.

Covers:
  * load_skills_config()  — assembles the dict from
    configs/pair_selection.json + configs/skills/*.json and produces what
    the existing resolve() consumes.
  * load_stage_config()    — refuses run presets, returns dict with comments
    stripped.
  * load_run_config()      — refuses per-stage configs, deep-merges
    stage_overrides.
  * merge_cli_with_config() — CLI > config precedence.
  * _deep_merge()          — nested dicts merge, scalars/lists replace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from pipeline.config import (
    STAGE_NAMES, _deep_merge, load_run_config, load_skills_config,
    load_stage_config, merge_cli_with_config, resolve,
)
from pipeline.skills.base import CONTENT_SKILLS, POSE_SKILLS


def test_skills_split_assembles_full_config():
    cfg = load_skills_config()
    assert set(cfg) == {"selection", "min_frame_gap_by_source",
                         "tasks", "content_skills"}
    assert set(cfg["tasks"]) == set(POSE_SKILLS)
    assert set(cfg["content_skills"]) == set(CONTENT_SKILLS)


def test_resolve_against_split_config_produces_same_taskgates():
    """resolve() over the split config should match the values that the
    historical configs/tasks.json snapshot encoded. We assert the few
    headline values that the published distribution calibration depends
    on (chat note 2026-04-24)."""
    cfg = load_skills_config()
    pc = resolve(cfg, "scannet")
    assert pc.source_floor == 40
    assert pc.pair_quality_min == 0.12
    assert pc.min_yaw_diff_deg == 30.0
    # Occlusion task has the +60 frame-gap bonus on ScanNet.
    occ = pc.tasks["cross_occlusion_visibility"]
    assert occ.min_frame_gap == 100  # 40 floor + 60 bonus
    assert occ.overlap_min == 0.40
    # Min frame-gap pre-filter = lowest task floor (no bonus on
    # spatial_transformation / depth_variation → 40).
    assert pc.min_frame_gap_pre == 40


def test_load_stage_config_strips_comments_and_returns_dict():
    cfg = load_stage_config("sample")
    # All standard knob keys present, none of the comment keys leak.
    for k in ("sampling", "frame_stride", "min_keyframes",
               "min_translation_m", "min_rotation_deg"):
        assert k in cfg
    assert not any(k.startswith("_") for k in cfg)


def test_load_stage_config_rejects_run_preset(tmp_path: Path):
    bad = tmp_path / "preset.json"
    bad.write_text(json.dumps({"stages": {"sample": "x"}}))
    with pytest.raises(ValueError, match="run preset"):
        load_stage_config("sample", bad)


def test_load_run_config_deep_merges_stage_overrides(tmp_path: Path):
    """Override one nested key inside the cosmic block; siblings keep
    their values from the base stage config (deep merge, not replace)."""
    preset = tmp_path / "preset.json"
    preset.write_text(json.dumps({
        "stages": {"sample": "configs/stages/sample.json"},
        "stage_overrides": {
            "sample": {"cosmic_union_coverage_min": 0.99}
        }
    }))
    rp = load_run_config(preset)
    s = rp.stages["sample"]
    assert s["cosmic_union_coverage_min"] == 0.99
    # Base values still present (deep merge — not replace).
    assert s["sampling"] == "stride"
    assert s["frame_stride"] == 50


def test_load_run_config_rejects_per_stage_file():
    with pytest.raises(ValueError, match="per-stage config"):
        load_run_config(Path("configs/stages/sample.json"))


def test_merge_cli_with_config_cli_wins():
    cfg = {"frame_stride": 50, "min_keyframes": 30}
    args = argparse.Namespace(frame_stride=25, min_keyframes=None)
    merge_cli_with_config(args, cfg, ("frame_stride", "min_keyframes"))
    assert args.frame_stride == 25  # CLI value preserved.
    assert args.min_keyframes == 30  # Filled from config.


def test_deep_merge_nested_and_scalar():
    base = {"a": {"b": 1, "c": 2}, "d": [1, 2]}
    over = {"a": {"b": 99}, "d": [99]}
    out = _deep_merge(base, over)
    assert out == {"a": {"b": 99, "c": 2}, "d": [99]}
    # Inputs untouched.
    assert base["a"] == {"b": 1, "c": 2}


def test_every_stage_has_a_default_config_file():
    """Each name in STAGE_NAMES must ship a configs/stages/<name>.json,
    or the precedence chain (CLI > config) has no defaults to fall back
    on. Pinned by this test so a missing file fails CI loudly."""
    for stage in STAGE_NAMES:
        load_stage_config(stage)
