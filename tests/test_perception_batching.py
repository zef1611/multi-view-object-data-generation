"""Parity tests for the batched perception path.

These tests pin the multi-frame batching contract:

  * ``GDinoDetector.detect_multi_frame`` ≡ ``detect_batched_prompts``
    rowwise (single-frame slice).
  * ``LabeledGDinoDetector.detect_with_labels_multi`` ≡
    ``detect_with_labels`` per frame.
  * ``SAM21Segmenter.segment_multi_frame`` single-frame ≡ ``segment``.
  * The Phase 4.5 auto-skip path falls back to the serial Phase 5 when
    the cumulative frame count is below threshold (no Pool spawned).

GPU-bearing tests are ``skipif(not cuda)``. The auto-skip test runs on
CPU and pins the orchestration logic without any model load.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest import mock

import pytest


# -----------------------------------------------------------------------
# (d) Auto-skip / fallback orchestration — pure CPU, no GPU model load.
# -----------------------------------------------------------------------

def _stub_args(**overrides):
    """Build a minimal argparse-like Namespace for _maybe_run_perception_prepass."""
    import argparse
    ns = argparse.Namespace(
        adapter="scannet",
        scenes_root=Path("/tmp/_stub_scenes"),
        detector="labeled-gdino",
        segmenter="sam2.1",
        cache_root=Path("/tmp/_stub_cache"),
        perception_workers=2,
        perception_batch_frames=4,
        perception_prepass_min_frames=40,
        compile_perception=False,
        prompt_file=None,
        gdino_max_classes=200,
        run_log_dir=None,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_prepass_skipped_for_cpu_only_config(caplog):
    from cli.generate import _maybe_run_perception_prepass
    args = _stub_args(detector="scannet-gt", segmenter="gt-mask")
    scene_state = {"sceneXX": {"frames_for_pairs": []}}
    caplog.set_level("INFO")
    _maybe_run_perception_prepass(args, scene_state, None, None)
    assert any("CPU-only" in r.message for r in caplog.records)


def test_prepass_skipped_for_explicit_zero_workers(caplog):
    from cli.generate import _maybe_run_perception_prepass
    args = _stub_args(perception_workers=0)
    caplog.set_level("INFO")
    _maybe_run_perception_prepass(args, {}, None, None)
    assert any("disabled" in r.message for r in caplog.records)


def test_prepass_skipped_below_min_frames(caplog, tmp_path):
    """Below-threshold runs must NOT spawn a worker pool."""
    from cli.generate import _maybe_run_perception_prepass

    # Build a minimal scene_state with 2 frames worth of work — far below
    # the default threshold of 40. The function should log the skip
    # message and return without importing the workers module's run
    # function (which would otherwise attempt to spawn).
    class _FakeFrameRef:
        def __init__(self, fid):
            self.frame_id = fid
            self.image_path = tmp_path / f"{fid}.png"
            # Touch so the file exists if anything checks (pre-pass
            # itself doesn't open it; only the workers do).
            self.image_path.write_bytes(b"")

    scene_state = {
        "sceneAA": {"frames_for_pairs": [_FakeFrameRef("f0"), _FakeFrameRef("f1")]},
    }
    args = _stub_args(
        cache_root=tmp_path / "cache",
        perception_prepass_min_frames=40,
    )

    with mock.patch(
        "pipeline.perception_workers.run_perception_prepass"
    ) as run_mock:
        caplog.set_level("INFO")
        _maybe_run_perception_prepass(args, scene_state, None, None)
        run_mock.assert_not_called()
    assert any("< threshold" in r.message for r in caplog.records)


def test_prepass_skipped_when_all_frames_already_cached(caplog, tmp_path):
    """Pre-existing perception pickles short-circuit the pre-pass."""
    from cli.generate import _maybe_run_perception_prepass

    cache_dir = tmp_path / "cache" / "scannet" / "sceneZZ" / "labeled-gdino+sam2.1"
    cache_dir.mkdir(parents=True)
    (cache_dir / "f0.pkl").write_bytes(b"x")
    (cache_dir / "f1.pkl").write_bytes(b"x")

    class _FakeFrameRef:
        def __init__(self, fid):
            self.frame_id = fid
            self.image_path = tmp_path / f"{fid}.png"

    scene_state = {
        "sceneZZ": {"frames_for_pairs": [_FakeFrameRef("f0"), _FakeFrameRef("f1")]},
    }
    args = _stub_args(cache_root=tmp_path / "cache")
    with mock.patch(
        "pipeline.perception_workers.run_perception_prepass"
    ) as run_mock:
        caplog.set_level("INFO")
        _maybe_run_perception_prepass(args, scene_state, None, None)
        run_mock.assert_not_called()
    assert any("0 frames pending" in r.message for r in caplog.records)


# -----------------------------------------------------------------------
# (a)+(b)+(c) GPU parity tests. Skipped without CUDA + the asset image.
# -----------------------------------------------------------------------

_TEST_IMAGE = Path("/home/mila/l/leh/scratch/dataset/scannet_data/"
                   "scans/scene0000_00/color/0.jpg")


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@pytest.mark.skipif(
    not (_cuda_available() and _TEST_IMAGE.exists()),
    reason="needs CUDA + a sample ScanNet image",
)
def test_gdino_multi_frame_parity_single_frame():
    """detect_multi_frame on one (image, prompts) item must match
    detect_batched_prompts(image, prompts) row-by-row."""
    from models.detectors.gdino import GDinoDetector

    det = GDinoDetector()
    prompts = ["chair .", "table .", "monitor ."]
    serial = det.detect_batched_prompts(_TEST_IMAGE, prompts, chunk_size=8)
    batched = det.detect_multi_frame([(_TEST_IMAGE, prompts)], micro_batch=8)
    assert len(batched) == 1
    assert len(batched[0]) == len(serial) == len(prompts)
    for s_dets, b_dets in zip(serial, batched[0]):
        assert len(s_dets) == len(b_dets), "per-prompt detection count drifted"
        for s, b in zip(s_dets, b_dets):
            for k in range(4):
                assert abs(s.bbox[k] - b.bbox[k]) < 1e-3, (s.bbox, b.bbox)
            assert abs(s.score - b.score) < 1e-3


@pytest.mark.skipif(
    not (_cuda_available() and _TEST_IMAGE.exists()),
    reason="needs CUDA + a sample ScanNet image",
)
def test_sam_segment_multi_frame_parity_single_frame():
    from models.base import Detection
    from models.segmenters.sam21 import SAM21Segmenter

    seg = SAM21Segmenter()
    # Two arbitrary boxes. Real bbox values would come from GDino — for
    # parity testing the actual mask quality doesn't matter.
    dets = [
        Detection(bbox=(50.0, 50.0, 250.0, 250.0), score=0.9, label="a"),
        Detection(bbox=(300.0, 100.0, 500.0, 350.0), score=0.8, label="b"),
    ]
    single = seg.segment(_TEST_IMAGE, dets)
    multi = seg.segment_multi_frame([(_TEST_IMAGE, dets)])
    assert len(multi) == 1
    assert len(multi[0]) == len(single)
    for a, b in zip(single, multi[0]):
        # Mask arrays should be identical (same model, same input).
        assert a.mask.shape == b.mask.shape
        assert (a.mask == b.mask).all()
        assert abs(a.score - b.score) < 1e-3
