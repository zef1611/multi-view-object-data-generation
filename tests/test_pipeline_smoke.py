"""End-to-end Phase-1 plumbing test using noop models (no GPU)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCENE_ROOT = Path("/home/mila/l/leh/scratch/dataset/scannet_data/scans/scene0000_00")


@pytest.mark.skipif(not SCENE_ROOT.exists(), reason="scene0000_00 missing")
def test_noop_pipeline_emits_records(tmp_path: Path):
    out = tmp_path / "smoke.jsonl"
    cmd = [
        sys.executable, "-m", "cli", "generate",
        "--scene", "scene0000_00",
        "--frame-stride", "50",
        "--limit-frames", "20",
        "--detector", "noop", "--segmenter", "noop",
        "--iou-min", "0.05", "--depth-tol", "0.5",
        "--cache-root", str(tmp_path / "cache"),
        "--out", str(out),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) > 0, "noop pipeline produced 0 emits — plumbing regression"
    rec = json.loads(lines[0])
    for k in ("scene_id", "frame_src", "frame_tgt", "point_src", "point_tgt",
              "X_world", "src_mask_id", "tgt_mask_id", "iou_src_to_tgt"):
        assert k in rec
    assert isinstance(rec["point_src"], list) and len(rec["point_src"]) == 2
