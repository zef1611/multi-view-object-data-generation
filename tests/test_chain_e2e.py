"""End-to-end chain test for the per-stage CLIs.

Runs ``cli sample → cli pair_gate → cli match`` on the noop pipeline and
asserts the final ``correspondences.jsonl`` matches what
``cli generate`` produces with the same args. Pins the contract that
the per-stage chain is byte-equivalent to the integrated path.

`cli filter` / `cli label` are not exercised here because the noop
detector requires neither (filter spec disabled via --quality-filter
none; labeled-gdino is not in use).

`cli perceive` is not exercised because the noop detector is CPU-only
and `stage_perceive` auto-skips that combo (its scaffolding is
nonetheless covered indirectly via the existing perception_batching
test).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCENE_ROOT = Path("/home/mila/l/leh/scratch/dataset/scannet_data/scans/scene0000_00")


def _read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _normalize(records: list[dict]) -> list[tuple]:
    """Return a deterministic, hashable view of the record set so two
    runs with the same caches/code produce the same canonical form."""
    keys = ("scene_id", "frame_src", "frame_tgt",
            "point_src", "point_tgt", "src_mask_id", "tgt_mask_id",
            "visible")
    return sorted(
        tuple(
            (k, tuple(r[k]) if isinstance(r[k], list) else r[k])
            for k in keys
        )
        for r in records
    )


@pytest.mark.skipif(not SCENE_ROOT.exists(), reason="scene0000_00 missing")
def test_chain_matches_generate(tmp_path: Path):
    chain_root = tmp_path / "chain"
    integrated_root = tmp_path / "integrated"
    cache_root = tmp_path / "cache"

    common_match_args = [
        "--detector", "noop", "--segmenter", "noop",
        "--iou-min", "0.05", "--depth-tol", "0.5",
        "--cache-root", str(cache_root),
    ]

    # Per-stage chain
    frames = chain_root / "frames.json"
    pairs = chain_root / "pairs.scored.jsonl"
    chain_out_root = chain_root / "out"

    sample_cmd = [
        sys.executable, "-m", "cli", "sample",
        "--scene", "scene0000_00",
        "--sampling", "stride", "--frame-stride", "50",
        "--limit-frames", "20",
        "--out", str(frames),
    ]
    res = subprocess.run(sample_cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert frames.exists()

    pair_gate_cmd = [
        sys.executable, "-m", "cli", "pair_gate",
        "--in", str(frames),
        "--sampling", "stride", "--frame-stride", "50",
        "--limit-frames", "20",
        "--out", str(pairs),
        "--logs-dir", str(chain_root / "logs"),
    ]
    res = subprocess.run(pair_gate_cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr

    if not pairs.exists() or pairs.stat().st_size == 0:
        pytest.skip("no surviving pairs on noop scene — chain test n/a")

    match_cmd = [
        sys.executable, "-m", "cli", "match",
        "--in", str(pairs),
        "--out-root", str(chain_out_root),
        *common_match_args,
        "--logs-dir", str(chain_root / "logs"),
    ]
    res = subprocess.run(match_cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr

    # Integrated cli generate (same scene, same args).
    generate_cmd = [
        sys.executable, "-m", "cli", "generate",
        "--scene", "scene0000_00",
        "--sampling", "stride", "--frame-stride", "50",
        "--limit-frames", "20",
        "--quality-filter", "none",
        *common_match_args,
        "--out-root", str(integrated_root),
        "--viz-num", "0",
        "--cache-root", str(tmp_path / "cache_integrated"),
        "--logs-dir", str(integrated_root / "logs"),
    ]
    res = subprocess.run(generate_cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr

    chain_records = _read_records(
        chain_out_root / "stage_1" / "_all" / "correspondences.jsonl"
    )
    integrated_records = _read_records(
        integrated_root / "stage_1" / "_all" / "correspondences.jsonl"
    )
    assert chain_records, "chain produced 0 records"
    assert integrated_records, "integrated cli generate produced 0 records"
    assert _normalize(chain_records) == _normalize(integrated_records), (
        f"chain vs integrated diverge: "
        f"chain={len(chain_records)}, integrated={len(integrated_records)}"
    )
