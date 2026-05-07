"""Round-trip projection sanity tests on synthetic + real ScanNet data."""

from pathlib import Path

import numpy as np
import pytest

from datasets.base import (
    Frame, default_reproject_with_depth, world_point_from_pixel,
)


def _synthetic_frame(pose: np.ndarray, depth_value: float = 2.0) -> Frame:
    K = np.array([[600.0, 0.0, 320.0],
                  [0.0, 600.0, 240.0],
                  [0.0, 0.0, 1.0]])
    depth = np.full((480, 640), depth_value, dtype=np.float32)
    return Frame(
        frame_id="syn",
        image_path=Path("/dev/null"),
        image_size=(640, 480),
        depth=depth,
        depth_size=(640, 480),
        K_color=K,
        pose_c2w=pose,
    )


def test_identity_pose_roundtrips_to_same_pixel():
    """Same pose, same K → reprojection returns the same pixel exactly."""
    pose = np.eye(4)
    src = _synthetic_frame(pose)
    tgt = _synthetic_frame(pose)
    rep = default_reproject_with_depth(src, (320.0, 240.0), tgt)
    assert rep is not None and rep.in_bounds
    assert abs(rep.u - 320.0) < 1e-3
    assert abs(rep.v - 240.0) < 1e-3
    assert abs(rep.depth_pred - 2.0) < 1e-6


def test_translation_pose_offsets_pixel_predictably():
    """Translate target camera +0.1 m along world +x; the same world point
    should land left of center in target (smaller u)."""
    src_pose = np.eye(4)
    tgt_pose = np.eye(4)
    tgt_pose[0, 3] = 0.1   # camera moved right; object appears left
    src = _synthetic_frame(src_pose)
    tgt = _synthetic_frame(tgt_pose)
    rep = default_reproject_with_depth(src, (320.0, 240.0), tgt)
    assert rep is not None and rep.in_bounds
    expected_u = 320.0 - (600.0 * 0.1 / 2.0)   # f * dx / z
    assert abs(rep.u - expected_u) < 1e-3
    assert abs(rep.v - 240.0) < 1e-3


def test_world_point_helper_consistent():
    pose = np.eye(4)
    src = _synthetic_frame(pose)
    X = world_point_from_pixel(src, 320.0, 240.0)
    assert X is not None
    np.testing.assert_allclose(X, [0.0, 0.0, 2.0], atol=1e-6)


def test_invalid_depth_returns_none():
    pose = np.eye(4)
    src = _synthetic_frame(pose, depth_value=0.0)
    tgt = _synthetic_frame(pose)
    rep = default_reproject_with_depth(src, (320.0, 240.0), tgt)
    assert rep is None


@pytest.mark.skipif(
    not Path("/home/mila/l/leh/scratch/dataset/scannet_data/scans/scene0000_00").exists(),
    reason="scene0000_00 not on disk",
)
def test_scannet_frame_roundtrips_to_self():
    """For a ScanNet frame, projecting any in-mask pixel into itself returns
    the same pixel within 1 px."""
    from datasets.scannet import ScanNetAdapter
    ad = ScanNetAdapter(Path("/home/mila/l/leh/scratch/dataset/scannet_data/scans/scene0000_00"))
    f = ad.load_frame("100")
    rep = default_reproject_with_depth(f, (640.0, 484.0), f)
    assert rep is not None and rep.in_bounds
    assert abs(rep.u - 640.0) < 1.0
    assert abs(rep.v - 484.0) < 1.0
