"""Proof of modularity: stand up the entire Phase 1 pipeline on a totally
synthetic in-memory adapter (no files on disk, no GPU, no real models).

If this passes, a new dataset adapter only needs to subclass
BaseSceneAdapter and implement list_frames() + load_frame(); core +
models work unchanged.
"""

from pathlib import Path

import numpy as np

from datasets.base import BaseSceneAdapter, Frame
from pipeline.dedup import VoxelSet
from pipeline.match import match_pair
from pipeline.pairs import select_pairs
from models.noop import NoopDetector, NoopSegmenter


def _R_y(theta_deg: float) -> np.ndarray:
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class MockAdapter(BaseSceneAdapter):
    """3 frames of a virtual world with a constant 2 m depth plane,
    cameras yawed at 0°, 30°, 60° around world +Y at the origin."""

    def __init__(self):
        self.scene_id = "mock"
        self._frames = ["f0", "f1", "f2"]
        self._yaw = {"f0": 0.0, "f1": 30.0, "f2": 60.0}
        self._K = np.array([[600.0, 0.0, 320.0],
                            [0.0, 600.0, 240.0],
                            [0.0, 0.0, 1.0]])
        self._depth = np.full((480, 640), 2.0, dtype=np.float32)

    def list_frames(self):
        return list(self._frames)

    def load_frame(self, frame_id):
        pose = np.eye(4)
        pose[:3, :3] = _R_y(self._yaw[frame_id])
        return Frame(
            frame_id=frame_id,
            image_path=Path(f"/tmp/mock_{frame_id}.jpg"),  # never read
            image_size=(640, 480),
            depth=self._depth,
            depth_size=(640, 480),
            K_color=self._K,
            pose_c2w=pose,
        )

    # Implement image_path so the noop detector doesn't try to PIL.open()
    # the fake path. We only need it to exist; noop ignores contents... but
    # NoopDetector DOES open the image to read its size. So write a tiny
    # placeholder file once, lazily.
    def image_path(self, frame_id):
        p = Path(f"/tmp/mock_{frame_id}.jpg")
        if not p.exists():
            from PIL import Image
            Image.new("RGB", (640, 480)).save(p)
        return p


def _seed_image_files(adapter):
    """NoopDetector reads image size from disk. Pre-create the placeholders."""
    for fid in adapter.list_frames():
        adapter.image_path(fid)


def test_select_pairs_runs_on_mock_adapter():
    ad = MockAdapter()
    _seed_image_files(ad)
    from pipeline.config import resolve
    cfg = resolve({
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
    pairs = select_pairs(
        ad, cfg,
        sampling="adaptive",
        min_translation_m=0.0,             # mock cameras share origin
        min_rotation_deg=10.0,             # 30° steps survive
    )
    # 3 frames at 30° steps → at least one pair (f0–f1, f1–f2) survives
    assert len(pairs) >= 1, "no pairs survived on mock adapter"


def test_match_pair_emits_on_mock_adapter():
    ad = MockAdapter()
    _seed_image_files(ad)
    f0 = ad.load_frame("f0")
    f1 = ad.load_frame("f1")
    det, seg = NoopDetector(grid=2), NoopSegmenter()
    from models._frame_ref import FrameRef
    r0 = FrameRef(image_path=ad.image_path("f0"), adapter="unknown",
                  scene_id="mock", frame_id="f0")
    r1 = FrameRef(image_path=ad.image_path("f1"), adapter="unknown",
                  scene_id="mock", frame_id="f1")
    masks_a = seg.segment(r0.image_path, det.detect(r0))
    masks_b = seg.segment(r1.image_path, det.detect(r1))
    matches = match_pair(
        ad, f0, masks_a, f1, masks_b,
        seed=42, seed_retries=5, depth_tol_m=1.0, iou_min=0.0,
    )
    # With 30° rotation and 2 m depth, some grid points should reproject
    # in-bounds. We don't insist on a count — just that the call returns
    # without crashing on a non-ScanNet adapter.
    assert isinstance(matches, list)


def test_voxel_dedup_works_independent_of_adapter():
    s = VoxelSet(0.05)
    assert s.add((0.0, 0.0, 0.0)) is True
    assert s.add((0.01, 0.0, 0.0)) is False
    assert s.add((0.10, 0.0, 0.0)) is True
