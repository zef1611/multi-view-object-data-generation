"""Matterport3D adapter.

Consumes the ScanNet-like layout produced by
`matterport3d-data-gen/scripts/convert_matterport_to_scannet_layout.py`.

Layout (per scene `<root>/<scene_id>/`):
  color/{i}.jpg                 1280x1024 RGB (symlink to source)
  depth/{i}.png                 1280x1024 uint16, 0.25 mm/unit (÷ 4000 → meters)
  pose/{i}.txt                  4x4 cam-to-world, OpenCV axes
  intrinsic/intrinsic_color.txt 4x4 padded K (camera-1 / level-tilt, global)
  intrinsic_per_frame/{i}.txt   4x4 padded K per frame (cameras i0/i1/i2 differ)
  scene_meta.txt                ScanNet-style meta header
  filelist.json                 per-frame metadata (viewpoint/camera/yaw)

Differences vs ScanNet:
  * Depth scale is 4000 (not 1000). We divide by 4000 here.
  * Color and depth are the same resolution (1280x1024) and already aligned.
  * Three cameras per pod have slightly different K; we prefer per-frame K
    when `intrinsic_per_frame/` exists, else fall back to the global K.
  * No *.aggregation.json / instance-filt PNGs — QC returns None.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .base import BaseSceneAdapter, Frame


_META_RE = re.compile(r"^(\w+)\s*=\s*(.+)$")

DEPTH_SCALE = 4000.0  # uint16 unit = 0.25 mm


def _parse_meta(meta_path: Path) -> dict:
    out = {}
    for line in meta_path.read_text().splitlines():
        m = _META_RE.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def _parse_matrix(text: str, shape: tuple[int, int]) -> np.ndarray:
    nums = [float(x) for x in text.replace(",", " ").split()]
    return np.array(nums, dtype=np.float64).reshape(shape)


class MatterportAdapter(BaseSceneAdapter):
    source_name = "matterport"

    def __init__(self, scene_dir: Path):
        self.scene_dir = Path(scene_dir)
        self.scene_id = self.scene_dir.name

        meta_path = self.scene_dir / "scene_meta.txt"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta: {meta_path}")
        meta = _parse_meta(meta_path)

        self.color_size = (int(meta["colorWidth"]), int(meta["colorHeight"]))
        self.depth_size = (int(meta["depthWidth"]), int(meta["depthHeight"]))

        K_path = self.scene_dir / "intrinsic" / "intrinsic_color.txt"
        K = _parse_matrix(K_path.read_text(), (4, 4))[:3, :3]
        self.K_color = K

        self._color_dir = self.scene_dir / "color"
        self._depth_dir = self.scene_dir / "depth"
        self._pose_dir = self.scene_dir / "pose"
        self._per_frame_K_dir = self.scene_dir / "intrinsic_per_frame"
        self._has_per_frame_K = self._per_frame_K_dir.exists()

    # -------- frame enumeration ---------------------------------------------

    def list_frames(self) -> list[str]:
        ids = []
        for p in self._color_dir.iterdir():
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                ids.append(int(p.stem))
            except ValueError:
                continue
        ids.sort()
        return [str(i) for i in ids]

    def load_pose(self, frame_id: str) -> np.ndarray:
        return _parse_matrix(
            (self._pose_dir / f"{frame_id}.txt").read_text(), (4, 4)
        )

    def image_path(self, frame_id: str):
        return self._color_dir / f"{frame_id}.jpg"

    @lru_cache(maxsize=4096)
    def _K_for_frame(self, frame_id: str) -> np.ndarray:
        if self._has_per_frame_K:
            p = self._per_frame_K_dir / f"{frame_id}.txt"
            if p.exists():
                return _parse_matrix(p.read_text(), (4, 4))[:3, :3]
        return self.K_color

    def load_frame(self, frame_id: str) -> Frame:
        img_path = self._color_dir / f"{frame_id}.jpg"
        depth_path = self._depth_dir / f"{frame_id}.png"
        pose_path = self._pose_dir / f"{frame_id}.txt"

        depth_raw = np.array(Image.open(depth_path), dtype=np.uint16)
        depth_m = depth_raw.astype(np.float32) / DEPTH_SCALE

        pose = _parse_matrix(pose_path.read_text(), (4, 4))
        if not np.all(np.isfinite(pose)):
            pose = np.full((4, 4), np.nan)

        return Frame(
            frame_id=frame_id,
            image_path=img_path,
            image_size=self.color_size,
            depth=depth_m,
            depth_size=self.depth_size,
            K_color=self._K_for_frame(frame_id),
            pose_c2w=pose,
        )

    # No dataset-shipped 2D instance GT in this Matterport layout.
    def qc_instance_mask(self, frame_id: str) -> Optional[tuple[np.ndarray, dict]]:
        return None
