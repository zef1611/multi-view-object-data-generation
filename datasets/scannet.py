"""ScanNet adapter.

Layout (per scene `<root>/<scene_id>/`):
  color/{i}.jpg               1296x968 RGB
  depth/{i}.png               640x480 uint16 mm
  pose/{i}.txt                4x4 cam-to-world (color sensor)
  intrinsic/intrinsic_color.txt   3x3 K_color (4x4 padded)
  intrinsic/extrinsic_color.txt   identity in practice; ignored
  <scene_id>.txt              meta with fx/fy/mx/my for color & depth
  *.aggregation.json          {segGroups: [{objectId, label, ...}]}
  instance-filt/{i}.png       OPTIONAL, color-resolution objectId mask
                              (extracted from <scene_id>_2d-instance-filt.zip)

We treat `pose/{i}.txt` as the color-sensor pose in world. The
color-to-depth offset (~3-4 cm) is small relative to our 0.1 m visibility
tolerance and is ignored, matching common ScanNet usage.
"""

import json
import re
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .base import BaseSceneAdapter, Frame


_META_RE = re.compile(r"^(\w+)\s*=\s*(.+)$")


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


class ScanNetAdapter(BaseSceneAdapter):
    source_name = "scannet"

    def __init__(self, scene_dir: Path):
        self.scene_dir = Path(scene_dir)
        self.scene_id = self.scene_dir.name

        meta_path = self.scene_dir / f"{self.scene_id}.txt"
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
        self._instance_dir = self.scene_dir / "instance-filt"
        self._instance_zip = (
            self.scene_dir / f"{self.scene_id}_2d-instance-filt.zip"
        )
        self._aggregation_path = self.scene_dir / f"{self.scene_id}.aggregation.json"

    # -------- frame enumeration ---------------------------------------------

    def list_frames(self) -> list[str]:
        ids = []
        for p in self._color_dir.glob("*.jpg"):
            try:
                ids.append(int(p.stem))
            except ValueError:
                continue
        ids.sort()
        return [str(i) for i in ids]

    def load_pose(self, frame_id: str) -> np.ndarray:
        """Cheap pose-only load — used by adaptive frame selection."""
        return _parse_matrix(
            (self._pose_dir / f"{frame_id}.txt").read_text(), (4, 4)
        )

    def image_path(self, frame_id: str):
        return self._color_dir / f"{frame_id}.jpg"

    def load_frame(self, frame_id: str) -> Frame:
        img_path = self._color_dir / f"{frame_id}.jpg"
        depth_path = self._depth_dir / f"{frame_id}.png"
        pose_path = self._pose_dir / f"{frame_id}.txt"

        depth_mm = np.array(Image.open(depth_path), dtype=np.uint16)
        depth_m = depth_mm.astype(np.float32) / 1000.0    # mm -> meters

        pose = _parse_matrix(pose_path.read_text(), (4, 4))
        # Some scans contain -inf rows for failed tracking. Detect & flag invalid.
        if not np.all(np.isfinite(pose)):
            pose = np.full((4, 4), np.nan)

        return Frame(
            frame_id=frame_id,
            image_path=img_path,
            image_size=self.color_size,
            depth=depth_m,
            depth_size=self.depth_size,
            K_color=self.K_color,
            pose_c2w=pose,
        )

    # -------- optional QC ---------------------------------------------------

    @lru_cache(maxsize=1)
    def _aggregation(self) -> dict[int, str]:
        """objectId-keyed label map, **shifted to match instance-filt PNG values**.

        ScanNet's `aggregation.json` numbers objects 0..N-1, but the
        `instance-filt/{i}.png` mask uses 1..N (with 0 reserved for
        background / no-label). We add 1 here so callers can do
        `label_map[mask[y, x]]` directly without an off-by-one.
        """
        if not self._aggregation_path.exists():
            return {}
        data = json.loads(self._aggregation_path.read_text())
        return {int(g["objectId"]) + 1: str(g["label"])
                for g in data.get("segGroups", [])}

    def _read_instance_png(self, frame_id: str) -> Optional[np.ndarray]:
        """Read instance-filt/{frame}.png. Falls back to extracting from zip."""
        p = self._instance_dir / f"{frame_id}.png"
        if p.exists():
            return np.array(Image.open(p))
        if self._instance_zip.exists():
            try:
                with zipfile.ZipFile(self._instance_zip) as zf:
                    # ScanNet zips use either "instance-filt/{i}.png" or "{i}.png"
                    candidates = (f"instance-filt/{frame_id}.png",
                                  f"{frame_id}.png")
                    for name in candidates:
                        if name in zf.namelist():
                            with zf.open(name) as f:
                                return np.array(Image.open(f))
            except (zipfile.BadZipFile, OSError):
                return None
        return None

    def qc_instance_mask(self, frame_id: str) -> Optional[tuple[np.ndarray, dict]]:
        mask = self._read_instance_png(frame_id)
        if mask is None:
            return None
        return mask, self._aggregation()
