"""Adapter ABC + Frame / Reprojection dataclasses.

The core pipeline is dataset-agnostic and only ever talks to adapters via
the interface here. Adding a new dataset = subclass `BaseSceneAdapter`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Frame:
    """Generic per-frame bundle returned by every adapter.

    INTERFACE CONTRACT (all adapters must satisfy):
      * Camera convention: **OpenCV** — camera local axes are +X right,
        +Y down, +Z forward (out of the lens). `pose_c2w` is the 4x4
        rigid transform expressing this camera frame in world coords.
        Adapters whose source data uses a different convention
        (e.g. OpenGL: +Y up, -Z forward) MUST convert at load time.
      * `K_color` is the 3x3 intrinsic in color-image pixel coordinates,
        i.e. `[u, v, 1]^T = K_color @ [X/Z, Y/Z, 1]^T` for camera-local (X,Y,Z).
      * `depth` is in meters. 0 means "invalid" — the pipeline treats 0 as
        a sentinel for missing depth and skips those pixels.
      * `depth` and `K_color` may live at different resolutions; the core
        rescales depth-pixel ↔ color-pixel via `image_size` / `depth_size`.
      * If an adapter has no per-pixel depth (e.g. ScanNet++ DSLR), it
        must override `BaseSceneAdapter.reproject` to use mesh raycast
        or SfM tracks; `depth` may be `None` in that case.
    """
    frame_id: str
    image_path: Path
    image_size: tuple[int, int]          # (W, H) of color image, pixels
    depth: np.ndarray                    # HxW float meters (depth-sensor res); 0 = invalid
    depth_size: tuple[int, int]          # (W_d, H_d)
    K_color: np.ndarray                  # 3x3, in color-image pixel space
    pose_c2w: np.ndarray                 # 4x4 cam-to-world (color sensor, OpenCV)


@dataclass
class Reprojection:
    """Output of reprojecting a source pixel into a target frame."""
    u: float                             # color-pixel coords (target)
    v: float
    depth_pred: float                    # predicted depth in target camera (meters)
    in_bounds: bool


class BaseSceneAdapter(ABC):
    scene_id: str
    # Short tag identifying the dataset this adapter loads from.
    # Used downstream to stratify / weight records by source.
    source_name: str = "unknown"

    @abstractmethod
    def list_frames(self) -> list[str]: ...

    @abstractmethod
    def load_frame(self, frame_id: str) -> Frame: ...

    # Optional cheap image-path lookup. Override for adapters where loading
    # the full Frame is expensive (e.g. ScanNet's depth + pose I/O).
    def image_path(self, frame_id: str) -> Path:
        return self.load_frame(frame_id).image_path

    def frame_ref(self, frame_id: str,
                  adapter_name: Optional[str] = None) -> "FrameRef":
        """Bundle (image_path, adapter, scene_id, frame_id) for cache keying.

        Defaults `adapter_name` to `self.source_name`. Pass an override
        when the CLI exposes a separate adapter flag that must win.
        """
        from models._frame_ref import FrameRef  # local import to avoid cycle
        return FrameRef(
            image_path=self.image_path(frame_id),
            adapter=adapter_name or self.source_name,
            scene_id=self.scene_id, frame_id=frame_id,
        )

    # Optional QC hook — used only by --qc mode to compare against
    # dataset-provided ground truth. Return (HxW int mask, {id: label}) or None.
    def qc_instance_mask(self, frame_id: str) -> Optional[tuple[np.ndarray, dict]]:
        return None

    # Adapters lacking dense per-frame depth (ScanNet++ DSLR, panorama, ...)
    # override this to use a mesh raycast or SfM tracks. Default uses depth.
    def reproject(self, src: Frame, p_src: tuple[float, float],
                  tgt: Frame) -> Optional[Reprojection]:
        return default_reproject_with_depth(src, p_src, tgt)


def _sample_depth(depth: np.ndarray, depth_size: tuple[int, int],
                  image_size: tuple[int, int],
                  u: float, v: float) -> float:
    """Nearest-neighbor depth lookup given a color-pixel (u, v)."""
    W, H = image_size
    Wd, Hd = depth_size
    ud = int(round(u * Wd / W))
    vd = int(round(v * Hd / H))
    if not (0 <= ud < Wd and 0 <= vd < Hd):
        return 0.0
    return float(depth[vd, ud])


def default_reproject_with_depth(
    src: Frame, p_src: tuple[float, float], tgt: Frame
) -> Optional[Reprojection]:
    """Depth-based cross-view reprojection in color-pixel coordinates.

    1. Sample depth at p_src (nearest-neighbor in depth resolution).
    2. Back-project to world via src K, src pose.
    3. Project into target via tgt pose, tgt K.
    """
    u, v = p_src
    z = _sample_depth(src.depth, src.depth_size, src.image_size, u, v)
    if z <= 0.0:
        return None

    K_inv = np.linalg.inv(src.K_color)
    p_cam = K_inv @ np.array([u, v, 1.0]) * z         # (3,) in src camera
    p_h = np.append(p_cam, 1.0)                       # (4,)
    X_world = src.pose_c2w @ p_h                      # (4,)

    pose_w2c_tgt = np.linalg.inv(tgt.pose_c2w)
    p_tgt_cam = (pose_w2c_tgt @ X_world)[:3]          # (3,)
    z_tgt = float(p_tgt_cam[2])
    if z_tgt <= 0.0:                                   # behind the target camera
        return Reprojection(u=-1.0, v=-1.0, depth_pred=z_tgt, in_bounds=False)

    p_img = tgt.K_color @ p_tgt_cam
    u_t = float(p_img[0] / p_img[2])
    v_t = float(p_img[1] / p_img[2])
    W, H = tgt.image_size
    in_bounds = (0.0 <= u_t < W) and (0.0 <= v_t < H)
    return Reprojection(u=u_t, v=v_t, depth_pred=z_tgt, in_bounds=in_bounds)


def world_point_from_pixel(src: Frame, u: float, v: float) -> Optional[np.ndarray]:
    """Helper: 3D world coordinate of a pixel via depth (None if invalid)."""
    z = _sample_depth(src.depth, src.depth_size, src.image_size, u, v)
    if z <= 0.0:
        return None
    K_inv = np.linalg.inv(src.K_color)
    p_cam = K_inv @ np.array([u, v, 1.0]) * z
    return (src.pose_c2w @ np.append(p_cam, 1.0))[:3]
