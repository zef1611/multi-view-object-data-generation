"""Vectorized projection helpers for IoU-based mask matching.

The per-point reprojection lives in `adapters.base.default_reproject_with_depth`
and is invoked through `adapter.reproject(src, p_src, tgt)`. This module
provides the bulk version: project every "in-mask" pixel from src into tgt
to compute reprojected-mask IoU.
"""

from __future__ import annotations

import numpy as np

from datasets.base import Frame


def reproject_mask(src: Frame, mask_src: np.ndarray, tgt: Frame,
                   subsample: int = 4) -> np.ndarray:
    """Project every src-mask pixel into the target color frame.

    Returns a binary mask in target color resolution (H_t, W_t) marking
    reprojected hits. `subsample` strides the src mask to keep this fast
    (dropping every Nth pixel; default 4 = 16x speedup, ~minor IoU error).
    """
    W, H = src.image_size
    Wt, Ht = tgt.image_size
    Wd, Hd = src.depth_size

    ys, xs = np.where(mask_src > 0)
    if subsample > 1:
        ys = ys[::subsample]
        xs = xs[::subsample]
    if xs.size == 0:
        return np.zeros((Ht, Wt), dtype=bool)

    # Sample depth at color pixels via NN downscale.
    xd = np.clip(np.round(xs * (Wd / W)).astype(int), 0, Wd - 1)
    yd = np.clip(np.round(ys * (Hd / H)).astype(int), 0, Hd - 1)
    z = src.depth[yd, xd]
    keep = z > 0
    xs, ys, z = xs[keep], ys[keep], z[keep]
    if xs.size == 0:
        return np.zeros((Ht, Wt), dtype=bool)

    # Back-project to camera coords.
    K_inv = np.linalg.inv(src.K_color)
    pts_h = np.stack([xs.astype(np.float64), ys.astype(np.float64),
                      np.ones_like(xs, dtype=np.float64)], axis=0)  # (3, N)
    cam = (K_inv @ pts_h) * z[None, :]                              # (3, N)
    world = src.pose_c2w @ np.vstack([cam, np.ones((1, cam.shape[1]))])  # (4, N)

    # Project into target.
    w2c_t = np.linalg.inv(tgt.pose_c2w)
    cam_t = (w2c_t @ world)[:3]                                     # (3, N)
    in_front = cam_t[2] > 0
    cam_t = cam_t[:, in_front]
    if cam_t.shape[1] == 0:
        return np.zeros((Ht, Wt), dtype=bool)

    img = tgt.K_color @ cam_t                                       # (3, N)
    u = img[0] / img[2]
    v = img[1] / img[2]
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    ok = (ui >= 0) & (ui < Wt) & (vi >= 0) & (vi < Ht)
    out = np.zeros((Ht, Wt), dtype=bool)
    out[vi[ok], ui[ok]] = True
    return out


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0
