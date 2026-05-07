"""Mask outline + point marker rendering on matplotlib axes.

Was duplicated across visualize_correspondences.py (lw=2.0),
visualize_perception.py / visualize_pair_match.py / visualize_gt.py
(lw=1.5), and visualize_pairs.py (lw=1.4). The single shared impl
defaults to `lw=1.5` and exposes the parameter for callers that want
thicker outlines (e.g. correspondences viz emphasizes matched pairs).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def draw_mask_outline(ax, mask: np.ndarray, color, *,
                      lw: float = 1.5, ls: str = "-") -> None:
    """Draw a mask silhouette via `ax.contour`.

    Replaces bbox drawing everywhere — bboxes are uninformative for thin
    or wraparound masks (a `wall` instance with two slivers spans the
    whole frame even though the actual visible pixels are thin). Contours
    show the real mask geometry.

    Returns silently if `mask` is None or all-False.
    """
    if mask is None or not mask.any():
        return
    ax.contour(mask.astype(np.uint8), levels=[0.5],
               colors=[color], linewidths=lw, linestyles=ls)


def draw_bbox(ax, bbox, color, *, lw: float = 2.0, ls: str = "-") -> None:
    """Draw an XYXY bbox as a hollow matplotlib Rectangle.

    Used by debug viz that doesn't have a mask in hand (raw detector
    output, manifest objects). Returns silently for missing / sentinel
    bboxes (``None`` or first coord ``< 0``).
    """
    if bbox is None or bbox[0] < 0:
        return
    # Local import keeps the module otherwise pyplot-free for fast
    # imports in headless tests.
    from matplotlib.patches import Rectangle
    x0, y0, x1, y1 = bbox
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                           fill=False, ec=color, lw=lw, ls=ls))


def mask_centroid(mask: np.ndarray) -> Optional[tuple[int, int]]:
    """Median of mask pixel coordinates.

    Median (not mean) is robust to weird shapes — for a U-shaped table
    mask, the mean falls inside the hole; the median lands on the
    table itself.
    """
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(np.median(xs)), int(np.median(ys))


def draw_src_point(ax, u: float, v: float, color, text: str) -> None:
    """Source-side point marker: filled colored circle with text label.

    Used for `point_src` in correspondence visualizations. Always
    visible — the source point is always observed (depth and image
    pixel are both valid).
    """
    ax.plot(u, v, "o", ms=10, mec="black", mfc=color, mew=1.5)
    ax.annotate(text, (u, v), xytext=(8, -8), textcoords="offset points",
                fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.75))


def draw_tgt_point(ax, u: float, v: float, visible: bool, text: str) -> None:
    """Target-side point marker.

    Visible matches: green ring; occluded matches (`visible=False`):
    red X. The contrast between the two visual states is intentional —
    occlusion negatives are an interesting signal and should pop in
    debug viz.
    """
    if visible:
        ax.plot(u, v, "o", ms=12, mec="lime", mfc="none", mew=2.5)
        fc = "darkgreen"
    else:
        ax.plot(u, v, "x", ms=14, mec="red", mew=3.0)
        fc = "darkred"
    ax.annotate(text, (u, v), xytext=(8, -8), textcoords="offset points",
                fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc=fc, alpha=0.75))
