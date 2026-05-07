"""Single source of truth for the visualization color palette.

Was duplicated across 6+ scripts (visualize_correspondences.py,
visualize_perception.py, visualize_pair_match.py, visualize_pairs.py,
visualize_gt.py, dryrun_inspect.py) with identical implementation.
"""

from __future__ import annotations

import hashlib

import matplotlib.pyplot as plt


# tab20 has 20 distinct, perceptually distinguishable colors — enough headroom
# for a typical scene's object inventory without recycling. Using `.colors`
# (not the colormap object) lets us index by integer modulo 20.
PALETTE = plt.get_cmap("tab20").colors


def color_for(key: str):
    """Deterministic SHA1-based color picker.

    Same `key` always maps to the same color, so a "chair" mask in frame 0
    and frame 216 are colored identically — important for visual matching.
    Different label strings hash to different palette slots.
    """
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)
    return PALETTE[h % len(PALETTE)]
