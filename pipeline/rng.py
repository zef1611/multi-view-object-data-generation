"""Seeded RNG factory.

Per-scene sub-seed = base_seed XOR hash(scene_id) so reruns are
deterministic per scene without entangling scenes.
"""

import hashlib
import random


def make_rng(base_seed: int, scene_id: str) -> random.Random:
    h = int(hashlib.sha1(scene_id.encode()).hexdigest()[:8], 16)
    return random.Random(base_seed ^ h)
