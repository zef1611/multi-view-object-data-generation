"""Default label blocklist for object-level reasoning.

Used by the COSMIC visibility-set gate (`pipeline.sampling.cosmic`) and
the GT-driven detectors (`models.gt`). Structural surfaces (walls,
floors, ceilings, door frames) are excluded so downstream skills reason
over discrete objects, not surfaces.
"""

from __future__ import annotations

DEFAULT_LABEL_BLOCKLIST: frozenset[str] = frozenset({
    "wall", "floor", "ceiling", "doorframe", "door frame",
})
