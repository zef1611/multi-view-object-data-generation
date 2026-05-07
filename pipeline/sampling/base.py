"""Sampling strategy protocol.

Concrete implementations live in sibling files. Each takes the adapter
plus strategy-specific kwargs and returns a list of selected frame IDs.

The orchestrator (`pipeline.sampling.sample_keyframes`) dispatches by
strategy name; this protocol is a marker for type checkers, not a
strict ABC — we keep duck typing because the concrete signatures
diverge (adaptive needs the adapter; stride only needs the frame
list).
"""

from __future__ import annotations

from typing import Protocol


class SamplingStrategy(Protocol):
    def __call__(self, *args, **kwargs) -> list[str]: ...
