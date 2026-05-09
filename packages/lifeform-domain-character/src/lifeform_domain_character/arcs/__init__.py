"""Reviewed narrative arcs shipped with the wheel.

Arcs are reviewed structured artifacts encoding a character's lived
sequence of decision-point scenes (see ``../narrative.py``). They drive
the :class:`ExperientialReplayDriver`; they are NOT free-form prose.
"""

from __future__ import annotations

from lifeform_domain_character.arcs.zhang_wuji_demo_arc import (
    build_zhang_wuji_demo_arc,
)


__all__ = [
    "build_zhang_wuji_demo_arc",
]
