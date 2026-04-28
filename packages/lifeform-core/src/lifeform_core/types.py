"""Shared frozen-dataclass types for the lifeform core layer.

These types are NOT part of the kernel's snapshot contract surface — they
describe lifeform-internal coordination state (tick clock, scene lifecycle,
follow-up scheduling). The kernel does not import them.

If a lifeform-internal concept eventually needs to be visible to the kernel
(e.g. for evaluation), promote it to a proper ``vz-cognition`` snapshot
through a normal ``RuntimeModule`` rather than reaching into these dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TickKind(str, Enum):
    """Three metabolic-tick kinds the lifeform fires on its own clock.

    Maps to NL multi-timescale levels (R1) at the lifeform layer:

    * ``SYSTEM`` — fires every ``system_tick_seconds``; coarsest cadence.
      Drives scene staleness checks and follow-up due-time evaluation.
    * ``ENERGY`` — physiological metabolism analogue. Doesn't update kernel
      owners directly; lifeform-evolution telemetry can correlate with PE
      signals to validate "fast adaptation does not block long-window".
    * ``CONTEXT`` — fires when a scene boundary is detected from idle time.
    """

    SYSTEM = "system"
    ENERGY = "energy"
    CONTEXT = "context"


@dataclass(frozen=True)
class TickEvent:
    """One firing of the metabolic tick.

    ``elapsed_seconds`` is monotonic from lifeform start, not wall clock —
    so the same lifeform replayed deterministically produces the same tick
    sequence regardless of host time.
    """

    tick_index: int
    kind: TickKind
    elapsed_seconds: float
    reason: str = ""


class SceneStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass(frozen=True)
class Scene:
    """A single conversational scene.

    A scene is the smallest unit at which slow reflection runs (R6) and at
    which the kernel's ``begin_new_context`` boundary is invoked.

    The kernel does not own scene identity; the lifeform decides when scenes
    open and close based on idle time, explicit user farewells, or external
    triggers. The kernel only sees the resulting ``begin_new_context`` calls.
    """

    scene_id: str
    started_at_tick: int
    closed_at_tick: int | None = None
    turn_count: int = 0
    last_turn_at_tick: int | None = None
    open_loops_at_close: tuple[str, ...] = ()
    commitments_at_close: tuple[str, ...] = ()

    @property
    def status(self) -> SceneStatus:
        return SceneStatus.CLOSED if self.closed_at_tick is not None else SceneStatus.OPEN

    @property
    def is_open(self) -> bool:
        return self.closed_at_tick is None


@dataclass(frozen=True)
class FollowupItem:
    """A scheduled future wake-up.

    Created from an unresolved ``open_loop`` or an ``at-risk`` commitment.
    The lifeform does not auto-execute follow-ups; it surfaces them to the
    product layer (CLI, service) via ``LifeformSession.due_followups()``.
    Whether to actually re-engage the user is a UX decision, not a kernel
    decision.
    """

    followup_id: str
    source: str  # "open_loop" | "commitment" | "scene-end"
    description: str
    due_at_tick: int
    priority: float = 0.5
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnSummary:
    """Compact summary of a single turn for lifeform-side bookkeeping.

    A super-thin readout of what the kernel produced; not a replacement for
    ``AgentTurnResult``. The lifeform keeps these for scene reporting and
    for feeding follow-up decisions.
    """

    turn_index: int
    scene_id: str
    user_input: str
    response_text: str
    active_regime: str | None
    active_abstract_action: str | None
    open_loop_count: int
    commitment_count: int
    pe_magnitude: float
    elapsed_at_tick: int
