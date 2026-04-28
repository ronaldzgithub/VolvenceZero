"""EtiquetteWatchdog — UX-only verdict on whether to speak right now.

Pure rule-based, deterministic, no LLM. Intentionally **not** a modification
gate: it does NOT block the kernel from learning. It only advises the
product layer (CLI / service) on whether emitting *another* turn is
appropriate.

Why is this in lifeform-expression and not lifeform-core?

* It is fundamentally about *expression timing* (when to use one's voice).
* Hard self-modification gating belongs to ``vz-cognition.credit.ModificationGate``.
* Conversation-initiation policy is a cross-cutting product concern; it
  reads scene state from lifeform-core but produces a verdict for whoever
  is actually about to call ``run_turn``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from lifeform_core.types import Scene, TickEvent, TickKind


class SpeakVerdict(str, Enum):
    SPEAK = "speak"
    WAIT = "wait"
    STAY_SILENT = "stay-silent"


@dataclass(frozen=True)
class EtiquetteVerdict:
    verdict: SpeakVerdict
    reason: str
    cooldown_remaining_ticks: int = 0


class EtiquetteWatchdog:
    """Stateful — tracks how recently the lifeform last spoke."""

    def __init__(
        self,
        *,
        min_cooldown_ticks: int = 2,
        max_consecutive_proactive_turns: int = 1,
        quiet_hours_start_tick: int | None = None,
        quiet_hours_end_tick: int | None = None,
    ) -> None:
        """Args:
            min_cooldown_ticks: minimum SYSTEM ticks between spoken turns.
            max_consecutive_proactive_turns: cap on how many follow-up
                proactive turns can be emitted without an intervening user
                turn (prevents runaway "I'm checking in!" spirals).
            quiet_hours_start_tick / quiet_hours_end_tick: optional periodic
                quiet window — both must be set to enable. The window is on
                the lifeform's monotonic tick clock, not wall time.
        """
        self._min_cooldown = min_cooldown_ticks
        self._max_consecutive_proactive = max_consecutive_proactive_turns
        self._quiet_start = quiet_hours_start_tick
        self._quiet_end = quiet_hours_end_tick
        self._last_spoken_tick: int | None = None
        self._consecutive_proactive_count = 0

    @property
    def last_spoken_tick(self) -> int | None:
        return self._last_spoken_tick

    @property
    def consecutive_proactive_count(self) -> int:
        return self._consecutive_proactive_count

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def evaluate(
        self,
        *,
        current_tick: int,
        scene: Scene | None = None,
        is_proactive: bool = False,
    ) -> EtiquetteVerdict:
        """Decide whether the lifeform should speak now.

        ``is_proactive=True`` means the speak-decision was triggered by the
        lifeform side (e.g. a follow-up due) rather than by the user.
        """
        if self._is_in_quiet_hours(current_tick):
            return EtiquetteVerdict(
                verdict=SpeakVerdict.STAY_SILENT,
                reason="in-quiet-hours",
            )

        if self._last_spoken_tick is not None:
            elapsed = current_tick - self._last_spoken_tick
            if elapsed < self._min_cooldown:
                return EtiquetteVerdict(
                    verdict=SpeakVerdict.WAIT,
                    reason="cooldown",
                    cooldown_remaining_ticks=self._min_cooldown - elapsed,
                )

        if is_proactive and self._consecutive_proactive_count >= self._max_consecutive_proactive:
            return EtiquetteVerdict(
                verdict=SpeakVerdict.WAIT,
                reason="proactive-cap-reached",
            )

        # Avoid speaking if the scene is empty and the call is proactive — we
        # don't want the lifeform to wake up and start monologuing.
        if is_proactive and (scene is None or scene.turn_count == 0):
            return EtiquetteVerdict(
                verdict=SpeakVerdict.STAY_SILENT,
                reason="no-active-scene",
            )

        return EtiquetteVerdict(verdict=SpeakVerdict.SPEAK, reason="ok")

    # ------------------------------------------------------------------
    # Bookkeeping — call after a turn is actually emitted
    # ------------------------------------------------------------------

    def record_spoken(self, *, current_tick: int, was_proactive: bool) -> None:
        self._last_spoken_tick = current_tick
        if was_proactive:
            self._consecutive_proactive_count += 1
        else:
            self._consecutive_proactive_count = 0

    def on_tick(self, event: TickEvent) -> None:
        # Reserved for future periodic decay, e.g. re-allow proactive turns
        # after long inactivity. Currently a no-op.
        if event.kind is TickKind.SYSTEM:
            return

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_in_quiet_hours(self, current_tick: int) -> bool:
        if self._quiet_start is None or self._quiet_end is None:
            return False
        if self._quiet_end > self._quiet_start:
            return self._quiet_start <= current_tick < self._quiet_end
        # Wrap-around (e.g. nightly quiet hours that span midnight on the
        # tick clock).
        return current_tick >= self._quiet_start or current_tick < self._quiet_end
