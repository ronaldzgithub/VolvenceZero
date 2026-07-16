"""Bounded-learned PE write gate for :class:`MemoryStore` (#89 residual).

Historically ``apply_prediction_error_signal`` wrote a prediction-error
memory whenever ``magnitude >= 0.15`` — a fixed, hand-crafted admission
threshold. This module turns that threshold into the *initialisation and
rollback point* of a bounded-learned gate, following the same calibrator
pattern the PE owner uses for severity/bias tables (C2):

- Every PE-driven write is parked with the strength it was written at.
- Two PE turns later the gate settles the write against the realized
  usefulness of the entry inside the store the gate's owner already
  controls: an entry that was retrieved again (touched without net
  strength loss) or promoted counts as *useful*; an entry that decayed,
  was deleted, or was never revisited counts as *unused*.
- Useful writes nudge the threshold down (admit more borderline PE);
  unused writes nudge it up (admit less noise). Drift is clamped to a
  hard envelope around the initial threshold so the gate can never
  silently collapse into "write everything" or "write nothing".

The gate is owner-internal state of the memory store (R5/R6): settlement
only reads entries the store itself owns, and the readout is published
through ``MemorySnapshot.lifecycle_metrics``. ``reset()`` restores the
historical fixed-threshold behaviour exactly (rollback path).
"""

from __future__ import annotations

from typing import Mapping

from volvence_zero.memory.contracts import MemoryEntry

PE_WRITE_GATE_INITIAL_THRESHOLD = 0.15


class PeWriteGate:
    """Bounded-learned admission threshold for PE-driven memory writes."""

    _ENVELOPE = 0.10
    _LEARNING_RATE = 0.01
    _SETTLE_DELAY_TURNS = 2

    def __init__(self) -> None:
        self._threshold = PE_WRITE_GATE_INITIAL_THRESHOLD
        self._turn_count = 0
        # (entry_id, parked_at_turn, strength_at_write)
        self._pending: list[tuple[str, int, float]] = []
        self._settled_useful = 0
        self._settled_unused = 0

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def settled_useful_count(self) -> int:
        return self._settled_useful

    @property
    def settled_unused_count(self) -> int:
        return self._settled_unused

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def begin_turn(self) -> None:
        self._turn_count += 1

    def should_write(self, magnitude: float) -> bool:
        return magnitude >= self._threshold

    def record_write(self, *, entry_id: str, strength: float) -> None:
        self._pending.append((entry_id, self._turn_count, float(strength)))

    def settle(self, entries: Mapping[str, MemoryEntry]) -> tuple[int, int]:
        """Settle due pending writes against realized entry usefulness.

        Returns ``(useful, unused)`` counts settled on this call. An
        entry is *useful* when it still exists, has not lost net
        strength (rules out the decay-path ``last_accessed_ms`` touch),
        and was accessed after creation (retrieval touch or promotion).
        """

        if not self._pending:
            return (0, 0)
        due: list[tuple[str, int, float]] = []
        remaining: list[tuple[str, int, float]] = []
        for item in self._pending:
            if self._turn_count - item[1] >= self._SETTLE_DELAY_TURNS:
                due.append(item)
            else:
                remaining.append(item)
        if not due:
            return (0, 0)
        useful = 0
        unused = 0
        for entry_id, _parked_at, written_strength in due:
            entry = entries.get(entry_id)
            if (
                entry is not None
                and entry.strength >= written_strength
                and entry.last_accessed_ms > entry.created_at_ms
            ):
                useful += 1
            else:
                unused += 1
        self._settled_useful += useful
        self._settled_unused += unused
        candidate = self._threshold + self._LEARNING_RATE * (unused - useful)
        self._threshold = self._clamp_to_envelope(candidate)
        self._pending = remaining
        return (useful, unused)

    def restore_threshold(self, value: float) -> None:
        """Restore a checkpointed threshold (pending writes do not survive)."""

        self._threshold = self._clamp_to_envelope(float(value))
        self._pending = []

    def reset(self) -> None:
        """Rollback to the historical fixed-threshold behaviour."""

        self._threshold = PE_WRITE_GATE_INITIAL_THRESHOLD
        self._pending = []
        self._settled_useful = 0
        self._settled_unused = 0

    @classmethod
    def _clamp_to_envelope(cls, value: float) -> float:
        low = PE_WRITE_GATE_INITIAL_THRESHOLD - cls._ENVELOPE
        high = PE_WRITE_GATE_INITIAL_THRESHOLD + cls._ENVELOPE
        return max(low, min(high, value))


__all__ = ["PE_WRITE_GATE_INITIAL_THRESHOLD", "PeWriteGate"]
