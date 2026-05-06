"""``InterlocutorStateModule`` - SHADOW owner of the 12-axis readout.

Reads the same six upstream snapshots that the duck-typed builder
consumes (``regime``, ``dual_track``, ``evaluation``,
``prediction_error``, ``memory``, ``commitment``), runs the pure
:func:`readout_interlocutor_state` function, and publishes a frozen
:class:`InterlocutorStateSnapshot`.

Wave 2 SSOT cleanup:

* Before W2: each downstream consumer (``prompt_planner`` /
  ``response_synthesizer`` / ``LifeformSession.interlocutor_state``)
  re-built the readout context independently by reaching into the
  six upstream snapshots. Three reconstruction paths -> three
  potential schema drifts.
* After W2: this owner is the SINGLE producer. Consumers read the
  ``interlocutor_state`` slot.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel

from volvence_zero.interlocutor.contracts import (
    InterlocutorState,
    InterlocutorStateSnapshot,
)
from volvence_zero.interlocutor.readout import (
    build_interlocutor_readout_context_from_snapshots,
    readout_interlocutor_state,
)


class InterlocutorStateModule(RuntimeModule[InterlocutorStateSnapshot]):
    """SHADOW owner of the 12-axis interlocutor readout (W2)."""

    slot_name: ClassVar[str] = "interlocutor_state"
    owner: ClassVar[str] = "InterlocutorStateModule"
    value_type: ClassVar[type[Any]] = InterlocutorStateSnapshot
    dependencies: ClassVar[tuple[str, ...]] = (
        "regime",
        "dual_track",
        "evaluation",
        "prediction_error",
        "memory",
        "commitment",
    )
    default_wiring_level: ClassVar[WiringLevel] = WiringLevel.SHADOW

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[InterlocutorStateSnapshot]:
        ctx = build_interlocutor_readout_context_from_snapshots(
            regime_snapshot=upstream.get("regime"),
            dual_track_snapshot=upstream.get("dual_track"),
            evaluation_snapshot=upstream.get("evaluation"),
            prediction_error_snapshot=upstream.get("prediction_error"),
            memory_snapshot=upstream.get("memory"),
            commitment_snapshot=upstream.get("commitment"),
        )
        state = readout_interlocutor_state(ctx)
        snapshot = InterlocutorStateSnapshot(
            state=state,
            description=_describe(state),
        )
        return self.publish(snapshot)


def _describe(state: InterlocutorState) -> str:
    """Owner-authored short description used in dashboards / logs."""

    zones: list[str] = []
    if state.repair_zone:
        zones.append("repair")
    if state.direct_task_zone:
        zones.append("direct_task")
    if state.emotional_render_zone:
        zones.append("emotional")
    if state.pace_pressure_zone:
        zones.append("pace_pressure")
    if state.cold_rapport_zone:
        zones.append("cold_rapport")
    if state.acknowledge_pressure_zone:
        zones.append("acknowledge_pressure")
    zone_str = ",".join(zones) if zones else "neutral"
    return (
        "InterlocutorStateModule readout "
        f"confidence={state.readout_confidence:.2f} "
        f"zones={zone_str} "
        f"trust={state.trust_signal:+.2f} "
        f"emotional={state.emotional_weight:.2f} "
        f"resistance={state.resistance_level:.2f}"
    )


__all__ = ["InterlocutorStateModule"]
