"""ProtocolPhaseModule (packet 5.0): PE-driven TemporalArc phase advancement.

Each loaded ``BehaviorProtocol`` declares a ``TemporalArc`` of
named phases (e.g. ``icebreaker`` → ``trust_building`` →
``value_anchor`` → ``long_term_companion``). Phase advancement
is driven by typed evidence — never by calendar / session day
counters.

The module:

1. Reads ``prediction_error`` / ``interlocutor_state`` / ``regime``
   / ``rupture_state`` upstream snapshots.
2. For each protocol in the registry, looks up the current phase
   pointer and evaluates the phase's ``exit_conditions`` (and
   the candidate next phase's ``entry_conditions``) against
   the typed signal evidence.
3. Counts consecutive firing turns per progression_signal; when
   ``ProgressionSignal.threshold`` (interpreted as min
   consecutive fires) is reached, the phase advances.
4. Publishes ``ProtocolPhaseSnapshot`` mapping protocol_id →
   current_phase_id + turns_in_current_phase.

Backwards compatibility: protocols with empty
``progression_signals`` pin at the first phase forever (a
``"long_term_companion"``-shape default for cheng_laoshi-like
fixtures). Empty registry → empty snapshot.

Owner placement: ``vz-application.protocol_runtime`` — this is
the same wheel as ``ProtocolRegistryModule`` (which consumes
``protocol_phase`` to fill ``current_phase_id``). The module
receives a registry handle via constructor injection (mirrors
how the registry receives application stores).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, ClassVar, Mapping

from volvence_zero.application.types import BoundaryPolicySnapshot
from volvence_zero.behavior_protocol import (
    BehaviorProtocol,
    BehaviorProtocolSignalSource,
    ProgressionSignal,
    ProtocolPhaseSnapshot,
    TemporalPhase,
)
from volvence_zero.dual_track import DualTrackSnapshot  # noqa: F401 (future use)
from volvence_zero.interlocutor import InterlocutorStateSnapshot
from volvence_zero.prediction import PredictionErrorSnapshot
from volvence_zero.regime import RegimeSnapshot
from volvence_zero.rupture_state import RuptureKind, RuptureStateSnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel

from volvence_zero.protocol_runtime.registry import ProtocolRegistry


# Floor: even when threshold is 0 / unset, require at least one
# observed firing turn before advancing the phase. Belt-and-suspenders
# guard against degenerate ``threshold=0`` ProgressionSignal entries
# auto-advancing on cold start.
_MIN_OBSERVED_FIRES: int = 1


class ProtocolPhaseModule(RuntimeModule[ProtocolPhaseSnapshot]):
    """SHADOW-default owner of the ``protocol_phase`` slot."""

    slot_name: ClassVar[str] = "protocol_phase"
    owner: ClassVar[str] = "ProtocolPhaseModule"
    value_type: ClassVar[type[Any]] = ProtocolPhaseSnapshot
    dependencies: ClassVar[tuple[str, ...]] = (
        "prediction_error",
        "interlocutor_state",
        "regime",
        "rupture_state",
        "boundary_policy",
    )
    default_wiring_level: ClassVar[WiringLevel] = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        wiring_level: WiringLevel | None = None,
        registry: ProtocolRegistry,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._registry = registry
        # Per-protocol current phase pointer and tenure.
        self._current_phase: dict[str, str] = {}
        self._turns_in_phase: dict[str, int] = {}
        # Per-(protocol, signal_id) consecutive-firing counter,
        # reset whenever the signal stops firing.
        self._fire_streaks: dict[tuple[str, str], int] = defaultdict(int)
        # De-dupe: skip duplicate / out-of-order PE turn_index.
        self._last_pe_turn_index: int | None = None

    @property
    def current_phase(self) -> Mapping[str, str]:
        """Read-only view of per-protocol current phase pointer."""
        return dict(self._current_phase)

    @property
    def turns_in_phase(self) -> Mapping[str, int]:
        return dict(self._turns_in_phase)

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[ProtocolPhaseSnapshot]:
        # Skip duplicate / replay PE turns. (PE is the canonical
        # tick counter for phase logic.)
        pe_turn_index = _read_pe_turn_index(upstream)
        if pe_turn_index is not None:
            if (
                self._last_pe_turn_index is not None
                and pe_turn_index <= self._last_pe_turn_index
            ):
                return self.publish(self._build_snapshot())
            self._last_pe_turn_index = pe_turn_index

        interlocutor = _read_snapshot(
            upstream, "interlocutor_state", InterlocutorStateSnapshot
        )
        rupture = _read_snapshot(upstream, "rupture_state", RuptureStateSnapshot)
        regime = _read_snapshot(upstream, "regime", RegimeSnapshot)
        boundary = _read_snapshot(
            upstream, "boundary_policy", BoundaryPolicySnapshot
        )

        loaded = self._registry.loaded()
        # Drop tracking for protocols no longer loaded.
        loaded_ids = {p.protocol_id for p in loaded}
        for stale in list(self._current_phase):
            if stale not in loaded_ids:
                del self._current_phase[stale]
                self._turns_in_phase.pop(stale, None)
        for streak_key in list(self._fire_streaks):
            if streak_key[0] not in loaded_ids:
                del self._fire_streaks[streak_key]

        for protocol in loaded:
            self._evaluate_protocol(
                protocol,
                interlocutor=interlocutor,
                rupture=rupture,
                regime=regime,
                boundary=boundary,
            )

        return self.publish(self._build_snapshot())

    # ------------------------------------------------------------------
    # Per-protocol evaluation
    # ------------------------------------------------------------------

    def _evaluate_protocol(
        self,
        protocol: BehaviorProtocol,
        *,
        interlocutor: InterlocutorStateSnapshot | None,
        rupture: RuptureStateSnapshot | None,
        regime: RegimeSnapshot | None,
        boundary: BoundaryPolicySnapshot | None,
    ) -> None:
        phases = protocol.temporal_arc.phases
        if not phases:
            return  # Nothing to track; current_phase stays absent.

        # Initialise phase pointer to first phase if unset.
        protocol_id = protocol.protocol_id
        if protocol_id not in self._current_phase:
            self._current_phase[protocol_id] = phases[0].phase_id
            self._turns_in_phase[protocol_id] = 0

        # Increment tenure each evaluated turn.
        self._turns_in_phase[protocol_id] = (
            self._turns_in_phase.get(protocol_id, 0) + 1
        )

        # If progression_signals are absent at all 3 levels (arc + phase
        # entry/exit), the protocol is using packet-1.0 default shape:
        # phase pinned to phases[0] forever (cheng_laoshi backwards
        # compat).
        if not _has_any_progression_signal(protocol):
            return

        current_phase_id = self._current_phase[protocol_id]
        current_phase = _phase_by_id(phases, current_phase_id)
        if current_phase is None:
            # Phase pointer drifted to an unknown id; reset to first.
            self._current_phase[protocol_id] = phases[0].phase_id
            self._turns_in_phase[protocol_id] = 0
            return

        # Update fire streaks per signal evaluated this turn.
        all_signals = (
            tuple(current_phase.entry_conditions)
            + tuple(current_phase.exit_conditions)
            + tuple(protocol.temporal_arc.progression_signals)
        )
        for signal in all_signals:
            firing = _signal_is_firing(
                signal,
                interlocutor=interlocutor,
                rupture=rupture,
                regime=regime,
                boundary=boundary,
            )
            key = (protocol_id, signal.signal_id)
            if firing:
                self._fire_streaks[key] = self._fire_streaks.get(key, 0) + 1
            else:
                self._fire_streaks[key] = 0

        # Try to ADVANCE: if the current phase's exit_conditions are
        # met (any one with firing streak ≥ threshold), or the next
        # phase's entry_conditions are met, advance pointer.
        next_phase = _phase_after(phases, current_phase_id)
        if next_phase is None:
            return  # Already at terminal phase.

        if self._exit_condition_met(protocol_id, current_phase) or (
            self._entry_condition_met(protocol_id, next_phase)
        ):
            self._current_phase[protocol_id] = next_phase.phase_id
            self._turns_in_phase[protocol_id] = 0
            # Reset all streaks for this protocol on phase change to
            # avoid carry-over re-triggering from stale fires.
            for key in list(self._fire_streaks):
                if key[0] == protocol_id:
                    self._fire_streaks[key] = 0

    # ------------------------------------------------------------------
    # Condition evaluation helpers
    # ------------------------------------------------------------------

    def _exit_condition_met(
        self, protocol_id: str, phase: TemporalPhase
    ) -> bool:
        for signal in phase.exit_conditions:
            min_fires = max(int(signal.threshold), _MIN_OBSERVED_FIRES)
            if self._fire_streaks.get((protocol_id, signal.signal_id), 0) >= min_fires:
                return True
        return False

    def _entry_condition_met(
        self, protocol_id: str, phase: TemporalPhase
    ) -> bool:
        if not phase.entry_conditions:
            return False
        for signal in phase.entry_conditions:
            min_fires = max(int(signal.threshold), _MIN_OBSERVED_FIRES)
            if self._fire_streaks.get((protocol_id, signal.signal_id), 0) >= min_fires:
                return True
        return False

    # ------------------------------------------------------------------
    # Snapshot construction
    # ------------------------------------------------------------------

    def _build_snapshot(self) -> ProtocolPhaseSnapshot:
        phase_pairs = tuple(
            sorted(
                (pid, phase) for pid, phase in self._current_phase.items()
            )
        )
        turn_pairs = tuple(
            sorted(
                (pid, turns) for pid, turns in self._turns_in_phase.items()
            )
        )
        description = (
            f"protocol_phase: {len(phase_pairs)} protocol(s) tracked"
        )
        return ProtocolPhaseSnapshot(
            phase_by_protocol_id=phase_pairs,
            turns_in_current_phase=turn_pairs,
            description=description,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_any_progression_signal(protocol: BehaviorProtocol) -> bool:
    if protocol.temporal_arc.progression_signals:
        return True
    for phase in protocol.temporal_arc.phases:
        if phase.entry_conditions or phase.exit_conditions:
            return True
    return False


def _phase_by_id(
    phases: tuple[TemporalPhase, ...], phase_id: str
) -> TemporalPhase | None:
    for phase in phases:
        if phase.phase_id == phase_id:
            return phase
    return None


def _phase_after(
    phases: tuple[TemporalPhase, ...], current_phase_id: str
) -> TemporalPhase | None:
    found = False
    for phase in phases:
        if found:
            return phase
        if phase.phase_id == current_phase_id:
            found = True
    return None


def _read_pe_turn_index(
    upstream: Mapping[str, Snapshot[Any]],
) -> int | None:
    pe = upstream.get("prediction_error")
    if pe is None:
        return None
    if not isinstance(pe.value, PredictionErrorSnapshot):
        return None
    if pe.value.bootstrap:
        return None
    return int(pe.value.turn_index)


def _read_snapshot(
    upstream: Mapping[str, Snapshot[Any]],
    slot: str,
    expected_type: type,
) -> Any:
    snap = upstream.get(slot)
    if snap is None:
        return None
    if not isinstance(snap.value, expected_type):
        return None
    return snap.value


def _signal_is_firing(
    signal: ProgressionSignal,
    *,
    interlocutor: InterlocutorStateSnapshot | None,
    rupture: RuptureStateSnapshot | None,
    regime: RegimeSnapshot | None,
    boundary: BoundaryPolicySnapshot | None,
) -> bool:
    """Mirror of ``activation._signal_is_firing`` for ProgressionSignal.

    Reuses the same kernel-readable detector vocabulary so phase
    progression is driven by the same typed evidence stream that
    drives context_match scoring.
    """

    source = signal.measurable_via
    if source is BehaviorProtocolSignalSource.INTERLOCUTOR_ZONE_TRANSITION:
        if interlocutor is None:
            return False
        st = interlocutor.state
        return bool(
            st.acknowledge_pressure_zone
            or st.repair_zone
            or st.direct_task_zone
            or st.emotional_render_zone
            or st.pace_pressure_zone
            or st.cold_rapport_zone
            or st.low_directness_zone
        )
    if source is BehaviorProtocolSignalSource.RUPTURE_KIND_FIRED:
        return bool(rupture is not None and rupture.rupture_kind is not None)
    if source is BehaviorProtocolSignalSource.BOUNDARY_VIOLATION_FIRED:
        return bool(boundary is not None and boundary.trigger_reasons)
    if source is BehaviorProtocolSignalSource.USER_DROPOUT_OBSERVED:
        return bool(
            rupture is not None and rupture.rupture_kind is RuptureKind.ABANDONED
        )
    if source is BehaviorProtocolSignalSource.REGIME_TRANSITION_RECENT:
        return bool(
            regime is not None and int(regime.turns_in_current_regime) <= 1
        )
    # RETRIEVAL_HITS_PRESENT / DRIVE_HOMEOSTASIS_* / others: deferred
    # (deliberately mirror activation.py — phase progression should
    # not light up signals that activation deems unsupported).
    return False


__all__ = ["ProtocolPhaseModule"]
