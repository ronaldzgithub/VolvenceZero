"""Packet 5.0: ProtocolPhaseModule + protocol_phase slot tests.

Asserts the contract for the new owner:

* SHADOW default + dependency declaration.
* Empty registry → empty snapshot.
* Protocol with empty progression_signals → phase pinned at first
  declared phase forever (cheng_laoshi backwards compat).
* Synthetic ProgressionSignal with threshold=N → phase advances
  after N consecutive firing turns.
* Bootstrap PE / duplicate turn_index do not advance phase.
* Cross-protocol independence: one protocol advancing doesn't
  reset another's streaks.
* Snapshot is consumed by ProtocolRegistryModule and surfaces
  in ``ActiveProtocolEntry.current_phase_id``.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace as _replace
from typing import Any

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from volvence_zero.behavior_protocol import (
    BehaviorProtocolSignalSource,
    ProgressionSignal,
    ProtocolPhaseSnapshot,
    TemporalArc,
    TemporalPhase,
)
from volvence_zero.interlocutor.contracts import (
    InterlocutorState,
    InterlocutorStateSnapshot,
    with_zones,
)
from volvence_zero.prediction import (
    ActualOutcome,
    PredictedOutcome,
    PredictionActionContext,
    PredictionError,
    PredictionErrorSnapshot,
)
from volvence_zero.protocol_runtime import (
    ProtocolPhaseModule,
    ProtocolRegistry,
    ProtocolRegistryModule,
)
from volvence_zero.runtime import Snapshot, WiringLevel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pe_snapshot(
    *, turn: int, signed_reward: float = 0.0
) -> Snapshot[PredictionErrorSnapshot]:
    ctx = PredictionActionContext()
    pred = PredictedOutcome(
        source_turn_index=turn,
        target_turn_index=turn + 1,
        predicted_task_progress=0.5,
        predicted_relationship_delta=0.5,
        predicted_regime_stability=0.5,
        predicted_action_payoff=0.5,
        confidence=0.5,
        description="",
        action_context=ctx,
    )
    actual = ActualOutcome(
        observed_turn_index=turn,
        task_progress=0.5,
        relationship_delta=0.5,
        regime_stability=0.5,
        action_payoff=0.5,
        description="",
        action_context=ctx,
    )
    pe = PredictionError(
        task_error=signed_reward,
        relationship_error=signed_reward,
        regime_error=signed_reward,
        action_error=signed_reward,
        magnitude=abs(signed_reward),
        signed_reward=signed_reward,
        description="",
    )
    return Snapshot(
        slot_name="prediction_error",
        owner="PredictionErrorModule",
        version=1,
        timestamp_ms=turn,
        value=PredictionErrorSnapshot(
            evaluated_prediction=pred,
            actual_outcome=actual,
            next_prediction=pred,
            error=pe,
            turn_index=turn,
            bootstrap=False,
            description="",
            action_context=ctx,
        ),
    )


def _interlocutor_snapshot(
    *, fire_acknowledge_pressure: bool
) -> Snapshot[InterlocutorStateSnapshot]:
    if fire_acknowledge_pressure:
        state = InterlocutorState(
            emotional_weight=0.80,
            resistance_level=0.55,
            trust_signal=-0.20,
            readout_confidence=0.85,
            rationale="active",
        )
    else:
        state = InterlocutorState(
            engagement_intensity=0.20,
            self_disclosure_level=0.30,
            task_focus_level=0.40,
            emotional_weight=0.30,
            cognitive_engagement=0.40,
            resistance_level=0.20,
            openness_to_guidance=0.50,
            directness=0.55,
            trust_signal=0.30,
            stability=0.55,
            rapport_warmth=0.55,
            pace_pressure=0.40,
            readout_confidence=0.85,
            rationale="neutral",
        )
    state = with_zones(state)
    return Snapshot(
        slot_name="interlocutor_state",
        owner="InterlocutorReadoutModule",
        version=1,
        timestamp_ms=0,
        value=InterlocutorStateSnapshot(state=state, description=""),
    )


def _build_arc_with_phases() -> TemporalArc:
    """Two-phase arc: 'icebreaker' → 'value_anchor' on 2 consecutive
    interlocutor zone firings."""
    return TemporalArc(
        phases=(
            TemporalPhase(
                phase_id="icebreaker",
                description="opening phase",
                exit_conditions=(
                    ProgressionSignal(
                        signal_id="exit_icebreaker_when_pressure",
                        measurable_via=BehaviorProtocolSignalSource.INTERLOCUTOR_ZONE_TRANSITION,
                        threshold=2,  # 2 consecutive firings
                        description="exit icebreaker after sustained pressure",
                    ),
                ),
            ),
            TemporalPhase(
                phase_id="value_anchor",
                description="anchor phase",
            ),
        ),
    )


def _make_test_protocol(*, with_phases: bool):
    bp = growth_advisor_profile_to_behavior_protocol(build_cheng_laoshi_profile())
    if with_phases:
        bp = _replace(bp, temporal_arc=_build_arc_with_phases())
    return bp


def _run_turn(module, upstream):
    return asyncio.run(module.process(upstream))


# ---------------------------------------------------------------------------
# Module shape
# ---------------------------------------------------------------------------


def test_phase_module_owns_protocol_phase_slot() -> None:
    registry = ProtocolRegistry()
    eng = ProtocolPhaseModule(registry=registry)
    assert eng.slot_name == "protocol_phase"
    assert eng.owner == "ProtocolPhaseModule"
    assert eng.value_type is ProtocolPhaseSnapshot


def test_phase_module_default_dependencies() -> None:
    assert ProtocolPhaseModule.dependencies == (
        "prediction_error",
        "interlocutor_state",
        "regime",
        "rupture_state",
        "boundary_policy",
    )


def test_phase_module_default_wiring_is_shadow() -> None:
    registry = ProtocolRegistry()
    eng = ProtocolPhaseModule(registry=registry)
    assert eng.wiring_level is WiringLevel.SHADOW


# ---------------------------------------------------------------------------
# Empty registry
# ---------------------------------------------------------------------------


def test_phase_module_empty_registry_publishes_empty_snapshot() -> None:
    registry = ProtocolRegistry()
    eng = ProtocolPhaseModule(registry=registry)
    snap = _run_turn(eng, {})
    assert snap.value.phase_by_protocol_id == ()
    assert snap.value.turns_in_current_phase == ()


# ---------------------------------------------------------------------------
# Empty progression_signals → phase pinned at phases[0]
# ---------------------------------------------------------------------------


def test_phase_module_empty_progression_pins_at_first_phase() -> None:
    """cheng_laoshi backwards compat: no progression_signals → phase fixed."""
    registry = ProtocolRegistry()
    bp = _make_test_protocol(with_phases=False)
    registry.load(bp)
    eng = ProtocolPhaseModule(registry=registry)

    # Run several turns; phase pointer should never advance.
    for turn in range(1, 6):
        _run_turn(
            eng,
            {
                "prediction_error": _pe_snapshot(turn=turn),
                "interlocutor_state": _interlocutor_snapshot(
                    fire_acknowledge_pressure=True
                ),
            },
        )

    # The phase pointer should be at the first declared phase
    # (whichever it is), not advanced by interlocutor pressure.
    snap = eng._build_snapshot()
    if snap.phase_by_protocol_id:
        first_phase_id = bp.temporal_arc.phases[0].phase_id
        # Single-protocol case: phase_by_protocol_id has 1 entry
        # and it equals first phase.
        for pid, phase in snap.phase_by_protocol_id:
            assert pid == bp.protocol_id
            assert phase == first_phase_id


# ---------------------------------------------------------------------------
# PE-driven phase advance
# ---------------------------------------------------------------------------


def test_phase_module_advances_after_threshold_consecutive_fires() -> None:
    """Synthetic 2-phase protocol; threshold=2 consecutive interlocutor
    zone firings → advance icebreaker → value_anchor."""
    registry = ProtocolRegistry()
    bp = _make_test_protocol(with_phases=True)
    registry.load(bp)
    eng = ProtocolPhaseModule(registry=registry)

    # Turn 1: signal not firing → no advance.
    snap1 = _run_turn(
        eng,
        {
            "prediction_error": _pe_snapshot(turn=1),
            "interlocutor_state": _interlocutor_snapshot(
                fire_acknowledge_pressure=False
            ),
        },
    )
    phase_t1 = dict(snap1.value.phase_by_protocol_id)[bp.protocol_id]
    assert phase_t1 == "icebreaker"

    # Turn 2: signal fires once → no advance yet (threshold=2).
    snap2 = _run_turn(
        eng,
        {
            "prediction_error": _pe_snapshot(turn=2),
            "interlocutor_state": _interlocutor_snapshot(
                fire_acknowledge_pressure=True
            ),
        },
    )
    phase_t2 = dict(snap2.value.phase_by_protocol_id)[bp.protocol_id]
    assert phase_t2 == "icebreaker"

    # Turn 3: signal fires consecutively → threshold reached → advance.
    snap3 = _run_turn(
        eng,
        {
            "prediction_error": _pe_snapshot(turn=3),
            "interlocutor_state": _interlocutor_snapshot(
                fire_acknowledge_pressure=True
            ),
        },
    )
    phase_t3 = dict(snap3.value.phase_by_protocol_id)[bp.protocol_id]
    assert phase_t3 == "value_anchor"


def test_phase_module_resets_streak_when_signal_stops() -> None:
    """One firing turn followed by a non-firing turn → streak resets."""
    registry = ProtocolRegistry()
    bp = _make_test_protocol(with_phases=True)
    registry.load(bp)
    eng = ProtocolPhaseModule(registry=registry)

    # Fire once.
    _run_turn(
        eng,
        {
            "prediction_error": _pe_snapshot(turn=1),
            "interlocutor_state": _interlocutor_snapshot(
                fire_acknowledge_pressure=True
            ),
        },
    )
    # Stop firing.
    _run_turn(
        eng,
        {
            "prediction_error": _pe_snapshot(turn=2),
            "interlocutor_state": _interlocutor_snapshot(
                fire_acknowledge_pressure=False
            ),
        },
    )
    # Fire once more — streak should be 1 (not 2).
    snap = _run_turn(
        eng,
        {
            "prediction_error": _pe_snapshot(turn=3),
            "interlocutor_state": _interlocutor_snapshot(
                fire_acknowledge_pressure=True
            ),
        },
    )
    phase = dict(snap.value.phase_by_protocol_id)[bp.protocol_id]
    assert phase == "icebreaker"


def test_phase_module_dedupes_replay_pe_turn() -> None:
    """Same turn_index PE arriving twice — second is a no-op."""
    registry = ProtocolRegistry()
    bp = _make_test_protocol(with_phases=True)
    registry.load(bp)
    eng = ProtocolPhaseModule(registry=registry)

    snap = _pe_snapshot(turn=5)
    _run_turn(
        eng,
        {
            "prediction_error": snap,
            "interlocutor_state": _interlocutor_snapshot(
                fire_acknowledge_pressure=True
            ),
        },
    )
    streak_after_first = eng._fire_streaks.get(
        (bp.protocol_id, "exit_icebreaker_when_pressure"), 0
    )

    _run_turn(
        eng,
        {
            "prediction_error": snap,  # Same turn_index
            "interlocutor_state": _interlocutor_snapshot(
                fire_acknowledge_pressure=True
            ),
        },
    )
    streak_after_dup = eng._fire_streaks.get(
        (bp.protocol_id, "exit_icebreaker_when_pressure"), 0
    )
    assert streak_after_first == streak_after_dup


def test_phase_module_skips_bootstrap_pe() -> None:
    registry = ProtocolRegistry()
    bp = _make_test_protocol(with_phases=True)
    registry.load(bp)
    eng = ProtocolPhaseModule(registry=registry)

    pe = _pe_snapshot(turn=1)
    bootstrap_pe = Snapshot(
        slot_name=pe.slot_name,
        owner=pe.owner,
        version=pe.version,
        timestamp_ms=pe.timestamp_ms,
        value=_replace(pe.value, bootstrap=True, evaluated_prediction=None),
    )
    snap = _run_turn(
        eng,
        {
            "prediction_error": bootstrap_pe,
            "interlocutor_state": _interlocutor_snapshot(
                fire_acknowledge_pressure=True
            ),
        },
    )
    # Bootstrap PE → process treated as "no PE turn"; streak dict
    # is still updated based on signal firing this turn (we don't
    # gate streak update on PE), but turn_index dedup uses PE so
    # next-turn equality still works. Phase pointer should remain
    # at first phase.
    if snap.value.phase_by_protocol_id:
        phase = dict(snap.value.phase_by_protocol_id)[bp.protocol_id]
        assert phase == "icebreaker"


# ---------------------------------------------------------------------------
# Cross-protocol independence
# ---------------------------------------------------------------------------


def test_phase_module_tracks_protocols_independently() -> None:
    """Loading two protocols → each has its own phase pointer."""
    registry = ProtocolRegistry()
    bp_a = _make_test_protocol(with_phases=True)
    bp_b = _replace(
        _make_test_protocol(with_phases=True),
        protocol_id="growth_advisor:cheng-laoshi-clone",
    )
    registry.load(bp_a)
    registry.load(bp_b)
    eng = ProtocolPhaseModule(registry=registry)

    # Run 3 firing turns; both should advance.
    for turn in range(1, 4):
        _run_turn(
            eng,
            {
                "prediction_error": _pe_snapshot(turn=turn),
                "interlocutor_state": _interlocutor_snapshot(
                    fire_acknowledge_pressure=True
                ),
            },
        )
    snap = eng._build_snapshot()
    by_id = dict(snap.phase_by_protocol_id)
    assert by_id[bp_a.protocol_id] == "value_anchor"
    assert by_id[bp_b.protocol_id] == "value_anchor"


# ---------------------------------------------------------------------------
# Integration with ProtocolRegistryModule
# ---------------------------------------------------------------------------


def test_registry_uses_phase_snapshot_for_current_phase_id() -> None:
    """ProtocolRegistryModule.process consumes protocol_phase upstream
    and fills ActiveProtocolEntry.current_phase_id from it."""
    registry = ProtocolRegistry()
    bp = _make_test_protocol(with_phases=True)
    registry.load(bp)

    # Synthesize a phase snapshot that says we're at value_anchor.
    phase_snap = Snapshot(
        slot_name="protocol_phase",
        owner="ProtocolPhaseModule",
        version=1,
        timestamp_ms=0,
        value=ProtocolPhaseSnapshot(
            phase_by_protocol_id=((bp.protocol_id, "value_anchor"),),
            turns_in_current_phase=((bp.protocol_id, 5),),
            description="test",
        ),
    )

    module = ProtocolRegistryModule(
        wiring_level=WiringLevel.SHADOW, registry=registry
    )
    snapshot = asyncio.run(module.process({"protocol_phase": phase_snap}))
    entry = snapshot.value.active_protocols[0]
    assert entry.current_phase_id == "value_anchor"


def test_registry_falls_back_to_first_phase_when_no_phase_snapshot() -> None:
    """Backwards compat: when protocol_phase upstream is missing,
    ActiveProtocolEntry.current_phase_id falls back to first declared
    phase (the pre-packet-5.0 default)."""
    registry = ProtocolRegistry()
    bp = _make_test_protocol(with_phases=True)
    registry.load(bp)
    module = ProtocolRegistryModule(
        wiring_level=WiringLevel.SHADOW, registry=registry
    )
    snapshot = asyncio.run(module.process({}))
    entry = snapshot.value.active_protocols[0]
    assert entry.current_phase_id == "icebreaker"


# ---------------------------------------------------------------------------
# final_wiring registration
# ---------------------------------------------------------------------------


def test_phase_module_registered_in_final_wiring() -> None:
    from volvence_zero.integration.final_wiring import (
        FinalRolloutConfig,
        build_final_runtime_modules,
    )
    from volvence_zero.substrate.adapter import PlaceholderSubstrateAdapter

    config = FinalRolloutConfig()
    modules = build_final_runtime_modules(
        config=config,
        substrate_adapter=PlaceholderSubstrateAdapter(model_id="phase-test"),
    )
    publishers = [m for m in modules if m.slot_name == "protocol_phase"]
    assert len(publishers) == 1
    assert isinstance(publishers[0], ProtocolPhaseModule)
    assert publishers[0].wiring_level is WiringLevel.SHADOW
