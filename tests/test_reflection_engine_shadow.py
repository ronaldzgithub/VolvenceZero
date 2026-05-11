"""Packet 3.1: ProtocolReflectionEngine SHADOW publish + dependency tests."""

from __future__ import annotations

import asyncio
from typing import Any

from volvence_zero.behavior_protocol import (
    ActiveMixtureSnapshot,
    ProtocolReflectionSnapshot,
)
from volvence_zero.integration.final_wiring import (
    FinalRolloutConfig,
    build_final_runtime_modules,
)
from volvence_zero.prediction import (
    ActualOutcome,
    PredictedOutcome,
    PredictionActionContext,
    PredictionError,
    PredictionErrorSnapshot,
)
from volvence_zero.reflection import ProtocolReflectionEngine
from volvence_zero.runtime import Snapshot, WiringLevel
from volvence_zero.substrate.adapter import PlaceholderSubstrateAdapter


# ---------------------------------------------------------------------------
# Module shape
# ---------------------------------------------------------------------------


def test_engine_owns_protocol_reflection_slot() -> None:
    eng = ProtocolReflectionEngine()
    assert eng.slot_name == "protocol_reflection"
    assert eng.owner == "ProtocolReflectionEngine"
    assert eng.value_type is ProtocolReflectionSnapshot


def test_engine_default_dependencies() -> None:
    assert ProtocolReflectionEngine.dependencies == (
        "prediction_error",
        "active_mixture",
    )


def test_engine_default_wiring_is_shadow() -> None:
    eng = ProtocolReflectionEngine()
    assert eng.wiring_level is WiringLevel.SHADOW


def test_engine_rejects_invalid_history_window() -> None:
    import pytest

    with pytest.raises(ValueError, match="history_window"):
        ProtocolReflectionEngine(history_window=0)


def test_engine_rejects_invalid_scan_period() -> None:
    import pytest

    with pytest.raises(ValueError, match="scan_period"):
        ProtocolReflectionEngine(scan_period=0)


# ---------------------------------------------------------------------------
# Per-turn behaviour: ingestion + scan periodicity
# ---------------------------------------------------------------------------


def _make_pe_snapshot(
    *,
    signed_reward: float = 0.0,
    turn_index: int = 1,
    bootstrap: bool = False,
) -> Snapshot[PredictionErrorSnapshot]:
    action_context = PredictionActionContext(
        segment_id=f"seg-{turn_index}",
        abstract_action_id="act",
        regime_id="r",
    )
    actual = ActualOutcome(
        observed_turn_index=turn_index,
        task_progress=0.5,
        relationship_delta=0.5,
        regime_stability=0.5,
        action_payoff=0.5,
        description="",
        action_context=action_context,
    )
    pred = PredictedOutcome(
        source_turn_index=turn_index,
        target_turn_index=turn_index + 1,
        predicted_task_progress=0.5,
        predicted_relationship_delta=0.5,
        predicted_regime_stability=0.5,
        predicted_action_payoff=0.5,
        confidence=0.5,
        description="",
        action_context=action_context,
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
    pe_snap = PredictionErrorSnapshot(
        evaluated_prediction=None if bootstrap else pred,
        actual_outcome=actual,
        next_prediction=pred,
        error=pe,
        turn_index=turn_index,
        bootstrap=bootstrap,
        description="",
        action_context=action_context,
    )
    return Snapshot(
        slot_name="prediction_error",
        owner="PredictionErrorModule",
        version=1,
        timestamp_ms=0,
        value=pe_snap,
    )


def _make_active_mixture_snapshot() -> Snapshot[ActiveMixtureSnapshot]:
    value = ActiveMixtureSnapshot(
        active_protocols=(),
        boundary_union_ids=(),
        revision_fingerprint="",
        description="",
    )
    return Snapshot(
        slot_name="active_mixture",
        owner="ProtocolRegistryModule",
        version=1,
        timestamp_ms=0,
        value=value,
    )


def _run_turn(eng, upstream):
    return asyncio.run(eng.process(upstream))


def test_engine_publishes_empty_snapshot_on_first_turn() -> None:
    eng = ProtocolReflectionEngine(scan_period=10)
    snap = _run_turn(eng, {})
    assert snap.value.protocol_revision_proposals == ()
    assert snap.value.observation_window_turns == 0
    assert snap.value.turns_since_last_scan == 1


def test_engine_ingests_pe_history_across_turns() -> None:
    eng = ProtocolReflectionEngine(scan_period=100)
    for t in range(1, 6):
        _run_turn(
            eng,
            {
                "prediction_error": _make_pe_snapshot(
                    signed_reward=0.1, turn_index=t
                ),
                "active_mixture": _make_active_mixture_snapshot(),
            },
        )
    assert len(eng.pe_history) == 5
    assert len(eng.active_mixture_history) == 5


def test_engine_skips_bootstrap_pe() -> None:
    eng = ProtocolReflectionEngine()
    _run_turn(
        eng,
        {
            "prediction_error": _make_pe_snapshot(
                signed_reward=0.1, turn_index=1, bootstrap=True
            ),
        },
    )
    assert len(eng.pe_history) == 0


def test_engine_dedupes_repeated_pe_turn_index() -> None:
    eng = ProtocolReflectionEngine()
    snap = _make_pe_snapshot(signed_reward=0.1, turn_index=5)
    _run_turn(eng, {"prediction_error": snap})
    _run_turn(eng, {"prediction_error": snap})
    assert len(eng.pe_history) == 1


def test_engine_runs_rules_on_scan_period() -> None:
    """After scan_period turns, _run_rules executes (returns empty
    until packet 3.2 fills the ruleset, but the counter resets)."""
    eng = ProtocolReflectionEngine(scan_period=3)
    for t in range(1, 4):
        snap = _run_turn(
            eng,
            {
                "prediction_error": _make_pe_snapshot(
                    signed_reward=0.1, turn_index=t
                ),
            },
        )
    # On the 3rd turn the scan should have fired and counter reset.
    assert snap.value.turns_since_last_scan == 0


# ---------------------------------------------------------------------------
# final_wiring integration
# ---------------------------------------------------------------------------


def test_engine_registered_in_final_wiring() -> None:
    config = FinalRolloutConfig()
    adapter = PlaceholderSubstrateAdapter(model_id="protocol-refl-test")
    modules = build_final_runtime_modules(
        config=config,
        substrate_adapter=adapter,
    )
    publishers = [m for m in modules if m.slot_name == "protocol_reflection"]
    assert len(publishers) == 1
    assert isinstance(publishers[0], ProtocolReflectionEngine)
    assert publishers[0].wiring_level is WiringLevel.SHADOW
