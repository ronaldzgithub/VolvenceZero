"""CP-12 owner prediction signal contract tests.

Proves: (a) the shared immutable signal validates fail-loud, (b) the five
first-wave semantic owners publish predictions and settle them on the next
turn from their own store state, and (c) only the PE owner turns settled
signals into mismatch records — consumers never read owner internals.
"""

from __future__ import annotations

import pytest

from volvence_zero.owner_prediction import (
    OwnerPredictionKind,
    OwnerPredictionSignal,
    settle_owner_prediction,
)


def _signal(**overrides) -> OwnerPredictionSignal:
    payload = dict(
        signal_id="relationship_state:opsig-1",
        prediction_id="relationship_state:oppred-1",
        source_owner="RelationshipStateModule",
        source_slot="relationship_state",
        track="self",
        kind=OwnerPredictionKind.RELATIONSHIP_TRUST_TRAJECTORY,
        predicted_vector=(0.6, 0.2),
        confidence=0.7,
        description="persistence-prior v1 forecast",
        source_turn_index=1,
        evidence=("trust_level=0.60 repair_pressure=0.20",),
    )
    payload.update(overrides)
    return OwnerPredictionSignal(**payload)


def test_signal_validates_track_vector_and_confidence() -> None:
    with pytest.raises(ValueError, match="track"):
        _signal(track="task")
    with pytest.raises(ValueError, match="predicted_vector"):
        _signal(predicted_vector=())
    with pytest.raises(ValueError, match="predicted_vector"):
        _signal(predicted_vector=(1.4,))
    with pytest.raises(ValueError, match="confidence"):
        _signal(confidence=1.2)


def test_settlement_is_one_shot_and_requires_matching_shape_and_evidence() -> None:
    signal = _signal()
    assert signal.settled is False
    settled = settle_owner_prediction(
        signal,
        settled_vector=(0.5, 0.3),
        outcome_evidence=("observed readout at turn 2",),
    )
    assert settled.settled is True
    with pytest.raises(ValueError, match="already settled"):
        settle_owner_prediction(
            settled, settled_vector=(0.5, 0.3), outcome_evidence=("again",)
        )
    with pytest.raises(ValueError, match="dimensionality"):
        settle_owner_prediction(
            signal, settled_vector=(0.5,), outcome_evidence=("bad shape",)
        )
    with pytest.raises(ValueError, match="outcome_evidence"):
        settle_owner_prediction(signal, settled_vector=(0.5, 0.3), outcome_evidence=())


FIRST_WAVE = (
    ("commitment", OwnerPredictionKind.COMMITMENT_FOLLOW_THROUGH, "self"),
    ("relationship_state", OwnerPredictionKind.RELATIONSHIP_TRUST_TRAJECTORY, "self"),
    ("goal_value", OwnerPredictionKind.GOAL_VALUE_ALIGNMENT, "world"),
    ("boundary_consent", OwnerPredictionKind.BOUNDARY_CONSENT_STABILITY, "self"),
    ("execution_result", OwnerPredictionKind.EXECUTION_RESULT_SUCCESS, "world"),
)

SECOND_WAVE = (
    ("plan_intent", OwnerPredictionKind.PLAN_INTENT_PROGRESS, "world"),
    ("open_loop", OwnerPredictionKind.OPEN_LOOP_CLOSURE, "world"),
    ("belief_assumption", OwnerPredictionKind.BELIEF_ASSUMPTION_STABILITY, "world"),
    ("user_model", OwnerPredictionKind.USER_MODEL_PACING, "self"),
)

ALL_WAVES = FIRST_WAVE + SECOND_WAVE


async def _owner_snapshot(module_type, store, turn_index):
    from volvence_zero.substrate import (
        SimulatedResidualSubstrateAdapter,
        SubstrateModule,
        build_training_trace,
    )
    from volvence_zero.memory import MemoryModule

    trace = build_training_trace(
        trace_id=f"op-sig-{turn_index}", source_text="steady waters in the harbor"
    )
    substrate = await SubstrateModule(
        adapter=SimulatedResidualSubstrateAdapter(trace=trace)
    ).process_standalone()
    memory = await MemoryModule().process_standalone(substrate_snapshot=substrate.value)
    module = module_type(store=store, turn_index=turn_index)
    return (await module.process({"substrate": substrate, "memory": memory})).value


@pytest.mark.parametrize(("slot", "kind", "track"), ALL_WAVES)
async def test_first_wave_owner_publishes_then_settles_prediction(
    slot: str, kind: OwnerPredictionKind, track: str
) -> None:
    from volvence_zero.semantic_state import SemanticStateStore
    from volvence_zero.semantic_state.owners import SEMANTIC_MODULE_TYPES

    module_type = next(m for m in SEMANTIC_MODULE_TYPES if m.slot_name == slot)
    store = SemanticStateStore()

    first = await _owner_snapshot(module_type, store, turn_index=1)
    first_signals = first.owner_prediction_signals
    assert len(first_signals) == 1
    assert first_signals[0].settled is False
    assert first_signals[0].kind is kind
    assert first_signals[0].track == track
    assert first_signals[0].source_slot == slot

    second = await _owner_snapshot(module_type, store, turn_index=2)
    second_signals = second.owner_prediction_signals
    assert len(second_signals) == 2
    settled, fresh = second_signals
    assert settled.settled is True
    assert settled.prediction_id == first_signals[0].prediction_id
    assert fresh.settled is False
    assert fresh.prediction_id != settled.prediction_id


@pytest.mark.parametrize(("slot", "kind", "track"), ALL_WAVES)
async def test_owner_forecast_is_v2_learned_and_trains_on_settlement(
    slot: str, kind: OwnerPredictionKind, track: str
) -> None:
    """W1.B: forecasts come from the store-held learner (v2-learned) and
    each settlement applies one training update."""

    from volvence_zero.semantic_state import SemanticStateStore
    from volvence_zero.semantic_state.owners import SEMANTIC_MODULE_TYPES

    module_type = next(m for m in SEMANTIC_MODULE_TYPES if m.slot_name == slot)
    store = SemanticStateStore()

    first = await _owner_snapshot(module_type, store, turn_index=1)
    fresh = first.owner_prediction_signals[-1]
    assert "v2-learned" in fresh.description
    updates, _ = store.owner_forecast_stats(slot)
    assert updates == 0  # nothing settled yet

    await _owner_snapshot(module_type, store, turn_index=2)
    updates, mae = store.owner_forecast_stats(slot)
    assert updates == 1
    assert 0.0 <= mae <= 1.0


def test_store_forecaster_cold_start_matches_persistence_prior() -> None:
    from volvence_zero.semantic_state import SemanticStateStore

    store = SemanticStateStore()
    observed = (0.62, 0.31)
    forecast = store.forecast_owner_vector("commitment", observed_vector=observed)
    assert forecast == observed


def test_store_forecaster_learns_a_drift_pattern() -> None:
    """Persistence prior always predicts 'no change'; the learned
    forecaster must beat it on a steadily drifting readout."""

    from volvence_zero.semantic_state import SemanticStateStore

    store = SemanticStateStore()
    slot = "relationship_state"
    series = [(0.1 + 0.02 * step, 0.5) for step in range(40)]
    learned_errors: list[float] = []
    prior_errors: list[float] = []
    for current, following in zip(series, series[1:], strict=False):
        forecast = store.forecast_owner_vector(slot, observed_vector=current)
        learned_errors.append(abs(forecast[0] - following[0]))
        prior_errors.append(abs(current[0] - following[0]))
        store.settle_owner_forecast(slot, observed_vector=following)
    late_learned = sum(learned_errors[-10:]) / 10
    late_prior = sum(prior_errors[-10:]) / 10
    assert late_learned < late_prior
    updates, mae = store.owner_forecast_stats(slot)
    assert updates == len(learned_errors)
    assert mae >= 0.0


def test_store_forecaster_rejects_dimension_drift() -> None:
    from volvence_zero.semantic_state import SemanticStateStore
    from volvence_zero.semantic_state.store import (
        OwnerForecastDimensionMismatchError,
    )

    store = SemanticStateStore()
    store.forecast_owner_vector("goal_value", observed_vector=(0.5, 0.5))
    with pytest.raises(OwnerForecastDimensionMismatchError):
        store.forecast_owner_vector("goal_value", observed_vector=(0.5,))


async def test_pe_owner_computes_settlement_from_settled_signals() -> None:
    """Two-turn session: PE snapshot carries typed settlements for the five
    first-wave owners without reading their internals."""

    from volvence_zero.agent.session import AgentSessionRunner

    runner = AgentSessionRunner(rare_heavy_enabled=False)
    await runner.run_turn("I promise to send you the tide tables tomorrow.")
    result = await runner.run_turn("Actually the plan changed, let's revisit.")

    pe_value = result.active_snapshots["prediction_error"].value
    settlements = pe_value.owner_prediction_settlements
    assert settlements, "second turn must settle first-turn owner predictions"
    slots = {s.source_slot for s in settlements}
    assert slots <= {slot for slot, _, _ in ALL_WAVES}
    # Second-wave owners (GAP-05) settle through the same single PE owner.
    assert slots & {slot for slot, _, _ in SECOND_WAVE}, (
        f"no second-wave owner settlement reached the PE owner; got {slots}"
    )
    for settlement in settlements:
        assert 0.0 <= settlement.mismatch_magnitude <= 1.0
        assert settlement.settled_turn_index == pe_value.turn_index
        assert settlement.description
