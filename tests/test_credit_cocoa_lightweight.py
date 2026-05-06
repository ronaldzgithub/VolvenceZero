"""Tests for the lightweight COCOA-style counterfactual contribution helper.

Phase 1.A of the Companion NL Uplift: a counterfactual contribution
estimate that reuses already-published statistics from regime / temporal
owners (selection weights + delayed payoff). The helper must be a no-op
when context is missing and must not change the legacy credit path.
"""

from __future__ import annotations

from volvence_zero.credit import (
    CreditLedger,
    CreditRecord,
    derive_counterfactual_contribution_records,
    record_nstep_outcomes_from_segment_closure,
)
from volvence_zero.prediction import (
    ActualOutcome,
    PredictedOutcome,
    PredictionActionContext,
    PredictionError,
    PredictionErrorSnapshot,
)
from volvence_zero.regime import (
    DelayedOutcomeAttribution,
    DelayedOutcomePayoff,
    RegimeIdentity,
    RegimeSelectionWeights,
    RegimeSnapshot,
)
from volvence_zero.temporal import (
    ControllerState,
    TemporalAbstractionSnapshot,
    TemporalSegmentClosure,
)


def _make_pe_snapshot(
    *,
    signed_reward: float,
    bootstrap: bool = False,
    regime_id: str = "comfort",
    abstract_action_id: str = "action_a",
    segment_id: str = "seg-1",
    turn_index: int = 1,
) -> PredictionErrorSnapshot:
    action_context = PredictionActionContext(
        segment_id=segment_id,
        abstract_action_id=abstract_action_id,
        regime_id=regime_id,
    )
    actual = ActualOutcome(
        observed_turn_index=turn_index,
        task_progress=signed_reward,
        relationship_delta=signed_reward,
        regime_stability=signed_reward,
        action_payoff=signed_reward,
        description="actual",
        action_context=action_context,
    )
    next_prediction = PredictedOutcome(
        source_turn_index=turn_index,
        target_turn_index=turn_index + 1,
        predicted_task_progress=0.0,
        predicted_relationship_delta=0.0,
        predicted_regime_stability=0.0,
        predicted_action_payoff=0.0,
        confidence=0.5,
        description="next",
        action_context=action_context,
    )
    error = PredictionError(
        task_error=signed_reward,
        relationship_error=signed_reward,
        regime_error=signed_reward,
        action_error=signed_reward,
        magnitude=abs(signed_reward),
        signed_reward=signed_reward,
        description="pe",
    )
    return PredictionErrorSnapshot(
        evaluated_prediction=None if bootstrap else next_prediction,
        actual_outcome=actual,
        next_prediction=next_prediction,
        error=error,
        turn_index=turn_index,
        bootstrap=bootstrap,
        description="pe-snapshot",
        action_context=action_context,
    )


def _make_regime_snapshot(
    *,
    weights: tuple[tuple[str, float], ...],
    payoffs: tuple[DelayedOutcomePayoff, ...],
    active_regime_id: str = "comfort",
) -> RegimeSnapshot:
    active = RegimeIdentity(
        regime_id=active_regime_id,
        name=active_regime_id,
        embedding=(0.1, 0.1),
        entry_conditions="",
        exit_conditions="",
        historical_effectiveness=0.5,
    )
    return RegimeSnapshot(
        active_regime=active,
        previous_regime=None,
        switch_reason="",
        candidate_regimes=weights,
        turns_in_current_regime=1,
        description="regime",
        delayed_attributions=(),
        delayed_payoffs=payoffs,
        selection_weights=RegimeSelectionWeights(weights=weights),
    )


def _make_temporal_snapshot(
    *,
    segment_id: str,
    abstract_action_id: str,
    closed: bool = True,
) -> TemporalAbstractionSnapshot:
    closures: tuple[TemporalSegmentClosure, ...] = ()
    if closed:
        closures = (
            TemporalSegmentClosure(
                segment_id=segment_id,
                open_turn_index=0,
                close_turn_index=1,
                abstract_action_id=abstract_action_id,
                z_t_digest=(0.1, 0.2),
                beta_open_digest=0.1,
                beta_close_digest=0.5,
                affordance_name=None,
                description="closure",
            ),
        )
    return TemporalAbstractionSnapshot(
        controller_state=ControllerState(
            code=(0.1, 0.2),
            code_dim=2,
            switch_gate=0.5,
            is_switching=False,
            steps_since_switch=1,
        ),
        active_abstract_action=abstract_action_id,
        controller_params_hash="hash",
        description="temporal",
        action_family_version=1,
        closed_segments=closures,
    )


def test_cocoa_records_zero_contribution_when_distribution_concentrated():
    """When one regime has effectively 100% weight, baseline equals its
    historical payoff and contribution collapses to (actual - that
    payoff)."""

    pe = _make_pe_snapshot(signed_reward=0.40)
    regime = _make_regime_snapshot(
        weights=(("comfort", 1.0),),
        payoffs=(
            DelayedOutcomePayoff(
                regime_id="comfort",
                abstract_action="action_a",
                action_family_version=1,
                sample_count=4,
                rolling_payoff=0.40,
                latest_outcome=0.40,
                last_source_wave_id="w-old",
            ),
        ),
    )
    temporal = _make_temporal_snapshot(
        segment_id="seg-1", abstract_action_id="action_a"
    )
    records = derive_counterfactual_contribution_records(
        regime_snapshot=regime,
        temporal_snapshot=temporal,
        prediction_error_snapshot=pe,
        timestamp_ms=10,
    )
    assert len(records) == 1
    assert records[0].level == "counterfactual_contribution"
    # actual ~ baseline → contribution ~ 0
    assert abs(records[0].credit_value) < 1e-6
    assert "baseline=0.400" in records[0].context
    assert "segment_closed=seg-1" in records[0].context


def test_cocoa_positive_when_actual_beats_uniform_baseline():
    """Uniform weights over regimes with low rolling payoffs and a
    high actual outcome must produce positive contribution."""

    pe = _make_pe_snapshot(signed_reward=0.80)
    regime = _make_regime_snapshot(
        weights=(("comfort", 1.0), ("repair", 1.0), ("planning", 1.0)),
        payoffs=(
            DelayedOutcomePayoff(
                regime_id="comfort",
                abstract_action="action_a",
                action_family_version=1,
                sample_count=4,
                rolling_payoff=0.10,
                latest_outcome=0.10,
                last_source_wave_id="w-old",
            ),
            DelayedOutcomePayoff(
                regime_id="repair",
                abstract_action=None,
                action_family_version=1,
                sample_count=4,
                rolling_payoff=0.05,
                latest_outcome=0.05,
                last_source_wave_id="w-old",
            ),
            DelayedOutcomePayoff(
                regime_id="planning",
                abstract_action=None,
                action_family_version=1,
                sample_count=4,
                rolling_payoff=-0.10,
                latest_outcome=-0.10,
                last_source_wave_id="w-old",
            ),
        ),
    )
    records = derive_counterfactual_contribution_records(
        regime_snapshot=regime,
        temporal_snapshot=None,
        prediction_error_snapshot=pe,
        timestamp_ms=10,
    )
    assert len(records) == 1
    record = records[0]
    # baseline = (0.10 + 0.05 - 0.10) / 3 ≈ 0.0167; actual = 0.80
    assert record.credit_value > 0.5
    assert record.source_event.startswith("cocoa:comfort")


def test_cocoa_negative_when_actual_below_baseline():
    """If actual is below the counterfactual baseline, contribution is
    negative even if signed_reward itself is positive."""

    pe = _make_pe_snapshot(signed_reward=0.10)
    regime = _make_regime_snapshot(
        weights=(("comfort", 1.0), ("repair", 1.0)),
        payoffs=(
            DelayedOutcomePayoff(
                regime_id="comfort",
                abstract_action="action_a",
                action_family_version=1,
                sample_count=4,
                rolling_payoff=0.50,
                latest_outcome=0.50,
                last_source_wave_id="w-old",
            ),
            DelayedOutcomePayoff(
                regime_id="repair",
                abstract_action=None,
                action_family_version=1,
                sample_count=4,
                rolling_payoff=0.50,
                latest_outcome=0.50,
                last_source_wave_id="w-old",
            ),
        ),
    )
    records = derive_counterfactual_contribution_records(
        regime_snapshot=regime,
        temporal_snapshot=None,
        prediction_error_snapshot=pe,
        timestamp_ms=10,
    )
    assert len(records) == 1
    assert records[0].credit_value < 0.0


def test_cocoa_returns_empty_when_bootstrap_or_missing_payoffs():
    bootstrap_pe = _make_pe_snapshot(signed_reward=0.5, bootstrap=True)
    regime = _make_regime_snapshot(
        weights=(("comfort", 1.0),),
        payoffs=(
            DelayedOutcomePayoff(
                regime_id="comfort",
                abstract_action="action_a",
                action_family_version=1,
                sample_count=2,
                rolling_payoff=0.2,
                latest_outcome=0.2,
                last_source_wave_id="w-old",
            ),
        ),
    )
    assert (
        derive_counterfactual_contribution_records(
            regime_snapshot=regime,
            temporal_snapshot=None,
            prediction_error_snapshot=bootstrap_pe,
            timestamp_ms=10,
        )
        == ()
    )

    pe = _make_pe_snapshot(signed_reward=0.5)
    no_payoff_regime = _make_regime_snapshot(
        weights=(("comfort", 1.0),),
        payoffs=(),
    )
    assert (
        derive_counterfactual_contribution_records(
            regime_snapshot=no_payoff_regime,
            temporal_snapshot=None,
            prediction_error_snapshot=pe,
            timestamp_ms=10,
        )
        == ()
    )

    assert (
        derive_counterfactual_contribution_records(
            regime_snapshot=None,
            temporal_snapshot=None,
            prediction_error_snapshot=pe,
            timestamp_ms=10,
        )
        == ()
    )


def test_record_nstep_outcomes_appends_when_segment_closes():
    pe = _make_pe_snapshot(signed_reward=0.7, segment_id="seg-7", abstract_action_id="famA")
    regime = _make_regime_snapshot(
        weights=(("comfort", 1.0),),
        payoffs=(
            DelayedOutcomePayoff(
                regime_id="comfort",
                abstract_action="famA",
                action_family_version=1,
                sample_count=1,
                rolling_payoff=0.0,
                latest_outcome=0.0,
                last_source_wave_id="w-old",
            ),
        ),
    )
    temporal = _make_temporal_snapshot(
        segment_id="seg-7", abstract_action_id="famA"
    )
    ledger = CreditLedger()
    assert ledger.snapshot().delayed_ledger_size == 0

    written = record_nstep_outcomes_from_segment_closure(
        ledger=ledger,
        prediction_error_snapshot=pe,
        temporal_snapshot=temporal,
        regime_snapshot=regime,
        timestamp_ms=42,
    )
    assert written == 1
    snapshot = ledger.snapshot()
    assert snapshot.delayed_ledger_size == 1
    # second close on the same action_id appends to the same entry, not a new one.
    record_nstep_outcomes_from_segment_closure(
        ledger=ledger,
        prediction_error_snapshot=pe,
        temporal_snapshot=temporal,
        regime_snapshot=regime,
        timestamp_ms=43,
    )
    assert ledger.snapshot().delayed_ledger_size == 1


def test_record_nstep_outcomes_noop_without_segment_close():
    pe = _make_pe_snapshot(signed_reward=0.7)
    no_close_temporal = _make_temporal_snapshot(
        segment_id="seg-1", abstract_action_id="action_a", closed=False
    )
    ledger = CreditLedger()
    written = record_nstep_outcomes_from_segment_closure(
        ledger=ledger,
        prediction_error_snapshot=pe,
        temporal_snapshot=no_close_temporal,
        regime_snapshot=None,
        timestamp_ms=10,
    )
    assert written == 0
    assert ledger.snapshot().delayed_ledger_size == 0


def test_cocoa_records_have_no_effect_on_legacy_consumers():
    """Legacy consumers that filter by record.level must not be
    affected by the new ``counterfactual_contribution`` level."""

    pe = _make_pe_snapshot(signed_reward=0.30)
    regime = _make_regime_snapshot(
        weights=(("comfort", 1.0), ("repair", 1.0)),
        payoffs=(
            DelayedOutcomePayoff(
                regime_id="comfort",
                abstract_action="action_a",
                action_family_version=1,
                sample_count=2,
                rolling_payoff=0.20,
                latest_outcome=0.20,
                last_source_wave_id="w-old",
            ),
            DelayedOutcomePayoff(
                regime_id="repair",
                abstract_action=None,
                action_family_version=1,
                sample_count=2,
                rolling_payoff=0.10,
                latest_outcome=0.10,
                last_source_wave_id="w-old",
            ),
        ),
    )
    records = derive_counterfactual_contribution_records(
        regime_snapshot=regime,
        temporal_snapshot=None,
        prediction_error_snapshot=pe,
        timestamp_ms=10,
    )
    assert all(isinstance(record, CreditRecord) for record in records)
    # Legacy abstract-action / session / turn filters all skip the new level.
    abstract_filter = tuple(r for r in records if r.level == "abstract_action")
    session_filter = tuple(r for r in records if r.level == "session")
    turn_filter = tuple(r for r in records if r.level == "turn")
    assert abstract_filter == ()
    assert session_filter == ()
    assert turn_filter == ()
