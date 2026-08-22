from __future__ import annotations

from dataclasses import replace

import pytest

from lifeform_domain_emogpt.relationship_action_contracts import RelationshipAction
from lifeform_domain_emogpt.relationship_action_gate import (
    RELATIONSHIP_ACTION_CREDIT_LEVEL,
    RelationshipActionGate,
    RelationshipActionGateMode,
    RelationshipGateAction,
    temporal_action_advisory_from_gate_decision,
)
from volvence_zero.credit import CreditRecord
from volvence_zero.memory import Track
from volvence_zero.runtime import WiringLevel
from volvence_zero.social_cognition import (
    PreferenceActionForecast,
    SocialActionCandidatePrediction,
    SocialActionOutcomeProbability,
)
from volvence_zero.substrate import SubstrateSnapshot, SurfaceKind
from volvence_zero.temporal import PlaceholderTemporalPolicy, TrackTemporalModule
from volvence_zero.temporal_types import TemporalActionAdvisoryStatus


_OUTCOMES = ("helped", "felt_heard", "missed", "over_directive")


def _candidate(
    action_id: str,
    probabilities: tuple[float, float, float, float],
) -> SocialActionCandidatePrediction:
    return SocialActionCandidatePrediction(
        action_id=action_id,
        outcomes=tuple(
            SocialActionOutcomeProbability(outcome_id, probability)
            for outcome_id, probability in zip(
                _OUTCOMES,
                probabilities,
                strict=True,
            )
        ),
    )


def _forecast(suffix: str = "1") -> PreferenceActionForecast:
    return PreferenceActionForecast(
        forecast_id=f"relationship-forecast-{suffix}",
        decision_id=f"relationship-decision-{suffix}",
        interlocutor_id="primary",
        candidate_predictions=(
            _candidate("stay_present_without_probe", (0.15, 0.55, 0.2, 0.1)),
            _candidate("respect_space_with_return_option", (0.2, 0.2, 0.2, 0.4)),
            _candidate("neutral_noop", (0.25, 0.25, 0.25, 0.25)),
        ),
        recommended_action_id="stay_present_without_probe",
        confidence=0.8,
        source_record_ids=("preference-record-1", "preference-record-2"),
        issued_turn=4,
        evidence=("runtime:bounded-owner-reader",),
        session_scope="closed-alpha-user-1",
    )


def _credit(decision, *, value: float) -> CreditRecord:
    return CreditRecord(
        record_id=f"relationship-credit:{decision.forecast_id}",
        level=RELATIONSHIP_ACTION_CREDIT_LEVEL,
        track=Track.SELF,
        source_event=f"social_pe:{decision.forecast_id}",
        credit_value=value,
        context="typed PE-derived relationship action credit",
        timestamp_ms=5000,
        prediction_id=decision.forecast_id,
        environment_outcome_id=f"external-outcome:{decision.forecast_id}",
        abstract_action_id=decision.selected_action_id,
    )


def _substrate() -> SubstrateSnapshot:
    return SubstrateSnapshot(
        model_id="frozen-test-substrate",
        is_frozen=True,
        surface_kind=SurfaceKind.PLACEHOLDER,
        token_logits=(),
        feature_surface=(),
        residual_activations=(),
        residual_sequence=(),
        unavailable_fields=(),
        description="gate self-temporal contract fixture",
    )


def test_control_modes_are_explicit_and_oracle_is_evaluator_only() -> None:
    forecast = _forecast()
    gate = RelationshipActionGate(random_seed="frozen-control-seed")

    noop = gate.decide(forecast, mode=RelationshipActionGateMode.NOOP)
    always = gate.decide(forecast, mode=RelationshipActionGateMode.ALWAYS)
    random_a = gate.decide(forecast, mode=RelationshipActionGateMode.RANDOM)
    random_b = gate.decide(forecast, mode=RelationshipActionGateMode.RANDOM)
    oracle = gate.decide(
        forecast,
        mode=RelationshipActionGateMode.ORACLE,
        oracle_action_id=RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION.value,
        evaluator_only=True,
    )

    assert noop.gate_action is RelationshipGateAction.NOOP
    assert noop.selected_action_id == RelationshipAction.NEUTRAL_NOOP.value
    assert always.gate_action is RelationshipGateAction.STEER
    assert always.selected_action_id == forecast.recommended_action_id
    assert random_a == random_b
    assert oracle.evaluator_only is True
    assert oracle.selected_action_id == "respect_space_with_return_option"
    with pytest.raises(ValueError, match="evaluator_only"):
        gate.decide(
            forecast,
            mode=RelationshipActionGateMode.ORACLE,
            oracle_action_id="stay_present_without_probe",
        )


def test_only_matching_pe_credit_updates_the_learned_gate_and_restores() -> None:
    gate = RelationshipActionGate()
    decision = gate.decide(_forecast(), mode=RelationshipActionGateMode.LEARNED)
    assert decision.gate_action is RelationshipGateAction.NOOP
    before = gate.parameter_state

    update = gate.observe_credit(_credit(decision, value=-0.8))

    assert gate.parameter_state != before
    assert update.update_count == 1
    assert update.old_state_sha256 != update.new_state_sha256
    checkpoint = gate.export_checkpoint()
    restored = RelationshipActionGate(checkpoint=checkpoint)
    assert restored.parameter_state == gate.parameter_state
    assert restored.update_count == 1
    next_decision = restored.decide(
        _forecast("2"),
        mode=RelationshipActionGateMode.LEARNED,
    )
    assert next_decision.steer_probability > 0.5
    assert next_decision.gate_action is RelationshipGateAction.STEER

    wrong_level = replace(
        _credit(next_decision, value=0.4),
        level="evaluation_score",
    )
    with pytest.raises(ValueError, match="PE-derived"):
        restored.observe_credit(wrong_level)
    with pytest.raises(ValueError, match="action lineage"):
        restored.observe_credit(
            replace(
                _credit(next_decision, value=0.4),
                abstract_action_id="neutral_noop",
            )
        )


async def test_self_temporal_records_unpromoted_advisory_in_shadow_only() -> None:
    decision = RelationshipActionGate().decide(
        _forecast(),
        mode=RelationshipActionGateMode.ALWAYS,
    )
    advisory = temporal_action_advisory_from_gate_decision(decision)
    module = TrackTemporalModule(
        track=Track.SELF,
        policy=PlaceholderTemporalPolicy(),
        wiring_level=WiringLevel.ACTIVE,
        action_advisory=advisory,
        action_advisory_level=WiringLevel.SHADOW,
    )

    snapshot = await module.process_standalone(substrate_snapshot=_substrate())

    assert snapshot.value.active_abstract_action == "placeholder-controller"
    assert snapshot.value.action_advisory == advisory
    assert (
        snapshot.value.action_advisory_status
        is TemporalActionAdvisoryStatus.SHADOW_RECORDED
    )
    with pytest.raises(ValueError, match="ACTIVE authorization"):
        TrackTemporalModule(
            track=Track.SELF,
            policy=PlaceholderTemporalPolicy(),
            action_advisory=advisory,
            action_advisory_level=WiringLevel.ACTIVE,
        )
    with pytest.raises(ValueError, match="only target self_temporal"):
        TrackTemporalModule(
            track=Track.WORLD,
            policy=PlaceholderTemporalPolicy(),
            action_advisory=advisory,
            action_advisory_level=WiringLevel.SHADOW,
        )
