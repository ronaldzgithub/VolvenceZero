from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from volvence_zero.credit import (
    RelationshipActionCommonBaselineCredit,
    derive_preference_action_common_baseline_credit_records,
)
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
)
from volvence_zero.social import (
    settle_preference_action_forecast,
    social_prediction_error_from_preference_action_forecast_settlement,
)
from volvence_zero.social_cognition import (
    PreferenceActionForecast,
    RelationshipConditionReadout,
    SocialActionCandidatePrediction,
    SocialActionOutcomeProbability,
)


_OUTCOMES = (
    DialogueExternalOutcomeKind.HELPED.value,
    DialogueExternalOutcomeKind.FELT_HEARD.value,
    DialogueExternalOutcomeKind.MISSED.value,
    DialogueExternalOutcomeKind.OVER_DIRECTIVE.value,
)


@dataclass(frozen=True)
class _MutableExtendedCandidate(SocialActionCandidatePrediction):
    mutable_notes: list[str]


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


def _forecast(suffix: str) -> PreferenceActionForecast:
    return PreferenceActionForecast(
        forecast_id=f"common-baseline-forecast-{suffix}",
        decision_id=f"common-baseline-decision-{suffix}",
        interlocutor_id="primary",
        candidate_predictions=(
            _candidate("stay_present", (1.0, 0.0, 0.0, 0.0)),
            _candidate("respect_space", (0.2, 0.3, 0.3, 0.2)),
            _candidate("neutral_noop", (0.25, 0.25, 0.25, 0.25)),
        ),
        recommended_action_id="stay_present",
        confidence=0.8,
        source_record_ids=("preference-record-1",),
        issued_turn=4,
        evidence=("typed-observation:common-baseline",),
        session_scope=f"common-baseline-session-{suffix}",
    )


def _external(
    forecast: PreferenceActionForecast,
    *,
    action_id: str,
    kind: DialogueExternalOutcomeKind = DialogueExternalOutcomeKind.HELPED,
    confidence: float = 1.0,
    source: DialogueExternalOutcomeEvidenceSource = (
        DialogueExternalOutcomeEvidenceSource.ENVIRONMENT
    ),
) -> DialogueExternalOutcomeEvidence:
    return DialogueExternalOutcomeEvidence(
        evidence_id=f"environment-outcome:{forecast.decision_id}:{action_id}",
        turn_index=5,
        kind=kind,
        source=source,
        confidence=confidence,
        evidence_ref=f"environment-ref:{forecast.decision_id}:{action_id}",
        description="Typed reactive-environment outcome.",
        session_scope=forecast.session_scope,
        action_turn_index=forecast.issued_turn,
        forecast_id=forecast.forecast_id,
        decision_id=forecast.decision_id,
        action_id=action_id,
    )


def _derive(
    forecast: PreferenceActionForecast,
    evidence: DialogueExternalOutcomeEvidence,
) -> RelationshipActionCommonBaselineCredit:
    settlement = settle_preference_action_forecast(
        forecast=forecast,
        evidence=evidence,
    )
    error = social_prediction_error_from_preference_action_forecast_settlement(
        settlement
    )
    records = derive_preference_action_common_baseline_credit_records(
        forecasts=(forecast,),
        external_evidence=(evidence,),
        settlements=(settlement,),
        social_errors=(error,),
        settled_at_turn=settlement.observed_turn,
        timestamp_ms=5000,
    )
    assert len(records) == 1
    return records[0]


def test_common_noop_baseline_recovers_advantage_when_action_pe_is_zero() -> None:
    forecast = _forecast("candidate")
    evidence = _external(forecast, action_id="stay_present")

    credit = _derive(forecast, evidence)

    assert credit.parent_action_credit.credit_value == 0.0
    assert credit.common_baseline_expected_utility == 0.0
    assert credit.delivered_expected_utility == 1.0
    assert credit.common_baseline_adjustment == 0.5
    assert credit.credit_value == 0.5
    assert credit.credit_value == pytest.approx(
        credit.parent_action_credit.credit_value
        + credit.common_baseline_adjustment
    )
    assert credit.to_payload()["record_id"] == credit.record_id
    assert credit.to_payload()["forecast_sha256"] == credit.forecast_sha256


def test_noop_delivery_has_zero_adjustment_and_preserves_action_pe() -> None:
    forecast = _forecast("noop")
    evidence = _external(forecast, action_id="neutral_noop")

    credit = _derive(forecast, evidence)

    assert credit.delivered_expected_utility == credit.common_baseline_expected_utility
    assert credit.common_baseline_adjustment == 0.0
    assert credit.credit_value == credit.parent_action_credit.credit_value == 0.5


def test_contract_rejects_algebra_and_parent_lineage_tampering() -> None:
    forecast = _forecast("tamper")
    credit = _derive(forecast, _external(forecast, action_id="stay_present"))

    with pytest.raises(ValueError, match="credit_value_hex mismatch"):
        replace(credit, credit_value_hex=(0.25).hex())

    with pytest.raises(ValueError, match="social PE replay mismatch"):
        replace(
            credit,
            social_prediction_error=replace(
                credit.social_prediction_error,
                magnitude=0.0,
            ),
        )

    with pytest.raises(ValueError, match="parent action-PE replay mismatch"):
        replace(
            credit,
            parent_action_credit=replace(
                credit.parent_action_credit,
                abstract_action_id="neutral_noop",
            ),
        )


@pytest.mark.parametrize(
    ("confidence", "source", "match"),
    (
        (
            0.8,
            DialogueExternalOutcomeEvidenceSource.ENVIRONMENT,
            "confidence 1.0",
        ),
        (
            1.0,
            DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
            "ENVIRONMENT",
        ),
    ),
)
def test_common_baseline_credit_rejects_postaction_weighting_and_non_environment_sources(
    confidence: float,
    source: DialogueExternalOutcomeEvidenceSource,
    match: str,
) -> None:
    forecast = _forecast(f"source-{source.value}-{confidence}")
    evidence = _external(
        forecast,
        action_id="stay_present",
        confidence=confidence,
        source=source,
    )
    with pytest.raises(ValueError, match=match):
        _derive(forecast, evidence)


def test_common_baseline_derivation_is_current_turn_only_and_strictly_typed() -> None:
    forecast = _forecast("historical")
    evidence = _external(forecast, action_id="stay_present")
    settlement = settle_preference_action_forecast(
        forecast=forecast,
        evidence=evidence,
    )
    error = social_prediction_error_from_preference_action_forecast_settlement(
        settlement
    )
    assert (
        derive_preference_action_common_baseline_credit_records(
            forecasts=(forecast,),
            external_evidence=(evidence,),
            settlements=(settlement,),
            social_errors=(error,),
            settled_at_turn=settlement.observed_turn + 1,
            timestamp_ms=5000,
        )
        == ()
    )

    with pytest.raises(TypeError, match="exact tuple"):
        derive_preference_action_common_baseline_credit_records(
            forecasts=[forecast],  # type: ignore[arg-type]
            external_evidence=(evidence,),
            settlements=(settlement,),
            social_errors=(error,),
            settled_at_turn=settlement.observed_turn,
            timestamp_ms=5000,
        )

    forged_numeric = replace(settlement, observed_utility=1)
    with pytest.raises(TypeError, match="exact float"):
        derive_preference_action_common_baseline_credit_records(
            forecasts=(forecast,),
            external_evidence=(evidence,),
            settlements=(forged_numeric,),
            social_errors=(
                social_prediction_error_from_preference_action_forecast_settlement(
                    forged_numeric
                ),
            ),
            settled_at_turn=forged_numeric.observed_turn,
            timestamp_ms=5000,
        )


def test_common_baseline_credit_rejects_mutable_condition_readout_scores() -> None:
    forecast = _forecast("mutable-readout")
    mutable_scores = [
        ("agency_displacement", 0.75),
        ("repair_readiness", 0.25),
    ]
    readout = RelationshipConditionReadout(
        condition_label="agency_displacement",
        confidence=0.8,
        normalized_margin=0.25,
        candidate_scores=mutable_scores,  # type: ignore[arg-type]
        reader_artifact_id="a" * 64,
        source_observation_sha256="b" * 64,
    )
    forecast_with_mutable_readout = replace(
        forecast,
        condition_readout=readout,
    )

    with pytest.raises(
        TypeError,
        match=r"condition_readout\.candidate_scores must be an exact tuple",
    ):
        _derive(
            forecast_with_mutable_readout,
            _external(
                forecast_with_mutable_readout,
                action_id="stay_present",
            ),
        )


def test_common_baseline_credit_rejects_mutable_parent_subclasses() -> None:
    forecast = _forecast("mutable-subclass")
    base_candidate = forecast.candidate_predictions[0]
    extended_candidate = _MutableExtendedCandidate(
        action_id=base_candidate.action_id,
        outcomes=base_candidate.outcomes,
        mutable_notes=[],
    )
    forecast_with_subclass = replace(
        forecast,
        candidate_predictions=(
            extended_candidate,
            *forecast.candidate_predictions[1:],
        ),
    )

    with pytest.raises(
        TypeError,
        match=r"candidate_predictions contains an invalid item type",
    ):
        _derive(
            forecast_with_subclass,
            _external(forecast_with_subclass, action_id="stay_present"),
        )
