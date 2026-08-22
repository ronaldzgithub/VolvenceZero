from __future__ import annotations

from volvence_zero.agent.session import AgentSessionRunner
from volvence_zero.social import (
    PreferenceActionForecastProposal,
    PreferenceActionForecastRequest,
)
from volvence_zero.social_cognition import (
    OtherMindRecord,
    OtherMindRecordKind,
    OtherMindRecordStatus,
    PreferenceActionOutcomeEvidence,
    SocialActionCandidatePrediction,
    SocialActionOutcomeProbability,
)
from volvence_zero.temporal import (
    TemporalActionAdvisoryProposal,
    TemporalActionAdvisoryStatus,
)


_ACTIONS = (
    "stay_present_without_probe",
    "respect_space_with_return_option",
    "neutral_noop",
)
_OUTCOMES = ("helped", "felt_heard", "missed", "over_directive")


class _ForecastRuntime:
    runtime_id = "runtime-advisory-contract-test"

    def propose(self, *, request, records, action_outcomes):
        assert records
        assert action_outcomes
        return PreferenceActionForecastProposal(
            candidate_predictions=tuple(
                SocialActionCandidatePrediction(
                    action_id=action_id,
                    outcomes=tuple(
                        SocialActionOutcomeProbability(
                            outcome_id=outcome_id,
                            probability=probability,
                        )
                        for outcome_id, probability in zip(
                            _OUTCOMES,
                            probabilities,
                            strict=True,
                        )
                    ),
                )
                for action_id, probabilities in (
                    (_ACTIONS[0], (0.1, 0.7, 0.1, 0.1)),
                    (_ACTIONS[1], (0.1, 0.1, 0.4, 0.4)),
                    (_ACTIONS[2], (0.25, 0.25, 0.25, 0.25)),
                )
            ),
            recommended_action_id=_ACTIONS[0],
            confidence=0.8,
            source_record_ids=tuple(record.record_id for record in records),
            evidence=("typed-owner-runtime-test",),
        )


async def test_runner_previews_owner_forecast_and_records_same_turn_shadow_advisory() -> None:
    runner = AgentSessionRunner(
        session_id="relationship-runtime-session",
        rare_heavy_enabled=False,
    )
    record = OtherMindRecord(
        record_id="preference-runtime-record",
        interlocutor_id="primary",
        kind=OtherMindRecordKind.PREFERENCE,
        summary="Typed preference summary.",
        detail="Typed preference detail.",
        confidence=0.8,
        status=OtherMindRecordStatus.ACTIVE,
        source_turn=0,
        prediction_error_refs=(),
        evidence="typed owner evidence",
    )
    runner.social_record_store.set_tom_records(
        "preference_about_other",
        (record,),
    )
    runner.social_record_store.set_preference_action_outcomes(
        (
            PreferenceActionOutcomeEvidence(
                evidence_id=record.record_id,
                interlocutor_id="primary",
                observation_summary="Earlier parallel situation.",
                action_id=_ACTIONS[0],
                observed_outcome_id="felt_heard",
                reaction_summary="The user resumed when presence remained available.",
                source_turn=0,
                evidence_refs=("typed-history:0",),
            ),
        )
    )
    request = PreferenceActionForecastRequest(
        decision_id="runtime-relationship-decision-1",
        interlocutor_id="primary",
        current_observation="A different surface situation with the same structure.",
        observation_ref="typed-current:1",
        candidate_action_ids=_ACTIONS,
        outcome_ids=_OUTCOMES,
        turn_index=1,
        session_scope="relationship-runtime-session",
    )

    forecast = await runner.preview_preference_action_forecast(
        request=request,
        runtime=_ForecastRuntime(),
    )

    assert forecast is not None
    advisory = TemporalActionAdvisoryProposal(
        advisory_id="runtime-advisory-1",
        decision_id=forecast.decision_id,
        prediction_id=forecast.forecast_id,
        action_id=forecast.recommended_action_id,
        confidence=forecast.confidence,
        policy_artifact_id="relationship-action-gate-test",
        policy_artifact_version=1,
        evidence_refs=forecast.evidence,
        rationale_codes=("test:typed-owner-forecast",),
    )
    runner.stage_self_temporal_action_advisory(advisory)
    result = await runner.run_turn("Run the canonical turn.")

    self_temporal = result.active_snapshots["self_temporal"].value
    assert self_temporal.action_advisory == advisory
    assert (
        self_temporal.action_advisory_status
        is TemporalActionAdvisoryStatus.SHADOW_RECORDED
    )
    assert self_temporal.active_abstract_action != advisory.action_id
    assert runner.social_record_store.preference_action_forecasts == (forecast,)
    assert runner.self_temporal_policy is not None
