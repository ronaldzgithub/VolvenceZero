from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from lifeform_domain_emogpt.lab import (
    CandidateOutcomePrediction,
    OutcomeProbability,
    P4CanaryArmPreActionRecord,
    P4LongitudinalCanaryArm,
    PreActionRelationshipDecision,
    RelationshipAction,
    RelationshipModelLineage,
    authorize_relationship_p4_lab_action_advisory,
    load_relationship_p4_longitudinal_canary_contract,
    load_relationship_p4_longitudinal_canary_evaluator_bundle,
    load_relationship_p4_longitudinal_canary_view,
    prepare_relationship_p4_longitudinal_canary,
    render_relationship_p4_canary_markdown,
    relationship_p4_lab_active_authorization,
    relationship_p4_longitudinal_canary_protocol_path,
    run_relationship_p4_longitudinal_canary_development,
    run_relationship_p4_preference_mutation_drill,
    write_relationship_p4_canary_artifact,
)
from volvence_zero.dialogue_trace import DialogueExternalOutcomeKind
from lifeform_domain_emogpt.relationship_action_gate import (
    RelationshipActionGate,
    RelationshipActionGateMode,
    temporal_action_advisory_from_gate_decision,
)
from volvence_zero.social_cognition import (
    PreferenceActionForecast,
    SocialActionCandidatePrediction,
    SocialActionOutcomeProbability,
)


_OUTCOMES = ("helped", "felt_heard", "missed", "over_directive")
_FORBIDDEN_PUBLIC_KEYS = {
    "scene_id",
    "preferred_action",
    "expected_action",
    "policy_id",
    "condition_id",
    "dynamic_id",
    "generator_truth",
    "environment_seed",
}


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


def _forecast() -> PreferenceActionForecast:
    return PreferenceActionForecast(
        forecast_id="p4-canary-forecast-test",
        decision_id="p4-canary-decision-test",
        interlocutor_id="primary",
        candidate_predictions=(
            _candidate("stay_present_without_probe", (0.1, 0.7, 0.1, 0.1)),
            _candidate("respect_space_with_return_option", (0.1, 0.1, 0.4, 0.4)),
            _candidate("neutral_noop", (0.25, 0.25, 0.25, 0.25)),
        ),
        recommended_action_id="stay_present_without_probe",
        confidence=0.8,
        source_record_ids=("p4-canary-owner-record",),
        issued_turn=4,
        evidence=("runtime:p4-canary-test",),
        session_scope="p4-canary-test-subject",
    )


def test_protocol_freezes_familiar_longitudinal_story_and_strong_baselines() -> None:
    contract = load_relationship_p4_longitudinal_canary_contract()
    view = load_relationship_p4_longitudinal_canary_view()
    evaluator = load_relationship_p4_longitudinal_canary_evaluator_bundle()
    preparation = prepare_relationship_p4_longitudinal_canary()

    assert "有时需要你别走" in contract.story_title
    assert contract.formal_minimum_independent_subjects == 20
    assert contract.formal_minimum_context_tokens == 32_768
    assert contract.required_arms == tuple(arm.value for arm in P4LongitudinalCanaryArm)
    assert len(view.subjects) == 2
    assert all(len(subject.onboarding_sessions) == 4 for subject in view.subjects)
    assert all(len(subject.decision_sessions) == 8 for subject in view.subjects)
    assert all(
        tuple(item.action_turn_index for item in subject.decision_sessions) == (4, 6, 8, 10, 12, 14, 16, 18)
        for subject in view.subjects
    )
    public_json = json.dumps(
        [subject.to_sut_payload() for subject in view.subjects],
        ensure_ascii=False,
        sort_keys=True,
    )
    assert not any(f'"{key}"' in public_json for key in _FORBIDDEN_PUBLIC_KEYS)
    assert len(evaluator.sessions) == 16
    assert preparation.model_output_count == 0
    assert preparation.formal_evidence_authorized is False
    assert preparation.correction_redaction_owner_status == ("implemented_p4_2_engineering_drill_available")
    assert preparation.arm_statuses[:2] == (
        (
            P4LongitudinalCanaryArm.QWEN_STEELMAN_FULL_HISTORY.value,
            "blocked_by_p1k_p1m_zero_output_rule",
        ),
        (
            P4LongitudinalCanaryArm.QWEN_STEELMAN_SELECTIVE_RAG.value,
            "blocked_by_p1k_p1m_zero_output_rule",
        ),
    )


def test_lab_active_authorization_is_hashed_scoped_and_rejects_oracle() -> None:
    contract = load_relationship_p4_longitudinal_canary_contract()
    authorization = relationship_p4_lab_active_authorization(contract)
    gate = RelationshipActionGate()
    learned = gate.decide(_forecast(), mode=RelationshipActionGateMode.LEARNED)

    product_advisory = temporal_action_advisory_from_gate_decision(learned)
    lab_advisory = authorize_relationship_p4_lab_action_advisory(
        learned,
        authorization=authorization,
    )

    assert product_advisory.active_authorized is False
    assert lab_advisory.active_authorized is True
    assert any(authorization.authorization_id in item for item in lab_advisory.evidence_refs)
    assert authorization.environment_consumer_only is True
    assert authorization.expression_authorized is False
    assert authorization.production_authorized is False
    assert authorization.evaluation_feedback_to_learning is False

    oracle = RelationshipActionGate().decide(
        _forecast(),
        mode=RelationshipActionGateMode.ORACLE,
        oracle_action_id="stay_present_without_probe",
        evaluator_only=True,
    )
    with pytest.raises(ValueError, match="oracle"):
        authorize_relationship_p4_lab_action_advisory(
            oracle,
            authorization=authorization,
        )
    with pytest.raises(ValueError, match="outside Lab authorization"):
        authorize_relationship_p4_lab_action_advisory(
            replace(learned, artifact_id="different-gate"),
            authorization=authorization,
        )


def test_arm_preaction_record_is_typed_hashed_and_outcome_free() -> None:
    view = load_relationship_p4_longitudinal_canary_view()
    session = view.subjects[0].decision_sessions[-1]
    uniform = tuple(
        OutcomeProbability(outcome_kind=kind, probability=0.25)
        for kind in (
            DialogueExternalOutcomeKind.HELPED,
            DialogueExternalOutcomeKind.FELT_HEARD,
            DialogueExternalOutcomeKind.MISSED,
            DialogueExternalOutcomeKind.OVER_DIRECTIVE,
        )
    )
    decision = PreActionRelationshipDecision(
        decision_id=session.decision_id,
        pre_action_timestamp="2026-08-22T00:00:00+00:00",
        candidate_predictions=tuple(
            CandidateOutcomePrediction(action_id=action, outcomes=uniform)
            for action in (
                RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
                RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
                RelationshipAction.NEUTRAL_NOOP,
            )
        ),
        chosen_action_id=RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
        source_snapshot_hashes=(hashlib.sha256(b"owner-snapshot").hexdigest(),),
        lineage=RelationshipModelLineage(
            model_id="frozen-qwen-placeholder",
            weights_sha256=hashlib.sha256(b"weights").hexdigest(),
            prompt_sha256=hashlib.sha256(b"prompt").hexdigest(),
            generation_config_sha256=hashlib.sha256(b"generation").hexdigest(),
            seed=17,
        ),
    )
    record = P4CanaryArmPreActionRecord(
        protocol_sha256=view.contract.protocol_sha256,
        public_plan_sha256=view.public_plan_sha256,
        arm=P4LongitudinalCanaryArm.QWEN_STEELMAN_FULL_HISTORY,
        subject_scope=view.subjects[0].subject_scope,
        session_id=session.session_id,
        phase_id=session.phase_id,
        public_context_tokens=32_768,
        latency_ms=1200.0,
        response_sha256=hashlib.sha256(b"response").hexdigest(),
        decision=decision,
    )

    payload = record.to_payload()
    assert len(record.record_sha256) == 64
    assert payload["decision"]["chosen_action_id"] == "stay_present_without_probe"
    assert "observed_outcome" not in json.dumps(payload, sort_keys=True)


async def test_development_canary_closes_typed_causal_loop_without_model_claim() -> None:
    report = await run_relationship_p4_longitudinal_canary_development()

    assert report.verdict == "engineering_mechanism_ready_formal_effect_not_run"
    assert report.model_output_count == 0
    assert report.expression_output_count == 0
    assert report.formal_evidence_authorized is False
    assert len(report.runs) == 4
    assert all(len(run.traces) == 8 for run in report.runs)
    assert all(run.process_restart_count == 11 for run in report.runs)
    assert all(trace.temporal_status == "applied" for run in report.runs for trace in run.traces)
    assert all(len({trace.owner_persistence_sha256 for trace in run.traces}) == 8 for run in report.runs)

    closed_loop = report.arm_summaries[0]
    noop = report.arm_summaries[1]
    assert closed_loop.arm is P4LongitudinalCanaryArm.VOLVENCE_CLOSED_LOOP
    assert noop.arm is P4LongitudinalCanaryArm.VOLVENCE_TYPED_NOOP_CONTROL
    assert closed_loop.typed_outcome_count == noop.typed_outcome_count == 16
    assert closed_loop.gate_update_count == 16
    assert noop.gate_update_count == 0
    assert closed_loop.positive_outcome_count == 10
    assert noop.positive_outcome_count == 7
    assert closed_loop.preferred_action_match_count == 3
    assert noop.preferred_action_match_count == 0
    assert all(
        trace.credit_applied_to_gate
        for run in report.runs
        if run.arm is P4LongitudinalCanaryArm.VOLVENCE_CLOSED_LOOP
        for trace in run.traces
    )
    assert all(
        not trace.credit_applied_to_gate
        for run in report.runs
        if run.arm is P4LongitudinalCanaryArm.VOLVENCE_TYPED_NOOP_CONTROL
        for trace in run.traces
    )
    serialized = json.dumps(report.to_payload(), ensure_ascii=False, sort_keys=True)
    assert "current_observation" not in serialized
    assert "rendered_user_reaction" not in serialized
    assert "scene_id" not in serialized
    markdown = render_relationship_p4_canary_markdown(report)
    assert "10/16" in markdown
    assert "3/16" in markdown
    assert "Qwen output=0" in markdown
    assert "不能据此声称 Volvence advantage" in markdown


async def test_p4_2_mutation_drill_closes_correction_redaction_and_restart() -> None:
    report = await run_relationship_p4_preference_mutation_drill()

    assert report.passed is True
    assert report.corrected_evidence_sha256 == (report.reader_observed_evidence_sha256)
    assert report.correction_invalidated_forecast_count >= 1
    assert report.redaction_invalidated_forecast_count >= 1
    assert report.correction_persisted_after_restart is True
    assert report.redaction_content_absent_after_restart is True
    assert report.redaction_tombstone_enforced is True
    assert report.process_restart_count >= 7
    assert report.model_output_count == 0
    assert report.evaluator_truth_used is False
    assert report.formal_evidence_authorized is False
    serialized = json.dumps(report.to_payload(), ensure_ascii=False)
    assert "user-corrected relationship observation" not in serialized
    assert "user-corrected relationship reaction" not in serialized


@pytest.mark.parametrize(
    ("section", "field", "tampered_value", "error_match"),
    (
        (
            "lab_active_authorization",
            "production_authorized",
            True,
            "production_authorized",
        ),
        (
            "firewall",
            "new_qwen_output_allowed",
            True,
            "firewall",
        ),
    ),
)
def test_protocol_tamper_fails_loudly(
    tmp_path,
    section: str,
    field: str,
    tampered_value: bool,
    error_match: str,
) -> None:
    raw = json.loads(relationship_p4_longitudinal_canary_protocol_path().read_text(encoding="utf-8"))
    raw[section][field] = tampered_value
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match=error_match):
        load_relationship_p4_longitudinal_canary_contract(path)


def test_canary_artifacts_are_create_only_and_byte_hashed(tmp_path) -> None:
    path = tmp_path / "preparation.json"
    payload = prepare_relationship_p4_longitudinal_canary().to_payload()

    digest = write_relationship_p4_canary_artifact(path, payload)

    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        write_relationship_p4_canary_artifact(path, payload)
