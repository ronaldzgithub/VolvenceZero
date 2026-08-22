from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from lifeform_service.alpha import AlphaServiceConfig
from lifeform_service.app import create_app
from lifeform_service.relationship_intelligence import (
    RELATIONSHIP_ACTION_EXPOSURE_BASELINE_NOOP,
    RELATIONSHIP_ACTION_EXPOSURE_SHADOW_COUNTERFACTUAL,
    RELATIONSHIP_RUNTIME_OUTCOME_EXPOSURE_BLOCKED,
    RELATIONSHIP_RUNTIME_OUTCOME_SUBMITTED,
    RELATIONSHIP_RUNTIME_OUTCOME_TYPING_BLOCKED,
    RELATIONSHIP_TRAINING_CANDIDATE_SCHEMA_VERSION,
    RELATIONSHIP_TYPING_COLLECTION_ONLY,
    RELATIONSHIP_TYPING_PASSED,
    RelationshipAlphaArtifactStore,
    RelationshipIntelligenceController,
    RelationshipOutcomeTypingQualification,
    load_relationship_outcome_typing_qualification,
)
from lifeform_service.relationship_outcome_typing import (
    RELATIONSHIP_OUTCOME_TYPING_RESULT_SCHEMA_VERSION,
    RelationshipOutcomeEvidenceBasis,
    RelationshipOutcomeTypingResult,
)
from lifeform_service.verticals import _try_companion
from lifeform_domain_emogpt.relationship_action_gate import (
    RelationshipActionGateMode,
)
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
)
from volvence_zero.semantic_state import BoundaryConsentSnapshot
from volvence_zero.social_cognition import (
    PreferenceActionForecast,
    SocialActionCandidatePrediction,
    SocialActionOutcomeProbability,
)
from volvence_zero.temporal import (
    ControllerState,
    TemporalAbstractionSnapshot,
    TemporalActionAdvisoryStatus,
)


_OUTCOMES = ("helped", "felt_heard", "missed", "over_directive")


def _typing_qualification(
    *,
    majority_agreement: float = 0.86,
) -> RelationshipOutcomeTypingQualification:
    return RelationshipOutcomeTypingQualification(
        qualification_id="relationship-typing-qualification:alpha-1",
        typing_method="llm_structured_output",
        structured_runtime_id="relationship-outcome-typer:2026-08-22",
        structured_output_schema_id=(
            RELATIONSHIP_OUTCOME_TYPING_RESULT_SCHEMA_VERSION
        ),
        rater_count=3,
        independent_rater_artifact_ids=(
            "rater-batch:r1",
            "rater-batch:r2",
            "rater-batch:r3",
        ),
        sample_count=120,
        hidden_labels=True,
        majority_agreement=majority_agreement,
        typing_anchor_agreement=0.89,
        preregistered_anchor_threshold=0.85,
        validation_anchor_only=True,
        learning_use_authorized=False,
        keyword_or_regex_routing_used=False,
        unknown_outcome_supported=True,
        reviewed_by=("reviewer-1", "reviewer-2", "reviewer-3"),
        qualified_at_iso="2026-08-22T09:00:00+08:00",
    )


def _candidate(action_id: str, values: tuple[float, ...]):
    return SocialActionCandidatePrediction(
        action_id=action_id,
        outcomes=tuple(
            SocialActionOutcomeProbability(outcome_id, probability)
            for outcome_id, probability in zip(_OUTCOMES, values, strict=True)
        ),
    )


class _FakeAlphaSession:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.turn_summaries: tuple[object, ...] = ()
        self.staged = None
        self.submitted_outcomes: list[DialogueExternalOutcomeEvidence] = []

    async def preview_preference_action_forecast(self, *, request, runtime):
        assert runtime.runtime_id == "relationship-p2-bounded-forecast.v1"
        return PreferenceActionForecast(
            forecast_id=(
                f"preference_about_other:{request.decision_id}:"
                f"forecast:{request.turn_index}"
            ),
            decision_id=request.decision_id,
            interlocutor_id="primary",
            candidate_predictions=(
                _candidate("stay_present_without_probe", (0.1, 0.7, 0.1, 0.1)),
                _candidate("respect_space_with_return_option", (0.1, 0.1, 0.4, 0.4)),
                _candidate("neutral_noop", (0.25, 0.25, 0.25, 0.25)),
            ),
            recommended_action_id="stay_present_without_probe",
            confidence=0.8,
            source_record_ids=("preference-record-1",),
            issued_turn=request.turn_index,
            evidence=("typed-owner-evidence",),
            session_scope=request.session_scope,
        )

    def stage_self_temporal_action_advisory(self, advisory) -> None:
        self.staged = advisory

    async def run_turn(self, user_input: str):
        assert user_input
        assert self.staged is not None
        temporal = TemporalAbstractionSnapshot(
            controller_state=ControllerState(
                code=(0.0, 0.0, 0.0),
                code_dim=3,
                switch_gate=0.0,
                is_switching=False,
                steps_since_switch=0,
            ),
            active_abstract_action="native-placeholder-controller",
            controller_params_hash="native-controller-hash",
            description="native action with SHADOW relationship advisory",
            action_advisory=self.staged,
            action_advisory_status=TemporalActionAdvisoryStatus.SHADOW_RECORDED,
        )
        boundary = BoundaryConsentSnapshot(
            granted_consents=(),
            missing_consents=(),
            denied_boundaries=(),
            memory_consent="unknown",
            external_action_consent="unknown",
            compliance_score=1.0,
            control_signal=0.0,
            description="typed boundary fixture",
            autonomy_risk=0.1,
            overreach_risk=0.2,
            external_action_blocked=False,
        )
        self.turn_summaries = (*self.turn_summaries, object())
        return SimpleNamespace(
            response=SimpleNamespace(
                text="ordinary baseline response",
                rationale_tags=("relationship-action:shadow",),
            ),
            active_snapshots={
                "self_temporal": SimpleNamespace(value=temporal),
                "boundary_consent": SimpleNamespace(value=boundary),
            },
            shadow_snapshots={},
        )

    def relationship_action_credits(self, *, settled_at_turn, timestamp_ms):
        assert settled_at_turn == timestamp_ms
        return ()

    def submit_dialogue_outcome(self, **kwargs):
        evidence = DialogueExternalOutcomeEvidence(
            evidence_id=f"runtime-evidence:{kwargs['forecast_id']}",
            turn_index=kwargs["turn_index"],
            kind=kwargs["kind"],
            source=kwargs["source"],
            confidence=kwargs["confidence"],
            evidence_ref=kwargs["evidence_ref"],
            session_scope=self.session_id,
            action_turn_index=kwargs["action_turn_index"],
            forecast_id=kwargs["forecast_id"],
            decision_id=kwargs["decision_id"],
            action_id=kwargs["action_id"],
            typing_qualification_id=kwargs["typing_qualification_id"],
            typing_qualification_sha256=kwargs["typing_qualification_sha256"],
            typing_runtime_id=kwargs["typing_runtime_id"],
            typing_schema_version=kwargs["typing_schema_version"],
        )
        self.submitted_outcomes.append(evidence)
        return evidence


class _FakeOutcomeTyper:
    runtime_id = "relationship-outcome-typer:2026-08-22"
    schema_version = RELATIONSHIP_OUTCOME_TYPING_RESULT_SCHEMA_VERSION

    def __init__(self) -> None:
        self.calls = 0

    def classify(self, outcome_text: str) -> RelationshipOutcomeTypingResult:
        assert outcome_text
        self.calls += 1
        return RelationshipOutcomeTypingResult(
            outcome_kind="missed",
            confidence=0.9,
            evidence_basis=RelationshipOutcomeEvidenceBasis.EXPLICIT_REPORT,
            needs_human_review=False,
            runtime_id=self.runtime_id,
        )


async def test_collection_only_shell_audits_turn_and_separates_training_root(
    tmp_path,
) -> None:
    evidence_root = tmp_path / "operational-evidence"
    training_root = tmp_path / "offline-training-candidates"
    state_root = tmp_path / "private-owner-state"
    controller = RelationshipIntelligenceController(
        artifact_store=RelationshipAlphaArtifactStore(
            evidence_root=evidence_root,
            training_candidate_root=training_root,
        ),
        state_root=state_root,
    )
    assert controller.typing_gate_status == RELATIONSHIP_TYPING_COLLECTION_ONLY
    session = _FakeAlphaSession("relationship-alpha-session")
    raw_user_text = "This raw private sentence must not enter evidence artifacts."

    turn = await controller.run_turn(
        session=session,
        user_input=raw_user_text,
        subject_scope="alpha:user-1",
        session_scope=session.session_id,
    )

    assert turn.status == "shadow_action_audited"
    assert turn.audit is not None
    assert turn.audit.applied_to_expression is False
    assert (
        turn.audit.temporal_advisory_status
        == TemporalActionAdvisoryStatus.SHADOW_RECORDED.value
    )
    receipt = await controller.submit_outcome_text(
        session=session,
        subject_scope="alpha:user-1",
        session_scope=session.session_id,
        forecast_id=turn.audit.forecast_id,
        decision_id=turn.audit.decision_id,
        action_id=turn.audit.selected_action_id,
        outcome_text="I did feel heard, but this channel is not qualified yet.",
        training_use_authorized=True,
    )
    assert receipt.submitted_to_runtime is False
    assert receipt.outcome_kind == "unknown"
    assert receipt.needs_human_review is True
    assert (
        receipt.action_exposure_status
        == RELATIONSHIP_ACTION_EXPOSURE_BASELINE_NOOP
    )
    assert (
        receipt.runtime_submission_status
        == RELATIONSHIP_RUNTIME_OUTCOME_TYPING_BLOCKED
    )
    assert receipt.runtime_evidence_id is None
    assert session.submitted_outcomes == []
    assert receipt.training_candidate_ref is not None
    assert receipt.operational_evidence_ref.startswith(
        "relationship-alpha-evidence:"
    )
    assert receipt.training_candidate_ref.startswith(
        "relationship-training-candidate:"
    )
    assert tuple(evidence_root.rglob("*.json"))
    assert tuple(training_root.rglob("*.json"))
    all_artifact_text = "\n".join(
        path.read_text(encoding="utf-8")
        for root in (evidence_root, training_root)
        for path in root.rglob("*.json")
    )
    assert raw_user_text not in all_artifact_text
    assert RELATIONSHIP_TRAINING_CANDIDATE_SCHEMA_VERSION in all_artifact_text
    assert tuple(state_root.rglob("*.json"))


async def test_passed_typing_gate_is_required_before_runtime_submission(
    tmp_path,
) -> None:
    typer = _FakeOutcomeTyper()
    controller = RelationshipIntelligenceController(
        artifact_store=RelationshipAlphaArtifactStore(
            evidence_root=tmp_path / "evidence",
        ),
        typing_qualification=_typing_qualification(),
        outcome_typer=typer,
    )
    assert controller.typing_gate_status == RELATIONSHIP_TYPING_PASSED
    session = _FakeAlphaSession("relationship-alpha-qualified-session")
    turn = await controller.run_turn(
        session=session,
        user_input="A typed observation for the qualified intake path.",
        subject_scope="alpha:user-2",
        session_scope=session.session_id,
    )
    assert turn.audit is not None

    receipt = await controller.submit_outcome_text(
        session=session,
        subject_scope="alpha:user-2",
        session_scope=session.session_id,
        forecast_id=turn.audit.forecast_id,
        decision_id=turn.audit.decision_id,
        action_id=turn.audit.selected_action_id,
        outcome_text="That did not meet what I needed from the conversation.",
        training_use_authorized=False,
    )

    assert receipt.submitted_to_runtime is True
    assert receipt.runtime_evidence_id is not None
    assert receipt.outcome_kind == DialogueExternalOutcomeKind.MISSED.value
    assert receipt.typing_runtime_id == _FakeOutcomeTyper.runtime_id
    assert receipt.needs_human_review is False
    assert receipt.runtime_submission_status == RELATIONSHIP_RUNTIME_OUTCOME_SUBMITTED
    assert receipt.typing_qualification_id == (
        "relationship-typing-qualification:alpha-1"
    )
    assert receipt.typing_qualification_sha256 == (
        controller.typing_qualification.content_sha256
    )
    assert len(session.submitted_outcomes) == 1
    evidence = session.submitted_outcomes[0]
    assert (
        evidence.source
        is DialogueExternalOutcomeEvidenceSource.QUALIFIED_USER_REPORT
    )
    assert evidence.forecast_id == turn.audit.forecast_id
    assert evidence.decision_id == turn.audit.decision_id
    assert evidence.action_id == turn.audit.selected_action_id

    retried = await controller.submit_outcome_text(
        session=session,
        subject_scope="alpha:user-2",
        session_scope=session.session_id,
        forecast_id=turn.audit.forecast_id,
        decision_id=turn.audit.decision_id,
        action_id=turn.audit.selected_action_id,
        outcome_text="That did not meet what I needed from the conversation.",
        training_use_authorized=False,
    )
    assert retried == receipt
    assert len(session.submitted_outcomes) == 1
    assert typer.calls == 1


async def test_shadow_counterfactual_is_never_credited_as_an_executed_action(
    tmp_path,
) -> None:
    controller = RelationshipIntelligenceController(
        artifact_store=RelationshipAlphaArtifactStore(
            evidence_root=tmp_path / "evidence",
            training_candidate_root=tmp_path / "training",
        ),
        typing_qualification=_typing_qualification(),
        outcome_typer=_FakeOutcomeTyper(),
    )
    session = _FakeAlphaSession("relationship-alpha-shadow-counterfactual")
    turn = await controller.run_turn(
        session=session,
        user_input="A turn whose suggested action remains invisible.",
        subject_scope="alpha:user-shadow",
        session_scope=session.session_id,
        gate_mode=RelationshipActionGateMode.ALWAYS,
    )
    assert turn.audit is not None
    assert turn.audit.gate_action == "steer"
    assert turn.audit.applied_to_expression is False

    receipt = await controller.submit_outcome_text(
        session=session,
        subject_scope="alpha:user-shadow",
        session_scope=session.session_id,
        forecast_id=turn.audit.forecast_id,
        decision_id=turn.audit.decision_id,
        action_id=turn.audit.selected_action_id,
        outcome_text="The baseline reply did not meet what I needed.",
        training_use_authorized=True,
    )

    assert receipt.submitted_to_runtime is False
    assert session.submitted_outcomes == []
    assert (
        receipt.action_exposure_status
        == RELATIONSHIP_ACTION_EXPOSURE_SHADOW_COUNTERFACTUAL
    )
    assert (
        receipt.runtime_submission_status
        == RELATIONSHIP_RUNTIME_OUTCOME_EXPOSURE_BLOCKED
    )
    assert receipt.training_candidate_ref is None


def test_typing_qualification_is_verified_and_cannot_be_replaced_by_a_boolean(
    tmp_path,
) -> None:
    qualification = _typing_qualification()
    path = tmp_path / "typing-qualification.json"
    path.write_text(
        json.dumps(qualification.to_json(), sort_keys=True),
        encoding="utf-8",
    )

    loaded = load_relationship_outcome_typing_qualification(path)

    assert loaded == qualification
    assert loaded.passed is True
    failed = _typing_qualification(majority_agreement=0.79)
    controller = RelationshipIntelligenceController(
        artifact_store=RelationshipAlphaArtifactStore(
            evidence_root=tmp_path / "failed-evidence",
        ),
        typing_qualification=failed,
    )
    assert controller.typing_gate_status == RELATIONSHIP_TYPING_COLLECTION_ONLY

    tampered = qualification.to_json()
    tampered["majority_agreement"] = 0.10
    path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash|verdict"):
        load_relationship_outcome_typing_qualification(path)


def test_service_requires_the_qualified_structured_llm_runtime(tmp_path) -> None:
    qualification_path = tmp_path / "typing-qualification.json"
    qualification_path.write_text(
        json.dumps(_typing_qualification().to_json(), sort_keys=True),
        encoding="utf-8",
    )
    config = AlphaServiceConfig(
        enabled=True,
        memory_scope_root_dir=str(tmp_path / "owner-state"),
        evidence_root_dir=str(tmp_path / "evidence"),
        relationship_intelligence_enabled=True,
        relationship_outcome_typing_qualification_path=str(qualification_path),
    )
    vertical = _try_companion()
    assert vertical is not None

    with pytest.raises(ValueError, match="structured-JSON LLM client"):
        create_app(vertical=vertical, alpha_config=config)

    app = create_app(
        vertical=vertical,
        alpha_config=config,
        external_llm_client=_FakeJsonClient(),
    )
    controller = app["relationship_intelligence_controller"]
    assert isinstance(controller, RelationshipIntelligenceController)
    assert controller.typing_gate_status == RELATIONSHIP_TYPING_PASSED
    relationship_routes = {
        route.resource.canonical
        for route in app.router.routes()
        if "relationship" in route.resource.canonical
    }
    assert {
        "/v1/sessions/{session_id}/relationship-turns",
        "/v1/sessions/{session_id}/relationship-outcomes",
        "/v1/sessions/{session_id}/relationship-followups/execute-due",
        "/v1/users/me/relationship-memory",
        "/v1/users/me/relationship-memory/{item_id}/action",
    }.issubset(relationship_routes)


class _FakeJsonClient:
    def complete_json(self, *, system_prompt: str, user_prompt: str):
        del system_prompt, user_prompt
        return {
            "schema_version": RELATIONSHIP_OUTCOME_TYPING_RESULT_SCHEMA_VERSION,
            "outcome_kind": "unknown",
            "confidence": 0.2,
            "evidence_basis": "mixed_or_ambiguous",
            "needs_human_review": True,
        }
