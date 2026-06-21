from __future__ import annotations

import pytest

from lifeform_protocol_runtime import (
    MentorIntakeKind,
    MentorIntakeRequest,
    build_protocol_revision_proposal,
    classify_mentor_intake,
    extract_reviewed_knowledge_from_guidance,
    extract_signature_case_from_guidance,
    resolve_experience_outcome_kind,
)
from volvence_zero.behavior_protocol import (
    ProtocolRevisionChangeKind,
    ProtocolRevisionTargetField,
)
from volvence_zero.dialogue_trace import DialogueExternalOutcomeKind


class _ClassifierClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict:
        self.calls.append((system_prompt, user_prompt))
        return self.payload


def test_classify_mentor_intake_returns_typed_decision():
    client = _ClassifierClient(
        {
            "intake_kind": "protocol",
            "routed_owner": "protocol_registry",
            "confidence": 0.91,
            "reason": "Changes the next-turn action posture.",
            "actionable_summary": "Clarify boundary before giving options.",
        }
    )

    decision = classify_mentor_intake(
        MentorIntakeRequest(
            guidance="When planning, confirm boundaries before advice.",
            mentor_id="mentor:alice",
        ),
        llm_client=client,
    )

    assert decision.intake_kind is MentorIntakeKind.PROTOCOL
    assert decision.routed_owner == "protocol_registry"
    assert decision.confidence == 0.91
    assert client.calls


def test_classify_mentor_intake_rejects_unknown_kind():
    client = _ClassifierClient(
        {
            "intake_kind": "prompt_hint",
            "confidence": 0.5,
            "reason": "invalid",
            "actionable_summary": "invalid",
        }
    )

    with pytest.raises(ValueError, match="unknown intake_kind"):
        classify_mentor_intake(
            MentorIntakeRequest(guidance="Do something different."),
            llm_client=client,
        )


def test_non_protocol_kinds_are_explicitly_unsupported_for_live_apply():
    client = _ClassifierClient(
        {
            "intake_kind": "experience",
            "confidence": 0.8,
            "reason": "This is a retrospective outcome note.",
            "actionable_summary": "Record the failed outcome.",
        }
    )

    decision = classify_mentor_intake(
        MentorIntakeRequest(guidance="The last dense reply caused dropout."),
        llm_client=client,
    )

    assert decision.intake_kind is MentorIntakeKind.EXPERIENCE
    assert decision.routed_owner == "experience_consolidation"
    # experience apply requires an explicit typed outcome_kind.
    assert "outcome_kind" in decision.unsupported_reason


def test_supported_kinds_carry_no_unsupported_reason():
    for kind, owner in (
        ("knowledge", "domain_knowledge"),
        ("case", "case_memory"),
        ("protocol_revision", "protocol_registry"),
    ):
        client = _ClassifierClient(
            {
                "intake_kind": kind,
                "routed_owner": owner,
                "confidence": 0.8,
                "reason": "r",
                "actionable_summary": "s",
            }
        )
        decision = classify_mentor_intake(
            MentorIntakeRequest(guidance="some guidance"),
            llm_client=client,
        )
        assert decision.unsupported_reason == ""


def test_extract_reviewed_knowledge_returns_draft():
    client = _ClassifierClient(
        {
            "summary": "A fragile child needs emotional safety before logic.",
            "detail": "Lead with co-regulation; defer reasoning.",
            "domain": "child-emotional-support",
            "confidence": 0.82,
            "relevance_hint": "when the child is distressed",
        }
    )
    draft = extract_reviewed_knowledge_from_guidance(
        "For a fragile child, give emotional support first.",
        llm_client=client,
        knowledge_id="k1",
    )
    assert draft.knowledge_id == "k1"
    assert draft.summary.startswith("A fragile child")
    assert 0.0 <= draft.confidence <= 1.0
    assert draft.detail


def test_extract_reviewed_knowledge_requires_summary():
    client = _ClassifierClient({"detail": "no summary"})
    with pytest.raises(ValueError, match="summary"):
        extract_reviewed_knowledge_from_guidance(
            "guidance", llm_client=client, knowledge_id="k1"
        )


def test_extract_signature_case_returns_case():
    client = _ClassifierClient(
        {
            "domain": "child-emotional-support",
            "problem_pattern": "child shuts down when corrected",
            "user_state_pattern": "high emotional weight, low openness",
            "intervention_ordering": ["emotional support", "then logic"],
            "outcome_label": "repaired",
            "risk_markers": [],
            "confidence": 0.7,
            "description": "A worked example of repair-first handling.",
        }
    )
    case = extract_signature_case_from_guidance(
        "Last time, leading with logic failed; emotional support first worked.",
        llm_client=client,
        case_id="c1",
    )
    assert case.case_id == "c1"
    assert case.intervention_ordering == ("emotional support", "then logic")


def test_extract_signature_case_requires_intervention_ordering():
    client = _ClassifierClient(
        {
            "domain": "x",
            "problem_pattern": "p",
            "intervention_ordering": [],
            "confidence": 0.5,
        }
    )
    with pytest.raises(ValueError, match="intervention_ordering"):
        extract_signature_case_from_guidance(
            "g", llm_client=client, case_id="c1"
        )


def test_build_protocol_revision_proposal_supported_combo():
    proposal = build_protocol_revision_proposal(
        proposal_id="p1",
        target_protocol_id="mentor:proto",
        target_field="strategy_prior",
        target_entry_id="rule-1",
        change_kind="weight_decay",
        summary="this strategy is over-firing",
    )
    assert proposal.target_field is ProtocolRevisionTargetField.STRATEGY_PRIOR
    assert proposal.change_kind is ProtocolRevisionChangeKind.WEIGHT_DECAY
    assert proposal.target_protocol_id == "mentor:proto"


def test_build_protocol_revision_proposal_rejects_unsupported_combo():
    with pytest.raises(ValueError, match="unsupported revision"):
        build_protocol_revision_proposal(
            proposal_id="p1",
            target_protocol_id="mentor:proto",
            target_field="knowledge_seed",
            target_entry_id="seed-1",
            change_kind="weight_decay",
            summary="bad combo",
        )


def test_build_protocol_revision_proposal_rejects_unknown_field():
    with pytest.raises(ValueError, match="unknown target_field"):
        build_protocol_revision_proposal(
            proposal_id="p1",
            target_protocol_id="mentor:proto",
            target_field="not-a-field",
            target_entry_id="x",
            change_kind="archive",
            summary="bad field",
        )


def test_resolve_experience_outcome_kind_valid():
    assert (
        resolve_experience_outcome_kind("helped")
        is DialogueExternalOutcomeKind.HELPED
    )


def test_resolve_experience_outcome_kind_requires_explicit():
    with pytest.raises(ValueError, match="explicit outcome_kind"):
        resolve_experience_outcome_kind(None)


def test_resolve_experience_outcome_kind_rejects_unknown():
    with pytest.raises(ValueError, match="unknown outcome_kind"):
        resolve_experience_outcome_kind("delighted")
