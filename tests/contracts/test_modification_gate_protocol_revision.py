"""Packet 3.4: ModificationGate evaluate_protocol_revision tests."""

from __future__ import annotations

import pytest

from volvence_zero.behavior_protocol import (
    ProposalEvidence,
    ProtocolRevisionChangeKind,
    ProtocolRevisionProposal,
    ProtocolRevisionTargetField,
    ReviewLevel,
)
from volvence_zero.protocol_runtime import (
    ApprovalOutcome,
    RevisionQueue,
    evaluate_protocol_revision,
)
from volvence_zero.protocol_runtime.revision_queue import (
    L3_AUTO_APPROVE_MIN_OBSERVATION_TURNS,
)


def _proposal(
    *,
    review_level: ReviewLevel,
    observation_window_turns: int = 10,
    pe_signature: str = "test_signature",
    target_field: ProtocolRevisionTargetField = ProtocolRevisionTargetField.STRATEGY_PRIOR,
    proposal_id: str = "prop:test",
) -> ProtocolRevisionProposal:
    return ProtocolRevisionProposal(
        proposal_id=proposal_id,
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=target_field,
        target_entry_id="rapport-empathy",
        change_kind=ProtocolRevisionChangeKind.WEIGHT_DECAY,
        evidence=ProposalEvidence(
            observation_window_turns=observation_window_turns,
            pe_signature=pe_signature,
            summary="test",
        ),
        required_review_level=review_level,
    )


# ---------------------------------------------------------------------------
# evaluate_protocol_revision
# ---------------------------------------------------------------------------


def test_l1_auto_approves() -> None:
    decision = evaluate_protocol_revision(
        _proposal(review_level=ReviewLevel.L1)
    )
    assert decision.outcome is ApprovalOutcome.AUTO_APPROVED


def test_l2_auto_approves() -> None:
    decision = evaluate_protocol_revision(
        _proposal(review_level=ReviewLevel.L2)
    )
    assert decision.outcome is ApprovalOutcome.AUTO_APPROVED


def test_l3_auto_approves_when_evidence_meets_threshold() -> None:
    proposal = _proposal(
        review_level=ReviewLevel.L3,
        observation_window_turns=L3_AUTO_APPROVE_MIN_OBSERVATION_TURNS,
    )
    decision = evaluate_protocol_revision(proposal)
    assert decision.outcome is ApprovalOutcome.AUTO_APPROVED


def test_l3_queues_when_evidence_window_too_short() -> None:
    proposal = _proposal(
        review_level=ReviewLevel.L3,
        observation_window_turns=L3_AUTO_APPROVE_MIN_OBSERVATION_TURNS - 1,
    )
    decision = evaluate_protocol_revision(proposal)
    assert decision.outcome is ApprovalOutcome.QUEUED_FOR_HUMAN


def test_l3_queues_when_pe_signature_empty() -> None:
    proposal = _proposal(
        review_level=ReviewLevel.L3,
        observation_window_turns=20,
        pe_signature=" ",
    )
    decision = evaluate_protocol_revision(proposal)
    assert decision.outcome is ApprovalOutcome.QUEUED_FOR_HUMAN


def test_l4_always_queues() -> None:
    """L4 fail-safe: NEVER auto-approves regardless of evidence."""
    proposal = _proposal(
        review_level=ReviewLevel.L4,
        observation_window_turns=1000,
        pe_signature="overwhelming evidence",
        target_field=ProtocolRevisionTargetField.BOUNDARY_CONTRACT,
    )
    decision = evaluate_protocol_revision(proposal)
    assert decision.outcome is ApprovalOutcome.QUEUED_FOR_HUMAN
    assert "fail-safe" in decision.rationale.lower()


# ---------------------------------------------------------------------------
# RevisionQueue
# ---------------------------------------------------------------------------


def test_queue_submit_only_accepts_queued_decisions() -> None:
    proposal = _proposal(review_level=ReviewLevel.L1)
    decision = evaluate_protocol_revision(proposal)
    assert decision.outcome is ApprovalOutcome.AUTO_APPROVED

    queue = RevisionQueue()
    with pytest.raises(ValueError, match="QUEUED_FOR_HUMAN"):
        queue.submit(proposal, decision)


def test_queue_submit_and_list() -> None:
    proposal = _proposal(review_level=ReviewLevel.L4)
    decision = evaluate_protocol_revision(proposal)
    queue = RevisionQueue()
    queue.submit(proposal, decision)
    pending = queue.list_pending()
    assert len(pending) == 1
    assert pending[0][0].proposal_id == proposal.proposal_id


def test_queue_approve_pops_entry() -> None:
    proposal = _proposal(review_level=ReviewLevel.L4, proposal_id="prop:queued")
    decision = evaluate_protocol_revision(proposal)
    queue = RevisionQueue()
    queue.submit(proposal, decision)

    approved = queue.approve("prop:queued", reviewer_id="ops-admin")
    assert approved.proposal_id == "prop:queued"
    assert len(queue) == 0


def test_queue_reject_pops_entry_and_returns_audit() -> None:
    proposal = _proposal(review_level=ReviewLevel.L4, proposal_id="prop:reject")
    decision = evaluate_protocol_revision(proposal)
    queue = RevisionQueue()
    queue.submit(proposal, decision)

    rejection = queue.reject(
        "prop:reject",
        reviewer_id="ops-admin",
        reason="evidence insufficient",
    )
    assert rejection.outcome is ApprovalOutcome.REJECTED
    assert "ops-admin" in rejection.rationale
    assert len(queue) == 0


def test_queue_approve_unknown_id_raises() -> None:
    queue = RevisionQueue()
    with pytest.raises(KeyError, match="never-existed"):
        queue.approve("never-existed", reviewer_id="x")


def test_queue_reject_empty_reason_raises() -> None:
    proposal = _proposal(review_level=ReviewLevel.L4, proposal_id="prop:bad-reject")
    decision = evaluate_protocol_revision(proposal)
    queue = RevisionQueue()
    queue.submit(proposal, decision)
    with pytest.raises(ValueError, match="reason"):
        queue.reject(
            "prop:bad-reject",
            reviewer_id="ops-admin",
            reason="",
        )
