"""Packet 7.4: protocol_revision_review CLI tests."""

from __future__ import annotations

import io

from volvence_zero.behavior_protocol import (
    ProposalEvidence,
    ProtocolRevisionChangeKind,
    ProtocolRevisionProposal,
    ProtocolRevisionTargetField,
    ReviewLevel,
)
from volvence_zero.cli.protocol_revision_review import (
    list_pending,
    run_review_session,
)
from volvence_zero.protocol_runtime import (
    ApprovalDecision,
    ApprovalOutcome,
    RevisionQueue,
)


def _proposal(proposal_id: str, level: ReviewLevel = ReviewLevel.L4) -> ProtocolRevisionProposal:
    return ProtocolRevisionProposal(
        proposal_id=proposal_id,
        target_protocol_id="growth_advisor:test",
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id="rule:test",
        change_kind=ProtocolRevisionChangeKind.WEIGHT_DECAY,
        evidence=ProposalEvidence(
            observation_window_turns=10,
            pe_signature="cli-test",
            summary="cli test summary",
        ),
        required_review_level=level,
    )


def _populate_queue(queue: RevisionQueue, n: int = 2) -> None:
    for i in range(n):
        p = _proposal(f"prop:{i}")
        decision = ApprovalDecision(
            proposal_id=p.proposal_id,
            outcome=ApprovalOutcome.QUEUED_FOR_HUMAN,
            rationale="L4 always queued",
            decided_at_iso="2026-01-01T00:00:00Z",
        )
        queue.submit(p, decision)


def test_list_pending_empty_queue() -> None:
    queue = RevisionQueue()
    out = io.StringIO()
    import contextlib
    with contextlib.redirect_stdout(out):
        count = list_pending(queue)
    assert count == 0
    assert "(no pending proposals)" in out.getvalue()


def test_list_pending_shows_proposals() -> None:
    queue = RevisionQueue()
    _populate_queue(queue, n=2)
    out = io.StringIO()
    import contextlib
    with contextlib.redirect_stdout(out):
        count = list_pending(queue)
    assert count == 2
    text = out.getvalue()
    assert "prop:0" in text
    assert "prop:1" in text


def test_run_review_session_auto_approve_all() -> None:
    queue = RevisionQueue()
    _populate_queue(queue, n=2)
    out = io.StringIO()
    summary = run_review_session(
        queue=queue,
        reviewer_id="alice",
        auto_decision="a",
        output_stream=out,
    )
    assert summary == {"approved": 2, "rejected": 0, "skipped": 0}


def test_run_review_session_auto_reject_all() -> None:
    queue = RevisionQueue()
    _populate_queue(queue, n=2)
    out = io.StringIO()
    summary = run_review_session(
        queue=queue,
        reviewer_id="alice",
        auto_decision="r",
        output_stream=out,
    )
    assert summary == {"approved": 0, "rejected": 2, "skipped": 0}


def test_run_review_session_auto_skip_keeps_queue() -> None:
    queue = RevisionQueue()
    _populate_queue(queue, n=2)
    out = io.StringIO()
    summary = run_review_session(
        queue=queue,
        reviewer_id="alice",
        auto_decision="s",
        output_stream=out,
    )
    assert summary == {"approved": 0, "rejected": 0, "skipped": 2}
    assert len(queue.list_pending()) == 2


def test_run_review_session_interactive_input() -> None:
    queue = RevisionQueue()
    _populate_queue(queue, n=2)
    inp = io.StringIO("a\nr\nrejection-reason\n")
    out = io.StringIO()
    summary = run_review_session(
        queue=queue,
        reviewer_id="bob",
        input_stream=inp,
        output_stream=out,
    )
    assert summary["approved"] == 1
    assert summary["rejected"] == 1


def test_main_list_only(capsys) -> None:
    from volvence_zero.cli.protocol_revision_review import main

    rc = main(["--reviewer", "test", "--list-only"])
    assert rc == 0
