from __future__ import annotations

import asyncio

from volvence_zero.agent.session import AgentSessionRunner
from volvence_zero.credit.gate import derive_dialogue_outcome_credit_records
from volvence_zero.dialogue_trace import (
    DialogueOutcomeEvidence,
    DialogueOutcomeEvidenceSource,
    DialogueOutcomeKind,
)


def test_derive_dialogue_outcome_credit_records_signs_by_kind() -> None:
    evidence = (
        DialogueOutcomeEvidence(
            evidence_id="e-clarified",
            source=DialogueOutcomeEvidenceSource.OWNER_SNAPSHOT,
            source_owner="CommitmentModule",
            outcome_kind=DialogueOutcomeKind.CLARIFIED,
            confidence=0.6,
        ),
        DialogueOutcomeEvidence(
            evidence_id="e-rejected",
            source=DialogueOutcomeEvidenceSource.OWNER_SNAPSHOT,
            source_owner="CommitmentModule",
            outcome_kind=DialogueOutcomeKind.REJECTED,
            confidence=0.6,
        ),
        DialogueOutcomeEvidence(
            evidence_id="e-scene-closed",
            source=DialogueOutcomeEvidenceSource.SCENE_EVENT,
            source_owner="SceneManager",
            outcome_kind=DialogueOutcomeKind.SCENE_CLOSED,
            confidence=0.9,
        ),
    )

    records = derive_dialogue_outcome_credit_records(
        outcome_evidence=evidence,
        timestamp_ms=10,
    )

    by_event = {record.source_event: record for record in records}
    assert any(
        record.source_event == "dialogue_outcome:clarified:CommitmentModule"
        and record.credit_value > 0.0
        for record in records
    )
    assert any(
        record.source_event == "dialogue_outcome:rejected:CommitmentModule"
        and record.credit_value < 0.0
        for record in records
    )
    # Scene closed has structural sign 0.0 -> no record.
    assert all(
        not record.source_event.startswith("dialogue_outcome:scene_closed")
        for record in records
    )
    assert all(record.level == "turn" for record in records)
    assert all("dialogue_outcome:" in event for event in by_event)


def test_dialogue_outcome_evidence_appears_in_credit_snapshot_after_turn() -> None:
    runner = AgentSessionRunner(session_id="dialogue-credit-session")
    asyncio.run(runner.run_turn("First turn for bootstrap."))
    second = asyncio.run(runner.run_turn("Second turn should produce typed evidence."))

    credit_snapshot = (
        second.active_snapshots.get("credit")
        or second.shadow_snapshots.get("credit")
    )
    assert credit_snapshot is not None
    dialogue_credits = tuple(
        record
        for record in credit_snapshot.value.recent_credits
        if record.source_event.startswith("dialogue_outcome:")
    )
    assert dialogue_credits, "expected dialogue outcome credit records on second turn"
    assert all(record.level == "turn" for record in dialogue_credits)


def test_dialogue_outcome_credit_disabled_when_producers_off() -> None:
    runner = AgentSessionRunner(
        session_id="dialogue-credit-disabled-session",
        dialogue_pe_continued_evidence_enabled=False,
        dialogue_commitment_outcome_evidence_enabled=False,
    )
    asyncio.run(runner.run_turn("First turn for bootstrap."))
    second = asyncio.run(runner.run_turn("Second turn should produce no typed evidence."))

    credit_snapshot = (
        second.active_snapshots.get("credit")
        or second.shadow_snapshots.get("credit")
    )
    assert credit_snapshot is not None
    dialogue_credits = tuple(
        record
        for record in credit_snapshot.value.recent_credits
        if record.source_event.startswith("dialogue_outcome:")
    )
    assert dialogue_credits == ()
