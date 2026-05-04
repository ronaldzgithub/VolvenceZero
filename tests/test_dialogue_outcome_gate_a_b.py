"""Gate A (Failure -> PE) and Gate B (Outcome -> Delayed Credit) tests.

These tests exercise the full ``AgentSessionRunner.run_turn`` path to
prove that ``submit_dialogue_outcome`` causes the right downstream
effects, routed through the ``dialogue_external_outcome`` snapshot slot
and consumed inside ``PredictionErrorModule`` / ``RegimeModule``'s own
``process`` methods (R8).

Gate A: an explicit ``MISSED`` must produce a non-trivial
``relationship_error`` on the PE snapshot published at the next turn,
with provenance referencing the external evidence id.

Gate B: a later ``DECISION_CLEARER`` must produce a
``DelayedOutcomeAttribution`` entry on the regime snapshot whose
``source_turn_index``, ``regime_id``, ``abstract_action`` and
``action_family_version`` can be traced back to the submitted outcome.
"""

from __future__ import annotations

import asyncio

from volvence_zero.agent import default_active_runner
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
)


def test_gate_a_missed_produces_relationship_error_on_next_pe_snapshot() -> None:
    runner = default_active_runner()
    first = asyncio.run(runner.run_turn("I just need to be heard right now."))
    assert "prediction_error" in first.active_snapshots

    evidence = runner.submit_dialogue_outcome(
        kind=DialogueExternalOutcomeKind.MISSED,
        source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
        confidence=0.95,
        description="Test: user says 'you missed me'.",
    )
    second = asyncio.run(runner.run_turn("You missed me."))

    external_outcome = second.active_snapshots["dialogue_external_outcome"].value
    assert any(
        entry.evidence_id == evidence.evidence_id for entry in external_outcome.entries
    ), "Submitted evidence must appear on the dialogue_external_outcome snapshot."

    pe_snapshot = second.active_snapshots["prediction_error"].value
    assert pe_snapshot.bootstrap is False
    # Gate A: relationship_error must be non-trivially non-zero.
    assert abs(pe_snapshot.error.relationship_error) > 0.1, (
        f"MISSED should bump relationship_error; got {pe_snapshot.error}"
    )
    # Provenance: the actual outcome records the external evidence id.
    assert evidence.evidence_id in pe_snapshot.actual_outcome.external_outcome_refs


def test_gate_b_decision_clearer_produces_delayed_attribution() -> None:
    runner = default_active_runner()
    first = asyncio.run(runner.run_turn("Help me weigh two options."))
    regime_before = first.active_snapshots.get("regime") or first.shadow_snapshots.get(
        "regime"
    )
    assert regime_before is not None
    active_before = regime_before.value.active_regime.regime_id

    evidence = runner.submit_dialogue_outcome(
        kind=DialogueExternalOutcomeKind.DECISION_CLEARER,
        source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
        confidence=0.9,
    )
    second = asyncio.run(runner.run_turn("I know what I want to do now."))

    regime_after = second.active_snapshots.get("regime") or second.shadow_snapshots.get(
        "regime"
    )
    assert regime_after is not None
    attributions = regime_after.value.delayed_attributions
    # Gate B: at least one DelayedOutcomeAttribution must come from our
    # external DECISION_CLEARER evidence.
    external_wave_id = f"external:{evidence.evidence_id}"
    matching = [
        attr
        for attr in attributions
        if attr.source_wave_id == external_wave_id
    ]
    assert matching, (
        f"Expected a DelayedOutcomeAttribution with source_wave_id="
        f"'{external_wave_id}' in {attributions}"
    )
    attribution = matching[0]
    # The score must be clearly positive (DECISION_CLEARER maps to 0.90
    # before confidence scaling; scaled by 0.9 confidence it lands near
    # 0.5 + 0.40 * 0.9 = 0.86).
    assert attribution.outcome_score > 0.7
    # Active regime field must be populated (not the empty placeholder).
    assert attribution.regime_id
    # If the regime is still the same as before, the attribution must
    # be tagged with that regime (the user scored the regime we were
    # actually in for turn 1).
    if active_before == regime_after.value.active_regime.regime_id:
        assert attribution.regime_id == active_before


def test_llm_source_rejected_by_default() -> None:
    runner = default_active_runner()
    asyncio.run(runner.run_turn("hi"))
    import pytest

    with pytest.raises(ValueError, match="allow_llm_outcome_proposals"):
        runner.submit_dialogue_outcome(
            kind=DialogueExternalOutcomeKind.MISSED,
            source=DialogueExternalOutcomeEvidenceSource.LLM_PROPOSAL,
            confidence=0.9,
        )


def test_submit_dialogue_outcome_attaches_structural_evidence_to_trace() -> None:
    runner = default_active_runner()
    asyncio.run(runner.run_turn("First turn."))
    evidence = runner.submit_dialogue_outcome(
        kind=DialogueExternalOutcomeKind.MISSED,
        source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
        confidence=0.9,
    )

    # The trace should now carry structural evidence bridging the external
    # kind into the conservative DialogueOutcomeKind vocabulary.
    trace_snapshot = runner._dialogue_trace_store.snapshot()  # noqa: SLF001 (test introspection)
    structural_ids: list[str] = []
    for trace in trace_snapshot.traces:
        for piece in trace.outcome.structured_evidence:
            structural_ids.append(piece.evidence_id)
    expected = f"external-bridge:{evidence.evidence_id}"
    assert expected in structural_ids, (
        f"expected structural bridge {expected}, got {structural_ids}"
    )
