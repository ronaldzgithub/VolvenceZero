"""External outcome -> conditioning bank attribution join (State KV P1).

Design plan exit condition (research/state_kv/01_state_kv_complete_design_plan.md
P1): per-turn audit records carry session_scope / turn_index /
selected_bank_set / bank fingerprints, and one end-to-end case proves that a
returned ``PURCHASE_CONFIRMED`` can be resolved to the bank set that was
live when the rated action was produced.
"""

from __future__ import annotations

import asyncio

import pytest

from volvence_zero.agent.conditioning_lineage import (
    resolve_conditioning_lineage_for_outcome,
)
from volvence_zero.agent.dialogue_trace import DialogueTraceStore
from volvence_zero.dialogue_trace import (
    ConditioningLineage,
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
)
from volvence_zero.environment import build_user_input_environment_event
from volvence_zero.prediction import PredictedOutcome


def _lineage(*, session_scope: str = "session-a", fingerprint: str = "abc123def456") -> ConditioningLineage:
    return ConditioningLineage(
        session_scope=session_scope,
        selected_bank_set=("personal",),
        bank_fingerprints=(("personal", fingerprint),),
    )


def _store_with_turns(
    *turn_lineages: ConditioningLineage | None,
) -> DialogueTraceStore:
    store = DialogueTraceStore()
    for offset, lineage in enumerate(turn_lineages):
        turn = offset + 1
        store.record_action(
            session_id="session-a",
            wave_id=f"wave-{turn}",
            turn_index=turn,
            environment_event=build_user_input_environment_event(
                event_id=f"event-{turn}",
                user_input=f"turn {turn} input",
                scene_id="scene-1",
                timestamp_ms=turn,
            ),
            active_regime="support",
            active_abstract_action="clarify",
            response_text=f"response {turn}",
            response_rationale=f"rationale {turn}",
            next_prediction=PredictedOutcome(turn, turn + 1, 0.5, 0.5, 0.5, 0.5, 0.8, "pred"),
            evaluated_prediction=None,
            actual_outcome=None,
            prediction_error=None,
            conditioning_lineage=lineage,
        )
    return store


def _evidence(
    *,
    session_scope: str = "session-a",
    action_turn_index: int = 1,
    kind: DialogueExternalOutcomeKind = DialogueExternalOutcomeKind.PURCHASE_CONFIRMED,
) -> DialogueExternalOutcomeEvidence:
    return DialogueExternalOutcomeEvidence(
        evidence_id=f"external:environment:{kind.value}:{action_turn_index}",
        turn_index=max(action_turn_index, 0),
        kind=kind,
        source=DialogueExternalOutcomeEvidenceSource.ENVIRONMENT,
        confidence=0.95,
        evidence_ref="crm:order-1042",
        session_scope=session_scope,
        action_turn_index=action_turn_index,
    )


def test_join_resolves_outcome_to_the_rated_turns_bank_set() -> None:
    turn_one = _lineage(fingerprint="fp-turn-one-0001")
    turn_two = _lineage(fingerprint="fp-turn-two-0002")
    store = _store_with_turns(turn_one, turn_two)

    resolved = resolve_conditioning_lineage_for_outcome(
        evidence=_evidence(action_turn_index=1),
        trace_snapshot=store.snapshot(),
    )

    # The join must land on turn 1's state version, not the latest turn's.
    assert resolved == turn_one
    assert resolved != turn_two
    assert resolved.selected_bank_set == ("personal",)
    assert resolved.bank_fingerprints == (("personal", "fp-turn-one-0001"),)


def test_unattributable_evidence_returns_none() -> None:
    store = _store_with_turns(_lineage())

    no_session = _evidence(session_scope="", action_turn_index=1)
    no_action_turn = _evidence(action_turn_index=-1)

    assert not no_session.is_attributable
    assert not no_action_turn.is_attributable
    for evidence in (no_session, no_action_turn):
        assert (
            resolve_conditioning_lineage_for_outcome(
                evidence=evidence,
                trace_snapshot=store.snapshot(),
            )
            is None
        )


def test_missing_turn_returns_none_instead_of_nearest_trace() -> None:
    store = _store_with_turns(_lineage())

    resolved = resolve_conditioning_lineage_for_outcome(
        evidence=_evidence(action_turn_index=99),
        trace_snapshot=store.snapshot(),
    )

    assert resolved is None


def test_turn_without_live_banks_returns_none() -> None:
    store = _store_with_turns(None)

    resolved = resolve_conditioning_lineage_for_outcome(
        evidence=_evidence(action_turn_index=1),
        trace_snapshot=store.snapshot(),
    )

    assert resolved is None


def test_cross_session_join_fails_loud() -> None:
    store = _store_with_turns(_lineage(session_scope="session-a"))

    with pytest.raises(ValueError, match="wrong session"):
        resolve_conditioning_lineage_for_outcome(
            evidence=_evidence(session_scope="session-b", action_turn_index=1),
            trace_snapshot=store.snapshot(),
        )


def test_replay_artifact_exports_lineage_for_audit_join() -> None:
    store = _store_with_turns(_lineage(fingerprint="fp-export-000001"), None)

    artifact = store.export_replay_artifact()
    rows = artifact["turns"]

    assert rows[0]["turn_index"] == 1
    assert rows[0]["conditioning_lineage"] == {
        "session_scope": "session-a",
        "selected_bank_set": ("personal",),
        "bank_fingerprints": (("personal", "fp-export-000001"),),
        "state_encoder_version": "",
        "prefix_generator_version": "",
        "router_version": "",
    }
    # "No bank was live" must stay distinguishable in the exported audit.
    assert rows[1]["conditioning_lineage"] is None


def test_purchase_confirmed_end_to_end_locates_the_live_bank_set() -> None:
    """P1 exit-condition case: a real runner turn records lineage, a later
    PURCHASE_CONFIRMED return joins back to exactly that turn's bank set."""

    from volvence_zero.agent.dialogue import build_standard_dialogue_runner
    from volvence_zero.agent.dialogue._legacy import DEFAULT_DIALOGUE_PROOF_CASES

    runner = build_standard_dialogue_runner(
        profile_label="state-kv-arm-e",
        case=DEFAULT_DIALOGUE_PROOF_CASES[0],
    )

    async def _run() -> None:
        await runner.run_turn("I prefer direct answers and I trust your judgment.")
        await runner.run_turn("My goal is to decide on the offer this week.")

    asyncio.run(_run())

    traces = {
        trace.turn_index: trace
        for trace in runner._dialogue_trace_store.snapshot().traces
    }
    rated_turn = min(traces)
    rated_lineage = traces[rated_turn].conditioning_lineage
    assert rated_lineage is not None
    assert rated_lineage.selected_bank_set == ("personal",)

    evidence = runner.submit_dialogue_outcome(
        kind=DialogueExternalOutcomeKind.PURCHASE_CONFIRMED,
        source=DialogueExternalOutcomeEvidenceSource.ENVIRONMENT,
        confidence=0.95,
        evidence_ref="crm:order-1042",
        action_turn_index=rated_turn,
    )
    assert evidence.is_attributable

    resolved = runner.resolve_dialogue_outcome_attribution(evidence)

    assert resolved == rated_lineage
    assert resolved.selected_bank_set == ("personal",)
    assert resolved.bank_fingerprints[0][0] == "personal"
    assert resolved.bank_fingerprints[0][1]

    # An undeclared action turn stays counted-but-unattributed.
    undeclared = runner.submit_dialogue_outcome(
        kind=DialogueExternalOutcomeKind.PURCHASE_CONFIRMED,
        source=DialogueExternalOutcomeEvidenceSource.ENVIRONMENT,
        confidence=0.95,
        evidence_ref="crm:order-1043",
        action_turn_index=-1,
    )
    assert not undeclared.is_attributable
    assert runner.resolve_dialogue_outcome_attribution(undeclared) is None
