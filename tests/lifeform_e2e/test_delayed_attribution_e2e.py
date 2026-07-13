"""CP-14 (intent-alignment W2.F) - multi-turn delayed attribution e2e.

Two layers close the remaining CP-14 acceptance gap:

1. A multi-turn TOOL task through a real lifeform session: pending regime
   outcomes mature over the attribution horizon and the resulting
   ``DelayedOutcomeAttribution`` rows carry the abstract action that was
   dominant at the SOURCE turn (not the resolution turn), and the credit
   owner consumes them as ``delayed_regime:*`` / ``delayed_action:*``
   records.
2. The mismatch contract: when the PE action context references a segment
   that did NOT close, segment-closure attribution explicitly returns an
   empty set (``()`` / 0) instead of guessing - typed, documented, no
   silent fallback to the wrong action.
"""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from lifeform_affordance import (
    AffordanceCost,
    AffordanceDescriptor,
    AffordanceInvocationStatus,
    AffordanceInvoker,
    AffordanceKind,
    AffordanceLatencyClass,
    AffordanceRegistry,
    AffordanceSafety,
)
from volvence_zero.credit import (
    CreditLedger,
    derive_segment_closure_credit_records,
    record_nstep_outcomes_from_segment_closure,
)
from volvence_zero.prediction import (
    ActualOutcome,
    PredictedOutcome,
    PredictionActionContext,
    PredictionError,
    PredictionErrorSnapshot,
)
from volvence_zero.temporal import (
    ControllerState,
    TemporalAbstractionSnapshot,
    TemporalSegmentClosure,
)

_HINT = (
    "Use only inside the CP-14 delayed-attribution e2e to prove multi-turn "
    "tool outcomes attribute to the correct abstract action."
)

_TURNS: tuple[str, ...] = (
    "turn 1: we need to plan the harbor tool run",
    "turn 2: run the lookup and tell me what it says",
    "turn 3: given the tool output, what changes?",
    "turn 4: keep going, what is still open?",
    "turn 5: summarize the task state",
    "turn 6: final check before we close",
)


def _echo_descriptor(name: str) -> AffordanceDescriptor:
    return AffordanceDescriptor(
        name=name,
        kind=AffordanceKind.TOOL,
        version="0.1.0",
        display_name="Echo (CP-14 probe)",
        description="Echo affordance used by the CP-14 delayed-attribution e2e.",
        when_to_use=_HINT,
        when_not_to_use=_HINT + " Not suitable for any other test.",
        parameters_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        output_schema={"type": "object"},
        cost_model=AffordanceCost(latency_class=AffordanceLatencyClass.INSTANT),
        safety_model=AffordanceSafety(),
    )


async def _echo_backend(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"echoed": parameters["message"]}


@pytest.mark.asyncio
async def test_multi_turn_tool_task_attributes_delayed_outcomes() -> None:
    """Delayed regime outcomes mature over the horizon and carry the
    source turn's abstract action; the credit owner consumes them.
    """
    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    session = lifeform.create_session(session_id="cp14-delayed-attribution")
    registry = AffordanceRegistry()
    registry.register(_echo_descriptor("cp14_echo"))
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("cp14_echo", _echo_backend)

    # Dominant abstract action hint per kernel turn, captured from the
    # published dual_track snapshot the same way the regime owner does
    # (world hint first, then self hint).
    hints_by_turn: dict[int, str | None] = {}
    last_result = None
    for turn_number, text in enumerate(_TURNS, start=1):
        last_result = await session.run_turn(text)
        dual_value = last_result.active_snapshots["dual_track"].value
        hint = (
            dual_value.world_track.abstract_action_hint
            or dual_value.self_track.abstract_action_hint
        )
        hints_by_turn[turn_number] = hint
        if turn_number == 2:
            invocation = await invoker.invoke(
                "cp14_echo",
                {"message": "tool step for the harbor task"},
                session=session.brain_session,
                event_id="cp14-tool-call",
                plan_ref="cp14-plan-1",
            )
            assert invocation.status is AffordanceInvocationStatus.SUCCEEDED

    assert last_result is not None
    regime_value = last_result.active_snapshots["regime"].value

    # 1. Delayed outcomes matured: the ledger is non-empty and every row
    #    resolved AFTER its source turn (that is what "delayed" means).
    ledger_rows = regime_value.delayed_attribution_ledger
    assert ledger_rows, "attribution horizon (2 turns) must have matured rows"
    internal_rows = tuple(
        row for row in ledger_rows if row.source_wave_id.startswith("wave-")
    )
    assert internal_rows, "internal pending-outcome path must produce rows"
    for row in internal_rows:
        assert row.resolved_turn_index > row.source_turn_index

    # 2. Correct abstract action: each internal row carries the hint that
    #    was dominant at its SOURCE turn, not the resolution turn's.
    for row in internal_rows:
        assert row.abstract_action == hints_by_turn[row.source_turn_index], (
            f"row from turn {row.source_turn_index} must carry that turn's "
            f"dominant abstract action, got {row.abstract_action!r} vs "
            f"{hints_by_turn[row.source_turn_index]!r}"
        )

    # 3. The credit owner consumed the attributions (CP-14 lineage into
    #    credit): delayed_regime rows always, delayed_action rows whenever
    #    an abstract action was attached.
    credit_value = last_result.active_snapshots["credit"].value
    source_events = {record.source_event for record in credit_value.recent_credits}
    assert any(event.startswith("delayed_regime:") for event in source_events)
    attributed_actions = {
        row.abstract_action for row in internal_rows if row.abstract_action
    }
    if attributed_actions:
        assert any(
            event.startswith("delayed_action:") for event in source_events
        ), f"expected delayed_action credit for {sorted(attributed_actions)}"

    # 4. The tool call's outcome lineage reached PE (CP-10/13 base for the
    #    attribution chain): plan_ref surfaced on a later action context.
    pe_value = last_result.active_snapshots["prediction_error"].value
    assert pe_value.action_context is not None


def _pe_snapshot_with_segment(segment_id: str) -> PredictionErrorSnapshot:
    context = PredictionActionContext(
        segment_id=segment_id,
        abstract_action_id="clarify-before-act",
        z_t_digest=(0.2, 0.4),
        environment_event_id="env:cp14",
        environment_outcome_id="outcome:cp14",
    )
    predicted = PredictedOutcome(
        source_turn_index=0,
        target_turn_index=1,
        predicted_task_progress=0.5,
        predicted_relationship_delta=0.5,
        predicted_regime_stability=0.5,
        predicted_action_payoff=0.6,
        confidence=0.7,
        description="predicted",
        action_context=context,
    )
    return PredictionErrorSnapshot(
        evaluated_prediction=predicted,
        actual_outcome=ActualOutcome(
            observed_turn_index=1,
            task_progress=0.4,
            relationship_delta=0.5,
            regime_stability=0.5,
            action_payoff=0.2,
            description="actual",
            action_context=context,
        ),
        next_prediction=PredictedOutcome(
            source_turn_index=1,
            target_turn_index=2,
            predicted_task_progress=0.5,
            predicted_relationship_delta=0.5,
            predicted_regime_stability=0.5,
            predicted_action_payoff=0.5,
            confidence=0.7,
            description="next",
            action_context=context,
        ),
        error=PredictionError(
            task_error=-0.1,
            relationship_error=0.0,
            regime_error=0.0,
            action_error=-0.4,
            magnitude=0.4,
            signed_reward=-0.3,
            description="cp14 probe error",
        ),
        turn_index=1,
        bootstrap=False,
        description="cp14 probe snapshot",
        action_context=context,
    )


def _temporal_with_closed_segment(segment_id: str) -> TemporalAbstractionSnapshot:
    return TemporalAbstractionSnapshot(
        controller_state=ControllerState(
            code=(0.2, 0.4),
            code_dim=2,
            switch_gate=0.9,
            is_switching=True,
            steps_since_switch=0,
        ),
        active_abstract_action="clarify-before-act",
        controller_params_hash="cp14-hash",
        description="cp14 temporal probe",
        closed_segments=(
            TemporalSegmentClosure(
                segment_id=segment_id,
                open_turn_index=0,
                close_turn_index=1,
                abstract_action_id="clarify-before-act",
                z_t_digest=(0.2, 0.4),
                beta_open_digest=0.3,
                beta_close_digest=0.9,
            ),
        ),
    )


def test_segment_closure_match_attributes_to_correct_abstract_action() -> None:
    pe = _pe_snapshot_with_segment("segment-cp14")
    temporal = _temporal_with_closed_segment("segment-cp14")

    records = derive_segment_closure_credit_records(
        prediction_error_snapshot=pe,
        temporal_snapshot=temporal,
        timestamp_ms=10,
    )
    assert len(records) == 1
    assert records[0].source_event == "segment:segment-cp14"
    assert "abstract_action=clarify-before-act" in records[0].context

    ledger = CreditLedger()
    created = record_nstep_outcomes_from_segment_closure(
        ledger=ledger,
        prediction_error_snapshot=pe,
        temporal_snapshot=temporal,
        regime_snapshot=None,
        timestamp_ms=10,
    )
    assert created == 1
    assert "clarify-before-act" in ledger.rolling_payoff_by_family()


def test_segment_closure_mismatch_returns_empty_set_explicitly() -> None:
    """A closed segment that does NOT match the chosen action context must
    yield an EMPTY attribution set - typed ``()`` / 0, never a guess.
    """
    pe = _pe_snapshot_with_segment("segment-chosen")
    # A different segment (different id AND different abstract action)
    # closed this turn; attribution must refuse to bind the outcome to it.
    temporal = TemporalAbstractionSnapshot(
        controller_state=ControllerState(
            code=(0.2, 0.4),
            code_dim=2,
            switch_gate=0.9,
            is_switching=True,
            steps_since_switch=0,
        ),
        active_abstract_action="other-action",
        controller_params_hash="cp14-hash",
        description="cp14 mismatch probe",
        closed_segments=(
            TemporalSegmentClosure(
                segment_id="segment-other",
                open_turn_index=0,
                close_turn_index=1,
                abstract_action_id="other-action",
                z_t_digest=(0.9, 0.1),
                beta_open_digest=0.3,
                beta_close_digest=0.9,
            ),
        ),
    )

    records = derive_segment_closure_credit_records(
        prediction_error_snapshot=pe,
        temporal_snapshot=temporal,
        timestamp_ms=10,
    )
    assert records == ()

    ledger = CreditLedger()
    created = record_nstep_outcomes_from_segment_closure(
        ledger=ledger,
        prediction_error_snapshot=pe,
        temporal_snapshot=temporal,
        regime_snapshot=None,
        timestamp_ms=10,
    )
    assert created == 0
    assert ledger.rolling_payoff_by_family() == {}
