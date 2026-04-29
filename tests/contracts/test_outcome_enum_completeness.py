"""Contract tests for Gap 10: outcome enum taxonomy on plan_intent
/ execution_result owners.

Invariants enforced here:

1. Every ``SemanticProposalOperation`` is mapped to a plan_intent
   outcome (possibly ``None``) and an execution_result outcome
   (possibly ``None``) \u2014 never raises, never silently drops a value.
2. ``PlanIntentOutcome`` and ``ExecutionResultOutcome`` are finite
   enums; if someone adds a value this test reminds them to update
   the mapping and module aggregates.
3. ``PlanIntentLifecycleEntry`` / ``ExecutionResultLifecycleEntry``
   enforce the same outcome-requires-evidence + outcome-requires-turn
   invariants as ``CommitmentLifecycleEntry`` (Gap 7).
"""

from __future__ import annotations

import pytest

from volvence_zero.semantic_state import (
    ExecutionResultLifecycleEntry,
    ExecutionResultOutcome,
    PlanIntentLifecycleEntry,
    PlanIntentOutcome,
    SemanticProposalOperation,
    execution_result_outcome_for_operation,
    plan_intent_outcome_for_operation,
)


# ---------------------------------------------------------------------------
# Enum coverage
# ---------------------------------------------------------------------------


def test_plan_intent_outcome_values_are_exhaustive() -> None:
    assert set(PlanIntentOutcome) == {
        PlanIntentOutcome.DECISION_MADE,
        PlanIntentOutcome.ASSUMPTION_RECORDED,
        PlanIntentOutcome.PROBLEM_PROGRESS_ASSESSED,
        PlanIntentOutcome.OUTCOME_OBSERVED,
    }


def test_execution_result_outcome_values_are_exhaustive() -> None:
    assert set(ExecutionResultOutcome) == {
        ExecutionResultOutcome.USER_FEEDBACK_RECEIVED,
        ExecutionResultOutcome.INSTRUCTION_RECEIVED,
        ExecutionResultOutcome.TOOL_OUTCOME,
        ExecutionResultOutcome.CRYSTAL_EVALUATION,
        ExecutionResultOutcome.CRYSTAL_SUPPRESSION,
        ExecutionResultOutcome.PACKAGE_PUBLICATION,
        ExecutionResultOutcome.BOOTSTRAP_CONSUMPTION,
    }


# ---------------------------------------------------------------------------
# Operation -> outcome mapping is total
# ---------------------------------------------------------------------------


def test_every_operation_has_plan_intent_outcome_mapping() -> None:
    for op in SemanticProposalOperation:
        value = plan_intent_outcome_for_operation(op)
        assert value is None or isinstance(value, PlanIntentOutcome), (
            f"plan_intent mapping for {op.value!r} returned non-outcome / non-None: {value!r}"
        )


def test_every_operation_has_execution_result_outcome_mapping() -> None:
    for op in SemanticProposalOperation:
        value = execution_result_outcome_for_operation(op)
        assert value is None or isinstance(value, ExecutionResultOutcome), (
            f"execution_result mapping for {op.value!r} returned non-outcome / non-None: {value!r}"
        )


def test_plan_intent_activate_and_revise_produce_decision_made() -> None:
    assert (
        plan_intent_outcome_for_operation(SemanticProposalOperation.ACTIVATE)
        is PlanIntentOutcome.DECISION_MADE
    )
    assert (
        plan_intent_outcome_for_operation(SemanticProposalOperation.REVISE)
        is PlanIntentOutcome.DECISION_MADE
    )


def test_plan_intent_complete_and_close_produce_outcome_observed() -> None:
    for op in (
        SemanticProposalOperation.COMPLETE,
        SemanticProposalOperation.CLOSE,
    ):
        assert plan_intent_outcome_for_operation(op) is PlanIntentOutcome.OUTCOME_OBSERVED


def test_execution_result_terminal_ops_produce_tool_outcome() -> None:
    for op in (
        SemanticProposalOperation.COMPLETE,
        SemanticProposalOperation.CLOSE,
        SemanticProposalOperation.BLOCK,
    ):
        assert (
            execution_result_outcome_for_operation(op)
            is ExecutionResultOutcome.TOOL_OUTCOME
        )


def test_execution_result_planning_ops_do_not_produce_outcome() -> None:
    """OBSERVE / CREATE / DEFER / ACTIVATE / REVISE are planning reads on
    ``execution_result``; the outcome must remain None so the caller
    does not overwrite a real outcome with a status read.
    """
    for op in (
        SemanticProposalOperation.OBSERVE,
        SemanticProposalOperation.CREATE,
        SemanticProposalOperation.DEFER,
        SemanticProposalOperation.ACTIVATE,
        SemanticProposalOperation.REVISE,
    ):
        assert execution_result_outcome_for_operation(op) is None


# ---------------------------------------------------------------------------
# Lifecycle entry invariants
# ---------------------------------------------------------------------------


def test_plan_intent_lifecycle_entry_requires_evidence_when_outcome_set() -> None:
    with pytest.raises(ValueError, match="last_outcome_evidence"):
        PlanIntentLifecycleEntry(
            record_id="plan-1",
            last_outcome=PlanIntentOutcome.DECISION_MADE,
            last_outcome_evidence="",
            last_outcome_at_turn=1,
        )


def test_plan_intent_lifecycle_entry_requires_turn_when_outcome_set() -> None:
    with pytest.raises(ValueError, match="last_outcome_at_turn"):
        PlanIntentLifecycleEntry(
            record_id="plan-1",
            last_outcome=PlanIntentOutcome.OUTCOME_OBSERVED,
            last_outcome_evidence="observed outcome",
            last_outcome_at_turn=-1,
        )


def test_plan_intent_lifecycle_entry_fresh_record_is_accepted() -> None:
    entry = PlanIntentLifecycleEntry(record_id="plan-1")
    assert entry.last_outcome is None
    assert entry.last_outcome_evidence == ""
    assert entry.last_outcome_at_turn == -1


def test_execution_result_lifecycle_entry_requires_evidence_when_outcome_set() -> None:
    with pytest.raises(ValueError, match="last_outcome_evidence"):
        ExecutionResultLifecycleEntry(
            record_id="exec-1",
            last_outcome=ExecutionResultOutcome.TOOL_OUTCOME,
            last_outcome_evidence="",
            last_outcome_at_turn=1,
        )


def test_execution_result_lifecycle_entry_requires_turn_when_outcome_set() -> None:
    with pytest.raises(ValueError, match="last_outcome_at_turn"):
        ExecutionResultLifecycleEntry(
            record_id="exec-1",
            last_outcome=ExecutionResultOutcome.CRYSTAL_EVALUATION,
            last_outcome_evidence="eval happened",
            last_outcome_at_turn=-5,
        )


def test_execution_result_lifecycle_entry_fresh_record_is_accepted() -> None:
    entry = ExecutionResultLifecycleEntry(record_id="exec-1")
    assert entry.last_outcome is None
    assert entry.last_outcome_evidence == ""
    assert entry.last_outcome_at_turn == -1
