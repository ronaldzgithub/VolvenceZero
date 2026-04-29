"""End-to-end tests for Gap 10's outcome tracking on plan_intent and
execution_result owners.

Verifies that driving the owner module with a ``SemanticProposalBatch``
through ``SemanticStateStore.apply`` produces a snapshot whose
``lifecycle_entries`` + outcome aggregate counts match the typed
operation mapping. No kernel-wide propagate \u2014 owner-scoped.
"""

from __future__ import annotations

import asyncio

from volvence_zero.memory import MemorySnapshot
from volvence_zero.runtime import WiringLevel
from volvence_zero.semantic_state import (
    ExecutionResultModule,
    ExecutionResultOutcome,
    NoOpSemanticProposalRuntime,
    PlanIntentModule,
    PlanIntentOutcome,
    SemanticProposal,
    SemanticProposalBatch,
    SemanticProposalOperation,
    SemanticStateStore,
)
from volvence_zero.substrate import FeatureSurfaceSubstrateAdapter


def _drive_owner_once(
    module,
    proposals: tuple[SemanticProposal, ...],
    *,
    turn_index: int,
) -> object:
    """Directly apply proposals to the store and run the owner's snapshot builder.

    Bypasses ``process()`` to avoid needing the full upstream dict.
    The store + module duo is the only thing we need to exercise the
    outcome lifecycle path.
    """
    batch = SemanticProposalBatch(
        proposals=proposals,
        runtime_id="test-runtime",
        schema_version=1,
        description="test batch",
    )
    records = module._store.apply(  # noqa: SLF001 \u2014 owner-scoped test peek
        slot=module.slot_name,
        proposals=proposals,
        turn_index=turn_index,
    )
    return module._build_snapshot(records=records, batch=batch)  # noqa: SLF001


def _proposal(
    *,
    slot: str,
    operation: SemanticProposalOperation,
    proposal_id: str,
    evidence: str = "test evidence",
    confidence: float = 0.8,
) -> SemanticProposal:
    return SemanticProposal(
        proposal_id=proposal_id,
        target_slot=slot,
        operation=operation,
        summary=f"{slot}:{operation.value}",
        detail=f"detail for {proposal_id}",
        confidence=confidence,
        evidence=evidence,
        control_signal=0.2,
    )


def _build_module(module_cls, store: SemanticStateStore, *, turn_index: int):
    return module_cls(
        store=store,
        proposal_runtime=NoOpSemanticProposalRuntime(),
        user_input=None,
        turn_index=turn_index,
        wiring_level=WiringLevel.ACTIVE,
    )


# ---------------------------------------------------------------------------
# PlanIntent outcome tracking
# ---------------------------------------------------------------------------


def test_plan_intent_activate_produces_decision_made_outcome() -> None:
    store = SemanticStateStore()
    module = _build_module(PlanIntentModule, store, turn_index=1)
    snapshot = _drive_owner_once(
        module,
        (
            _proposal(
                slot="plan_intent",
                operation=SemanticProposalOperation.ACTIVATE,
                proposal_id="plan-a",
            ),
        ),
        turn_index=1,
    )
    assert snapshot.outcome_decision_made_count == 1
    assert snapshot.outcome_observed_count == 0
    lifecycle = snapshot.lifecycle_for("plan-a")
    assert lifecycle is not None
    assert lifecycle.last_outcome is PlanIntentOutcome.DECISION_MADE
    assert lifecycle.last_outcome_at_turn == 1
    assert lifecycle.last_outcome_evidence != ""


def test_plan_intent_complete_produces_outcome_observed() -> None:
    store = SemanticStateStore()
    module = _build_module(PlanIntentModule, store, turn_index=2)
    snapshot = _drive_owner_once(
        module,
        (
            _proposal(
                slot="plan_intent",
                operation=SemanticProposalOperation.COMPLETE,
                proposal_id="plan-done",
            ),
        ),
        turn_index=2,
    )
    assert snapshot.outcome_observed_count == 1
    assert snapshot.outcome_decision_made_count == 0
    lifecycle = snapshot.lifecycle_for("plan-done")
    assert lifecycle is not None
    assert lifecycle.last_outcome is PlanIntentOutcome.OUTCOME_OBSERVED


def test_plan_intent_observe_produces_no_outcome() -> None:
    store = SemanticStateStore()
    module = _build_module(PlanIntentModule, store, turn_index=3)
    snapshot = _drive_owner_once(
        module,
        (
            _proposal(
                slot="plan_intent",
                operation=SemanticProposalOperation.OBSERVE,
                proposal_id="plan-obs",
            ),
        ),
        turn_index=3,
    )
    # No typed outcome \u2014 lifecycle entry exists but with None outcome.
    assert snapshot.outcome_decision_made_count == 0
    assert snapshot.outcome_observed_count == 0
    assert snapshot.outcome_assumption_recorded_count == 0
    lifecycle = snapshot.lifecycle_for("plan-obs")
    assert lifecycle is not None
    assert lifecycle.last_outcome is None


def test_plan_intent_aggregates_multiple_outcomes() -> None:
    store = SemanticStateStore()
    module = _build_module(PlanIntentModule, store, turn_index=5)
    snapshot = _drive_owner_once(
        module,
        (
            _proposal(
                slot="plan_intent",
                operation=SemanticProposalOperation.CREATE,
                proposal_id="plan-c1",
            ),
            _proposal(
                slot="plan_intent",
                operation=SemanticProposalOperation.ACTIVATE,
                proposal_id="plan-c2",
            ),
            _proposal(
                slot="plan_intent",
                operation=SemanticProposalOperation.DEFER,
                proposal_id="plan-c3",
            ),
            _proposal(
                slot="plan_intent",
                operation=SemanticProposalOperation.COMPLETE,
                proposal_id="plan-c4",
            ),
        ),
        turn_index=5,
    )
    assert snapshot.outcome_assumption_recorded_count == 1
    assert snapshot.outcome_decision_made_count == 1
    assert snapshot.outcome_problem_progress_assessed_count == 1
    assert snapshot.outcome_observed_count == 1


# ---------------------------------------------------------------------------
# ExecutionResult outcome tracking
# ---------------------------------------------------------------------------


def test_execution_result_complete_produces_tool_outcome() -> None:
    store = SemanticStateStore()
    module = _build_module(ExecutionResultModule, store, turn_index=1)
    snapshot = _drive_owner_once(
        module,
        (
            _proposal(
                slot="execution_result",
                operation=SemanticProposalOperation.COMPLETE,
                proposal_id="exec-ok",
            ),
        ),
        turn_index=1,
    )
    assert snapshot.outcome_tool_outcome_count == 1
    lifecycle = snapshot.lifecycle_for("exec-ok")
    assert lifecycle is not None
    assert lifecycle.last_outcome is ExecutionResultOutcome.TOOL_OUTCOME


def test_execution_result_block_produces_tool_outcome() -> None:
    """A failed tool run (BLOCK operation) still counts as a
    ``tool_outcome`` \u2014 the outcome enum captures the channel, not
    the success/failure direction.
    """
    store = SemanticStateStore()
    module = _build_module(ExecutionResultModule, store, turn_index=1)
    snapshot = _drive_owner_once(
        module,
        (
            _proposal(
                slot="execution_result",
                operation=SemanticProposalOperation.BLOCK,
                proposal_id="exec-fail",
            ),
        ),
        turn_index=1,
    )
    assert snapshot.outcome_tool_outcome_count == 1


def test_execution_result_observe_does_not_create_outcome() -> None:
    store = SemanticStateStore()
    module = _build_module(ExecutionResultModule, store, turn_index=1)
    snapshot = _drive_owner_once(
        module,
        (
            _proposal(
                slot="execution_result",
                operation=SemanticProposalOperation.OBSERVE,
                proposal_id="exec-obs",
            ),
        ),
        turn_index=1,
    )
    assert snapshot.outcome_tool_outcome_count == 0
    lifecycle = snapshot.lifecycle_for("exec-obs")
    assert lifecycle is not None
    assert lifecycle.last_outcome is None


# ---------------------------------------------------------------------------
# Cross-turn persistence of outcome
# ---------------------------------------------------------------------------


def test_outcome_persists_across_turns_for_distinct_records() -> None:
    """An ACTIVATE on turn 1 stamps decision_made on record A; an
    OBSERVE of DIFFERENT record B on turn 3 adds a no-outcome
    lifecycle entry without resetting A.
    """
    store = SemanticStateStore()
    # Turn 1: ACTIVATE produces DECISION_MADE on plan-a
    module_t1 = _build_module(PlanIntentModule, store, turn_index=1)
    snapshot_t1 = _drive_owner_once(
        module_t1,
        (
            _proposal(
                slot="plan_intent",
                operation=SemanticProposalOperation.ACTIVATE,
                proposal_id="plan-a",
            ),
        ),
        turn_index=1,
    )
    assert snapshot_t1.outcome_decision_made_count == 1
    lifecycle_a = snapshot_t1.lifecycle_for("plan-a")
    assert lifecycle_a is not None
    assert lifecycle_a.last_outcome_at_turn == 1
    # Turn 3: OBSERVE on a different record does NOT touch plan-a.
    module_t3 = _build_module(PlanIntentModule, store, turn_index=3)
    snapshot_t3 = _drive_owner_once(
        module_t3,
        (
            _proposal(
                slot="plan_intent",
                operation=SemanticProposalOperation.OBSERVE,
                proposal_id="plan-b",
            ),
        ),
        turn_index=3,
    )
    # plan-a's DECISION_MADE is still there; plan-b got no outcome.
    assert snapshot_t3.outcome_decision_made_count == 1
    lifecycle_a_t3 = snapshot_t3.lifecycle_for("plan-a")
    assert lifecycle_a_t3 is not None
    assert lifecycle_a_t3.last_outcome is PlanIntentOutcome.DECISION_MADE
    assert lifecycle_a_t3.last_outcome_at_turn == 1
    lifecycle_b = snapshot_t3.lifecycle_for("plan-b")
    assert lifecycle_b is not None
    assert lifecycle_b.last_outcome is None
