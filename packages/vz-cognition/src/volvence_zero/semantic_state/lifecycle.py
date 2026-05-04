"""Semantic state lifecycle dispatch (Gap 7 / Gap 10).

Maps :class:`SemanticProposalOperation` values to commitment /
plan_intent / execution_result lifecycle transitions, follow-up
policies, and typed outcome records. Pure functional module; no I/O
and no dependence on the runtime store.

Slice S.1 (2026-05-04): extracted from the previous monolithic
``semantic_state/__init__.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from volvence_zero.semantic_state.contracts import (
    AdvocacyState,
    AlignmentState,
    CommitmentOutcomeKind,
    ExecutionResultOutcome,
    FollowupPolicy,
    PlanIntentOutcome,
    SemanticProposalOperation,
)


# Map each SemanticProposal operation to a lifecycle transition for
# commitment records. The mapping is intentionally narrow \u2014 only
# operations that meaningfully advance advocacy or alignment are
# represented; the rest leave the existing state in place. Missing
# entries keep whatever the previous operation set.
_COMMITMENT_LIFECYCLE_TRANSITIONS: dict[
    SemanticProposalOperation, tuple[AdvocacyState | None, AlignmentState | None]
] = {
    SemanticProposalOperation.OBSERVE: (AdvocacyState.NOT_READY, AlignmentState.UNKNOWN),
    SemanticProposalOperation.CREATE: (AdvocacyState.NOT_READY, AlignmentState.UNKNOWN),
    SemanticProposalOperation.DEFER: (AdvocacyState.READY, None),
    SemanticProposalOperation.ACTIVATE: (AdvocacyState.PROPOSED, None),
    SemanticProposalOperation.REVISE: (AdvocacyState.PROPOSED, AlignmentState.MODIFY),
    SemanticProposalOperation.COMPLETE: (AdvocacyState.PROPOSED, AlignmentState.AGREE),
    SemanticProposalOperation.CLOSE: (AdvocacyState.PROPOSED, None),
    SemanticProposalOperation.BLOCK: (AdvocacyState.PROPOSED, AlignmentState.REJECT),
}


# Map each operation to the follow-up policy that best fits the resulting
# lifecycle state. ``None`` means "leave whatever was set before". A
# freshly-observed commitment starts with ``GENTLE_CHECKIN`` so that the
# FollowupManager treats it as a normal engage-on-due item; ``BLOCK``
# (user rejected) and ``DEFER`` (explicit hold) flip it to ``DEFER_ONLY``
# so the lifeform does not badger the user about a commitment they just
# pushed back against.
_COMMITMENT_FOLLOWUP_POLICY_TRANSITIONS: dict[
    SemanticProposalOperation, FollowupPolicy | None
] = {
    SemanticProposalOperation.OBSERVE: FollowupPolicy.GENTLE_CHECKIN,
    SemanticProposalOperation.CREATE: FollowupPolicy.GENTLE_CHECKIN,
    SemanticProposalOperation.DEFER: FollowupPolicy.DEFER_ONLY,
    SemanticProposalOperation.ACTIVATE: FollowupPolicy.GENTLE_CHECKIN,
    SemanticProposalOperation.REVISE: FollowupPolicy.GENTLE_CHECKIN,
    SemanticProposalOperation.COMPLETE: None,
    SemanticProposalOperation.CLOSE: None,
    SemanticProposalOperation.BLOCK: FollowupPolicy.DEFER_ONLY,
}


# Map each operation to the typed outcome it produces, if any. Used by
# reflection writeback to record a single canonical outcome enum per
# commitment transition so downstream consumers (ETA credit, regime
# calibration, case_memory) can key off a stable label rather than
# reparse lifecycle state pairs. ``None`` means the operation does not
# represent a meaningful outcome (merely an observation / advocacy
# move), so no outcome is recorded.
_COMMITMENT_OUTCOME_TRANSITIONS: dict[
    SemanticProposalOperation, CommitmentOutcomeKind | None
] = {
    SemanticProposalOperation.OBSERVE: None,
    SemanticProposalOperation.CREATE: None,
    SemanticProposalOperation.DEFER: CommitmentOutcomeKind.STALLED,
    SemanticProposalOperation.ACTIVATE: CommitmentOutcomeKind.PROGRESSED,
    SemanticProposalOperation.REVISE: CommitmentOutcomeKind.PROGRESSED,
    SemanticProposalOperation.COMPLETE: CommitmentOutcomeKind.COMPLETED,
    SemanticProposalOperation.CLOSE: CommitmentOutcomeKind.STALLED,
    SemanticProposalOperation.BLOCK: CommitmentOutcomeKind.REJECTED,
}

def commitment_lifecycle_for_operation(
    operation: SemanticProposalOperation,
    *,
    previous: tuple[AdvocacyState, AlignmentState] | None = None,
) -> tuple[AdvocacyState, AlignmentState]:
    """Pure helper exposing the operation \u2192 lifecycle map.

    Public so reflection writeback / evaluation / tests can derive the
    same lifecycle the commitment owner uses, without duplicating the
    truth table. ``previous`` lets a transition leave one axis untouched
    (e.g. ``ACTIVATE`` advances advocacy but leaves alignment as
    whatever the user-side signals last said).
    """
    base_advocacy = previous[0] if previous else AdvocacyState.NOT_READY
    base_alignment = previous[1] if previous else AlignmentState.UNKNOWN
    advocacy, alignment = _COMMITMENT_LIFECYCLE_TRANSITIONS.get(
        operation, (None, None)
    )
    return (
        advocacy if advocacy is not None else base_advocacy,
        alignment if alignment is not None else base_alignment,
    )


def commitment_followup_policy_for_operation(
    operation: SemanticProposalOperation,
    *,
    previous: FollowupPolicy | None = None,
) -> FollowupPolicy:
    """Pure helper exposing the operation \u2192 follow-up policy map.

    Defaults to ``GENTLE_CHECKIN`` when both ``previous`` is None and the
    operation is unmapped, so that callers constructing a fresh lifecycle
    entry always get a usable policy.
    """
    policy = _COMMITMENT_FOLLOWUP_POLICY_TRANSITIONS.get(operation)
    if policy is not None:
        return policy
    return previous or FollowupPolicy.GENTLE_CHECKIN


def commitment_outcome_for_operation(
    operation: SemanticProposalOperation,
) -> CommitmentOutcomeKind | None:
    """Pure helper exposing the operation \u2192 outcome enum.

    ``None`` means the operation did not produce a durable outcome
    (observe / create is a status read, not an outcome). Callers must
    treat ``None`` as "leave the previous outcome in place" \u2014 never
    overwrite a real outcome with nothing.
    """
    return _COMMITMENT_OUTCOME_TRANSITIONS.get(operation)


# Gap 10: plan-intent outcome taxonomy. A plan / intent lifecycle
# maps to four named outcome kinds. OBSERVE / CREATE are "status
# reads" and intentionally do NOT produce a typed outcome; callers
# treat ``None`` as "leave the previous outcome in place".
_PLAN_INTENT_OUTCOME_TRANSITIONS: dict[
    SemanticProposalOperation, PlanIntentOutcome | None
] = {
    SemanticProposalOperation.OBSERVE: None,
    SemanticProposalOperation.CREATE: PlanIntentOutcome.ASSUMPTION_RECORDED,
    SemanticProposalOperation.DEFER: PlanIntentOutcome.PROBLEM_PROGRESS_ASSESSED,
    SemanticProposalOperation.ACTIVATE: PlanIntentOutcome.DECISION_MADE,
    SemanticProposalOperation.REVISE: PlanIntentOutcome.DECISION_MADE,
    SemanticProposalOperation.COMPLETE: PlanIntentOutcome.OUTCOME_OBSERVED,
    SemanticProposalOperation.CLOSE: PlanIntentOutcome.OUTCOME_OBSERVED,
    SemanticProposalOperation.BLOCK: PlanIntentOutcome.PROBLEM_PROGRESS_ASSESSED,
}


def plan_intent_outcome_for_operation(
    operation: SemanticProposalOperation,
) -> PlanIntentOutcome | None:
    """Pure helper exposing the operation \u2192 plan_intent outcome enum."""
    return _PLAN_INTENT_OUTCOME_TRANSITIONS.get(operation)


# Gap 10: execution-result outcome taxonomy. The execution_result
# owner receives external signals via adapters (tool results, profile
# events, etc.) so the mapping here focuses on the status bucket the
# owner actually writes. Product-specific subtypes (crystal
# evaluation / suppression, package publication, bootstrap
# consumption) do not arise from ``SemanticProposalOperation`` alone
# \u2014 they come in through reviewed-knowledge / task events / test
# writes. For those, ``None`` is returned here and the caller passes
# an explicit ``ExecutionResultOutcome`` to the store when writing.
_EXECUTION_RESULT_OUTCOME_TRANSITIONS: dict[
    SemanticProposalOperation, ExecutionResultOutcome | None
] = {
    SemanticProposalOperation.OBSERVE: None,
    SemanticProposalOperation.CREATE: None,
    SemanticProposalOperation.DEFER: None,
    SemanticProposalOperation.ACTIVATE: None,
    SemanticProposalOperation.REVISE: None,
    SemanticProposalOperation.COMPLETE: ExecutionResultOutcome.TOOL_OUTCOME,
    SemanticProposalOperation.CLOSE: ExecutionResultOutcome.TOOL_OUTCOME,
    SemanticProposalOperation.BLOCK: ExecutionResultOutcome.TOOL_OUTCOME,
}


def execution_result_outcome_for_operation(
    operation: SemanticProposalOperation,
) -> ExecutionResultOutcome | None:
    """Pure helper exposing the operation \u2192 execution_result outcome.

    Returns ``None`` for status-read / planning operations so the
    caller can leave the previous outcome in place. Callers that have
    direct typed information (e.g. a ``user_feedback_received`` event
    from a tool_result adapter) should pass an explicit outcome to
    the store rather than rely on this mapping.
    """
    return _EXECUTION_RESULT_OUTCOME_TRANSITIONS.get(operation)


@dataclass(frozen=True)
class _CommitmentOutcomeRecord:
    """Internal record: typed outcome + anchoring turn + evidence."""

    outcome: CommitmentOutcomeKind
    turn_index: int
    evidence: str


@dataclass(frozen=True)
class _PlanIntentOutcomeRecord:
    """Internal record for plan_intent outcome (Gap 10)."""

    outcome: PlanIntentOutcome
    turn_index: int
    evidence: str


@dataclass(frozen=True)
class _ExecutionResultOutcomeRecord:
    """Internal record for execution_result outcome (Gap 10)."""

    outcome: ExecutionResultOutcome
    turn_index: int
    evidence: str


# Per-slot dispatch for operation \u2192 outcome helpers. Lets ``apply``
# call the right helper per slot instead of branching on slot name.
# ``None`` means the slot does not participate in typed-outcome
# tracking (only commitment / plan_intent / execution_result do).
def _outcome_dispatch_for_slot(slot: str, operation: SemanticProposalOperation):
    if slot == "commitment":
        return commitment_outcome_for_operation(operation)
    if slot == "plan_intent":
        return plan_intent_outcome_for_operation(operation)
    if slot == "execution_result":
        return execution_result_outcome_for_operation(operation)
    return None


def _outcome_record_for_slot(
    slot: str,
    outcome: Any,
    *,
    turn_index: int,
    evidence: str,
):
    if slot == "commitment":
        return _CommitmentOutcomeRecord(
            outcome=outcome, turn_index=turn_index, evidence=evidence
        )
    if slot == "plan_intent":
        return _PlanIntentOutcomeRecord(
            outcome=outcome, turn_index=turn_index, evidence=evidence
        )
    if slot == "execution_result":
        return _ExecutionResultOutcomeRecord(
            outcome=outcome, turn_index=turn_index, evidence=evidence
        )
    raise ValueError(
        f"Unsupported outcome-tracking slot {slot!r}; expected one of "
        "commitment / plan_intent / execution_result."
    )
