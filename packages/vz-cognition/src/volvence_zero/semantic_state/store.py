"""Semantic state store (Gap 7 / Gap 10 single writer).

:class:`SemanticStateStore` is the single in-process writer for the
nine semantic owner slots. Owner modules observe it via read accessors
(``records_for`` / ``completed_refs_for`` / ``lifecycle_for`` ...) and
mutate it via ``apply(...)`` which routes each proposal through
:mod:`volvence_zero.semantic_state.lifecycle` dispatch.

Slice S.1 (2026-05-04): extracted from the previous monolithic
``semantic_state/__init__.py``.
"""

from __future__ import annotations

from typing import Any

from volvence_zero.semantic_state.contracts import (
    SEMANTIC_OWNER_SLOTS,
    AdvocacyState,
    AlignmentState,
    CommitmentOutcomeKind,
    ExecutionResultOutcome,
    FollowupPolicy,
    PlanIntentOutcome,
    SemanticProposal,
    SemanticProposalOperation,
    SemanticRecord,
    _clamp,
)
from volvence_zero.semantic_state.lifecycle import (
    _CommitmentOutcomeRecord,
    _ExecutionResultOutcomeRecord,
    _PlanIntentOutcomeRecord,
    _outcome_dispatch_for_slot,
    _outcome_record_for_slot,
    commitment_followup_policy_for_operation,
    commitment_lifecycle_for_operation,
    commitment_outcome_for_operation,
    execution_result_outcome_for_operation,
    plan_intent_outcome_for_operation,
)


class SemanticStateStore:
    def __init__(self) -> None:
        self._records: dict[str, tuple[SemanticRecord, ...]] = {slot: () for slot in SEMANTIC_OWNER_SLOTS}
        self._completed_refs: dict[str, tuple[str, ...]] = {slot: () for slot in SEMANTIC_OWNER_SLOTS}
        self._revision_counts: dict[str, int] = {slot: 0 for slot in SEMANTIC_OWNER_SLOTS}
        # Per-record lifecycle state for the commitment owner (and any
        # other owner that later wants to consume it). Stored as
        # ``slot -> {record_id -> (advocacy, alignment)}`` so the latest
        # operation's transition wins and prior operations' state on the
        # untouched axis is preserved (see
        # ``commitment_lifecycle_for_operation``'s ``previous`` semantics).
        self._record_lifecycle: dict[
            str, dict[str, tuple[AdvocacyState, AlignmentState]]
        ] = {slot: {} for slot in SEMANTIC_OWNER_SLOTS}
        # Per-record follow-up policy. Same GC semantics as lifecycle.
        self._record_followup_policy: dict[str, dict[str, FollowupPolicy]] = {
            slot: {} for slot in SEMANTIC_OWNER_SLOTS
        }
        # Per-record typed outcome, anchored to the turn it was produced
        # and carrying non-empty evidence. Value type varies per slot:
        # - commitment   -> _CommitmentOutcomeRecord
        # - plan_intent  -> _PlanIntentOutcomeRecord  (Gap 10)
        # - execution_result -> _ExecutionResultOutcomeRecord  (Gap 10)
        # Other slots never populate this map.
        self._record_outcome: dict[str, dict[str, Any]] = {
            slot: {} for slot in SEMANTIC_OWNER_SLOTS
        }

    def apply(self, *, slot: str, proposals: tuple[SemanticProposal, ...], turn_index: int) -> tuple[SemanticRecord, ...]:
        existing = list(self._records[slot])
        completed_refs = list(self._completed_refs[slot])
        revision_count = self._revision_counts[slot]
        lifecycle_map = self._record_lifecycle[slot]
        policy_map = self._record_followup_policy[slot]
        outcome_map = self._record_outcome[slot]
        for proposal in proposals:
            if proposal.target_slot != slot:
                continue
            if proposal.operation in {SemanticProposalOperation.REVISE, SemanticProposalOperation.ACTIVATE}:
                revision_count += 1
            if proposal.operation in {SemanticProposalOperation.COMPLETE, SemanticProposalOperation.CLOSE}:
                completed_refs.append(proposal.proposal_id)
            status = {
                SemanticProposalOperation.DEFER: "deferred",
                SemanticProposalOperation.COMPLETE: "completed",
                SemanticProposalOperation.CLOSE: "closed",
                SemanticProposalOperation.BLOCK: "blocked",
            }.get(proposal.operation, "active")
            existing.append(
                SemanticRecord(
                    record_id=proposal.proposal_id,
                    summary=proposal.summary,
                    detail=proposal.detail,
                    confidence=_clamp(proposal.confidence),
                    status=status,
                    source_turn=turn_index,
                    evidence=proposal.evidence,
                    control_signal=_clamp(proposal.control_signal),
                )
            )
            previous = lifecycle_map.get(proposal.proposal_id)
            lifecycle_map[proposal.proposal_id] = (
                commitment_lifecycle_for_operation(
                    proposal.operation, previous=previous
                )
            )
            # Follow-up policy: keep previous if the operation does not
            # prescribe one; default is GENTLE_CHECKIN via the helper.
            policy_map[proposal.proposal_id] = commitment_followup_policy_for_operation(
                proposal.operation,
                previous=policy_map.get(proposal.proposal_id),
            )
            # Outcome: only record when the operation produces a typed
            # outcome. Evidence MUST be non-empty \u2014 fall back to the
            # proposal's evidence field or (as last resort) a short
            # operation+summary trace so the outcome never ships with an
            # empty audit string. Never silently overwrite an existing
            # outcome with None. Per-slot dispatch lets commitment /
            # plan_intent / execution_result each carry their own
            # outcome taxonomy without a mega-if.
            outcome_kind = _outcome_dispatch_for_slot(slot, proposal.operation)
            if outcome_kind is not None:
                evidence_text = proposal.evidence.strip() or (
                    f"op={proposal.operation.value} summary={proposal.summary}".strip()
                )
                if not evidence_text:
                    evidence_text = (
                        f"op={proposal.operation.value} "
                        f"record_id={proposal.proposal_id}"
                    )
                outcome_map[proposal.proposal_id] = _outcome_record_for_slot(
                    slot,
                    outcome_kind,
                    turn_index=turn_index,
                    evidence=evidence_text[:320],
                )
        self._records[slot] = tuple(existing[-12:])
        self._completed_refs[slot] = tuple(completed_refs[-12:])
        self._revision_counts[slot] = revision_count
        # Garbage-collect lifecycle / policy / outcome entries whose
        # record id has fallen out of the bounded window. Avoids
        # unbounded growth across long sessions while still letting
        # late-arriving proposals reuse earlier ids during the same
        # session.
        live_ids = {record.record_id for record in self._records[slot]}
        for record_id in tuple(lifecycle_map.keys()):
            if record_id not in live_ids:
                del lifecycle_map[record_id]
        for record_id in tuple(policy_map.keys()):
            if record_id not in live_ids:
                del policy_map[record_id]
        for record_id in tuple(outcome_map.keys()):
            if record_id not in live_ids:
                del outcome_map[record_id]
        return self._records[slot]

    def records_for(self, slot: str) -> tuple[SemanticRecord, ...]:
        return self._records[slot]

    def completed_refs_for(self, slot: str) -> tuple[str, ...]:
        return self._completed_refs[slot]

    def revision_count_for(self, slot: str) -> int:
        return self._revision_counts[slot]

    def lifecycle_for(
        self, slot: str
    ) -> dict[str, tuple[AdvocacyState, AlignmentState]]:
        """Return a copy of the per-record lifecycle map for ``slot``."""
        return dict(self._record_lifecycle[slot])

    def followup_policy_for(self, slot: str) -> dict[str, FollowupPolicy]:
        """Return a copy of the per-record follow-up policy map for ``slot``."""
        return dict(self._record_followup_policy[slot])

    def outcome_for(self, slot: str) -> dict[str, Any]:
        """Return a copy of the per-record typed-outcome map for ``slot``.

        Value type varies per slot (see ``_record_outcome`` attribute
        docstring). Callers that care about the typed enum should
        inspect ``record.outcome`` after lookup.
        """
        return dict(self._record_outcome[slot])


def clone_semantic_store(source: SemanticStateStore) -> SemanticStateStore:
    target = SemanticStateStore()
    for slot in SEMANTIC_OWNER_SLOTS:
        target._records[slot] = source.records_for(slot)
        target._completed_refs[slot] = source.completed_refs_for(slot)
        target._revision_counts[slot] = source.revision_count_for(slot)
        target._record_lifecycle[slot] = source.lifecycle_for(slot)
        target._record_followup_policy[slot] = source.followup_policy_for(slot)
        target._record_outcome[slot] = source.outcome_for(slot)
    return target
