"""Pending-revision queue + ModificationGate (packet 3.4).

Reflection produces ``ProtocolRevisionProposal``s. Whether
those land directly into the registry, queue for human review,
or get rejected outright is decided by
:func:`evaluate_protocol_revision`. The decision uses the
proposal's ``required_review_level`` plus PE-evidence
heuristics:

* L1 (CASE)      → auto-approve (low risk, easy to revert).
* L2 (KNOWLEDGE) → auto-approve (knowledge churn is low risk).
* L3 (STRATEGY)  → auto-approve when the rule's evidence
  meets the ``min_observation_window`` + magnitude
  thresholds; otherwise queue for human.
* L4 (BOUNDARY / IDENTITY) → ALWAYS queue (fail-safe; never
  auto-mutate a boundary contract).

The queue itself is a simple in-memory store; persistence
is deferred to a follow-up packet (DLaaS platform).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock

from volvence_zero.behavior_protocol import (
    ProtocolRevisionProposal,
    ReviewLevel,
)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------


# L3 auto-approve requires at least this many turns of observation.
L3_AUTO_APPROVE_MIN_OBSERVATION_TURNS: int = 8

# L3 auto-approve requires the proposal evidence to summarize a
# magnitude meeting this threshold (presence-test only — the
# rule's own threshold owns the actual numeric test; here we
# just sanity-check the evidence has been articulated).
L3_AUTO_APPROVE_REQUIRE_NONEMPTY_PE_SIGNATURE: bool = True


# ---------------------------------------------------------------------------
# Decision types
# ---------------------------------------------------------------------------


class ApprovalOutcome(str, Enum):
    """Result of a gate evaluation."""

    AUTO_APPROVED = "auto_approved"
    QUEUED_FOR_HUMAN = "queued_for_human"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ApprovalDecision:
    """Audit record of a gate decision."""

    proposal_id: str
    outcome: ApprovalOutcome
    rationale: str
    decided_at_iso: str


# ---------------------------------------------------------------------------
# evaluate_protocol_revision
# ---------------------------------------------------------------------------


def evaluate_protocol_revision(
    proposal: ProtocolRevisionProposal,
) -> ApprovalDecision:
    """R10 ModificationGate: decide if a proposal can auto-apply."""

    level = proposal.required_review_level

    if level is ReviewLevel.L4:
        return ApprovalDecision(
            proposal_id=proposal.proposal_id,
            outcome=ApprovalOutcome.QUEUED_FOR_HUMAN,
            rationale=(
                "L4 (boundary / identity) revisions never auto-apply; "
                "fail-safe gate routes to human queue."
            ),
            decided_at_iso=_now_iso(),
        )

    if level is ReviewLevel.L3:
        evidence = proposal.evidence
        if (
            evidence.observation_window_turns
            < L3_AUTO_APPROVE_MIN_OBSERVATION_TURNS
        ):
            return ApprovalDecision(
                proposal_id=proposal.proposal_id,
                outcome=ApprovalOutcome.QUEUED_FOR_HUMAN,
                rationale=(
                    f"L3 evidence window "
                    f"{evidence.observation_window_turns} turns "
                    f"below auto-approve threshold "
                    f"{L3_AUTO_APPROVE_MIN_OBSERVATION_TURNS}"
                ),
                decided_at_iso=_now_iso(),
            )
        if (
            L3_AUTO_APPROVE_REQUIRE_NONEMPTY_PE_SIGNATURE
            and not evidence.pe_signature.strip()
        ):
            return ApprovalDecision(
                proposal_id=proposal.proposal_id,
                outcome=ApprovalOutcome.QUEUED_FOR_HUMAN,
                rationale=(
                    "L3 auto-approve requires a non-empty pe_signature; "
                    "queueing for human review."
                ),
                decided_at_iso=_now_iso(),
            )
        return ApprovalDecision(
            proposal_id=proposal.proposal_id,
            outcome=ApprovalOutcome.AUTO_APPROVED,
            rationale=(
                f"L3 auto-approve: window "
                f"{evidence.observation_window_turns} >= "
                f"{L3_AUTO_APPROVE_MIN_OBSERVATION_TURNS} and "
                f"pe_signature present"
            ),
            decided_at_iso=_now_iso(),
        )

    # L1 / L2 → auto-approve.
    return ApprovalDecision(
        proposal_id=proposal.proposal_id,
        outcome=ApprovalOutcome.AUTO_APPROVED,
        rationale=(
            f"{level.value} revision auto-approved (low risk class)"
        ),
        decided_at_iso=_now_iso(),
    )


# ---------------------------------------------------------------------------
# RevisionQueue (in-memory pending review queue)
# ---------------------------------------------------------------------------


@dataclass
class _PendingEntry:
    proposal: ProtocolRevisionProposal
    decision: ApprovalDecision


class RevisionQueue:
    """In-memory pending-review queue for L3 (insufficient evidence) + L4 proposals."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._pending: dict[str, _PendingEntry] = {}

    def submit(
        self,
        proposal: ProtocolRevisionProposal,
        decision: ApprovalDecision,
    ) -> None:
        if decision.outcome is not ApprovalOutcome.QUEUED_FOR_HUMAN:
            raise ValueError(
                "RevisionQueue.submit only accepts QUEUED_FOR_HUMAN "
                f"decisions; got {decision.outcome.value!r}"
            )
        with self._lock:
            self._pending[proposal.proposal_id] = _PendingEntry(
                proposal=proposal, decision=decision
            )

    def list_pending(self) -> tuple[tuple[ProtocolRevisionProposal, ApprovalDecision], ...]:
        with self._lock:
            return tuple(
                (e.proposal, e.decision) for e in self._pending.values()
            )

    def approve(self, proposal_id: str, *, reviewer_id: str) -> ProtocolRevisionProposal:
        with self._lock:
            entry = self._pending.pop(proposal_id, None)
            if entry is None:
                raise KeyError(
                    f"RevisionQueue.approve: no pending proposal with id "
                    f"{proposal_id!r}"
                )
            return entry.proposal

    def reject(self, proposal_id: str, *, reviewer_id: str, reason: str) -> ApprovalDecision:
        if not reason.strip():
            raise ValueError("reject reason must be non-empty")
        with self._lock:
            entry = self._pending.pop(proposal_id, None)
            if entry is None:
                raise KeyError(
                    f"RevisionQueue.reject: no pending proposal with id "
                    f"{proposal_id!r}"
                )
        return ApprovalDecision(
            proposal_id=proposal_id,
            outcome=ApprovalOutcome.REJECTED,
            rationale=(
                f"rejected by {reviewer_id}: {reason}"
            ),
            decided_at_iso=_now_iso(),
        )

    def __len__(self) -> int:
        with self._lock:
            return len(self._pending)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


__all__ = [
    "ApprovalDecision",
    "ApprovalOutcome",
    "L3_AUTO_APPROVE_MIN_OBSERVATION_TURNS",
    "RevisionQueue",
    "evaluate_protocol_revision",
]
