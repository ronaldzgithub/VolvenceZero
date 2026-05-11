"""Candidate review helpers (packet 2.4).

DocumentUptake yields a :class:`BehaviorProtocolCandidate` with
``requires_review=True``. Before the kernel-side
``ProtocolRegistryModule`` will load it, the candidate must run
through this review layer to:

1. Validate the candidate's claimed review level matches its
   highest-risk content (boundary L4 admin / strategy L3 / etc.)
2. Stamp an approval audit (reviewer_id + evidence + timestamp)
3. Return a *new* :class:`BehaviorProtocol` with
   ``review_status=SHADOW`` (auto path) or ``ACTIVE`` (after a
   second human approval — out-of-scope for this packet).

Rejection produces an explicit :class:`CandidateRejection`
record (no silent failure) so the upstream uptake pipeline
records why a candidate was discarded.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, replace as _replace

from volvence_zero.behavior_protocol import (
    BehaviorProtocol,
    BehaviorProtocolCandidate,
    ProtocolRevision,
    ReviewLevel,
    ReviewStatus,
)


@dataclass(frozen=True)
class CandidateApproval:
    """Audit record produced when a candidate is approved into SHADOW."""

    candidate_id: str
    reviewer_id: str
    approved_at_iso: str
    review_level_required: ReviewLevel
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class CandidateRejection:
    """Audit record produced when a candidate is rejected.

    Returned in lieu of a protocol when the candidate fails the
    review gate. Caller MUST handle this case explicitly (no
    silent fallback to ``None``).
    """

    candidate_id: str
    reviewer_id: str
    rejected_at_iso: str
    reason: str


def required_review_level(
    candidate: BehaviorProtocolCandidate,
) -> ReviewLevel:
    """Derive the highest review level required by the candidate's content.

    * Has any boundary contract → L4 (boundary mutation requires
      admin sign-off — this is the safety floor).
    * Has any strategy prior (and no boundary) → L3.
    * Has knowledge / case content only → L2.
    * Otherwise (identity / metadata only) → L1.

    Reviewers can override the default by passing an explicit
    ``minimum_level`` to :func:`approve_candidate`.
    """

    inner = candidate.protocol
    if inner.boundary_contracts:
        return ReviewLevel.L4
    if inner.strategy_priors:
        return ReviewLevel.L3
    if inner.knowledge_seeds or inner.signature_cases:
        return ReviewLevel.L2
    return ReviewLevel.L1


def approve_candidate(
    candidate: BehaviorProtocolCandidate,
    *,
    reviewer_id: str,
    evidence: tuple[str, ...] = (),
    minimum_level: ReviewLevel | None = None,
    target_status: ReviewStatus = ReviewStatus.SHADOW,
) -> tuple[BehaviorProtocol, CandidateApproval]:
    """Approve a candidate into SHADOW (default) or ACTIVE review_status.

    Behaviour:

    * Computes :func:`required_review_level` and ensures
      ``minimum_level`` (if provided) is at least as high. If
      not, raises ``PermissionError`` (the reviewer doesn't
      meet the bar).
    * Returns a *new* ``BehaviorProtocol`` (frozen replace)
      with ``review_status`` set to ``target_status``.
    * Appends a :class:`ProtocolRevision` entry recording the
      approval.

    Caller is the lifeform-side review pipeline; the returned
    protocol can be passed to
    ``ProtocolRegistryModule.load_protocol_candidate`` (packet
    2.4 owner-side helper).
    """

    if not reviewer_id.strip():
        raise ValueError("approve_candidate.reviewer_id must be non-empty")

    required = required_review_level(candidate)
    if minimum_level is not None:
        # Convert to numeric for comparison.
        if _level_ord(minimum_level) < _level_ord(required):
            raise PermissionError(
                f"reviewer authorised at {minimum_level.value!r} but "
                f"candidate requires {required.value!r}"
            )

    approved_at = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()

    approval = CandidateApproval(
        candidate_id=candidate.protocol.protocol_id,
        reviewer_id=reviewer_id,
        approved_at_iso=approved_at,
        review_level_required=required,
        evidence=evidence,
    )

    revision = ProtocolRevision(
        revision_id=f"approval:{candidate.protocol.protocol_id}:{approved_at}",
        revised_at_tick=0,
        revised_by=reviewer_id,
        description=(
            f"DocumentUptake candidate approved at "
            f"{required.value} by {reviewer_id}; "
            f"target_status={target_status.value}"
        ),
        affected_field="review_status",
    )

    inner = candidate.protocol
    approved = _replace(
        inner,
        review_status=target_status,
        revision_log=inner.revision_log + (revision,),
    )
    return approved, approval


def reject_candidate(
    candidate: BehaviorProtocolCandidate,
    *,
    reviewer_id: str,
    reason: str,
) -> CandidateRejection:
    """Reject a candidate; produce an audit record.

    Use when the candidate fails review (e.g. provenance
    confidence too low, extracted content not matching
    expected schema). Caller must NOT pass a rejected
    candidate to ``load_protocol_candidate``.
    """

    if not reviewer_id.strip():
        raise ValueError("reject_candidate.reviewer_id must be non-empty")
    if not reason.strip():
        raise ValueError("reject_candidate.reason must be non-empty")

    return CandidateRejection(
        candidate_id=candidate.protocol.protocol_id,
        reviewer_id=reviewer_id,
        rejected_at_iso=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_LEVEL_ORDER = {
    ReviewLevel.L1: 1,
    ReviewLevel.L2: 2,
    ReviewLevel.L3: 3,
    ReviewLevel.L4: 4,
}


def _level_ord(level: ReviewLevel) -> int:
    return _LEVEL_ORDER[level]


__all__ = [
    "CandidateApproval",
    "CandidateRejection",
    "approve_candidate",
    "reject_candidate",
    "required_review_level",
]
