"""Semantic state store (Gap 7 / Gap 10 single writer).

:class:`SemanticStateStore` is the single in-process writer for the
nine semantic owner slots. Owner modules observe it via read accessors
(``records_for`` / ``completed_refs_for`` / ``lifecycle_for`` ...) and
mutate it via ``apply(...)`` which routes each proposal through
:mod:`volvence_zero.semantic_state.lifecycle` dispatch.

Slice S.1 (2026-05-04): extracted from the previous monolithic
``semantic_state/__init__.py``.

Packet D (2026-05-12): implements ``HydratableOwnerProtocol`` so the
store's nine slots survive across BrainSession / process boundaries.
See ``docs/specs/owner-hydration.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from volvence_zero.owner_hydration import (
    HydrationOwnerMismatchError,
    HydrationPayloadInvalidError,
    HydrationVersionMismatchError,
    OwnerPersistenceSnapshot,
)
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


_SEMANTIC_STATE_OWNER_NAME = "semantic_state"
_SEMANTIC_STATE_SCHEMA_VERSION = 1


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

    # -----------------------------------------------------------------
    # Packet D (long-horizon-closure): HydratableOwnerProtocol
    # -----------------------------------------------------------------

    def export_persistence_snapshot(self) -> OwnerPersistenceSnapshot:
        """Dump the store's nine slots to a versioned, JSON-friendly
        ``OwnerPersistenceSnapshot``.

        Read-only on store state (no version bump, no side effect).
        Round-trip stable: calling ``export -> hydrate -> export``
        on a fresh store yields the same snapshot. Verified by
        ``tests/contracts/test_owner_hydration_protocol.py``.
        """
        return OwnerPersistenceSnapshot(
            owner_name=_SEMANTIC_STATE_OWNER_NAME,
            schema_version=_SEMANTIC_STATE_SCHEMA_VERSION,
            payload={
                "records": {
                    slot: [_serialize_semantic_record(r) for r in records]
                    for slot, records in self._records.items()
                },
                "completed_refs": {
                    slot: list(refs) for slot, refs in self._completed_refs.items()
                },
                "revision_counts": dict(self._revision_counts),
                "record_lifecycle": {
                    slot: {
                        record_id: [advocacy.value, alignment.value]
                        for record_id, (advocacy, alignment) in lifecycle.items()
                    }
                    for slot, lifecycle in self._record_lifecycle.items()
                },
                "record_followup_policy": {
                    slot: {
                        record_id: policy.value
                        for record_id, policy in policies.items()
                    }
                    for slot, policies in self._record_followup_policy.items()
                },
                "record_outcome": {
                    slot: {
                        record_id: _serialize_outcome_record(slot, outcome)
                        for record_id, outcome in outcomes.items()
                    }
                    for slot, outcomes in self._record_outcome.items()
                },
            },
            description=(
                f"SemanticStateStore snapshot v{_SEMANTIC_STATE_SCHEMA_VERSION} "
                f"({sum(len(r) for r in self._records.values())} live records "
                f"across {len(SEMANTIC_OWNER_SLOTS)} slots)"
            ),
        )

    def hydrate_from_persistence(
        self, snapshot: OwnerPersistenceSnapshot
    ) -> None:
        """Replace the store's nine slots from a previously-exported
        snapshot.

        Idempotent (applying the same snapshot twice yields the same
        store state). Fail-loud on:

        - ``snapshot.owner_name != "semantic_state"`` (wiring bug)
        - ``snapshot.schema_version`` not equal to the owner's known
          version (no migration registered)
        - structurally-broken payload (missing required keys / wrong
          slot names / unknown enum values)
        """
        if snapshot.owner_name != _SEMANTIC_STATE_OWNER_NAME:
            raise HydrationOwnerMismatchError(
                f"SemanticStateStore.hydrate_from_persistence: owner_name "
                f"mismatch — expected {_SEMANTIC_STATE_OWNER_NAME!r}, "
                f"got {snapshot.owner_name!r}"
            )
        if snapshot.schema_version != _SEMANTIC_STATE_SCHEMA_VERSION:
            raise HydrationVersionMismatchError(
                f"SemanticStateStore.hydrate_from_persistence: unknown "
                f"schema_version={snapshot.schema_version!r}; this build "
                f"only knows version {_SEMANTIC_STATE_SCHEMA_VERSION}."
            )
        payload = snapshot.payload
        try:
            records_blob = payload["records"]
            completed_refs_blob = payload["completed_refs"]
            revision_counts_blob = payload["revision_counts"]
            lifecycle_blob = payload["record_lifecycle"]
            policy_blob = payload["record_followup_policy"]
            outcome_blob = payload["record_outcome"]
        except KeyError as exc:
            raise HydrationPayloadInvalidError(
                f"SemanticStateStore.hydrate_from_persistence: missing "
                f"required key {exc.args[0]!r} in payload"
            ) from exc
        # Validate that every slot referenced is in our registry.
        for blob_name, blob in (
            ("records", records_blob),
            ("completed_refs", completed_refs_blob),
            ("revision_counts", revision_counts_blob),
            ("record_lifecycle", lifecycle_blob),
            ("record_followup_policy", policy_blob),
            ("record_outcome", outcome_blob),
        ):
            unknown_slots = set(blob).difference(SEMANTIC_OWNER_SLOTS)
            if unknown_slots:
                raise HydrationPayloadInvalidError(
                    f"SemanticStateStore.hydrate_from_persistence: "
                    f"payload[{blob_name!r}] references unknown slot(s) "
                    f"{sorted(unknown_slots)!r}; expected subset of "
                    f"{SEMANTIC_OWNER_SLOTS!r}"
                )
        # Apply (full reset on each slot — hydration is "replace", not "merge").
        new_records: dict[str, tuple[SemanticRecord, ...]] = {}
        new_completed: dict[str, tuple[str, ...]] = {}
        new_revisions: dict[str, int] = {}
        new_lifecycle: dict[str, dict[str, tuple[AdvocacyState, AlignmentState]]] = {}
        new_followup: dict[str, dict[str, FollowupPolicy]] = {}
        new_outcome: dict[str, dict[str, Any]] = {}
        for slot in SEMANTIC_OWNER_SLOTS:
            new_records[slot] = tuple(
                _deserialize_semantic_record(item)
                for item in records_blob.get(slot, ())
            )
            new_completed[slot] = tuple(completed_refs_blob.get(slot, ()))
            new_revisions[slot] = int(revision_counts_blob.get(slot, 0))
            new_lifecycle[slot] = {
                record_id: (
                    AdvocacyState(values[0]),
                    AlignmentState(values[1]),
                )
                for record_id, values in lifecycle_blob.get(slot, {}).items()
            }
            new_followup[slot] = {
                record_id: FollowupPolicy(value)
                for record_id, value in policy_blob.get(slot, {}).items()
            }
            new_outcome[slot] = {
                record_id: _deserialize_outcome_record(slot, outcome_blob_entry)
                for record_id, outcome_blob_entry in outcome_blob.get(slot, {}).items()
            }
        self._records = new_records
        self._completed_refs = new_completed
        self._revision_counts = new_revisions
        self._record_lifecycle = new_lifecycle
        self._record_followup_policy = new_followup
        self._record_outcome = new_outcome


def _serialize_semantic_record(record: SemanticRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "summary": record.summary,
        "detail": record.detail,
        "confidence": record.confidence,
        "status": record.status,
        "source_turn": record.source_turn,
        "evidence": record.evidence,
        "control_signal": record.control_signal,
    }


def _deserialize_semantic_record(blob: Mapping[str, Any]) -> SemanticRecord:
    try:
        return SemanticRecord(
            record_id=str(blob["record_id"]),
            summary=str(blob["summary"]),
            detail=str(blob["detail"]),
            confidence=float(blob["confidence"]),
            status=str(blob["status"]),
            source_turn=int(blob["source_turn"]),
            evidence=str(blob["evidence"]),
            control_signal=float(blob.get("control_signal", 0.0)),
        )
    except KeyError as exc:
        raise HydrationPayloadInvalidError(
            f"semantic_state record missing required key {exc.args[0]!r}; "
            f"got blob={blob!r}"
        ) from exc


def _serialize_outcome_record(slot: str, outcome: Any) -> dict[str, Any]:
    return {
        "kind": outcome.__class__.__name__,
        "outcome_value": outcome.outcome.value,
        "turn_index": outcome.turn_index,
        "evidence": outcome.evidence,
    }


def _deserialize_outcome_record(slot: str, blob: Mapping[str, Any]) -> Any:
    """Reconstruct the per-slot outcome record. The slot determines
    which dataclass + which enum to use.
    """
    try:
        kind = str(blob["kind"])
        outcome_value = blob["outcome_value"]
        turn_index = int(blob["turn_index"])
        evidence = str(blob["evidence"])
    except KeyError as exc:
        raise HydrationPayloadInvalidError(
            f"semantic_state outcome blob missing required key "
            f"{exc.args[0]!r}; slot={slot!r}, blob={blob!r}"
        ) from exc
    if kind == "_CommitmentOutcomeRecord":
        return _CommitmentOutcomeRecord(
            outcome=CommitmentOutcomeKind(outcome_value),
            turn_index=turn_index,
            evidence=evidence,
        )
    if kind == "_PlanIntentOutcomeRecord":
        return _PlanIntentOutcomeRecord(
            outcome=PlanIntentOutcome(outcome_value),
            turn_index=turn_index,
            evidence=evidence,
        )
    if kind == "_ExecutionResultOutcomeRecord":
        return _ExecutionResultOutcomeRecord(
            outcome=ExecutionResultOutcome(outcome_value),
            turn_index=turn_index,
            evidence=evidence,
        )
    raise HydrationPayloadInvalidError(
        f"semantic_state outcome blob has unknown kind={kind!r} "
        f"(slot={slot!r}); known kinds are _CommitmentOutcomeRecord / "
        f"_PlanIntentOutcomeRecord / _ExecutionResultOutcomeRecord"
    )


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
