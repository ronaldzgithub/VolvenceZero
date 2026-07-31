"""Product-side action ledger and typed adapters for relationship memory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from volvence_zero.dialogue_trace import DialogueExternalOutcomeKind
from volvence_zero.reflection import RelationshipUpdateProposal
from volvence_zero.semantic_state import (
    ExternalSemanticEventBatch,
    GenericSemanticEvent,
    SemanticProposalOperation,
)


class RelationshipMemoryAction(str, Enum):
    KEEP = "keep"
    SESSION_ONLY = "session_only"
    DELETE = "delete"
    REWRITE = "rewrite"
    MARK_SENSITIVE = "mark_sensitive"
    NO_PROACTIVE_MENTION = "no_proactive_mention"


class RelationshipMemoryCorrectionKind(str, Enum):
    CONTENT_INACCURATE = "content_inaccurate"
    WRONG_USER_ATTRIBUTION = "wrong_user_attribution"
    STALE = "stale"
    BOUNDARY_PREFERENCE = "boundary_preference"


@dataclass(frozen=True)
class RelationshipMemoryActionRecord:
    action_id: str
    user_id: str
    session_id: str
    item_id: str
    action: str
    request_fingerprint: str
    status: str
    owner_operations: tuple[str, ...]
    replacement_entry_id: str | None
    correction_kind: str | None
    dialogue_outcome_evidence_id: str | None
    dialogue_outcome_kind: str | None
    created_at_ms: int

    def to_json(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "item_id": self.item_id,
            "action": self.action,
            "status": self.status,
            "owner_operations": list(self.owner_operations),
            "replacement_entry_id": self.replacement_entry_id,
            "correction_kind": self.correction_kind,
            "dialogue_outcome_evidence_id": self.dialogue_outcome_evidence_id,
            "dialogue_outcome_kind": self.dialogue_outcome_kind,
            "created_at_ms": self.created_at_ms,
        }


class RelationshipMemoryActionConflictError(ValueError):
    pass


class RelationshipMemoryActionLedger:
    """Owner of durable console action/idempotency evidence, not relationship state."""

    _SCHEMA_VERSION = 1

    def __init__(self, *, persistence_root: str | Path | None = None) -> None:
        self._records: dict[
            tuple[str, str, str, str], RelationshipMemoryActionRecord
        ] = {}
        self._resolved_proposals: dict[tuple[str, str, str], str] = {}
        self._persistence_path = (
            Path(persistence_root) / "relationship-memory-console-actions.json"
            if persistence_root is not None
            else None
        )
        if self._persistence_path is not None and self._persistence_path.exists():
            self._hydrate(self._persistence_path)

    @staticmethod
    def request_fingerprint(
        *,
        action: RelationshipMemoryAction,
        replacement: str | None,
        correction_kind: RelationshipMemoryCorrectionKind | None = None,
    ) -> str:
        payload = (
            f"{action.value}\0{replacement or ''}\0"
            f"{correction_kind.value if correction_kind is not None else ''}"
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def existing(
        self,
        *,
        user_id: str,
        session_id: str,
        item_id: str,
        request_fingerprint: str,
    ) -> RelationshipMemoryActionRecord | None:
        return self._records.get(
            (user_id, session_id, item_id, request_fingerprint)
        )

    def ensure_proposal_open(
        self,
        *,
        user_id: str,
        session_id: str,
        proposal_id: str,
        request_fingerprint: str,
    ) -> None:
        resolved_fingerprint = self._resolved_proposals.get(
            (user_id, session_id, proposal_id)
        )
        if (
            resolved_fingerprint is not None
            and resolved_fingerprint != request_fingerprint
        ):
            raise RelationshipMemoryActionConflictError(
                "relationship memory proposal already has a different console action"
            )

    def record(
        self,
        *,
        user_id: str,
        session_id: str,
        item_id: str,
        action: RelationshipMemoryAction,
        request_fingerprint: str,
        status: str,
        owner_operations: tuple[str, ...],
        replacement_entry_id: str | None,
        correction_kind: RelationshipMemoryCorrectionKind | None,
        dialogue_outcome_evidence_id: str | None,
        dialogue_outcome_kind: DialogueExternalOutcomeKind | None,
        created_at_ms: int,
        resolves_proposal: bool,
    ) -> RelationshipMemoryActionRecord:
        key = (user_id, session_id, item_id, request_fingerprint)
        if key in self._records:
            raise RelationshipMemoryActionConflictError(
                "relationship memory item action must be checked for idempotency first"
            )
        record = RelationshipMemoryActionRecord(
            action_id=f"relationship-memory-action:{uuid4().hex}",
            user_id=user_id,
            session_id=session_id,
            item_id=item_id,
            action=action.value,
            request_fingerprint=request_fingerprint,
            status=status,
            owner_operations=owner_operations,
            replacement_entry_id=replacement_entry_id,
            correction_kind=(
                correction_kind.value if correction_kind is not None else None
            ),
            dialogue_outcome_evidence_id=dialogue_outcome_evidence_id,
            dialogue_outcome_kind=(
                dialogue_outcome_kind.value
                if dialogue_outcome_kind is not None
                else None
            ),
            created_at_ms=created_at_ms,
        )
        self._records[key] = record
        proposal_key = (user_id, session_id, item_id)
        if resolves_proposal:
            self._resolved_proposals[proposal_key] = request_fingerprint
        try:
            self._persist()
        except OSError:
            del self._records[key]
            if resolves_proposal:
                del self._resolved_proposals[proposal_key]
            raise
        return record

    def resolved_proposal_ids(self, *, user_id: str, session_id: str) -> frozenset[str]:
        return frozenset(
            proposal_id
            for (record_user, record_session, proposal_id) in self._resolved_proposals
            if record_user == user_id and record_session == session_id
        )

    def records_for_user(self, *, user_id: str) -> tuple[RelationshipMemoryActionRecord, ...]:
        return tuple(
            sorted(
                (
                    record
                    for record in self._records.values()
                    if record.user_id == user_id
                ),
                key=lambda record: (record.created_at_ms, record.action_id),
            )
        )

    def _persist(self) -> None:
        if self._persistence_path is None:
            return
        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self._SCHEMA_VERSION,
            "records": [
                _action_record_to_persistence_json(record)
                for record in sorted(
                    self._records.values(), key=lambda item: item.action_id
                )
            ],
            "resolved_proposals": [
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "proposal_id": proposal_id,
                    "request_fingerprint": fingerprint,
                }
                for (user_id, session_id, proposal_id), fingerprint in sorted(
                    self._resolved_proposals.items()
                )
            ],
        }
        temporary = self._persistence_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self._persistence_path)

    def _hydrate(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != self._SCHEMA_VERSION:
            raise ValueError(
                "relationship memory action ledger schema_version mismatch"
            )
        records = payload.get("records")
        resolved = payload.get("resolved_proposals")
        if not isinstance(records, list) or not isinstance(resolved, list):
            raise ValueError("relationship memory action ledger payload is invalid")
        for item in records:
            record = _action_record_from_persistence_json(item)
            key = (
                record.user_id,
                record.session_id,
                record.item_id,
                record.request_fingerprint,
            )
            if key in self._records:
                raise ValueError("relationship memory action ledger has duplicate record")
            self._records[key] = record
        for item in resolved:
            if not isinstance(item, dict):
                raise ValueError("relationship memory resolved proposal is invalid")
            key = (
                str(item["user_id"]),
                str(item["session_id"]),
                str(item["proposal_id"]),
            )
            self._resolved_proposals[key] = str(item["request_fingerprint"])


def dialogue_outcome_kind_for_action(
    action: RelationshipMemoryAction,
) -> DialogueExternalOutcomeKind | None:
    if action in {
        RelationshipMemoryAction.DELETE,
        RelationshipMemoryAction.REWRITE,
    }:
        return DialogueExternalOutcomeKind.MISSED
    if action in {
        RelationshipMemoryAction.MARK_SENSITIVE,
        RelationshipMemoryAction.NO_PROACTIVE_MENTION,
    }:
        return DialogueExternalOutcomeKind.OVER_DIRECTIVE
    return None


def _action_record_to_persistence_json(
    record: RelationshipMemoryActionRecord,
) -> dict[str, Any]:
    payload = record.to_json()
    payload["request_fingerprint"] = record.request_fingerprint
    return payload


def _action_record_from_persistence_json(payload: object) -> RelationshipMemoryActionRecord:
    if not isinstance(payload, dict):
        raise ValueError("relationship memory action record is invalid")
    owner_operations = payload["owner_operations"]
    if not isinstance(owner_operations, list):
        raise ValueError("relationship memory owner_operations must be a list")
    return RelationshipMemoryActionRecord(
        action_id=str(payload["action_id"]),
        user_id=str(payload["user_id"]),
        session_id=str(payload["session_id"]),
        item_id=str(payload["item_id"]),
        action=str(payload["action"]),
        request_fingerprint=str(payload["request_fingerprint"]),
        status=str(payload["status"]),
        owner_operations=tuple(str(item) for item in owner_operations),
        replacement_entry_id=(
            str(payload["replacement_entry_id"])
            if payload["replacement_entry_id"] is not None
            else None
        ),
        correction_kind=(
            str(payload["correction_kind"])
            if payload["correction_kind"] is not None
            else None
        ),
        dialogue_outcome_evidence_id=(
            str(payload["dialogue_outcome_evidence_id"])
            if payload["dialogue_outcome_evidence_id"] is not None
            else None
        ),
        dialogue_outcome_kind=(
            str(payload["dialogue_outcome_kind"])
            if payload["dialogue_outcome_kind"] is not None
            else None
        ),
        created_at_ms=int(payload["created_at_ms"]),
    )


def proposal_to_json(proposal: RelationshipUpdateProposal) -> dict[str, Any]:
    return {
        "proposal_id": proposal.proposal_id,
        "target_owner_slot": proposal.target_owner_slot,
        "operation": proposal.operation,
        "description": proposal.human_readable_description,
        "source_evidence": list(proposal.source_evidence),
        "confidence": proposal.confidence,
        "requires_user_confirmation": proposal.requires_user_confirmation,
        "shadow_only": proposal.shadow_only,
    }


def proposal_memory_entry_id(proposal: RelationshipUpdateProposal) -> str | None:
    matches = tuple(
        evidence.removeprefix("memory_entry:")
        for evidence in proposal.source_evidence
        if evidence.startswith("memory_entry:")
    )
    if not matches:
        return None
    if len(matches) != 1 or not matches[0]:
        raise ValueError(
            "relationship memory proposal must identify one memory_entry"
        )
    return matches[0]


def semantic_event_for_action(
    *,
    proposal: RelationshipUpdateProposal | None,
    item_id: str,
    action: RelationshipMemoryAction,
    replacement: str | None,
    created_at_ms: int,
) -> ExternalSemanticEventBatch:
    if action in {
        RelationshipMemoryAction.MARK_SENSITIVE,
        RelationshipMemoryAction.NO_PROACTIVE_MENTION,
    }:
        target_slot = "boundary_consent"
        operation = SemanticProposalOperation.BLOCK
        summary = (
            "User marked a relationship memory item as sensitive."
            if action is RelationshipMemoryAction.MARK_SENSITIVE
            else "User disallowed proactive mention of a relationship memory item."
        )
        detail = summary
    else:
        if proposal is None or proposal.target_owner_slot == "memory":
            raise ValueError(
                "semantic relationship action requires a semantic-owner proposal"
            )
        target_slot = proposal.target_owner_slot
        operation = {
            RelationshipMemoryAction.KEEP: SemanticProposalOperation.REVISE,
            RelationshipMemoryAction.DELETE: SemanticProposalOperation.CLOSE,
            RelationshipMemoryAction.REWRITE: SemanticProposalOperation.REVISE,
        }.get(action)
        if operation is None:
            raise ValueError(
                f"action {action.value!r} does not produce a semantic event"
            )
        summary = (
            replacement
            if action is RelationshipMemoryAction.REWRITE and replacement is not None
            else proposal.human_readable_description
        )
        detail = summary

    event_id = f"relationship-memory-console:{uuid4().hex}"
    event = GenericSemanticEvent(
        event_id=event_id,
        target_slot=target_slot,
        operation=operation,
        summary=summary,
        detail=detail,
        confidence=1.0,
        evidence=(
            f"relationship-memory-console:{action.value}:{item_id}:"
            f"timestamp-{created_at_ms}"
        ),
        control_signal=0.0,
        requires_confirmation=False,
    )
    return ExternalSemanticEventBatch(
        events=(event,),
        source="relationship-memory-console",
        description=(
            f"User-confirmed relationship memory action {action.value} "
            f"for {target_slot}."
        ),
    )


__all__ = [
    "RelationshipMemoryAction",
    "RelationshipMemoryActionConflictError",
    "RelationshipMemoryActionLedger",
    "RelationshipMemoryActionRecord",
    "RelationshipMemoryCorrectionKind",
    "dialogue_outcome_kind_for_action",
    "proposal_memory_entry_id",
    "proposal_to_json",
    "semantic_event_for_action",
]
