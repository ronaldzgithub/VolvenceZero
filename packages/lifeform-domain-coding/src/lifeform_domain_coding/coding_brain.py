"""Memory-first Coding Brain product controller.

The controller owns only bounded product lineage and idempotency. Cognitive
state stays in the existing Memory, semantic-state, Prediction Error, credit,
and temporal owners and is reached exclusively through ``LifeformSession``.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field

from lifeform_core import LifeformSession, TurnTriggerKind
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
)
from volvence_zero.memory import (
    MemoryEntry,
    MemoryStratum,
    MemoryWriteRequest,
    RetrievalQuery,
    Track,
)
from volvence_zero.prediction import PredictionErrorSnapshot

from lifeform_domain_coding.coding_brain_contracts import (
    CodingAdviceSnapshot,
    CodingContextPackSnapshot,
    CodingContextRequest,
    CodingOutcomeKind,
    CodingOutcomeReceipt,
    CodingOutcomeReport,
    CodingOutcomeRoute,
    stable_content_sha256,
)


class CodingBrainError(RuntimeError):
    """Base class for product-controller failures."""


class CodingBrainConflictError(CodingBrainError):
    """An idempotency key was reused with a different immutable payload."""


class CodingBrainLineageError(CodingBrainError):
    """An outcome references an unknown or cross-session Context Pack."""


class CodingBrainReadOnlyError(CodingBrainError):
    """A mutating Coding Brain operation targeted historical state."""


@dataclass(frozen=True)
class _ContextLineage:
    request_digest: str
    snapshot: CodingContextPackSnapshot


@dataclass(frozen=True)
class _OutcomeLineage:
    report_digest: str
    receipt: CodingOutcomeReceipt


@dataclass
class _SessionLedger:
    contexts_by_request_id: dict[str, _ContextLineage] = field(default_factory=dict)
    contexts_by_pack_id: dict[str, CodingContextPackSnapshot] = field(
        default_factory=dict
    )
    outcomes_by_outcome_id: dict[str, _OutcomeLineage] = field(default_factory=dict)


_NEGATIVE_OUTCOMES = frozenset(
    {
        CodingOutcomeKind.TASK_REGRESSED,
        CodingOutcomeKind.REVIEW_CHANGES_REQUESTED,
        CodingOutcomeKind.REVERTED,
    }
)

_TASK_STATUS = {
    CodingOutcomeKind.TASK_VERIFIED: "completed",
    CodingOutcomeKind.TASK_REGRESSED: "failed",
    CodingOutcomeKind.REVIEW_APPROVED: "completed",
    CodingOutcomeKind.REVIEW_CHANGES_REQUESTED: "failed",
    CodingOutcomeKind.MERGED: "completed",
    CodingOutcomeKind.REVERTED: "failed",
}


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _retrieval_facets(request: CodingContextRequest) -> tuple[str, ...]:
    facets = [
        "coding-brain",
        f"project:{_short_hash(request.project_id)}",
        f"repository:{_short_hash(request.repository_id)}",
        f"task-kind:{request.task_kind.value}",
    ]
    facets.extend(f"path:{_short_hash(path)}" for path in request.target_paths)
    return tuple(facets)


def _retrieval_query_text(request: CodingContextRequest) -> str:
    parts = [
        request.task_summary,
        f"task kind {request.task_kind.value}",
        *(f"target path {path}" for path in request.target_paths),
    ]
    # Mechanical cap over explicit host fields; no semantic classification.
    return "\n".join(parts)[:16_000]


def _render_memory(
    entries: tuple[MemoryEntry, ...],
    *,
    max_chars: int,
) -> tuple[str, tuple[str, ...], bool]:
    header = "[coding-brain memory context v1]"
    if not entries:
        empty = (
            f"{header}\n"
            "No prior episodic or durable coding outcome matched this task."
        )
        return empty[:max_chars], (), len(empty) > max_chars

    rendered = header
    source_ids: list[str] = []
    truncated = False
    for index, entry in enumerate(entries, start=1):
        block = f"\n\n[{index}] entry_id={entry.entry_id}\n{entry.content}"
        remaining = max_chars - len(rendered)
        if remaining <= 0:
            truncated = True
            break
        if len(block) <= remaining:
            rendered += block
            source_ids.append(entry.entry_id)
            continue
        rendered += block[:remaining]
        source_ids.append(entry.entry_id)
        truncated = True
        break
    if len(source_ids) < len(entries):
        truncated = True
    return rendered, tuple(source_ids), truncated


def _prediction_error_snapshot(turn_result: object) -> PredictionErrorSnapshot:
    active_snapshots = turn_result.active_snapshots
    published = active_snapshots.get("prediction_error")
    if published is None:
        raise RuntimeError("Coding Brain turn published no prediction_error snapshot")
    if not isinstance(published.value, PredictionErrorSnapshot):
        raise TypeError(
            "prediction_error slot must publish PredictionErrorSnapshot"
        )
    return published.value


def _context_digest_payload(
    *,
    session_id: str,
    request: CodingContextRequest,
    generated_at_ms: int,
    source_turn_index: int,
    rendered_context: str,
    source_entry_ids: tuple[str, ...],
    retrieval_facets: tuple[str, ...],
    memory_entry_count: int,
    truncated: bool,
    settled_outcome_evidence_refs: tuple[str, ...],
    pe_magnitude: float,
    pe_bootstrap: bool,
    advice: CodingAdviceSnapshot,
) -> dict[str, object]:
    return {
        "schema_version": "coding-context-pack.v1",
        "session_id": session_id,
        "request": request.to_json(),
        "generated_at_ms": generated_at_ms,
        "source_turn_index": source_turn_index,
        "rendered_context": rendered_context,
        "source_entry_ids": list(source_entry_ids),
        "retrieval_facets": list(retrieval_facets),
        "memory_entry_count": memory_entry_count,
        "truncated": truncated,
        "settled_outcome_evidence_refs": list(settled_outcome_evidence_refs),
        "pe_magnitude": pe_magnitude,
        "pe_bootstrap": pe_bootstrap,
        "advice": advice.to_json(),
        "wiring_level": "active",
    }


def _outcome_memory_content(
    *,
    request: CodingContextRequest,
    report: CodingOutcomeReport,
) -> str:
    changed_paths = ", ".join(report.changed_paths) or "none-declared"
    target_paths = ", ".join(request.target_paths) or "none-declared"
    return (
        "[coding-brain typed outcome]\n"
        f"project={request.project_id}\n"
        f"repository={request.repository_id}\n"
        f"task={request.task_id}\n"
        f"task_kind={request.task_kind.value}\n"
        f"outcome={report.kind.value}\n"
        f"source={report.source.value}\n"
        f"summary={report.summary}\n"
        f"detail={report.detail}\n"
        f"target_paths={target_paths}\n"
        f"changed_paths={changed_paths}\n"
        f"evidence_ref={report.evidence_ref}"
    )


class CodingBrainController:
    """Context Pack/outcome adapter with bounded non-cognitive lineage."""

    def __init__(
        self,
        *,
        max_records_per_session: int = 512,
        max_session_ledgers: int = 1_024,
    ) -> None:
        if isinstance(max_records_per_session, bool) or max_records_per_session < 1:
            raise ValueError("max_records_per_session must be positive")
        if isinstance(max_session_ledgers, bool) or max_session_ledgers < 1:
            raise ValueError("max_session_ledgers must be positive")
        self._max_records_per_session = max_records_per_session
        self._max_session_ledgers = max_session_ledgers
        self._ledgers: dict[str, _SessionLedger] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def build_context_pack(
        self,
        *,
        session: LifeformSession,
        request: CodingContextRequest,
        generated_at_ms: int | None = None,
    ) -> CodingContextPackSnapshot:
        snapshot, _created = await self.publish_context_pack(
            session=session,
            request=request,
            generated_at_ms=generated_at_ms,
        )
        return snapshot

    async def publish_context_pack(
        self,
        *,
        session: LifeformSession,
        request: CodingContextRequest,
        generated_at_ms: int | None = None,
    ) -> tuple[CodingContextPackSnapshot, bool]:
        """Publish or replay a pack, returning ``(snapshot, created)``."""

        if session.historical_readonly:
            raise CodingBrainReadOnlyError(
                "historical read-only sessions cannot build Coding Context Packs"
            )
        session_id = session.session_id
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            ledger = self._ledger_for(session_id)
            request_digest = stable_content_sha256(request.to_json())
            existing = ledger.contexts_by_request_id.get(request.request_id)
            if existing is not None:
                if existing.request_digest != request_digest:
                    raise CodingBrainConflictError(
                        f"request_id {request.request_id!r} was reused with a different payload"
                    )
                return existing.snapshot, False

            observed_at_ms = (
                int(time.time() * 1_000)
                if generated_at_ms is None
                else generated_at_ms
            )
            if isinstance(observed_at_ms, bool) or observed_at_ms < 0:
                raise ValueError("generated_at_ms must be a non-negative integer")

            observation = (
                "[coding-brain context request v1]\n"
                f"project_id={request.project_id}\n"
                f"repository_id={request.repository_id}\n"
                f"task_id={request.task_id}\n"
                f"task_kind={request.task_kind.value}\n"
                f"task_summary={request.task_summary}\n"
                f"repository_revision={request.repository_revision}\n"
                f"target_paths={','.join(request.target_paths)}"
            )
            result = await session.run_turn(
                observation,
                trigger_kind=TurnTriggerKind.USER_INPUT,
                environment_provenance="lifeform-domain-coding:context-request.v1",
            )
            pe = _prediction_error_snapshot(result)
            facets = _retrieval_facets(request)
            retrieval = session.retrieve_memory(
                RetrievalQuery(
                    text=_retrieval_query_text(request),
                    track=Track.WORLD,
                    strata=(MemoryStratum.EPISODIC, MemoryStratum.DURABLE),
                    limit=request.memory_limit,
                    facets=facets,
                ),
                timestamp_ms=observed_at_ms,
            )
            rendered, source_entry_ids, truncated = _render_memory(
                retrieval.entries,
                max_chars=request.max_context_chars,
            )
            advice_payload = {
                "session_id": session_id,
                "request_id": request.request_id,
                "source_turn_index": pe.turn_index,
                "candidate_regime_id": result.active_regime or "",
                "candidate_abstract_action": result.active_abstract_action or "",
                "evidence_entry_ids": list(source_entry_ids),
                "wiring_level": "shadow",
                "applied": False,
            }
            advice_digest = stable_content_sha256(advice_payload)
            advice = CodingAdviceSnapshot(
                advice_id=f"coding-advice:{advice_digest}",
                source_turn_index=pe.turn_index,
                candidate_regime_id=result.active_regime or "",
                candidate_abstract_action=result.active_abstract_action or "",
                evidence_entry_ids=source_entry_ids,
                rationale=(
                    "Projection of owner-published regime/action readouts; "
                    "SHADOW only and excluded from rendered_context."
                ),
            )
            settled_refs = tuple(pe.actual_outcome.external_outcome_refs)
            context_payload = _context_digest_payload(
                session_id=session_id,
                request=request,
                generated_at_ms=observed_at_ms,
                source_turn_index=pe.turn_index,
                rendered_context=rendered,
                source_entry_ids=source_entry_ids,
                retrieval_facets=facets,
                memory_entry_count=len(retrieval.entries),
                truncated=truncated,
                settled_outcome_evidence_refs=settled_refs,
                pe_magnitude=float(pe.error.magnitude),
                pe_bootstrap=pe.bootstrap,
                advice=advice,
            )
            context_digest = stable_content_sha256(context_payload)
            snapshot = CodingContextPackSnapshot(
                context_pack_id=f"coding-context-pack:{context_digest}",
                content_sha256=context_digest,
                request=request,
                generated_at_ms=observed_at_ms,
                source_turn_index=pe.turn_index,
                rendered_context=rendered,
                source_entry_ids=source_entry_ids,
                retrieval_facets=facets,
                memory_entry_count=len(retrieval.entries),
                truncated=truncated,
                settled_outcome_evidence_refs=settled_refs,
                pe_magnitude=float(pe.error.magnitude),
                pe_bootstrap=pe.bootstrap,
                advice=advice,
            )
            lineage = _ContextLineage(
                request_digest=request_digest,
                snapshot=snapshot,
            )
            ledger.contexts_by_request_id[request.request_id] = lineage
            ledger.contexts_by_pack_id[snapshot.context_pack_id] = snapshot
            self._trim_contexts(ledger)
            return snapshot, True

    async def record_outcome(
        self,
        *,
        session: LifeformSession,
        report: CodingOutcomeReport,
    ) -> CodingOutcomeReceipt:
        receipt, _created = await self.publish_outcome(
            session=session,
            report=report,
        )
        return receipt

    async def publish_outcome(
        self,
        *,
        session: LifeformSession,
        report: CodingOutcomeReport,
    ) -> tuple[CodingOutcomeReceipt, bool]:
        """Publish or replay an outcome, returning ``(receipt, created)``."""

        if session.historical_readonly:
            raise CodingBrainReadOnlyError(
                "historical read-only sessions cannot record Coding outcomes"
            )
        session_id = session.session_id
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            ledger = self._ledger_for(session_id)
            report_digest = stable_content_sha256(report.to_json())
            existing = ledger.outcomes_by_outcome_id.get(report.outcome_id)
            if existing is not None:
                if existing.report_digest != report_digest:
                    raise CodingBrainConflictError(
                        f"outcome_id {report.outcome_id!r} was reused with a different payload"
                    )
                return existing.receipt, False

            context_pack = ledger.contexts_by_pack_id.get(report.context_pack_id)
            if context_pack is None:
                raise CodingBrainLineageError(
                    "outcome must reference a Context Pack issued for this live session"
                )
            request = context_pack.request
            task_event_seed = _short_hash(f"{session_id}:{report.outcome_id}")
            task_event_ids = session.submit_task_event(
                event_id=f"coding-outcome:{task_event_seed}:task",
                task_id=request.task_id,
                status=_TASK_STATUS[report.kind],
                summary=report.summary,
                detail=report.detail,
                confidence=1.0,
            )

            external_evidence_id = ""
            learning_route = CodingOutcomeRoute.EXECUTION_RESULT
            if report.deterministic_environment_outcome:
                external_kind = (
                    DialogueExternalOutcomeKind.TASK_VERIFIED
                    if report.kind is CodingOutcomeKind.TASK_VERIFIED
                    else DialogueExternalOutcomeKind.TASK_REGRESSED
                )
                evidence = session.submit_dialogue_outcome(
                    kind=external_kind,
                    source=DialogueExternalOutcomeEvidenceSource.ENVIRONMENT,
                    confidence=1.0,
                    evidence_ref=report.evidence_ref,
                    description=(
                        f"Coding outcome {report.kind.value} from "
                        f"{report.source.value}: {report.summary}. {report.detail}"
                    )[:8_000],
                    action_turn_index=context_pack.source_turn_index,
                )
                external_evidence_id = evidence.evidence_id
                learning_route = CodingOutcomeRoute.DIALOGUE_EXTERNAL_OUTCOME

            tags = (
                *_retrieval_facets(request),
                f"outcome:{report.kind.value}",
                f"source:{report.source.value}",
                *(f"path:{_short_hash(path)}" for path in report.changed_paths),
            )
            memory_entry = session.write_memory(
                MemoryWriteRequest(
                    content=_outcome_memory_content(request=request, report=report),
                    track=Track.WORLD,
                    stratum=MemoryStratum.EPISODIC,
                    tags=tuple(dict.fromkeys(tags)),
                    strength=0.9 if report.kind in _NEGATIVE_OUTCOMES else 0.65,
                ),
                timestamp_ms=report.observed_at_ms,
            )
            memory_persisted = session.persist_memory()
            receipt_payload = {
                "schema_version": "coding-outcome-receipt.v1",
                "session_id": session_id,
                "project_id": request.project_id,
                "repository_id": request.repository_id,
                "task_id": request.task_id,
                "report": report.to_json(),
                "action_turn_index": context_pack.source_turn_index,
                "memory_entry_id": memory_entry.entry_id,
                "memory_persisted": memory_persisted,
                "task_event_ids": list(task_event_ids),
                "external_outcome_evidence_id": external_evidence_id,
                "learning_route": learning_route.value,
                "settlement_state": "pending_next_context_turn",
            }
            receipt_digest = stable_content_sha256(receipt_payload)
            receipt = CodingOutcomeReceipt(
                receipt_id=f"coding-outcome-receipt:{receipt_digest}",
                content_sha256=receipt_digest,
                session_id=session_id,
                project_id=request.project_id,
                repository_id=request.repository_id,
                task_id=request.task_id,
                report=report,
                action_turn_index=context_pack.source_turn_index,
                memory_entry_id=memory_entry.entry_id,
                memory_persisted=memory_persisted,
                task_event_ids=task_event_ids,
                external_outcome_evidence_id=external_evidence_id,
                learning_route=learning_route,
            )
            ledger.outcomes_by_outcome_id[report.outcome_id] = _OutcomeLineage(
                report_digest=report_digest,
                receipt=receipt,
            )
            self._trim_outcomes(ledger)
            return receipt, True

    def drop_session(self, session_id: str) -> None:
        """Discard bounded product lineage for a closed live session."""

        self._ledgers.pop(session_id, None)
        self._locks.pop(session_id, None)

    def _trim_contexts(self, ledger: _SessionLedger) -> None:
        while len(ledger.contexts_by_request_id) > self._max_records_per_session:
            oldest_request_id = next(iter(ledger.contexts_by_request_id))
            removed = ledger.contexts_by_request_id.pop(oldest_request_id)
            ledger.contexts_by_pack_id.pop(removed.snapshot.context_pack_id, None)

    def _ledger_for(self, session_id: str) -> _SessionLedger:
        ledger = self._ledgers.get(session_id)
        if ledger is not None:
            return ledger
        ledger = _SessionLedger()
        self._ledgers[session_id] = ledger
        while len(self._ledgers) > self._max_session_ledgers:
            removed = False
            for candidate in tuple(self._ledgers):
                if candidate == session_id:
                    continue
                candidate_lock = self._locks.get(candidate)
                if candidate_lock is not None and candidate_lock.locked():
                    continue
                self._ledgers.pop(candidate, None)
                self._locks.pop(candidate, None)
                removed = True
                break
            if not removed:
                break
        return ledger

    def _trim_outcomes(self, ledger: _SessionLedger) -> None:
        while len(ledger.outcomes_by_outcome_id) > self._max_records_per_session:
            oldest_outcome_id = next(iter(ledger.outcomes_by_outcome_id))
            ledger.outcomes_by_outcome_id.pop(oldest_outcome_id)


__all__ = (
    "CodingBrainConflictError",
    "CodingBrainController",
    "CodingBrainError",
    "CodingBrainLineageError",
    "CodingBrainReadOnlyError",
)
