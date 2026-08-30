"""Memory-first Coding Brain product controller.

The controller owns only bounded product lineage and idempotency. Cognitive
state stays in the existing Memory, semantic-state, Prediction Error, credit,
and temporal owners and is reached exclusively through ``LifeformSession``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field

from lifeform_core import (
    BoundedContentCandidate,
    BoundedContentPolicy,
    BoundedContentPolicyCheckpoint,
    BoundedContentPolicyCredit,
    BoundedContentPolicyDecision,
    BoundedContentPolicyUpdateReceipt,
    CONTENT_POLICY_NOOP_CANDIDATE_ID,
    LifeformSession,
    TurnTriggerKind,
    default_bounded_content_policy_checkpoint,
)
from volvence_zero.credit import CreditRecord, CreditSnapshot
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
from volvence_zero.runtime import WiringLevel

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


class CodingBrainSettlementPendingError(CodingBrainError):
    """A policy-bearing deterministic outcome is awaiting next-turn settlement."""


class CodingBrainMemoryContractError(CodingBrainError):
    """Persisted content-policy state violates the Coding memory contract."""


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
    latest_context_pack_id: str = ""
    pending_external_outcome_evidence_id: str = ""
    policy_checkpoints_by_scope: dict[str, BoundedContentPolicyCheckpoint] = field(
        default_factory=dict
    )


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

_CONTENT_POLICY_FEATURE_ORDER = (
    "owner_rank",
    "memory_strength",
    "recency",
    "durable",
    "pe_magnitude",
)
_CONTENT_POLICY_ARTIFACT_ID = "coding-content-position-policy.v1"
_PE_CREDIT_SOURCES = ("pe:task", "pe:relationship", "pe:regime", "pe:action")


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


def _credit_snapshot(turn_result: object) -> CreditSnapshot:
    published = turn_result.active_snapshots.get("credit")
    if published is None:
        published = turn_result.shadow_snapshots.get("credit")
    if published is None:
        raise RuntimeError("Coding Brain turn published no credit snapshot")
    if not isinstance(published.value, CreditSnapshot):
        raise TypeError("credit slot must publish CreditSnapshot")
    return published.value


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _content_policy_scope(request: CodingContextRequest) -> str:
    return f"{request.project_id}\x1f{request.repository_id}"


def _content_policy_scope_facets(request: CodingContextRequest) -> tuple[str, ...]:
    return (
        f"project:{_short_hash(request.project_id)}",
        f"repository:{_short_hash(request.repository_id)}",
    )


def _content_policy_candidates(
    entries: tuple[MemoryEntry, ...],
    *,
    observed_at_ms: int,
    pe_magnitude: float,
) -> tuple[BoundedContentCandidate, ...]:
    candidates: list[BoundedContentCandidate] = []
    for index, entry in enumerate(entries[1:], start=2):
        age_days = max(0.0, observed_at_ms - entry.created_at_ms) / 86_400_000.0
        values = (
            1.0 / float(index),
            _clamp(entry.strength),
            1.0 / (1.0 + age_days),
            1.0 if entry.stratum == MemoryStratum.DURABLE.value else 0.0,
            _clamp(pe_magnitude),
        )
        candidates.append(
            BoundedContentCandidate(
                entry_id=entry.entry_id,
                feature_values=tuple(
                    zip(_CONTENT_POLICY_FEATURE_ORDER, values, strict=True)
                ),
            )
        )
    return tuple(candidates)


def _ordered_entries(
    entries: tuple[MemoryEntry, ...],
    decision: BoundedContentPolicyDecision | None,
) -> tuple[MemoryEntry, ...]:
    if decision is None:
        return entries
    by_id = {entry.entry_id: entry for entry in entries}
    if set(by_id) != set(decision.output_entry_ids):
        raise CodingBrainLineageError("content policy output diverged from retrieval")
    return tuple(by_id[entry_id] for entry_id in decision.output_entry_ids)


def _settled_content_policy_credit(
    *,
    prediction_error: PredictionErrorSnapshot,
    credit_snapshot: CreditSnapshot,
    decision: BoundedContentPolicyDecision,
    external_outcome_evidence_id: str,
) -> BoundedContentPolicyCredit:
    if prediction_error.bootstrap:
        raise ValueError("bootstrap PE cannot settle Coding content policy")
    if external_outcome_evidence_id not in (
        prediction_error.actual_outcome.external_outcome_refs
    ):
        raise ValueError("Coding policy PE external-outcome lineage mismatch")
    context = prediction_error.actual_outcome.action_context
    evaluated = prediction_error.evaluated_prediction
    if (
        evaluated is None
        or evaluated.prediction_id != decision.source_prediction_id
    ):
        raise ValueError("Coding policy PE did not settle its source prediction")
    if context.prediction_id and context.prediction_id != decision.source_prediction_id:
        raise ValueError("Coding policy PE action-context prediction diverged")
    matching: tuple[CreditRecord, ...] = (
        credit_snapshot.recent_prediction_error_credits
    )
    by_source = {record.source_event: record for record in matching}
    if len(matching) != 4 or set(by_source) != set(_PE_CREDIT_SOURCES):
        raise ValueError("Coding policy requires exactly four owner PE credits")
    if any(
        record.prediction_id
        and record.prediction_id != decision.source_prediction_id
        for record in matching
    ):
        raise ValueError("Coding policy credit prediction lineage diverged")
    if any(
        record.environment_outcome_id != context.environment_outcome_id
        for record in matching
    ):
        raise ValueError("Coding policy credit outcome lineage diverged")
    ordered = tuple(by_source[source] for source in _PE_CREDIT_SOURCES)
    return BoundedContentPolicyCredit.create(
        policy_decision_id=decision.policy_decision_id,
        credited_candidate_id=(
            decision.selected_entry_id
            if decision.intervened
            else CONTENT_POLICY_NOOP_CANDIDATE_ID
        ),
        prediction_id=decision.source_prediction_id,
        settlement_ref=external_outcome_evidence_id,
        signed_prediction_error=_clamp(
            prediction_error.error.signed_reward,
            -1.0,
            1.0,
        ),
        source_credit_record_ids=tuple(item.record_id for item in ordered),
        observed_at_ms=max(item.timestamp_ms for item in ordered),
    )


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
    content_policy_decision: BoundedContentPolicyDecision | None,
    settled_policy_credits: tuple[BoundedContentPolicyCredit, ...],
    policy_updates: tuple[BoundedContentPolicyUpdateReceipt, ...],
    content_policy_wiring_level: WiringLevel,
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
        "content_policy_decision": (
            content_policy_decision.to_json()
            if content_policy_decision is not None
            else None
        ),
        "settled_policy_credits": [
            item.to_json() for item in settled_policy_credits
        ],
        "policy_updates": [item.to_json() for item in policy_updates],
        "content_policy_wiring_level": content_policy_wiring_level.value,
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
        content_policy: BoundedContentPolicy | None = None,
        content_policy_wiring_level: WiringLevel = WiringLevel.ACTIVE,
        max_records_per_session: int = 512,
        max_session_ledgers: int = 1_024,
    ) -> None:
        if isinstance(max_records_per_session, bool) or max_records_per_session < 1:
            raise ValueError("max_records_per_session must be positive")
        if isinstance(max_session_ledgers, bool) or max_session_ledgers < 1:
            raise ValueError("max_session_ledgers must be positive")
        if content_policy_wiring_level not in {
            WiringLevel.ACTIVE,
            WiringLevel.DISABLED,
        }:
            raise ValueError("content_policy_wiring_level must be ACTIVE or DISABLED")
        self._max_records_per_session = max_records_per_session
        self._max_session_ledgers = max_session_ledgers
        self._content_policy = content_policy or BoundedContentPolicy()
        self._content_policy_wiring_level = content_policy_wiring_level
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

            policy_scope = _content_policy_scope(request)
            if (
                self._content_policy_wiring_level is WiringLevel.ACTIVE
                and policy_scope not in ledger.policy_checkpoints_by_scope
            ):
                ledger.policy_checkpoints_by_scope[policy_scope] = (
                    self._restore_policy_checkpoint(
                        session=session,
                        request=request,
                        timestamp_ms=observed_at_ms,
                    )
                )

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
            settled_policy_credits: tuple[BoundedContentPolicyCredit, ...] = ()
            policy_updates: tuple[BoundedContentPolicyUpdateReceipt, ...] = ()
            pending_evidence_id = ledger.pending_external_outcome_evidence_id
            if pending_evidence_id:
                if self._content_policy_wiring_level is WiringLevel.DISABLED:
                    raise RuntimeError(
                        "Coding content policy was disabled with settlement pending"
                    )
                if pending_evidence_id not in pe.actual_outcome.external_outcome_refs:
                    raise RuntimeError(
                        "pending Coding deterministic outcome was not settled by the next context turn"
                    )
                settled_policy_credits, policy_updates = self._settle_policy_update(
                    session=session,
                    ledger=ledger,
                    result=result,
                    prediction_error=pe,
                    external_outcome_evidence_id=pending_evidence_id,
                    timestamp_ms=observed_at_ms,
                )
                ledger.pending_external_outcome_evidence_id = ""
            facets = _retrieval_facets(request)
            retrieval = session.retrieve_memory(
                RetrievalQuery(
                    text=_retrieval_query_text(request),
                    track=Track.WORLD,
                    strata=(MemoryStratum.EPISODIC, MemoryStratum.DURABLE),
                    limit=min(80, request.memory_limit * 4),
                    facets=facets,
                ),
                timestamp_ms=observed_at_ms,
            )
            coding_entries = tuple(
                entry for entry in retrieval.entries if "coding-brain" in entry.tags
            )[: request.memory_limit]
            content_policy_decision: BoundedContentPolicyDecision | None = None
            if self._content_policy_wiring_level is WiringLevel.ACTIVE:
                checkpoint = ledger.policy_checkpoints_by_scope[policy_scope]
                content_policy_decision = self._content_policy.decide(
                    owner_order=tuple(entry.entry_id for entry in coding_entries),
                    challengers=_content_policy_candidates(
                        coding_entries,
                        observed_at_ms=observed_at_ms,
                        pe_magnitude=float(pe.error.magnitude),
                    ),
                    source_prediction_id=pe.next_prediction.prediction_id,
                    checkpoint=checkpoint,
                )
            positioned_entries = _ordered_entries(
                coding_entries,
                content_policy_decision,
            )
            rendered, source_entry_ids, truncated = _render_memory(
                positioned_entries,
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
                memory_entry_count=len(coding_entries),
                truncated=truncated,
                settled_outcome_evidence_refs=settled_refs,
                pe_magnitude=float(pe.error.magnitude),
                pe_bootstrap=pe.bootstrap,
                advice=advice,
                content_policy_decision=content_policy_decision,
                settled_policy_credits=settled_policy_credits,
                policy_updates=policy_updates,
                content_policy_wiring_level=self._content_policy_wiring_level,
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
                memory_entry_count=len(coding_entries),
                truncated=truncated,
                settled_outcome_evidence_refs=settled_refs,
                pe_magnitude=float(pe.error.magnitude),
                pe_bootstrap=pe.bootstrap,
                advice=advice,
                content_policy_decision=content_policy_decision,
                settled_policy_credits=settled_policy_credits,
                policy_updates=policy_updates,
                content_policy_wiring_level=self._content_policy_wiring_level,
            )
            lineage = _ContextLineage(
                request_digest=request_digest,
                snapshot=snapshot,
            )
            ledger.contexts_by_request_id[request.request_id] = lineage
            ledger.contexts_by_pack_id[snapshot.context_pack_id] = snapshot
            ledger.latest_context_pack_id = snapshot.context_pack_id
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
                if context_pack.content_policy_decision is not None:
                    if report.context_pack_id != ledger.latest_context_pack_id:
                        raise CodingBrainLineageError(
                            "policy-bearing deterministic outcomes must reference the latest Context Pack"
                        )
                    if ledger.pending_external_outcome_evidence_id:
                        raise CodingBrainSettlementPendingError(
                            "a deterministic Coding outcome is already pending the next Context Pack turn"
                        )
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
                "source_content_policy_decision_id": (
                    context_pack.content_policy_decision.policy_decision_id
                    if context_pack.content_policy_decision is not None
                    else ""
                ),
                "content_policy_action_applied": (
                    context_pack.content_policy_decision is not None
                ),
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
                source_content_policy_decision_id=(
                    context_pack.content_policy_decision.policy_decision_id
                    if context_pack.content_policy_decision is not None
                    else ""
                ),
                content_policy_action_applied=(
                    context_pack.content_policy_decision is not None
                ),
            )
            ledger.outcomes_by_outcome_id[report.outcome_id] = _OutcomeLineage(
                report_digest=report_digest,
                receipt=receipt,
            )
            self._trim_outcomes(ledger)
            if (
                report.deterministic_environment_outcome
                and context_pack.content_policy_decision is not None
            ):
                ledger.pending_external_outcome_evidence_id = external_evidence_id
            return receipt, True

    def _restore_policy_checkpoint(
        self,
        *,
        session: LifeformSession,
        request: CodingContextRequest,
        timestamp_ms: int,
    ) -> BoundedContentPolicyCheckpoint:
        scope_facets = _content_policy_scope_facets(request)
        retrieval = session.retrieve_memory(
            RetrievalQuery(
                text="coding content position policy checkpoint",
                track=Track.WORLD,
                strata=(MemoryStratum.EPISODIC, MemoryStratum.DURABLE),
                limit=80,
                facets=("coding-content-policy-checkpoint", *scope_facets),
            ),
            timestamp_ms=timestamp_ms,
        )
        checkpoints: list[BoundedContentPolicyCheckpoint] = []
        for entry in retrieval.entries:
            if "coding-content-policy-checkpoint" not in entry.tags or any(
                facet not in entry.tags for facet in scope_facets
            ):
                continue
            try:
                payload = json.loads(entry.content)
            except json.JSONDecodeError as exc:
                raise CodingBrainMemoryContractError(
                    f"content policy checkpoint {entry.entry_id!r} is not JSON"
                ) from exc
            if not isinstance(payload, dict) or set(payload) != {
                "schema_version",
                "project_id",
                "repository_id",
                "checkpoint",
            }:
                raise CodingBrainMemoryContractError(
                    f"content policy checkpoint {entry.entry_id!r} has invalid shape"
                )
            if payload["schema_version"] != "coding-content-policy-memory-record.v1":
                raise CodingBrainMemoryContractError(
                    f"content policy checkpoint {entry.entry_id!r} has unsupported schema"
                )
            if (
                payload["project_id"] != request.project_id
                or payload["repository_id"] != request.repository_id
            ):
                raise CodingBrainMemoryContractError(
                    "Coding content policy scope does not match retrieval facets"
                )
            checkpoint_payload = payload["checkpoint"]
            if not isinstance(checkpoint_payload, dict):
                raise CodingBrainMemoryContractError(
                    f"content policy checkpoint {entry.entry_id!r} payload is invalid"
                )
            try:
                checkpoint = BoundedContentPolicyCheckpoint.from_json(
                    checkpoint_payload
                )
            except (TypeError, ValueError) as exc:
                raise CodingBrainMemoryContractError(
                    f"content policy checkpoint {entry.entry_id!r} violates contract: {exc}"
                ) from exc
            if checkpoint.artifact_id != _CONTENT_POLICY_ARTIFACT_ID:
                raise CodingBrainMemoryContractError(
                    "Coding content policy artifact id mismatch"
                )
            checkpoints.append(checkpoint)
        if not checkpoints:
            checkpoint = default_bounded_content_policy_checkpoint(
                artifact_id=_CONTENT_POLICY_ARTIFACT_ID,
                feature_order=_CONTENT_POLICY_FEATURE_ORDER,
            )
            self._persist_policy_checkpoint(
                session=session,
                request=request,
                checkpoint=checkpoint,
                timestamp_ms=timestamp_ms,
            )
            return checkpoint
        maximum_update_count = max(item.update_count for item in checkpoints)
        latest = tuple(
            item for item in checkpoints if item.update_count == maximum_update_count
        )
        if len({item.checkpoint_id for item in latest}) != 1:
            raise CodingBrainMemoryContractError(
                "Coding content policy has divergent checkpoints at one update count"
            )
        return latest[0]

    @staticmethod
    def _persist_policy_checkpoint(
        *,
        session: LifeformSession,
        request: CodingContextRequest,
        checkpoint: BoundedContentPolicyCheckpoint,
        timestamp_ms: int,
    ) -> None:
        session.write_memory(
            MemoryWriteRequest(
                content=json.dumps(
                    {
                        "schema_version": "coding-content-policy-memory-record.v1",
                        "project_id": request.project_id,
                        "repository_id": request.repository_id,
                        "checkpoint": checkpoint.to_json(),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                track=Track.WORLD,
                stratum=MemoryStratum.EPISODIC,
                tags=(
                    "coding-content-policy-checkpoint",
                    *_content_policy_scope_facets(request),
                    f"policy-artifact:{checkpoint.artifact_id}",
                    f"policy-update-count:{checkpoint.update_count}",
                ),
                strength=0.95,
            ),
            timestamp_ms=timestamp_ms,
        )
        session.persist_memory()

    def _settle_policy_update(
        self,
        *,
        session: LifeformSession,
        ledger: _SessionLedger,
        result: object,
        prediction_error: PredictionErrorSnapshot,
        external_outcome_evidence_id: str,
        timestamp_ms: int,
    ) -> tuple[
        tuple[BoundedContentPolicyCredit, ...],
        tuple[BoundedContentPolicyUpdateReceipt, ...],
    ]:
        lineage = next(
            (
                item
                for item in ledger.outcomes_by_outcome_id.values()
                if item.receipt.external_outcome_evidence_id
                == external_outcome_evidence_id
            ),
            None,
        )
        if lineage is None:
            raise CodingBrainLineageError(
                "settled deterministic outcome has no Coding report lineage"
            )
        context_pack = ledger.contexts_by_pack_id.get(
            lineage.receipt.report.context_pack_id
        )
        if context_pack is None:
            raise CodingBrainLineageError(
                "policy settlement Context Pack is absent from the live ledger"
            )
        decision = context_pack.content_policy_decision
        if decision is None:
            raise CodingBrainLineageError(
                "pending policy outcome references Context Pack without a decision"
            )
        if (
            lineage.receipt.source_content_policy_decision_id
            != decision.policy_decision_id
            or not lineage.receipt.content_policy_action_applied
        ):
            raise CodingBrainLineageError(
                "Coding outcome content policy application lineage mismatch"
            )
        scope = _content_policy_scope(context_pack.request)
        checkpoint = ledger.policy_checkpoints_by_scope.get(scope)
        if checkpoint is None:
            raise RuntimeError("Coding content policy checkpoint is unavailable")
        if checkpoint.checkpoint_id != decision.checkpoint_id:
            raise CodingBrainLineageError(
                "Coding content policy checkpoint advanced outside exact settlement"
            )
        credit = _settled_content_policy_credit(
            prediction_error=prediction_error,
            credit_snapshot=_credit_snapshot(result),
            decision=decision,
            external_outcome_evidence_id=external_outcome_evidence_id,
        )
        next_checkpoint, update = self._content_policy.observe_credit(
            checkpoint=checkpoint,
            decision=decision,
            credit=credit,
        )
        ledger.policy_checkpoints_by_scope[scope] = next_checkpoint
        self._persist_policy_checkpoint(
            session=session,
            request=context_pack.request,
            checkpoint=next_checkpoint,
            timestamp_ms=timestamp_ms,
        )
        return (credit,), (update,)

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
