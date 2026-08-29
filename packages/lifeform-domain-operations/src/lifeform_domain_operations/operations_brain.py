"""Memory-first Operations Brain product controller for AutoCompany.

The controller owns bounded live-session idempotency and product lineage only.
It reaches cognitive state exclusively through ``LifeformSession`` facades and
never reads AutoCompany ledgers or Volvence owner stores directly.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
import weakref
from dataclasses import dataclass, field
from typing import Protocol

from lifeform_core import LifeformSession, TurnTriggerKind
from volvence_zero.environment import (
    EnvironmentEventKind,
    EnvironmentMeasurement,
    EnvironmentOutcome,
)
from volvence_zero.memory import (
    MemoryEntry,
    MemoryStratum,
    MemoryWriteRequest,
    RetrievalQuery,
    Track,
)
from volvence_zero.prediction import PredictionErrorSnapshot

from lifeform_domain_operations.operations_brain_contracts import (
    EXPERIENCE_RECORD_SCHEMA_VERSION,
    OperationsAdviceCandidate,
    OperationsAdviceSnapshot,
    OperationsContextPackSnapshot,
    OperationsContextRequest,
    OperationsOutcomeReceipt,
    OperationsOutcomeReport,
    OperationsOutcomeRoute,
    OperationsOutcomeVerdict,
    OperationsRecalledExperience,
    OperationsReversibility,
    OperationsRiskLevel,
    OperationsSettlementState,
    stable_content_sha256,
)


class OperationsBrainError(RuntimeError):
    """Base class for Operations Brain controller failures."""


class OperationsBrainConflictError(OperationsBrainError):
    """An idempotency key was reused with a different immutable payload."""


class OperationsBrainLineageError(OperationsBrainError):
    """An outcome references unknown, stale, or cross-session lineage."""


class OperationsBrainReadOnlyError(OperationsBrainError):
    """A mutating operation targeted a historical read-only session."""


class OperationsBrainSettlementPendingError(OperationsBrainError):
    """A PE-eligible outcome is already waiting for the next context turn."""


class OperationsBrainMemoryContractError(OperationsBrainError):
    """A operations-tagged memory entry violates the Operations record contract."""


class OperationsAdviceProvider(Protocol):
    """Structured SHADOW-only proposal seam.

    Providers return frozen candidates. The controller revalidates every
    evidence and memory reference against the current Context Pack before it
    publishes them. A provider never receives a mutation or actuator handle.
    """

    async def propose(
        self,
        *,
        request: OperationsContextRequest,
        recalled_experiences: tuple[OperationsRecalledExperience, ...],
        source_turn_index: int,
        candidate_regime_id: str,
        candidate_abstract_action: str,
    ) -> tuple[OperationsAdviceCandidate, ...]: ...


class EmptyOperationsAdviceProvider:
    """Explicit no-candidate provider used when no qualified advisor is wired."""

    async def propose(
        self,
        *,
        request: OperationsContextRequest,
        recalled_experiences: tuple[OperationsRecalledExperience, ...],
        source_turn_index: int,
        candidate_regime_id: str,
        candidate_abstract_action: str,
    ) -> tuple[OperationsAdviceCandidate, ...]:
        del (
            request,
            recalled_experiences,
            source_turn_index,
            candidate_regime_id,
            candidate_abstract_action,
        )
        return ()


@dataclass(frozen=True)
class _ContextLineage:
    request_digest: str
    snapshot: OperationsContextPackSnapshot


@dataclass(frozen=True)
class _OutcomeLineage:
    report_digest: str
    receipt: OperationsOutcomeReceipt


@dataclass
class _SessionLedger:
    session_id: str
    session_lineage_id: str
    contexts_by_request_id: dict[str, _ContextLineage] = field(default_factory=dict)
    contexts_by_pack_id: dict[str, OperationsContextPackSnapshot] = field(default_factory=dict)
    outcomes_by_outcome_id: dict[str, _OutcomeLineage] = field(default_factory=dict)
    latest_context_pack_id: str = ""
    pending_environment_outcome_id: str = ""


_VERDICT_MEASUREMENT = {
    OperationsOutcomeVerdict.FAVORABLE: 1.0,
    OperationsOutcomeVerdict.UNFAVORABLE: -1.0,
    OperationsOutcomeVerdict.MIXED: 0.0,
    OperationsOutcomeVerdict.INCONCLUSIVE: 0.0,
}


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _retrieval_facets(request: OperationsContextRequest) -> tuple[str, ...]:
    facets = [
        "operations-brain",
        f"company:{_short_hash(request.company_id)}",
        f"decision-point:{request.decision_point.value}",
    ]
    if request.workstream_id:
        facets.append(f"workstream:{_short_hash(request.workstream_id)}")
    facets.extend(f"division:{_short_hash(division_id)}" for division_id in request.division_ids)
    facets.extend(f"fact-kind:{fact.kind.value}" for fact in request.confirmed_facts)
    return tuple(dict.fromkeys(facets))


def _retrieval_query_text(request: OperationsContextRequest) -> str:
    parts = [
        f"decision point {request.decision_point.value}",
        *(fact.statement for fact in request.confirmed_facts),
        *(constraint.description for constraint in request.constraints),
        *(uncertainty.statement for uncertainty in request.uncertainties),
    ]
    return "\n".join(parts)[:16_000]


def _context_observation(request: OperationsContextRequest) -> str:
    return (
        "[operations-brain structured context request v1]\n"
        f"company_id={request.company_id}\n"
        f"cycle_id={request.cycle_id}\n"
        f"workstream_id={request.workstream_id}\n"
        f"decision_id={request.decision_id}\n"
        f"decision_point={request.decision_point.value}\n"
        f"division_ids={','.join(request.division_ids)}\n"
        f"action_catalog_ids={','.join(request.action_catalog_ids)}\n"
        f"explicit_context={_retrieval_query_text(request)}"
    )


def _prediction_error_snapshot(turn_result: object) -> PredictionErrorSnapshot:
    published = turn_result.active_snapshots.get("prediction_error")
    if published is None:
        raise RuntimeError("Operations Brain turn published no prediction_error snapshot")
    if not isinstance(published.value, PredictionErrorSnapshot):
        raise TypeError("prediction_error slot must publish PredictionErrorSnapshot")
    return published.value


def _experience_record_payload(
    *,
    request: OperationsContextRequest,
    report: OperationsOutcomeReport,
) -> dict[str, object]:
    return {
        "schema_version": EXPERIENCE_RECORD_SCHEMA_VERSION,
        "company_id": request.company_id,
        "cycle_id": request.cycle_id,
        "workstream_id": request.workstream_id,
        "decision_id": request.decision_id,
        "source_context_pack_id": report.context_pack_id,
        "report": report.to_json(),
    }


def _experience_from_memory(entry: MemoryEntry) -> OperationsRecalledExperience:
    try:
        payload = json.loads(entry.content)
    except json.JSONDecodeError as exc:
        raise OperationsBrainMemoryContractError(
            f"operations memory entry {entry.entry_id!r} is not valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise OperationsBrainMemoryContractError(f"operations memory entry {entry.entry_id!r} must be a JSON object")
    expected = {
        "schema_version",
        "company_id",
        "cycle_id",
        "workstream_id",
        "decision_id",
        "source_context_pack_id",
        "report",
    }
    unknown = set(payload) - expected
    missing = expected - set(payload)
    if unknown or missing:
        raise OperationsBrainMemoryContractError(
            f"operations memory entry {entry.entry_id!r} has unknown={sorted(unknown)!r} missing={sorted(missing)!r}"
        )
    if payload["schema_version"] != EXPERIENCE_RECORD_SCHEMA_VERSION:
        raise OperationsBrainMemoryContractError(
            f"operations memory entry {entry.entry_id!r} has unsupported schema_version"
        )
    report_payload = payload["report"]
    if not isinstance(report_payload, dict):
        raise OperationsBrainMemoryContractError(f"operations memory entry {entry.entry_id!r} report must be an object")
    try:
        report = OperationsOutcomeReport.from_json(report_payload)
        return OperationsRecalledExperience(
            memory_entry_id=entry.entry_id,
            company_id=payload["company_id"],
            cycle_id=payload["cycle_id"],
            workstream_id=payload["workstream_id"],
            decision_id=payload["decision_id"],
            source_context_pack_id=payload["source_context_pack_id"],
            created_at_ms=entry.created_at_ms,
            report=report,
        )
    except (TypeError, ValueError) as exc:
        raise OperationsBrainMemoryContractError(
            f"operations memory entry {entry.entry_id!r} violates the record contract: {exc}"
        ) from exc


def _render_context(
    *,
    request: OperationsContextRequest,
    experiences: tuple[OperationsRecalledExperience, ...],
) -> tuple[str, tuple[OperationsRecalledExperience, ...], bool]:
    uncertainty_lines = (
        "\n".join(
            f"- {item.uncertainty_id}: p=[{item.probability_lower:.3f},{item.probability_upper:.3f}] {item.statement}"
            for item in request.uncertainties
        )
        if request.uncertainties
        else "- none declared by AutoCompany"
    )
    rendered = (
        "[operations-brain active context v1]\n"
        "Evidence class and outcome route below are AutoCompany-declared typed facts.\n"
        "Current uncertainty:\n"
        f"{uncertainty_lines}\n\n"
        "Cross-cycle recalled outcomes:"
    )
    included: list[OperationsRecalledExperience] = []
    truncated = False
    if not experiences:
        rendered += "\n- no operations-tagged episodic or durable outcome matched"
        if len(rendered) > request.max_context_chars:
            return rendered[: request.max_context_chars], (), True
        return rendered, (), False

    for index, experience in enumerate(experiences, start=1):
        report = experience.report
        execution = report.execution_outcome
        metric_summary = ",".join(
            f"{metric.metric_id}[{metric.unit}]:{metric.baseline_value:g}->{metric.observed_value:g}"
            for metric in execution.metrics
        ) or "none"
        block = (
            f"\n\n[{index}] memory_entry_id={experience.memory_entry_id}\n"
            f"work_order_ref={report.work_order_ref}\n"
            f"evidence_class={report.evidence_class.value}\n"
            f"outcome_kind={report.outcome_kind.value}\n"
            f"decision={report.decision.value}\n"
            f"verdict={report.verdict.value}\n"
            f"summary={report.summary}\n"
            f"detail={report.detail}\n"
            f"objective_result={execution.objective_result.value}\n"
            f"metrics={metric_summary}\n"
            f"realized_costs_minor={execution.realized_costs.total_minor}\n"
            f"currency={execution.currency}\n"
            f"elapsed_ms={execution.elapsed_ms}\n"
            f"blocker_duration_ms={execution.blocker_duration_ms}\n"
            f"rework_count={execution.rework_count}\n"
            f"incident_count={execution.incident_count}\n"
            f"human_minutes={execution.human_minutes}\n"
            f"risk_level={execution.risk_level.value}\n"
            f"reversibility={execution.reversibility.value}\n"
            "evidence_ref_ids=" + ",".join(reference.ref_id for reference in report.evidence_refs)
        )
        remaining = request.max_context_chars - len(rendered)
        if remaining <= 0:
            truncated = True
            break
        if len(block) <= remaining:
            rendered += block
            included.append(experience)
            continue
        rendered += block[:remaining]
        included.append(experience)
        truncated = True
        break
    if len(included) < len(experiences):
        truncated = True
    return rendered, tuple(included), truncated


def _context_digest_payload(
    *,
    session_id: str,
    session_lineage_id: str,
    request: OperationsContextRequest,
    generated_at_ms: int,
    source_turn_index: int,
    rendered_context: str,
    recalled_experiences: tuple[OperationsRecalledExperience, ...],
    source_entry_ids: tuple[str, ...],
    source_evidence_ref_ids: tuple[str, ...],
    retrieval_facets: tuple[str, ...],
    memory_entry_count: int,
    truncated: bool,
    settled_outcome_ids: tuple[str, ...],
    settled_evidence_ref_ids: tuple[str, ...],
    pe_magnitude: float,
    pe_bootstrap: bool,
    advice: OperationsAdviceSnapshot,
) -> dict[str, object]:
    return {
        "schema_version": "operations-context-pack.v1",
        "session_id": session_id,
        "session_lineage_id": session_lineage_id,
        "request": request.to_json(),
        "generated_at_ms": generated_at_ms,
        "source_turn_index": source_turn_index,
        "rendered_context": rendered_context,
        "recalled_experiences": [item.to_json() for item in recalled_experiences],
        "source_entry_ids": list(source_entry_ids),
        "source_evidence_ref_ids": list(source_evidence_ref_ids),
        "retrieval_facets": list(retrieval_facets),
        "memory_entry_count": memory_entry_count,
        "truncated": truncated,
        "current_uncertainties": [item.to_json() for item in request.uncertainties],
        "settled_outcome_ids": list(settled_outcome_ids),
        "settled_evidence_ref_ids": list(settled_evidence_ref_ids),
        "pe_magnitude": pe_magnitude,
        "pe_bootstrap": pe_bootstrap,
        "advice": advice.to_json(),
        "wiring_level": "active",
    }


def _environment_outcome_for_report(
    *,
    request: OperationsContextRequest,
    report: OperationsOutcomeReport,
) -> EnvironmentOutcome:
    normalized = _VERDICT_MEASUREMENT[report.verdict]
    return EnvironmentOutcome(
        outcome_id=f"operations-field-outcome:{_short_hash(report.outcome_id)}",
        event_id=f"operations-report:{_short_hash(report.outcome_id)}",
        outcome_kind=EnvironmentEventKind.SCENE_EVENT,
        action_id=f"autocompany-work-order:{_short_hash(report.work_order_ref)}",
        status=f"autocompany_{report.verdict.value}",
        summary=report.summary,
        detail=report.detail,
        confidence=1.0,
        evidence=tuple(f"autocompany-evidence:{reference.ref_id}" for reference in report.evidence_refs),
        monetary_cost=0.0,
        reversibility=report.execution_outcome.reversibility.value,
        environment_state_delta_kind="autocompany_field_operation_result",
        measurement=EnvironmentMeasurement(
            task_progress=normalized,
            action_payoff=normalized,
            terminal=True,
            discrete_milestone=True,
            unit="autocompany_multiobjective_operation_verdict.v1",
        ),
        situation_summary=(
            f"company={request.company_id}; decision={request.decision_id}; "
            f"work_order={report.work_order_ref}; point={request.decision_point.value}"
        ),
    )


class OperationsBrainController:
    """Context Pack/outcome adapter with bounded non-cognitive lineage."""

    def __init__(
        self,
        *,
        advice_provider: OperationsAdviceProvider | None = None,
        max_records_per_session: int = 512,
        max_session_ledgers: int = 1_024,
    ) -> None:
        if (
            isinstance(max_records_per_session, bool)
            or not isinstance(max_records_per_session, int)
            or max_records_per_session < 1
        ):
            raise ValueError("max_records_per_session must be positive")
        if isinstance(max_session_ledgers, bool) or not isinstance(max_session_ledgers, int) or max_session_ledgers < 1:
            raise ValueError("max_session_ledgers must be positive")
        self._advice_provider = advice_provider or EmptyOperationsAdviceProvider()
        self._max_records_per_session = max_records_per_session
        self._max_session_ledgers = max_session_ledgers
        self._ledgers: dict[str, _SessionLedger] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._session_keys: weakref.WeakKeyDictionary[LifeformSession, str] = (
            weakref.WeakKeyDictionary()
        )

    async def build_context_pack(
        self,
        *,
        session: LifeformSession,
        request: OperationsContextRequest,
        generated_at_ms: int | None = None,
    ) -> OperationsContextPackSnapshot:
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
        request: OperationsContextRequest,
        generated_at_ms: int | None = None,
    ) -> tuple[OperationsContextPackSnapshot, bool]:
        """Publish or replay a Context Pack, returning ``(snapshot, created)``."""

        if session.historical_readonly:
            raise OperationsBrainReadOnlyError("historical read-only sessions cannot build Operations Context Packs")
        session_id = session.session_id
        ledger_key = self._ledger_key_for(session)
        lock = self._locks.setdefault(ledger_key, asyncio.Lock())
        async with lock:
            ledger = self._ledger_for(ledger_key, session_id=session_id)
            request_digest = stable_content_sha256(request.to_json())
            existing = ledger.contexts_by_request_id.get(request.request_id)
            if existing is not None:
                if existing.request_digest != request_digest:
                    raise OperationsBrainConflictError(
                        f"request_id {request.request_id!r} was reused with a different payload"
                    )
                return existing.snapshot, False

            observed_at_ms = int(time.time() * 1_000) if generated_at_ms is None else generated_at_ms
            if isinstance(observed_at_ms, bool) or not isinstance(observed_at_ms, int) or observed_at_ms < 0:
                raise ValueError("generated_at_ms must be a non-negative integer")

            result = await session.run_turn(
                _context_observation(request),
                trigger_kind=TurnTriggerKind.USER_INPUT,
                environment_provenance="lifeform-domain-operations:context-request.v1",
            )
            pe = _prediction_error_snapshot(result)
            environment_outcome_id = pe.actual_outcome.action_context.environment_outcome_id
            settled_outcome_ids = (environment_outcome_id,) if environment_outcome_id else ()
            pending = ledger.pending_environment_outcome_id
            if pending and pending not in settled_outcome_ids:
                raise RuntimeError("pending Operations field outcome was not settled by the next context turn")
            if pending:
                ledger.pending_environment_outcome_id = ""
            settled_evidence_ref_ids = self._settled_evidence_refs(
                ledger=ledger,
                settled_outcome_ids=settled_outcome_ids,
            )

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
            company_facet = f"company:{_short_hash(request.company_id)}"
            operations_entries = tuple(
                entry
                for entry in retrieval.entries
                if "operations-brain" in entry.tags and company_facet in entry.tags
            )[: request.memory_limit]
            experiences = tuple(_experience_from_memory(entry) for entry in operations_entries)
            if any(experience.company_id != request.company_id for experience in experiences):
                raise OperationsBrainMemoryContractError(
                    "operations memory company payload does not match its retrieval facet"
                )
            rendered, included_experiences, truncated = _render_context(
                request=request,
                experiences=experiences,
            )
            source_entry_ids = tuple(experience.memory_entry_id for experience in included_experiences)
            source_evidence_ref_ids = tuple(
                dict.fromkeys(
                    reference.ref_id
                    for experience in included_experiences
                    for reference in experience.report.evidence_refs
                )
            )
            candidates = await self._advice_provider.propose(
                request=request,
                recalled_experiences=included_experiences,
                source_turn_index=pe.turn_index,
                candidate_regime_id=result.active_regime or "",
                candidate_abstract_action=result.active_abstract_action or "",
            )
            if not isinstance(candidates, tuple):
                raise TypeError("OperationsAdviceProvider must return a tuple")
            self._validate_advice_lineage(
                request=request,
                experiences=included_experiences,
                candidates=candidates,
            )
            advice_payload = {
                "session_id": session_id,
                "request_id": request.request_id,
                "source_turn_index": pe.turn_index,
                "candidate_regime_id": result.active_regime or "",
                "candidate_abstract_action": result.active_abstract_action or "",
                "candidates": [candidate.to_json() for candidate in candidates],
                "wiring_level": "shadow",
                "applied": False,
            }
            advice_digest = stable_content_sha256(advice_payload)
            advice = OperationsAdviceSnapshot(
                advice_id=f"operations-advice:{advice_digest}",
                source_turn_index=pe.turn_index,
                candidate_regime_id=result.active_regime or "",
                candidate_abstract_action=result.active_abstract_action or "",
                candidates=candidates,
                rationale=(
                    "Strict structured candidate projection; SHADOW only, "
                    "excluded from rendered_context, and never an AutoCompany decision."
                    if candidates
                    else "No qualified structured advisor candidate was supplied; "
                    "the SHADOW advice set is explicitly empty."
                ),
            )
            context_payload = _context_digest_payload(
                session_id=session_id,
                session_lineage_id=ledger.session_lineage_id,
                request=request,
                generated_at_ms=observed_at_ms,
                source_turn_index=pe.turn_index,
                rendered_context=rendered,
                recalled_experiences=included_experiences,
                source_entry_ids=source_entry_ids,
                source_evidence_ref_ids=source_evidence_ref_ids,
                retrieval_facets=facets,
                memory_entry_count=len(experiences),
                truncated=truncated,
                settled_outcome_ids=settled_outcome_ids,
                settled_evidence_ref_ids=settled_evidence_ref_ids,
                pe_magnitude=float(pe.error.magnitude),
                pe_bootstrap=pe.bootstrap,
                advice=advice,
            )
            context_digest = stable_content_sha256(context_payload)
            snapshot = OperationsContextPackSnapshot(
                context_pack_id=f"operations-context-pack:{context_digest}",
                content_sha256=context_digest,
                session_id=session_id,
                session_lineage_id=ledger.session_lineage_id,
                request=request,
                generated_at_ms=observed_at_ms,
                source_turn_index=pe.turn_index,
                rendered_context=rendered,
                recalled_experiences=included_experiences,
                source_entry_ids=source_entry_ids,
                source_evidence_ref_ids=source_evidence_ref_ids,
                retrieval_facets=facets,
                memory_entry_count=len(experiences),
                truncated=truncated,
                current_uncertainties=request.uncertainties,
                settled_outcome_ids=settled_outcome_ids,
                settled_evidence_ref_ids=settled_evidence_ref_ids,
                pe_magnitude=float(pe.error.magnitude),
                pe_bootstrap=pe.bootstrap,
                advice=advice,
            )
            ledger.contexts_by_request_id[request.request_id] = _ContextLineage(
                request_digest=request_digest,
                snapshot=snapshot,
            )
            ledger.contexts_by_pack_id[snapshot.context_pack_id] = snapshot
            ledger.latest_context_pack_id = snapshot.context_pack_id
            self._trim_contexts(ledger)
            return snapshot, True

    async def record_outcome(
        self,
        *,
        session: LifeformSession,
        report: OperationsOutcomeReport,
    ) -> OperationsOutcomeReceipt:
        receipt, _created = await self.publish_outcome(
            session=session,
            report=report,
        )
        return receipt

    async def publish_outcome(
        self,
        *,
        session: LifeformSession,
        report: OperationsOutcomeReport,
    ) -> tuple[OperationsOutcomeReceipt, bool]:
        """Publish or replay an outcome, returning ``(receipt, created)``."""

        if session.historical_readonly:
            raise OperationsBrainReadOnlyError("historical read-only sessions cannot record Operations outcomes")
        session_id = session.session_id
        ledger_key = self._ledger_key_for(session)
        lock = self._locks.setdefault(ledger_key, asyncio.Lock())
        async with lock:
            ledger = self._ledger_for(ledger_key, session_id=session_id)
            report_digest = stable_content_sha256(report.to_json())
            existing = ledger.outcomes_by_outcome_id.get(report.outcome_id)
            if existing is not None:
                if existing.report_digest != report_digest:
                    raise OperationsBrainConflictError(
                        f"outcome_id {report.outcome_id!r} was reused with a different payload"
                    )
                return existing.receipt, False

            context_pack = ledger.contexts_by_pack_id.get(report.context_pack_id)
            if context_pack is None:
                raise OperationsBrainLineageError(
                    "outcome must reference an Operations Context Pack issued for this live session"
                )
            request = context_pack.request
            if report.decision_id != request.decision_id:
                raise OperationsBrainLineageError("outcome decision_id must match its Operations Context Pack")
            if report.execution_outcome.currency != request.operating_window.currency:
                raise OperationsBrainLineageError("outcome currency must match the Context Pack resource window")
            if report.pe_eligible:
                if report.context_pack_id != ledger.latest_context_pack_id:
                    raise OperationsBrainLineageError(
                        "PE-eligible delayed outcomes must reference the latest live Context Pack"
                    )
                if ledger.pending_environment_outcome_id:
                    raise OperationsBrainSettlementPendingError(
                        "a field_operation_result is already pending the next Context Pack turn"
                    )

            environment_outcome = (
                _environment_outcome_for_report(request=request, report=report) if report.pe_eligible else None
            )
            task_event_seed = _short_hash(f"{session_id}:{report.outcome_id}")
            task_event_ids = session.submit_task_event(
                event_id=f"operations-outcome:{task_event_seed}:record",
                task_id=report.work_order_ref,
                status="completed",
                summary=report.summary,
                detail=report.detail,
                confidence=1.0,
            )
            if environment_outcome is not None:
                session.submit_environment_outcome(environment_outcome)

            tags = (
                *_retrieval_facets(request),
                f"evidence-class:{report.evidence_class.value}",
                f"outcome-kind:{report.outcome_kind.value}",
                f"decision:{report.decision.value}",
                *(f"evidence-role:{reference.role.value}" for reference in report.evidence_refs),
            )
            memory_entry = session.write_memory(
                MemoryWriteRequest(
                    content=json.dumps(
                        _experience_record_payload(request=request, report=report),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    track=Track.WORLD,
                    stratum=MemoryStratum.EPISODIC,
                    tags=tuple(dict.fromkeys(tags)),
                    strength=0.85 if report.evidence_class.value == "field" else 0.55,
                ),
                timestamp_ms=report.observed_at_ms,
            )
            memory_persisted = session.persist_memory()
            environment_outcome_id = environment_outcome.outcome_id if environment_outcome is not None else ""
            route = (
                OperationsOutcomeRoute.FIELD_OPERATION_PE_MEMORY_AND_EXECUTION_RESULT
                if report.pe_eligible
                else OperationsOutcomeRoute.MEMORY_AND_EXECUTION_RESULT
            )
            settlement_state = (
                OperationsSettlementState.PENDING_NEXT_CONTEXT_TURN
                if report.pe_eligible
                else OperationsSettlementState.NOT_PE_ELIGIBLE
            )
            receipt_payload = {
                "schema_version": "operations-outcome-receipt.v1",
                "session_id": session_id,
                "session_lineage_id": ledger.session_lineage_id,
                "company_id": request.company_id,
                "cycle_id": request.cycle_id,
                "workstream_id": request.workstream_id,
                "decision_id": request.decision_id,
                "work_order_ref": report.work_order_ref,
                "report": report.to_json(),
                "action_turn_index": context_pack.source_turn_index,
                "source_advice_id": context_pack.advice.advice_id,
                "source_advice_applied": False,
                "memory_entry_id": memory_entry.entry_id,
                "memory_persisted": memory_persisted,
                "task_event_ids": list(task_event_ids),
                "environment_outcome_id": environment_outcome_id,
                "learning_route": route.value,
                "settlement_state": settlement_state.value,
            }
            receipt_digest = stable_content_sha256(receipt_payload)
            receipt = OperationsOutcomeReceipt(
                receipt_id=f"operations-outcome-receipt:{receipt_digest}",
                content_sha256=receipt_digest,
                session_id=session_id,
                session_lineage_id=ledger.session_lineage_id,
                company_id=request.company_id,
                cycle_id=request.cycle_id,
                workstream_id=request.workstream_id,
                decision_id=request.decision_id,
                work_order_ref=report.work_order_ref,
                report=report,
                action_turn_index=context_pack.source_turn_index,
                source_advice_id=context_pack.advice.advice_id,
                source_advice_applied=False,
                memory_entry_id=memory_entry.entry_id,
                memory_persisted=memory_persisted,
                task_event_ids=task_event_ids,
                environment_outcome_id=environment_outcome_id,
                learning_route=route,
                settlement_state=settlement_state,
            )
            ledger.outcomes_by_outcome_id[report.outcome_id] = _OutcomeLineage(
                report_digest=report_digest,
                receipt=receipt,
            )
            if environment_outcome_id:
                ledger.pending_environment_outcome_id = environment_outcome_id
            self._trim_outcomes(ledger)
            return receipt, True

    def drop_session(self, session: LifeformSession | str) -> None:
        """Discard bounded product lineage for a closed live session."""

        if isinstance(session, LifeformSession):
            ledger_key = self._session_keys.pop(session, None)
            keys = (ledger_key,) if ledger_key is not None else ()
        else:
            keys = tuple(
                key
                for key, ledger in self._ledgers.items()
                if ledger.session_id == session
            )
        for key in keys:
            self._ledgers.pop(key, None)
            self._locks.pop(key, None)

    @staticmethod
    def _validate_advice_lineage(
        *,
        request: OperationsContextRequest,
        experiences: tuple[OperationsRecalledExperience, ...],
        candidates: tuple[OperationsAdviceCandidate, ...],
    ) -> None:
        if any(not isinstance(candidate, OperationsAdviceCandidate) for candidate in candidates):
            raise TypeError("OperationsAdviceProvider returned a non-candidate value")
        if len(candidates) > 32:
            raise ValueError("OperationsAdviceProvider returned more than 32 candidates")
        if len(candidates) > request.operating_window.maximum_work_orders:
            raise ValueError(
                "OperationsAdviceProvider exceeded operating_window.maximum_work_orders"
            )
        candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("OperationsAdviceProvider candidate ids must be unique")
        available_evidence = {reference.ref_id for reference in request.evidence_refs}
        available_evidence.update(
            reference.ref_id for experience in experiences for reference in experience.report.evidence_refs
        )
        available_entries = {experience.memory_entry_id for experience in experiences}
        available_fact_ids = {fact.fact_id for fact in request.confirmed_facts}
        available_division_ids = set(request.division_ids)
        available_action_catalog_ids = set(request.action_catalog_ids)
        for candidate in candidates:
            unknown_evidence = set(candidate.evidence_ref_ids) - available_evidence
            unknown_entries = set(candidate.source_entry_ids) - available_entries
            if unknown_evidence or unknown_entries:
                raise ValueError(f"advice candidate {candidate.candidate_id!r} has unknown lineage")
            if candidate.target_division_id not in available_division_ids:
                raise ValueError(
                    f"advice candidate {candidate.candidate_id!r} targets an unknown division"
                )
            if candidate.action_catalog_id not in available_action_catalog_ids:
                raise ValueError(
                    f"advice candidate {candidate.candidate_id!r} uses an unknown action catalog id"
                )
            unknown_prerequisites = set(candidate.prerequisite_fact_ids) - available_fact_ids
            if unknown_prerequisites:
                raise ValueError(
                    f"advice candidate {candidate.candidate_id!r} has unknown prerequisite facts"
                )
            if candidate.maximum_cost_minor > request.operating_window.maximum_external_cost_minor:
                raise ValueError(
                    f"advice candidate {candidate.candidate_id!r} exceeds the operating cost bound"
                )
            if candidate.maximum_human_minutes > request.operating_window.maximum_human_minutes:
                raise ValueError(
                    f"advice candidate {candidate.candidate_id!r} exceeds the human capacity bound"
                )
            approval_required = (
                candidate.risk_level in {OperationsRiskLevel.HIGH, OperationsRiskLevel.CRITICAL}
                or candidate.reversibility is OperationsReversibility.IRREVERSIBLE
            )
            if approval_required and not candidate.requires_human_approval:
                raise ValueError(
                    f"advice candidate {candidate.candidate_id!r} requires explicit human approval"
                )
            for prediction_range in candidate.prediction_ranges:
                if (
                    prediction_range.horizon_start_ms
                    < request.operating_window.starts_at_ms
                    or prediction_range.horizon_end_ms
                    > request.operating_window.ends_at_ms
                ):
                    raise ValueError(
                        f"advice candidate {candidate.candidate_id!r} predicts outside "
                        "the operating window"
                    )
                if not set(prediction_range.evidence_ref_ids).issubset(set(candidate.evidence_ref_ids)):
                    raise ValueError(
                        f"advice candidate {candidate.candidate_id!r} prediction "
                        "range must use candidate evidence lineage"
                    )

    @staticmethod
    def _settled_evidence_refs(
        *,
        ledger: _SessionLedger,
        settled_outcome_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        settled = set(settled_outcome_ids)
        return tuple(
            dict.fromkeys(
                reference.ref_id
                for lineage in ledger.outcomes_by_outcome_id.values()
                if lineage.receipt.environment_outcome_id in settled
                for reference in lineage.receipt.report.evidence_refs
            )
        )

    def _trim_contexts(self, ledger: _SessionLedger) -> None:
        while len(ledger.contexts_by_request_id) > self._max_records_per_session:
            oldest_request_id = next(iter(ledger.contexts_by_request_id))
            removed = ledger.contexts_by_request_id.pop(oldest_request_id)
            ledger.contexts_by_pack_id.pop(removed.snapshot.context_pack_id, None)

    def _trim_outcomes(self, ledger: _SessionLedger) -> None:
        while len(ledger.outcomes_by_outcome_id) > self._max_records_per_session:
            removed = False
            for outcome_id, lineage in tuple(ledger.outcomes_by_outcome_id.items()):
                if lineage.receipt.environment_outcome_id == ledger.pending_environment_outcome_id:
                    continue
                ledger.outcomes_by_outcome_id.pop(outcome_id)
                removed = True
                break
            if not removed:
                break

    def _ledger_key_for(self, session: LifeformSession) -> str:
        ledger_key = self._session_keys.get(session)
        if ledger_key is None:
            ledger_key = f"operations-controller-session:{secrets.token_hex(32)}"
            self._session_keys[session] = ledger_key
        return ledger_key

    def _ledger_for(self, ledger_key: str, *, session_id: str) -> _SessionLedger:
        ledger = self._ledgers.get(ledger_key)
        if ledger is not None:
            return ledger
        ledger = _SessionLedger(
            session_id=session_id,
            session_lineage_id=f"operations-live-session:{secrets.token_hex(32)}"
        )
        self._ledgers[ledger_key] = ledger
        while len(self._ledgers) > self._max_session_ledgers:
            removed = False
            for candidate in tuple(self._ledgers):
                if candidate == ledger_key:
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


__all__ = (
    "EmptyOperationsAdviceProvider",
    "OperationsAdviceProvider",
    "OperationsBrainConflictError",
    "OperationsBrainController",
    "OperationsBrainError",
    "OperationsBrainLineageError",
    "OperationsBrainMemoryContractError",
    "OperationsBrainReadOnlyError",
    "OperationsBrainSettlementPendingError",
)
