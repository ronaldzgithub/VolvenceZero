from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from volvence_zero.application.runtime import (
    ApplicationPriorUpdate,
    ApplicationPriorWritebackReport,
    ApplicationOutcomeAttribution,
    ApplicationSequencePayoff,
    ConversationKnowledgeCandidate,
    DelayedCreditSummary,
    ExperienceDelta,
    KnowledgeHit,
)
from volvence_zero.evaluation.backbone import EvaluationReport
from volvence_zero.integration import SessionPostWritebackRequest
from volvence_zero.credit.gate import SelfModificationRecord
from volvence_zero.reflection import WritebackResult
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel


@dataclass(frozen=True)
class SessionPostSlowLoopJob:
    job_id: str
    context_session_id: str
    closed_at_turn: int
    session_report: EvaluationReport
    prior_session_report_count: int
    trace_count: int
    substrate_batch_count: int
    prediction_error_summary: tuple[tuple[str, float], ...]
    writeback_request: SessionPostWritebackRequest
    description: str
    case_problem_patterns: tuple[str, ...] = ()
    case_risk_markers: tuple[str, ...] = ()
    knowledge_domains: tuple[str, ...] = ()
    knowledge_hits: tuple[KnowledgeHit, ...] = ()
    conversation_knowledge_candidates: tuple[ConversationKnowledgeCandidate, ...] = ()
    boundary_trigger_reasons: tuple[str, ...] = ()
    regime_id: str | None = None
    abstract_action: str | None = None
    action_family_version: int = 0
    retrieval_policy_id: str | None = None
    knowledge_weight: float = 0.0
    experience_weight: float = 0.0
    experience_domains: tuple[str, ...] = ()
    regime_sequence: tuple[str, ...] = ()
    case_hit_count: int = 0
    playbook_rule_count: int = 0
    continuum_profile_id: str | None = None
    case_band_ids: tuple[str, ...] = ()
    case_mean_continuum_position: float = 0.0
    playbook_band_ids: tuple[str, ...] = ()
    retrieval_fast_prior_strength: float = 0.0
    retrieval_fast_prior_attribution_count: int = 0
    retrieval_fast_prior_sequence_count: int = 0
    retrieval_regime_bias: float = 0.0
    retrieval_action_bias: float = 0.0
    retrieval_family_bias: float = 0.0
    retrieval_knowledge_weight_bias: float = 0.0
    retrieval_experience_weight_bias: float = 0.0


@dataclass(frozen=True)
class SessionPostSlowLoopResult:
    job_id: str
    context_session_id: str
    closed_at_turn: int
    writeback_result: WritebackResult | None
    applied: bool
    blocked: bool
    description: str
    experience_deltas: tuple[ExperienceDelta, ...] = ()
    delayed_outcome_ledger: tuple[ApplicationOutcomeAttribution, ...] = ()
    sequence_payoffs: tuple[ApplicationSequencePayoff, ...] = ()
    delayed_credit_summary: DelayedCreditSummary | None = None
    conversation_knowledge_candidates: tuple[ConversationKnowledgeCandidate, ...] = ()
    application_prior_update: ApplicationPriorUpdate | None = None
    application_prior_writeback_report: ApplicationPriorWritebackReport | None = None
    application_prior_audits: tuple[SelfModificationRecord, ...] = ()
    continuum_profile_id: str | None = None
    case_band_ids: tuple[str, ...] = ()
    playbook_band_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SessionPostSlowLoopQueueState:
    pending_job_count: int
    running_job_count: int
    completed_job_count: int
    last_completed_job_id: str | None
    last_completed_context_session_id: str | None
    description: str


@dataclass(frozen=True)
class SessionPostSlowLoopResultSummary:
    job_id: str
    context_session_id: str
    closed_at_turn: int
    applied_operation_count: int
    blocked_operation_count: int
    applied: bool
    blocked: bool
    description: str


@dataclass(frozen=True)
class SessionPostSlowLoopSnapshot:
    queue_state: SessionPostSlowLoopQueueState
    recent_results: tuple[SessionPostSlowLoopResultSummary, ...]
    last_completed_job_id: str | None
    last_completed_context_session_id: str | None
    description: str


class SessionPostSlowLoopQueue:
    def __init__(
        self,
        *,
        worker: Callable[[SessionPostSlowLoopJob], Awaitable[SessionPostSlowLoopResult]],
    ) -> None:
        self._worker = worker
        self._pending_jobs: deque[SessionPostSlowLoopJob] = deque()
        self._completed_results: deque[SessionPostSlowLoopResult] = deque()
        self._worker_task: asyncio.Task[None] | None = None
        self._completed_job_count = 0
        self._last_completed_job_id: str | None = None
        self._last_completed_context_session_id: str | None = None


    def enqueue(self, job: SessionPostSlowLoopJob) -> None:
        self._pending_jobs.append(job)

    def schedule(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._worker_task is not None and not self._worker_task.done():
            return
        if not self._pending_jobs:
            return
        self._worker_task = loop.create_task(self._drain())

    async def wait_for_idle(self) -> None:
        self.schedule()
        if self._worker_task is not None:
            await self._worker_task

    def consume_completed_results(self) -> tuple[SessionPostSlowLoopResult, ...]:
        results = tuple(self._completed_results)
        self._completed_results.clear()
        return results

    def snapshot(self) -> SessionPostSlowLoopQueueState:
        running_job_count = int(self._worker_task is not None and not self._worker_task.done())
        return SessionPostSlowLoopQueueState(
            pending_job_count=len(self._pending_jobs),
            running_job_count=running_job_count,
            completed_job_count=self._completed_job_count,
            last_completed_job_id=self._last_completed_job_id,
            last_completed_context_session_id=self._last_completed_context_session_id,
            description=(
                f"Session-post slow loop pending={len(self._pending_jobs)} "
                f"running={running_job_count} completed={self._completed_job_count}."
            ),
        )

    async def _drain(self) -> None:
        while self._pending_jobs:
            job = self._pending_jobs.popleft()
            result = await self._worker(job)
            self._completed_results.append(result)
            self._completed_job_count += 1
            self._last_completed_job_id = result.job_id
            self._last_completed_context_session_id = result.context_session_id


class SessionPostSlowLoopModule(RuntimeModule[SessionPostSlowLoopSnapshot]):
    slot_name = "session_post_slow_loop"
    owner = "SessionPostSlowLoopModule"
    value_type = SessionPostSlowLoopSnapshot
    dependencies = ()
    default_wiring_level = WiringLevel.ACTIVE

    def publish_snapshot(
        self,
        *,
        queue_state: SessionPostSlowLoopQueueState,
        completed_results: tuple[SessionPostSlowLoopResult, ...] = (),
    ) -> Snapshot[SessionPostSlowLoopSnapshot]:
        recent_results = tuple(
            SessionPostSlowLoopResultSummary(
                job_id=result.job_id,
                context_session_id=result.context_session_id,
                closed_at_turn=result.closed_at_turn,
                applied_operation_count=(
                    len(result.writeback_result.applied_operations) if result.writeback_result is not None else 0
                ),
                blocked_operation_count=(
                    len(result.writeback_result.blocked_operations) if result.writeback_result is not None else 0
                ),
                applied=result.applied,
                blocked=result.blocked,
                description=result.description,
            )
            for result in completed_results[-4:]
        )
        return self.publish(
            SessionPostSlowLoopSnapshot(
                queue_state=queue_state,
                recent_results=recent_results,
                last_completed_job_id=queue_state.last_completed_job_id,
                last_completed_context_session_id=queue_state.last_completed_context_session_id,
                description=(
                    f"Session-post slow loop slot pending={queue_state.pending_job_count} "
                    f"running={queue_state.running_job_count} completed={queue_state.completed_job_count} "
                    f"recent_results={len(recent_results)}."
                ),
            )
        )

    async def process(self, upstream: Mapping[str, Snapshot[object]]) -> Snapshot[SessionPostSlowLoopSnapshot]:
        raise NotImplementedError("SessionPostSlowLoopModule is published via process_standalone().")

    async def process_standalone(
        self,
        *,
        queue_state: SessionPostSlowLoopQueueState,
        completed_results: tuple[SessionPostSlowLoopResult, ...] = (),
    ) -> Snapshot[SessionPostSlowLoopSnapshot]:
        return self.publish_snapshot(
            queue_state=queue_state,
            completed_results=completed_results,
        )
