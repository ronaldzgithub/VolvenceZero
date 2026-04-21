from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from volvence_zero.evaluation import EvaluationReport
from volvence_zero.integration import SessionPostWritebackRequest
from volvence_zero.reflection import WritebackResult


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


@dataclass(frozen=True)
class SessionPostSlowLoopResult:
    job_id: str
    context_session_id: str
    closed_at_turn: int
    writeback_result: WritebackResult | None
    applied: bool
    blocked: bool
    description: str


@dataclass(frozen=True)
class SessionPostSlowLoopQueueState:
    pending_job_count: int
    running_job_count: int
    completed_job_count: int
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
