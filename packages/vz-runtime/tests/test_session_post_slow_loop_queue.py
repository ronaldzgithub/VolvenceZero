from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import cast

import pytest

from volvence_zero.agent.session_post_slow_loop import (
    SessionPostSlowLoopJob,
    SessionPostSlowLoopQueue,
    SessionPostSlowLoopResult,
)


@dataclass(frozen=True)
class _TestJob:
    job_id: str
    context_session_id: str
    payload: str


def _as_job(job: _TestJob) -> SessionPostSlowLoopJob:
    return cast(SessionPostSlowLoopJob, job)


def test_queue_executes_identical_job_id_once_and_reports_duplicate() -> None:
    executions: list[str] = []

    async def worker(job: SessionPostSlowLoopJob) -> SessionPostSlowLoopResult:
        executions.append(job.job_id)
        return SessionPostSlowLoopResult(
            job_id=job.job_id,
            context_session_id=job.context_session_id,
            closed_at_turn=1,
            writeback_result=None,
            applied=False,
            blocked=False,
            description="test result",
        )

    queue = SessionPostSlowLoopQueue(worker=worker)
    job = _as_job(_TestJob("job-1", "session-1", "closed-evidence"))

    assert queue.enqueue(job) is True
    assert queue.enqueue(job) is False
    asyncio.run(queue.wait_for_idle())

    state = queue.snapshot()
    assert executions == ["job-1"]
    assert state.completed_job_count == 1
    assert state.duplicate_job_count == 1
    assert state.pending_job_count == 0


def test_queue_rejects_same_job_id_with_different_payload() -> None:
    async def worker(job: SessionPostSlowLoopJob) -> SessionPostSlowLoopResult:
        raise AssertionError(f"worker must not run for {job.job_id}")

    queue = SessionPostSlowLoopQueue(worker=worker)
    assert queue.enqueue(_as_job(_TestJob("job-1", "session-1", "first"))) is True

    with pytest.raises(ValueError, match="job_id collision"):
        queue.enqueue(_as_job(_TestJob("job-1", "session-1", "changed")))
