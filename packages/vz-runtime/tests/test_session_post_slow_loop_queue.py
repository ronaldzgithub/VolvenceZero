from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import MethodType
from typing import cast

import pytest

from volvence_zero.agent.session import AgentSessionRunner
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


async def test_session_turn_lock_covers_complete_turn_response() -> None:
    runner = AgentSessionRunner(rare_heavy_enabled=False)
    active = 0
    max_active = 0
    order: list[str] = []

    async def fake_serialized_turn(
        _self: AgentSessionRunner,
        user_input: str,
        **_kwargs: object,
    ) -> str:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        order.append(f"start:{user_input}")
        await asyncio.sleep(0.02)
        order.append(f"end:{user_input}")
        active -= 1
        return user_input

    runner._run_turn_serialized = MethodType(  # type: ignore[method-assign]
        fake_serialized_turn,
        runner,
    )
    first, second = await asyncio.gather(
        runner.run_turn("first"),
        runner.run_turn("second"),
    )

    assert (first, second) == ("first", "second")
    assert max_active == 1
    assert order == ["start:first", "end:first", "start:second", "end:second"]
