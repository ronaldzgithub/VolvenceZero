"""Tests for the rare-heavy training-job executor (P1)."""

from __future__ import annotations

import asyncio

from dlaas_platform_contracts import (
    TrainingJob,
    TrainingJobStatus,
    TrainingJobType,
)
from dlaas_platform_api.training_executor import (
    InMemoryTrainingJobStore,
    RunnerResult,
    SyntheticTrainingJobRunner,
    TrainingJobExecutor,
)


def _job(job_id: str = "j1") -> TrainingJob:
    return TrainingJob(
        job_id=job_id,
        ai_id="ai_1",
        contract_id="ctr_1",
        job_type=TrainingJobType.ADAPTER_CANDIDATE,
        status=TrainingJobStatus.PENDING,
        source_ref="plan-1",
    )


class _FailingRunner:
    async def run(self, job):
        return RunnerResult(succeeded=False, detail="nope")


class _RaisingRunner:
    async def run(self, job):
        raise RuntimeError("boom")


async def test_synthetic_runner_succeeds_and_sets_artifact() -> None:
    store = InMemoryTrainingJobStore()
    await store.put(_job())
    ex = TrainingJobExecutor(store=store, runner=SyntheticTrainingJobRunner())
    await ex._run_one(ai_id="ai_1", job_id="j1")
    done = store.get(ai_id="ai_1", job_id="j1")
    assert done.status is TrainingJobStatus.SUCCEEDED
    assert done.artifact_ref == "artifact:j1"


async def test_failing_runner_marks_failed() -> None:
    store = InMemoryTrainingJobStore()
    await store.put(_job())
    ex = TrainingJobExecutor(store=store, runner=_FailingRunner())
    await ex._run_one(ai_id="ai_1", job_id="j1")
    assert store.get(ai_id="ai_1", job_id="j1").status is TrainingJobStatus.FAILED


async def test_raising_runner_captured_as_failed() -> None:
    store = InMemoryTrainingJobStore()
    await store.put(_job())
    ex = TrainingJobExecutor(store=store, runner=_RaisingRunner())
    await ex._run_one(ai_id="ai_1", job_id="j1")
    assert store.get(ai_id="ai_1", job_id="j1").status is TrainingJobStatus.FAILED


async def test_non_pending_job_is_skipped() -> None:
    store = InMemoryTrainingJobStore()
    await store.put(_job().with_status(TrainingJobStatus.CANCELLED))
    ex = TrainingJobExecutor(store=store, runner=SyntheticTrainingJobRunner())
    await ex._run_one(ai_id="ai_1", job_id="j1")
    # Cancelled stays cancelled (not resurrected to running/succeeded).
    assert store.get(ai_id="ai_1", job_id="j1").status is TrainingJobStatus.CANCELLED


async def test_worker_drains_enqueued_job() -> None:
    store = InMemoryTrainingJobStore()
    await store.put(_job("j2"))
    ex = TrainingJobExecutor(store=store, runner=SyntheticTrainingJobRunner())
    ex.start()
    ex.enqueue(ai_id="ai_1", job_id="j2")
    # Wait for the worker to process the queue.
    for _ in range(50):
        await asyncio.sleep(0.01)
        if store.get(ai_id="ai_1", job_id="j2").status is TrainingJobStatus.SUCCEEDED:
            break
    await ex.stop()
    assert store.get(ai_id="ai_1", job_id="j2").status is TrainingJobStatus.SUCCEEDED


async def test_requeue_pending_counts() -> None:
    store = InMemoryTrainingJobStore()
    await store.put(_job("a"))
    await store.put(_job("b"))
    await store.put(_job("c").with_status(TrainingJobStatus.SUCCEEDED))
    ex = TrainingJobExecutor(store=store, runner=SyntheticTrainingJobRunner())
    count = await ex.requeue_pending()
    assert count == 2
