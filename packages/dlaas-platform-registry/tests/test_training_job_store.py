"""Tests for the persisted training-job store (registry table)."""

from __future__ import annotations

from dlaas_platform_contracts import (
    TrainingJob,
    TrainingJobStatus,
    TrainingJobType,
)
from dlaas_platform_registry import Registry, TrainingJobStore


def _job(job_id: str, status: TrainingJobStatus) -> TrainingJob:
    return TrainingJob(
        job_id=job_id,
        ai_id="ai_1",
        contract_id="ctr_1",
        job_type=TrainingJobType.ADAPTER_CANDIDATE,
        status=status,
        source_ref="plan",
    )


async def test_put_get_round_trip() -> None:
    store = TrainingJobStore(Registry(db_path=":memory:"))
    await store.put(_job("j1", TrainingJobStatus.PENDING), tenant_id="t1")
    got = store.get(ai_id="ai_1", job_id="j1")
    assert got is not None
    assert got.job_id == "j1"
    assert got.status is TrainingJobStatus.PENDING
    assert store.get(ai_id="ai_1", job_id="missing") is None


async def test_status_update_persists() -> None:
    store = TrainingJobStore(Registry(db_path=":memory:"))
    await store.put(_job("j1", TrainingJobStatus.PENDING))
    await store.put(_job("j1", TrainingJobStatus.SUCCEEDED).with_artifact_ref("artifact:j1"))
    got = store.get(ai_id="ai_1", job_id="j1")
    assert got.status is TrainingJobStatus.SUCCEEDED
    assert got.artifact_ref == "artifact:j1"


async def test_list_by_status() -> None:
    store = TrainingJobStore(Registry(db_path=":memory:"))
    await store.put(_job("a", TrainingJobStatus.PENDING))
    await store.put(_job("b", TrainingJobStatus.PENDING))
    await store.put(_job("c", TrainingJobStatus.SUCCEEDED))
    pending = store.list_by_status(TrainingJobStatus.PENDING)
    assert {j.job_id for j in pending} == {"a", "b"}
    assert len(store.list_all()) == 3
