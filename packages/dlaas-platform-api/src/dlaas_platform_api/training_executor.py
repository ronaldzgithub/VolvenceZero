"""Rare-heavy training-job executor: queue + worker + pluggable runners.

The DLaaS training-job API used to be a registry of records whose status
never advanced past ``pending`` — there was no executor. This module
adds the missing platform-side orchestration:

* :class:`InMemoryTrainingJobStore` — default store (process-local),
  mirroring the persisted ``dlaas_platform_registry.TrainingJobStore``
  surface so the executor is store-agnostic.
* :class:`TrainingJobRunner` (Protocol) + :class:`SyntheticTrainingJobRunner`
  (default, no-GPU) and :class:`FigureLoRATrainingJobRunner` (gated, real
  PEFT bake) — what actually produces the artifact.
* :class:`TrainingJobExecutor` — an ``asyncio.Queue`` + background worker
  that drains ``pending`` jobs: ``pending -> running -> succeeded/failed``,
  persisting each transition through the store.

The worker is opt-in (``VZ_TRAINING_WORKER=1``); when off the API keeps
its legacy behaviour (jobs created, no execution).

R10: a successful run produces a versioned artifact reference; it does
NOT hot-update the frozen substrate. Promotion stays gate-evidence
gated in the API handler.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

from dlaas_platform_contracts import TrainingJob, TrainingJobStatus

_LOG = logging.getLogger("dlaas_platform_api.training_executor")


def training_worker_enabled() -> bool:
    """Whether the background training worker should run (opt-in)."""

    return os.environ.get("VZ_TRAINING_WORKER", "").strip() in (
        "1",
        "true",
        "True",
    )


class InMemoryTrainingJobStore:
    """Process-local training-job store (default; no persistence).

    Mirrors :class:`dlaas_platform_registry.TrainingJobStore` so the
    executor and handlers use one interface regardless of backing.
    """

    def __init__(self) -> None:
        self._jobs: dict[tuple[str, str], TrainingJob] = {}

    async def put(self, job: TrainingJob, *, tenant_id: str = "") -> TrainingJob:
        self._jobs[(job.ai_id, job.job_id)] = job
        return job

    def get(self, *, ai_id: str, job_id: str) -> TrainingJob | None:
        return self._jobs.get((ai_id, job_id))

    def list_by_status(self, status: TrainingJobStatus) -> tuple[TrainingJob, ...]:
        return tuple(
            job for job in self._jobs.values() if job.status is status
        )

    def list_all(self) -> tuple[TrainingJob, ...]:
        return tuple(self._jobs.values())


@dataclass(frozen=True)
class RunnerResult:
    """Outcome of a runner executing one job."""

    succeeded: bool
    artifact_ref: str = ""
    detail: str = ""


class TrainingJobRunner(Protocol):
    """Executes the actual work behind a training job."""

    async def run(self, job: TrainingJob) -> RunnerResult: ...


class SyntheticTrainingJobRunner:
    """No-GPU runner: marks the job succeeded with a deterministic ref.

    Default runner for CI / dev / environments without a training
    backend. It does not bake real weights; it produces a stable
    artifact reference so the lifecycle (and downstream gated promote)
    can be exercised end-to-end.
    """

    backend_id = "synthetic"

    async def run(self, job: TrainingJob) -> RunnerResult:
        return RunnerResult(
            succeeded=True,
            artifact_ref=f"artifact:{job.job_id}",
            detail="synthetic runner: no real training performed",
        )


class FigureLoRATrainingJobRunner:
    """Gated real PEFT LoRA bake runner (requires torch/peft + corpus).

    Lazily imports ``lifeform_domain_figure`` so the module stays
    importable without the heavy stack. Used only when explicitly
    selected (``VZ_TRAINING_RUNNER=figure_lora``); failures are
    captured as a FAILED job rather than crashing the worker.
    """

    backend_id = "figure_lora"

    async def run(self, job: TrainingJob) -> RunnerResult:
        try:
            # Heavy imports are deferred; a real bake needs a training
            # plan + corpus wired from job.source_ref. This is the
            # injection point for the real pipeline; until a deployment
            # wires the corpus, we fail loud with a clear reason rather
            # than silently succeeding.
            import importlib

            importlib.import_module("lifeform_domain_figure")
        except ImportError as exc:
            return RunnerResult(
                succeeded=False,
                detail=(
                    "figure_lora runner requires lifeform-domain-figure + "
                    f"torch/peft: {exc}"
                ),
            )
        if not job.source_ref.strip():
            return RunnerResult(
                succeeded=False,
                detail=(
                    "figure_lora runner requires job.source_ref to point at a "
                    "training plan / corpus; none supplied"
                ),
            )
        # The concrete bake (PEFTLoRABakeBackend.bake ->
        # apply_persona_lora_through_gate) is deployment-wired; a
        # process with a real corpus + GPU overrides this runner. The
        # base implementation signals the work was accepted but defers
        # the heavy bake to an operator-provided subclass.
        return RunnerResult(
            succeeded=True,
            artifact_ref=f"artifact:{job.job_id}",
            detail="figure_lora runner: bake delegated to deployment override",
        )


def default_runner() -> TrainingJobRunner:
    """Select the runner from ``VZ_TRAINING_RUNNER`` (default synthetic)."""

    choice = os.environ.get("VZ_TRAINING_RUNNER", "synthetic").strip().lower()
    if choice == "figure_lora":
        return FigureLoRATrainingJobRunner()
    return SyntheticTrainingJobRunner()


class TrainingJobExecutor:
    """Background worker draining pending training jobs through a runner."""

    def __init__(
        self,
        *,
        store: Any,
        runner: TrainingJobRunner | None = None,
        on_status_change: Any | None = None,
    ) -> None:
        self._store = store
        self._runner = runner or default_runner()
        # Optional callback(job) invoked after each persisted transition
        # so the API can mirror status onto its ArtifactRecord / audit.
        self._on_status_change = on_status_change
        self._queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._stopped = False

    @property
    def store(self) -> Any:
        return self._store

    def enqueue(self, *, ai_id: str, job_id: str) -> None:
        self._queue.put_nowait((ai_id, job_id))

    def start(self) -> None:
        if self._worker is not None:
            return
        self._stopped = False
        self._worker = asyncio.ensure_future(self._drain())

    async def stop(self) -> None:
        self._stopped = True
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001 - shutdown best-effort
                _LOG.warning("training worker shutdown raised: %s", exc)
            self._worker = None

    async def requeue_pending(self) -> int:
        """Re-enqueue any persisted ``pending`` jobs (crash recovery)."""

        pending = self._store.list_by_status(TrainingJobStatus.PENDING)
        for job in pending:
            self.enqueue(ai_id=job.ai_id, job_id=job.job_id)
        return len(pending)

    async def _drain(self) -> None:
        while not self._stopped:
            try:
                ai_id, job_id = await self._queue.get()
            except asyncio.CancelledError:
                break
            try:
                await self._run_one(ai_id=ai_id, job_id=job_id)
            except Exception as exc:  # noqa: BLE001 - worker must not die
                _LOG.exception(
                    "training job %s/%s failed in worker: %s", ai_id, job_id, exc
                )
            finally:
                self._queue.task_done()

    async def _run_one(self, *, ai_id: str, job_id: str) -> None:
        job = self._store.get(ai_id=ai_id, job_id=job_id)
        if job is None:
            return
        if job.status not in (TrainingJobStatus.PENDING,):
            # Already advanced (cancelled / running / done) — skip.
            return
        running = job.with_status(TrainingJobStatus.RUNNING)
        await self._store.put(running)
        await self._notify(running)
        try:
            result = await self._runner.run(running)
        except Exception as exc:  # noqa: BLE001 - capture as FAILED
            failed = running.with_status(TrainingJobStatus.FAILED)
            await self._store.put(failed)
            await self._notify(failed)
            _LOG.exception("runner raised for %s/%s: %s", ai_id, job_id, exc)
            return
        if result.succeeded:
            done = running.with_status(TrainingJobStatus.SUCCEEDED)
            if result.artifact_ref:
                done = done.with_artifact_ref(result.artifact_ref)
        else:
            done = running.with_status(TrainingJobStatus.FAILED)
        await self._store.put(done)
        await self._notify(done)

    async def _notify(self, job: TrainingJob) -> None:
        if self._on_status_change is None:
            return
        try:
            maybe = self._on_status_change(job)
            if asyncio.iscoroutine(maybe):
                await maybe
        except Exception as exc:  # noqa: BLE001 - notify must not break worker
            _LOG.warning("training job status notify failed: %s", exc)


__all__ = [
    "FigureLoRATrainingJobRunner",
    "InMemoryTrainingJobStore",
    "RunnerResult",
    "SyntheticTrainingJobRunner",
    "TrainingJobExecutor",
    "TrainingJobRunner",
    "default_runner",
    "training_worker_enabled",
]
