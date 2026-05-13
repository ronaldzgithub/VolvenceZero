"""Shared fixtures for ``tests/perf/``.

These fixtures are intentionally **scaffolds**: they expose the public
shape that performance tests will consume, but the heavy concurrent
``LifeformSession`` / GPU-tracker plumbing lands incrementally as
debt #45 advances from SHADOW → ACTIVE (see
:doc:`docs/specs/perf-baseline.md`).

The fixtures live here (instead of ``tests/conftest.py``) so the
``tests/perf/`` directory remains an isolated subtree that can be
collected independently::

    pytest tests/perf/ -m perf -k concurrent_sessions

Design constraints (R8 / R12):

* Performance tests **only read** owner snapshots / metrics; they
  never write into kernel owners.
* GPU memory measurements are observational telemetry (a readout),
  not a learning signal.
* Concurrency harnesses must use ``asyncio`` rather than threads so
  they exercise the same scheduling path as the real
  ``lifeform-service`` HTTP layer.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import Awaitable, Callable
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Register ``@pytest.mark.perf`` so it is not flagged as unknown."""

    config.addinivalue_line(
        "markers",
        "perf: production-grade performance / concurrency tests "
        "(skipped by default; run with `pytest tests/perf/ -m perf`)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip every test in ``tests/perf/`` unless ``-m perf`` is requested.

    This guards CI latency: a default ``pytest`` run never executes
    perf tests, but a developer running ``pytest tests/perf/ -m perf``
    will get the real (slow) execution.
    """

    if config.getoption("-m") and "perf" in str(config.getoption("-m")):
        return  # explicit opt-in
    skip_marker = pytest.mark.skip(reason="perf tests skipped by default; run with -m perf")
    for item in items:
        if "perf" in item.keywords:
            item.add_marker(skip_marker)


# ---------------------------------------------------------------------------
# Concurrency harness
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LatencyBucket:
    """A latency observation bucket from one concurrent run."""

    label: str
    samples: tuple[float, ...]

    def percentile(self, p: float) -> float:
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = max(0, min(len(sorted_samples) - 1, int(p * len(sorted_samples))))
        return sorted_samples[idx]


@dataclasses.dataclass(frozen=True)
class ConcurrencyResult:
    """Aggregated result from an ``asyncio_harness`` run."""

    n_workers: int
    total_seconds: float
    per_worker_latencies: tuple[LatencyBucket, ...]
    failures: tuple[str, ...]

    @property
    def all_samples(self) -> tuple[float, ...]:
        out: list[float] = []
        for bucket in self.per_worker_latencies:
            out.extend(bucket.samples)
        return tuple(out)


@pytest.fixture
def asyncio_harness() -> Callable[..., ConcurrencyResult]:
    """Launch ``n_workers`` async coroutines in parallel and report latency.

    Usage in a test::

        async def one_turn(worker_id: int) -> float:
            t0 = time.perf_counter()
            await some_real_call(worker_id)
            return time.perf_counter() - t0

        result = asyncio_harness(coro_factory=one_turn, n_workers=20)
        assert result.percentile_overall(0.99) < 3.0
    """

    def _run(
        *,
        coro_factory: Callable[[int], Awaitable[float]],
        n_workers: int = 20,
        rounds_per_worker: int = 1,
    ) -> ConcurrencyResult:
        async def _worker(worker_id: int) -> LatencyBucket:
            samples: list[float] = []
            for _ in range(rounds_per_worker):
                duration = await coro_factory(worker_id)
                samples.append(duration)
            return LatencyBucket(label=f"worker-{worker_id}", samples=tuple(samples))

        async def _runner() -> ConcurrencyResult:
            t0 = time.perf_counter()
            buckets = await asyncio.gather(
                *[_worker(i) for i in range(n_workers)], return_exceptions=True
            )
            total = time.perf_counter() - t0
            ok: list[LatencyBucket] = []
            failures: list[str] = []
            for b in buckets:
                if isinstance(b, BaseException):
                    failures.append(repr(b))
                else:
                    ok.append(b)
            return ConcurrencyResult(
                n_workers=n_workers,
                total_seconds=total,
                per_worker_latencies=tuple(ok),
                failures=tuple(failures),
            )

        return asyncio.run(_runner())

    return _run


# ---------------------------------------------------------------------------
# LifeformSession factory (SHADOW placeholder)
# ---------------------------------------------------------------------------


@pytest.fixture
def concurrent_lifeform_factory() -> Callable[..., Any]:
    """Build N independent ``LifeformSession`` instances for concurrent stress.

    SHADOW placeholder: returns a callable that raises
    ``NotImplementedError`` until the real wire-up lands as part of
    F-A subtask 2 (see :doc:`docs/moving forward/cross-cutting-foundation-packet.md`
    §2.1).

    The signature is locked now so downstream test modules can
    reference it without churn when the real implementation arrives.
    """

    def _build(
        *,
        vertical: str,
        n: int,
        memory_root: str | None = None,
    ) -> tuple[Any, ...]:
        raise NotImplementedError(
            "concurrent_lifeform_factory will land in F-A subtask 2 "
            "(see docs/moving forward/cross-cutting-foundation-packet.md §2.1). "
            f"Requested vertical={vertical!r} n={n} memory_root={memory_root!r}."
        )

    return _build


# ---------------------------------------------------------------------------
# GPU memory tracker (SHADOW placeholder)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GpuMemSample:
    timestamp: float
    allocated_mb: float
    reserved_mb: float


@pytest.fixture
def gpu_mem_tracker() -> Callable[[], list[GpuMemSample]]:
    """Sample CUDA memory at the call site.

    SHADOW: returns an empty list when CUDA is unavailable so tests
    can call this unconditionally. Real tracker (PyTorch ``cuda``
    hook) lands with the GPU-bearing perf tests in F-A subtask 3.
    """

    samples: list[GpuMemSample] = []

    def _sample() -> list[GpuMemSample]:
        try:
            import torch  # noqa: PLC0415

            if not torch.cuda.is_available():
                return list(samples)
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            samples.append(
                GpuMemSample(
                    timestamp=time.time(),
                    allocated_mb=float(allocated),
                    reserved_mb=float(reserved),
                )
            )
        except (ImportError, RuntimeError):
            pass
        return list(samples)

    return _sample
