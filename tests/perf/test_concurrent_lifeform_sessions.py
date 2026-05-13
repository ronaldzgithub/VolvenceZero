"""F-A subtask 2: concurrent ``LifeformSession`` SLO baseline.

Verifies that with ``N=20`` concurrent sessions on a single substrate
runtime, the P99 turn latency stays under the SLO defined in
:doc:`docs/specs/perf-baseline.md`.

This is a SHADOW scaffold: it wires the harness + asserts the SLO
shape, but the actual session factory raises ``NotImplementedError``
until F-A subtask 2 lands the real wire-up. CI default-skips perf
tests (see ``conftest.py``), so this scaffold is safe to merge.

Debt: known-debts.md #45 (生产并发实测床)
Packet: docs/moving forward/cross-cutting-foundation-packet.md §2.1
"""

from __future__ import annotations

import pytest


pytestmark = [pytest.mark.perf]


# Per-vertical SLO targets (P99 seconds per turn, 20 concurrent sessions).
# These are ACTIVE-tier SLOs; SHADOW just records the observed value.
P99_LATENCY_SLO_SECONDS: dict[str, float] = {
    "companion": 3.0,
    "figure": 5.0,
    "growth_advisor": 3.0,
}


@pytest.mark.parametrize("vertical", ["companion", "figure", "growth_advisor"])
def test_p99_turn_latency_under_slo(
    vertical: str,
    asyncio_harness,  # noqa: ANN001
    concurrent_lifeform_factory,  # noqa: ANN001
) -> None:
    """SHADOW scaffold: assert harness shape; real wire-up pending."""

    pytest.skip(
        f"SHADOW scaffold: concurrent_lifeform_factory(vertical={vertical!r}) "
        "raises NotImplementedError until F-A subtask 2 lands. "
        "Per-vertical P99 SLO target = "
        f"{P99_LATENCY_SLO_SECONDS[vertical]:.1f}s. "
        "See docs/moving forward/cross-cutting-foundation-packet.md §2.1."
    )

    # Below is the intended structure once the factory is real:
    # sessions = concurrent_lifeform_factory(vertical=vertical, n=20)
    #
    # async def _one_turn(worker_id: int) -> float:
    #     session = sessions[worker_id]
    #     t0 = time.perf_counter()
    #     await session.run_turn(...)
    #     return time.perf_counter() - t0
    #
    # result = asyncio_harness(coro_factory=_one_turn, n_workers=20)
    # p99 = max(b.percentile(0.99) for b in result.per_worker_latencies)
    # assert p99 < P99_LATENCY_SLO_SECONDS[vertical], (
    #     f"P99 latency {p99:.2f}s exceeds SLO "
    #     f"{P99_LATENCY_SLO_SECONDS[vertical]:.2f}s for vertical={vertical}"
    # )
