"""F-A subtask 2: multi-vertical owner snapshot propagation under load.

Verifies that 5 parallel verticals (companion / coding / character /
figure / growth-advisor — see ``PARALLEL_VERTICAL_PAIRS`` in
``tests/contracts/test_import_boundaries.py``) running N=5 sessions
each through a single ``InstanceManager`` do **not** experience owner
snapshot contention or PE-owner cross-pollution.

Debt: known-debts.md #45 (生产并发实测床)
Packet: docs/moving forward/cross-cutting-foundation-packet.md §2.1
"""

from __future__ import annotations

import pytest


pytestmark = [pytest.mark.perf]


PARALLEL_VERTICAL_NAMES: tuple[str, ...] = (
    "companion",
    "coding",
    "character",
    "figure",
    "growth_advisor",
)

# Each vertical's snapshot dispatch median ms target (ACTIVE SLO).
# SHADOW just records observed.
SNAPSHOT_DISPATCH_MEDIAN_MS_SLO: float = 50.0


def test_multi_vertical_owner_snapshot_no_contention(
    asyncio_harness,  # noqa: ANN001
    concurrent_lifeform_factory,  # noqa: ANN001
) -> None:
    """SHADOW scaffold."""

    pytest.skip(
        "SHADOW scaffold: 5 vertical × 5 session concurrent owner-snapshot "
        "test will be wired once F-A subtask 2 ships. Target: snapshot "
        f"dispatch median < {SNAPSHOT_DISPATCH_MEDIAN_MS_SLO:.0f}ms under "
        "25 concurrent sessions. "
        "See docs/moving forward/cross-cutting-foundation-packet.md §2.1."
    )


def test_pe_owner_no_cross_vertical_pollution(
    asyncio_harness,  # noqa: ANN001
    concurrent_lifeform_factory,  # noqa: ANN001
) -> None:
    """SHADOW scaffold: PE owner per vertical must remain isolated.

    ACTIVE-tier intent: after running 25 concurrent sessions across
    5 verticals, each vertical's ``PredictionErrorModule`` snapshot
    should reflect *only* turns from that vertical's sessions. Any
    leak (a coding-regime PE event showing up in companion's window)
    is a contract violation.
    """

    pytest.skip(
        "SHADOW scaffold: PE owner cross-vertical isolation assertion "
        "lands with F-A subtask 2."
    )
