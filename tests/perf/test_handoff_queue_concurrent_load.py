"""G-E: handoff queue concurrent load SLO (debt #70).

Two test tiers:

1. **Single-process N=10 async stress (v0.2 SHADOW)** — runs against
   the real ``HandoffTicketStore`` SQLite backend in-process. No
   GPU / no real production load needed; validates that the store's
   ``async create / list_for_ai / submit_human_reply`` behave
   correctly under concurrent calls. Locked by debt #70 v0.2 spec
   §9. Default-skipped under ``@pytest.mark.perf`` so PR CI stays
   fast; ``pytest tests/perf/ -m perf`` runs it explicitly (~< 5s).

2. **N=50 end_user × 10 tenant production load (v1.0 ACTIVE)** —
   depends on F-A perf bed + real production-shaped registry.
   Targets:
     * per-tenant queue isolation (tenant A's queue length doesn't
       affect tenant B's pickup latency)
     * P50 pickup latency < 10s
     * P99 pickup latency < 30s
     * fallback STRICT_REFUSE triggers at 5 min deadline (not LLM
       free-form)

The store lives at
``packages/dlaas-platform-registry/src/dlaas_platform_registry/handoff.py``
(NOT ``dlaas-platform-ops`` as debt #70 originally suggested —
packet G-E §"代码现状对账" recorded this discrepancy).

Refs:
    docs/moving forward/growth-advisor-pilot-packet.md §2.5 G-E
    docs/specs/handoff-queue-slo.md §8 / §9
    docs/known-debts.md #70
"""

from __future__ import annotations

import asyncio
import pathlib

import pytest


pytestmark = [pytest.mark.perf]


P50_PICKUP_LATENCY_SLO_S: float = 10.0
P99_PICKUP_LATENCY_SLO_S: float = 30.0
FALLBACK_DEADLINE_S: float = 300.0


# ---------------------------------------------------------------------------
# v0.2 SHADOW: single-process N=10 async stress (no F-A dependency)
# ---------------------------------------------------------------------------


def test_handoff_queue_single_process_concurrent_creates_ok(
    tmp_path: pathlib.Path,
) -> None:
    """N=10 concurrent ``store.create(...)`` in the same SQLite registry.

    Validates:

    * 10 unique ticket_ids
    * list_for_ai returns 10 entries, all OPEN
    * cross-concurrent ticket_id collisions do not occur
    """

    from dlaas_platform_contracts import HandoffStatus
    from dlaas_platform_registry.db import Registry
    from dlaas_platform_registry.handoff import HandoffTicketStore

    async def _drive() -> None:
        registry = Registry(db_path=str(tmp_path / "registry.db"))
        try:
            # Seed minimal foreign-key chain (tenants + templates +
            # contracts) so handoff_tickets.contract_id FK passes.
            registry.conn.execute(
                "INSERT INTO tenants (tenant_id, tenant_name, "
                "contact_email, api_key, api_secret_hash, created_at_ms) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("tnt-001", "Brand A", "ops@brand-a", "k", "h", 0),
            )
            registry.conn.execute(
                "INSERT INTO templates (template_id, tenant_id, "
                "template_name, created_at_ms) VALUES (?, ?, ?, ?)",
                ("tmpl-001", "tnt-001", "tmpl", 0),
            )
            registry.conn.execute(
                "INSERT INTO contracts (contract_id, tenant_id, "
                "template_id, shell_id, ai_id, created_at_ms) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("ctr-001", "tnt-001", "tmpl-001", "shell-001",
                 "cheng-laoshi", 0),
            )
            registry.conn.commit()
            store = HandoffTicketStore(registry)

            async def _one(idx: int):
                return await store.create(
                    ai_id="cheng-laoshi",
                    contract_id="ctr-001",
                    end_user_ref=f"alice-{idx:02d}",
                    session_id=f"sess-{idx:02d}",
                    trigger_reason="user_requested_human",
                    confidence_aggregate=0.9,
                )

            results = await asyncio.gather(*[_one(i) for i in range(10)])
            ticket_ids = {ticket.ticket_id for ticket in results}
            assert len(ticket_ids) == 10, (
                f"ticket_id collisions: {len(ticket_ids)} unique out of 10"
            )

            listed = await store.list_for_ai(
                ai_id="cheng-laoshi", status=HandoffStatus.OPEN
            )
            assert len(listed) == 10
            for ticket in listed:
                assert ticket.status is HandoffStatus.OPEN
        finally:
            registry.close()

    asyncio.run(_drive())


# ---------------------------------------------------------------------------
# v1.0 ACTIVE: N=50 × 10 tenant production load (depends on F-A)
# ---------------------------------------------------------------------------


def test_handoff_queue_per_tenant_isolation_and_pickup_slo(
    asyncio_harness,  # noqa: ANN001
    concurrent_lifeform_factory,  # noqa: ANN001
) -> None:
    """ACTIVE scaffold; depends on F-A perf bed."""

    pytest.skip(
        "G-E ACTIVE: production-load handoff test lands with G-E ACTIVE "
        "(Phase A W8). Targets: P50 < "
        f"{P50_PICKUP_LATENCY_SLO_S:.0f}s, P99 < "
        f"{P99_PICKUP_LATENCY_SLO_S:.0f}s, fallback STRICT_REFUSE at "
        f"{FALLBACK_DEADLINE_S:.0f}s. Depends on horizontal F-A perf bed."
    )


def test_handoff_state_persists_across_restart() -> None:
    """ACTIVE scaffold: handoff queue resumes after service restart."""

    pytest.skip(
        "G-E ACTIVE: cross-restart resume test lands with G-E ACTIVE."
    )
