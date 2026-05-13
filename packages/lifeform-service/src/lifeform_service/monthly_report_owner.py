"""MonthlyReportOwner — aggregates per-tenant monthly metrics (debt #67).

Aggregation pipeline:

1. read source owner snapshots (rupture / boundary / drive / handoff
   queue / archetype state) for end users in this tenant
2. roll up per-tenant + per-archetype counts / rates
3. emit a typed ``MonthlyReportSnapshot`` (immutable, R8 owner)

This module is the **single owner** of the monthly report; downstream
consumers (admin dashboard / customer-facing PDF) read the snapshot
and never re-derive aggregates. AST contract test in
``tests/contracts/test_monthly_report_schema_stability.py`` enforces
schema version stability across releases (a customer-facing report
must read the same field names month over month).

Refs:

* docs/moving forward/growth-advisor-pilot-packet.md §2.4 G-D
* docs/specs/growth-advisor-monthly-report.md
* docs/known-debts.md #67
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping


REPORT_SCHEMA_VERSION = "v0.1"


@dataclasses.dataclass(frozen=True)
class MonthlyReportSnapshot:
    """Frozen monthly report for one (tenant_id, month) pair.

    All fields are read-only: regeneration produces a new snapshot,
    never mutates an existing one. The customer-facing PDF / web
    rendering reads these fields verbatim.
    """

    report_schema_version: str
    tenant_id: str
    month_iso: str  # "YYYY-MM"
    period_start_ms: int
    period_end_ms: int
    # User activity
    end_user_count_total: int
    end_user_count_active: int
    new_end_user_count: int
    average_turns_per_active_user: float
    # Boundary policy
    boundary_trigger_total: int
    boundary_trigger_per_policy: dict[str, int]  # bp-no-hard-sell etc.
    # Rupture / repair
    rupture_count: int
    repair_count: int
    repair_rate: float
    # Archetype
    archetype_distribution: dict[str, float]  # 5 archetypes → fraction
    # Handoff
    handoff_triggered_count: int
    handoff_completed_count: int
    handoff_p99_seconds: float
    # Day-cohort
    day_cohort_active_counts: dict[str, int]  # "day1".."day7+"

    def __post_init__(self) -> None:
        if self.report_schema_version != REPORT_SCHEMA_VERSION:
            raise ValueError(
                f"MonthlyReportSnapshot.report_schema_version="
                f"{self.report_schema_version!r} != {REPORT_SCHEMA_VERSION!r}"
            )
        if not self.tenant_id or not self.tenant_id.strip():
            raise ValueError("MonthlyReportSnapshot.tenant_id must be non-empty")
        if not self.month_iso or "-" not in self.month_iso:
            raise ValueError(
                "MonthlyReportSnapshot.month_iso must be 'YYYY-MM' format"
            )
        if self.end_user_count_total < 0 or self.end_user_count_active < 0:
            raise ValueError("user counts must be non-negative")
        archetype_total = sum(self.archetype_distribution.values())
        if self.archetype_distribution and not (0.99 <= archetype_total <= 1.01):
            raise ValueError(
                f"archetype_distribution sums to {archetype_total:.4f}; "
                f"expected ~1.0"
            )


@dataclasses.dataclass(frozen=True)
class MonthlyReportInputs:
    """Aggregator inputs from upstream owner snapshots.

    Each field is a small immutable summary — the aggregator does
    NOT walk evidence_root_dir or memory store directly (that would
    duplicate ownership; R8). Upstream owners (boundary policy /
    rupture state / archetype classifier / handoff queue) publish
    their own per-end-user summaries which this aggregator consumes.
    """

    tenant_id: str
    month_iso: str
    period_start_ms: int
    period_end_ms: int
    per_end_user_turn_counts: Mapping[str, int]  # end_user_id → turn count
    new_end_user_ids: tuple[str, ...]
    boundary_trigger_per_policy: Mapping[str, int]
    rupture_count: int
    repair_count: int
    archetype_distribution: Mapping[str, float]
    handoff_triggered_count: int
    handoff_completed_count: int
    handoff_p99_seconds: float
    day_cohort_active_counts: Mapping[str, int]


class MonthlyReportOwner:
    """SSOT for monthly report generation.

    Consumers call ``aggregate(inputs)`` to get a frozen
    ``MonthlyReportSnapshot``; the owner caches per (tenant, month)
    in memory until the next aggregation overwrites it.

    SHADOW: aggregation is pure-function and deterministic. ACTIVE
    will add: persistence + PDF/HTML rendering + admin endpoint
    serving (the corresponding ``GET /v1/tenants/{tid}/admin/monthly-report``
    handler skeleton lives in protocol_routes.py per #67 ACTIVE).
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], MonthlyReportSnapshot] = {}

    def aggregate(self, inputs: MonthlyReportInputs) -> MonthlyReportSnapshot:
        end_user_count_total = len(inputs.per_end_user_turn_counts)
        end_user_count_active = sum(
            1 for n in inputs.per_end_user_turn_counts.values() if n > 0
        )
        average_turns = (
            sum(inputs.per_end_user_turn_counts.values()) / max(1, end_user_count_active)
        )
        snapshot = MonthlyReportSnapshot(
            report_schema_version=REPORT_SCHEMA_VERSION,
            tenant_id=inputs.tenant_id,
            month_iso=inputs.month_iso,
            period_start_ms=inputs.period_start_ms,
            period_end_ms=inputs.period_end_ms,
            end_user_count_total=end_user_count_total,
            end_user_count_active=end_user_count_active,
            new_end_user_count=len(inputs.new_end_user_ids),
            average_turns_per_active_user=average_turns,
            boundary_trigger_total=sum(inputs.boundary_trigger_per_policy.values()),
            boundary_trigger_per_policy=dict(inputs.boundary_trigger_per_policy),
            rupture_count=inputs.rupture_count,
            repair_count=inputs.repair_count,
            repair_rate=(
                inputs.repair_count / inputs.rupture_count
                if inputs.rupture_count > 0
                else 0.0
            ),
            archetype_distribution=dict(inputs.archetype_distribution),
            handoff_triggered_count=inputs.handoff_triggered_count,
            handoff_completed_count=inputs.handoff_completed_count,
            handoff_p99_seconds=inputs.handoff_p99_seconds,
            day_cohort_active_counts=dict(inputs.day_cohort_active_counts),
        )
        self._cache[(inputs.tenant_id, inputs.month_iso)] = snapshot
        return snapshot

    def latest_for(self, tenant_id: str, month_iso: str) -> MonthlyReportSnapshot | None:
        return self._cache.get((tenant_id, month_iso))


__all__ = (
    "MonthlyReportInputs",
    "MonthlyReportOwner",
    "MonthlyReportSnapshot",
    "REPORT_SCHEMA_VERSION",
)
