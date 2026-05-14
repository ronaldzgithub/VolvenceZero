"""Contract test: MonthlyReportSnapshot field set is stable (debt #67).

Customer-facing monthly reports must read the same field names
month over month. This contract pins the field set + version
string; bumping either requires explicit acknowledgement (test
update) so a silent schema change cannot ship.

Refs:

* docs/specs/growth-advisor-monthly-report.md
* docs/known-debts.md #67
"""

from __future__ import annotations

import dataclasses

from lifeform_service.monthly_report_owner import (
    REPORT_SCHEMA_VERSION,
    MonthlyReportInputs,
    MonthlyReportOwner,
    MonthlyReportSnapshot,
)


_PINNED_SCHEMA_VERSION = "v0.2"


_PINNED_SNAPSHOT_FIELDS = frozenset(
    {
        "report_schema_version",
        "tenant_id",
        "month_iso",
        "period_start_ms",
        "period_end_ms",
        "end_user_count_total",
        "end_user_count_active",
        "new_end_user_count",
        "average_turns_per_active_user",
        "boundary_trigger_total",
        "boundary_trigger_per_policy",
        "rupture_count",
        "repair_count",
        "repair_rate",
        "archetype_distribution",
        "handoff_triggered_count",
        "handoff_completed_count",
        "handoff_p99_seconds",
        # v0.2: replaces day_cohort_active_counts (calendar-day routing
        # deprecated 2026-05-14; phase ids come from
        # BehaviorProtocol.TemporalArc.progression_signals).
        "protocol_phase_cohort_active_counts",
        # v0.2: deletion auditability (debt #49).
        "deleted_end_user_count",
        "deletion_event_count",
    }
)


_PINNED_INPUTS_FIELDS = frozenset(
    {
        "tenant_id",
        "month_iso",
        "period_start_ms",
        "period_end_ms",
        "per_end_user_turn_counts",
        "new_end_user_ids",
        "boundary_trigger_per_policy",
        "rupture_count",
        "repair_count",
        "archetype_distribution",
        "handoff_triggered_count",
        "handoff_completed_count",
        "handoff_p99_seconds",
        "protocol_phase_cohort_active_counts",
        "deleted_end_user_count",
        "deletion_event_count",
    }
)


def test_schema_version_pinned() -> None:
    assert REPORT_SCHEMA_VERSION == _PINNED_SCHEMA_VERSION, (
        f"REPORT_SCHEMA_VERSION drifted from pinned {_PINNED_SCHEMA_VERSION!r} "
        f"to {REPORT_SCHEMA_VERSION!r}; if this is intentional, bump _PINNED_SCHEMA_VERSION "
        "AND notify downstream PDF rendering team (customer-facing reports)."
    )


def test_snapshot_fields_pinned() -> None:
    actual = {f.name for f in dataclasses.fields(MonthlyReportSnapshot)}
    diff_added = actual - _PINNED_SNAPSHOT_FIELDS
    diff_removed = _PINNED_SNAPSHOT_FIELDS - actual
    assert not diff_added, (
        f"MonthlyReportSnapshot added fields {sorted(diff_added)}; "
        "bump REPORT_SCHEMA_VERSION + update _PINNED_SNAPSHOT_FIELDS "
        "+ notify customer-facing PDF rendering team."
    )
    assert not diff_removed, (
        f"MonthlyReportSnapshot removed fields {sorted(diff_removed)}; "
        "this is breaking for customers reading historical reports. "
        "Use migration shim instead (see growth-advisor-monthly-report.md §4)."
    )


def test_inputs_fields_pinned() -> None:
    actual = {f.name for f in dataclasses.fields(MonthlyReportInputs)}
    diff_added = actual - _PINNED_INPUTS_FIELDS
    diff_removed = _PINNED_INPUTS_FIELDS - actual
    assert not diff_added, f"MonthlyReportInputs added fields {sorted(diff_added)}"
    assert not diff_removed, f"MonthlyReportInputs removed fields {sorted(diff_removed)}"


def test_owner_aggregate_returns_snapshot() -> None:
    """Smoke: owner.aggregate() returns immutable snapshot with all pinned fields."""
    owner = MonthlyReportOwner()
    inputs = MonthlyReportInputs(
        tenant_id="brand_a",
        month_iso="2026-04",
        period_start_ms=1_711_897_200_000,
        period_end_ms=1_714_489_200_000,
        per_end_user_turn_counts={"alice": 23, "bob": 0, "carol": 47},
        new_end_user_ids=("dan",),
        boundary_trigger_per_policy={
            "bp-no-hard-sell": 12,
            "bp-no-overclaim": 7,
            "bp-no-flooding": 3,
            "bp-no-judgmental": 4,
        },
        rupture_count=8,
        repair_count=7,
        archetype_distribution={
            "anxious": 0.34,
            "comparing": 0.18,
            "standard_seeking": 0.22,
            "venting": 0.16,
            "product_seeking": 0.10,
        },
        handoff_triggered_count=3,
        handoff_completed_count=2,
        handoff_p99_seconds=18.4,
        protocol_phase_cohort_active_counts={
            "icebreaker": 4,
            "baseline": 3,
            "empathy": 5,
            "pain_mining": 2,
            "rapport": 1,
            "targeted_advice": 1,
            "summary_hook": 2,
        },
        deleted_end_user_count=1,
        deletion_event_count=2,
    )
    snapshot = owner.aggregate(inputs)
    assert snapshot.report_schema_version == _PINNED_SCHEMA_VERSION
    assert snapshot.tenant_id == "brand_a"
    assert snapshot.end_user_count_total == 3
    assert snapshot.end_user_count_active == 2
    assert snapshot.repair_rate == 7 / 8
    assert snapshot.deleted_end_user_count == 1
    assert snapshot.deletion_event_count == 2
    assert snapshot.protocol_phase_cohort_active_counts["icebreaker"] == 4
