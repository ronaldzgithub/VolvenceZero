"""MonthlyReportOwner — aggregates per-tenant monthly metrics (debt #67).

Aggregation pipeline:

1. read source owner snapshots (rupture / boundary / archetype state /
   handoff queue / protocol phase) for end users in this tenant
2. roll up per-tenant + per-archetype counts / rates
3. emit a typed ``MonthlyReportSnapshot`` (immutable, R8 owner)

This module is the **single owner** of the monthly report; downstream
consumers (admin dashboard / customer-facing PDF) read the snapshot
and never re-derive aggregates. AST contract test in
``tests/contracts/test_monthly_report_schema_stability.py`` enforces
schema version stability across releases (a customer-facing report
must read the same field names month over month).

Schema version log:

* v0.1 (initial scaffold)
* v0.2 (2026-05-14):
  * day_cohort_active_counts → protocol_phase_cohort_active_counts
    (calendar-day routing was deprecated; relationship phase routing
    flows through ``BehaviorProtocol.TemporalArc.progression_signals``)
  * deleted_end_user_count + deletion_event_count fields added
    (debt #49 — GDPR/PIPL compliance auditability without leaking
    deleted end-user content)

Refs:

* docs/moving forward/growth-advisor-pilot-packet.md §2.4 G-D
* docs/specs/growth-advisor-monthly-report.md
* docs/known-debts.md #67 + #49
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path


REPORT_SCHEMA_VERSION = "v0.2"


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
    # Archetype (5 mom archetypes; values are fractions summing to ~1.0
    # when there is at least one active end_user, else may be empty).
    archetype_distribution: dict[str, float]
    # Handoff
    handoff_triggered_count: int
    handoff_completed_count: int
    handoff_p99_seconds: float
    # Protocol-phase cohort (PE-driven onboarding-arc phase, replaces
    # the previous calendar day_cohort_active_counts; phase ids come
    # from BehaviorProtocol.TemporalArc.progression_signals snapshot
    # — typically icebreaker / baseline / empathy / pain_mining /
    # rapport / targeted_advice / summary_hook).
    protocol_phase_cohort_active_counts: dict[str, int]
    # Deletion auditability (debt #49). Never disclose what was
    # deleted — only that deletion happened, and how many end users
    # were affected. Both default to 0 for tenants with no deletion
    # events in the window.
    deleted_end_user_count: int = 0
    deletion_event_count: int = 0

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
        if self.deleted_end_user_count < 0 or self.deletion_event_count < 0:
            raise ValueError("deletion counts must be non-negative")
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
    # Replaces day_cohort_active_counts (calendar-day routing
    # deprecated 2026-05-14). Phase ids match
    # BehaviorProtocol.TemporalArc.progression_signals snapshot.
    protocol_phase_cohort_active_counts: Mapping[str, int]
    # Deletion auditability (debt #49). Defaults to 0 so legacy
    # callers that haven't wired the deletion ledger yet still
    # construct an inputs object successfully.
    deleted_end_user_count: int = 0
    deletion_event_count: int = 0


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
            protocol_phase_cohort_active_counts=dict(
                inputs.protocol_phase_cohort_active_counts
            ),
            deleted_end_user_count=inputs.deleted_end_user_count,
            deletion_event_count=inputs.deletion_event_count,
        )
        self._cache[(inputs.tenant_id, inputs.month_iso)] = snapshot
        return snapshot

    def latest_for(self, tenant_id: str, month_iso: str) -> MonthlyReportSnapshot | None:
        return self._cache.get((tenant_id, month_iso))


@dataclasses.dataclass(frozen=True)
class _RuptureSnapshotShim:
    """Minimal contract surface the builder reads from rupture_state.

    Spec §SSOT: the aggregator never re-derives rupture / repair from
    raw memory — it only consumes the typed snapshot the
    ``vz-cognition.rupture_state`` owner publishes. This shim is the
    documented contract; production passes the real snapshot type.
    """

    rupture_count: int
    repair_count: int


@dataclasses.dataclass(frozen=True)
class _BoundarySnapshotShim:
    """Contract surface for boundary policy owner snapshot.

    ``per_policy_trigger_count`` keys the 4 anti-sales policies
    (``bp-no-hard-sell`` / ``bp-no-overclaim`` / ``bp-no-flooding`` /
    ``bp-no-judgmental``); other keys allowed for forward compat.
    """

    per_policy_trigger_count: Mapping[str, int]


@dataclasses.dataclass(frozen=True)
class _ArchetypeSnapshotShim:
    """Contract surface for ArchetypeStateOwner snapshot.

    ``distribution`` is the per-archetype fraction over end_users
    active in the window; sums to ~1.0 (validated downstream).
    """

    distribution: Mapping[str, float]


@dataclasses.dataclass(frozen=True)
class _HandoffSnapshotShim:
    """Contract surface for HandoffTicketStore.list_for_tenant readout."""

    triggered_count: int
    completed_count: int
    p99_seconds: float


@dataclasses.dataclass(frozen=True)
class _ProtocolPhaseSnapshotShim:
    """Contract surface for BehaviorProtocol.TemporalArc.progression_signals.

    ``active_counts`` keys the onboarding-arc phase ids
    (``icebreaker`` / ``baseline`` / ``empathy`` / ``pain_mining`` /
    ``rapport`` / ``targeted_advice`` / ``summary_hook``).
    """

    active_counts: Mapping[str, int]


class MonthlyReportInputsBuilder:
    """Build :class:`MonthlyReportInputs` from owner snapshots (debt #67).

    Per R8 + spec §SSOT, the builder is a **read-only consumer** of
    typed snapshots published by upstream owners. It does NOT walk
    ``evidence_root_dir`` / scoped memory directly (that would
    duplicate ownership). The 5 snapshot inputs match the 5 owners
    documented in
    ``docs/specs/growth-advisor-monthly-report.md`` §5
    (Aggregation 公式).

    The builder is intentionally stateless so callers can construct
    different inputs for different (tenant, month) pairs without
    side-effects.
    """

    def build_inputs(
        self,
        *,
        tenant_id: str,
        month_iso: str,
        period_start_ms: int,
        period_end_ms: int,
        per_end_user_turn_counts: Mapping[str, int],
        new_end_user_ids: tuple[str, ...],
        rupture: _RuptureSnapshotShim,
        boundary: _BoundarySnapshotShim,
        archetype: _ArchetypeSnapshotShim,
        handoff: _HandoffSnapshotShim,
        protocol_phase: _ProtocolPhaseSnapshotShim,
        deleted_end_user_count: int = 0,
        deletion_event_count: int = 0,
    ) -> MonthlyReportInputs:
        """Read the 5 typed snapshots → frozen MonthlyReportInputs.

        ``deleted_end_user_count`` / ``deletion_event_count`` typically
        come from :func:`count_deletion_events_in_window` reading the
        ``evidence_deletion_ledger.jsonl`` for the same window; the
        builder accepts them as already-counted ints so the contract
        between "ledger reader" and "monthly report aggregator" stays
        a single typed handoff (rather than the aggregator opening
        the ledger itself, which would muddy R8 boundaries).

        ``per_end_user_turn_counts`` and ``new_end_user_ids`` come
        from the lifeform-service session_manager turn-counter; we
        accept them as inputs because session_manager doesn't expose
        a "monthly slice" snapshot directly — extracting that slice
        is the caller's job (typically a small adapter in
        ``lifeform_service.protocol_routes``).
        """

        return MonthlyReportInputs(
            tenant_id=tenant_id,
            month_iso=month_iso,
            period_start_ms=period_start_ms,
            period_end_ms=period_end_ms,
            per_end_user_turn_counts=dict(per_end_user_turn_counts),
            new_end_user_ids=new_end_user_ids,
            boundary_trigger_per_policy=dict(boundary.per_policy_trigger_count),
            rupture_count=rupture.rupture_count,
            repair_count=rupture.repair_count,
            archetype_distribution=dict(archetype.distribution),
            handoff_triggered_count=handoff.triggered_count,
            handoff_completed_count=handoff.completed_count,
            handoff_p99_seconds=handoff.p99_seconds,
            protocol_phase_cohort_active_counts=dict(protocol_phase.active_counts),
            deleted_end_user_count=deleted_end_user_count,
            deletion_event_count=deletion_event_count,
        )


def count_deletion_events_in_window(
    *,
    evidence_root: Path,
    period_start_ms: int,
    period_end_ms: int,
    tenant_id: str | None = None,
) -> tuple[int, int]:
    """Count deletion events + distinct deleted end_users from the ledger.

    Reads ``evidence_deletion_ledger-YYYYMMDD.jsonl`` files under
    ``evidence_root`` and counts entries whose ``timestamp_iso`` falls
    in the ``[period_start_ms, period_end_ms)`` half-open window. When
    ``tenant_id`` is supplied, only entries whose ledger
    ``tenant_id`` matches contribute (single-tenant reports). The
    aggregator never reads the ledger ``deleted_file_sha256_set`` or
    actual evidence files; this preserves the GDPR/PIPL contract:
    "deletion is auditable; deleted content stays deleted".

    Returns ``(deletion_event_count, deleted_end_user_count)``. The
    second figure de-duplicates entries by ``end_user_id`` so a single
    end_user issuing multiple deletes within the window counts once.
    Returns ``(0, 0)`` when ``evidence_root`` does not exist (tenant
    has no recorded deletions yet).
    """

    from datetime import datetime

    root = Path(evidence_root)
    if not root.exists():
        return (0, 0)

    event_count = 0
    deleted_end_users: set[str] = set()
    for ledger_path in sorted(root.glob("evidence_deletion_ledger-*.jsonl")):
        try:
            lines = ledger_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise OSError(
                f"failed to read deletion ledger {ledger_path}: {exc}"
            ) from exc
        for line in lines:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entry_tenant = entry.get("tenant_id")
            if tenant_id is not None and entry_tenant != tenant_id:
                continue
            timestamp_iso = entry.get("timestamp_iso")
            if not timestamp_iso:
                continue
            entry_ms = int(
                datetime.fromisoformat(timestamp_iso).timestamp() * 1000.0
            )
            if not (period_start_ms <= entry_ms < period_end_ms):
                continue
            event_count += 1
            entry_end_user = entry.get("end_user_id")
            if entry_end_user:
                deleted_end_users.add(entry_end_user)
    return (event_count, len(deleted_end_users))


__all__ = (
    "MonthlyReportInputs",
    "MonthlyReportInputsBuilder",
    "MonthlyReportOwner",
    "MonthlyReportSnapshot",
    "REPORT_SCHEMA_VERSION",
    "count_deletion_events_in_window",
    # Snapshot shims (typed contract surface for upstream owner snapshots)
    "_RuptureSnapshotShim",
    "_BoundarySnapshotShim",
    "_ArchetypeSnapshotShim",
    "_HandoffSnapshotShim",
    "_ProtocolPhaseSnapshotShim",
)
