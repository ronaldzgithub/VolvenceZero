"""Phase 2 W2.C: cross-session longitudinal aggregator over per-scenario
``FamilyReport``s.

Background
----------

R12 / acceptance question #5 ("does the system carry forward across
scenes?") needs evidence that survives a single
``compute_family_report`` window. Within one scenario the F3
relationship-continuity family already reports closed scene count and
final ``bond_warmth``, but it does not say whether trust eroded across
N independent sessions.

This module aggregates a sequence of per-session ``FamilyReport``
objects into a longitudinal report. The aggregator is **pure**: it
takes frozen per-session reports and emits a frozen longitudinal
report. The CLI (``lifeform-bench --longitudinal-rounds N``) is the
producer of those per-session reports; this file only computes the
aggregate metrics + acceptance gate.

Aggregate F3 metrics
--------------------

* ``closed_scenes_total`` (sum across rounds) — at least one closed
  scene per round means the session-post slow loop fired in every
  session.
* ``bond_warmth_first`` / ``bond_warmth_last`` — companion vertical
  drive at session end, first session vs last session.
* ``bond_warmth_trend`` — last minus first; positive means trust grew.
* ``trust_no_drift`` — gate boolean: ``bond_warmth_trend >= -0.10``.
* ``continuity_improved_vs_baseline`` — gate boolean: trend >= 0 AND
  every round closed at least one scene.

Both gates are conservative: ``trust_no_drift`` allows small noise,
``continuity_improved_vs_baseline`` is the strict version that requires
non-negative trend across all rounds.

The longitudinal report does not invent new F3 metrics; it tracks the
across-session derivative of the existing per-session F3 metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

from lifeform_evolution.family_report import (
    FamilyId,
    FamilyReport,
)


_BOND_WARMTH_DRIFT_TOLERANCE: float = 0.10


@dataclass(frozen=True)
class LongitudinalFamilyReport:
    """Aggregate of N per-session ``FamilyReport``s, focused on F3."""

    rounds: int
    scenario_id: str
    closed_scenes_total: int
    bond_warmth_first: float | None
    bond_warmth_last: float | None
    bond_warmth_trend: float
    trust_no_drift: bool
    continuity_improved_vs_baseline: bool
    per_round_closed_scene_count: tuple[int, ...]
    per_round_bond_warmth: tuple[float | None, ...]
    description: str

    @property
    def passed(self) -> bool:
        """The longitudinal acceptance gate.

        A run passes when both ``trust_no_drift`` (no significant
        warmth erosion) and ``continuity_improved_vs_baseline`` (the
        strict every-round-closed + non-negative trend gate) hold.
        """
        return self.trust_no_drift and self.continuity_improved_vs_baseline


def _bond_warmth_metric(report: FamilyReport) -> float | None:
    """Pull ``f3.bond_warmth_final`` from one ``FamilyReport`` if present.

    Vertical packs that lack a ``bond_warmth`` drive (e.g. coding
    vertical) skip the metric; we forward ``None`` rather than 0 so
    the trend can short-circuit.
    """
    fam = report.family(FamilyId.F3_RELATIONSHIP_CONTINUITY)
    for metric in fam.metrics:
        if metric.metric_id == "f3.bond_warmth_final":
            return float(metric.value)
    return None


def _closed_scene_count(report: FamilyReport) -> int:
    fam = report.family(FamilyId.F3_RELATIONSHIP_CONTINUITY)
    for metric in fam.metrics:
        if metric.metric_id == "f3.closed_scene_count":
            return int(metric.value)
    return 0


def compute_longitudinal_family_report(
    reports: tuple[FamilyReport, ...],
) -> LongitudinalFamilyReport:
    """Aggregate per-session family reports into a longitudinal view.

    ``reports`` MUST be ordered by session/round index (oldest first);
    the aggregator does NOT re-sort. An empty input yields a report
    with both gates ``False`` so the CLI gate fails loudly rather than
    silently passing on no evidence.
    """
    rounds = len(reports)
    scenario_id = reports[0].scenario_id if reports else ""
    per_round_closed = tuple(_closed_scene_count(report) for report in reports)
    per_round_bond = tuple(_bond_warmth_metric(report) for report in reports)
    closed_total = sum(per_round_closed)

    bond_first = per_round_bond[0] if per_round_bond else None
    bond_last = per_round_bond[-1] if per_round_bond else None
    if bond_first is None or bond_last is None:
        bond_trend = 0.0
        trust_no_drift = False
    else:
        bond_trend = float(bond_last) - float(bond_first)
        trust_no_drift = bond_trend >= -_BOND_WARMTH_DRIFT_TOLERANCE

    every_round_closed = rounds > 0 and all(count >= 1 for count in per_round_closed)
    continuity_improved = (
        rounds > 0
        and every_round_closed
        and bond_first is not None
        and bond_last is not None
        and bond_trend >= 0.0
    )

    description = (
        f"Longitudinal F3 over {rounds} round(s): "
        f"closed_scenes_total={closed_total} "
        f"bond_warmth first={bond_first} last={bond_last} trend={bond_trend:+.3f} "
        f"trust_no_drift={trust_no_drift} continuity_improved={continuity_improved}"
    )
    return LongitudinalFamilyReport(
        rounds=rounds,
        scenario_id=scenario_id,
        closed_scenes_total=closed_total,
        bond_warmth_first=bond_first,
        bond_warmth_last=bond_last,
        bond_warmth_trend=bond_trend,
        trust_no_drift=trust_no_drift,
        continuity_improved_vs_baseline=continuity_improved,
        per_round_closed_scene_count=per_round_closed,
        per_round_bond_warmth=per_round_bond,
        description=description,
    )


def format_longitudinal_family_report(report: LongitudinalFamilyReport) -> str:
    lines = [
        "== Longitudinal family report (F3) ==",
        f"scenario={report.scenario_id} rounds={report.rounds} "
        f"closed_scenes_total={report.closed_scenes_total}",
        f"bond_warmth_first={report.bond_warmth_first}",
        f"bond_warmth_last={report.bond_warmth_last}",
        f"bond_warmth_trend={report.bond_warmth_trend:+.3f}",
        f"trust_no_drift                   = {'PASS' if report.trust_no_drift else 'FAIL'}",
        f"continuity_improved_vs_baseline  = "
        f"{'PASS' if report.continuity_improved_vs_baseline else 'FAIL'}",
        f"overall                          = {'PASS' if report.passed else 'FAIL'}",
        "per-round closed scenes : "
        + ", ".join(str(count) for count in report.per_round_closed_scene_count),
        "per-round bond_warmth   : "
        + ", ".join(
            f"{level:.3f}" if level is not None else "None"
            for level in report.per_round_bond_warmth
        ),
    ]
    return "\n".join(lines)


def longitudinal_family_report_to_dict(
    report: LongitudinalFamilyReport,
) -> dict[str, object]:
    return {
        "rounds": report.rounds,
        "scenario_id": report.scenario_id,
        "closed_scenes_total": report.closed_scenes_total,
        "bond_warmth_first": report.bond_warmth_first,
        "bond_warmth_last": report.bond_warmth_last,
        "bond_warmth_trend": report.bond_warmth_trend,
        "trust_no_drift": report.trust_no_drift,
        "continuity_improved_vs_baseline": report.continuity_improved_vs_baseline,
        "passed": report.passed,
        "per_round_closed_scene_count": list(report.per_round_closed_scene_count),
        "per_round_bond_warmth": list(report.per_round_bond_warmth),
        "description": report.description,
    }


__all__ = [
    "LongitudinalFamilyReport",
    "compute_longitudinal_family_report",
    "format_longitudinal_family_report",
    "longitudinal_family_report_to_dict",
]
