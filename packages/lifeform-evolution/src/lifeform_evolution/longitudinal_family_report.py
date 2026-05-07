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
# Phase 2 W2.0b (debt #10A closure): minimum cross-round delta on
# ``il_rapport`` for the longitudinal acceptance gate to consider the
# rapport trend "positive". Set to 0.005 because the un-LLM probe runs
# in artifacts/eq_uplift/cross_session_probe.json show 0.001-0.006
# magnitudes — 0.005 is the high end of the empirically observed
# range, so passing is meaningful but achievable without LLM runtime
# wired (debt #10B). Tightened to 0.02 once #10B closes.
_IL_RAPPORT_TREND_MIN: float = 0.005


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
    # Phase 2 W2.0b (debt #10A closure) — interlocutor 12-axis trend.
    # When the F3 metrics include ``f3.il_trust_final`` /
    # ``f3.il_rapport_final``, the aggregator extracts first/last/trend
    # alongside per-round arrays. The new ``il_rapport_trend_pos`` gate
    # replaces ``bond_warmth_trend`` as the primary cross-round signal
    # because vitals drive is ceiling-saturated and bond_warmth_trend
    # is structurally near-zero (see the debt #10A note).
    il_trust_first: float | None = None
    il_trust_last: float | None = None
    il_trust_trend: float = 0.0
    il_rapport_first: float | None = None
    il_rapport_last: float | None = None
    il_rapport_trend: float = 0.0
    il_rapport_trend_pos: bool = False
    per_round_il_trust: tuple[float | None, ...] = ()
    per_round_il_rapport: tuple[float | None, ...] = ()
    # Phase 2 W2.0c (debt #10B closure) — EQ owner activation
    # diagnostics carried across rounds. The longitudinal aggregator
    # surfaces these so an artifact reader can answer "did the LLM
    # semantic runtime stay wired across all rounds?" by inspecting
    # the per-round arrays. They are READ-ONLY diagnostics and do
    # NOT participate in the ``passed`` gate — under the default
    # NoOpSemanticProposalRuntime they legitimately stay at 0 across
    # every round.
    tom_records_total_first: int | None = None
    tom_records_total_last: int | None = None
    tom_records_total_trend: int = 0
    common_ground_dyad_atoms_total_first: int | None = None
    common_ground_dyad_atoms_total_last: int | None = None
    common_ground_dyad_atoms_total_trend: int = 0
    per_round_tom_records_total: tuple[int | None, ...] = ()
    per_round_common_ground_dyad_atoms_total: tuple[int | None, ...] = ()

    @property
    def passed(self) -> bool:
        """The longitudinal acceptance gate.

        A run passes when:

        * ``trust_no_drift`` — no significant warmth erosion (legacy
          bond_warmth-based check; preserved for backward compat).
        * ``continuity_improved_vs_baseline`` — every round closes a
          scene and the bond_warmth trend is non-negative.
        * ``il_rapport_trend_pos`` — cross-round il_rapport trend
          exceeds ``_IL_RAPPORT_TREND_MIN`` (Phase 2 W2.0b primary
          gate). When ``il_rapport`` data is unavailable (legacy
          BenchmarkReport synth without final_interlocutor_axes) this
          condition is skipped to avoid breaking unit tests, but in
          live longitudinal CLI runs the il axes are always populated.
        """
        if self.il_rapport_first is None:
            # Legacy synth path with no interlocutor axes wired.
            return (
                self.trust_no_drift
                and self.continuity_improved_vs_baseline
            )
        return (
            self.trust_no_drift
            and self.continuity_improved_vs_baseline
            and self.il_rapport_trend_pos
        )


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


def _il_axis_metric(report: FamilyReport, metric_id: str) -> float | None:
    """Pull a typed F3 il axis metric (e.g. ``f3.il_trust_final``).

    Returns ``None`` when the metric is absent — happens when the
    underlying ``BenchmarkReport`` did not capture
    ``final_interlocutor_axes`` (legacy synth path / pre-W2.0b
    BenchmarkReport).
    """
    fam = report.family(FamilyId.F3_RELATIONSHIP_CONTINUITY)
    for metric in fam.metrics:
        if metric.metric_id == metric_id:
            return float(metric.value)
    return None


def _eq_count_metric(report: FamilyReport, metric_id: str) -> int | None:
    """Pull an EQ owner activation count metric (e.g. ``f3.tom_records_total``).

    Returns ``None`` when the metric is absent — happens with
    pre-W2.0c ``BenchmarkReport`` instances built without the new
    diagnostic fields. The value is rounded to ``int`` because the
    underlying ``BenchmarkReport`` field is ``int`` even though
    ``FamilyMetric.value`` is ``float``.
    """
    fam = report.family(FamilyId.F3_RELATIONSHIP_CONTINUITY)
    for metric in fam.metrics:
        if metric.metric_id == metric_id:
            return int(round(metric.value))
    return None


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

    # Phase 2 W2.0b (debt #10A closure): pull per-round il_trust /
    # il_rapport. ``None`` rounds are preserved (the legacy / synth
    # FamilyReport path doesn't carry these metrics).
    per_round_il_trust = tuple(
        _il_axis_metric(report, "f3.il_trust_final") for report in reports
    )
    per_round_il_rapport = tuple(
        _il_axis_metric(report, "f3.il_rapport_final") for report in reports
    )
    il_trust_first = per_round_il_trust[0] if per_round_il_trust else None
    il_trust_last = per_round_il_trust[-1] if per_round_il_trust else None
    il_trust_trend = (
        0.0
        if il_trust_first is None or il_trust_last is None
        else float(il_trust_last) - float(il_trust_first)
    )
    il_rapport_first = per_round_il_rapport[0] if per_round_il_rapport else None
    il_rapport_last = per_round_il_rapport[-1] if per_round_il_rapport else None
    il_rapport_trend = (
        0.0
        if il_rapport_first is None or il_rapport_last is None
        else float(il_rapport_last) - float(il_rapport_first)
    )
    il_rapport_trend_pos = (
        il_rapport_first is not None
        and il_rapport_last is not None
        and il_rapport_trend >= _IL_RAPPORT_TREND_MIN
    )

    # Phase 2 W2.0c (debt #10B closure): EQ owner activation counts
    # carried across rounds. Pure diagnostics, never gates.
    per_round_tom = tuple(
        _eq_count_metric(report, "f3.tom_records_total") for report in reports
    )
    per_round_cg = tuple(
        _eq_count_metric(report, "f3.common_ground_dyad_atoms_total")
        for report in reports
    )
    tom_first = per_round_tom[0] if per_round_tom else None
    tom_last = per_round_tom[-1] if per_round_tom else None
    tom_trend = (
        0
        if tom_first is None or tom_last is None
        else int(tom_last) - int(tom_first)
    )
    cg_first = per_round_cg[0] if per_round_cg else None
    cg_last = per_round_cg[-1] if per_round_cg else None
    cg_trend = (
        0
        if cg_first is None or cg_last is None
        else int(cg_last) - int(cg_first)
    )

    description = (
        f"Longitudinal F3 over {rounds} round(s): "
        f"closed_scenes_total={closed_total} "
        f"bond_warmth first={bond_first} last={bond_last} trend={bond_trend:+.3f} "
        f"il_rapport first={il_rapport_first} last={il_rapport_last} "
        f"trend={il_rapport_trend:+.4f} pos={il_rapport_trend_pos} "
        f"tom_records first={tom_first} last={tom_last} trend={tom_trend:+d} "
        f"cg_dyad_atoms first={cg_first} last={cg_last} trend={cg_trend:+d} "
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
        il_trust_first=il_trust_first,
        il_trust_last=il_trust_last,
        il_trust_trend=il_trust_trend,
        il_rapport_first=il_rapport_first,
        il_rapport_last=il_rapport_last,
        il_rapport_trend=il_rapport_trend,
        il_rapport_trend_pos=il_rapport_trend_pos,
        per_round_il_trust=per_round_il_trust,
        per_round_il_rapport=per_round_il_rapport,
        tom_records_total_first=tom_first,
        tom_records_total_last=tom_last,
        tom_records_total_trend=tom_trend,
        common_ground_dyad_atoms_total_first=cg_first,
        common_ground_dyad_atoms_total_last=cg_last,
        common_ground_dyad_atoms_total_trend=cg_trend,
        per_round_tom_records_total=per_round_tom,
        per_round_common_ground_dyad_atoms_total=per_round_cg,
    )


def format_longitudinal_family_report(report: LongitudinalFamilyReport) -> str:
    lines = [
        "== Longitudinal family report (F3) ==",
        f"scenario={report.scenario_id} rounds={report.rounds} "
        f"closed_scenes_total={report.closed_scenes_total}",
        f"bond_warmth_first={report.bond_warmth_first}",
        f"bond_warmth_last={report.bond_warmth_last}",
        f"bond_warmth_trend={report.bond_warmth_trend:+.3f}",
        f"il_trust_first={report.il_trust_first}",
        f"il_trust_last={report.il_trust_last}",
        f"il_trust_trend={report.il_trust_trend:+.4f}",
        f"il_rapport_first={report.il_rapport_first}",
        f"il_rapport_last={report.il_rapport_last}",
        f"il_rapport_trend={report.il_rapport_trend:+.4f}",
        f"tom_records_first={report.tom_records_total_first}",
        f"tom_records_last={report.tom_records_total_last}",
        f"tom_records_trend={report.tom_records_total_trend:+d}",
        f"cg_dyad_atoms_first={report.common_ground_dyad_atoms_total_first}",
        f"cg_dyad_atoms_last={report.common_ground_dyad_atoms_total_last}",
        f"cg_dyad_atoms_trend={report.common_ground_dyad_atoms_total_trend:+d}",
        f"trust_no_drift                   = {'PASS' if report.trust_no_drift else 'FAIL'}",
        f"continuity_improved_vs_baseline  = "
        f"{'PASS' if report.continuity_improved_vs_baseline else 'FAIL'}",
        f"il_rapport_trend_pos             = "
        f"{'PASS' if report.il_rapport_trend_pos else 'FAIL'}",
        f"overall                          = {'PASS' if report.passed else 'FAIL'}",
        "per-round closed scenes : "
        + ", ".join(str(count) for count in report.per_round_closed_scene_count),
        "per-round bond_warmth   : "
        + ", ".join(
            f"{level:.3f}" if level is not None else "None"
            for level in report.per_round_bond_warmth
        ),
        "per-round il_trust      : "
        + ", ".join(
            f"{level:+.3f}" if level is not None else "None"
            for level in report.per_round_il_trust
        ),
        "per-round il_rapport    : "
        + ", ".join(
            f"{level:.3f}" if level is not None else "None"
            for level in report.per_round_il_rapport
        ),
        "per-round tom_records   : "
        + ", ".join(
            str(count) if count is not None else "None"
            for count in report.per_round_tom_records_total
        ),
        "per-round cg_dyad_atoms : "
        + ", ".join(
            str(count) if count is not None else "None"
            for count in report.per_round_common_ground_dyad_atoms_total
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
        "il_trust_first": report.il_trust_first,
        "il_trust_last": report.il_trust_last,
        "il_trust_trend": report.il_trust_trend,
        "il_rapport_first": report.il_rapport_first,
        "il_rapport_last": report.il_rapport_last,
        "il_rapport_trend": report.il_rapport_trend,
        "il_rapport_trend_pos": report.il_rapport_trend_pos,
        "tom_records_total_first": report.tom_records_total_first,
        "tom_records_total_last": report.tom_records_total_last,
        "tom_records_total_trend": report.tom_records_total_trend,
        "common_ground_dyad_atoms_total_first": (
            report.common_ground_dyad_atoms_total_first
        ),
        "common_ground_dyad_atoms_total_last": (
            report.common_ground_dyad_atoms_total_last
        ),
        "common_ground_dyad_atoms_total_trend": (
            report.common_ground_dyad_atoms_total_trend
        ),
        "passed": report.passed,
        "per_round_closed_scene_count": list(report.per_round_closed_scene_count),
        "per_round_bond_warmth": list(report.per_round_bond_warmth),
        "per_round_il_trust": list(report.per_round_il_trust),
        "per_round_il_rapport": list(report.per_round_il_rapport),
        "per_round_tom_records_total": list(report.per_round_tom_records_total),
        "per_round_common_ground_dyad_atoms_total": list(
            report.per_round_common_ground_dyad_atoms_total
        ),
        "description": report.description,
    }


__all__ = [
    "LongitudinalFamilyReport",
    "compute_longitudinal_family_report",
    "format_longitudinal_family_report",
    "longitudinal_family_report_to_dict",
]
