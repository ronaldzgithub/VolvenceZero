"""Phase 2 W2.C contract test: longitudinal family-report aggregator.

The aggregator takes N per-session ``FamilyReport`` instances (oldest
first) and emits typed cross-session F3 metrics + an acceptance gate.
This test pins:

1. Empty input fails the gate (no evidence).
2. Trust-no-drift gate fires when warmth erodes beyond tolerance.
3. Continuity-improved gate requires every round to close at least
   one scene AND the warmth trend to be non-negative.
4. Both gates passing yields ``passed=True``.
5. Vertical packs without ``f3.bond_warmth_final`` (e.g. coding) yield
   ``trust_no_drift=False`` (we cannot prove no drift without warmth).
"""

from __future__ import annotations

from lifeform_evolution.family_report import (
    FamilyEvaluation,
    FamilyId,
    FamilyMetric,
    FamilyReport,
)
from lifeform_evolution.longitudinal_family_report import (
    compute_longitudinal_family_report,
    longitudinal_family_report_to_dict,
)


def _f3_eval(
    *,
    closed_scenes: int,
    bond_warmth: float | None,
    il_trust: float | None = None,
    il_rapport: float | None = None,
) -> FamilyEvaluation:
    metrics: list[FamilyMetric] = [
        FamilyMetric(
            metric_id="f3.closed_scene_count",
            name="scenes closed",
            value=float(closed_scenes),
            threshold=1.0,
            higher_is_better=True,
            note="test",
        ),
    ]
    if bond_warmth is not None:
        metrics.append(
            FamilyMetric(
                metric_id="f3.bond_warmth_final",
                name="final bond_warmth level",
                value=float(bond_warmth),
                threshold=0.4,
                higher_is_better=True,
                note="test",
            )
        )
    if il_trust is not None:
        metrics.append(
            FamilyMetric(
                metric_id="f3.il_trust_final",
                name="final il_trust",
                value=float(il_trust),
                threshold=None,
                higher_is_better=True,
                note="test",
            )
        )
    if il_rapport is not None:
        metrics.append(
            FamilyMetric(
                metric_id="f3.il_rapport_final",
                name="final il_rapport",
                value=float(il_rapport),
                threshold=None,
                higher_is_better=True,
                note="test",
            )
        )
    return FamilyEvaluation(
        family_id=FamilyId.F3_RELATIONSHIP_CONTINUITY,
        family_name="relationship continuity",
        metrics=tuple(metrics),
    )


def _stub_family_report(
    *,
    scenario_id: str,
    closed_scenes: int,
    bond_warmth: float | None,
    il_trust: float | None = None,
    il_rapport: float | None = None,
) -> FamilyReport:
    """Build a FamilyReport that only carries the F3 evaluation we need.

    Other family slots are stubbed empty so the aggregator's family
    lookup works. ``family(...)`` raises ``KeyError`` on unknown ids,
    so we still need to populate the F3 entry.
    """
    return FamilyReport(
        scenario_id=scenario_id,
        families=(
            _f3_eval(
                closed_scenes=closed_scenes,
                bond_warmth=bond_warmth,
                il_trust=il_trust,
                il_rapport=il_rapport,
            ),
        ),
    )


def test_empty_input_fails_both_gates() -> None:
    longitudinal = compute_longitudinal_family_report(())
    assert longitudinal.rounds == 0
    assert longitudinal.passed is False
    assert longitudinal.trust_no_drift is False
    assert longitudinal.continuity_improved_vs_baseline is False


def test_warmth_erosion_beyond_tolerance_fails_trust_gate() -> None:
    reports = (
        _stub_family_report(scenario_id="s1", closed_scenes=1, bond_warmth=0.80),
        _stub_family_report(scenario_id="s1", closed_scenes=1, bond_warmth=0.60),
        _stub_family_report(scenario_id="s1", closed_scenes=1, bond_warmth=0.40),
    )
    longitudinal = compute_longitudinal_family_report(reports)
    assert longitudinal.bond_warmth_first == 0.80
    assert longitudinal.bond_warmth_last == 0.40
    assert longitudinal.bond_warmth_trend < -0.10
    assert longitudinal.trust_no_drift is False
    assert longitudinal.continuity_improved_vs_baseline is False
    assert longitudinal.passed is False


def test_warmth_small_dip_within_tolerance_passes_trust_gate_only() -> None:
    """Trend of -0.05 is within the tolerance -> trust_no_drift passes,
    but continuity_improved still fails because trend is negative.
    """
    reports = (
        _stub_family_report(scenario_id="s1", closed_scenes=1, bond_warmth=0.50),
        _stub_family_report(scenario_id="s1", closed_scenes=1, bond_warmth=0.45),
    )
    longitudinal = compute_longitudinal_family_report(reports)
    assert longitudinal.trust_no_drift is True
    assert longitudinal.continuity_improved_vs_baseline is False
    assert longitudinal.passed is False


def test_continuity_improved_requires_every_round_closed_a_scene() -> None:
    reports = (
        _stub_family_report(scenario_id="s1", closed_scenes=1, bond_warmth=0.50),
        _stub_family_report(scenario_id="s1", closed_scenes=0, bond_warmth=0.55),
        _stub_family_report(scenario_id="s1", closed_scenes=1, bond_warmth=0.60),
    )
    longitudinal = compute_longitudinal_family_report(reports)
    assert longitudinal.bond_warmth_trend > 0.0
    assert longitudinal.trust_no_drift is True
    assert longitudinal.continuity_improved_vs_baseline is False  # round 2 closed 0
    assert longitudinal.passed is False


def test_passing_run_satisfies_both_gates() -> None:
    reports = (
        _stub_family_report(scenario_id="s1", closed_scenes=1, bond_warmth=0.50),
        _stub_family_report(scenario_id="s1", closed_scenes=1, bond_warmth=0.55),
        _stub_family_report(scenario_id="s1", closed_scenes=2, bond_warmth=0.62),
    )
    longitudinal = compute_longitudinal_family_report(reports)
    assert longitudinal.closed_scenes_total == 4
    assert longitudinal.bond_warmth_trend > 0.0
    assert longitudinal.trust_no_drift is True
    assert longitudinal.continuity_improved_vs_baseline is True
    assert longitudinal.passed is True


def test_no_bond_warmth_drives_fails_trust_gate() -> None:
    """Vertical packs without a bond_warmth drive (coding vertical)
    cannot prove trust_no_drift; we explicitly fail the gate rather
    than silently passing on missing evidence.
    """
    reports = (
        _stub_family_report(scenario_id="coding-1", closed_scenes=1, bond_warmth=None),
        _stub_family_report(scenario_id="coding-1", closed_scenes=1, bond_warmth=None),
    )
    longitudinal = compute_longitudinal_family_report(reports)
    assert longitudinal.trust_no_drift is False
    assert longitudinal.continuity_improved_vs_baseline is False
    assert longitudinal.passed is False


def test_to_dict_round_trips_typed_fields() -> None:
    reports = (
        _stub_family_report(scenario_id="s1", closed_scenes=1, bond_warmth=0.50),
        _stub_family_report(scenario_id="s1", closed_scenes=1, bond_warmth=0.55),
    )
    longitudinal = compute_longitudinal_family_report(reports)
    d = longitudinal_family_report_to_dict(longitudinal)
    assert d["rounds"] == 2
    assert d["scenario_id"] == "s1"
    assert d["bond_warmth_first"] == 0.50
    assert d["bond_warmth_last"] == 0.55
    assert d["passed"] is longitudinal.passed


# ---------------------------------------------------------------------------
# Phase 2 W2.0b (debt #10A closure) — il_trust / il_rapport aggregation
# ---------------------------------------------------------------------------


def test_il_axes_absent_falls_back_to_legacy_gate() -> None:
    """When per-round FamilyReport lacks ``f3.il_*_final`` metrics
    (legacy synth path / pre-W2.0b BenchmarkReport), the aggregator
    sets the il fields to ``None`` / ``0`` and ``passed`` falls back
    to the legacy bond_warmth-only gate.
    """
    reports = (
        _stub_family_report(scenario_id="s1", closed_scenes=1, bond_warmth=0.50),
        _stub_family_report(scenario_id="s1", closed_scenes=1, bond_warmth=0.55),
    )
    longitudinal = compute_longitudinal_family_report(reports)
    assert longitudinal.il_rapport_first is None
    assert longitudinal.il_trust_first is None
    assert longitudinal.il_rapport_trend == 0.0
    assert longitudinal.il_trust_trend == 0.0
    assert longitudinal.il_rapport_trend_pos is False
    # Legacy fallback: with il axes absent, gate ignores
    # il_rapport_trend_pos and only checks bond_warmth + closed.
    assert longitudinal.passed is True


def test_il_rapport_trend_pos_blocks_passed_when_below_threshold() -> None:
    """il_rapport trend below the W2.0b minimum (0.005) blocks the
    new acceptance gate even though bond_warmth + closed gates pass.
    """
    reports = (
        _stub_family_report(
            scenario_id="s1",
            closed_scenes=1,
            bond_warmth=0.50,
            il_trust=0.10,
            il_rapport=0.80,
        ),
        _stub_family_report(
            scenario_id="s1",
            closed_scenes=1,
            bond_warmth=0.55,
            il_trust=0.10,
            il_rapport=0.801,  # +0.001 trend, below 0.005 threshold
        ),
    )
    longitudinal = compute_longitudinal_family_report(reports)
    assert longitudinal.il_rapport_first == 0.80
    assert abs(longitudinal.il_rapport_trend - 0.001) < 1e-6
    assert longitudinal.il_rapport_trend_pos is False
    assert longitudinal.trust_no_drift is True
    assert longitudinal.continuity_improved_vs_baseline is True
    # il_rapport_trend_pos is the new primary gate when il axes are
    # populated; failing it blocks passed even with the other two
    # gates green.
    assert longitudinal.passed is False


def test_il_rapport_trend_pos_passes_when_above_threshold() -> None:
    """When all three gates fire green ``passed`` is True."""
    reports = (
        _stub_family_report(
            scenario_id="s1",
            closed_scenes=1,
            bond_warmth=0.50,
            il_trust=0.10,
            il_rapport=0.80,
        ),
        _stub_family_report(
            scenario_id="s1",
            closed_scenes=1,
            bond_warmth=0.55,
            il_trust=0.105,
            il_rapport=0.81,  # +0.010 trend, above 0.005 threshold
        ),
        _stub_family_report(
            scenario_id="s1",
            closed_scenes=2,
            bond_warmth=0.60,
            il_trust=0.115,
            il_rapport=0.82,
        ),
    )
    longitudinal = compute_longitudinal_family_report(reports)
    assert longitudinal.il_rapport_first == 0.80
    assert longitudinal.il_rapport_last == 0.82
    assert abs(longitudinal.il_rapport_trend - 0.02) < 1e-6
    assert longitudinal.il_rapport_trend_pos is True
    assert longitudinal.passed is True


def test_il_axes_dict_round_trips() -> None:
    reports = (
        _stub_family_report(
            scenario_id="s1",
            closed_scenes=1,
            bond_warmth=0.50,
            il_trust=0.10,
            il_rapport=0.80,
        ),
        _stub_family_report(
            scenario_id="s1",
            closed_scenes=1,
            bond_warmth=0.55,
            il_trust=0.115,
            il_rapport=0.81,
        ),
    )
    longitudinal = compute_longitudinal_family_report(reports)
    d = longitudinal_family_report_to_dict(longitudinal)
    assert d["il_trust_first"] == 0.10
    assert d["il_rapport_first"] == 0.80
    assert d["il_rapport_last"] == 0.81
    assert abs(d["il_rapport_trend"] - 0.01) < 1e-6
    assert d["il_rapport_trend_pos"] is True
    assert d["per_round_il_rapport"] == [0.80, 0.81]
    assert d["per_round_il_trust"] == [0.10, 0.115]
    assert d["per_round_closed_scene_count"] == [1, 1]
    assert d["per_round_bond_warmth"] == [0.50, 0.55]
