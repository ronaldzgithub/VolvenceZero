"""Tests for the R12 six-family evaluation grouping.

These pin the contract that ``compute_family_report`` produces:

* All six families (F1\u2013F6) appear in the report, in canonical order.
* Each family's ``passed`` predicate honours its metric thresholds.
* Family-level ``family_passed`` requires ALL gated metrics to pass.
* Companion-vertical-specific metrics (``f3.bond_warmth_final``) appear
  only when the underlying drive is present in the bench report.
* The JSON serialiser produces a structurally stable artifact.
* The end-to-end ``lifeform-bench`` runner feeds the family computer
  with all the fields it needs (i.e. the new ``temporal_switch_gate`` /
  ``response_length`` / ``final_vitals_*`` plumbing actually works).
"""

from __future__ import annotations

import json

import pytest

from lifeform_evolution import (
    BenchmarkReport,
    FamilyId,
    TurnReport,
    compute_family_report,
    family_report_to_dict,
    format_family_report,
)


def _turn(
    *,
    index: int,
    text: str = "I am responding with enough text to clear the floor.",
    regime: str = "emotional_support",
    intent: str = "support-first",
    pe: float = 0.4,
    open_loop_count: int = 0,
    regime_match: bool = True,
    pe_threshold_met: bool = True,
    abstract_action: str | None = "support_listen",
    switch_gate: float = 0.05,
    refer_out: bool = False,
) -> TurnReport:
    return TurnReport(
        turn_index=index,
        user_input=f"u{index}",
        response_text=text,
        active_regime=regime,
        active_abstract_action=abstract_action,
        expression_intent=intent,
        pe_magnitude=pe,
        open_loop_count=open_loop_count,
        regime_match=regime_match,
        pe_threshold_met=pe_threshold_met,
        temporal_switch_gate=switch_gate,
        continuum_target_position=0.7,
        refer_out_required=refer_out,
        response_length=len(text),
    )


def _bench(
    *,
    turns: tuple[TurnReport, ...],
    regime_match_rate: float = 1.0,
    pe_threshold_match_rate: float = 1.0,
    response_non_empty_rate: float = 1.0,
    closed_scene_count: int = 1,
    final_vitals_total_pe: float = 0.3,
    final_vitals_drive_levels: tuple[tuple[str, float], ...] = (
        ("bond_warmth", 0.55),
        ("user_engagement", 0.7),
    ),
    pending_followup_count: int = 1,
    proactive_followup_count: int = 0,
) -> BenchmarkReport:
    return BenchmarkReport(
        scenario_id="unit-test-scenario",
        turn_reports=turns,
        regime_match_rate=regime_match_rate,
        pe_threshold_match_rate=pe_threshold_match_rate,
        response_non_empty_rate=response_non_empty_rate,
        closed_scene_count=closed_scene_count,
        final_vitals_total_pe=final_vitals_total_pe,
        final_vitals_drive_levels=final_vitals_drive_levels,
        pending_followup_count=pending_followup_count,
        proactive_followup_count=proactive_followup_count,
    )


# ---------------------------------------------------------------------------
# Structural contract
# ---------------------------------------------------------------------------


def test_family_report_includes_all_six_families_in_canonical_order():
    bench = _bench(turns=(_turn(index=1), _turn(index=2)))
    report = compute_family_report(bench=bench)
    ids = [fam.family_id for fam in report.families]
    assert ids == [
        FamilyId.F1_TASK_CAPABILITY,
        FamilyId.F2_INTERACTION_QUALITY,
        FamilyId.F3_RELATIONSHIP_CONTINUITY,
        FamilyId.F4_LEARNING_QUALITY,
        FamilyId.F5_ABSTRACTION_QUALITY,
        FamilyId.F6_SAFETY_BOUNDEDNESS,
    ]


def test_strong_run_passes_every_family():
    turns = tuple(_turn(index=i) for i in range(1, 5))
    bench = _bench(turns=turns)
    report = compute_family_report(bench=bench)
    assert report.overall_passed is True
    for fam in report.families:
        assert fam.family_passed, (
            f"family {fam.family_id.value} did not pass: "
            f"{[(m.metric_id, m.value, m.threshold, m.passed) for m in fam.metrics]}"
        )


# ---------------------------------------------------------------------------
# F1: Task capability
# ---------------------------------------------------------------------------


def test_f1_fails_when_response_is_too_short():
    turns = tuple(_turn(index=i, text="hi") for i in range(1, 4))
    bench = _bench(turns=turns)
    report = compute_family_report(bench=bench, response_length_min=20)
    f1 = report.family(FamilyId.F1_TASK_CAPABILITY)
    short_metric = next(m for m in f1.metrics if m.metric_id == "f1.short_response_rate")
    assert short_metric.value == pytest.approx(1.0)
    assert short_metric.passed is False
    assert f1.family_passed is False


def test_f1_fails_on_empty_response_rate():
    turns = (_turn(index=1), _turn(index=2, text="ok"))
    bench = _bench(
        turns=turns,
        response_non_empty_rate=0.5,  # half the turns produced empty text
    )
    report = compute_family_report(bench=bench, response_length_min=5)
    f1 = report.family(FamilyId.F1_TASK_CAPABILITY)
    assert any(
        m.metric_id == "f1.response_non_empty_rate" and not m.passed
        for m in f1.metrics
    )
    assert f1.family_passed is False


# ---------------------------------------------------------------------------
# F3: relationship continuity \u2014 vertical-specific metric appears only when
# bond_warmth ships in the drive levels
# ---------------------------------------------------------------------------


def test_f3_includes_bond_warmth_metric_only_when_present():
    turns = tuple(_turn(index=i) for i in range(1, 3))
    bench_with = _bench(turns=turns)  # default has bond_warmth
    bench_without = _bench(
        turns=turns,
        final_vitals_drive_levels=(),  # no drives shipped
    )
    with_metrics = compute_family_report(bench=bench_with).family(
        FamilyId.F3_RELATIONSHIP_CONTINUITY
    )
    without_metrics = compute_family_report(bench=bench_without).family(
        FamilyId.F3_RELATIONSHIP_CONTINUITY
    )
    assert any(m.metric_id == "f3.bond_warmth_final" for m in with_metrics.metrics)
    assert all(
        m.metric_id != "f3.bond_warmth_final" for m in without_metrics.metrics
    )


def test_f3_fails_when_no_scene_closed():
    turns = (_turn(index=1),)
    bench = _bench(turns=turns, closed_scene_count=0)
    f3 = compute_family_report(bench=bench).family(FamilyId.F3_RELATIONSHIP_CONTINUITY)
    closed_metric = next(m for m in f3.metrics if m.metric_id == "f3.closed_scene_count")
    assert closed_metric.passed is False
    assert f3.family_passed is False


# ---------------------------------------------------------------------------
# F4: learning quality \u2014 PE recovery
# ---------------------------------------------------------------------------


def test_f4_pe_recovery_is_positive_when_second_half_calmer():
    turns = (
        _turn(index=1, pe=0.8),
        _turn(index=2, pe=0.9),
        _turn(index=3, pe=0.3),
        _turn(index=4, pe=0.2),
    )
    bench = _bench(turns=turns)
    f4 = compute_family_report(bench=bench).family(FamilyId.F4_LEARNING_QUALITY)
    recovery = next(m for m in f4.metrics if m.metric_id == "f4.pe_recovery_delta")
    assert recovery.value > 0.0


def test_f4_pe_recovery_zero_when_second_half_louder():
    turns = (
        _turn(index=1, pe=0.1),
        _turn(index=2, pe=0.2),
        _turn(index=3, pe=0.7),
        _turn(index=4, pe=0.8),
    )
    bench = _bench(turns=turns)
    f4 = compute_family_report(bench=bench).family(FamilyId.F4_LEARNING_QUALITY)
    recovery = next(m for m in f4.metrics if m.metric_id == "f4.pe_recovery_delta")
    assert recovery.value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# F5: abstraction quality \u2014 distinct actions / persistence / switch gate
# ---------------------------------------------------------------------------


def test_f5_distinct_actions_counts_unique_named_actions():
    turns = (
        _turn(index=1, abstract_action="a"),
        _turn(index=2, abstract_action="a"),
        _turn(index=3, abstract_action="b"),
        _turn(index=4, abstract_action=None),
    )
    bench = _bench(turns=turns)
    f5 = compute_family_report(bench=bench).family(FamilyId.F5_ABSTRACTION_QUALITY)
    distinct = next(
        m for m in f5.metrics if m.metric_id == "f5.distinct_abstract_action_count"
    )
    assert distinct.value == pytest.approx(2.0)


def test_f5_persistence_increases_with_consecutive_same_actions():
    consecutive = tuple(_turn(index=i, abstract_action="a") for i in range(1, 5))
    flapping = (
        _turn(index=1, abstract_action="a"),
        _turn(index=2, abstract_action="b"),
        _turn(index=3, abstract_action="a"),
        _turn(index=4, abstract_action="b"),
    )
    cons_persist = next(
        m
        for m in compute_family_report(bench=_bench(turns=consecutive))
        .family(FamilyId.F5_ABSTRACTION_QUALITY)
        .metrics
        if m.metric_id == "f5.mean_action_persistence"
    )
    flap_persist = next(
        m
        for m in compute_family_report(bench=_bench(turns=flapping))
        .family(FamilyId.F5_ABSTRACTION_QUALITY)
        .metrics
        if m.metric_id == "f5.mean_action_persistence"
    )
    assert cons_persist.value > flap_persist.value


# ---------------------------------------------------------------------------
# F6: safety \u2014 runaway response length
# ---------------------------------------------------------------------------


def test_f6_runaway_rate_fails_on_oversize_response():
    long_text = "x" * 1500
    turns = (_turn(index=1, text=long_text), _turn(index=2))
    bench = _bench(turns=turns)
    f6 = compute_family_report(bench=bench, response_length_max=1200).family(
        FamilyId.F6_SAFETY_BOUNDEDNESS
    )
    runaway = next(
        m for m in f6.metrics if m.metric_id == "f6.response_runaway_rate"
    )
    assert runaway.value > 0.0
    assert runaway.passed is False
    assert f6.family_passed is False


# ---------------------------------------------------------------------------
# JSON / formatter
# ---------------------------------------------------------------------------


def test_family_report_to_dict_roundtrips_through_json():
    bench = _bench(turns=(_turn(index=1), _turn(index=2)))
    report = compute_family_report(bench=bench)
    payload = family_report_to_dict(report)
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["scenario_id"] == "unit-test-scenario"
    family_ids = [fam["family_id"] for fam in decoded["families"]]
    assert family_ids == ["F1", "F2", "F3", "F4", "F5", "F6"]


def test_format_family_report_includes_all_family_lines():
    bench = _bench(turns=(_turn(index=1),))
    report = compute_family_report(bench=bench)
    text = format_family_report(report)
    for fid in ("F1", "F2", "F3", "F4", "F5", "F6"):
        assert f"[{fid}]" in text
    assert "PASS" in text or "FAIL" in text


# ---------------------------------------------------------------------------
# End-to-end \u2014 the actual benchmark runner populates the new fields
# ---------------------------------------------------------------------------


async def test_run_benchmark_populates_fields_required_by_family_report():
    """The runner produces every field that ``compute_family_report`` reads.

    Two paths are exercised:

    * Default path (``run_benchmark_async()`` with no ``lifeform``) \u2014
      vertical-agnostic, no vitals wired. Family report still works but
      the F3 ``bond_warmth_final`` metric is absent.
    * Companion path (``lifeform=build_companion_lifeform()``) \u2014 vitals
      ship and ``final_vitals_drive_levels`` is populated end-to-end.
    """
    from lifeform_domain_emogpt import build_companion_lifeform
    from lifeform_evolution import (
        run_benchmark_async,
        low_mood_disclosure_scenario,
    )

    bench_default = await run_benchmark_async(scenario=low_mood_disclosure_scenario())
    assert any(t.response_length > 0 for t in bench_default.turn_reports)
    default_report = compute_family_report(bench=bench_default)
    assert len(default_report.families) == 6
    # F3 has at least the closed_scene_count + pending_followup_count metrics
    # even without vitals.
    f3_default = default_report.family(FamilyId.F3_RELATIONSHIP_CONTINUITY)
    metric_ids = {m.metric_id for m in f3_default.metrics}
    assert "f3.closed_scene_count" in metric_ids
    assert "f3.pending_followup_count" in metric_ids

    # Companion path: build a fully-configured lifeform and feed it in.
    companion = build_companion_lifeform()
    bench_companion = await run_benchmark_async(
        scenario=low_mood_disclosure_scenario(),
        lifeform=companion,
    )
    assert bench_companion.final_vitals_drive_levels, (
        "companion vertical ships vitals; running it through the benchmark "
        "should populate final_vitals_drive_levels"
    )
    drive_names = {name for name, _level in bench_companion.final_vitals_drive_levels}
    assert "bond_warmth" in drive_names
    companion_report = compute_family_report(bench=bench_companion)
    f3_companion = companion_report.family(FamilyId.F3_RELATIONSHIP_CONTINUITY)
    assert any(
        m.metric_id == "f3.bond_warmth_final" for m in f3_companion.metrics
    )
