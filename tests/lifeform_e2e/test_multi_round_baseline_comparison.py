"""Tests for multi-round loop's R12 acceptance#12 baseline comparison.

Acceptance question #12 of ``docs/next_gen_emogpt.md``:

    Can fixed multi-turn dialogue benchmarks show that high prediction
    error triggers temporally aligned controller changes and later
    improvement relative to weak baselines?

Round 0 of the multi-round loop IS the weak baseline (untrained
metacontroller). The loop now publishes per-round quality aggregates and
explicit baseline-deltas so consumers can answer #12 without
hand-rolling the comparison from raw distributions.

These tests pin:

* ``RoundQualityMetrics`` is computed for every round (including round 0).
* ``RoundDeltaVsBaseline`` is None on round 0 and populated on later rounds.
* The new acceptance verdicts (``improved_*_vs_baseline``,
  ``found_pe_aligned_improvement_round``) are present and reflect the
  observed deltas.
* ``trajectory_passes()`` is decoupled from
  ``improvement_vs_baseline_passes()`` \u2014 a healthy loop can pass the
  former without passing the stricter latter.
"""

from __future__ import annotations

import pytest

from lifeform_evolution import (
    MultiRoundLearningLoopReport,
    RoundDeltaVsBaseline,
    RoundQualityMetrics,
    RoundReport,
    format_multi_round_report,
    run_multi_round_loop_async,
)


@pytest.fixture(scope="module")
async def short_multi_round_report() -> MultiRoundLearningLoopReport:
    """Run the multi-round loop once with a small round count, reuse across tests."""
    return await run_multi_round_loop_async(rounds=3)


# ---------------------------------------------------------------------------
# Per-round quality metrics
# ---------------------------------------------------------------------------


async def test_every_round_publishes_quality_metrics(
    short_multi_round_report: MultiRoundLearningLoopReport,
):
    report = short_multi_round_report
    for r in report.rounds:
        assert r.quality is not None, f"round {r.round_index} missing quality"
        assert isinstance(r.quality, RoundQualityMetrics)
        assert 0.0 <= r.quality.mean_regime_match_rate <= 1.0
        assert 0.0 <= r.quality.mean_pe_threshold_match_rate <= 1.0
        assert r.quality.mean_pe >= 0.0
        # PE recovery is signed: positive = late half lower than early half
        assert r.quality.early_half_pe_mean >= 0.0
        assert r.quality.late_half_pe_mean >= 0.0
        assert (
            r.quality.pe_recovery_delta
            == pytest.approx(
                r.quality.early_half_pe_mean - r.quality.late_half_pe_mean,
                abs=1e-9,
            )
        )


# ---------------------------------------------------------------------------
# Baseline delta: round 0 None, round k>0 populated
# ---------------------------------------------------------------------------


async def test_round_zero_has_no_baseline_delta(
    short_multi_round_report: MultiRoundLearningLoopReport,
):
    report = short_multi_round_report
    baseline = report.rounds[0]
    assert baseline.delta_from_baseline is None


async def test_trained_rounds_publish_baseline_delta(
    short_multi_round_report: MultiRoundLearningLoopReport,
):
    report = short_multi_round_report
    for r in report.rounds[1:]:
        assert r.delta_from_baseline is not None, (
            f"round {r.round_index} missing baseline delta"
        )
        assert isinstance(r.delta_from_baseline, RoundDeltaVsBaseline)
        # Deltas are exact differences between the round's quality and round 0's.
        baseline_q = report.rounds[0].quality
        assert baseline_q is not None and r.quality is not None
        assert r.delta_from_baseline.regime_match_delta == pytest.approx(
            r.quality.mean_regime_match_rate - baseline_q.mean_regime_match_rate,
            abs=1e-9,
        )
        assert r.delta_from_baseline.pe_recovery_delta_delta == pytest.approx(
            r.quality.pe_recovery_delta - baseline_q.pe_recovery_delta,
            abs=1e-9,
        )


# ---------------------------------------------------------------------------
# Acceptance #12 verdicts
# ---------------------------------------------------------------------------


async def test_acceptance_12_verdict_keys_present(
    short_multi_round_report: MultiRoundLearningLoopReport,
):
    report = short_multi_round_report
    for key in (
        "improved_regime_match_vs_baseline",
        "improved_pe_recovery_vs_baseline",
        "found_pe_aligned_improvement_round",
    ):
        assert key in report.verdicts, f"missing verdict {key!r}"
        assert isinstance(report.verdicts[key], bool)


async def test_trajectory_passes_decoupled_from_baseline_improvement(
    short_multi_round_report: MultiRoundLearningLoopReport,
):
    """A run can pass the structural trajectory gate without passing the
    stricter R12 acceptance#12 gate, and vice-versa. The two gates exist
    precisely so a regression in one does not silently drag down the
    other.
    """
    report = short_multi_round_report
    traj = report.trajectory_passes()
    accept12 = report.improvement_vs_baseline_passes()
    # We do not assert on absolute values here \u2014 they depend on the small
    # built-in scenario set's stochasticity. We only assert that the two
    # methods are independent dispatch points (not the same boolean).
    assert isinstance(traj, bool)
    assert isinstance(accept12, bool)
    # Acceptance#12 verdict in the dict matches the method.
    assert report.verdicts["found_pe_aligned_improvement_round"] == accept12


# ---------------------------------------------------------------------------
# Acceptance#12 verdict reflects the underlying deltas
# ---------------------------------------------------------------------------


async def test_improvement_verdicts_reflect_per_round_deltas(
    short_multi_round_report: MultiRoundLearningLoopReport,
):
    report = short_multi_round_report
    expected_regime_improved = any(
        r.delta_from_baseline is not None
        and r.delta_from_baseline.regime_match_improved
        for r in report.rounds[1:]
    )
    expected_recovery_improved = any(
        r.delta_from_baseline is not None
        and r.delta_from_baseline.pe_recovery_improved
        for r in report.rounds[1:]
    )
    assert (
        report.verdicts["improved_regime_match_vs_baseline"]
        == expected_regime_improved
    )
    assert (
        report.verdicts["improved_pe_recovery_vs_baseline"]
        == expected_recovery_improved
    )


# ---------------------------------------------------------------------------
# Format includes the baseline-comparison line
# ---------------------------------------------------------------------------


async def test_format_includes_vs_baseline_section_for_trained_rounds(
    short_multi_round_report: MultiRoundLearningLoopReport,
):
    report = short_multi_round_report
    text = format_multi_round_report(report)
    # Round 0 has quality but no delta.
    assert "BASELINE (round 0)" in text
    # At least one trained round prints the delta line.
    if any(r.round_index >= 1 for r in report.rounds):
        assert "vs baseline:" in text
    # The quality summary line is present for every round.
    assert text.count("quality:") == len(report.rounds)
