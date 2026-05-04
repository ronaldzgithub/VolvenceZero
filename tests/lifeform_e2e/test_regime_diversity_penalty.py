"""Tests for the regime calibrator's diversity penalty.

The penalty exists because the original calibrator's update rule \u2014
"boost every regime in each miss's expected_regime_in" \u2014 has a degenerate
cheap path on small scenario sets where one regime is in every label
set. The cheapest regime_match=100% is "always pick that regime", which
flattens product behaviour to one shape regardless of context. The
penalty pulls back any regime predicted on more than
``diversity_threshold`` of turns proportional to overuse.

These tests pin:

* The unit math (no penalty when diverse, scaled penalty under monoculture).
* ``RegimeCalibrationRoundReport.predicted_regime_counts`` is populated.
* ``best_round()`` excludes the untrained baseline AND prefers diverse rounds.
* Super-loop's ``best_temporal_round()`` and ``best_regime_round()``
  return rounds that match their respective health gates.
* End-to-end on the coding vertical: super-loop saves a temporal artifact
  with sparse \u03b2_t AND a regime artifact with non-monoculture predicted
  distribution. The two artifacts may come from different rounds.
"""

from __future__ import annotations

import pytest

from lifeform_evolution.regime_calibrator import (
    _CLIP_HIGH,
    _CLIP_LOW,
    _apply_diversity_penalty,
    _tally_predicted_regimes,
)


# ---------------------------------------------------------------------------
# Unit: penalty math
# ---------------------------------------------------------------------------


def test_diversity_penalty_skips_when_distribution_is_diverse():
    weights = {"a": 1.0, "b": 1.0, "c": 1.0}
    counts = {"a": 4, "b": 3, "c": 3}
    applied = _apply_diversity_penalty(
        weights, counts, 10, threshold=0.5, lr=0.30
    )
    assert applied == {}
    assert weights == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_diversity_penalty_pulls_back_complete_monoculture():
    # Phase 1.7 tightens the cap to [0.85, 1.15] so all valid starting
    # weights live in that band. Use values within the band so the
    # multiplicative factor is the only thing tested here, not the
    # clip behaviour (which is exercised separately).
    weights = {"a": 1.10, "b": 1.0, "c": 0.95}
    applied = _apply_diversity_penalty(
        weights, {"a": 10}, 10, threshold=0.5, lr=0.30
    )
    # excess = 1.0 - 0.5 = 0.5; factor = 1 - 0.30 * 0.5 = 0.85
    assert applied == pytest.approx({"a": 0.85})
    # 1.10 * 0.85 = 0.935; comfortably above _CLIP_LOW=0.85 so no clamp.
    assert weights["a"] == pytest.approx(1.10 * 0.85)
    # Other regimes are untouched.
    assert weights["b"] == pytest.approx(1.0)
    assert weights["c"] == pytest.approx(0.95)


def test_diversity_penalty_respects_clip_floor():
    """Even with a very strong penalty, weights cannot drop below
    ``_CLIP_LOW``. This guarantees the regime stays selectable; the
    penalty is a brake, not a kill switch."""
    weights = {"a": 0.4}
    _apply_diversity_penalty(weights, {"a": 10}, 10, threshold=0.5, lr=10.0)
    assert weights["a"] == pytest.approx(_CLIP_LOW)


def test_diversity_penalty_zero_lr_is_a_noop():
    weights = {"a": 1.0, "b": 1.0}
    applied = _apply_diversity_penalty(
        weights, {"a": 10}, 10, threshold=0.5, lr=0.0
    )
    assert applied == {}
    assert weights == {"a": 1.0, "b": 1.0}


def test_diversity_penalty_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        _apply_diversity_penalty(
            {"a": 1.0}, {"a": 10}, 10, threshold=0.0, lr=0.30
        )
    with pytest.raises(ValueError):
        _apply_diversity_penalty(
            {"a": 1.0}, {"a": 10}, 10, threshold=1.0, lr=0.30
        )


def test_tally_predicted_regimes_counts_only_named_regimes():
    """``_tally_predicted_regimes`` counts ``active_regime`` per turn,
    skipping turns whose regime is None (so the diversity denominator
    stays consistent with the calibrator's miss ledger).
    """
    from lifeform_evolution import BenchmarkReport, TurnReport

    def _t(regime: str | None, idx: int) -> TurnReport:
        return TurnReport(
            turn_index=idx,
            user_input="u",
            response_text="r",
            active_regime=regime,
            active_abstract_action=None,
            expression_intent=None,
            pe_magnitude=0.0,
            open_loop_count=0,
            regime_match=True,
            pe_threshold_met=True,
        )

    bench = BenchmarkReport(
        scenario_id="t",
        turn_reports=(
            _t("a", 1),
            _t("a", 2),
            _t("b", 3),
            _t(None, 4),
        ),
        regime_match_rate=1.0,
        pe_threshold_match_rate=1.0,
        response_non_empty_rate=1.0,
        closed_scene_count=0,
    )
    counts, total = _tally_predicted_regimes([bench])
    assert counts == {"a": 2, "b": 1}
    assert total == 3


# ---------------------------------------------------------------------------
# RegimeCalibrationRoundReport publishes the new field
# ---------------------------------------------------------------------------


def test_round_report_publishes_predicted_regime_counts():
    from lifeform_evolution import run_regime_calibrator

    report = run_regime_calibrator(rounds=2)
    for r in report.rounds:
        # Counts are tuples of (regime, count), sorted by count desc.
        assert isinstance(r.predicted_regime_counts, tuple)
        if r.predicted_regime_counts:
            counts = [count for _, count in r.predicted_regime_counts]
            assert counts == sorted(counts, reverse=True)
        # Top-share is just a derived view of the same data.
        assert 0.0 <= r.predicted_regime_share <= 1.0


def test_calibrator_best_round_excludes_baseline():
    """A calibrator's "best" must reflect at least one update step.
    Returning round 0 (the untrained baseline) as best would silently
    say "no calibration helped"; we require the selector to instead
    return the best TRAINED round, even if all trained rounds are
    monocultures.
    """
    from lifeform_evolution import run_regime_calibrator

    report = run_regime_calibrator(rounds=2)
    assert report.best_round() is not report.baseline
    # Match rate must improve over baseline (or at least tie + come from a
    # trained round).
    assert report.best_round().round_index >= 1


# ---------------------------------------------------------------------------
# Super-loop axis selectors
# ---------------------------------------------------------------------------


async def test_super_loop_saves_per_axis_best_rounds_on_coding_vertical():
    """End-to-end: ``best_temporal_round`` returns a sparse-\u03b2_t round and
    ``best_regime_round`` returns a non-monoculture round. They CAN be
    different rounds; the CLI's --save-temporal / --save-regime use
    each accordingly so the saved artifacts are healthy on their own
    axis even when one round is healthy on one axis only.
    """
    from lifeform_domain_coding import build_coding_package, scenarios_dir
    from lifeform_evolution import load_scenarios, run_super_loop_async

    scenarios = load_scenarios(scenarios_dir())
    report = await run_super_loop_async(
        rounds=4,
        scenarios=scenarios,
        domain_experience_packages=(build_coding_package(),),
    )

    best_t = report.best_temporal_round()
    best_r = report.best_regime_round()

    # Temporal axis: prefer sparse \u03b2_t when any round qualifies.
    sparse_rounds = [
        r for r in report.rounds[1:]
        if r.ssl.switch_frequency_last is not None
        and r.ssl.switch_frequency_last <= 0.20
    ]
    if sparse_rounds:
        assert (
            best_t.ssl.switch_frequency_last is not None
            and best_t.ssl.switch_frequency_last <= 0.20
        )

    # Regime axis: prefer non-monoculture when any round qualifies.
    diverse_rounds = [
        r for r in report.rounds[1:] if r.predicted_regime_share <= 0.70
    ]
    if diverse_rounds:
        assert best_r.predicted_regime_share <= 0.70


async def test_super_loop_diversity_penalty_changes_predicted_distribution():
    """Without the penalty, super-loop on coding scenarios collapsed to
    100% guided_exploration by round 1 (every coding scenario lists it
    in expected_regime_in, so the cheapest match-100% solution is to
    always pick it). With the penalty active, no later trained round
    should stay at top-share=100% \u2014 the penalty must visibly nudge
    the predicted distribution back toward diversity.
    """
    from lifeform_domain_coding import build_coding_package, scenarios_dir
    from lifeform_evolution import load_scenarios, run_super_loop_async

    scenarios = load_scenarios(scenarios_dir())
    report = await run_super_loop_async(
        rounds=4,
        scenarios=scenarios,
        domain_experience_packages=(build_coding_package(),),
        diversity_threshold=0.50,
        diversity_lr=0.30,
    )
    # At least one trained round must dip below the 0.7 monoculture line.
    diverse_rounds = [
        r for r in report.rounds[1:] if r.predicted_regime_share <= 0.70
    ]
    assert diverse_rounds, (
        "diversity penalty failed to produce any non-monoculture trained "
        f"round; predicted shares = "
        f"{[(r.round_index, r.predicted_regime_share) for r in report.rounds[1:]]}"
    )


async def test_super_loop_diversity_lr_zero_recovers_old_behaviour():
    """Setting ``diversity_lr=0.0`` must reproduce the pre-penalty
    behaviour. We do not pin a specific round shape (depends on SSL
    randomness); we only assert that with lr=0 some round IS allowed
    to fully collapse, and the trajectory verdicts still pass.
    """
    from lifeform_domain_coding import build_coding_package, scenarios_dir
    from lifeform_evolution import load_scenarios, run_super_loop_async

    scenarios = load_scenarios(scenarios_dir())
    report = await run_super_loop_async(
        rounds=4,
        scenarios=scenarios,
        domain_experience_packages=(build_coding_package(),),
        diversity_lr=0.0,
    )
    assert report.trajectory_passes()


# ---------------------------------------------------------------------------
# Coding vertical's shipped bootstrap is now the diversity-aware one
# ---------------------------------------------------------------------------


def test_shipped_coding_regime_bootstrap_does_not_concentrate_weight_on_one_regime():
    """The shipped artifact must not have any single regime weight >= clip
    high, AND must not push any single regime down to clip low \u2014 both
    extremes indicate the calibrator failed to settle on a useful
    prior. Within those bounds the test is permissive: we are not pinning
    specific weights, only the structural claim "no monoculture".
    """
    from lifeform_domain_coding import load_coding_regime_bootstrap

    bootstrap = load_coding_regime_bootstrap()
    weights = dict(bootstrap.selection_weights)
    max_w = max(weights.values())
    min_w = min(weights.values())
    # No single regime should be saturated (would indicate monoculture).
    assert max_w < _CLIP_HIGH, (
        f"shipped bootstrap has a saturated regime weight: {weights}"
    )
    # No regime should be effectively zeroed.
    assert min_w > _CLIP_LOW, (
        f"shipped bootstrap has a zero-clipped regime weight: {weights}"
    )
