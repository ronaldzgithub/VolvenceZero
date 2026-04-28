"""Trace-driven regime selection-weights calibrator.

Real-learning analogue of the metacontroller SSL pipeline, applied to the
regime classifier. Workflow:

1. Run scripted scenarios through a Lifeform; for each turn, the kernel's
   ``RegimeModule`` predicts an active regime.
2. Each scenario turn carries an ``expected_regime_in: tuple[str, ...]``
   set of acceptable answers (loaded from the JSON scenario pack).
3. For every misclassified turn, push a small fixed-rate update on
   ``RegimeModule._selection_weights`` \u2014 scale UP the expected regimes,
   scale DOWN the wrongly-predicted one. Multiple passes converge.
4. Pack the final weights into a ``RegimeBootstrap`` artifact, persist
   via ``regime_io.save_regime_bootstrap``, inject into a fresh ``Lifeform``
   with ``temporal_bootstrap=...`` (well, ``regime_bootstrap=...``).

Why a separate fast learner instead of the kernel's online
``_selection_weight_lr = 0.02`` drift:

* The kernel's online rule fires on **delayed credit attributions** which
  require completed sessions and aggregated outcomes. That is correct for
  long-horizon regime tuning but slow to demonstrate behavioural shift on
  scripted benchmarks.
* This calibrator is a **scripted-supervision** layer that uses the
  benchmark's own labels (``expected_regime_in``) as a teacher signal.
  It never bypasses the kernel's online learning; it just produces a
  bootstrap that lifts us out of the flat-weights cold-start basin.

Invariants:
* The calibrator never mutates a Lifeform that is in use \u2014 it only writes
  to its own staging dictionary, then publishes a frozen
  ``RegimeBootstrap``.
* Selection weights are clipped to ``[0.3, 2.0]`` (same range as the
  kernel's online rule), so a runaway calibration cannot push any regime
  out of the candidate set entirely.
* If a turn's ``expected_regime_in`` is empty, the turn is treated as
  unlabelled and skipped (no update applied).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from volvence_zero.regime import REGIME_TEMPLATES, RegimeBootstrap

from lifeform_core import Lifeform, LifeformConfig
from lifeform_domain_emogpt import build_companion_package

from lifeform_evolution.benchmark import (
    BenchmarkReport,
    ScriptedScenario,
    all_built_in_scenarios,
)
from lifeform_evolution.learning_loop import _run_scenario_with_lifeform


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeCalibrationRoundReport:
    round_index: int
    regime_match_rate: float
    misclassified_turn_count: int
    selection_weights: tuple[tuple[str, float], ...]
    benchmark_reports: tuple[BenchmarkReport, ...]


@dataclass(frozen=True)
class RegimeCalibrationReport:
    scenarios: tuple[str, ...]
    rounds: tuple[RegimeCalibrationRoundReport, ...]
    final_bootstrap: RegimeBootstrap
    description: str = ""

    @property
    def baseline(self) -> RegimeCalibrationRoundReport:
        return self.rounds[0]

    @property
    def final(self) -> RegimeCalibrationRoundReport:
        return self.rounds[-1]

    def best_round(self) -> RegimeCalibrationRoundReport:
        """Return the round with the highest regime match rate.

        Ties broken by *latest* round, on the assumption that later rounds
        ran on more updated weights and are thus more representative of
        the calibrated state.
        """
        best = self.rounds[0]
        for r in self.rounds[1:]:
            if r.regime_match_rate >= best.regime_match_rate:
                best = r
        return best


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


_DEFAULT_LR = 0.18
_DEFAULT_NEGATIVE_LR_RATIO = 0.5
_CLIP_LOW = 0.3
_CLIP_HIGH = 2.0


async def run_regime_calibrator_async(
    *,
    rounds: int = 4,
    scenarios: tuple[ScriptedScenario, ...] | None = None,
    learning_rate: float = _DEFAULT_LR,
) -> RegimeCalibrationReport:
    if rounds < 2:
        raise ValueError(
            "Regime calibrator needs at least 2 rounds (baseline + at least 1 update)."
        )
    chosen = scenarios or all_built_in_scenarios()

    base_config = LifeformConfig().with_domain_experience(
        (build_companion_package(),)
    )
    from dataclasses import replace as _replace
    base_config = _replace(
        base_config,
        brain_config=_replace(base_config.brain_config, rare_heavy_enabled=False),
    )

    # Per-regime running weight, mutated across rounds. Same defaults as
    # ``RegimeModule.__init__``.
    weights: dict[str, float] = {
        template.regime_id: 1.0 for template in REGIME_TEMPLATES
    }

    round_reports: list[RegimeCalibrationRoundReport] = []
    for round_index in range(rounds):
        bootstrap = _build_bootstrap_from_weights(weights)
        lifeform = Lifeform(base_config, regime_bootstrap=bootstrap)

        bench_reports: list[BenchmarkReport] = []
        for scenario in chosen:
            bench_reports.append(
                await _run_scenario_with_lifeform(
                    scenario=scenario, lifeform=lifeform
                )
            )

        match_count, total_labelled, misses = _tally_matches(chosen, bench_reports)
        match_rate = match_count / max(total_labelled, 1)

        round_reports.append(
            RegimeCalibrationRoundReport(
                round_index=round_index,
                regime_match_rate=match_rate,
                misclassified_turn_count=len(misses),
                selection_weights=tuple(sorted(weights.items())),
                benchmark_reports=tuple(bench_reports),
            )
        )

        # Apply updates only between rounds, so the last round's weights are
        # the ones in the final bootstrap.
        if round_index < rounds - 1:
            _apply_updates_in_place(weights, misses, lr=learning_rate)

    final_bootstrap = _build_bootstrap_from_weights(
        weights,
        description=(
            f"Regime calibrator: {len(chosen)} scenarios x {rounds} rounds, "
            f"final regime_match_rate={round_reports[-1].regime_match_rate:.0%}"
        ),
    )

    description = (
        f"Regime calibrator: {len(chosen)} scenarios x {rounds} rounds  "
        f"baseline match {round_reports[0].regime_match_rate:.0%} \u2192 "
        f"final {round_reports[-1].regime_match_rate:.0%}"
    )

    return RegimeCalibrationReport(
        scenarios=tuple(s.scenario_id for s in chosen),
        rounds=tuple(round_reports),
        final_bootstrap=final_bootstrap,
        description=description,
    )


def run_regime_calibrator(
    *,
    rounds: int = 4,
    scenarios: tuple[ScriptedScenario, ...] | None = None,
    learning_rate: float = _DEFAULT_LR,
) -> RegimeCalibrationReport:
    """Sync wrapper."""
    import asyncio

    return asyncio.run(
        run_regime_calibrator_async(
            rounds=rounds,
            scenarios=scenarios,
            learning_rate=learning_rate,
        )
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _build_bootstrap_from_weights(
    weights: dict[str, float], *, description: str = ""
) -> RegimeBootstrap:
    return RegimeBootstrap(
        selection_weights=tuple(sorted(weights.items())),
        description=description,
    )


def _tally_matches(
    scenarios: tuple[ScriptedScenario, ...],
    bench_reports: list[BenchmarkReport],
) -> tuple[int, int, list[tuple[str, str]]]:
    """Count matched / unmatched labelled turns.

    Returns ``(match_count, labelled_total, misses)`` where ``misses`` is a
    list of ``(actual_regime, expected_regime_in_csv)`` pairs \u2014 enough info
    for the update step to push weights toward the expected set and away
    from the wrongly-predicted regime.
    """
    if len(scenarios) != len(bench_reports):
        raise RuntimeError("scenario / benchmark report length mismatch")
    match_count = 0
    labelled_total = 0
    misses: list[tuple[str, str]] = []
    for scenario, bench in zip(scenarios, bench_reports):
        for scripted_turn, turn_report in zip(scenario.turns, bench.turn_reports):
            expected = tuple(scripted_turn.expected_regime_in)
            if not expected:
                continue
            labelled_total += 1
            if turn_report.active_regime in expected:
                match_count += 1
            else:
                misses.append(
                    (
                        turn_report.active_regime or "",
                        ",".join(expected),
                    )
                )
    return match_count, labelled_total, misses


def _apply_updates_in_place(
    weights: dict[str, float],
    misses: list[tuple[str, str]],
    *,
    lr: float,
) -> None:
    """Scale up expected regimes, scale down the wrongly-predicted one.

    Multiplicative update like the kernel's online rule, but with a much
    higher learning rate (~0.18 vs 0.02) because we have explicit teacher
    labels rather than delayed credit signals.
    """
    if lr <= 0.0:
        return
    neg_lr = lr * _DEFAULT_NEGATIVE_LR_RATIO
    for actual_regime, expected_csv in misses:
        expected_set = tuple(part for part in expected_csv.split(",") if part)
        for regime in expected_set:
            if regime in weights:
                weights[regime] = max(_CLIP_LOW, min(_CLIP_HIGH, weights[regime] * (1.0 + lr)))
        if actual_regime and actual_regime in weights:
            weights[actual_regime] = max(
                _CLIP_LOW, min(_CLIP_HIGH, weights[actual_regime] * (1.0 - neg_lr))
            )


def format_regime_calibration_report(report: RegimeCalibrationReport) -> str:
    lines: list[str] = []
    lines.append(
        f"== Regime calibrator: {len(report.rounds)} rounds x "
        f"{len(report.scenarios)} scenarios =="
    )
    for r in report.rounds:
        tag = "BASELINE (round 0)" if r.round_index == 0 else f"ROUND {r.round_index}"
        lines.append(f"   --- {tag} ---")
        lines.append(
            f"      regime_match_rate: {r.regime_match_rate:.0%}  "
            f"misclassified: {r.misclassified_turn_count}"
        )
        lines.append(
            "      weights: "
            + ", ".join(f"{regime}={weight:.2f}" for regime, weight in r.selection_weights)
        )
    best = report.best_round()
    lines.append(
        f"   best round: round {best.round_index}  "
        f"regime_match_rate={best.regime_match_rate:.0%}"
    )
    return "\n".join(lines)
