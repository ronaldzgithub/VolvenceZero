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
* Selection weights are clipped to ``[0.85, 1.15]`` (same range as the
  kernel's online rule, post Gap 9 phase 1.7). The narrow cap keeps
  ``selection_weights`` a soft prior on top of the per-turn
  ``base_score`` rather than a winner-takes-all hard switch. The
  previous ``[0.3, 2.0]`` cap let any weight >= ~1.5 dominate every
  prompt because ``base_score * weight`` clamped at 1.0; that produced
  the monoculture observed in phase 1.5 / 1.6 calibration.
* If a turn's ``expected_regime_in`` is empty, the turn is treated as
  unlabelled and skipped (no update applied).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from volvence_zero.application import DomainExperiencePackage
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
    # Sorted descending by count. Lets the diversity penalty's effect be
    # audited round-by-round: when one regime dominates, the next round's
    # weight should drop accordingly.
    predicted_regime_counts: tuple[tuple[str, int], ...] = ()

    @property
    def predicted_regime_share(self) -> float:
        """Fraction of turns claimed by the most-predicted regime.

        ``1.0`` means complete monoculture, ``0.0`` means no turns were
        labelled with a regime at all (which only happens on a degenerate
        run). For a 6-regime kernel a perfectly diverse run hits 1/6.
        """
        if not self.predicted_regime_counts:
            return 0.0
        total = sum(count for _, count in self.predicted_regime_counts)
        if total <= 0:
            return 0.0
        return self.predicted_regime_counts[0][1] / total


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

    # Diversity gate aligned with ``SuperLoopReport.best_round``: prefer
    # rounds whose predicted regime distribution does not collapse to a
    # single regime, even at the cost of a lower regime_match rate.
    _DIVERSITY_MAX_TOP_SHARE: ClassVar[float] = 0.70

    def best_round(self) -> RegimeCalibrationRoundReport:
        """Return the calibrator's best **trained** round.

        Round 0 is the untrained baseline and is always excluded \u2014 a
        "best" round must reflect at least one update step. Within the
        trained rounds:

        1. Prefer **diverse** rounds (top-share \u2264 ``_DIVERSITY_MAX_TOP_SHARE``).
        2. Highest ``regime_match_rate`` within the chosen pool.
        3. Ties broken by latest round (more update steps applied).

        Falls back to the trained-round set even when none qualify as
        diverse \u2014 a monoculture trained round is still preferable to
        returning the untrained baseline as "best".
        """
        trained = list(self.rounds[1:]) if len(self.rounds) > 1 else list(self.rounds)
        diverse = [
            r for r in trained
            if r.predicted_regime_share <= self._DIVERSITY_MAX_TOP_SHARE
        ]
        pool = diverse or trained
        best = pool[0]
        for r in pool[1:]:
            if r.regime_match_rate >= best.regime_match_rate:
                best = r
        return best


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


_DEFAULT_LR = 0.18
_DEFAULT_NEGATIVE_LR_RATIO = 0.5
_DEFAULT_DIVERSITY_THRESHOLD = 0.50
_DEFAULT_DIVERSITY_LR = 0.30
_CLIP_LOW = 0.85
_CLIP_HIGH = 1.15


async def run_regime_calibrator_async(
    *,
    rounds: int = 4,
    scenarios: tuple[ScriptedScenario, ...] | None = None,
    learning_rate: float = _DEFAULT_LR,
    domain_experience_packages: tuple[DomainExperiencePackage, ...] | None = None,
    diversity_threshold: float = _DEFAULT_DIVERSITY_THRESHOLD,
    diversity_lr: float = _DEFAULT_DIVERSITY_LR,
) -> RegimeCalibrationReport:
    """Calibrate regime selection_weights against scripted ``expected_regime_in``.

    ``domain_experience_packages`` lets a different vertical drive the
    calibration with its own ``DomainExperiencePackage``. Defaults to
    the companion pack so existing call sites keep their behaviour.

    ``diversity_threshold`` and ``diversity_lr`` parameterise the
    anti-monoculture penalty (see ``_apply_diversity_penalty``). With a
    small scenario pack where one regime appears in every
    ``expected_regime_in``, the bare update rule converges to "always
    pick that regime" \u2014 technically a 100% match rate but pragmatically
    a single-regime collapse that flattens product behaviour. The
    diversity penalty kicks in when the predicted distribution is
    dominated (>``diversity_threshold`` of turns by one regime) and pulls
    the over-represented regime's weight back proportional to overuse.
    Set ``diversity_lr=0.0`` to recover the pre-2026-04-29 behaviour.
    """
    if rounds < 2:
        raise ValueError(
            "Regime calibrator needs at least 2 rounds (baseline + at least 1 update)."
        )
    chosen = scenarios or all_built_in_scenarios()
    if domain_experience_packages is None:
        domain_experience_packages = (build_companion_package(),)

    base_config = LifeformConfig().with_domain_experience(
        domain_experience_packages
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
        predicted_counts, predicted_total = _tally_predicted_regimes(bench_reports)

        round_reports.append(
            RegimeCalibrationRoundReport(
                round_index=round_index,
                regime_match_rate=match_rate,
                misclassified_turn_count=len(misses),
                selection_weights=tuple(sorted(weights.items())),
                benchmark_reports=tuple(bench_reports),
                predicted_regime_counts=tuple(
                    sorted(predicted_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                ),
            )
        )

        # Apply updates only between rounds, so the last round's weights are
        # the ones in the final bootstrap.
        if round_index < rounds - 1:
            _apply_updates_in_place(weights, misses, lr=learning_rate)
            _apply_diversity_penalty(
                weights,
                predicted_counts,
                predicted_total,
                threshold=diversity_threshold,
                lr=diversity_lr,
            )

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
    domain_experience_packages: tuple[DomainExperiencePackage, ...] | None = None,
    diversity_threshold: float = _DEFAULT_DIVERSITY_THRESHOLD,
    diversity_lr: float = _DEFAULT_DIVERSITY_LR,
) -> RegimeCalibrationReport:
    """Sync wrapper."""
    import asyncio

    return asyncio.run(
        run_regime_calibrator_async(
            rounds=rounds,
            scenarios=scenarios,
            learning_rate=learning_rate,
            domain_experience_packages=domain_experience_packages,
            diversity_threshold=diversity_threshold,
            diversity_lr=diversity_lr,
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


def _tally_predicted_regimes(
    bench_reports: list[BenchmarkReport],
) -> tuple[dict[str, int], int]:
    """Return ``(predicted_regime_counts, labelled_total)`` over all turns.

    Used by ``_apply_diversity_penalty`` to detect monoculture: when one
    regime is predicted on more than ``diversity_threshold`` of turns,
    its weight is pulled back so subsequent rounds explore alternatives.
    """
    counts: dict[str, int] = {}
    total = 0
    for bench in bench_reports:
        for turn_report in bench.turn_reports:
            regime = turn_report.active_regime
            if regime is None:
                continue
            total += 1
            counts[regime] = counts.get(regime, 0) + 1
    return counts, total


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


def _apply_diversity_penalty(
    weights: dict[str, float],
    predicted_counts: dict[str, int],
    total_turns: int,
    *,
    threshold: float,
    lr: float,
) -> dict[str, float]:
    """Down-weight any regime predicted on more than ``threshold`` of turns.

    Why this exists: ``_apply_updates_in_place`` boosts EVERY regime in a
    miss's expected set. With small scenario packs where one regime is in
    every ``expected_regime_in`` (e.g. ``guided_exploration`` in all
    coding scenarios), that regime gets boosted on every miss \u2014 and the
    cheapest path to "100% match" is to collapse to that regime. The
    regime classifier is technically right ("guided_exploration is in
    every allowed set"), but the lifeform's responses become indistinguishable
    across scenarios, defeating the point of having multiple regimes.

    The diversity penalty fixes this by detecting monoculture in the
    PREDICTED distribution (independent of the misses ledger) and pulling
    over-represented regimes' weights back proportional to
    ``(predicted_share - threshold)``. When predictions are already
    diverse, the penalty is zero.

    Returns the actual penalties applied (regime \u2192 multiplicative factor)
    so the round report can audit which regimes got pulled back. Mutates
    ``weights`` in place.
    """
    if total_turns <= 0 or lr <= 0.0:
        return {}
    if not 0.0 < threshold < 1.0:
        raise ValueError(
            f"diversity threshold must be in (0, 1), got {threshold!r}"
        )
    applied: dict[str, float] = {}
    for regime, count in predicted_counts.items():
        if regime not in weights:
            continue
        share = count / total_turns
        excess = share - threshold
        if excess <= 0.0:
            continue
        # excess in (0, 1 - threshold]; lr scales it linearly into a
        # multiplicative pull-back. lr=0.10, threshold=0.5, share=1.0
        # \u2192 factor 1 - 0.10 * 0.5 = 0.95 (5% reduction per round).
        factor = max(0.0, 1.0 - lr * excess)
        new_weight = max(_CLIP_LOW, min(_CLIP_HIGH, weights[regime] * factor))
        applied[regime] = factor
        weights[regime] = new_weight
    return applied


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
            f"misclassified: {r.misclassified_turn_count}  "
            f"top-regime share: {r.predicted_regime_share:.0%}"
        )
        if r.predicted_regime_counts:
            lines.append(
                "      predicted: "
                + ", ".join(
                    f"{regime}={count}"
                    for regime, count in r.predicted_regime_counts
                )
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
