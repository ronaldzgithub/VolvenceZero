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
from typing import Callable, ClassVar

from volvence_zero.application import DomainExperiencePackage
from volvence_zero.regime import REGIME_TEMPLATES, RegimeBootstrap

from lifeform_core import Lifeform, LifeformConfig
from volvence_zero.substrate import SubstrateAdapter
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
    strategy_priors: tuple[tuple[str, float], ...]
    feature_weights: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    scenario_pass_rate: float
    regime_bootstrap: RegimeBootstrap
    benchmark_reports: tuple[BenchmarkReport, ...]
    # Sorted descending by count. Lets the diversity penalty's effect be
    # audited round-by-round: when one regime dominates, the next round's
    # weight should drop accordingly.
    predicted_regime_counts: tuple[tuple[str, int], ...] = ()
    selected_candidate_label: str = ""

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
        2. Highest scenario-level pass rate within the chosen pool.
        3. Highest turn-level ``regime_match_rate``.
        4. Ties broken by latest round (more update steps applied).

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
            if (
                r.scenario_pass_rate,
                r.regime_match_rate,
                r.round_index,
            ) >= (
                best.scenario_pass_rate,
                best.regime_match_rate,
                best.round_index,
            ):
                best = r
        return best


@dataclass(frozen=True)
class _CalibrationCandidate:
    label: str
    selection_weights: dict[str, float]
    strategy_priors: dict[str, float]
    feature_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class _CandidateEvaluation:
    candidate: _CalibrationCandidate
    benchmark_reports: tuple[BenchmarkReport, ...]
    regime_match_rate: float
    scenario_pass_rate: float
    misclassified_turn_count: int
    predicted_regime_counts: tuple[tuple[str, int], ...]
    misses: tuple[tuple[str, str, tuple[tuple[str, float], ...]], ...]

    @property
    def predicted_regime_share(self) -> float:
        if not self.predicted_regime_counts:
            return 0.0
        total = sum(count for _, count in self.predicted_regime_counts)
        if total <= 0:
            return 0.0
        return self.predicted_regime_counts[0][1] / total


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
    lifeform_factory: Callable[[RegimeBootstrap], Lifeform] | None = None,
    substrate_adapter_factory: Callable[[str, int], SubstrateAdapter] | None = None,
    diversity_threshold: float = _DEFAULT_DIVERSITY_THRESHOLD,
    diversity_lr: float = _DEFAULT_DIVERSITY_LR,
    search_candidate_limit: int = 12,
) -> RegimeCalibrationReport:
    """Calibrate regime selection_weights against scripted ``expected_regime_in``.

    ``domain_experience_packages`` lets a different vertical drive the
    calibration with its own ``DomainExperiencePackage``. Defaults to
    the companion pack so existing call sites keep their behaviour.

    ``lifeform_factory`` is the most faithful path for vertical-owned
    calibration: it receives the round's candidate ``RegimeBootstrap`` and
    must return a fresh Lifeform with that bootstrap wired into the same
    substrate / temporal / vitals setup used by production benchmarks.
    When set it takes precedence over ``domain_experience_packages`` and
    ``substrate_adapter_factory``.

    ``substrate_adapter_factory`` lets a vertical calibrate against the
    same fallback feature surface it uses at runtime. Without this hook
    the calibrator sees a different substrate contract than the benchmark
    path and learns weights for the wrong evidence surface.

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

    ``search_candidate_limit`` bounds the coordinate-search fanout per
    round. The calibrator optimizes scenario-level pass rate directly,
    but it must stay cheap enough for local and CI runs.
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

    # Per-regime running weights, mutated across rounds. Same defaults as
    # ``RegimeModule.__init__``.
    weights: dict[str, float] = {
        template.regime_id: 1.0 for template in REGIME_TEMPLATES
    }
    strategy_priors: dict[str, float] = {
        template.regime_id: 0.0 for template in REGIME_TEMPLATES
    }
    feature_weights: dict[str, dict[str, float]] = {
        template.regime_id: {} for template in REGIME_TEMPLATES
    }

    round_reports: list[RegimeCalibrationRoundReport] = []
    for round_index in range(rounds):
        current = _CalibrationCandidate(
            label="current",
            selection_weights=_copy_weights(weights),
            strategy_priors=_copy_priors(strategy_priors),
            feature_weights=_copy_feature_weights(feature_weights),
        )
        current_eval = await _evaluate_candidate(
            candidate=current,
            scenarios=chosen,
            base_config=base_config,
            lifeform_factory=lifeform_factory,
            substrate_adapter_factory=substrate_adapter_factory,
        )

        round_reports.append(
            RegimeCalibrationRoundReport(
                round_index=round_index,
                regime_match_rate=current_eval.regime_match_rate,
                misclassified_turn_count=current_eval.misclassified_turn_count,
                selection_weights=tuple(sorted(weights.items())),
                strategy_priors=tuple(sorted(strategy_priors.items())),
                feature_weights=_freeze_feature_weights(feature_weights),
                scenario_pass_rate=current_eval.scenario_pass_rate,
                regime_bootstrap=_build_bootstrap_from_weights(
                    weights,
                    strategy_priors=strategy_priors,
                    feature_weights=feature_weights,
                ),
                benchmark_reports=current_eval.benchmark_reports,
                predicted_regime_counts=current_eval.predicted_regime_counts,
                selected_candidate_label="",
            )
        )

        # Select the next round's bootstrap via a small coordinate search.
        # Candidate evaluation, not the miss ledger itself, decides what
        # survives.
        if round_index < rounds - 1:
            candidates = _generate_search_candidates(
                current=current,
                misses=list(current_eval.misses),
                predicted_counts=dict(current_eval.predicted_regime_counts),
                predicted_total=sum(
                    count for _, count in current_eval.predicted_regime_counts
                ),
                learning_rate=learning_rate,
                diversity_threshold=diversity_threshold,
                diversity_lr=diversity_lr,
                limit=max(1, search_candidate_limit),
            )
            candidate_evals = [
                await _evaluate_candidate(
                    candidate=candidate,
                    scenarios=chosen,
                    base_config=base_config,
                    lifeform_factory=lifeform_factory,
                    substrate_adapter_factory=substrate_adapter_factory,
                )
                for candidate in candidates
            ]
            selected = _select_best_evaluation((current_eval, *candidate_evals))
            weights = _copy_weights(selected.candidate.selection_weights)
            strategy_priors = _copy_priors(selected.candidate.strategy_priors)
            feature_weights = _copy_feature_weights(selected.candidate.feature_weights)
            last = round_reports[-1]
            round_reports[-1] = RegimeCalibrationRoundReport(
                round_index=last.round_index,
                regime_match_rate=last.regime_match_rate,
                misclassified_turn_count=last.misclassified_turn_count,
                selection_weights=last.selection_weights,
                strategy_priors=last.strategy_priors,
                feature_weights=last.feature_weights,
                scenario_pass_rate=last.scenario_pass_rate,
                regime_bootstrap=last.regime_bootstrap,
                benchmark_reports=last.benchmark_reports,
                predicted_regime_counts=last.predicted_regime_counts,
                selected_candidate_label=selected.candidate.label,
            )

    provisional_report = RegimeCalibrationReport(
        scenarios=tuple(s.scenario_id for s in chosen),
        rounds=tuple(round_reports),
        final_bootstrap=round_reports[-1].regime_bootstrap,
        description="",
    )
    best_round = provisional_report.best_round()
    final_bootstrap = _replace_bootstrap_description(
        best_round.regime_bootstrap,
        description=(
            f"Regime calibrator: {len(chosen)} scenarios x {rounds} rounds, "
            f"selected round {best_round.round_index}, "
            f"scenario_pass_rate={best_round.scenario_pass_rate:.0%}, "
            f"regime_match_rate={best_round.regime_match_rate:.0%}"
        ),
    )

    description = (
        f"Regime calibrator: {len(chosen)} scenarios x {rounds} rounds  "
        f"baseline match {round_reports[0].regime_match_rate:.0%} \u2192 "
        f"selected round {best_round.round_index} "
        f"({best_round.scenario_pass_rate:.0%} scenario pass, "
        f"{best_round.regime_match_rate:.0%} turn match)"
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
    lifeform_factory: Callable[[RegimeBootstrap], Lifeform] | None = None,
    substrate_adapter_factory: Callable[[str, int], SubstrateAdapter] | None = None,
    diversity_threshold: float = _DEFAULT_DIVERSITY_THRESHOLD,
    diversity_lr: float = _DEFAULT_DIVERSITY_LR,
    search_candidate_limit: int = 12,
) -> RegimeCalibrationReport:
    """Sync wrapper."""
    import asyncio

    return asyncio.run(
        run_regime_calibrator_async(
            rounds=rounds,
            scenarios=scenarios,
            learning_rate=learning_rate,
            domain_experience_packages=domain_experience_packages,
            lifeform_factory=lifeform_factory,
            substrate_adapter_factory=substrate_adapter_factory,
            diversity_threshold=diversity_threshold,
            diversity_lr=diversity_lr,
            search_candidate_limit=search_candidate_limit,
        )
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _copy_weights(weights: dict[str, float]) -> dict[str, float]:
    return dict(weights)


def _copy_priors(priors: dict[str, float]) -> dict[str, float]:
    return dict(priors)


def _copy_feature_weights(
    feature_weights: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {regime_id: dict(weights) for regime_id, weights in feature_weights.items()}


def _build_bootstrap_from_weights(
    weights: dict[str, float],
    *,
    strategy_priors: dict[str, float] | None = None,
    feature_weights: dict[str, dict[str, float]] | None = None,
    description: str = "",
) -> RegimeBootstrap:
    return RegimeBootstrap(
        selection_weights=tuple(sorted(weights.items())),
        strategy_priors=tuple(sorted((strategy_priors or {}).items())),
        feature_weights=_freeze_feature_weights(feature_weights or {}),
        description=description,
    )


def _replace_bootstrap_description(
    bootstrap: RegimeBootstrap,
    *,
    description: str,
) -> RegimeBootstrap:
    return RegimeBootstrap(
        selection_weights=bootstrap.selection_weights,
        historical_effectiveness=bootstrap.historical_effectiveness,
        strategy_priors=bootstrap.strategy_priors,
        feature_weights=getattr(bootstrap, "feature_weights", ()),
        description=description,
    )


def _freeze_feature_weights(
    feature_weights: dict[str, dict[str, float]],
) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
    return tuple(
        (regime_id, tuple(sorted(weights.items())))
        for regime_id, weights in sorted(feature_weights.items())
        if weights
    )


async def _evaluate_candidate(
    *,
    candidate: _CalibrationCandidate,
    scenarios: tuple[ScriptedScenario, ...],
    base_config: LifeformConfig,
    lifeform_factory: Callable[[RegimeBootstrap], Lifeform] | None,
    substrate_adapter_factory: Callable[[str, int], SubstrateAdapter] | None,
) -> _CandidateEvaluation:
    bootstrap = _build_bootstrap_from_weights(
        candidate.selection_weights,
        strategy_priors=candidate.strategy_priors,
        feature_weights=candidate.feature_weights,
    )
    if lifeform_factory is not None:
        lifeform = lifeform_factory(bootstrap)
    else:
        lifeform = Lifeform(
            base_config,
            regime_bootstrap=bootstrap,
            substrate_adapter_factory=substrate_adapter_factory,
        )

    bench_reports: list[BenchmarkReport] = []
    for scenario in scenarios:
        bench_reports.append(
            await _run_scenario_with_lifeform(scenario=scenario, lifeform=lifeform)
        )
    match_count, total_labelled, misses = _tally_matches(scenarios, bench_reports)
    predicted_counts, _predicted_total = _tally_predicted_regimes(bench_reports)
    scenario_pass_rate = sum(1 for report in bench_reports if report.passed()) / max(
        len(bench_reports), 1
    )
    return _CandidateEvaluation(
        candidate=candidate,
        benchmark_reports=tuple(bench_reports),
        regime_match_rate=match_count / max(total_labelled, 1),
        scenario_pass_rate=scenario_pass_rate,
        misclassified_turn_count=len(misses),
        predicted_regime_counts=tuple(
            sorted(predicted_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        misses=tuple(misses),
    )


def _select_best_evaluation(
    evaluations: tuple[_CandidateEvaluation, ...],
) -> _CandidateEvaluation:
    best = evaluations[0]
    for evaluation in evaluations[1:]:
        key = (
            evaluation.scenario_pass_rate,
            evaluation.regime_match_rate,
            -evaluation.predicted_regime_share,
        )
        best_key = (
            best.scenario_pass_rate,
            best.regime_match_rate,
            -best.predicted_regime_share,
        )
        if key > best_key:
            best = evaluation
    return best


def _generate_search_candidates(
    *,
    current: _CalibrationCandidate,
    misses: list[tuple[str, str, tuple[tuple[str, float], ...]]],
    predicted_counts: dict[str, int],
    predicted_total: int,
    learning_rate: float,
    diversity_threshold: float,
    diversity_lr: float,
    limit: int,
) -> tuple[_CalibrationCandidate, ...]:
    candidates: list[_CalibrationCandidate] = []

    selection_weights = _copy_weights(current.selection_weights)
    _apply_updates_in_place(selection_weights, misses, lr=learning_rate)
    _apply_diversity_penalty(
        selection_weights,
        predicted_counts,
        predicted_total,
        threshold=diversity_threshold,
        lr=diversity_lr,
    )
    candidates.append(
        _CalibrationCandidate(
            label="selection-ledger",
            selection_weights=selection_weights,
            strategy_priors=_copy_priors(current.strategy_priors),
            feature_weights=_copy_feature_weights(current.feature_weights),
        )
    )

    priors = _copy_priors(current.strategy_priors)
    _apply_prior_updates_in_place(priors, misses, lr=learning_rate * 0.25)
    candidates.append(
        _CalibrationCandidate(
            label="strategy-ledger",
            selection_weights=_copy_weights(current.selection_weights),
            strategy_priors=priors,
            feature_weights=_copy_feature_weights(current.feature_weights),
        )
    )

    feature_weights = _copy_feature_weights(current.feature_weights)
    _apply_feature_updates_in_place(
        feature_weights,
        misses,
        lr=learning_rate * 0.35,
    )
    candidates.append(
        _CalibrationCandidate(
            label="feature-ledger",
            selection_weights=_copy_weights(current.selection_weights),
            strategy_priors=_copy_priors(current.strategy_priors),
            feature_weights=feature_weights,
        )
    )

    combined_selection = _copy_weights(selection_weights)
    combined_priors = _copy_priors(priors)
    combined_features = _copy_feature_weights(feature_weights)
    candidates.append(
        _CalibrationCandidate(
            label="combined-ledger",
            selection_weights=combined_selection,
            strategy_priors=combined_priors,
            feature_weights=combined_features,
        )
    )

    for label, candidate in _coordinate_candidates(
        current=current,
        misses=misses,
        learning_rate=learning_rate,
    ):
        candidates.append(candidate)
        if len(candidates) >= limit:
            break

    unique: list[_CalibrationCandidate] = []
    seen: set[
        tuple[
            tuple[tuple[str, float], ...],
            tuple[tuple[str, float], ...],
            tuple[tuple[str, tuple[tuple[str, float], ...]], ...],
        ]
    ] = set()
    for candidate in candidates:
        fingerprint = (
            tuple(sorted(candidate.selection_weights.items())),
            tuple(sorted(candidate.strategy_priors.items())),
            _freeze_feature_weights(candidate.feature_weights),
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(candidate)
        if len(unique) >= limit:
            break
    return tuple(unique)


def _coordinate_candidates(
    *,
    current: _CalibrationCandidate,
    misses: list[tuple[str, str, tuple[tuple[str, float], ...]]],
    learning_rate: float,
) -> tuple[tuple[str, _CalibrationCandidate], ...]:
    regime_gradients: dict[str, float] = {}
    feature_gradients: dict[tuple[str, str], float] = {}
    for actual_regime, expected_csv, features in misses:
        expected_set = tuple(part for part in expected_csv.split(",") if part)
        for regime in expected_set:
            regime_gradients[regime] = regime_gradients.get(regime, 0.0) + 1.0
            for feature_name, value in features:
                feature_gradients[(regime, feature_name)] = (
                    feature_gradients.get((regime, feature_name), 0.0) + value
                )
        if actual_regime:
            regime_gradients[actual_regime] = regime_gradients.get(actual_regime, 0.0) - 0.5
            for feature_name, value in features:
                feature_gradients[(actual_regime, feature_name)] = (
                    feature_gradients.get((actual_regime, feature_name), 0.0) - value * 0.5
                )

    candidates: list[tuple[str, _CalibrationCandidate]] = []
    top_regime_gradients = tuple(
        sorted(regime_gradients.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
    )
    top_feature_gradients = tuple(
        sorted(feature_gradients.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
    )

    for regime, gradient in top_regime_gradients:
        candidates.append(_selection_coordinate_candidate(current, regime, gradient, learning_rate))
        candidates.append(_prior_coordinate_candidate(current, regime, gradient, learning_rate))

    for (regime, feature_name), gradient in top_feature_gradients:
        candidates.append(
            _feature_coordinate_candidate(
                current,
                regime,
                feature_name,
                gradient,
                learning_rate,
            )
        )

    for regime_gradient, feature_gradient in zip(
        top_regime_gradients[:2],
        top_feature_gradients[:2],
        strict=False,
    ):
        regime, prior_gradient = regime_gradient
        (feature_regime, feature_name), feature_delta = feature_gradient
        priors = _copy_priors(current.strategy_priors)
        prior_delta = learning_rate * 0.25 * (1.0 if prior_gradient > 0.0 else -0.5)
        priors[regime] = max(-0.5, min(0.5, priors.get(regime, 0.0) + prior_delta))
        feature_weights = _copy_feature_weights(current.feature_weights)
        current_value = feature_weights.setdefault(feature_regime, {}).get(feature_name, 0.0)
        weight_delta = learning_rate * 0.35 * (1.0 if feature_delta > 0.0 else -0.5)
        feature_weights[feature_regime][feature_name] = max(
            -0.75, min(0.75, current_value + weight_delta)
        )
        label = (
            f"coord-prior+feature:{regime}:"
            f"{feature_regime}.{feature_name}:"
            f"{'up' if prior_gradient > 0.0 else 'down'}:"
            f"{'up' if feature_delta > 0.0 else 'down'}"
        )
        candidates.append(
            (
                label,
                _CalibrationCandidate(
                    label=label,
                    selection_weights=_copy_weights(current.selection_weights),
                    strategy_priors=priors,
                    feature_weights=feature_weights,
                ),
            )
        )
    return tuple(candidates)


def _selection_coordinate_candidate(
    current: _CalibrationCandidate,
    regime: str,
    gradient: float,
    learning_rate: float,
) -> tuple[str, _CalibrationCandidate]:
    selection = _copy_weights(current.selection_weights)
    factor = (
        1.0 + learning_rate
        if gradient > 0.0
        else 1.0 - learning_rate * _DEFAULT_NEGATIVE_LR_RATIO
    )
    selection[regime] = max(_CLIP_LOW, min(_CLIP_HIGH, selection[regime] * factor))
    label = f"coord-selection:{regime}:{'up' if gradient > 0.0 else 'down'}"
    return (
        label,
        _CalibrationCandidate(
            label=label,
            selection_weights=selection,
            strategy_priors=_copy_priors(current.strategy_priors),
            feature_weights=_copy_feature_weights(current.feature_weights),
        ),
    )


def _prior_coordinate_candidate(
    current: _CalibrationCandidate,
    regime: str,
    gradient: float,
    learning_rate: float,
) -> tuple[str, _CalibrationCandidate]:
    priors = _copy_priors(current.strategy_priors)
    delta = learning_rate * 0.25 * (1.0 if gradient > 0.0 else -0.5)
    priors[regime] = max(-0.5, min(0.5, priors.get(regime, 0.0) + delta))
    label = f"coord-prior:{regime}:{'up' if gradient > 0.0 else 'down'}"
    return (
        label,
        _CalibrationCandidate(
            label=label,
            selection_weights=_copy_weights(current.selection_weights),
            strategy_priors=priors,
            feature_weights=_copy_feature_weights(current.feature_weights),
        ),
    )


def _feature_coordinate_candidate(
    current: _CalibrationCandidate,
    regime: str,
    feature_name: str,
    gradient: float,
    learning_rate: float,
) -> tuple[str, _CalibrationCandidate]:
    feature_weights = _copy_feature_weights(current.feature_weights)
    current_value = feature_weights.setdefault(regime, {}).get(feature_name, 0.0)
    delta = learning_rate * 0.35 * (1.0 if gradient > 0.0 else -0.5)
    feature_weights[regime][feature_name] = max(
        -0.75, min(0.75, current_value + delta)
    )
    label = f"coord-feature:{regime}.{feature_name}:{'up' if gradient > 0.0 else 'down'}"
    return (
        label,
        _CalibrationCandidate(
            label=label,
            selection_weights=_copy_weights(current.selection_weights),
            strategy_priors=_copy_priors(current.strategy_priors),
            feature_weights=feature_weights,
        ),
    )


def _tally_matches(
    scenarios: tuple[ScriptedScenario, ...],
    bench_reports: list[BenchmarkReport],
) -> tuple[int, int, list[tuple[str, str, tuple[tuple[str, float], ...]]]]:
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
    misses: list[tuple[str, str, tuple[tuple[str, float], ...]]] = []
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
                        _features_from_turn_report(turn_report),
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


_LEARNABLE_FEATURE_NAMES: tuple[str, ...] = (
    "task_pressure",
    "support_presence",
    "repair_pressure",
    "social_pressure",
    "task_dominance",
    "support_dominance",
)


def _features_from_turn_report(turn_report: TurnReport) -> tuple[tuple[str, float], ...]:
    metrics = dict(turn_report.evaluation_metrics)
    task_pressure = metrics.get("task_pressure", 0.0)
    support_presence = metrics.get("support_presence", 0.0)
    repair_pressure = metrics.get("repair_pressure", 0.0)
    social_pressure = metrics.get("social_pressure", 0.0)
    features = {
        "task_pressure": task_pressure,
        "support_presence": support_presence,
        "repair_pressure": repair_pressure,
        "social_pressure": social_pressure,
        "task_dominance": max(task_pressure - support_presence, 0.0),
        "support_dominance": max(support_presence - task_pressure, 0.0),
        "low_pressure": max(1.0 - max(task_pressure, support_presence, repair_pressure), 0.0),
    }
    return tuple((name, features[name]) for name in _LEARNABLE_FEATURE_NAMES)


def _apply_updates_in_place(
    weights: dict[str, float],
    misses: list[tuple[str, str, tuple[tuple[str, float], ...]]],
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
    for actual_regime, expected_csv, _features in misses:
        expected_set = tuple(part for part in expected_csv.split(",") if part)
        for regime in expected_set:
            if regime in weights:
                weights[regime] = max(_CLIP_LOW, min(_CLIP_HIGH, weights[regime] * (1.0 + lr)))
        if actual_regime and actual_regime in weights:
            weights[actual_regime] = max(
                _CLIP_LOW, min(_CLIP_HIGH, weights[actual_regime] * (1.0 - neg_lr))
            )


def _apply_feature_updates_in_place(
    feature_weights: dict[str, dict[str, float]],
    misses: list[tuple[str, str, tuple[tuple[str, float], ...]]],
    *,
    lr: float,
) -> None:
    if lr <= 0.0:
        return
    neg_lr = lr * _DEFAULT_NEGATIVE_LR_RATIO
    for actual_regime, expected_csv, features in misses:
        expected_set = tuple(part for part in expected_csv.split(",") if part)
        for feature_name, value in features:
            if value <= 0.0:
                continue
            for regime in expected_set:
                if regime in feature_weights:
                    current = feature_weights[regime].get(feature_name, 0.0)
                    feature_weights[regime][feature_name] = max(
                        -0.75, min(0.75, current + lr * value)
                    )
            if actual_regime and actual_regime in feature_weights:
                current = feature_weights[actual_regime].get(feature_name, 0.0)
                feature_weights[actual_regime][feature_name] = max(
                    -0.75, min(0.75, current - neg_lr * value)
                )


def _apply_prior_updates_in_place(
    priors: dict[str, float],
    misses: list[tuple[str, str, tuple[tuple[str, float], ...]]],
    *,
    lr: float,
) -> None:
    if lr <= 0.0:
        return
    neg_lr = lr * _DEFAULT_NEGATIVE_LR_RATIO
    for actual_regime, expected_csv, _features in misses:
        expected_set = tuple(part for part in expected_csv.split(",") if part)
        for regime in expected_set:
            if regime in priors:
                priors[regime] = max(-0.5, min(0.5, priors[regime] + lr))
        if actual_regime and actual_regime in priors:
            priors[actual_regime] = max(
                -0.5,
                min(0.5, priors[actual_regime] - neg_lr),
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
            f"scenario_pass_rate: {r.scenario_pass_rate:.0%}  "
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
        nonzero_priors = tuple(
            (regime, prior) for regime, prior in r.strategy_priors if abs(prior) > 1e-9
        )
        if nonzero_priors:
            lines.append(
                "      strategy_priors: "
                + ", ".join(f"{regime}={prior:+.2f}" for regime, prior in nonzero_priors)
            )
        if r.feature_weights:
            lines.append(
                "      feature_weights: "
                + "; ".join(
                    f"{regime}("
                    + ", ".join(f"{feature}={weight:.2f}" for feature, weight in entries)
                    + ")"
                    for regime, entries in r.feature_weights
                )
            )
        if r.selected_candidate_label:
            lines.append(f"      selected_next: {r.selected_candidate_label}")
    best = report.best_round()
    lines.append(
        f"   best round: round {best.round_index}  "
        f"scenario_pass_rate={best.scenario_pass_rate:.0%}  "
        f"regime_match_rate={best.regime_match_rate:.0%}"
    )
    return "\n".join(lines)
