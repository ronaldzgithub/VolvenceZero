"""Super-loop: jointly calibrate temporal (\u03b2_t / z_t) and regime axes.

Two axes have been learnable independently for a while:

* ``MetacontrollerSSLTrainer`` finds sparse temporal abstractions
  (\u03b2_t) from substrate residuals (R3 / R4).
* ``run_regime_calibrator`` learns regime selection_weights from
  scripted ``expected_regime_in`` labels (R14 surface).

The super-loop runs them in lockstep so each round trains BOTH axes on
the same scenario surface, and the next round uses the freshly-updated
state of BOTH bootstraps. Behaviourally this is the smallest version of
"NL multi-timescale learning" we can demonstrate: temporal abstraction
and regime identity co-evolving against the same evidence.

Per round:

1. **Eval pass** \u2014 run scenarios on a fresh lifeform built with the
   *current* (temporal_bootstrap, regime_bootstrap). Capture
   ``DistributionSnapshot`` for the report.
2. **Trace pass** \u2014 collect SSL training traces with the same bootstraps
   so the next round's metacontroller sees the regime-calibrated
   behaviour.
3. **SSL train** \u2014 mutate the shared ``FullLearnedTemporalPolicy`` in
   place; export new metacontroller snapshot.
4. **Regime calibration** \u2014 tally misclassified turns from the eval
   pass; nudge ``selection_weights`` per ``regime_calibrator`` rules;
   build a fresh ``RegimeBootstrap``.

Verdicts are "shape-of-trajectory" (same discipline as
``multi_round_loop``): we assert that *some* round produces
simultaneously sparse \u03b2_t AND meaningful regime drift, not that the
final round monotonically beats baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

from volvence_zero.application import DomainExperiencePackage
from volvence_zero.regime import REGIME_TEMPLATES, RegimeBootstrap
from volvence_zero.temporal import (
    FullLearnedTemporalPolicy,
    MetacontrollerParameterSnapshot,
)

from lifeform_core import Lifeform, LifeformConfig
from lifeform_domain_emogpt import build_companion_package

from lifeform_evolution.benchmark import (
    BenchmarkReport,
    ScriptedScenario,
    all_built_in_scenarios,
)
from lifeform_evolution.dataset_adapter import trace_records_to_training_dataset
from lifeform_evolution.learning_loop import (
    DistributionSnapshot,
    _run_scenario_with_lifeform,
    _summarise_reports,
)
from lifeform_evolution.regime_calibrator import (
    _apply_diversity_penalty,
    _apply_updates_in_place,
    _tally_matches,
    _tally_predicted_regimes,
)
from lifeform_evolution.ssl_demo import SSLDemoReport, run_ssl_demo
from lifeform_evolution.trace_collector import TraceCollector


# ---------------------------------------------------------------------------
# Per-round + top-level reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SuperLoopRoundReport:
    """Both axes' state after one round."""

    round_index: int
    distribution: DistributionSnapshot
    ssl: SSLDemoReport
    regime_match_rate: float
    misclassified_turn_count: int
    selection_weights: tuple[tuple[str, float], ...]
    temporal_snapshot: MetacontrollerParameterSnapshot
    regime_bootstrap: RegimeBootstrap
    distance_to_baseline: float
    predicted_regime_counts: tuple[tuple[str, int], ...] = ()

    @property
    def predicted_regime_share(self) -> float:
        if not self.predicted_regime_counts:
            return 0.0
        total = sum(count for _, count in self.predicted_regime_counts)
        if total <= 0:
            return 0.0
        return self.predicted_regime_counts[0][1] / total


@dataclass(frozen=True)
class SuperLoopReport:
    scenarios: tuple[str, ...]
    rounds: tuple[SuperLoopRoundReport, ...]
    final_temporal_snapshot: MetacontrollerParameterSnapshot
    final_regime_bootstrap: RegimeBootstrap
    verdicts: dict[str, bool] = field(default_factory=dict)
    description: str = ""

    @property
    def baseline(self) -> SuperLoopRoundReport:
        return self.rounds[0]

    @property
    def final(self) -> SuperLoopRoundReport:
        return self.rounds[-1]

    def trajectory_passes(self) -> bool:
        return all(self.verdicts.values())

    # Above this share, a round is treated as a regime-collapsed run \u2014
    # technically high regime_match but the lifeform's responses become
    # indistinguishable across scenarios. Diversity is the higher-priority
    # gate when picking the best round.
    _DIVERSITY_MAX_TOP_SHARE: ClassVar[float] = 0.70
    # Below this gate is "sparse \u03b2_t" \u2014 healthy ETA temporal abstraction.
    _SPARSE_BETA_THRESHOLD: ClassVar[float] = 0.20

    def best_round(self) -> SuperLoopRoundReport:
        """Round that best evidences the design intent.

        Two independent gates: **diverse** (multi-regime predicted
        distribution) and **sparse** (\u03b2_t fires occasionally rather
        than every step). Selection priority:

        1. **diverse \u2229 sparse** \u2014 ideal: temporal abstraction is healthy
           AND multiple regimes are exercised.
        2. **diverse only** \u2014 we keep regime variety even at the cost of
           saturated \u03b2_t. A monoculture round defeats the point of
           having a regime axis at all.
        3. **sparse only** \u2014 fall back to ETA-healthy rounds when no
           round managed multi-regime predictions.
        4. Any trained round, by highest regime_match_rate.

        Within each pool, ties broken by ``regime_match_rate``.
        """
        trained = list(self.rounds[1:]) or list(self.rounds)

        def _is_sparse(r: SuperLoopRoundReport) -> bool:
            sf = r.ssl.switch_frequency_last
            return sf is not None and sf <= self._SPARSE_BETA_THRESHOLD

        def _is_diverse(r: SuperLoopRoundReport) -> bool:
            return r.predicted_regime_share <= self._DIVERSITY_MAX_TOP_SHARE

        for predicate in (
            lambda r: _is_diverse(r) and _is_sparse(r),
            _is_diverse,
            _is_sparse,
        ):
            pool = [r for r in trained if predicate(r)]
            if pool:
                return max(pool, key=lambda r: r.regime_match_rate)
        return max(trained, key=lambda r: r.regime_match_rate)

    def best_temporal_round(self) -> SuperLoopRoundReport:
        """Best round for the temporal (\u03b2_t / z_t) axis.

        Picks the sparse-\u03b2_t round with the highest regime_match_rate
        (regime quality is a useful tie-break among healthy temporal
        rounds). Falls back to whichever round has the closest-to-target
        switch frequency when no round qualifies as sparse \u2014 measured by
        absolute distance to ``_SPARSE_BETA_THRESHOLD``.

        Used by ``lifeform-super-loop --save-temporal`` so the saved
        temporal artifact reflects the cleanest \u03b2_t state, even when
        regime calibration kept improving in later (over-trained)
        rounds.
        """
        trained = list(self.rounds[1:]) or list(self.rounds)
        sparse = [
            r for r in trained
            if r.ssl.switch_frequency_last is not None
            and r.ssl.switch_frequency_last <= self._SPARSE_BETA_THRESHOLD
        ]
        if sparse:
            return max(sparse, key=lambda r: r.regime_match_rate)
        return min(
            trained,
            key=lambda r: abs(
                (r.ssl.switch_frequency_last or 1.0) - self._SPARSE_BETA_THRESHOLD
            ),
        )

    def best_regime_round(self) -> SuperLoopRoundReport:
        """Best round for the regime axis.

        Picks the most diverse round (predicted top-share \u2264 threshold)
        with the highest regime_match_rate. When all rounds are mono\u00ad
        cultures, falls back to the lowest-top-share round. Used by
        ``lifeform-super-loop --save-regime``.
        """
        trained = list(self.rounds[1:]) or list(self.rounds)
        diverse = [
            r for r in trained
            if r.predicted_regime_share <= self._DIVERSITY_MAX_TOP_SHARE
        ]
        if diverse:
            return max(diverse, key=lambda r: r.regime_match_rate)
        return min(trained, key=lambda r: r.predicted_regime_share)


# ---------------------------------------------------------------------------
# Hellinger-style distance reused from multi_round_loop
# ---------------------------------------------------------------------------


def _distance(a: DistributionSnapshot, b: DistributionSnapshot) -> float:
    keys: set[tuple[str, str]] = set()
    p_total = max(a.turn_count, 1)
    q_total = max(b.turn_count, 1)
    p: dict[tuple[str, str], float] = {}
    q: dict[tuple[str, str], float] = {}
    for label, items in (
        ("regime", a.regime_counts),
        ("intent", a.intent_counts),
        ("action", a.abstract_action_counts),
    ):
        for key, count in items:
            p[(label, key)] = count / p_total
            keys.add((label, key))
    for label, items in (
        ("regime", b.regime_counts),
        ("intent", b.intent_counts),
        ("action", b.abstract_action_counts),
    ):
        for key, count in items:
            q[(label, key)] = count / q_total
            keys.add((label, key))
    summed = 0.0
    for k in keys:
        diff = math.sqrt(p.get(k, 0.0)) - math.sqrt(q.get(k, 0.0))
        summed += diff * diff
    return min(1.0, math.sqrt(summed / (2.0 * 3.0)))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_super_loop_async(
    *,
    rounds: int = 3,
    scenarios: tuple[ScriptedScenario, ...] | None = None,
    n_z: int = 3,
    alpha: float = 0.1,
    regime_learning_rate: float = 0.18,
    diversity_threshold: float = 0.50,
    diversity_lr: float = 0.30,
    domain_experience_packages: tuple[DomainExperiencePackage, ...] | None = None,
    substrate_runtime: object | None = None,
) -> SuperLoopReport:
    """Jointly train (\u03b2_t / z_t) and regime axes over a vertical's scenarios.

    Args:
        rounds: total number of rounds, including round 0 baseline.
        scenarios: scenario set to drive the loop. Defaults to the
            built-in companion scenarios.
        n_z, alpha: SSL hyperparams.
        regime_learning_rate: regime calibrator step size.
        domain_experience_packages: vertical-specific
            ``DomainExperiencePackage`` set baked into every Lifeform
            constructed by the loop. Defaults to the companion package
            so existing call sites do not need to change. A different
            vertical (e.g. ``lifeform-domain-coding``) passes its own
            ``(build_coding_package(),)`` in.
    """
    if rounds < 2:
        raise ValueError(
            "Super loop needs at least 2 rounds (round 0 baseline + at least 1 trained)."
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
    # Slice 2a phase 1.5: when an explicit substrate runtime is supplied
    # (typically real Qwen / HF transformer), force ``substrate_mode``
    # to ``injected`` so all eval lifeforms across rounds share THAT one
    # runtime instance. Loading Qwen weights once per round would burn
    # 30+ seconds of overhead per round otherwise.
    if substrate_runtime is not None:
        base_config = _replace(
            base_config,
            brain_config=_replace(
                base_config.brain_config, substrate_mode="injected"
            ),
        )

    # Persistent state across rounds.
    temporal_policy = FullLearnedTemporalPolicy()
    regime_weights: dict[str, float] = {
        template.regime_id: 1.0 for template in REGIME_TEMPLATES
    }
    regime_priors: dict[str, float] = {
        template.regime_id: 0.0 for template in REGIME_TEMPLATES
    }

    round_reports: list[SuperLoopRoundReport] = []

    for round_index in range(rounds):
        temporal_seed = (
            None if round_index == 0 else round_reports[-1].temporal_snapshot
        )
        regime_bootstrap = _build_regime_bootstrap(
            regime_weights,
            strategy_priors=regime_priors,
        )

        # 1) Eval pass: per-scenario benchmark with the current bootstraps.
        eval_lifeform = Lifeform(
            base_config,
            temporal_bootstrap=temporal_seed,
            regime_bootstrap=regime_bootstrap,
            substrate_runtime=substrate_runtime,
        )
        bench_reports: list[BenchmarkReport] = []
        for scenario in chosen:
            bench_reports.append(
                await _run_scenario_with_lifeform(
                    scenario=scenario, lifeform=eval_lifeform
                )
            )
        distribution = _summarise_reports(tuple(bench_reports))

        match_count, total_labelled, misses = _tally_matches(
            tuple(chosen), bench_reports
        )
        regime_match_rate = match_count / max(total_labelled, 1)
        predicted_counts, predicted_total = _tally_predicted_regimes(
            bench_reports
        )

        # 2) Trace pass: collect SSL training data with the SAME bootstraps,
        #    so the next metacontroller update reflects the post-calibration
        #    behaviour rather than a stale baseline.
        collector = TraceCollector(
            temporal_bootstrap=temporal_seed,
            substrate_runtime=substrate_runtime,
        )
        try:
            for scenario in chosen:
                await collector.collect_scenario_async(scenario)
            collected_records = collector.records
        finally:
            collector.close()

        # 3) SSL train (mutates ``temporal_policy`` in place; cumulative
        #    across rounds, same as multi_round_loop).
        dataset = trace_records_to_training_dataset(collected_records)
        ssl_report: SSLDemoReport = run_ssl_demo(
            dataset=dataset,
            policy=temporal_policy,
            n_z=n_z,
            alpha=alpha,
        )
        new_temporal_snapshot = temporal_policy.export_rare_heavy_snapshot()

        # 4) Regime calibration: nudge weights AFTER eval pass so the next
        #    round sees the update. Includes the diversity penalty so a
        #    small scenario pack does not collapse the calibrator into
        #    "always predict the regime that's in every label set".
        if round_index < rounds - 1:
            _apply_updates_in_place(
                regime_weights, misses, lr=regime_learning_rate
            )
            _apply_diversity_penalty(
                regime_weights,
                predicted_counts,
                predicted_total,
                threshold=diversity_threshold,
                lr=diversity_lr,
            )
            _apply_diversity_prior_penalty(
                regime_priors,
                predicted_counts,
                predicted_total,
                threshold=diversity_threshold,
                lr=diversity_lr,
            )
        new_regime_bootstrap = _build_regime_bootstrap(
            regime_weights,
            strategy_priors=regime_priors,
        )

        if round_reports:
            distance_to_baseline = _distance(
                round_reports[0].distribution, distribution
            )
        else:
            distance_to_baseline = 0.0

        round_reports.append(
            SuperLoopRoundReport(
                round_index=round_index,
                distribution=distribution,
                ssl=ssl_report,
                regime_match_rate=regime_match_rate,
                misclassified_turn_count=len(misses),
                selection_weights=tuple(sorted(regime_weights.items())),
                temporal_snapshot=new_temporal_snapshot,
                regime_bootstrap=new_regime_bootstrap,
                distance_to_baseline=distance_to_baseline,
                predicted_regime_counts=tuple(
                    sorted(
                        predicted_counts.items(),
                        key=lambda kv: (-kv[1], kv[0]),
                    )
                ),
            )
        )

    final_temporal = round_reports[-1].temporal_snapshot
    final_regime = _build_regime_bootstrap(
        regime_weights,
        strategy_priors=regime_priors,
        description=(
            f"Super loop final regime bootstrap: "
            f"baseline match {round_reports[0].regime_match_rate:.0%} -> "
            f"final {round_reports[-1].regime_match_rate:.0%}"
        ),
    )

    verdicts = _build_verdicts(round_reports)
    description = (
        f"Super loop on {len(chosen)} scenarios x {rounds} rounds: "
        f"regime match {round_reports[0].regime_match_rate:.0%} -> "
        f"{round_reports[-1].regime_match_rate:.0%}, "
        f"\u03b2_t {round_reports[0].ssl.switch_frequency_last or 0.0:.3f} -> "
        f"{round_reports[-1].ssl.switch_frequency_last or 0.0:.3f}"
    )
    return SuperLoopReport(
        scenarios=tuple(s.scenario_id for s in chosen),
        rounds=tuple(round_reports),
        final_temporal_snapshot=final_temporal,
        final_regime_bootstrap=final_regime,
        verdicts=verdicts,
        description=description,
    )


def run_super_loop(
    *,
    rounds: int = 3,
    scenarios: tuple[ScriptedScenario, ...] | None = None,
    n_z: int = 3,
    alpha: float = 0.1,
    regime_learning_rate: float = 0.18,
    diversity_threshold: float = 0.50,
    diversity_lr: float = 0.30,
    domain_experience_packages: tuple[DomainExperiencePackage, ...] | None = None,
    substrate_runtime: object | None = None,
) -> SuperLoopReport:
    import asyncio

    return asyncio.run(
        run_super_loop_async(
            rounds=rounds,
            scenarios=scenarios,
            n_z=n_z,
            alpha=alpha,
            regime_learning_rate=regime_learning_rate,
            diversity_threshold=diversity_threshold,
            diversity_lr=diversity_lr,
            domain_experience_packages=domain_experience_packages,
            substrate_runtime=substrate_runtime,
        )
    )


# ---------------------------------------------------------------------------
# Verdicts + formatter
# ---------------------------------------------------------------------------


def _build_regime_bootstrap(
    weights: dict[str, float],
    *,
    strategy_priors: dict[str, float] | None = None,
    description: str = "",
) -> RegimeBootstrap:
    return RegimeBootstrap(
        selection_weights=tuple(sorted(weights.items())),
        strategy_priors=tuple(sorted((strategy_priors or {}).items())),
        description=description,
    )


def _apply_diversity_prior_penalty(
    priors: dict[str, float],
    predicted_counts: dict[str, int],
    total_turns: int,
    *,
    threshold: float,
    lr: float,
) -> None:
    """Persist anti-monoculture pressure in regime strategy priors.

    Selection weights are intentionally clipped to a tight band so they do
    not override per-turn evidence. When a tiny vertical scenario pack
    collapses to one regime, a bounded negative prior on the overrepresented
    regime provides a stable memory of that pressure across rounds.
    """
    if total_turns <= 0 or lr <= 0.0:
        return
    if not 0.0 < threshold < 1.0:
        raise ValueError(
            f"diversity threshold must be in (0, 1), got {threshold!r}"
        )
    severe_threshold = max(threshold, 0.85)
    for regime, count in predicted_counts.items():
        if regime not in priors:
            continue
        share = count / total_turns
        if share <= severe_threshold:
            continue
        excess = share - threshold
        priors[regime] = max(-0.5, min(0.5, priors[regime] - lr * excess))


def _build_verdicts(rounds: list[SuperLoopRoundReport]) -> dict[str, bool]:
    if len(rounds) < 2:
        return {"sufficient_rounds": False}
    baseline = rounds[0]
    best_match = max(r.regime_match_rate for r in rounds)
    regime_improved = best_match > baseline.regime_match_rate

    snapshot_fingerprints = {repr(r.temporal_snapshot) for r in rounds}
    temporal_evolved = len(snapshot_fingerprints) > 1

    sparse_and_calibrated = any(
        r.regime_match_rate > baseline.regime_match_rate
        and r.ssl.switch_frequency_last is not None
        and r.ssl.switch_frequency_last <= 0.20
        for r in rounds[1:]
    )

    return {
        "sufficient_rounds": True,
        "regime_match_improved": regime_improved,
        "temporal_state_evolved": temporal_evolved,
        "joint_sparse_and_calibrated": sparse_and_calibrated,
    }


def format_super_loop_report(report: SuperLoopReport) -> str:
    lines: list[str] = []
    lines.append(
        f"== Super loop ({len(report.rounds)} rounds x "
        f"{len(report.scenarios)} scenarios) =="
    )
    lines.append(f"   scenarios: {', '.join(report.scenarios)}")
    lines.append("")
    for r in report.rounds:
        tag = "BASELINE (round 0)" if r.round_index == 0 else f"ROUND {r.round_index}"
        lines.append(f"   --- {tag} ---")
        lines.append(_format_distribution(r))
        lines.append(
            f"      regime:  match={r.regime_match_rate:.0%}  "
            f"misclassified={r.misclassified_turn_count}  "
            f"top-share={r.predicted_regime_share:.0%}"
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
            f"      ssl:     switch_freq "
            f"{r.ssl.switch_frequency_first or 0.0:.3f} -> "
            f"{r.ssl.switch_frequency_last or 0.0:.3f}  "
            f"loss {r.ssl.prediction_loss_first:.3f} -> "
            f"{r.ssl.prediction_loss_last:.3f}"
        )
        lines.append(f"      distance_to_baseline: {r.distance_to_baseline:.3f}")
        lines.append("")
    lines.append("   verdicts:")
    for key, value in sorted(report.verdicts.items()):
        flag = "OK" if value else "FAIL"
        lines.append(f"      {flag}  {key}={value}")
    best = report.best_round()
    lines.append(
        f"   best round: round {best.round_index}  "
        f"regime_match={best.regime_match_rate:.0%}  "
        f"\u03b2_t={best.ssl.switch_frequency_last or 0.0:.3f}"
    )
    lines.append(f"   trajectory passes: {report.trajectory_passes()}")
    return "\n".join(lines)


def _format_distribution(r: SuperLoopRoundReport) -> str:
    dist = r.distribution
    parts: list[str] = []
    parts.append(
        f"      turns={dist.turn_count}  non-empty={dist.response_non_empty_rate:.0%}  "
        f"pe_mean={dist.pe_mean:.3f}"
    )
    if dist.regime_counts:
        parts.append(
            "      regimes:  " + ", ".join(f"{k}={v}" for k, v in dist.regime_counts)
        )
    if dist.intent_counts:
        parts.append(
            "      intents:  " + ", ".join(f"{k}={v}" for k, v in dist.intent_counts)
        )
    return "\n".join(parts)
