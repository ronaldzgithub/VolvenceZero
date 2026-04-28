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
from lifeform_evolution.regime_calibrator import _apply_updates_in_place, _tally_matches
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

    def best_round(self) -> SuperLoopRoundReport:
        """Round combining sparse \u03b2_t with the highest regime match rate.

        Sparse threshold: ``switch_frequency_last <= 0.20``. Among such
        rounds the one with the highest regime match rate wins. Falls back
        to highest regime match rate if no round qualifies as sparse.
        """
        sparse = [
            r for r in self.rounds[1:]
            if r.ssl.switch_frequency_last is not None
            and r.ssl.switch_frequency_last <= 0.20
        ]
        candidates = sparse or list(self.rounds[1:])
        return max(candidates, key=lambda r: r.regime_match_rate)


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
) -> SuperLoopReport:
    if rounds < 2:
        raise ValueError(
            "Super loop needs at least 2 rounds (round 0 baseline + at least 1 trained)."
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

    # Persistent state across rounds.
    temporal_policy = FullLearnedTemporalPolicy()
    regime_weights: dict[str, float] = {
        template.regime_id: 1.0 for template in REGIME_TEMPLATES
    }

    round_reports: list[SuperLoopRoundReport] = []

    for round_index in range(rounds):
        temporal_seed = (
            None if round_index == 0 else round_reports[-1].temporal_snapshot
        )
        regime_bootstrap = _build_regime_bootstrap(regime_weights)

        # 1) Eval pass: per-scenario benchmark with the current bootstraps.
        eval_lifeform = Lifeform(
            base_config,
            temporal_bootstrap=temporal_seed,
            regime_bootstrap=regime_bootstrap,
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

        # 2) Trace pass: collect SSL training data with the SAME bootstraps,
        #    so the next metacontroller update reflects the post-calibration
        #    behaviour rather than a stale baseline.
        collector = TraceCollector(temporal_bootstrap=temporal_seed)
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
        #    round sees the update.
        if round_index < rounds - 1:
            _apply_updates_in_place(
                regime_weights, misses, lr=regime_learning_rate
            )
        new_regime_bootstrap = _build_regime_bootstrap(regime_weights)

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
            )
        )

    final_temporal = round_reports[-1].temporal_snapshot
    final_regime = _build_regime_bootstrap(
        regime_weights,
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
) -> SuperLoopReport:
    import asyncio

    return asyncio.run(
        run_super_loop_async(
            rounds=rounds,
            scenarios=scenarios,
            n_z=n_z,
            alpha=alpha,
            regime_learning_rate=regime_learning_rate,
        )
    )


# ---------------------------------------------------------------------------
# Verdicts + formatter
# ---------------------------------------------------------------------------


def _build_regime_bootstrap(
    weights: dict[str, float], *, description: str = ""
) -> RegimeBootstrap:
    return RegimeBootstrap(
        selection_weights=tuple(sorted(weights.items())),
        description=description,
    )


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
            f"misclassified={r.misclassified_turn_count}"
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
