"""Multi-round learning loop \u2014 the R13 evidence harness.

R13 ("training loop must alternate compression and reinforcement") is the
NL/ETA invariant that says reinforcement must act on a compressed,
structured internal substrate that itself is updated by compression. At
the lifeform layer this becomes:

* round 0: untrained policy \u2192 traces are baseline
* round k (k > 0):
    a. **collect** traces with the policy as it is at the start of the round
    b. **compress** \u2014 SSL training updates the policy in place
    c. **reinforce** (lite version, until Internal RL is wired in) \u2014 the
       same policy is reinjected into a fresh ``Lifeform`` for the next round
    d. **observe** \u2014 record the regime / intent / action distribution

What the harness establishes:

* the policy and the traces co-evolve (round 2's traces reflect round 1's
  policy, etc.) \u2014 that is the actual ETA wake-sleep loop, not a one-shot
* drift from baseline accumulates or saturates rather than oscillating
  randomly \u2014 published as machine-readable per-round metrics
* the ``MetacontrollerParameterSnapshot`` at every round is preserved, so
  later harnesses (Internal RL, real-substrate evidence) can rewind to any
  intermediate generation without re-running upstream rounds.

The verdicts here are **direction**-only, same discipline as the single-round
``learning_loop``: with synthetic substrate and 10 turns/round, claiming
specific magnitudes would be over-fitting to noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

from volvence_zero.application import DomainExperiencePackage
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
from lifeform_evolution.ssl_demo import SSLDemoReport, run_ssl_demo
from lifeform_evolution.trace_collector import TraceCollector


# ---------------------------------------------------------------------------
# Per-round + top-level reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundQualityMetrics:
    """Per-round behavioural quality, aggregated across all scripted scenarios.

    These let the multi-round loop answer R12 acceptance question #12 \u2014
    "does high PE trigger controller change AND later improvement vs weak
    baseline?" \u2014 directly, instead of leaving it implicit in the Hellinger
    distance between distributions. Round 0's metrics are the canonical
    weak baseline; later rounds publish their delta vs round 0.
    """

    mean_regime_match_rate: float
    mean_pe_threshold_match_rate: float
    mean_pe: float
    early_half_pe_mean: float
    late_half_pe_mean: float
    pe_recovery_delta: float
    mean_switch_gate: float


@dataclass(frozen=True)
class RoundDeltaVsBaseline:
    """Per-metric delta of this round versus round 0.

    Positive ``regime_match_delta`` and ``pe_recovery_delta_delta`` are
    improvements; negative ``mean_switch_gate_delta`` means the trained
    policy is switching less often (sparser \u03b2_t \u2192 more committed
    abstract actions \u2192 R3 healthy).
    """

    regime_match_delta: float
    pe_threshold_delta: float
    pe_recovery_delta_delta: float
    mean_pe_delta: float
    mean_switch_gate_delta: float

    @property
    def regime_match_improved(self) -> bool:
        return self.regime_match_delta > 0.0

    @property
    def pe_recovery_improved(self) -> bool:
        return self.pe_recovery_delta_delta > 0.0

    @property
    def switching_sparser(self) -> bool:
        return self.mean_switch_gate_delta < 0.0


@dataclass(frozen=True)
class RoundReport:
    """One round (compress + reinforce + observe) of the multi-round loop."""

    round_index: int
    distribution: DistributionSnapshot
    ssl: SSLDemoReport
    snapshot: MetacontrollerParameterSnapshot
    distance_to_baseline: float
    distance_to_previous: float
    quality: RoundQualityMetrics | None = None
    delta_from_baseline: RoundDeltaVsBaseline | None = None


@dataclass(frozen=True)
class MultiRoundLearningLoopReport:
    """Roll-up of all rounds.

    Fields used by lifeform-evolution dashboards:

    * ``rounds[i].distribution`` \u2014 regime / intent / action histogram per round
    * ``rounds[i].distance_to_baseline`` \u2014 Hellinger-style distance from round 0
    * ``rounds[i].ssl.switch_frequency_last`` \u2014 \u03b2_t firing rate per round

    The verdict ``policy_state_evolved`` is the multi-round analogue of
    ``learning_loop.loop_closed``: it asserts that consecutive rounds produced
    structurally different policy snapshots (so SSL is doing more than zero
    work each round).
    """

    scenarios: tuple[str, ...]
    rounds: tuple[RoundReport, ...]
    verdicts: dict[str, bool] = field(default_factory=dict)
    description: str = ""

    @property
    def baseline(self) -> RoundReport:
        return self.rounds[0]

    @property
    def final(self) -> RoundReport:
        return self.rounds[-1]

    # The verdict dict is split into two semantic groups:
    #
    # * **Trajectory-shape verdicts** (the original gate set) describe
    #   whether the loop is wired and producing structurally meaningful
    #   policy evolution. They are deterministic on the built-in scenarios
    #   and SHOULD pass on every healthy run.
    # * **Acceptance#12 verdicts** describe whether a trained round
    #   actually improved on the weak baseline (round 0). These can
    #   legitimately fail on small scenario sets / over-trained final
    #   rounds, and consumers gate on them via
    #   ``improvement_vs_baseline_passes`` or the ``--require-improvement-vs-
    #   baseline`` CLI flag, NOT via ``trajectory_passes``.
    _TRAJECTORY_VERDICT_KEYS: ClassVar[tuple[str, ...]] = (
        "sufficient_rounds",
        "all_trained",
        "policy_state_evolved",
        "max_distance_meaningful",
        "found_sparse_drifted_regime",
    )
    _ACCEPTANCE_12_VERDICT_KEYS: ClassVar[tuple[str, ...]] = (
        "improved_regime_match_vs_baseline",
        "improved_pe_recovery_vs_baseline",
        "found_pe_aligned_improvement_round",
    )

    def trajectory_passes(self) -> bool:
        """Original gate: did the loop wire end-to-end and evolve the policy?

        Excludes the R12 acceptance#12 improvement verdicts, which gate
        a stricter claim and can legitimately fail when training overshoots.
        """
        for key in self._TRAJECTORY_VERDICT_KEYS:
            if key in self.verdicts and not self.verdicts[key]:
                return False
        return True

    def improvement_vs_baseline_passes(self) -> bool:
        """R12 acceptance#12 gate: did some trained round beat round 0?

        True iff at least one round shows improvement-vs-baseline AND that
        round also has sparse switching plus surface drift (the strict
        reading of "high PE \u2192 controller change \u2192 later improvement").
        """
        return self.verdicts.get("found_pe_aligned_improvement_round", False)

    def best_round(self) -> RoundReport:
        """Return the round that combines maximal drift with sparse \u03b2_t.

        Sparse threshold: ``switch_frequency_last <= 0.20``. Among such rounds
        the one with the highest distance from baseline wins. Falls back to
        the round with maximum distance if no round qualifies as sparse.
        """
        sparse = [
            r for r in self.rounds[1:]
            if r.ssl.switch_frequency_last is not None
            and r.ssl.switch_frequency_last <= 0.20
        ]
        candidates = sparse or list(self.rounds[1:])
        return max(candidates, key=lambda r: r.distance_to_baseline)


# ---------------------------------------------------------------------------
# Distribution distance
# ---------------------------------------------------------------------------


def _hellinger_distance(
    a: DistributionSnapshot,
    b: DistributionSnapshot,
) -> float:
    """Hellinger distance between two distributions over the union of regimes,
    intents, and abstract actions.

    Hellinger is in [0, 1]: 0 when distributions are identical, 1 when they
    have disjoint support. We use it (instead of KL) because the supports are
    small, often disjoint, and we want a symmetric / bounded measure for
    dashboards.
    """
    keys: set[tuple[str, str]] = set()
    p_total = max(a.turn_count, 1)
    q_total = max(b.turn_count, 1)

    p: dict[tuple[str, str], float] = {}
    q: dict[tuple[str, str], float] = {}
    for label, items in (("regime", a.regime_counts), ("intent", a.intent_counts), ("action", a.abstract_action_counts)):
        for key, count in items:
            p[(label, key)] = count / p_total
            keys.add((label, key))
    for label, items in (("regime", b.regime_counts), ("intent", b.intent_counts), ("action", b.abstract_action_counts)):
        for key, count in items:
            q[(label, key)] = count / q_total
            keys.add((label, key))

    summed = 0.0
    for k in keys:
        diff = math.sqrt(p.get(k, 0.0)) - math.sqrt(q.get(k, 0.0))
        summed += diff * diff
    # Hellinger is normally 1/sqrt(2) * sqrt(sum). We average across the three
    # axes (regime/intent/action) to keep it in [0, 1].
    return min(1.0, math.sqrt(summed / (2.0 * 3.0)))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_multi_round_loop_async(
    *,
    rounds: int = 3,
    scenarios: tuple[ScriptedScenario, ...] | None = None,
    n_z: int = 3,
    alpha: float = 0.1,
    domain_experience_packages: tuple[DomainExperiencePackage, ...] | None = None,
) -> MultiRoundLearningLoopReport:
    """Run rounds of (collect \u2192 SSL \u2192 reinject) on a vertical's scenarios.

    ``domain_experience_packages`` lets a different vertical (e.g.
    ``lifeform-domain-coding``) drive the loop with its own
    ``DomainExperiencePackage`` instead of the default companion pack.
    """
    if rounds < 2:
        raise ValueError(
            "Multi-round loop needs at least 2 rounds (round 0 baseline + at least 1 trained)."
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

    # Single shared policy across rounds: SSL training mutates this in place,
    # so each round picks up where the previous left off. This is the
    # cumulative-compression invariant of R13.
    policy = FullLearnedTemporalPolicy()

    round_reports: list[RoundReport] = []

    for round_index in range(rounds):
        # 1. Build the lifeform that will run this round's benchmark and
        #    collect its traces. From round 1 onward, seed it with the
        #    snapshot exported at the end of the previous round so traces
        #    reflect the trained policy's behaviour.
        if round_index == 0:
            seed_snapshot: MetacontrollerParameterSnapshot | None = None
        else:
            seed_snapshot = round_reports[-1].snapshot

        eval_lifeform = Lifeform(base_config, temporal_bootstrap=seed_snapshot)
        bench_reports: list[BenchmarkReport] = []
        for scenario in chosen:
            bench_report = await _run_scenario_with_lifeform(
                scenario=scenario,
                lifeform=eval_lifeform,
            )
            bench_reports.append(bench_report)
        distribution = _summarise_reports(tuple(bench_reports))

        # 2. Collect a fresh batch of traces with the same seed snapshot
        #    (separate sessions from the benchmark above so behaviour
        #    distribution and trace dataset are independently sampled).
        collector = TraceCollector(temporal_bootstrap=seed_snapshot)
        try:
            for scenario in chosen:
                await collector.collect_scenario_async(scenario)
            collected_records = collector.records
        finally:
            collector.close()

        # 3. SSL training. We train every round, including round 0, so the
        #    snapshot exported at the end of round k reflects k+1 SSL passes.
        #    The "baseline" anchor is round 0's *distribution* (which used
        #    the fresh policy, before any training in this loop ran), not
        #    round 0's snapshot. Skipping SSL at round 0 was a previous
        #    design choice that made round 1's distribution identical to
        #    round 0's because round 1's seed snapshot was still untrained;
        #    that hid the very behavioural shift the loop is meant to show.
        dataset = trace_records_to_training_dataset(collected_records)
        ssl_report: SSLDemoReport = run_ssl_demo(
            dataset=dataset,
            policy=policy,  # in-place mutation \u2014 cumulative across rounds
            n_z=n_z,
            alpha=alpha,
        )

        snapshot = policy.export_rare_heavy_snapshot()

        if round_reports:
            distance_to_baseline = _hellinger_distance(
                round_reports[0].distribution, distribution
            )
            distance_to_previous = _hellinger_distance(
                round_reports[-1].distribution, distribution
            )
        else:
            distance_to_baseline = 0.0
            distance_to_previous = 0.0

        # Per-round quality aggregates and baseline delta. Round 0 publishes
        # its own quality but no delta (it IS the baseline); later rounds
        # publish both. Quality is always present so consumers do not have
        # to special-case round 0.
        quality = _compute_round_quality(tuple(bench_reports))
        if round_reports:
            delta = _compute_delta_vs_baseline(
                round_reports[0].quality, quality
            )
        else:
            delta = None

        round_reports.append(
            RoundReport(
                round_index=round_index,
                distribution=distribution,
                ssl=ssl_report,
                snapshot=snapshot,
                distance_to_baseline=distance_to_baseline,
                distance_to_previous=distance_to_previous,
                quality=quality,
                delta_from_baseline=delta,
            )
        )

    verdicts = _build_multi_round_verdicts(round_reports)
    description = (
        f"Multi-round learning loop on {len(chosen)} scenarios x {rounds} rounds: "
        f"final distance to baseline = {round_reports[-1].distance_to_baseline:.3f}, "
        f"final \u03b2_t = "
        f"{round_reports[-1].ssl.switch_frequency_last if round_reports[-1].ssl else 'n/a'}"
    )

    return MultiRoundLearningLoopReport(
        scenarios=tuple(s.scenario_id for s in chosen),
        rounds=tuple(round_reports),
        verdicts=verdicts,
        description=description,
    )


def run_multi_round_loop(
    *,
    rounds: int = 3,
    scenarios: tuple[ScriptedScenario, ...] | None = None,
    n_z: int = 3,
    alpha: float = 0.1,
    domain_experience_packages: tuple[DomainExperiencePackage, ...] | None = None,
) -> MultiRoundLearningLoopReport:
    import asyncio

    return asyncio.run(
        run_multi_round_loop_async(
            rounds=rounds,
            scenarios=scenarios,
            n_z=n_z,
            alpha=alpha,
            domain_experience_packages=domain_experience_packages,
        )
    )


# ---------------------------------------------------------------------------
# Verdicts + formatter
# ---------------------------------------------------------------------------


def _compute_round_quality(reports: tuple[BenchmarkReport, ...]) -> RoundQualityMetrics:
    """Aggregate ``BenchmarkReport``s into one round-level quality summary.

    Round-level quality is the per-scenario mean: each scenario contributes
    equally regardless of turn count. PE recovery is computed at the
    PER-SCENARIO level (early half vs late half) and then averaged \u2014
    averaging globally over all turns can hide per-scenario recovery when
    one scenario starts hot and another ends hot.
    """
    if not reports:
        return RoundQualityMetrics(
            mean_regime_match_rate=0.0,
            mean_pe_threshold_match_rate=0.0,
            mean_pe=0.0,
            early_half_pe_mean=0.0,
            late_half_pe_mean=0.0,
            pe_recovery_delta=0.0,
            mean_switch_gate=0.0,
        )

    n = len(reports)
    regime_match = sum(r.regime_match_rate for r in reports) / n
    pe_threshold = sum(r.pe_threshold_match_rate for r in reports) / n

    pe_values: list[float] = []
    early_means: list[float] = []
    late_means: list[float] = []
    switch_gates: list[float] = []
    for report in reports:
        turn_pes = [t.pe_magnitude for t in report.turn_reports]
        pe_values.extend(turn_pes)
        switch_gates.extend(t.temporal_switch_gate for t in report.turn_reports)
        if len(turn_pes) >= 2:
            half = len(turn_pes) // 2
            early = sum(turn_pes[:half]) / half if half else 0.0
            late_count = len(turn_pes) - half
            late = sum(turn_pes[half:]) / late_count if late_count else 0.0
            early_means.append(early)
            late_means.append(late)

    mean_pe = sum(pe_values) / len(pe_values) if pe_values else 0.0
    early_half = sum(early_means) / len(early_means) if early_means else 0.0
    late_half = sum(late_means) / len(late_means) if late_means else 0.0
    mean_switch_gate = (
        sum(switch_gates) / len(switch_gates) if switch_gates else 0.0
    )
    return RoundQualityMetrics(
        mean_regime_match_rate=regime_match,
        mean_pe_threshold_match_rate=pe_threshold,
        mean_pe=mean_pe,
        early_half_pe_mean=early_half,
        late_half_pe_mean=late_half,
        pe_recovery_delta=early_half - late_half,
        mean_switch_gate=mean_switch_gate,
    )


def _compute_delta_vs_baseline(
    baseline: RoundQualityMetrics | None,
    current: RoundQualityMetrics,
) -> RoundDeltaVsBaseline | None:
    if baseline is None:
        return None
    return RoundDeltaVsBaseline(
        regime_match_delta=current.mean_regime_match_rate
        - baseline.mean_regime_match_rate,
        pe_threshold_delta=current.mean_pe_threshold_match_rate
        - baseline.mean_pe_threshold_match_rate,
        pe_recovery_delta_delta=current.pe_recovery_delta - baseline.pe_recovery_delta,
        mean_pe_delta=current.mean_pe - baseline.mean_pe,
        mean_switch_gate_delta=current.mean_switch_gate - baseline.mean_switch_gate,
    )


def _build_multi_round_verdicts(rounds: list[RoundReport]) -> dict[str, bool]:
    """Verdicts for the multi-round trajectory.

    R13 claim is "compression-reinforcement should ALTERNATE", not "training
    forever on the same data should keep getting better". Over-training on a
    small fixed dataset is *expected* to eventually overshoot and collapse
    \u03b2_t back toward random switching \u2014 the harness sees this in practice
    around round 3 with the built-in scenario set.

    We therefore frame the verdicts as **trajectory-shape** properties:

    * Did SSL run at every round?
    * Did the policy state actually evolve (snapshots differ)?
    * Did at least one round produce meaningful surface drift?
    * Did at least one round combine sparse \u03b2_t with meaningful drift?
      (the "found a healthy regime" claim)

    R12 acceptance question #12 adds three more:

    * ``improved_regime_match_vs_baseline`` \u2014 some trained round had a
      higher mean regime-match rate than round 0.
    * ``improved_pe_recovery_vs_baseline`` \u2014 some trained round had a
      larger PE recovery delta than round 0 (early-half PE \u2192 late-half PE
      decline was steeper). This is the "later improvement" leg of #12.
    * ``found_pe_aligned_improvement_round`` \u2014 there exists a round where
      sparse switching, surface drift, and improvement-vs-baseline all
      coincide. That is the strict reading of "high PE triggers temporally
      aligned controller changes AND later improvement vs weak baseline".
    """
    if len(rounds) < 2:
        return {"sufficient_rounds": False}

    all_trained = all(r.ssl.trained_step_count > 0 for r in rounds)
    snapshot_fingerprints = {_snapshot_fingerprint(r.snapshot) for r in rounds}
    policy_state_evolved = len(snapshot_fingerprints) > 1
    max_distance = max(r.distance_to_baseline for r in rounds)
    max_distance_meaningful = max_distance > 0.0

    sparse_and_drifted = any(
        r.distance_to_baseline > 0.0
        and r.ssl.switch_frequency_last is not None
        and r.ssl.switch_frequency_last <= 0.20
        for r in rounds[1:]
    )

    improved_regime = any(
        r.delta_from_baseline is not None and r.delta_from_baseline.regime_match_improved
        for r in rounds[1:]
    )
    improved_recovery = any(
        r.delta_from_baseline is not None and r.delta_from_baseline.pe_recovery_improved
        for r in rounds[1:]
    )
    pe_aligned_improvement = any(
        r.delta_from_baseline is not None
        and (
            r.delta_from_baseline.regime_match_improved
            or r.delta_from_baseline.pe_recovery_improved
        )
        and r.distance_to_baseline > 0.0
        and r.ssl.switch_frequency_last is not None
        and r.ssl.switch_frequency_last <= 0.20
        for r in rounds[1:]
    )

    return {
        "sufficient_rounds": True,
        "all_trained": all_trained,
        "policy_state_evolved": policy_state_evolved,
        "max_distance_meaningful": max_distance_meaningful,
        "found_sparse_drifted_regime": sparse_and_drifted,
        "improved_regime_match_vs_baseline": improved_regime,
        "improved_pe_recovery_vs_baseline": improved_recovery,
        "found_pe_aligned_improvement_round": pe_aligned_improvement,
    }


def _snapshot_fingerprint(snapshot: MetacontrollerParameterSnapshot) -> str:
    """Stable string fingerprint of a snapshot for change detection.

    We only care whether two snapshots are equal-or-not, not the actual
    parameter values; ``repr`` of a frozen-dataclass-of-tuples is deterministic
    enough for that and keeps this helper free of any export coupling.
    """
    return repr(snapshot)


def format_multi_round_report(report: MultiRoundLearningLoopReport) -> str:
    lines: list[str] = []
    lines.append(
        f"== Multi-round learning loop: {len(report.rounds)} rounds x {len(report.scenarios)} scenarios =="
    )
    lines.append(f"   scenarios: {', '.join(report.scenarios)}")
    lines.append("")
    for r in report.rounds:
        tag = "BASELINE (round 0)" if r.round_index == 0 else f"ROUND {r.round_index}"
        lines.append(f"   --- {tag} ---")
        lines.append(_format_round_distribution(r))
        lines.append(
            f"      ssl: traces={r.ssl.trace_count} steps={r.ssl.trained_step_count}  "
            f"switch_freq {r.ssl.switch_frequency_first or 0.0:.3f} \u2192 "
            f"{r.ssl.switch_frequency_last or 0.0:.3f}  "
            f"loss {r.ssl.prediction_loss_first:.3f} \u2192 {r.ssl.prediction_loss_last:.3f}"
        )
        lines.append(
            f"      distance: to baseline={r.distance_to_baseline:.3f}  "
            f"to previous={r.distance_to_previous:.3f}"
        )
        if r.quality is not None:
            lines.append(
                f"      quality:  regime_match={r.quality.mean_regime_match_rate:.2f}  "
                f"pe_recovery={r.quality.pe_recovery_delta:+.3f}  "
                f"early_pe={r.quality.early_half_pe_mean:.3f} \u2192 "
                f"late_pe={r.quality.late_half_pe_mean:.3f}  "
                f"switch_gate_avg={r.quality.mean_switch_gate:.3f}"
            )
        if r.delta_from_baseline is not None:
            d = r.delta_from_baseline
            lines.append(
                f"      vs baseline: regime_match \u0394={d.regime_match_delta:+.2f}  "
                f"pe_recovery \u0394={d.pe_recovery_delta_delta:+.3f}  "
                f"switch_gate \u0394={d.mean_switch_gate_delta:+.3f}  "
                + (
                    "[improved_regime]" if d.regime_match_improved else ""
                )
                + (" [improved_recovery]" if d.pe_recovery_improved else "")
                + (" [sparser]" if d.switching_sparser else "")
            )
        lines.append("")
    lines.append("   verdicts:")
    for key, value in sorted(report.verdicts.items()):
        flag = "OK" if value else "FAIL"
        lines.append(f"      {flag}  {key}={value}")
    best = report.best_round()
    lines.append(
        f"   best round: round {best.round_index}  "
        f"distance={best.distance_to_baseline:.3f}  "
        f"\u03b2_t={best.ssl.switch_frequency_last or 0.0:.3f}"
    )
    lines.append(f"   trajectory passes: {report.trajectory_passes()}")
    return "\n".join(lines)


def _format_round_distribution(r: RoundReport) -> str:
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
    if dist.abstract_action_counts:
        parts.append(
            "      actions:  " + ", ".join(f"{k}={v}" for k, v in dist.abstract_action_counts)
        )
    return "\n".join(parts)
