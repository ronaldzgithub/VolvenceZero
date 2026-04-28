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
class RoundReport:
    """One round (compress + reinforce + observe) of the multi-round loop."""

    round_index: int
    distribution: DistributionSnapshot
    ssl: SSLDemoReport
    snapshot: MetacontrollerParameterSnapshot
    distance_to_baseline: float
    distance_to_previous: float


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

    def trajectory_passes(self) -> bool:
        return all(self.verdicts.values())


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
) -> MultiRoundLearningLoopReport:
    if rounds < 2:
        raise ValueError(
            "Multi-round loop needs at least 2 rounds (round 0 baseline + at least 1 trained)."
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

        round_reports.append(
            RoundReport(
                round_index=round_index,
                distribution=distribution,
                ssl=ssl_report,
                snapshot=snapshot,
                distance_to_baseline=distance_to_baseline,
                distance_to_previous=distance_to_previous,
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
) -> MultiRoundLearningLoopReport:
    import asyncio

    return asyncio.run(
        run_multi_round_loop_async(
            rounds=rounds,
            scenarios=scenarios,
            n_z=n_z,
            alpha=alpha,
        )
    )


# ---------------------------------------------------------------------------
# Verdicts + formatter
# ---------------------------------------------------------------------------


def _build_multi_round_verdicts(rounds: list[RoundReport]) -> dict[str, bool]:
    if len(rounds) < 2:
        return {"sufficient_rounds": False}

    # Final round must have at least non-zero distance to baseline. If the
    # trained metacontroller had no surface-level effect, the loop is not
    # closing in any meaningful sense.
    final_distance = rounds[-1].distance_to_baseline
    distance_meaningful = final_distance > 0.0

    # Every round must have actually trained (round 0 included now).
    all_trained = all(r.ssl.trained_step_count > 0 for r in rounds)

    # Snapshot fingerprints across rounds must change (SSL is not a no-op at
    # the parameter level). We compare via ``repr`` of the frozen-dataclass
    # snapshot, which is deterministic for tuples of floats; if all snapshots
    # compare equal we treat it as no evolution.
    snapshot_fingerprints = {_snapshot_fingerprint(r.snapshot) for r in rounds}
    policy_state_evolved = len(snapshot_fingerprints) > 1

    # Switch frequency at the LAST trained round must not be wildly more
    # random than at the FIRST trained round. We allow up to +0.10 noise so
    # tiny fluctuations between saturated rounds do not flip the verdict.
    first_ssl = rounds[0].ssl
    last_ssl = rounds[-1].ssl
    switch_freq_first = first_ssl.switch_frequency_last
    switch_freq_last = last_ssl.switch_frequency_last
    if switch_freq_first is not None and switch_freq_last is not None:
        switch_did_not_explode = switch_freq_last <= switch_freq_first + 0.10
    else:
        switch_did_not_explode = True

    return {
        "sufficient_rounds": True,
        "all_trained": all_trained,
        "distance_to_baseline_meaningful": distance_meaningful,
        "policy_state_evolved": policy_state_evolved,
        "switch_freq_did_not_explode": switch_did_not_explode,
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
        lines.append("")
    lines.append("   verdicts:")
    for key, value in sorted(report.verdicts.items()):
        flag = "OK" if value else "FAIL"
        lines.append(f"      {flag}  {key}={value}")
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
