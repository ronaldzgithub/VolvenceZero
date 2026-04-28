"""``lifeform-bench`` and ``lifeform-trace`` CLI entry points."""

from __future__ import annotations

import argparse
import sys

from lifeform_evolution.benchmark import (
    all_built_in_scenarios,
    casual_social_checkin_scenario,
    format_report,
    low_mood_disclosure_scenario,
    run_benchmark,
    trust_rupture_repair_scenario,
)
from lifeform_evolution.learning_loop import (
    format_learning_loop_report,
    run_learning_loop,
)
from lifeform_evolution.multi_round_loop import (
    format_multi_round_report,
    run_multi_round_loop,
)
from lifeform_evolution.ssl_demo import (
    format_ssl_demo_report,
    run_ssl_demo,
    run_ssl_demo_from_ndjson,
)
from lifeform_evolution.trace_collector import TraceCollector


_SCENARIOS = {
    "low-mood-disclosure": low_mood_disclosure_scenario,
    "trust-rupture-repair": trust_rupture_repair_scenario,
    "casual-social-checkin": casual_social_checkin_scenario,
}


# ---------------------------------------------------------------------------
# lifeform-bench
# ---------------------------------------------------------------------------


def _build_bench_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lifeform-bench",
        description=(
            "Run a scripted multi-turn benchmark on a Lifeform built from "
            "VolvenceZero + the EmoGPT-style companion vertical."
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=tuple(_SCENARIOS.keys()) + ("all",),
        default="low-mood-disclosure",
        help='Scripted scenario to run (or "all" to run every built-in scenario).',
    )
    parser.add_argument(
        "--min-regime-match-rate",
        type=float,
        default=0.5,
        help="Minimum regime-match rate for the run to be considered passed (default 0.5).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_bench_parser()
    args = parser.parse_args(argv)

    if args.scenario == "all":
        ok = True
        for scenario in all_built_in_scenarios():
            report = run_benchmark(scenario=scenario)
            print(format_report(report))
            print()
            ok = ok and report.passed(min_regime_match_rate=args.min_regime_match_rate)
        return 0 if ok else 1

    scenario = _SCENARIOS[args.scenario]()
    report = run_benchmark(scenario=scenario)
    print(format_report(report))
    return 0 if report.passed(min_regime_match_rate=args.min_regime_match_rate) else 1


# ---------------------------------------------------------------------------
# lifeform-trace
# ---------------------------------------------------------------------------


def _build_trace_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lifeform-trace",
        description=(
            "Run scripted scenarios and emit a line-delimited JSON trace "
            "suitable for offline vz-temporal SSL training."
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=tuple(_SCENARIOS.keys()) + ("all",),
        default="all",
        help='Scenario to capture (or "all"). Default: all.',
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for the .ndjson trace file.",
    )
    return parser


def main_trace(argv: list[str] | None = None) -> int:
    parser = _build_trace_parser()
    args = parser.parse_args(argv)

    if args.scenario == "all":
        scenarios = all_built_in_scenarios()
    else:
        scenarios = (_SCENARIOS[args.scenario](),)

    collector = TraceCollector(output_path=args.out)
    try:
        for scenario in scenarios:
            report = collector.collect_scenario(scenario)
            print(
                f"== Trace: {report.scenario_id} =="
                f" rows={report.record_count}"
                f" regimes={','.join(report.distinct_regimes) or '-'}"
                f" intents={','.join(report.distinct_intents) or '-'}"
                f" pe_max={report.pe_max:.3f}"
                f" pe_mean={report.pe_mean:.3f}"
            )
        print(f"Trace written to {collector.output_path}")
        return 0
    finally:
        collector.close()


# ---------------------------------------------------------------------------
# lifeform-ssl
# ---------------------------------------------------------------------------


def _build_ssl_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lifeform-ssl",
        description=(
            "Run the SSL training demo: collect traces (or read from disk), "
            "run vz-temporal MetacontrollerSSLTrainer, report metrics."
        ),
    )
    parser.add_argument(
        "--from-ndjson",
        default=None,
        help=(
            "Path to a previously-written .ndjson trace file. If omitted, "
            "the runner collects all built-in scenarios fresh."
        ),
    )
    parser.add_argument(
        "--n-z",
        type=int,
        default=3,
        help="Latent code dimensionality (default 3).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Variational bottleneck weight (default 0.1).",
    )
    parser.add_argument(
        "--require-loss-decrease",
        action="store_true",
        help="Exit non-zero if prediction loss did not decrease across the run.",
    )
    return parser


def main_ssl(argv: list[str] | None = None) -> int:
    parser = _build_ssl_parser()
    args = parser.parse_args(argv)

    if args.from_ndjson:
        report = run_ssl_demo_from_ndjson(
            path=args.from_ndjson,
            n_z=args.n_z,
            alpha=args.alpha,
        )
    else:
        report = run_ssl_demo(n_z=args.n_z, alpha=args.alpha)
    print(format_ssl_demo_report(report))
    if args.require_loss_decrease and not report.loss_decreased():
        return 1
    return 0


# ---------------------------------------------------------------------------
# lifeform-loop \u2014 closed feedback loop (collect \u2192 SSL \u2192 reinject \u2192 re-bench)
# ---------------------------------------------------------------------------


def _build_loop_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lifeform-loop",
        description=(
            "Closed feedback loop: run baseline lifeform benchmark, "
            "collect traces, run SSL training, inject the trained "
            "metacontroller into a fresh Lifeform, and re-run the benchmark "
            "to compare distributions."
        ),
    )
    parser.add_argument(
        "--n-z", type=int, default=3,
        help="Latent code dimensionality (default 3).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1,
        help="Variational bottleneck weight (default 0.1).",
    )
    parser.add_argument(
        "--require-loop-closed", action="store_true",
        help="Exit non-zero if the loop did not close (any verdict failed).",
    )
    return parser


def main_loop(argv: list[str] | None = None) -> int:
    parser = _build_loop_parser()
    args = parser.parse_args(argv)
    report = run_learning_loop(n_z=args.n_z, alpha=args.alpha)
    print(format_learning_loop_report(report))
    if args.require_loop_closed and not report.loop_closed():
        return 1
    return 0


# ---------------------------------------------------------------------------
# lifeform-multi-loop \u2014 multi-round R13 evidence
# ---------------------------------------------------------------------------


def _build_multi_loop_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lifeform-multi-loop",
        description=(
            "Run the multi-round closed feedback loop. Round 0 is the "
            "untrained baseline; subsequent rounds collect with the "
            "previous round's policy, train, and re-benchmark. Reports "
            "per-round Hellinger distance from baseline and from the "
            "previous round."
        ),
    )
    parser.add_argument(
        "--rounds", type=int, default=4,
        help="Total number of rounds including round 0 baseline (default 4).",
    )
    parser.add_argument(
        "--n-z", type=int, default=3,
        help="Latent code dimensionality (default 3).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1,
        help="Variational bottleneck weight (default 0.1).",
    )
    parser.add_argument(
        "--require-trajectory-passes", action="store_true",
        help="Exit non-zero if any per-round trajectory verdict failed.",
    )
    return parser


def main_multi_loop(argv: list[str] | None = None) -> int:
    parser = _build_multi_loop_parser()
    args = parser.parse_args(argv)
    report = run_multi_round_loop(
        rounds=args.rounds,
        n_z=args.n_z,
        alpha=args.alpha,
    )
    print(format_multi_round_report(report))
    if args.require_trajectory_passes and not report.trajectory_passes():
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
