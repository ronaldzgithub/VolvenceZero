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


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
