"""``lifeform-bench`` and ``lifeform-trace`` CLI entry points."""

from __future__ import annotations

import argparse
import sys

from lifeform_evolution.benchmark import (
    ScriptedScenario,
    all_built_in_scenarios,
    casual_social_checkin_scenario,
    format_report,
    low_mood_disclosure_scenario,
    run_benchmark,
    trust_rupture_repair_scenario,
)
from lifeform_evolution.family_report import (
    compute_family_report,
    family_report_to_dict,
    format_family_report,
)
from lifeform_evolution.companion_evidence import (
    companion_evidence_report_to_dict,
    format_companion_evidence_report,
    run_companion_evidence,
)
from lifeform_evolution.scenario_pack import load_scenarios
from lifeform_evolution.learning_loop import (
    format_learning_loop_report,
    run_learning_loop,
)
from lifeform_evolution.multi_round_loop import (
    format_multi_round_report,
    run_multi_round_loop,
)
from lifeform_evolution.regime_calibrator import (
    format_regime_calibration_report,
    run_regime_calibrator,
)
from lifeform_evolution.regime_io import (
    load_regime_bootstrap_only,
    save_regime_bootstrap,
)
from lifeform_evolution.super_loop import (
    format_super_loop_report,
    run_super_loop,
)
from lifeform_evolution.snapshot_io import (
    SnapshotArtifact,
    load_snapshot,
    load_snapshot_only,
    save_snapshot,
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
# Vertical resolution \u2014 a vertical name maps to (scenarios_dir, domain_pack)
#
# We deliberately resolve verticals by importing their public package
# functions rather than going through ``lifeform_service.verticals``: the
# evolution layer must run without lifeform-service installed (it can be
# installed independently in a training-only environment).
# ---------------------------------------------------------------------------


_KNOWN_VERTICALS: tuple[str, ...] = ("companion", "coding")


def _resolve_vertical(name: str | None) -> tuple[
    tuple[ScriptedScenario, ...] | None,
    tuple[object, ...] | None,
]:
    """Resolve ``--vertical NAME`` into ``(scenarios, domain_packages)``.

    Returns ``(None, None)`` when ``name`` is None/empty so the caller can
    fall back to its own default. Both elements are populated when the
    vertical resolves successfully.

    Raises ``SystemExit`` (via argparse-shaped error) when ``name`` is
    set but does not resolve \u2014 better to fail loudly than silently use
    the wrong vertical's scenarios with the wrong vertical's package.
    """
    if not name:
        return (None, None)
    if name == "companion":
        from lifeform_domain_emogpt import (
            build_companion_package,
            scenarios_dir,
        )

        return (load_scenarios(scenarios_dir()), (build_companion_package(),))
    if name == "coding":
        try:
            from lifeform_domain_coding import (
                build_coding_package,
                scenarios_dir as coding_scenarios_dir,
            )
        except ImportError as exc:
            raise SystemExit(
                f"--vertical coding requires lifeform-domain-coding to be installed: {exc}"
            ) from exc
        return (
            load_scenarios(coding_scenarios_dir()),
            (build_coding_package(),),
        )
    raise SystemExit(
        f"Unknown vertical {name!r}. Available: {_KNOWN_VERTICALS!r}."
    )


def _build_vertical_lifeform(name: str | None) -> object | None:
    """Construct a fully-wired Lifeform for the named vertical.

    Used by ``lifeform-bench --vertical NAME`` so the vertical's shipped
    bootstraps are picked up automatically. Returns ``None`` for an
    empty name; callers fall back to their default Lifeform.
    """
    if not name:
        return None
    if name == "companion":
        from lifeform_domain_emogpt import build_companion_lifeform

        return build_companion_lifeform()
    if name == "coding":
        from lifeform_domain_coding import build_coding_lifeform

        return build_coding_lifeform()
    raise SystemExit(f"Unknown vertical {name!r}.")


def _run_with_lifeform(
    *, scenario: ScriptedScenario, lifeform: object
) -> object:
    """Run one scenario against a pre-built Lifeform synchronously."""
    import asyncio

    from lifeform_evolution.benchmark import run_benchmark_async

    return asyncio.run(
        run_benchmark_async(scenario=scenario, lifeform=lifeform)
    )


def _resolve_scenarios_flag(path: str | None) -> tuple[ScriptedScenario, ...] | None:
    """Return the scenarios named by ``--scenarios PATH``, or None if unset.

    ``PATH`` may be a single ``.json`` scenario file or a directory of them.
    """
    if not path:
        return None
    return load_scenarios(path)


# ---------------------------------------------------------------------------
# lifeform-bench
# ---------------------------------------------------------------------------


def _build_bench_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lifeform-bench",
        description=(
            "Run a scripted multi-turn benchmark on a Lifeform. Default is "
            "the companion vertical with built-in scenarios. Pass "
            "--vertical coding to benchmark the pair-programmer vertical "
            "end-to-end (its scenarios + its DomainExperiencePackage + its "
            "shipped bootstraps), or --scenarios PATH to point at any custom "
            "JSON scenario set."
        ),
    )
    parser.add_argument(
        "--vertical",
        choices=_KNOWN_VERTICALS,
        default=None,
        help=(
            "Resolve scenarios + DomainExperiencePackage + shipped bootstraps "
            "from a known vertical. Default: companion when neither --vertical "
            "nor --scenarios is given. When set, the matching ``build_*_lifeform`` "
            "is used (so e.g. shipped coding bootstraps load automatically)."
        ),
    )
    parser.add_argument(
        "--scenario",
        default="low-mood-disclosure",
        help=(
            'Built-in scenario name (or "all"). Choices: '
            + ", ".join(tuple(_SCENARIOS.keys()) + ("all",))
            + ". Ignored when --scenarios or --vertical is given."
        ),
    )
    parser.add_argument(
        "--scenarios",
        default=None,
        help=(
            "Path to a JSON scenario file or a directory of them. When set, "
            "overrides --scenario; combines with --vertical to use that "
            "vertical's domain pack but a different scenario set."
        ),
    )
    parser.add_argument(
        "--min-regime-match-rate",
        type=float,
        default=0.5,
        help="Minimum regime-match rate for the run to be considered passed (default 0.5).",
    )
    parser.add_argument(
        "--bootstrap-snapshot",
        default=None,
        help="Path to a saved metacontroller snapshot to inject into the Lifeform.",
    )
    parser.add_argument(
        "--regime-bootstrap",
        default=None,
        help="Path to a saved regime calibration artifact to inject into the Lifeform.",
    )
    parser.add_argument(
        "--family-report",
        action="store_true",
        help=(
            "After the flat benchmark output, also print the R12 six-family "
            "evaluation grouping (task / interaction / relationship / "
            "learning / abstraction / safety)."
        ),
    )
    parser.add_argument(
        "--family-report-json",
        default=None,
        help=(
            "Optional path to write the family report(s) as a JSON artifact. "
            "Implies --family-report. The artifact contains a list of "
            "scenario reports keyed by ``scenario_id``."
        ),
    )
    parser.add_argument(
        "--require-family-pass",
        action="store_true",
        help=(
            "Exit non-zero if any family on any scenario fails. Implies "
            "--family-report."
        ),
    )
    parser.add_argument(
        "--companion-evidence-report",
        action="store_true",
        help=(
            "Run the companion C1-C4 evidence report after the benchmark. "
            "Valid only for the companion/default vertical."
        ),
    )
    parser.add_argument(
        "--companion-evidence-json",
        default=None,
        help="Optional path to write the companion evidence report as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_bench_parser()
    args = parser.parse_args(argv)
    bootstrap = load_snapshot_only(args.bootstrap_snapshot) if args.bootstrap_snapshot else None
    if bootstrap is not None:
        print(f"[bench] using bootstrap snapshot: {args.bootstrap_snapshot}")
    regime_bootstrap = (
        load_regime_bootstrap_only(args.regime_bootstrap)
        if args.regime_bootstrap
        else None
    )
    if regime_bootstrap is not None:
        print(f"[bench] using regime bootstrap: {args.regime_bootstrap}")

    vertical_scenarios, _ = _resolve_vertical(args.vertical)
    custom = _resolve_scenarios_flag(args.scenarios)
    if custom is not None:
        scenarios: tuple[ScriptedScenario, ...] = custom
        print(f"[bench] using {len(scenarios)} scenarios from {args.scenarios}")
    elif vertical_scenarios is not None:
        scenarios = vertical_scenarios
        print(f"[bench] using {len(scenarios)} scenarios from --vertical {args.vertical}")
    elif args.scenario == "all":
        scenarios = all_built_in_scenarios()
    else:
        if args.scenario not in _SCENARIOS:
            parser.error(
                f"unknown built-in scenario {args.scenario!r}; "
                f"valid: {sorted(_SCENARIOS.keys()) + ['all']}"
            )
        scenarios = (_SCENARIOS[args.scenario](),)

    # When --vertical is set, build a fully-configured Lifeform via that
    # vertical's factory (so its drives + shipped bootstraps load
    # automatically). When unset, ``run_benchmark`` falls back to its
    # vanilla companion-pack lifeform \u2014 the historic behaviour.
    vertical_lifeform = _build_vertical_lifeform(args.vertical) if args.vertical else None

    want_family_report = (
        args.family_report
        or args.family_report_json is not None
        or args.require_family_pass
    )
    want_companion_evidence = (
        args.companion_evidence_report or args.companion_evidence_json is not None
    )
    ok = True
    family_pass = True
    family_artifacts: list[dict[str, object]] = []
    for scenario in scenarios:
        if vertical_lifeform is not None:
            # Per-scenario fresh session via the vertical lifeform; the
            # same Lifeform instance is reused across scenarios so its
            # Brain (and any pre-loaded bootstraps) is built once.
            report = _run_with_lifeform(
                scenario=scenario, lifeform=vertical_lifeform
            )
        else:
            report = run_benchmark(
                scenario=scenario,
                temporal_bootstrap=bootstrap,
                regime_bootstrap=regime_bootstrap,
            )
        print(format_report(report))
        if want_family_report:
            family = compute_family_report(bench=report)
            print(format_family_report(family))
            family_artifacts.append(family_report_to_dict(family))
            if not family.overall_passed:
                family_pass = False
        if len(scenarios) > 1:
            print()
        ok = ok and report.passed(min_regime_match_rate=args.min_regime_match_rate)

    if args.family_report_json:
        import json
        import pathlib

        path = pathlib.Path(args.family_report_json).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"scenarios": family_artifacts}, indent=2),
            encoding="utf-8",
        )
        print(f"[bench] wrote family report artifact to {path}")

    if want_companion_evidence:
        if args.vertical not in (None, "companion"):
            parser.error("--companion-evidence-report is only valid for the companion vertical")
        companion_evidence = run_companion_evidence()
        print(format_companion_evidence_report(companion_evidence))
        if args.companion_evidence_json:
            import json
            import pathlib

            path = pathlib.Path(args.companion_evidence_json).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(companion_evidence_report_to_dict(companion_evidence), indent=2),
                encoding="utf-8",
            )
            print(f"[bench] wrote companion evidence artifact to {path}")
        ok = ok and companion_evidence.passed

    if args.require_family_pass and not family_pass:
        return 1
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# lifeform-trace
# ---------------------------------------------------------------------------


def _build_trace_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lifeform-trace",
        description=(
            "Run scripted scenarios and emit a line-delimited JSON trace "
            "suitable for offline vz-temporal SSL training. Pass "
            "--scenarios PATH to use a custom JSON pack instead of built-ins."
        ),
    )
    parser.add_argument(
        "--scenario",
        default="all",
        help=(
            'Built-in scenario name (or "all"). Ignored when --scenarios is given.'
        ),
    )
    parser.add_argument(
        "--scenarios",
        default=None,
        help="Path to a JSON scenario file or a directory of them.",
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

    custom = _resolve_scenarios_flag(args.scenarios)
    if custom is not None:
        scenarios: tuple[ScriptedScenario, ...] = custom
    elif args.scenario == "all":
        scenarios = all_built_in_scenarios()
    else:
        if args.scenario not in _SCENARIOS:
            parser.error(f"unknown built-in scenario {args.scenario!r}")
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
    parser.add_argument(
        "--save-snapshot",
        default=None,
        help="Path to write the trained metacontroller snapshot after the loop runs.",
    )
    parser.add_argument(
        "--scenarios",
        default=None,
        help="Path to a JSON scenario file or directory; overrides built-ins.",
    )
    return parser


def main_loop(argv: list[str] | None = None) -> int:
    parser = _build_loop_parser()
    args = parser.parse_args(argv)
    custom = _resolve_scenarios_flag(args.scenarios)
    report = run_learning_loop(
        scenarios=custom,
        n_z=args.n_z,
        alpha=args.alpha,
    )
    print(format_learning_loop_report(report))
    if args.save_snapshot:
        # The single-round loop does not currently expose its trained
        # ``MetacontrollerParameterSnapshot`` on the report. We re-run a
        # tiny SSL pass against the same trace set for the export. This is
        # acceptable because ``run_ssl_demo`` is fast on the built-in
        # scenarios; if a future user needs avoid the duplicate train, we
        # will lift the snapshot onto the report itself. For now, prefer
        # ``lifeform-multi-loop --save-best`` for high-quality snapshots.
        from lifeform_evolution.benchmark import all_built_in_scenarios
        from lifeform_evolution.dataset_adapter import (
            trace_records_to_training_dataset,
        )
        from lifeform_evolution.ssl_demo import run_ssl_demo
        from lifeform_evolution.trace_collector import TraceCollector
        from volvence_zero.temporal import FullLearnedTemporalPolicy

        collector = TraceCollector()
        try:
            for scenario in all_built_in_scenarios():
                collector.collect_scenario(scenario)
        finally:
            collector.close()
        policy = FullLearnedTemporalPolicy()
        run_ssl_demo(
            dataset=trace_records_to_training_dataset(collector.records),
            policy=policy,
            n_z=args.n_z,
            alpha=args.alpha,
        )
        snapshot = policy.export_rare_heavy_snapshot()
        path = save_snapshot(
            snapshot,
            args.save_snapshot,
            metadata={
                "produced_by": "lifeform-loop",
                "scenarios": [s.scenario_id for s in all_built_in_scenarios()],
            },
        )
        print(f"[loop] saved trained snapshot to {path}")
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
    parser.add_argument(
        "--require-improvement-vs-baseline",
        action="store_true",
        help=(
            "Exit non-zero unless at least one trained round shows "
            "improvement-vs-baseline AND coincides with sparse switching "
            "and surface drift (R12 acceptance #12)."
        ),
    )
    parser.add_argument(
        "--save-best",
        default=None,
        help=(
            "Path to write the snapshot of the best round (max drift among "
            "sparse-\u03b2_t rounds). Use it later with "
            "``lifeform-bench --bootstrap-snapshot PATH`` or "
            "``lifeform-loop --bootstrap-snapshot PATH``."
        ),
    )
    parser.add_argument(
        "--scenarios",
        default=None,
        help="Path to a JSON scenario file or directory; overrides built-ins.",
    )
    return parser


def main_multi_loop(argv: list[str] | None = None) -> int:
    parser = _build_multi_loop_parser()
    args = parser.parse_args(argv)
    custom = _resolve_scenarios_flag(args.scenarios)
    report = run_multi_round_loop(
        rounds=args.rounds,
        scenarios=custom,
        n_z=args.n_z,
        alpha=args.alpha,
    )
    print(format_multi_round_report(report))
    if args.save_best:
        best = report.best_round()
        path = save_snapshot(
            best.snapshot,
            args.save_best,
            metadata={
                "scenarios": list(report.scenarios),
                "round_index": best.round_index,
                "rounds_total": len(report.rounds),
                "distance_to_baseline": best.distance_to_baseline,
                "switch_frequency_last": best.ssl.switch_frequency_last,
                "trained_step_count_at_best": best.ssl.trained_step_count,
            },
        )
        print(
            f"[multi-loop] saved best round {best.round_index}'s snapshot to {path}"
        )
    if args.require_trajectory_passes and not report.trajectory_passes():
        return 1
    if args.require_improvement_vs_baseline and not report.verdicts.get(
        "found_pe_aligned_improvement_round", False
    ):
        return 1
    return 0


# ---------------------------------------------------------------------------
# lifeform-calibrate \u2014 trace-driven regime classifier calibration
# ---------------------------------------------------------------------------


def _build_calibrate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lifeform-calibrate",
        description=(
            "Run the trace-driven regime calibrator: per scripted scenario, "
            "compare the kernel's predicted regime against the scenario's "
            "expected_regime_in label set; nudge selection_weights toward "
            "expected regimes; iterate. Output a RegimeBootstrap artifact."
        ),
    )
    parser.add_argument(
        "--rounds", type=int, default=4,
        help="Number of calibration rounds (default 4).",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.18,
        help="Multiplicative update rate for matched regimes (default 0.18).",
    )
    parser.add_argument(
        "--diversity-threshold", type=float, default=0.50,
        help=(
            "Anti-monoculture threshold (default 0.50). When predicted "
            "share of one regime exceeds this, the diversity penalty pulls "
            "its weight back. Set 0.99 to effectively disable."
        ),
    )
    parser.add_argument(
        "--diversity-lr", type=float, default=0.30,
        help=(
            "Diversity penalty strength (default 0.30). Set 0.0 to recover "
            "pre-2026-04-29 behaviour."
        ),
    )
    parser.add_argument(
        "--vertical",
        choices=_KNOWN_VERTICALS,
        default=None,
        help=(
            "Resolve scenarios + DomainExperiencePackage from a known "
            "vertical name (default: companion when neither --vertical nor "
            "--scenarios is given)."
        ),
    )
    parser.add_argument(
        "--scenarios",
        default=None,
        help="Path to a JSON scenario file or directory; overrides --vertical's scenarios.",
    )
    parser.add_argument(
        "--save-best",
        default=None,
        help=(
            "Path to write the best-round regime bootstrap artifact. Use it "
            "later with ``lifeform-bench --regime-bootstrap PATH``."
        ),
    )
    parser.add_argument(
        "--require-improvement",
        action="store_true",
        help=(
            "Exit non-zero if the final round's regime_match_rate is not "
            "strictly better than the baseline's."
        ),
    )
    return parser


def main_calibrate(argv: list[str] | None = None) -> int:
    parser = _build_calibrate_parser()
    args = parser.parse_args(argv)
    vertical_scenarios, vertical_packages = _resolve_vertical(args.vertical)
    custom = _resolve_scenarios_flag(args.scenarios) or vertical_scenarios
    report = run_regime_calibrator(
        rounds=args.rounds,
        scenarios=custom,
        learning_rate=args.learning_rate,
        diversity_threshold=args.diversity_threshold,
        diversity_lr=args.diversity_lr,
        domain_experience_packages=vertical_packages,
    )
    print(format_regime_calibration_report(report))
    if args.save_best:
        best = report.best_round()
        path = save_regime_bootstrap(
            report.final_bootstrap,
            args.save_best,
            metadata={
                "scenarios": list(report.scenarios),
                "round_index_best": best.round_index,
                "rounds_total": len(report.rounds),
                "regime_match_rate_baseline": report.baseline.regime_match_rate,
                "regime_match_rate_best": best.regime_match_rate,
                "regime_match_rate_final": report.final.regime_match_rate,
                "learning_rate": args.learning_rate,
            },
        )
        print(f"[calibrate] saved final regime bootstrap to {path}")
    if args.require_improvement:
        if report.final.regime_match_rate <= report.baseline.regime_match_rate:
            return 1
    return 0


# ---------------------------------------------------------------------------
# lifeform-super-loop \u2014 joint temporal + regime calibration
# ---------------------------------------------------------------------------


def _build_super_loop_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lifeform-super-loop",
        description=(
            "Joint multi-round calibration of the metacontroller (\u03b2_t / z_t) "
            "AND the regime classifier in lockstep. Each round runs scenarios "
            "with the current bootstraps, trains both axes against the same "
            "evidence, and emits the next round's bootstraps."
        ),
    )
    parser.add_argument(
        "--rounds", type=int, default=3,
        help="Number of rounds including round 0 baseline (default 3).",
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
        "--regime-learning-rate", type=float, default=0.18,
        help="Multiplicative update rate for matched regimes (default 0.18).",
    )
    parser.add_argument(
        "--diversity-threshold", type=float, default=0.50,
        help=(
            "Anti-monoculture threshold for the regime calibrator. When one "
            "regime is predicted on more than this fraction of turns, its "
            "weight is pulled back proportional to overuse. Set 0.99 to "
            "effectively disable. Default 0.50."
        ),
    )
    parser.add_argument(
        "--diversity-lr", type=float, default=0.30,
        help=(
            "Strength of the diversity penalty. ``factor = 1 - "
            "diversity_lr * (predicted_share - threshold)``. Set 0.0 to "
            "recover pre-2026-04-29 calibrator behaviour. Default 0.30."
        ),
    )
    parser.add_argument(
        "--vertical",
        choices=_KNOWN_VERTICALS,
        default=None,
        help=(
            "Resolve scenarios AND DomainExperiencePackage from a known "
            "vertical name (default: companion when neither --vertical nor "
            "--scenarios is given). When set, --scenarios is ignored unless "
            "you ALSO need a custom scenario pack on top of the vertical's "
            "domain package."
        ),
    )
    parser.add_argument(
        "--scenarios",
        default=None,
        help=(
            "Path to a JSON scenario file or directory; overrides "
            "--vertical's scenarios but keeps its domain package."
        ),
    )
    parser.add_argument(
        "--save-temporal",
        default=None,
        help="Path to write the best-round's metacontroller snapshot.",
    )
    parser.add_argument(
        "--save-regime",
        default=None,
        help="Path to write the best-round's regime bootstrap artifact.",
    )
    parser.add_argument(
        "--require-trajectory-passes", action="store_true",
        help="Exit non-zero if any per-round trajectory verdict failed.",
    )
    return parser


def main_super_loop(argv: list[str] | None = None) -> int:
    parser = _build_super_loop_parser()
    args = parser.parse_args(argv)
    vertical_scenarios, vertical_packages = _resolve_vertical(args.vertical)
    custom = _resolve_scenarios_flag(args.scenarios) or vertical_scenarios
    report = run_super_loop(
        rounds=args.rounds,
        scenarios=custom,
        n_z=args.n_z,
        alpha=args.alpha,
        regime_learning_rate=args.regime_learning_rate,
        diversity_threshold=args.diversity_threshold,
        diversity_lr=args.diversity_lr,
        domain_experience_packages=vertical_packages,
    )
    print(format_super_loop_report(report))
    if args.save_temporal:
        # Temporal artifact: pick the sparsest-\u03b2_t round (avoids saving
        # an over-trained snapshot from a later round where SSL has
        # collapsed \u03b2_t to 1.0).
        best_t = report.best_temporal_round()
        path = save_snapshot(
            best_t.temporal_snapshot,
            args.save_temporal,
            metadata={
                "produced_by": "lifeform-super-loop",
                "scenarios": list(report.scenarios),
                "round_index_best_temporal": best_t.round_index,
                "rounds_total": len(report.rounds),
                "regime_match_rate_at_best_temporal": best_t.regime_match_rate,
                "switch_frequency_at_best_temporal": best_t.ssl.switch_frequency_last,
            },
        )
        print(
            f"[super-loop] saved temporal snapshot to {path} "
            f"(round {best_t.round_index}, \u03b2_t="
            f"{best_t.ssl.switch_frequency_last:.3f})"
        )
    if args.save_regime:
        # Regime artifact: pick the most-diverse round (avoids saving a
        # monoculture bootstrap that would collapse all turns to one
        # regime). The two artifacts can come from different rounds \u2014
        # they encode different axes.
        best_r = report.best_regime_round()
        path = save_regime_bootstrap(
            best_r.regime_bootstrap,
            args.save_regime,
            metadata={
                "produced_by": "lifeform-super-loop",
                "scenarios": list(report.scenarios),
                "round_index_best_regime": best_r.round_index,
                "rounds_total": len(report.rounds),
                "regime_match_rate_at_best_regime": best_r.regime_match_rate,
                "regime_match_rate_baseline": report.baseline.regime_match_rate,
                "predicted_top_share_at_best_regime": best_r.predicted_regime_share,
            },
        )
        print(
            f"[super-loop] saved regime bootstrap to {path} "
            f"(round {best_r.round_index}, top-share="
            f"{best_r.predicted_regime_share:.0%})"
        )
    if args.require_trajectory_passes and not report.trajectory_passes():
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
