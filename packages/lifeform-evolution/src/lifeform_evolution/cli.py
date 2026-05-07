"""``lifeform-bench`` and ``lifeform-trace`` CLI entry points."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from lifeform_evolution.benchmark import (
    ScriptedScenario,
    all_built_in_scenarios,
    casual_social_checkin_scenario,
    emotional_decision_support_scenario,
    format_report,
    low_mood_disclosure_scenario,
    run_benchmark,
    trust_rupture_repair_scenario,
)
from lifeform_evolution.family_report import (
    FamilyReport,
    compute_family_report,
    family_report_to_dict,
    format_family_report,
)
from lifeform_evolution.longitudinal_family_report import (
    compute_longitudinal_family_report,
    format_longitudinal_family_report,
    longitudinal_family_report_to_dict,
)
from lifeform_evolution.companion_evidence import (
    companion_evidence_report_to_dict,
    format_companion_evidence_report,
    run_companion_evidence,
)
from lifeform_evolution.closed_alpha_preflight import (
    format_closed_alpha_preflight_report,
    run_closed_alpha_preflight,
)
from lifeform_evolution.social_cognition_evidence import (
    format_social_cognition_evidence_report,
    run_social_cognition_evidence,
    social_cognition_evidence_report_to_dict,
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
from lifeform_evolution.relationship_repair_alpha_gate import (
    format_relationship_repair_alpha_report,
    run_relationship_repair_alpha_gate,
)
from lifeform_evolution.super_loop import (
    format_super_loop_report,
    run_super_loop,
)
from lifeform_evolution.snapshot_io import (
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
    "emotional-decision-support": emotional_decision_support_scenario,
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
        "--longitudinal-rounds",
        type=int,
        default=0,
        help=(
            "Phase 2 W2.C cross-session evidence. Run every selected "
            "scenario N times sequentially against the SAME lifeform "
            "instance so memory + session-post slow loop carry forward "
            "across rounds. After all rounds, aggregate the per-round "
            "F3 metrics into a longitudinal report. 0 (default) "
            "disables the longitudinal pass; 3 to 5 is the recommended "
            "minimum for meaningful trend evidence. Requires "
            "--vertical (the longitudinal pass needs a stable Lifeform "
            "instance to share memory across rounds)."
        ),
    )
    parser.add_argument(
        "--longitudinal-report",
        action="store_true",
        help=(
            "Print the longitudinal family report after running "
            "--longitudinal-rounds. Implies --family-report."
        ),
    )
    parser.add_argument(
        "--longitudinal-json",
        default=None,
        help=(
            "Optional path to write the longitudinal report as JSON. "
            "Implies --longitudinal-report."
        ),
    )
    parser.add_argument(
        "--require-longitudinal-pass",
        action="store_true",
        help=(
            "Exit non-zero if the longitudinal acceptance gate "
            "(``trust_no_drift`` AND ``continuity_improved_vs_baseline``) "
            "fails. Implies --longitudinal-report and "
            "--longitudinal-rounds > 0."
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
    parser.add_argument(
        "--social-cognition-evidence-report",
        action="store_true",
        help="Run the social cognition R16-R20 evidence report after the benchmark.",
    )
    parser.add_argument(
        "--social-cognition-evidence-json",
        default=None,
        help="Optional path to write the social cognition evidence report as JSON.",
    )
    parser.add_argument(
        "--eta-open-weight-paper-suite",
        action="store_true",
        help=(
            "Run the ETA open-weight residual paper suite using the real "
            "Qwen/Qwen2.5-0.5B-Instruct backend. Pre-cache the model with "
            "`huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct` once."
        ),
    )
    parser.add_argument(
        "--eta-open-weight-tier",
        choices=("ci-smoke", "paper-suite-small", "paper-suite-full"),
        default="ci-smoke",
        help=(
            "Tier for the ETA open-weight paper suite. ci-smoke verifies "
            "wiring only (caps claim status at weak); paper-suite-small "
            "and paper-suite-full are required for retain. Default ci-smoke."
        ),
    )
    parser.add_argument(
        "--eta-open-weight-output-dir",
        default=None,
        help=(
            "Output directory for the ETA open-weight paper suite artifact "
            "bundle. Default: artifacts/eta_open_weight/<tier>/."
        ),
    )
    parser.add_argument(
        "--eta-open-weight-require-retain",
        action="store_true",
        help=(
            "Exit non-zero if claim_eta_real_open_weight_residual_control "
            "is not retain. ci-smoke can never reach retain, so this should "
            "only be combined with paper-suite-small or paper-suite-full."
        ),
    )
    parser.add_argument(
        "--require-sparse-reward-heldout",
        action="store_true",
        help=(
            "Phase 2 W2.D acceptance gate. Exit non-zero unless the ETA "
            "open-weight paper suite produced a 'retain' verdict for "
            "``claim_eta_internal_rl_sparse_reward_advantage`` (held-out "
            "sparse-reward success above matched controls). Implies "
            "--eta-open-weight-paper-suite. The verdict surfaces the "
            "F4 (learning quality) acceptance question 'does the system "
            "improve from sparse, delayed signal?'."
        ),
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
        # Phase 2 W2.C: the longitudinal pass aggregates per-round
        # FamilyReport instances, so it implicitly needs the family
        # report to be computed each round.
        or args.longitudinal_rounds > 0
        or args.longitudinal_report
        or args.longitudinal_json is not None
        or args.require_longitudinal_pass
    )
    want_longitudinal_report = (
        args.longitudinal_rounds > 0
        or args.longitudinal_report
        or args.longitudinal_json is not None
        or args.require_longitudinal_pass
    )
    if want_longitudinal_report and args.longitudinal_rounds <= 0:
        parser.error(
            "--longitudinal-report / --longitudinal-json / "
            "--require-longitudinal-pass require --longitudinal-rounds > 0"
        )
    if want_longitudinal_report and vertical_lifeform is None:
        parser.error(
            "longitudinal pass needs --vertical so the same Lifeform "
            "instance can be reused across rounds (memory + session-post "
            "slow loop carry forward only on a shared Lifeform)."
        )
    want_companion_evidence = (
        args.companion_evidence_report or args.companion_evidence_json is not None
    )
    want_social_cognition_evidence = (
        args.social_cognition_evidence_report
        or args.social_cognition_evidence_json is not None
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

    longitudinal_pass = True
    longitudinal_artifacts: list[dict[str, object]] = []
    if want_longitudinal_report:
        for scenario in scenarios:
            per_round_reports: list[FamilyReport] = []
            for round_index in range(args.longitudinal_rounds):
                report = _run_with_lifeform(
                    scenario=scenario, lifeform=vertical_lifeform
                )
                per_round_reports.append(compute_family_report(bench=report))
                print(
                    f"[bench] longitudinal round {round_index + 1}/"
                    f"{args.longitudinal_rounds} for scenario "
                    f"{scenario.scenario_id}: "
                    f"closed_scenes={report.closed_scene_count}"
                )
            longitudinal = compute_longitudinal_family_report(
                tuple(per_round_reports)
            )
            print(format_longitudinal_family_report(longitudinal))
            longitudinal_artifacts.append(
                longitudinal_family_report_to_dict(longitudinal)
            )
            if not longitudinal.passed:
                longitudinal_pass = False

    if args.longitudinal_json:
        import json
        import pathlib

        path = pathlib.Path(args.longitudinal_json).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"scenarios": longitudinal_artifacts}, indent=2),
            encoding="utf-8",
        )
        print(f"[bench] wrote longitudinal report artifact to {path}")

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

    if want_social_cognition_evidence:
        social_cognition_evidence = run_social_cognition_evidence()
        print(format_social_cognition_evidence_report(social_cognition_evidence))
        if args.social_cognition_evidence_json:
            import json
            import pathlib

            path = pathlib.Path(args.social_cognition_evidence_json).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    social_cognition_evidence_report_to_dict(social_cognition_evidence),
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[bench] wrote social cognition evidence artifact to {path}")
        ok = ok and social_cognition_evidence.passed

    sparse_reward_retain = False
    if args.require_sparse_reward_heldout and not args.eta_open_weight_paper_suite:
        parser.error(
            "--require-sparse-reward-heldout implies "
            "--eta-open-weight-paper-suite (the gate reads the paper "
            "suite's claim verdicts)."
        )
    if args.eta_open_weight_paper_suite:
        eta_result = _run_eta_open_weight_paper_suite_from_bench(args=args)
        ok = eta_result.passed and ok
        sparse_reward_retain = eta_result.sparse_reward_advantage_retain
        if args.require_sparse_reward_heldout:
            print(
                "[bench] sparse-reward held-out gate: "
                f"status={eta_result.sparse_reward_advantage_status} "
                f"retain={sparse_reward_retain}"
            )

    if args.require_family_pass and not family_pass:
        return 1
    if args.require_longitudinal_pass and not longitudinal_pass:
        return 1
    if args.require_sparse_reward_heldout and not sparse_reward_retain:
        return 1
    return 0 if ok else 1


@dataclass(frozen=True)
class _EtaPaperSuiteRunResult:
    """Outcome of one ``--eta-open-weight-paper-suite`` invocation.

    Two views:

    * ``passed`` — the historical bool the CLI used to gate the
      overall exit code. ``--eta-open-weight-require-retain`` flips
      this to False when the real residual claim is not retain.
    * ``sparse_reward_advantage_*`` — typed surface for the Phase 2
      W2.D ``--require-sparse-reward-heldout`` gate. The CLI reads
      these without re-parsing the artifact bundle.
    """

    passed: bool
    sparse_reward_advantage_status: str
    sparse_reward_advantage_retain: bool


def _run_eta_open_weight_paper_suite_from_bench(
    *, args: argparse.Namespace
) -> _EtaPaperSuiteRunResult:
    import json
    import pathlib

    from volvence_zero.agent import (
        ETAOpenWeightRuntimeConfig,
        build_eta_open_weight_paper_suite_manifest,
        export_eta_internal_rl_paper_suite_artifact_bundle,
        run_eta_internal_rl_paper_suite,
    )

    tier: str = args.eta_open_weight_tier
    output_dir = pathlib.Path(
        args.eta_open_weight_output_dir or f"artifacts/eta_open_weight/{tier}"
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ETAOpenWeightRuntimeConfig()
    manifest = build_eta_open_weight_paper_suite_manifest(suite_tier=tier)
    print(
        f"[bench] ETA open-weight paper suite tier={tier} "
        f"model={config.model_id} layers={config.layer_indices} "
        f"output_dir={output_dir}"
    )
    report = run_eta_internal_rl_paper_suite(
        manifest=manifest,
        open_weight_config=config,
        output_dir=str(output_dir),
    )
    export_eta_internal_rl_paper_suite_artifact_bundle(report, output_dir=output_dir)

    real_claim = next(
        (
            verdict
            for verdict in report.claim_verdicts
            if verdict.claim_id == "claim_eta_real_open_weight_residual_control"
        ),
        None,
    )
    print(
        "[bench] ETA open-weight paper suite claim verdicts:"
    )
    for verdict in report.claim_verdicts:
        print(f"[bench]   {verdict.claim_id}: {verdict.status}")
    if real_claim is not None:
        print("[bench] real claim evidence:")
        for key, value in real_claim.evidence:
            print(f"[bench]   {key}: {value}")
    print(
        f"[bench] ETA open-weight paper suite bundle exported to {output_dir.resolve()}"
    )

    provenance_descriptor = dict(report.provenance.runtime_descriptor)
    print("[bench] runtime provenance descriptor:")
    for key in sorted(provenance_descriptor):
        print(f"[bench]   {key}: {provenance_descriptor[key]}")
    runtime_provenance_path = output_dir / "eta_open_weight_runtime_provenance.json"
    runtime_provenance_path.write_text(
        json.dumps(
            {
                "git_sha": report.provenance.git_sha,
                "git_branch": report.provenance.git_branch,
                "working_tree_dirty": report.provenance.working_tree_dirty,
                "python_version": report.provenance.python_version,
                "platform": report.provenance.platform,
                "runtime_descriptor": provenance_descriptor,
                "manifest_hash": report.provenance.manifest_hash,
                "dependency_digest": report.provenance.dependency_digest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[bench] runtime provenance written to {runtime_provenance_path}"
    )

    sparse_reward_claim = next(
        (
            verdict
            for verdict in report.claim_verdicts
            if verdict.claim_id == "claim_eta_internal_rl_sparse_reward_advantage"
        ),
        None,
    )
    sparse_reward_status = (
        sparse_reward_claim.status if sparse_reward_claim is not None else "missing"
    )
    sparse_reward_retain = (
        sparse_reward_claim is not None and sparse_reward_claim.status == "retain"
    )

    overall_passed = True
    if args.eta_open_weight_require_retain:
        if real_claim is None or real_claim.status != "retain":
            status = "missing" if real_claim is None else real_claim.status
            print(
                f"[bench] ERROR: --eta-open-weight-require-retain set, "
                f"but claim status is {status} (need retain)."
            )
            overall_passed = False
    return _EtaPaperSuiteRunResult(
        passed=overall_passed,
        sparse_reward_advantage_status=sparse_reward_status,
        sparse_reward_advantage_retain=sparse_reward_retain,
    )


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


# ---------------------------------------------------------------------------
# lifeform-repair-alpha-gate — closed-alpha repair evidence
# ---------------------------------------------------------------------------


def _build_repair_alpha_gate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lifeform-repair-alpha-gate",
        description=(
            "Run the closed-alpha relationship repair matched-control gate. "
            "The report verifies typed rupture observation, repair-alpha "
            "expression, observed repair memory, same-user recall, and "
            "cross-user isolation."
        ),
    )
    parser.add_argument(
        "--out",
        default="artifacts/relationship_repair_alpha_gate/report.json",
        help="Path to write the JSON gate report.",
    )
    parser.add_argument(
        "--scope-root",
        default="artifacts/relationship_repair_alpha_gate/scope",
        help="Filesystem root for scoped memory stores used by the gate.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human-readable summary; JSON is still written.",
    )
    return parser


def main_repair_alpha_gate(argv: list[str] | None = None) -> int:
    parser = _build_repair_alpha_gate_parser()
    args = parser.parse_args(argv)
    report = run_relationship_repair_alpha_gate(
        out_path=args.out,
        scope_root_dir=args.scope_root,
    )
    if not args.quiet:
        print(format_relationship_repair_alpha_report(report))
        print(f"[repair-alpha-gate] wrote report to {args.out}")
    return 0 if report.passed else 1


# ---------------------------------------------------------------------------
# lifeform-alpha-preflight — aggregate closed-alpha gates
# ---------------------------------------------------------------------------


def _build_alpha_preflight_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lifeform-alpha-preflight",
        description=(
            "Run closed-alpha preflight gates and write a single manifest. "
            "Currently includes the open-dialogue v0 gate and the "
            "relationship repair alpha gate."
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts/closed_alpha_preflight",
        help="Directory where gate artifacts and the preflight report are written.",
    )
    parser.add_argument(
        "--scope-root",
        default="artifacts/closed_alpha_preflight_scope",
        help="Filesystem root for scoped memory stores used by gates.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable summary; JSON artifacts are still written.",
    )
    return parser


def main_alpha_preflight(argv: list[str] | None = None) -> int:
    parser = _build_alpha_preflight_parser()
    args = parser.parse_args(argv)
    report = run_closed_alpha_preflight(
        artifacts_dir=args.artifacts_dir,
        scope_root_dir=args.scope_root,
    )
    if not args.quiet:
        print(format_closed_alpha_preflight_report(report))
        print(
            "[alpha-preflight] wrote report to "
            f"{args.artifacts_dir}/closed_alpha_preflight_report.json"
        )
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
