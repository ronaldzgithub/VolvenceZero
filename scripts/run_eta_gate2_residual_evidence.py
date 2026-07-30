from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from volvence_zero.agent import (
    ETAOpenWeightRuntimeConfig,
    build_eta_gate2_control_summary_diagnostic_manifest,
    build_eta_gate2_recent_k_diagnostic_manifest,
    build_eta_gate2_recent_k2_formal_manifest,
    build_eta_gate2_residual_manifest,
    export_eta_gate2_recent_k_diagnostic_bundle,
    export_eta_gate2_control_summary_diagnostic_bundle,
    export_eta_gate2_residual_bundle,
    run_eta_internal_rl_paper_suite,
)

V36_SOURCE_ARTIFACT = (
    "artifacts/"
    "eta_gate2_residual_causal_v36_shadow_fullwidth896_"
    "qwen25_05b_cpu_1seed_probe_20260729"
)
V37_SOURCE_ARTIFACT = (
    "artifacts/"
    "eta_gate2_v37_recent_k2_fresh_formal_probe_fullwidth896_"
    "qwen25_05b_cpu_seed0_20260730"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run ETA Gate 2 identity/zero/shuffled/reversed residual-control "
            "matched ablations."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--suite-tier",
        choices=("ci-smoke", "paper-suite-small", "paper-suite-full"),
        default="ci-smoke",
    )
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--train-epochs", type=int, default=None)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--activation-width", type=int, default=None)
    parser.add_argument("--max-prefix-steps", type=int, default=None)
    parser.add_argument("--target-samples", type=int, default=None)
    parser.add_argument("--audit-samples", type=int, default=None)
    parser.add_argument(
        "--model-id",
        default=ETAOpenWeightRuntimeConfig().model_id,
    )
    parser.add_argument("--model-source", default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--shadow-control-window",
        type=int,
        choices=(1, 2),
        default=None,
        help=(
            "Run a development-only v36 recent-k diagnostic. Reuses "
            "observed v36 routes and cannot emit a formal promotion verdict."
        ),
    )
    parser.add_argument(
        "--recent-k2-formal",
        action="store_true",
        help=(
            "Run the preregistered v37 recent-k=2 fresh formal SHADOW "
            "admission suite."
        ),
    )
    parser.add_argument(
        "--control-summary-development",
        action="store_true",
        help=(
            "Run the preregistered v38 development-only bounded "
            "committed-control summary state diagnostic on observed v36 "
            "routes."
        ),
    )
    args = parser.parse_args()
    if args.seeds is not None and args.seeds < 1:
        parser.error("--seeds must be at least 1")
    if (
        args.shadow_control_window is not None
        and args.seeds not in {None, 1}
    ):
        parser.error(
            "recent-k reuses observed v36 routes and is limited to one "
            "development seed"
        )
    selected_special_modes = sum(
        (
            args.shadow_control_window is not None,
            args.recent_k2_formal,
            args.control_summary_development,
        )
    )
    if selected_special_modes > 1:
        parser.error(
            "--shadow-control-window, --recent-k2-formal, and "
            "--control-summary-development are mutually exclusive"
        )
    if args.recent_k2_formal:
        if args.suite_tier != "ci-smoke":
            parser.error(
                "v37 recent-k2 formal is frozen to suite-tier=ci-smoke"
            )
        if args.seeds not in {None, 1, 3}:
            parser.error(
                "v37 recent-k2 formal only permits the seed-0 probe or "
                "the frozen three-seed schedule"
            )
        if args.train_epochs not in {None, 2}:
            parser.error("v37 recent-k2 formal freezes train epochs at 2")
        if args.device != "cpu":
            parser.error("v37 recent-k2 formal requires device=cpu")
        if args.activation_width not in {None, 896}:
            parser.error(
                "v37 recent-k2 formal freezes activation width at 896"
            )
        if args.max_prefix_steps not in {None, 8}:
            parser.error(
                "v37 recent-k2 formal freezes max prefix steps at 8"
            )
        if args.target_samples not in {None, 1}:
            parser.error(
                "v37 recent-k2 formal freezes target samples at 1"
            )
        if args.audit_samples not in {None, 1}:
            parser.error(
                "v37 recent-k2 formal freezes audit samples at 1"
            )
        if args.allow_download:
            parser.error(
                "v37 recent-k2 formal requires the frozen local model"
            )
    if args.control_summary_development:
        if args.suite_tier != "ci-smoke":
            parser.error(
                "v38 control-summary development is frozen to "
                "suite-tier=ci-smoke"
            )
        if args.seeds not in {None, 1}:
            parser.error(
                "v38 control-summary development is limited to seed 0"
            )
        if args.train_epochs not in {None, 2}:
            parser.error(
                "v38 control-summary development freezes train epochs at 2"
            )
        if args.device != "cpu":
            parser.error(
                "v38 control-summary development requires device=cpu"
            )
        if args.activation_width not in {None, 896}:
            parser.error(
                "v38 control-summary development freezes activation "
                "width at 896"
            )
        if args.max_prefix_steps not in {None, 8}:
            parser.error(
                "v38 control-summary development freezes max prefix "
                "steps at 8"
            )
        if args.target_samples not in {None, 1}:
            parser.error(
                "v38 control-summary development freezes target samples "
                "at 1"
            )
        if args.audit_samples not in {None, 1}:
            parser.error(
                "v38 control-summary development freezes audit samples "
                "at 1"
            )
        if args.allow_download:
            parser.error(
                "v38 control-summary development requires the frozen "
                "local model"
            )
    if args.train_epochs is not None and args.train_epochs < 1:
        parser.error("--train-epochs must be at least 1")
    if args.activation_width is not None and args.activation_width < 1:
        parser.error("--activation-width must be at least 1")
    if args.max_prefix_steps is not None and args.max_prefix_steps < 3:
        parser.error(
            "--max-prefix-steps must be at least 3 so the realized "
            "primary and audit segments both exist"
        )
    if args.target_samples is not None and args.target_samples < 1:
        parser.error("--target-samples must be at least 1")
    if args.audit_samples is not None and args.audit_samples < 1:
        parser.error("--audit-samples must be at least 1")

    if args.control_summary_development:
        manifest = build_eta_gate2_control_summary_diagnostic_manifest(
            suite_tier=args.suite_tier,
        )
    elif args.recent_k2_formal:
        manifest = build_eta_gate2_recent_k2_formal_manifest(
            suite_tier=args.suite_tier,
        )
    elif args.shadow_control_window is not None:
        manifest = build_eta_gate2_recent_k_diagnostic_manifest(
            committed_control_window=args.shadow_control_window,
            suite_tier=args.suite_tier,
        )
    else:
        manifest = build_eta_gate2_residual_manifest(
            suite_tier=args.suite_tier,
        )
    if args.seeds is not None:
        manifest = replace(
            manifest,
            repeat_count=args.seeds,
            seed_schedule=tuple(range(args.seeds)),
        )
    if args.train_epochs is not None:
        manifest = replace(
            manifest,
            case_groups=tuple(
                (
                    name,
                    (str(args.train_epochs),)
                    if name == "train_epochs"
                    else values,
                )
                for name, values in manifest.case_groups
            ),
        )
    if args.activation_width is not None:
        manifest = replace(
            manifest,
            case_groups=tuple(
                (
                    name,
                    (str(args.activation_width),)
                    if name == "real_residual_activation_width"
                    else values,
                )
                for name, values in manifest.case_groups
            ),
        )
    sample_overrides = {
        "counterfactual_target_sample_count": args.target_samples,
        "counterfactual_audit_sample_count": args.audit_samples,
    }
    if any(value is not None for value in sample_overrides.values()):
        manifest = replace(
            manifest,
            case_groups=tuple(
                (
                    name,
                    (str(sample_overrides[name]),)
                    if name in sample_overrides
                    and sample_overrides[name] is not None
                    else values,
                )
                for name, values in manifest.case_groups
            ),
        )
    manifest_case_groups = dict(manifest.case_groups)
    activation_width = (
        args.activation_width
        if args.activation_width is not None
        else int(
            manifest_case_groups.get(
                "real_residual_activation_width",
                ("8",),
            )[0]
        )
    )
    config = ETAOpenWeightRuntimeConfig(
        model_id=args.model_id,
        model_source=args.model_source,
        device=args.device,
        activation_width=activation_width,
        local_files_only=not args.allow_download,
    )
    if args.recent_k2_formal or args.control_summary_development:
        config = replace(config, max_prefix_steps=8)
    elif args.max_prefix_steps is not None:
        config = replace(config, max_prefix_steps=args.max_prefix_steps)
    report = run_eta_internal_rl_paper_suite(
        manifest=manifest,
        open_weight_config=config,
        output_dir=(
            None
            if (
                args.shadow_control_window is not None
                or args.control_summary_development
            )
            else args.output_dir
        ),
    )
    if args.control_summary_development:
        written = export_eta_gate2_control_summary_diagnostic_bundle(
            report,
            output_dir=args.output_dir,
            source_v37_artifact=V37_SOURCE_ARTIFACT,
        )
        diagnostic = json.loads(
            (
                args.output_dir / "control_summary_diagnostic.json"
            ).read_text(encoding="utf-8")
        )
        status = (
            "development-pass"
            if diagnostic["development_gate_passed"]
            else "development-fail"
        )
        kill_conditions = []
    elif args.shadow_control_window is not None:
        written = export_eta_gate2_recent_k_diagnostic_bundle(
            report,
            output_dir=args.output_dir,
            source_v36_artifact=V36_SOURCE_ARTIFACT,
        )
        diagnostic = json.loads(
            (args.output_dir / "recent_k_diagnostic.json").read_text(
                encoding="utf-8"
            )
        )
        status = (
            "development-pass"
            if diagnostic["development_gate_passed"]
            else "development-fail"
        )
        kill_conditions = []
    else:
        written = export_eta_gate2_residual_bundle(
            report,
            output_dir=args.output_dir,
        )
        verdict = json.loads(
            (args.output_dir / "promotion_verdict.json").read_text(
                encoding="utf-8"
            )
        )
        status = verdict["status"]
        kill_conditions = verdict["kill_conditions"]
        if args.recent_k2_formal:
            if report.manifest.repeat_count == 1:
                status = (
                    "formal-probe-go"
                    if verdict["shadow_single_seed_stoploss_passed"]
                    else "formal-probe-no-go"
                )
            else:
                status = (
                    "shadow-admission-supported"
                    if verdict["shadow_admission_allowed"]
                    else "shadow-admission-not-supported"
                )
    print(
        json.dumps(
            {
                "suite_id": report.manifest.suite_id,
                "status": status,
                "kill_conditions": kill_conditions,
                "artifact_files": [path.name for path in written],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
