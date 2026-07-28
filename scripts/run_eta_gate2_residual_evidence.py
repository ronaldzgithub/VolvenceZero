from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from volvence_zero.agent import (
    ETAOpenWeightRuntimeConfig,
    build_eta_gate2_residual_manifest,
    export_eta_gate2_residual_bundle,
    run_eta_internal_rl_paper_suite,
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
    parser.add_argument(
        "--model-id",
        default=ETAOpenWeightRuntimeConfig().model_id,
    )
    parser.add_argument("--model-source", default=None)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if args.seeds is not None and args.seeds < 1:
        parser.error("--seeds must be at least 1")
    if args.train_epochs is not None and args.train_epochs < 1:
        parser.error("--train-epochs must be at least 1")
    if args.activation_width is not None and args.activation_width < 1:
        parser.error("--activation-width must be at least 1")

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
    report = run_eta_internal_rl_paper_suite(
        manifest=manifest,
        open_weight_config=config,
        output_dir=args.output_dir,
    )
    written = export_eta_gate2_residual_bundle(
        report,
        output_dir=args.output_dir,
    )
    verdict = json.loads(
        (args.output_dir / "promotion_verdict.json").read_text(
            encoding="utf-8"
        )
    )
    print(
        json.dumps(
            {
                "suite_id": report.manifest.suite_id,
                "status": verdict["status"],
                "kill_conditions": verdict["kill_conditions"],
                "artifact_files": [path.name for path in written],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
