#!/usr/bin/env bash
# Run the ETA open-weight residual paper suite using the real
# Qwen/Qwen2.5-0.5B-Instruct backend (default ETAOpenWeightRuntimeConfig).
#
# Required: Qwen/Qwen2.5-0.5B-Instruct must be available in the local HF
# cache (run `huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct` once).
#
# Usage:
#   bash scripts/run_eta_open_weight_paper_suite.sh [OUTPUT_DIR] [SUITE_TIER]
# Defaults:
#   OUTPUT_DIR=artifacts/eta_open_weight_paper_suite
#   SUITE_TIER=paper-suite-small  (alternatives: ci-smoke, paper-suite-full)
set -euo pipefail

OUTPUT_DIR="${1:-artifacts/eta_open_weight_paper_suite}"
SUITE_TIER="${2:-paper-suite-small}"

python - "$OUTPUT_DIR" "$SUITE_TIER" <<'PY'
import json
import sys
from pathlib import Path

from volvence_zero.agent import (
    ETAOpenWeightRuntimeConfig,
    build_eta_open_weight_paper_suite_manifest,
    export_eta_internal_rl_paper_suite_artifact_bundle,
    run_eta_internal_rl_paper_suite,
)


def main(output_dir: str, suite_tier: str) -> None:
    manifest = build_eta_open_weight_paper_suite_manifest(suite_tier=suite_tier)
    open_weight_config = ETAOpenWeightRuntimeConfig()
    report = run_eta_internal_rl_paper_suite(
        manifest=manifest,
        open_weight_config=open_weight_config,
        output_dir=output_dir,
    )
    output_path = Path(output_dir)
    export_eta_internal_rl_paper_suite_artifact_bundle(report, output_dir=output_path)

    print(report.description)
    print()
    print("Primary metric summaries:")
    for summary in report.primary_metric_summaries:
        print(
            f"  {summary.metric_name}: mean={summary.mean:.3f} "
            f"ci=[{summary.ci_low:.3f}, {summary.ci_high:.3f}]"
        )

    print()
    print("Real open-weight evidence (secondary):")
    real_metric_names = {
        "real_open_weight_capture_rate",
        "real_open_weight_hook_coverage",
        "real_open_weight_token_step_coverage",
        "real_open_weight_residual_sequence_present",
        "real_open_weight_intervention_protocol_valid",
        "real_open_weight_fallback_rate",
        "intervention_application_count",
        "episode_replacement_effect_delta",
        "residual_signal_quality",
    }
    for summary in report.secondary_metric_summaries:
        if summary.metric_name in real_metric_names:
            print(
                f"  {summary.metric_name}: mean={summary.mean:.3f} "
                f"ci=[{summary.ci_low:.3f}, {summary.ci_high:.3f}]"
            )

    print()
    print("Claim verdicts:")
    for verdict in report.claim_verdicts:
        print(f"  {verdict.claim_id}: status={verdict.status}")
    real_claim = next(
        (verdict for verdict in report.claim_verdicts if verdict.claim_id == "claim_eta_real_open_weight_residual_control"),
        None,
    )
    if real_claim is not None:
        print()
        print("claim_eta_real_open_weight_residual_control evidence:")
        for key, value in real_claim.evidence:
            print(f"  {key}: {value}")
    print()
    print(f"Bundle exported to: {output_path.resolve()}")


main(sys.argv[1], sys.argv[2])
PY
