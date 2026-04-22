#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-artifacts/eta_paper_suite}"
SUITE_TIER="${2:-paper-suite-small}"

python - "$OUTPUT_DIR" "$SUITE_TIER" <<'PY'
import sys
from pathlib import Path

from volvence_zero.agent import (
    build_eta_proof_paper_suite_manifest,
    export_eta_internal_rl_paper_suite_artifact_bundle,
    run_eta_internal_rl_paper_suite,
)


def main(output_dir: str, suite_tier: str) -> None:
    manifest = build_eta_proof_paper_suite_manifest(suite_tier=suite_tier)
    report = run_eta_internal_rl_paper_suite(
        manifest=manifest,
        output_dir=output_dir,
    )
    export_eta_internal_rl_paper_suite_artifact_bundle(
        report,
        output_dir=Path(output_dir),
    )
    print(report.description)
    for summary in report.primary_metric_summaries:
        print(
            f"{summary.metric_name}: mean={summary.mean:.3f} "
            f"ci=[{summary.ci_low:.3f}, {summary.ci_high:.3f}]"
        )


main(sys.argv[1], sys.argv[2])
PY
