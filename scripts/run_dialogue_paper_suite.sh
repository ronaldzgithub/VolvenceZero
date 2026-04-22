#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-artifacts/dialogue_paper_suite}"
SUITE_TIER="${2:-paper-suite-small}"

python - "$OUTPUT_DIR" "$SUITE_TIER" <<'PY'
import asyncio
import sys
from pathlib import Path

from volvence_zero.agent import (
    DialogueNLEssenceAcceptanceConfig,
    build_dialogue_paper_suite_manifest,
    export_dialogue_paper_suite_artifact_bundle,
    run_dialogue_paper_suite_repeated_benchmark,
)
from volvence_zero.substrate import LocalSubstrateRuntimeMode


async def main(output_dir: str, suite_tier: str) -> None:
    manifest = build_dialogue_paper_suite_manifest(suite_tier=suite_tier)
    report = await run_dialogue_paper_suite_repeated_benchmark(
        manifest=manifest,
        runtime_mode=LocalSubstrateRuntimeMode.BUILTIN_ONLY,
        output_dir=output_dir,
        progress_callback=print,
    )
    export_dialogue_paper_suite_artifact_bundle(
        report,
        output_dir=Path(output_dir),
    )
    print(report.description)
    for summary in report.primary_metric_summaries:
        print(
            f"{summary.metric_name}: mean={summary.mean:.3f} "
            f"ci=[{summary.ci_low:.3f}, {summary.ci_high:.3f}]"
        )


asyncio.run(main(sys.argv[1], sys.argv[2]))
PY
