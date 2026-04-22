#!/usr/bin/env bash
set -euo pipefail

OUTPUT_PATH="${1:-artifacts/emergence_dashboard.json}"

python - "$OUTPUT_PATH" <<'PY'
import asyncio
import sys
from pathlib import Path

from volvence_zero.agent import (
    DialogueNLEssenceAcceptanceConfig,
    default_dialogue_real_proof_config,
    export_dialogue_emergence_dashboard_artifact,
    run_real_dialogue_pe_eta_comprehensive_benchmark,
)
from volvence_zero.substrate import LocalSubstrateRuntimeMode


async def main(output_path: str) -> None:
    report = await run_real_dialogue_pe_eta_comprehensive_benchmark(
        config=default_dialogue_real_proof_config(
            runtime_mode=LocalSubstrateRuntimeMode.BUILTIN_ONLY,
        ),
        essence_acceptance_config=DialogueNLEssenceAcceptanceConfig(),
    )
    path = export_dialogue_emergence_dashboard_artifact(
        report,
        output_path=output_path,
    )
    print(f"wrote emergence dashboard to {Path(path).resolve()}")
    print(report.emergence_dashboard.description)


asyncio.run(main(sys.argv[1]))
PY
