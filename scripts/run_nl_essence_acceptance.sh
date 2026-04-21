#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import asyncio
from volvence_zero.agent import (
    DialogueNLEssenceAcceptanceConfig,
    DialogueRealComprehensiveBenchmarkConfig,
    run_real_dialogue_pe_eta_comprehensive_benchmark,
)
from volvence_zero.substrate import LocalSubstrateRuntimeMode


async def main() -> None:
    report = await run_real_dialogue_pe_eta_comprehensive_benchmark(
        config=DialogueRealComprehensiveBenchmarkConfig(
            runtime_mode=LocalSubstrateRuntimeMode.BUILTIN_ONLY,
            profile_labels=("pe-eta", "eta-no-pe", "heuristic-baseline"),
            canonical_case_limit=2,
            perturbation_variant_limit=1,
            replay_family_limit=1,
            replay_seeds=(0,),
            selection_top_k=1,
            candidate_config_limit=1,
        ),
        essence_acceptance_config=DialogueNLEssenceAcceptanceConfig(),
    )
    print(report.essence_report.description)
    print(report.essence_acceptance.description)
    for gate in report.essence_report.gates:
        print(f"{gate.gate_id}: {'PASS' if gate.passed else 'FAIL'}")
    if not report.essence_acceptance.accepted:
        blocked = ", ".join(report.essence_acceptance.blocked_gate_ids) or "unknown"
        reasons = ", ".join(report.essence_acceptance.reasons) or "unspecified"
        raise SystemExit(
            f"NL essence acceptance failed. blocked_gates=[{blocked}] reasons=[{reasons}]"
        )


asyncio.run(main())
PY
