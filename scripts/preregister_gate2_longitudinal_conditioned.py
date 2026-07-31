"""Freeze the Gate 2 relationship-conditioned longitudinal packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from volvence_zero.agent.gate2_longitudinal_capture import (
    build_gate2_longitudinal_capture_runtime,
)
from volvence_zero.agent.gate2_longitudinal_conditioned import (
    build_gate2_conditioned_preregistration,
    validate_gate2_conditioned_preregistration,
    write_gate2_conditioned_preregistration,
)
from volvence_zero.agent.shared_settled_trace import (
    shared_trace_runtime_fingerprint,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--candidate-artifact",
        type=Path,
        default=Path(
            "artifacts/"
            "eta_gate2_residual_causal_v35_selector_null_fresh_"
            "fullwidth896_qwen25_05b_cpu_1seed_20260729/"
            "counterfactual_outcomes.jsonl"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runtime, _basis = build_gate2_longitudinal_capture_runtime()
    payload = build_gate2_conditioned_preregistration(
        repo_root=args.repo_root,
        candidate_artifact_path=args.candidate_artifact,
        substrate_fingerprint=shared_trace_runtime_fingerprint(runtime),
    )
    validate_gate2_conditioned_preregistration(
        payload,
        repo_root=args.repo_root,
    )
    manifest = write_gate2_conditioned_preregistration(
        payload=payload,
        output_path=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
