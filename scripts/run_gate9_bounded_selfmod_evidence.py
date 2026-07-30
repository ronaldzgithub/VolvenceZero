from __future__ import annotations

import argparse
import json
from pathlib import Path

from volvence_zero.agent.gate9_bounded_selfmod_evidence import (
    export_gate9_evidence_bundle,
    run_gate9_evidence,
    verify_gate9_evidence_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered Gate 9 M3 and PE-gated matched controls."
        )
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if (args.output_dir is None) == (args.verify is None):
        parser.error("choose exactly one of --output-dir or --verify")
    if args.verify is not None:
        payload = verify_gate9_evidence_bundle(args.verify)
    else:
        assert args.output_dir is not None
        report = run_gate9_evidence()
        written = export_gate9_evidence_bundle(
            report,
            output_dir=args.output_dir,
        )
        payload = verify_gate9_evidence_bundle(args.output_dir)
        payload["artifact_files"] = [path.name for path in written]
        payload["optimizer_metrics"] = report.optimizer_metrics
        payload["memory_metrics"] = report.memory_metrics
        payload["mechanism_gates"] = report.mechanism_gates
        payload["optimizer_causal_gates"] = (
            report.optimizer_causal_gates
        )
        payload["memory_causal_gates"] = report.memory_causal_gates
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
