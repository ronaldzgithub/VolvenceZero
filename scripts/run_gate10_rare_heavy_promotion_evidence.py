from __future__ import annotations

import argparse
import json
from pathlib import Path

from volvence_zero.agent.gate10_rare_heavy_promotion_evidence import (
    export_gate10_evidence_bundle,
    run_gate10_evidence,
    verify_gate10_evidence_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered Gate 10 four-arm rare-heavy promotion "
            "and full-chain rollback drill."
        )
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if (args.output_dir is None) == (args.verify is None):
        parser.error("choose exactly one of --output-dir or --verify")
    if args.verify is not None:
        payload = verify_gate10_evidence_bundle(args.verify)
    else:
        assert args.output_dir is not None
        report = run_gate10_evidence()
        written = export_gate10_evidence_bundle(
            report,
            output_dir=args.output_dir,
        )
        payload = verify_gate10_evidence_bundle(args.output_dir)
        payload["artifact_files"] = [path.name for path in written]
        payload["aggregate_metrics"] = report.aggregate_metrics
        payload["mechanism_gates"] = report.mechanism_gates
        payload["causal_gates"] = report.causal_gates
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
