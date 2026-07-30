from __future__ import annotations

import argparse
import json
from pathlib import Path

from volvence_zero.agent.gate8_wake_sleep_evidence import (
    export_gate8_evidence_bundle,
    run_gate8_evidence,
    verify_gate8_evidence_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the preregistered Gate 8 wake/sleep campaign."
    )
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--evaluation-limit", type=int)
    args = parser.parse_args()
    if args.verify is not None:
        payload = verify_gate8_evidence_bundle(args.verify)
    else:
        if args.trace_root is None:
            parser.error("run requires --trace-root")
        report = run_gate8_evidence(
            trace_root=args.trace_root,
            seed_schedule=(701,) if args.development else (701, 709, 719),
            partition=(
                "trace-development-heldout"
                if args.development
                else "trace-locked-confirmation"
            ),
            evaluation_limit=args.evaluation_limit,
            formal_locked_run=not args.development,
        )
        if args.development:
            payload = {
                "schema_version": report.schema_version,
                "partition": report.partition,
                "formal_locked_run": report.formal_locked_run,
                "verdict": report.verdict,
                "aggregate_metrics": report.aggregate_metrics,
                "mechanism_gates": report.mechanism_gates,
                "causal_gates": report.causal_gates,
            }
        else:
            if args.output_dir is None:
                parser.error("formal run requires --output-dir")
            written = export_gate8_evidence_bundle(
                report,
                output_dir=args.output_dir,
            )
            payload = verify_gate8_evidence_bundle(args.output_dir)
            payload["artifact_files"] = [path.name for path in written]
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
