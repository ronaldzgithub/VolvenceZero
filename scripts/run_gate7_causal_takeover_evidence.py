from __future__ import annotations

import argparse
import json
from pathlib import Path

from volvence_zero.agent.gate7_causal_takeover_evidence import (
    export_gate7_evidence_bundle,
    run_gate7_evidence,
    verify_gate7_evidence_bundle,
)
from volvence_zero.agent.gate78_shared_trace import (
    GATE7_V3_TRACE_PROFILE,
    GATE78_V2_TRACE_PROFILE,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the preregistered Gate 7 five-arm causal takeover campaign."
    )
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--evaluation-limit", type=int)
    parser.add_argument(
        "--profile",
        choices=("v2", "gate7-v3"),
        help="Explicit source profile for a development or formal run.",
    )
    args = parser.parse_args()
    if args.verify is not None:
        payload = verify_gate7_evidence_bundle(args.verify)
    else:
        if args.trace_root is None:
            parser.error("run requires --trace-root")
        if args.profile is None:
            parser.error("run requires an explicit --profile")
        source_profile = (
            GATE7_V3_TRACE_PROFILE
            if args.profile == "gate7-v3"
            else GATE78_V2_TRACE_PROFILE
        )
        report = run_gate7_evidence(
            trace_root=args.trace_root,
            seed_schedule=(
                (source_profile.seeds[0],)
                if args.development
                else source_profile.seeds
            ),
            source_profile=source_profile,
            partition=(
                "trace-development-heldout"
                if args.development
                else "trace-locked-confirmation"
            ),
            train_limit=args.train_limit,
            evaluation_limit=args.evaluation_limit,
            formal_locked_run=not args.development,
        )
        if args.development and args.output_dir is None:
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
                parser.error("artifact export requires --output-dir")
            written = export_gate7_evidence_bundle(
                report,
                output_dir=args.output_dir,
            )
            payload = verify_gate7_evidence_bundle(args.output_dir)
            payload["artifact_files"] = [path.name for path in written]
            payload["aggregate_metrics"] = report.aggregate_metrics
            payload["mechanism_gates"] = report.mechanism_gates
            payload["causal_gates"] = report.causal_gates
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
