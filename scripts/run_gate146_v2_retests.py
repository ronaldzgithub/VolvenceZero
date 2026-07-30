from __future__ import annotations

import argparse
import json
from pathlib import Path

from volvence_zero.agent.gate1_pe_causal_v2_retest import (
    export_gate1_v2_bundle,
    run_gate1_v2_retest,
    verify_gate1_v2_bundle,
)
from volvence_zero.agent.gate4_active_learning_v2_retest import (
    export_gate4_v2_bundle,
    run_gate4_v2_retest,
    verify_gate4_v2_bundle,
)
from volvence_zero.agent.gate6_meta_init_v2_retest import (
    export_gate6_v2_bundle,
    run_gate6_v2_retest,
    verify_gate6_v2_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one preregistered Gate 1, 4, or 6 v2 retest."
    )
    parser.add_argument("gate", choices=("1", "4", "6"))
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--evaluation-limit", type=int)
    args = parser.parse_args()
    runners = {
        "1": (
            run_gate1_v2_retest,
            export_gate1_v2_bundle,
            verify_gate1_v2_bundle,
        ),
        "4": (
            run_gate4_v2_retest,
            export_gate4_v2_bundle,
            verify_gate4_v2_bundle,
        ),
        "6": (
            run_gate6_v2_retest,
            export_gate6_v2_bundle,
            verify_gate6_v2_bundle,
        ),
    }
    runner, exporter, verifier = runners[args.gate]
    if args.verify is not None:
        payload = verifier(args.verify)
    else:
        if args.trace_root is None:
            parser.error("run requires --trace-root")
        kwargs = {
            "trace_root": args.trace_root,
            "seed_schedule": (
                (701,)
                if args.development and not args.all_seeds
                else (701, 709, 719)
            ),
            "partition": (
                "trace-development-heldout"
                if args.development
                else "trace-locked-confirmation"
            ),
            "formal_locked_run": not args.development,
        }
        if args.gate in {"1", "6"}:
            kwargs["evaluation_limit"] = args.evaluation_limit
        report = runner(**kwargs)
        payload = {
            "partition": report.partition,
            "formal_locked_run": report.formal_locked_run,
            "verdict": report.verdict,
            "aggregate_metrics": report.aggregate_metrics,
            "mechanism_gates": report.mechanism_gates,
            "causal_gates": report.causal_gates,
        }
        if args.output_dir is not None:
            written = exporter(report, output_dir=args.output_dir)
            payload["verification"] = verifier(args.output_dir)
            payload["artifact_files"] = [
                path.name for path in written
            ]
        elif not args.development:
            parser.error("formal run requires --output-dir")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
