from __future__ import annotations

import argparse
import json
from pathlib import Path

from volvence_zero.agent.gate11_per_user_continuity_evidence import (
    export_gate11_per_user_continuity_bundle,
    reconcile_gate11_preregistered_verdict,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the preregistered Gate 11 per-user continuity suite."
    )
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument(
        "--reconcile-source",
        type=Path,
        help="Correct a completed v1 evaluator without rerunning arms.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.reconcile_source is not None:
        if args.trace_root is not None:
            parser.error(
                "--reconcile-source cannot be combined with --trace-root"
            )
        written = reconcile_gate11_preregistered_verdict(
            source_bundle=args.reconcile_source,
            output_dir=args.output_dir,
        )
    else:
        if args.trace_root is None:
            parser.error(
                "--trace-root is required unless --reconcile-source is used"
            )
        written = export_gate11_per_user_continuity_bundle(
            trace_root=args.trace_root,
            output_dir=args.output_dir,
        )
    verdict = json.loads(
        (args.output_dir / "promotion_verdict.json").read_text(
            encoding="utf-8"
        )
    )
    print(
        json.dumps(
            {
                "status": verdict["status"],
                "mechanism_passed": verdict["mechanism_passed"],
                "longitudinal_passed": verdict["longitudinal_passed"],
                "failed_gates": verdict["failed_gates"],
                "artifact_files": [path.name for path in written],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
