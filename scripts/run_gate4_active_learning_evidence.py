from __future__ import annotations

import argparse
import json
from pathlib import Path

from volvence_zero.agent.gate4_active_learning_evidence import (
    export_gate4_active_learning_bundle,
    verify_gate4_active_learning_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered Gate 4 segment-aware active-learning "
            "campaign."
        )
    )
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--development-no-locked",
        action="store_true",
        help="Run a development probe without consuming locked labels.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing output bundle without replaying.",
    )
    args = parser.parse_args()
    if args.verify_only:
        print(
            json.dumps(
                verify_gate4_active_learning_bundle(args.output_dir),
                indent=2,
                sort_keys=True,
            )
        )
        return
    written = export_gate4_active_learning_bundle(
        trace_root=args.trace_root,
        output_dir=args.output_dir,
        consume_locked=not args.development_no_locked,
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
                "causal_passed": verdict["causal_passed"],
                "failed_gates": verdict["failed_gates"],
                "artifact_files": [path.name for path in written],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
