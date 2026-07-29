from __future__ import annotations

import argparse
import json
from pathlib import Path

from volvence_zero.agent.shared_settled_trace import (
    SHARED_SETTLED_TRACE_SEEDS,
    generate_shared_settled_trace_sync,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or resume a Gate 4/5/6 shared settled trace."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        choices=SHARED_SETTLED_TRACE_SEEDS,
        required=True,
    )
    parser.add_argument(
        "--max-transitions",
        type=int,
        help="Stop after this total prefix length for a development probe.",
    )
    args = parser.parse_args()
    written = generate_shared_settled_trace_sync(
        output_dir=args.output_dir,
        seed=args.seed,
        max_transitions=args.max_transitions,
    )
    verdict = json.loads(
        (args.output_dir / "promotion_verdict.json").read_text(
            encoding="utf-8"
        )
    )
    progress = json.loads(
        (args.output_dir / "progress.json").read_text(
            encoding="utf-8"
        )
    )
    print(
        json.dumps(
            {
                "status": verdict["status"],
                "completed_transition_count": progress[
                    "completed_transition_count"
                ],
                "total_transition_count": progress[
                    "total_transition_count"
                ],
                "failed_gates": verdict["failed_gates"],
                "artifact_files": [path.name for path in written],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
