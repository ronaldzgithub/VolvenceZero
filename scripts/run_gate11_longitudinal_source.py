from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from volvence_zero.agent.gate11_longitudinal_source import (
    GATE11_LONGITUDINAL_SOURCE_SEEDS,
    aggregate_gate11_longitudinal_source,
    generate_gate11_longitudinal_source,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or aggregate the fresh Gate 11 longitudinal source."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        choices=GATE11_LONGITUDINAL_SOURCE_SEEDS,
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate all preregistered seed directories.",
    )
    parser.add_argument(
        "--max-transitions",
        type=int,
        help="Stop after this total prefix length for a development probe.",
    )
    args = parser.parse_args()
    if args.aggregate:
        if args.seed is not None or args.max_transitions is not None:
            parser.error("--aggregate cannot be combined with --seed/--max-transitions")
        written = aggregate_gate11_longitudinal_source(
            campaign_dir=args.output_dir
        )
        payload = json.loads(
            (args.output_dir / "aggregate_verdict.json").read_text(
                encoding="utf-8"
            )
        )
    else:
        if args.seed is None:
            parser.error("--seed is required unless --aggregate is used")
        written = asyncio.run(
            generate_gate11_longitudinal_source(
                output_dir=args.output_dir,
                seed=args.seed,
                max_transitions=args.max_transitions,
            )
        )
        payload = json.loads(
            (args.output_dir / "promotion_verdict.json").read_text(
                encoding="utf-8"
            )
        )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "consumer_admission": payload.get(
                    "consumer_admission", "n/a"
                ),
                "failed_gates": payload["failed_gates"],
                "artifact_files": [path.name for path in written],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
