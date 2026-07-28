#!/usr/bin/env python3
"""Aggregate State-KV retention reports across generation seeds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in sorted((REPO_ROOT / "packages").glob("*/src")):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from volvence_zero.state_kv_generation_seed_gate import (  # noqa: E402
    build_generation_seed_gate_report,
    load_generation_seed_panel,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retention-report",
        action="append",
        required=True,
        help="retention report path; repeat once per generation seed",
    )
    parser.add_argument("--min-generation-seeds", type=int, default=3)
    parser.add_argument(
        "--output",
        default=(
            "artifacts/state_kv/p2-state-strategy-routed-generation-seeds/"
            "verdict_generation_seed_gate.json"
        ),
    )
    args = parser.parse_args(argv)

    report = build_generation_seed_gate_report(
        panels=tuple(
            load_generation_seed_panel(path)
            for path in args.retention_report
        ),
        min_generation_seeds=args.min_generation_seeds,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.to_json() + "\n", encoding="utf-8")

    print(f"gate_state = {report.gate_state.value}")
    print(f"generation_seeds = {report.generation_seeds}")
    for claim in report.claims:
        print(f"  {claim.name:38s} {claim.state.value:18s} {claim.detail}")
    print(f"generation-seed gate: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
