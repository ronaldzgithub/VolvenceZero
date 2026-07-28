#!/usr/bin/env python3
"""Aggregate State-KV retention reports into a multi-judge court artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in sorted((REPO_ROOT / "packages").glob("*/src")):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from volvence_zero.state_kv_judge_court import (  # noqa: E402
    build_judge_court_report,
    load_judge_panel,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retention-report",
        action="append",
        required=True,
        help=(
            "path to a verdict_retention_gate.json; repeat once per judge "
            "panel"
        ),
    )
    parser.add_argument(
        "--min-judges",
        type=int,
        default=2,
        help="minimum distinct judge_model_id values required for a court pass",
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts/state_kv/p2-state-strategy-routed-judge-court/"
            "verdict_judge_court.json"
        ),
        help="where to write the multi-judge court artifact",
    )
    args = parser.parse_args(argv)

    panels = tuple(load_judge_panel(path) for path in args.retention_report)
    report = build_judge_court_report(
        panels=panels,
        min_judges=args.min_judges,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.to_json() + "\n", encoding="utf-8")

    print(f"court_state = {report.court_state.value}")
    print(f"judges = {', '.join(report.judge_model_ids)}")
    for claim in report.claims:
        print(f"  {claim.name:36s} {claim.state.value:18s} {claim.detail}")
    for note in report.notes:
        print(f"  note: {note}")
    print(f"judge court: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
