#!/usr/bin/env python3
"""Aggregate State-KV identification verdicts into a retention gate artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in sorted((REPO_ROOT / "packages").glob("*/src")):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from volvence_zero.state_kv_retention_gate import (  # noqa: E402
    build_retention_gate_report,
    load_retention_evidence,
)

DEFAULT_P2_PAIRS: tuple[str, ...] = (
    "repair-vs-execute",
    "boundary-vs-commit",
)
DEFAULT_BOOTSTRAP_SEEDS: tuple[int, ...] = (
    20260726,
    20260727,
    20260728,
    1701,
    31337,
)


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _split_int_csv(value: str) -> tuple[int, ...]:
    items = _split_csv(value)
    if not items:
        raise ValueError("expected at least one integer seed")
    return tuple(int(item) for item in items)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verdict",
        action="append",
        required=True,
        help=(
            "path to a verdict_identification.json; may be repeated. The "
            "adjacent substrate_fingerprint.json is read automatically"
        ),
    )
    parser.add_argument(
        "--required-p2-pairs",
        default=",".join(DEFAULT_P2_PAIRS),
        help="comma-separated P2 pair ids required for held-out coverage",
    )
    parser.add_argument(
        "--bootstrap-seeds",
        default=",".join(str(seed) for seed in DEFAULT_BOOTSTRAP_SEEDS),
        help="comma-separated seeds used to recompute aggregate bootstrap CIs",
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts/state_kv/p2-state-strategy-routed-retention/"
            "verdict_retention_gate.json"
        ),
        help="where to write the aggregate retention gate artifact",
    )
    args = parser.parse_args(argv)

    evidences = tuple(load_retention_evidence(path) for path in args.verdict)
    report = build_retention_gate_report(
        evidences=evidences,
        required_p2_pairs=_split_csv(args.required_p2_pairs),
        bootstrap_seeds=_split_int_csv(args.bootstrap_seeds),
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.to_json() + "\n", encoding="utf-8")

    print(f"gate_state = {report.gate_state.value}")
    for claim in report.claims:
        print(f"  {claim.name:38s} {claim.state.value:18s} {claim.detail}")
    for aggregate in report.aggregates:
        print(
            f"  aggregate {aggregate.arm_label}: "
            f"{aggregate.correct}/{aggregate.total} "
            f"accuracy={aggregate.accuracy:.3f} "
            f"CI=({aggregate.ci_low_min:.3f}, {aggregate.ci_high_max:.3f})"
        )
    for note in report.notes:
        print(f"  note: {note}")
    print(f"retention gate: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
