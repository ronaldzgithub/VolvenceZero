#!/usr/bin/env python3
"""Freeze or score the blinded Relationship Lab P1L human-anchor packet.

This script never loads Qwen.  It may run while P1j is in progress.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _relative in (
    "packages/lifeform-domain-emogpt/src",
    "packages/lifeform-evolution/src",
    "packages/vz-contracts/src",
):
    sys.path.insert(0, str(_REPO_ROOT / _relative))

from lifeform_domain_emogpt.lab import (  # noqa: E402
    load_relationship_consumer_training_view,
)
from lifeform_evolution.relationship_lab_packet1l import (  # noqa: E402
    assess_relationship_p1l_ratings,
    freeze_relationship_p1l_protocol,
    load_relationship_p1l_protocol,
    load_relationship_p1l_ratings,
    write_relationship_p1l_packet,
    write_relationship_p1l_report,
)


_DEFAULT_OUTPUT_DIR = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "packet1l_v3_human_anchor_20260821"
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Freeze the blinded rater packet and sealed key, then stop.",
    )
    parser.add_argument(
        "--ratings",
        nargs="*",
        default=(),
        help="One or more JSON/CSV rating files to score against the sealed key.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    output_dir = pathlib.Path(args.output_dir)
    dataset = load_relationship_consumer_training_view().training_dataset
    protocol_path = output_dir / "packet1l_protocol.json"
    if protocol_path.is_file():
        protocol = load_relationship_p1l_protocol(protocol_path)
        units_protocol, units = freeze_relationship_p1l_protocol(
            dataset=dataset,
            frozen_at_iso=protocol.frozen_at_iso,
        )
        if units_protocol.protocol_id != protocol.protocol_id:
            raise ValueError("P1L existing protocol diverges from v3 public evidence")
    else:
        if args.ratings:
            raise FileNotFoundError("P1L scoring requires a frozen packet")
        protocol, units = freeze_relationship_p1l_protocol(dataset=dataset)
        write_relationship_p1l_packet(
            protocol=protocol,
            units=units,
            output_dir=output_dir,
        )
        print(
            json.dumps(
                {
                    "stage": "prepared",
                    "protocol_id": protocol.protocol_id,
                    "required_units": protocol.required_units,
                    "rater_packet": str(output_dir / "rater_packet.csv"),
                    "sealed_key": str(output_dir / "sealed_answer_key.json"),
                    "next_action": protocol.next_action,
                },
                ensure_ascii=False,
            )
        )
        if args.prepare_only or not args.ratings:
            return 0

    ratings = tuple(
        rating
        for path in args.ratings
        for rating in load_relationship_p1l_ratings(pathlib.Path(path))
    )
    report = assess_relationship_p1l_ratings(
        protocol=protocol,
        units=units,
        ratings=ratings,
    )
    report_path = write_relationship_p1l_report(report=report, output_dir=output_dir)
    print(
        json.dumps(
            {
                "stage": "scored" if ratings else "pending",
                "report": str(report_path),
                "report_artifact_id": report.artifact_id,
                "verdict": report.verdict.value,
                "next_action": report.next_action,
                "majority_agreement": report.majority_agreement,
                "majority_accuracy": report.majority_accuracy,
            },
            ensure_ascii=False,
        )
    )
    return 0 if report.verdict.value != "human_anchor_failed_development" else 2


if __name__ == "__main__":
    raise SystemExit(main())
