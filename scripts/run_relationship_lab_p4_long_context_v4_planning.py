#!/usr/bin/env python3
"""Publish or validate the P4.7 v4a zero-output planning freeze."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _source_root in sorted((_REPO_ROOT / "packages").glob("*/src")):
    if _source_root.is_dir():
        sys.path.insert(0, str(_source_root))

from lifeform_evolution.relationship_lab_p4_long_context_causal_campaign import (  # noqa: E402
    prepare_relationship_p4_long_context_v4_zero_output_plan,
    validate_relationship_p4_long_context_v4_zero_output_plan,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Publish or validate the v4a planning-only protocol and abstract six-cell schedule")
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "validate-existing"):
        subcommand = commands.add_parser(command)
        subcommand.add_argument("--output-dir", type=pathlib.Path, required=True)
        subcommand.add_argument(
            "--v3-preparation-dir",
            type=pathlib.Path,
            required=True,
        )
        subcommand.add_argument(
            "--v2-admission-dir",
            type=pathlib.Path,
            required=True,
        )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    common = {
        "output_dir": args.output_dir,
        "v3_preparation_dir": args.v3_preparation_dir,
        "v2_admission_dir": args.v2_admission_dir,
    }
    if args.command == "prepare":
        result = prepare_relationship_p4_long_context_v4_zero_output_plan(**common)
    elif args.command == "validate-existing":
        result = validate_relationship_p4_long_context_v4_zero_output_plan(**common)
    else:
        raise AssertionError(f"unreachable command: {args.command}")
    print(
        json.dumps(
            {
                "artifact_id": result.artifact_id,
                "protocol_id": result.protocol_id,
                "scientific_v3_protocol_id": result.scientific_v3_protocol_id,
                "power_admission_v2_artifact_id": result.power_admission_v2_artifact_id,
                "status": result.status,
                "first_necessary_screen_passing_root_count": (result.first_necessary_screen_passing_root_count),
                "first_positive_mean_gate_capable_root_count": (result.first_positive_mean_gate_capable_root_count),
                "cartesian_candidate_tuple_count": (result.cartesian_candidate_tuple_count),
                "candidate_schedule_block_count": (result.candidate_schedule_block_count),
                "power_contract_determinate": result.power_contract_determinate,
                "source_grid_resolved": result.source_grid_resolved,
                "selected_formal_root_count": result.selected_formal_root_count,
                "source_materialization_authorized": (result.source_materialization_authorized),
                "development_authorized": result.development_authorized,
                "model_output_authorized": result.model_output_authorized,
                "formal_authorized": result.formal_authorized,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
