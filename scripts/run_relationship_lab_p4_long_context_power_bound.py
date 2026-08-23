#!/usr/bin/env python3
"""Publish current power admission or validate its preserved v1 history."""

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
    prepare_relationship_p4_long_context_power_admission_certificate,
    validate_relationship_p4_long_context_power_admission_certificate,
    validate_relationship_p4_long_context_power_failure_certificate,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Publish the current zero-output v3 power-contract admission finding or validate v1 history")
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output-dir", type=pathlib.Path, required=True)
    prepare.add_argument("--preparation-dir", type=pathlib.Path, required=True)

    validate = commands.add_parser("validate-existing")
    validate.add_argument("--output-dir", type=pathlib.Path, required=True)
    validate.add_argument("--preparation-dir", type=pathlib.Path, required=True)

    validate_v1 = commands.add_parser("validate-v1-existing")
    validate_v1.add_argument("--output-dir", type=pathlib.Path, required=True)
    validate_v1.add_argument("--preparation-dir", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    common = {
        "output_dir": args.output_dir,
        "preparation_dir": args.preparation_dir,
    }
    if args.command == "prepare":
        result = prepare_relationship_p4_long_context_power_admission_certificate(**common)
    elif args.command == "validate-existing":
        result = validate_relationship_p4_long_context_power_admission_certificate(**common)
    elif args.command == "validate-v1-existing":
        result = validate_relationship_p4_long_context_power_failure_certificate(**common)
    else:
        raise AssertionError(f"unreachable command: {args.command}")
    if args.command == "validate-v1-existing":
        payload = {
            "artifact_id": result.artifact_id,
            "protocol_id": result.protocol_id,
            "preparation_artifact_id": result.preparation_artifact_id,
            "status": result.status,
            "point_gate_power": {
                "numerator": str(result.point_gate_power_numerator),
                "denominator": str(result.point_gate_power_denominator),
                "display_decimal": result.point_gate_power_display_decimal,
            },
            "historical_numeric_decisive_failure": result.decisive_failure,
            "scientific_admission": result.scientific_admission,
            "full_joint_grid_completed": result.full_joint_grid_completed,
            "development_authorized": result.development_authorized,
            "formal_authorized": result.formal_authorized,
        }
    else:
        payload = {
            "artifact_id": result.artifact_id,
            "admission_protocol_id": result.admission_protocol_id,
            "scientific_protocol_id": result.scientific_protocol_id,
            "preparation_artifact_id": result.preparation_artifact_id,
            "status": result.status,
            "conditional_numeric_bound": {
                "numerator": str(result.conditional_bound_numerator),
                "denominator": str(result.conditional_bound_denominator),
                "display_decimal": result.conditional_bound_display_decimal,
            },
            "power_contract_determinate": result.power_contract_determinate,
            "v1_unconditional_scientific_admission_valid": (result.v1_unconditional_scientific_admission_valid),
            "development_authorized": result.development_authorized,
            "formal_authorized": result.formal_authorized,
        }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
