#!/usr/bin/env python3
"""Prepare or validate the zero-output P4.7 scientific preregistration."""

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
    P4_LONG_CONTEXT_PROTOCOL_ID_V1,
    P4_LONG_CONTEXT_PROTOCOL_ID_V2,
    P4_LONG_CONTEXT_PROTOCOL_ID_V3,
    load_relationship_p4_long_context_scientific_prereg,
    prepare_relationship_p4_long_context_scientific_prereg,
    relationship_p4_long_context_protocol_path,
    validate_relationship_p4_long_context_scientific_prereg,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze or validate the zero-output P4.7 independent long-context causal-campaign scientific design"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    show = commands.add_parser("show-protocol")
    show_source = show.add_mutually_exclusive_group()
    show_source.add_argument("--protocol", type=pathlib.Path)
    show_source.add_argument(
        "--protocol-version",
        choices=("v1", "v2", "v3"),
        default="v3",
    )

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output-dir", type=pathlib.Path, required=True)
    prepare.add_argument("--protocol", type=pathlib.Path)

    validate = commands.add_parser("validate-existing")
    validate.add_argument("--output-dir", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.command == "show-protocol":
        protocol_path = args.protocol
        if protocol_path is None:
            protocol_id = {
                "v1": P4_LONG_CONTEXT_PROTOCOL_ID_V1,
                "v2": P4_LONG_CONTEXT_PROTOCOL_ID_V2,
                "v3": P4_LONG_CONTEXT_PROTOCOL_ID_V3,
            }[args.protocol_version]
            protocol_path = relationship_p4_long_context_protocol_path(protocol_id)
        protocol = load_relationship_p4_long_context_scientific_prereg(protocol_path)
        print(
            json.dumps(
                {
                    "protocol_id": protocol.protocol_id,
                    "schema_version": protocol.schema_version,
                    "superseded": protocol.superseded,
                    "formal_subject_count": protocol.formal_subject_count,
                    "minimum_complete_paired_subjects": (protocol.minimum_complete_paired_subjects),
                    "arm_count": len(protocol.arm_matrix),
                    "minimum_public_history_tokens": (protocol.minimum_public_history_tokens),
                    "minimum_native_context_window_tokens": (protocol.minimum_native_context_window_tokens),
                    "execution_enabled": protocol.execution_enabled,
                    "formal_run_authorized": protocol.formal_run_authorized,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "prepare":
        result = prepare_relationship_p4_long_context_scientific_prereg(
            output_dir=args.output_dir,
            protocol_path=args.protocol,
        )
    elif args.command == "validate-existing":
        result = validate_relationship_p4_long_context_scientific_prereg(
            output_dir=args.output_dir,
        )
    else:
        raise AssertionError(f"unreachable command: {args.command}")
    print(
        json.dumps(
            {
                "artifact_id": result.artifact_id,
                "protocol_id": result.protocol_id,
                "status": result.status,
                "execution_enabled": result.execution_enabled,
                "formal_run_authorized": result.formal_run_authorized,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
