#!/usr/bin/env python3
"""Publish or validate the P4.7 source-opportunity zero-output preflight."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import stat
import sys


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_EXACT_SOURCE_ROOTS = (
    _REPO_ROOT / "packages" / "lifeform-evolution" / "src",
    _REPO_ROOT / "packages" / "lifeform-domain-emogpt" / "src",
)


def _reject_reparse_components(path: pathlib.Path, label: str) -> None:
    for candidate in (path, *path.parents):
        if not os.path.lexists(candidate):
            continue
        if candidate.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink: {candidate}")
        if os.name == "nt":
            attributes = os.lstat(candidate).st_file_attributes
            if attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise ValueError(f"{label} must not traverse a Windows reparse point: {candidate}")


def _install_exact_source_roots() -> None:
    for source_root in _EXACT_SOURCE_ROOTS:
        _reject_reparse_components(source_root, "P4.7 source preflight source root")
        if not source_root.is_dir():
            raise FileNotFoundError(f"P4.7 source preflight source root is missing: {source_root}")
    for source_root in reversed(_EXACT_SOURCE_ROOTS):
        source_text = str(source_root)
        while source_text in sys.path:
            sys.path.remove(source_text)
        sys.path.insert(0, source_text)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Publish or validate the source-opportunity zero-output preflight contract")
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "validate-existing"):
        subcommand = commands.add_parser(command)
        subcommand.add_argument("--output-dir", type=pathlib.Path, required=True)
        subcommand.add_argument(
            "--v4a-planning-dir",
            type=pathlib.Path,
            required=True,
        )
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
    _install_exact_source_roots()
    from lifeform_evolution.relationship_lab_p4_long_context_causal_campaign import (
        prepare_relationship_p4_long_context_source_opportunity_preflight,
        validate_relationship_p4_long_context_source_opportunity_preflight,
    )

    common = {
        "output_dir": args.output_dir,
        "v4a_planning_dir": args.v4a_planning_dir,
        "v3_preparation_dir": args.v3_preparation_dir,
        "v2_admission_dir": args.v2_admission_dir,
    }
    if args.command == "prepare":
        result = prepare_relationship_p4_long_context_source_opportunity_preflight(**common)
    elif args.command == "validate-existing":
        result = validate_relationship_p4_long_context_source_opportunity_preflight(**common)
    else:
        raise AssertionError(f"unreachable command: {args.command}")
    print(
        json.dumps(
            {
                "artifact_id": result.artifact_id,
                "certificate_id": result.certificate_id,
                "protocol_id": result.protocol_id,
                "v4a_planning_artifact_id": result.v4a_planning_artifact_id,
                "status": result.status,
                "zero_output_preflight_contract_frozen": (result.zero_output_preflight_contract_frozen),
                "source_opportunity_stage_completed": (result.source_opportunity_stage_completed),
                "source_structural_inventory_materialized": (result.source_structural_inventory_materialized),
                "unresolved_tuple_count": result.unresolved_tuple_count,
                "selected_formal_root_count": result.selected_formal_root_count,
                "current_source_execution_authorized": (result.current_source_execution_authorized),
                "tuple_feasibility_authorized": (result.tuple_feasibility_authorized),
                "model_output_authorized": result.model_output_authorized,
                "development_authorized": result.development_authorized,
                "qualification_authorized": result.qualification_authorized,
                "formal_authorized": result.formal_authorized,
                "cuda_planner_authorized": result.cuda_planner_authorized,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
