#!/usr/bin/env python3
"""Freeze or replay the local P4.7 A0 publication-anchor request."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import stat
import sys


_REPO_ROOT = pathlib.Path(os.path.abspath(__file__)).parents[1]
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
        _reject_reparse_components(source_root, "P4.7 A0 source root")
        if not source_root.is_dir():
            raise FileNotFoundError(f"P4.7 A0 source root is missing: {source_root}")
    for source_root in reversed(_EXACT_SOURCE_ROOTS):
        source_text = str(source_root)
        while source_text in sys.path:
            sys.path.remove(source_text)
        sys.path.insert(0, source_text)


def _add_artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--source-preflight-dir", type=pathlib.Path, required=True)
    parser.add_argument("--v4a-planning-dir", type=pathlib.Path, required=True)
    parser.add_argument("--v3-preparation-dir", type=pathlib.Path, required=True)
    parser.add_argument("--v2-admission-dir", type=pathlib.Path, required=True)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze or replay the zero-output A0 public-anchor request")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("show-protocol")
    _add_artifact_arguments(commands.add_parser("prepare-request"))
    _add_artifact_arguments(commands.add_parser("validate-request"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    _install_exact_source_roots()
    from lifeform_evolution.relationship_lab_p4_long_context_causal_campaign import (
        load_relationship_p4_long_context_external_anchor_request_protocol,
        prepare_relationship_p4_long_context_external_anchor_request,
        validate_relationship_p4_long_context_external_anchor_request,
    )

    if args.command == "show-protocol":
        protocol = load_relationship_p4_long_context_external_anchor_request_protocol()
        payload = {
            "protocol_id": protocol.protocol_id,
            "schema_version": protocol.schema_version,
            "status": protocol.status,
            "provider": protocol.provider,
            "expected_owner_login": protocol.expected_owner_login,
            "required_filename": protocol.required_filename,
            "publication_request_contract_frozen": protocol.publication_request_contract_frozen,
            "external_publication_anchor_present": protocol.external_publication_anchor_present,
            "structural_inventory_materialization_authorized": (
                protocol.structural_inventory_materialization_authorized
            ),
        }
    else:
        common = {
            "output_dir": args.output_dir,
            "source_preflight_dir": args.source_preflight_dir,
            "v4a_planning_dir": args.v4a_planning_dir,
            "v3_preparation_dir": args.v3_preparation_dir,
            "v2_admission_dir": args.v2_admission_dir,
        }
        if args.command == "prepare-request":
            result = prepare_relationship_p4_long_context_external_anchor_request(**common)
        elif args.command == "validate-request":
            result = validate_relationship_p4_long_context_external_anchor_request(**common)
        else:
            raise AssertionError(f"unreachable command: {args.command}")
        payload = {
            "artifact_id": result.artifact_id,
            "request_id": result.request_id,
            "protocol_id": result.protocol_id,
            "status": result.status,
            "publication_request_contract_frozen": result.publication_request_contract_frozen,
            "external_request_dispatched": result.external_request_dispatched,
            "publication_performed": result.publication_performed,
            "external_publication_anchor_present": result.external_publication_anchor_present,
            "external_anchor_admitted": result.external_anchor_admitted,
            "structural_inventory_materialization_authorized": (result.structural_inventory_materialization_authorized),
            "source_execution_authorized": result.source_execution_authorized,
            "tuple_feasibility_authorized": result.tuple_feasibility_authorized,
            "model_output_authorized": result.model_output_authorized,
            "cuda_planner_authorized": result.cuda_planner_authorized,
        }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
