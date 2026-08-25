#!/usr/bin/env python3
"""Prepare or validate the zero-model relationship reader qualification split."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import stat
import sys


_REPO_ROOT = pathlib.Path(os.path.abspath(__file__)).parents[1]
_SOURCE_ROOTS = tuple(sorted((_REPO_ROOT / "packages").glob("*/src")))


def _reject_reparse_components(path: pathlib.Path, label: str) -> None:
    for candidate in (path, *path.parents):
        if not os.path.lexists(candidate):
            continue
        if candidate.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink: {candidate}")
        if os.name == "nt":
            attributes = os.lstat(candidate).st_file_attributes
            if attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise ValueError(
                    f"{label} must not traverse a Windows reparse point: {candidate}"
                )


def _install_source_roots() -> None:
    if not _SOURCE_ROOTS:
        raise FileNotFoundError("qualification package source roots are missing")
    for source_root in _SOURCE_ROOTS:
        _reject_reparse_components(source_root, "qualification source root")
        if not source_root.is_dir():
            raise FileNotFoundError(
                f"qualification source root is missing: {source_root}"
            )
    for source_root in reversed(_SOURCE_ROOTS):
        source_text = str(source_root)
        while source_text in sys.path:
            sys.path.remove(source_text)
        sys.path.insert(0, source_text)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or validate the model-free source-separated relationship "
            "condition reader qualification preflight"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("show-protocol")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--preflight-root", type=pathlib.Path, required=True)
    prepare.add_argument(
        "--proposed-execution-root", type=pathlib.Path, required=True
    )
    validate = commands.add_parser("validate-preflight")
    validate.add_argument("--preflight-root", type=pathlib.Path, required=True)
    validate.add_argument("--expected-protocol-id", required=True)
    validate.add_argument(
        "--expected-publication-request-artifact-id", required=True
    )
    validate.add_argument(
        "--expected-proposed-execution-root", type=pathlib.Path, required=True
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    _install_source_roots()
    from lifeform_evolution.relationship_condition_reader_qualification import (
        load_relationship_condition_reader_qualification_protocol,
        prepare_relationship_condition_reader_qualification_preflight,
        validate_relationship_condition_reader_qualification_preflight,
    )

    if args.command == "show-protocol":
        protocol = load_relationship_condition_reader_qualification_protocol()
        payload = {
            "schema_version": (
                "relationship-condition-reader-qualification-protocol-summary.v1"
            ),
            "protocol_id": protocol.protocol_id,
            "protocol_raw_sha256": protocol.raw_sha256,
            "protocol_raw_bytes": protocol.raw_bytes,
            "training_source_protocol_id": protocol.training_source.protocol_id,
            "challenge_source_protocol_id": protocol.challenge_source.protocol_id,
            "qualification_execution_authorized": False,
            "model_or_cuda_used": False,
        }
    elif args.command == "prepare":
        payload = prepare_relationship_condition_reader_qualification_preflight(
            preflight_root=args.preflight_root,
            proposed_execution_root=args.proposed_execution_root,
        )
    elif args.command == "validate-preflight":
        payload = validate_relationship_condition_reader_qualification_preflight(
            preflight_root=args.preflight_root,
            expected_protocol_id=args.expected_protocol_id,
            expected_publication_request_artifact_id=(
                args.expected_publication_request_artifact_id
            ),
            expected_proposed_execution_root=args.expected_proposed_execution_root,
        )
    else:  # pragma: no cover - argparse owns the command set
        raise AssertionError(f"unreachable command: {args.command}")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
