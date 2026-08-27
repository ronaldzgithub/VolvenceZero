#!/usr/bin/env python3
"""Materialize or model-free validate the source-v5 public embedding table."""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _source_root in sorted((_REPO_ROOT / "packages").glob("*/src")):
    sys.path.insert(0, str(_source_root))

from lifeform_domain_emogpt.lab.contracts import canonical_json  # noqa: E402
from lifeform_evolution.relationship_product_horizon_source_v5_embedding_table import (  # noqa: E402
    SourceV5AdmissionValidationInputs,
    load_relationship_product_horizon_source_v5_embedding_table_protocol,
    materialize_relationship_product_horizon_source_v5_embedding_table,
    validate_relationship_product_horizon_source_v5_embedding_table,
)


_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


def _add_admission_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-v5-admission-root", required=True)
    parser.add_argument("--source-v3-admission-root", required=True)
    parser.add_argument("--source-v4-admission-root", required=True)
    parser.add_argument("--development-reader-root", required=True)
    parser.add_argument("--attempt03-embedding-table", required=True)
    parser.add_argument("--attempt03-reobservation", required=True)
    parser.add_argument("--qualification-v5-embedding-table", required=True)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("show-protocol")

    materialize = subparsers.add_parser("materialize")
    _add_admission_arguments(materialize)
    materialize.add_argument("--output-dir", required=True)
    materialize.add_argument("--implementation-git-commit", required=True)
    materialize.add_argument("--bge-snapshot-path", required=True)

    validate = subparsers.add_parser("validate-existing")
    _add_admission_arguments(validate)
    validate.add_argument("--output-dir", required=True)
    validate.add_argument("--expected-protocol-id", required=True)
    validate.add_argument("--expected-artifact-id", required=True)
    return parser.parse_args(argv)


def _emit(payload: object) -> None:
    print(canonical_json(payload), flush=True)


def _verify_implementation_commit(expected_commit: str) -> None:
    if _GIT_COMMIT.fullmatch(expected_commit) is None:
        raise ValueError("implementation git commit must be lowercase 40-hex")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected_commit:
        raise ValueError("implementation git commit does not match current HEAD")
    protocol = load_relationship_product_horizon_source_v5_embedding_table_protocol()
    closure = [str(value) for value in protocol.payload["direct_execution_closure"]]
    tracked = subprocess.run(
        ["git", "ls-files", "--", *closure],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if set(tracked) != set(closure):
        raise ValueError("source-v5 embedding execution closure is not fully tracked")
    clean = subprocess.run(
        ["git", "diff", "--quiet", expected_commit, "--", *closure],
        cwd=_REPO_ROOT,
        check=False,
    )
    if clean.returncode != 0:
        raise ValueError("source-v5 embedding execution closure differs from frozen commit")


def _validation_inputs(args: argparse.Namespace) -> SourceV5AdmissionValidationInputs:
    return SourceV5AdmissionValidationInputs(
        source_v3_admission_root=pathlib.Path(args.source_v3_admission_root),
        source_v4_admission_root=pathlib.Path(args.source_v4_admission_root),
        development_reader_root=pathlib.Path(args.development_reader_root),
        attempt03_embedding_table_path=pathlib.Path(
            args.attempt03_embedding_table
        ),
        attempt03_reobservation_path=pathlib.Path(args.attempt03_reobservation),
        qualification_v5_embedding_table_path=pathlib.Path(
            args.qualification_v5_embedding_table
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    protocol = load_relationship_product_horizon_source_v5_embedding_table_protocol()
    if args.command == "show-protocol":
        _emit(
            {
                "schema_version": protocol.payload["schema_version"],
                "protocol_id": protocol.protocol_id,
                "evidence_tier": protocol.payload["evidence_tier"],
                "batch_size": protocol.execution_contract["batch_size"],
                "reader_fit_count": protocol.output_contract["reader_fit_count"],
                "campaign_execution_authorized": protocol.claims_ceiling[
                    "campaign_execution_authorized"
                ],
            }
        )
        return 0
    validation_inputs = _validation_inputs(args)
    if args.command == "materialize":
        _verify_implementation_commit(args.implementation_git_commit)
        manifest = materialize_relationship_product_horizon_source_v5_embedding_table(
            source_v5_admission_root=pathlib.Path(args.source_v5_admission_root),
            admission_validation_inputs=validation_inputs,
            output_dir=pathlib.Path(args.output_dir),
            implementation_git_commit=args.implementation_git_commit,
            bge_snapshot_path=pathlib.Path(args.bge_snapshot_path),
        )
    elif args.command == "validate-existing":
        manifest = validate_relationship_product_horizon_source_v5_embedding_table(
            source_v5_admission_root=pathlib.Path(args.source_v5_admission_root),
            admission_validation_inputs=validation_inputs,
            output_dir=pathlib.Path(args.output_dir),
            expected_protocol_id=args.expected_protocol_id,
            expected_artifact_id=args.expected_artifact_id,
        )
    else:
        raise AssertionError(f"unreachable command: {args.command}")
    _emit(
        {
            "status": manifest["status"],
            "protocol_id": manifest["protocol_id"],
            "artifact_id": manifest["artifact_id"],
            "embedding_table_artifact_id": manifest[
                "embedding_table_artifact_id"
            ],
            "embedding_table_record_count": manifest[
                "embedding_table_record_count"
            ],
            "reader_fit_count": manifest["reader_fit_count"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
