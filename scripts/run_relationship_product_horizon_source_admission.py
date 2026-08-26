#!/usr/bin/env python3
"""Materialize or validate the model-free Product Horizon source-v4 admission."""

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
from lifeform_evolution.relationship_product_horizon_source_admission import (  # noqa: E402
    load_relationship_product_horizon_source_admission_protocol,
    materialize_relationship_product_horizon_source_admission,
    validate_relationship_product_horizon_source_admission,
)


_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_OWNED_PATHS = (
    "packages/lifeform-evolution/src/lifeform_evolution/relationship_product_horizon_source_admission.py",
    "packages/lifeform-evolution/src/lifeform_evolution/protocols/relationship_product_horizon_source_v4_campaign_admission_v1.json",
    "scripts/run_relationship_product_horizon_source_admission.py",
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("show-protocol")

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--output-dir", required=True)
    materialize.add_argument("--implementation-git-commit", required=True)

    validate = subparsers.add_parser("validate-existing")
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
    protocol, _ = load_relationship_product_horizon_source_admission_protocol()
    closure_paths = [str(item["path"]) for item in protocol["direct_execution_closure"]]
    owned_paths = [*_OWNED_PATHS, *closure_paths]
    tracked = subprocess.run(
        ["git", "ls-files", "--", *owned_paths],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if set(tracked) != set(owned_paths):
        raise ValueError("source-v4 admission implementation closure is not fully tracked")
    clean = subprocess.run(
        ["git", "diff", "--quiet", expected_commit, "--", *owned_paths],
        cwd=_REPO_ROOT,
        check=False,
    )
    if clean.returncode != 0:
        raise ValueError("source-v4 admission implementation closure differs from frozen commit")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    if args.command == "show-protocol":
        protocol, protocol_id = load_relationship_product_horizon_source_admission_protocol()
        _emit(
            {
                "schema_version": protocol["schema_version"],
                "protocol_id": protocol_id,
                "evidence_tier": protocol["evidence_tier"],
                "campaign_execution_authorized": protocol["claims"][
                    "campaign_execution_authorized"
                ],
            }
        )
        return 0
    if args.command == "materialize":
        _verify_implementation_commit(args.implementation_git_commit)
        _, protocol_id = load_relationship_product_horizon_source_admission_protocol()
        manifest = materialize_relationship_product_horizon_source_admission(
            pathlib.Path(args.output_dir),
            implementation_git_commit=args.implementation_git_commit,
        )
        validated = validate_relationship_product_horizon_source_admission(
            pathlib.Path(args.output_dir),
            expected_protocol_id=protocol_id,
            expected_artifact_id=str(manifest["artifact_id"]),
        )
        if validated != manifest:
            raise ValueError("source-v4 admission validation drifted from created manifest")
        _emit(validated)
        return 0
    if args.command == "validate-existing":
        _emit(
            validate_relationship_product_horizon_source_admission(
                pathlib.Path(args.output_dir),
                expected_protocol_id=args.expected_protocol_id,
                expected_artifact_id=args.expected_artifact_id,
            )
        )
        return 0
    raise AssertionError(f"unreachable command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
