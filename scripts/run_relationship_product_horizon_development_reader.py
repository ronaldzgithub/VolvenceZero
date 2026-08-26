"""Materialize or model-free validate Product Horizon development reader inputs."""

from __future__ import annotations

import argparse
import pathlib

from lifeform_domain_emogpt.lab.contracts import canonical_json
from lifeform_evolution.relationship_product_horizon_development_reader import (
    materialize_relationship_product_horizon_development_reader,
    validate_relationship_product_horizon_development_reader,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("materialize", "validate-existing"):
        command = subparsers.add_parser(name)
        command.add_argument("--preflight-root", type=pathlib.Path, required=True)
        command.add_argument(
            "--source-v4-admission-root",
            type=pathlib.Path,
            required=True,
        )
        command.add_argument("--output-dir", type=pathlib.Path, required=True)
        if name == "materialize":
            command.add_argument("--implementation-git-commit", required=True)
        else:
            command.add_argument("--expected-protocol-id", required=True)
            command.add_argument("--expected-artifact-id", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "materialize":
        manifest = materialize_relationship_product_horizon_development_reader(
            preflight_root=args.preflight_root,
            source_v4_admission_root=args.source_v4_admission_root,
            output_dir=args.output_dir,
            implementation_git_commit=args.implementation_git_commit,
        )
    else:
        manifest = validate_relationship_product_horizon_development_reader(
            preflight_root=args.preflight_root,
            source_v4_admission_root=args.source_v4_admission_root,
            output_dir=args.output_dir,
            expected_protocol_id=args.expected_protocol_id,
            expected_artifact_id=args.expected_artifact_id,
        )
    print(
        canonical_json(
            {
                "status": manifest["status"],
                "protocol_id": manifest["protocol_id"],
                "artifact_id": manifest["artifact_id"],
                "reader_artifact_id": manifest["reader_artifact_id"],
                "embedding_table_artifact_id": manifest[
                    "embedding_table_artifact_id"
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
