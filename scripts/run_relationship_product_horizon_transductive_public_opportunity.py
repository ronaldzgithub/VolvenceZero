"""Materialize or validate the Product Horizon public opportunity scanner."""

from __future__ import annotations

import argparse
import pathlib

from lifeform_domain_emogpt.lab.contracts import canonical_json
from lifeform_evolution.relationship_product_horizon_transductive_public_opportunity import (
    materialize_relationship_product_horizon_transductive_public_opportunity,
    validate_relationship_product_horizon_transductive_public_opportunity,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("materialize", "validate-existing"):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--source-v4-admission-root",
            type=pathlib.Path,
            required=True,
        )
        command.add_argument("--reader-root", type=pathlib.Path, required=True)
        command.add_argument("--theta0-v2-root", type=pathlib.Path, required=True)
        command.add_argument("--output-dir", type=pathlib.Path, required=True)
        if name == "materialize":
            command.add_argument("--implementation-git-commit", required=True)
        else:
            command.add_argument("--expected-protocol-id", required=True)
            command.add_argument("--expected-artifact-id", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    shared = {
        "source_v4_admission_root": args.source_v4_admission_root,
        "reader_root": args.reader_root,
        "theta0_v2_root": args.theta0_v2_root,
        "output_dir": args.output_dir,
    }
    if args.command == "materialize":
        manifest = (
            materialize_relationship_product_horizon_transductive_public_opportunity(
                **shared,
                implementation_git_commit=args.implementation_git_commit,
            )
        )
    else:
        manifest = (
            validate_relationship_product_horizon_transductive_public_opportunity(
                **shared,
                expected_protocol_id=args.expected_protocol_id,
                expected_artifact_id=args.expected_artifact_id,
            )
        )
    print(
        canonical_json(
            {
                "status": manifest["status"],
                "protocol_id": manifest["protocol_id"],
                "artifact_id": manifest["artifact_id"],
                "theta0_v2_artifact_id": manifest["theta0_v2_artifact_id"],
                "temporal_delivered_nonnoop_counts_by_category": manifest[
                    "temporal_delivered_nonnoop_counts_by_category"
                ],
                "witness_pass_count": manifest["witness_pass_count"],
                "collection_prefix_protocol_freeze_authorized": manifest["claims"][
                    "collection_prefix_protocol_freeze_authorized"
                ],
                "collection_prefix_execution_authorized": manifest["claims"][
                    "collection_prefix_execution_authorized"
                ],
                "campaign_execution_authorized": manifest["claims"][
                    "campaign_execution_authorized"
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
