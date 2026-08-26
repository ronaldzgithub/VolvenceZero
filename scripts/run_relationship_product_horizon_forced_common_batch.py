"""Materialize or validate Product Horizon root-local common batches."""

from __future__ import annotations

import argparse
import pathlib

from lifeform_domain_emogpt.lab.contracts import canonical_json
from lifeform_evolution.relationship_product_horizon_forced_common_batch import (
    materialize_relationship_product_horizon_forced_common_batch,
    validate_relationship_product_horizon_forced_common_batch,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("materialize", "validate-existing"):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--source-v4-admission-root", type=pathlib.Path, required=True
        )
        command.add_argument("--reader-root", type=pathlib.Path, required=True)
        command.add_argument("--theta0-v2-root", type=pathlib.Path, required=True)
        command.add_argument("--scanner-root", type=pathlib.Path, required=True)
        command.add_argument("--dynamic-root", type=pathlib.Path, required=True)
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
        "scanner_root": args.scanner_root,
        "dynamic_root": args.dynamic_root,
        "output_dir": args.output_dir,
    }
    if args.command == "materialize":
        manifest = materialize_relationship_product_horizon_forced_common_batch(
            **shared,
            implementation_git_commit=args.implementation_git_commit,
        )
    else:
        manifest = validate_relationship_product_horizon_forced_common_batch(
            **shared,
            expected_protocol_id=args.expected_protocol_id,
            expected_artifact_id=args.expected_artifact_id,
        )
    print(
        canonical_json(
            {
                "status": manifest["status"],
                "protocol_id": manifest["protocol_id"],
                "artifact_id": manifest["artifact_id"],
                "completed_root_count": manifest["completed_root_count"],
                "root_batch_count": manifest["root_batch_count"],
                "parameter_delta_nonzero_root_count": manifest[
                    "parameter_delta_nonzero_root_count"
                ],
                "campaign_protocol_freeze_authorized": manifest["claims"][
                    "campaign_protocol_freeze_authorized"
                ],
                "campaign_execution_authorized": manifest["claims"][
                    "campaign_execution_authorized"
                ],
                "effect_tested": manifest["claims"]["effect_tested"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
