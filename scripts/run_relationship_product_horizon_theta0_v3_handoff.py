#!/usr/bin/env python3
"""Materialize or validate the accepted theta0-v3 typed handoff."""

from __future__ import annotations

import argparse
import pathlib
import sys


_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _source_root in sorted((_REPOSITORY_ROOT / "packages").glob("*/src")):
    sys.path.insert(0, str(_source_root))

from lifeform_domain_emogpt.lab.contracts import canonical_json  # noqa: E402
from lifeform_evolution.relationship_product_horizon_theta0_v3_handoff import (  # noqa: E402
    load_relationship_product_horizon_theta0_v3_handoff_protocol,
    materialize_relationship_product_horizon_theta0_v3_handoff,
    validate_relationship_product_horizon_theta0_v3_handoff,
)


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-v4-admission-root", required=True)
    parser.add_argument("--development-reader-root", required=True)
    parser.add_argument("--theta0-root", required=True)
    parser.add_argument("--historical-validation-report", required=True)
    parser.add_argument("--output-dir", required=True)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("show-protocol")
    materialize = commands.add_parser("materialize")
    _add_shared_arguments(materialize)
    materialize.add_argument("--implementation-git-commit", required=True)
    validate = commands.add_parser("validate-existing")
    _add_shared_arguments(validate)
    validate.add_argument("--expected-protocol-id", required=True)
    validate.add_argument("--expected-artifact-id", required=True)
    return parser.parse_args(argv)


def _shared(args: argparse.Namespace) -> dict[str, pathlib.Path]:
    return {
        "source_v4_admission_root": pathlib.Path(args.source_v4_admission_root),
        "development_reader_root": pathlib.Path(args.development_reader_root),
        "theta0_root": pathlib.Path(args.theta0_root),
        "historical_validation_report_path": pathlib.Path(args.historical_validation_report),
        "output_dir": pathlib.Path(args.output_dir),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    protocol = load_relationship_product_horizon_theta0_v3_handoff_protocol()
    if args.command == "show-protocol":
        print(
            canonical_json(
                {
                    "schema_version": protocol.payload["schema_version"],
                    "protocol_id": protocol.protocol_id,
                    "evidence_tier": protocol.payload["evidence_tier"],
                    "historical_theta0_artifact_id": protocol.theta0_input["artifact_id"],
                    "theta_handoff_materialized_on_success": protocol.payload["claims_ceiling"][
                        "theta_handoff_materialized"
                    ],
                    "campaign_execution_authorized": protocol.payload["claims_ceiling"][
                        "campaign_execution_authorized"
                    ],
                }
            ),
            flush=True,
        )
        return 0
    if args.command == "materialize":
        manifest = materialize_relationship_product_horizon_theta0_v3_handoff(
            **_shared(args),
            implementation_git_commit=args.implementation_git_commit,
        )
    elif args.command == "validate-existing":
        manifest = validate_relationship_product_horizon_theta0_v3_handoff(
            **_shared(args),
            expected_protocol_id=args.expected_protocol_id,
            expected_artifact_id=args.expected_artifact_id,
        )
    else:
        raise AssertionError(f"unreachable command: {args.command}")
    print(
        canonical_json(
            {
                "status": manifest["status"],
                "protocol_id": manifest["protocol_id"],
                "artifact_id": manifest["artifact_id"],
                "theta0_artifact_id": manifest["theta0_artifact_id"],
                "theta0_authorization_id": manifest["theta0_authorization_id"],
                "current_full_typed_federation_rehydrated": manifest["current_full_typed_federation_rehydrated"],
                "campaign_execution_authorized": manifest["claims"]["campaign_execution_authorized"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
