"""Materialize or model-free validate the development theta0 v3 bootstrap."""

from __future__ import annotations

import argparse
import pathlib
import sys


_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _source_root in sorted((_REPOSITORY_ROOT / "packages").glob("*/src")):
    sys.path.insert(0, str(_source_root))

from lifeform_domain_emogpt.lab.contracts import canonical_json  # noqa: E402
from lifeform_evolution.relationship_product_horizon_theta0_v3_bootstrap import (  # noqa: E402
    load_relationship_product_horizon_theta0_v3_bootstrap_protocol,
    materialize_relationship_product_horizon_theta0_v3_bootstrap,
    validate_relationship_product_horizon_theta0_v3_bootstrap,
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
        "output_dir": args.output_dir,
    }
    if args.command == "materialize":
        manifest = materialize_relationship_product_horizon_theta0_v3_bootstrap(
            **shared,
            implementation_git_commit=args.implementation_git_commit,
        )
        protocol = load_relationship_product_horizon_theta0_v3_bootstrap_protocol()
        validated = validate_relationship_product_horizon_theta0_v3_bootstrap(
            **shared,
            expected_protocol_id=protocol.protocol_id,
            expected_artifact_id=manifest["artifact_id"],
        )
        if validated != manifest:
            raise ValueError("theta0 v3 post-materialization validation drifted")
        manifest = validated
    else:
        manifest = validate_relationship_product_horizon_theta0_v3_bootstrap(
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
                "parent_schedule_artifact_id": manifest[
                    "parent_schedule_artifact_id"
                ],
                "federated_gate_batch_id": manifest["federated_gate_batch_id"],
                "published_theta0_artifact_id": manifest[
                    "published_theta0_artifact_id"
                ],
                "child_transition_count": manifest["child_transition_count"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
