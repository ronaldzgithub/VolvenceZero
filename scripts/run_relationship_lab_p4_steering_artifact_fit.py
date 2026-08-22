#!/usr/bin/env python3
"""Run or offline-validate the P4 Windows/CUDA steering fit prerequisite."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys


_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _source_root in sorted((_REPOSITORY_ROOT / "packages").glob("*/src")):
    if _source_root.is_dir():
        sys.path.insert(0, str(_source_root))

from volvence_zero.agent.relationship_p4_steering_artifact_fit import (  # noqa: E402
    run_relationship_p4_steering_artifact_fit,
    validate_relationship_p4_steering_artifact_fit,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create or offline-validate the fresh Qwen2.5-1.5B Windows/CUDA "
            "steering artifact fit prerequisite"
        )
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run")
    run.add_argument("--output-dir", type=pathlib.Path, required=True)
    run.add_argument("--protocol", type=pathlib.Path)
    validate = subcommands.add_parser("validate-existing")
    validate.add_argument("--output-dir", type=pathlib.Path, required=True)
    validate.add_argument("--protocol", type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.command == "validate-existing":
        result = validate_relationship_p4_steering_artifact_fit(
            output_dir=args.output_dir,
            protocol_path=args.protocol,
        )
    elif args.command == "run":
        if os.environ.get("PYTHONNOUSERSITE") != "1":
            raise RuntimeError(
                "P4 steering-fit run requires PYTHONNOUSERSITE=1 so the "
                "frozen CUDA environment cannot inherit user-site packages"
            )
        result = run_relationship_p4_steering_artifact_fit(
            output_dir=args.output_dir,
            protocol_path=args.protocol,
            progress=lambda message: print(
                f"[p4-steering-fit] {message}",
                flush=True,
            ),
        )
    else:
        raise AssertionError(f"unreachable command: {args.command}")
    print(
        json.dumps(
            {
                "artifact_id": result.artifact_id,
                "protocol_id": result.protocol_id,
                "bundle_id": result.bundle_id,
                "execution_attestation_id": (
                    result.execution_attestation_id
                ),
                "prerequisite_passed": result.prerequisite_passed,
                "verdict": result.verdict,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if args.command == "run" and not result.prerequisite_passed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
