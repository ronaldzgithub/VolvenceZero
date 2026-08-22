"""CLI for the development-only P4 physical residual-actuation preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from volvence_zero.agent.relationship_p4_physical_residual_actuation import (
    run_relationship_p4_physical_residual_actuation,
    validate_relationship_p4_physical_residual_actuation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or GPU-free validate the frozen Windows/CUDA P4 physical residual-actuation development preflight."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "validate-existing"):
        child = subparsers.add_parser(command)
        child.add_argument("--output-dir", type=Path, required=True)
        child.add_argument("--input-fit-root", type=Path, required=True)
        child.add_argument("--campaign-manifest", type=Path, required=True)
        child.add_argument("--protocol", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        result = run_relationship_p4_physical_residual_actuation(
            output_dir=args.output_dir,
            input_fit_root=args.input_fit_root,
            campaign_manifest_path=args.campaign_manifest,
            protocol_path=args.protocol,
            progress=lambda message: print(message, flush=True),
        )
    else:
        result = validate_relationship_p4_physical_residual_actuation(
            output_dir=args.output_dir,
            input_fit_root=args.input_fit_root,
            campaign_manifest_path=args.campaign_manifest,
            protocol_path=args.protocol,
        )
    print(
        json.dumps(
            {
                "artifact_id": result.artifact_id,
                "protocol_id": result.protocol_id,
                "input_fit_artifact_id": result.input_fit_artifact_id,
                "execution_attestation_id": result.execution_attestation_id,
                "preflight_passed": result.preflight_passed,
                "verdict": result.verdict,
                "output_dir": str(result.output_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result.preflight_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
