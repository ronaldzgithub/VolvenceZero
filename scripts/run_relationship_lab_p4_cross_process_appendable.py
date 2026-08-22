#!/usr/bin/env python3
"""Run or validate the P4 real child-process owner-hydration preflight."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _source_root in sorted((_REPO_ROOT / "packages").glob("*/src")):
    if _source_root.is_dir():
        sys.path.insert(0, str(_source_root))

from lifeform_evolution.relationship_lab_p4_cross_process_appendable import (  # noqa: E402
    run_relationship_p4_cross_process_appendable_preflight,
    run_relationship_p4_cross_process_worker,
    validate_relationship_p4_cross_process_report_files,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P4 real child-process owner-hydration preflight",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run")
    run.add_argument("--output-dir", type=pathlib.Path, required=True)
    run.add_argument("--protocol", type=pathlib.Path)
    run.add_argument("--python-executable", default=sys.executable)

    worker = subcommands.add_parser("worker-pulse")
    worker.add_argument("--request", type=pathlib.Path, required=True)
    worker.add_argument("--receipt", type=pathlib.Path, required=True)
    worker.add_argument("--run-root", type=pathlib.Path, required=True)

    validate = subcommands.add_parser("validate-existing")
    validate.add_argument("--output-dir", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.command == "worker-pulse":
        run_relationship_p4_cross_process_worker(
            request_path=args.request,
            receipt_path=args.receipt,
            run_root=args.run_root,
        )
        return 0
    if args.command == "validate-existing":
        validate_relationship_p4_cross_process_report_files(
            output_dir=args.output_dir,
        )
        print(json.dumps({"valid": True}, sort_keys=True))
        return 0
    if args.command == "run":
        if os.environ.get("PYTHONNOUSERSITE") != "1":
            raise RuntimeError(
                "P4 cross-process runner requires PYTHONNOUSERSITE=1 so child "
                "lineage cannot inherit an unrelated user-site environment"
            )
        report = run_relationship_p4_cross_process_appendable_preflight(
            output_dir=args.output_dir,
            protocol_path=args.protocol,
            worker_script=pathlib.Path(__file__).resolve(),
            python_executable=args.python_executable,
        )
        print(
            json.dumps(
                {
                    "artifact_id": report.artifact_id,
                    "invocation_count": len(report.pulses),
                    "correct_empty_forecast_presence_change_count": (
                        report.correct_empty_forecast_presence_change_count
                    ),
                    "correct_swapped_recommended_action_change_count": (
                        report.correct_swapped_recommended_action_change_count
                    ),
                    "formal_evidence_authorized": (
                        report.formal_evidence_authorized
                    ),
                    "verdict": report.verdict,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(f"unreachable command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
