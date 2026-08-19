#!/usr/bin/env python3
"""Run Relationship Lab v0 Gate 0 calibration.

Default mode is an honest machinery-only run.  It should produce
``machinery_ready=true`` and ``gate0_passed=false`` until a content-addressed
real-substrate stateless/raw baseline attestation is supplied.

Exit codes:

* 0: requested verdict passed;
* 2: machinery or Gate 0 did not pass (report is still written);
* uncaught exception: malformed input or infrastructure failure.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(
    0,
    str(_REPO_ROOT / "packages" / "lifeform-domain-emogpt" / "src"),
)
sys.path.insert(
    0,
    str(_REPO_ROOT / "packages" / "lifeform-evolution" / "src"),
)

from lifeform_evolution.relationship_lab_gate0 import (  # noqa: E402
    Gate0CalibrationConfig,
    load_frozen_baseline_attestation,
    run_relationship_gate0_calibration,
    write_relationship_gate0_report,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "artifacts" / "relationship_lab" / f"gate0_{int(time.time())}"),
    )
    parser.add_argument("--baseline-attestation", default="")
    parser.add_argument("--samples-per-action", type=int, default=256)
    parser.add_argument("--minimum-action-effect", type=float, default=0.5)
    parser.add_argument("--maximum-baseline-accuracy", type=float, default=0.85)
    parser.add_argument("--minimum-baseline-decisions", type=int, default=24)
    parser.add_argument(
        "--machinery-only",
        action="store_true",
        help="Exit successfully when machinery_ready is true even though the real-substrate baseline tooth is pending.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    baseline = (
        load_frozen_baseline_attestation(pathlib.Path(args.baseline_attestation)) if args.baseline_attestation else None
    )
    config = Gate0CalibrationConfig(
        samples_per_action=args.samples_per_action,
        minimum_action_effect=args.minimum_action_effect,
        maximum_baseline_accuracy=args.maximum_baseline_accuracy,
        minimum_baseline_decisions=args.minimum_baseline_decisions,
    )
    report = run_relationship_gate0_calibration(config=config, baseline=baseline)
    json_path, _markdown_path = write_relationship_gate0_report(
        report,
        pathlib.Path(args.output_dir),
    )
    print(
        json.dumps(
            {
                "artifact_id": report.artifact_id,
                "machinery_ready": report.machinery_ready,
                "gate0_passed": report.gate0_passed,
                "report": str(json_path),
            },
            ensure_ascii=False,
        )
    )
    requested_pass = report.machinery_ready if args.machinery_only else report.gate0_passed
    return 0 if requested_pass else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
