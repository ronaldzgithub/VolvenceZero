#!/usr/bin/env python3
"""Evaluate a completed formal seven-day simulated companion run matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from volvence_zero.agent.seven_day_companion_evidence import (
    export_seven_day_ablation_bundle,
    load_seven_day_run_envelopes,
)
from volvence_zero.agent.seven_day_companion_preregistration import (
    validate_seven_day_companion_preregistration,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    preregistration = json.loads(
        args.preregistration.read_text(encoding="utf-8")
    )
    validate_seven_day_companion_preregistration(
        preregistration,
        repo_root=args.repo_root,
    )
    result = export_seven_day_ablation_bundle(
        runs=load_seven_day_run_envelopes(args.runs_dir),
        preregistration=preregistration,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "passed": result.passed,
                "case_count": result.case_count,
                "run_count": result.run_count,
            },
            sort_keys=True,
        )
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
