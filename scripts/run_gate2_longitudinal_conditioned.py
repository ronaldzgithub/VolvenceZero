"""Run/resume the preregistered Gate 2 conditioned longitudinal lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from volvence_zero.agent.gate2_longitudinal_conditioned import (
    run_gate2_conditioned_evidence,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-training-records", type=int)
    parser.add_argument("--max-evaluation-records", type=int)
    args = parser.parse_args()

    verdict = run_gate2_conditioned_evidence(
        repo_root=args.repo_root,
        preregistration_path=args.preregistration,
        output_root=args.output_root,
        max_training_records=args.max_training_records,
        max_evaluation_records=args.max_evaluation_records,
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
