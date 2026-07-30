"""Run/resume Gate 2 longitudinal v35 companion capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from volvence_zero.agent.gate2_longitudinal_capture import (
    run_gate2_longitudinal_companion_capture,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--selector-artifact", type=Path, required=True)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()
    verdict = run_gate2_longitudinal_companion_capture(
        source_root=args.source_root,
        selector_artifact_path=args.selector_artifact,
        candidate_artifact_path=args.candidate_artifact,
        output_root=args.output_root,
        seeds=args.seeds,
        max_records=args.max_records,
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
