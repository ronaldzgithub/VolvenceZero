"""Freeze Gate 8/11 pilot-analysis implementation details."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from volvence_zero.agent.gate811_human_anchor_analysis import (
    build_gate811_analysis_preregistration,
    write_gate811_analysis_preregistration,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--human-anchor-preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--created-at-unix-ms", type=int)
    args = parser.parse_args()
    created_at = args.created_at_unix_ms
    if created_at is None:
        created_at = time.time_ns() // 1_000_000
    payload = build_gate811_analysis_preregistration(
        repo_root=args.repo_root,
        human_anchor_preregistration_path=(
            args.human_anchor_preregistration
        ),
        created_at_unix_ms=created_at,
    )
    manifest = write_gate811_analysis_preregistration(
        payload=payload,
        output_path=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
