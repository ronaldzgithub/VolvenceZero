#!/usr/bin/env python3
"""Write the immutable seven-day simulated companion preregistration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from volvence_zero.agent.seven_day_companion_preregistration import (
    build_seven_day_companion_preregistration,
    write_seven_day_companion_preregistration,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--created-at-unix-ms", type=int)
    args = parser.parse_args()
    created_at = args.created_at_unix_ms or int(time.time() * 1000)
    payload = build_seven_day_companion_preregistration(
        repo_root=args.repo_root,
        created_at_unix_ms=created_at,
    )
    digest = write_seven_day_companion_preregistration(
        payload=payload,
        output_path=args.output,
    )
    print(
        json.dumps(
            {"output": str(args.output), "sha256": digest},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
