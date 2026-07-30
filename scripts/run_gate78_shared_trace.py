from __future__ import annotations

import argparse
import json
from pathlib import Path

from volvence_zero.agent.gate78_shared_trace import (
    GATE7_V3_TRACE_PROFILE,
    GATE78_V2_TRACE_PROFILE,
    export_gate78_shared_trace_bundle,
    verify_gate78_shared_trace_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or verify a preregistered Gate 7 source corpus."
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument(
        "--profile",
        choices=("v2", "gate7-v3"),
        default="v2",
    )
    args = parser.parse_args()
    profile = (
        GATE7_V3_TRACE_PROFILE
        if args.profile == "gate7-v3"
        else GATE78_V2_TRACE_PROFILE
    )
    if (args.output_dir is None) == (args.verify is None):
        parser.error("choose exactly one of --output-dir or --verify")
    if args.verify is not None:
        payload = verify_gate78_shared_trace_bundle(
            args.verify,
            profile=profile,
        )
    else:
        assert args.output_dir is not None
        written = export_gate78_shared_trace_bundle(
            output_dir=args.output_dir,
            profile=profile,
        )
        payload = verify_gate78_shared_trace_bundle(
            args.output_dir,
            profile=profile,
        )
        payload["artifact_files"] = [
            str(path.relative_to(args.output_dir)) for path in written
        ]
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
