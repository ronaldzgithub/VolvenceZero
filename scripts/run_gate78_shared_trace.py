from __future__ import annotations

import argparse
import json
from pathlib import Path

from volvence_zero.agent.gate78_shared_trace import (
    export_gate78_shared_trace_bundle,
    verify_gate78_shared_trace_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or verify the preregistered Gate 7/8 v2 source corpus."
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if (args.output_dir is None) == (args.verify is None):
        parser.error("choose exactly one of --output-dir or --verify")
    if args.verify is not None:
        payload = verify_gate78_shared_trace_bundle(args.verify)
    else:
        assert args.output_dir is not None
        written = export_gate78_shared_trace_bundle(
            output_dir=args.output_dir
        )
        payload = verify_gate78_shared_trace_bundle(args.output_dir)
        payload["artifact_files"] = [
            str(path.relative_to(args.output_dir)) for path in written
        ]
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
