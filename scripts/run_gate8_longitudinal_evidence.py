from __future__ import annotations

import argparse
import json
from pathlib import Path

from volvence_zero.agent.gate8_longitudinal_evidence import (
    export_gate8_longitudinal_bundle,
    verify_gate8_longitudinal_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Gate 8 cross-session wake/sleep evidence."
    )
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--transition-limit", type=int)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        payload = verify_gate8_longitudinal_bundle(args.verify)
    else:
        if args.trace_root is None or args.output_dir is None:
            parser.error("run requires --trace-root and --output-dir")
        written = export_gate8_longitudinal_bundle(
            trace_root=args.trace_root,
            output_dir=args.output_dir,
            transition_limit=args.transition_limit,
            formal_locked_run=not args.development,
        )
        payload = verify_gate8_longitudinal_bundle(args.output_dir)
        payload["artifact_files"] = [path.name for path in written]
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
