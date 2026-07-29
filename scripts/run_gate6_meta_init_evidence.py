from __future__ import annotations

import argparse
import json
from pathlib import Path

from volvence_zero.agent.gate6_meta_init_evidence import (
    export_gate6_meta_init_bundle,
    run_gate6_development_probe,
    verify_gate6_meta_init_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the preregistered Gate 6 nested meta-init campaign."
    )
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--seed", type=int, default=401)
    args = parser.parse_args()
    if args.verify is not None:
        payload = verify_gate6_meta_init_bundle(args.verify)
    elif args.development:
        if args.trace_root is None:
            parser.error("--development requires --trace-root")
        payload = run_gate6_development_probe(
            trace_root=args.trace_root,
            seed=args.seed,
        )
    else:
        if args.trace_root is None or args.output_dir is None:
            parser.error("formal run requires --trace-root and --output-dir")
        written = export_gate6_meta_init_bundle(
            trace_root=args.trace_root,
            output_dir=args.output_dir,
        )
        payload = verify_gate6_meta_init_bundle(args.output_dir)
        payload["artifact_files"] = [path.name for path in written]
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
