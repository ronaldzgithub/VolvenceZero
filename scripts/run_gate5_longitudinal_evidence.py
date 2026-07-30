from __future__ import annotations

import argparse
import json
from pathlib import Path

from volvence_zero.agent.gate5_longitudinal_evidence import (
    export_gate5_longitudinal_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the preregistered Gate 5 cross-session CMS suite."
    )
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    written = export_gate5_longitudinal_bundle(
        trace_root=args.trace_root,
        output_dir=args.output_dir,
    )
    verdict = json.loads(
        (args.output_dir / "promotion_verdict.json").read_text(
            encoding="utf-8"
        )
    )
    print(
        json.dumps(
            {
                "status": verdict["status"],
                "mechanism_passed": verdict["mechanism_passed"],
                "longitudinal_passed": verdict["longitudinal_passed"],
                "failed_gates": verdict["failed_gates"],
                "artifact_files": [path.name for path in written],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
