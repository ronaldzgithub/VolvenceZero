from __future__ import annotations

import argparse
import json
from pathlib import Path

from volvence_zero.agent.gate5_cms_pareto_evidence import (
    export_gate5_cms_pareto_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the preregistered Gate 5 CMS Pareto campaign."
    )
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    written = export_gate5_cms_pareto_bundle(
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
                "causal_passed": verdict["causal_passed"],
                "failed_gates": verdict["failed_gates"],
                "artifact_files": [path.name for path in written],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
