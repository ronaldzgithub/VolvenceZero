from __future__ import annotations

import argparse
import json
from pathlib import Path

from volvence_zero.agent.gate1_pe_causal_evidence import (
    export_gate1_pe_causal_bundle_sync,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the preregistered Gate 1 PE-drive causal packet."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--full-matrix",
        action="store_true",
        help="Resume seed 101 and run seeds 211/307 only if the probe passes.",
    )
    args = parser.parse_args()
    written = export_gate1_pe_causal_bundle_sync(
        output_dir=args.output_dir,
        full_matrix=args.full_matrix,
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
                "causal_status": verdict["causal_status"],
                "failed_gates": verdict["failed_gates"],
                "artifact_files": [path.name for path in written],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
