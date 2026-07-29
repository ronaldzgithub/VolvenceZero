from __future__ import annotations

import argparse
import json
from pathlib import Path

from volvence_zero.agent.gate1_pe_mechanism_evidence import (
    export_gate1_pe_mechanism_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the preregistered Gate 1 PE/LSS mechanism packet."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    written = export_gate1_pe_mechanism_bundle(
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
                "failed_gates": verdict["failed_gates"],
                "artifact_files": [path.name for path in written],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
