"""Export Gate 2 longitudinal admission for the frozen v35 selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from volvence_zero.agent.gate2_longitudinal_readout import (
    export_gate2_longitudinal_readout_admission,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--selector-artifact", type=Path, required=True)
    parser.add_argument("--companion-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    verdict = export_gate2_longitudinal_readout_admission(
        source_root=args.source_root,
        selector_artifact_path=args.selector_artifact,
        output_dir=args.output_dir,
        companion_root=args.companion_root,
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
