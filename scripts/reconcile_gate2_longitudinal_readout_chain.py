"""Correct Gate 2 companion rows to the readout-only provenance label."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from volvence_zero.agent.gate2_longitudinal_capture import (
    reconcile_gate2_longitudinal_readout_chain,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    count = reconcile_gate2_longitudinal_readout_chain(
        output_root=args.output_root,
    )
    print(json.dumps({"updated_outcome_count": count}))


if __name__ == "__main__":
    main()
