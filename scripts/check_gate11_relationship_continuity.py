#!/usr/bin/env python3
"""CI check for an existing Gate 11 four-arm evidence artifact."""

from __future__ import annotations

import argparse
import json

from volvence_zero.agent.gate11_per_user_continuity_evidence import (
    evaluate_gate11_continuity_regression,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root")
    args = parser.parse_args()
    result = evaluate_gate11_continuity_regression(args.artifact_root)
    print(
        json.dumps(
            {
                "passed": result.passed,
                "failed_gates": list(result.failed_gates),
                "comparison_gains": dict(result.comparison_gains),
                "schema_version": result.schema_version,
            },
            sort_keys=True,
        )
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
