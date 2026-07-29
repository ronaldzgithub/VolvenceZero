from __future__ import annotations

import argparse
import json
from pathlib import Path

from volvence_zero.agent.shared_settled_trace import (
    aggregate_shared_settled_trace_campaign,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate the three Gate 4/5/6 shared trace seeds."
    )
    parser.add_argument("--campaign-dir", type=Path, required=True)
    args = parser.parse_args()
    written = aggregate_shared_settled_trace_campaign(
        campaign_dir=args.campaign_dir,
    )
    verdict = json.loads(
        (args.campaign_dir / "aggregate_verdict.json").read_text(
            encoding="utf-8"
        )
    )
    print(
        json.dumps(
            {
                "status": verdict["status"],
                "consumer_admission": verdict["consumer_admission"],
                "failed_gates": verdict["failed_gates"],
                "artifact_files": [path.name for path in written],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
