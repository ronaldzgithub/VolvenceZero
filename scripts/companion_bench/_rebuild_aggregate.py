"""Rebuild aggregate_results.json from per-system summary.json files.

Recovery utility for situations where score_reference_systems was
killed mid-loop (e.g. a downstream SUT subprocess hung) and the
aggregate file was never written. Walks ``--artifact-dir`` for
subdirectories containing ``summary.json`` and rebuilds the aggregate.

Usage::

    python scripts/companion_bench/_rebuild_aggregate.py \\
        --artifact-dir artifacts/companion_bench_smoke
"""

from __future__ import annotations

import argparse
import json
import pathlib


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="_rebuild_aggregate")
    parser.add_argument("--artifact-dir", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    if not args.artifact_dir.is_dir():
        print(f"error: artifact-dir {args.artifact_dir} does not exist")
        return 2

    aggregate: dict = {"systems": []}
    for sub_dir in sorted(args.artifact_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        summary_path = sub_dir / "summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = payload.get("manifest", {})
        aggregate["systems"].append(
            {
                "submission_id": manifest.get("submission_id") or sub_dir.name,
                "system_name": manifest.get("system_name"),
                "leaderboard_category": manifest.get("leaderboard_category", "bespoke"),
                "summary": payload,
            }
        )

    out_path = args.artifact_dir / "aggregate_results.json"
    out_path.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"rebuilt {out_path} with {len(aggregate['systems'])} systems")
    for s in aggregate["systems"]:
        summ = s["summary"]
        aggp = summ.get("aggregate", {})
        cost = summ.get("cost", {})
        final = aggp.get("final_mean")
        final_str = f"{final:6.2f}" if isinstance(final, (int, float)) else "  n/a"
        cost_usd = cost.get("total_usd")
        cost_str = f"${cost_usd:.4f}" if isinstance(cost_usd, (int, float)) else "n/a"
        print(
            f"  {s['submission_id']:42s} "
            f"final={final_str}  arcs={aggp.get('arc_count', 0)}  cost={cost_str}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
