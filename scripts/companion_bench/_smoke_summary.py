"""Print key smoke-run metrics from site detail JSON.

Used to feed SMOKE_REPORT.md content. Reads
``site/data/submissions/<id>.json`` (build_site output) and prints
per-axis means / per-family means / final score / arc-level
breakdown.
"""

from __future__ import annotations

import argparse
import json
import pathlib


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="_smoke_summary")
    parser.add_argument(
        "--detail-json",
        type=pathlib.Path,
        required=True,
        help="Path to site/data/submissions/<id>.json",
    )
    args = parser.parse_args(argv)

    data = json.loads(args.detail_json.read_text(encoding="utf-8"))
    print(f"=== Submission: {data['submission_id']} ({data.get('system_name')}) ===")
    print(f"category: {data.get('leaderboard_category')}")
    agg = data.get("aggregate", {})
    print(f"companionbench_final: {agg.get('companionbench_final')}")
    print(f"final_ci95: {agg.get('final_ci95')}")
    print(f"arc_count: {agg.get('arc_count')}")
    print(f"a6_cap_applied: {agg.get('a6_cap_applied')}")
    print(f"trueskill_conservative: {agg.get('trueskill_conservative')}")
    print(f"bradley_terry_score: {agg.get('bradley_terry_score')}")
    print()
    print("=== Per-axis means ===")
    for axis, val in (agg.get("axis_means") or {}).items():
        ci = (agg.get("axis_ci95") or {}).get(axis, [None, None])
        print(f"  {axis}: {val:6.2f}  CI95=[{ci[0]:.2f}, {ci[1]:.2f}]")
    print()
    print("=== Per-family means ===")
    for fam, fdata in (data.get("family_means") or {}).items():
        ci = fdata.get("ci95") or [None, None]
        print(f"  {fam}: mean={fdata.get('mean'):6.2f}  arcs={fdata.get('arc_count')}")
    print()
    print("=== Per-arc breakdown ===")
    for arc in data.get("arcs", []):
        print(
            f"  {arc.get('scenario_id'):30s}  family={arc.get('family')}  "
            f"final={arc.get('final_score'):6.2f}  cap={arc.get('a6_cap_applied')}"
        )
    print()
    print("=== Cost ===")
    cost = data.get("cost", {})
    totals = cost.get("totals", {})
    print(f"  total_usd: {totals.get('total_usd')}")
    print(f"  sut_usd: {totals.get('sut_usd')}")
    print(f"  perturn_usd: {totals.get('perturn_usd')}")
    print(f"  arc_usd: {totals.get('arc_usd')}")
    print(f"  missing_models: {cost.get('missing_models')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
