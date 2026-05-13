"""P1 #63: per-figure cost summary from audit logs.

Reads all audit records under ``data/figure_audit/<figure_id>/``,
groups by ``bundle_id``, accumulates ``cost_breakdown`` field
(introduced in audit.py via debt #63), outputs a per-bundle cost
table + a per-action breakdown + a projected price recommendation.

This is the data behind ``docs/business/figure-bake-cost-actuals.md``
which replaces the §6.2 estimated 5-15万 COGS with measured numbers.

Run::

    python scripts/figure_cost_summary.py --figure-id einstein

Output:
    artifacts/figure_cost_summary/<figure_id>-<date>.json + .md

Refs:
    docs/moving forward/figure-evidence-packet.md §2.6
    docs/business/figure-bake-cost-actuals.md
    docs/known-debts.md #63
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
from pathlib import Path


_DEFAULT_RATE_TABLE = {
    "engineer_hours": 800.0,  # 元/天 ≈ 100/小时 × 8
    "reviewer_hours": 200.0,  # 元/小时
    "gpu_hours": 5.0,         # 元/小时（A100 spot 基线）
    "archive_fetch_wallclock_hours": 50.0,  # 元/小时（人工值守）
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-id", default="einstein")
    parser.add_argument(
        "--audit-root",
        default="data/figure_audit",
    )
    parser.add_argument("--output-dir", default="artifacts/figure_cost_summary")
    parser.add_argument(
        "--rate-engineer-per-day",
        type=float,
        default=_DEFAULT_RATE_TABLE["engineer_hours"],
    )
    parser.add_argument(
        "--rate-reviewer-per-hour",
        type=float,
        default=_DEFAULT_RATE_TABLE["reviewer_hours"],
    )
    parser.add_argument(
        "--rate-gpu-per-hour",
        type=float,
        default=_DEFAULT_RATE_TABLE["gpu_hours"],
    )
    return parser


def _load_audit_records(audit_root: Path, figure_id: str) -> list[dict]:
    figure_dir = audit_root / figure_id
    if not figure_dir.is_dir():
        return []
    out: list[dict] = []
    for path in sorted(figure_dir.iterdir()):
        if path.suffix != ".json":
            continue
        out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


def _accumulate_cost(records: list[dict]) -> dict:
    per_bundle: dict[str, dict] = collections.defaultdict(
        lambda: {"actions": [], "cost_breakdown": collections.defaultdict(float)}
    )
    for record in records:
        bundle_id = record.get("bundle_id", "unknown")
        per_bundle[bundle_id]["actions"].append(record.get("action", "unknown"))
        breakdown = record.get("cost_breakdown") or {}
        for k, v in breakdown.items():
            per_bundle[bundle_id]["cost_breakdown"][k] += float(v)
    return {
        bundle_id: {
            "actions": data["actions"],
            "cost_breakdown": dict(data["cost_breakdown"]),
        }
        for bundle_id, data in per_bundle.items()
    }


def _project_rmb(cost_breakdown: dict, args: argparse.Namespace) -> float:
    total = 0.0
    total += cost_breakdown.get("engineer_hours", 0.0) * (args.rate_engineer_per_day / 8)
    total += cost_breakdown.get("reviewer_hours", 0.0) * args.rate_reviewer_per_hour
    total += cost_breakdown.get("gpu_hours", 0.0) * args.rate_gpu_per_hour
    return total


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()
    json_path = out_dir / f"{args.figure_id}-{today}.json"
    md_path = out_dir / f"{args.figure_id}-{today}.md"

    records = _load_audit_records(Path(args.audit_root), args.figure_id)
    by_bundle = _accumulate_cost(records)
    per_bundle_rmb = {
        bid: {
            "actions": data["actions"],
            "cost_breakdown": data["cost_breakdown"],
            "projected_rmb": _project_rmb(data["cost_breakdown"], args),
        }
        for bid, data in by_bundle.items()
    }

    json_payload = {
        "figure_id": args.figure_id,
        "report_date": today,
        "audit_record_count": len(records),
        "rate_table": {
            "engineer_per_day_rmb": args.rate_engineer_per_day,
            "reviewer_per_hour_rmb": args.rate_reviewer_per_hour,
            "gpu_per_hour_rmb": args.rate_gpu_per_hour,
        },
        "per_bundle": per_bundle_rmb,
        "total_rmb": sum(b["projected_rmb"] for b in per_bundle_rmb.values()),
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    md_lines = [
        f"# Figure cost summary: {args.figure_id}",
        f"",
        f"> Generated: {today}",
        f"> Audit records: {len(records)}",
        f"",
        f"## Per-bundle cost",
        f"",
        f"| bundle_id | actions | engineer_h | reviewer_h | gpu_h | projected ¥ |",
        f"|---|---|---|---|---|---|",
    ]
    for bid, data in per_bundle_rmb.items():
        breakdown = data["cost_breakdown"]
        md_lines.append(
            f"| `{bid}` | {len(data['actions'])} | "
            f"{breakdown.get('engineer_hours', 0):.1f} | "
            f"{breakdown.get('reviewer_hours', 0):.1f} | "
            f"{breakdown.get('gpu_hours', 0):.1f} | "
            f"¥{data['projected_rmb']:.0f} |"
        )
    md_lines.append("")
    md_lines.append(
        f"**Total projected**: ¥{json_payload['total_rmb']:.0f}"
    )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"wrote: {json_path}")
    print(f"wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
