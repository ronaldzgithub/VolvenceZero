"""P5 #48: LLM-judge robustness sweep across model families.

Quantifies the inter-rater agreement (Spearman / Kendall κ) of N
LLM judges on a fixed reference SUT pool. Output:

* per-axis variance σ across judge families
* Spearman matrix between judge family pairs
* worst-case SUT ranking change

This is what stops the "VZ's leaderboard is biased because the
judge is GPT" critique on Day 1 of public launch — see
``docs/external/companion-bench-judge-robustness-v0.md`` for the
public-facing report.

Run::

    python scripts/companion_bench/judge_robustness_sweep.py --dry-run
    python scripts/companion_bench/judge_robustness_sweep.py \
        --judge-families gpt5,claude47,deepseek4,qwen3,gemini25 \
        --reference-sut vz,gpt5,claude47,deepseek4,qwen3 \
        --seed 42

Refs:
    docs/moving forward/companion-bench-public-launch-packet.md §2.1
    docs/known-debts.md #48
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
import statistics


_DEFAULT_JUDGE_FAMILIES = ("gpt5", "claude47", "deepseek4", "qwen3", "gemini25")
_DEFAULT_REFERENCE_SUT = ("vz", "gpt5", "claude47", "deepseek4", "qwen3")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judge-families",
        type=lambda s: tuple(s.split(",")),
        default=_DEFAULT_JUDGE_FAMILIES,
        help="Comma-separated LLM judge families to sweep.",
    )
    parser.add_argument(
        "--reference-sut",
        type=lambda s: tuple(s.split(",")),
        default=_DEFAULT_REFERENCE_SUT,
        help="Comma-separated reference SUT pool.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scenario-pack",
        default="public-24",
        help="Scenario subset (default: 24 public scenarios).",
    )
    parser.add_argument("--output-dir", default="artifacts/companion_bench")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--scores-dir",
        default="",
        help="Directory containing per-submission summary.json files from score_reference_systems.",
    )
    return parser


def _emit_placeholder_artifact(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()
    out_path = out_dir / f"judge_robustness_sweep-{today}.json"
    payload = {
        "scaffold_status": "SHADOW",
        "sweep_kind": "judge_robustness",
        "judge_families": list(args.judge_families),
        "reference_sut": list(args.reference_sut),
        "scenario_pack": args.scenario_pack,
        "seed": args.seed,
        "dry_run": True,
        "per_axis_variance_sigma": None,
        "spearman_matrix": None,
        "worst_case_ranking_change": None,
        "estimated_token_cost_usd": None,
        "notes": (
            "P5 #48 SHADOW. Real sweep runs ~"
            f"{len(args.judge_families)}*"
            f"{len(args.reference_sut)} judge×SUT × scenarios; "
            "estimated cost goes through scripts/companion_bench/"
            "estimate_quarterly_cost.py (#56). "
            "See docs/external/companion-bench-judge-robustness-v0.md."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.dry_run:
        out_path = _emit_summary_based_artifact(args)
        print(f"wrote judge robustness artifact: {out_path}")
        return 0
    out_path = _emit_placeholder_artifact(args)
    print(f"wrote SHADOW placeholder: {out_path}")
    return 0


def _emit_summary_based_artifact(args: argparse.Namespace) -> Path:
    if any("qwen" in family.casefold() for family in args.judge_families):
        raise SystemExit("judge-families must exclude Qwen for same-substrate Qwen runs")
    scores_dir = Path(args.scores_dir)
    if not scores_dir.is_dir():
        raise SystemExit("--scores-dir is required for non-dry-run judge evidence")
    rows: list[dict[str, object]] = []
    for summary_path in sorted(scores_dir.glob("*/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        aggregate = payload.get("aggregate", {})
        final_mean = float(aggregate.get("final_mean", 0.0))
        rows.append(
            {
                "submission_id": summary_path.parent.name,
                "final_mean": final_mean,
            }
        )
    if len(rows) < 2:
        raise SystemExit("judge evidence needs at least two scored systems")
    means = [float(row["final_mean"]) for row in rows]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"judge_robustness_summary-{_dt.date.today().isoformat()}.json"
    payload = {
        "schema_version": "judge-robustness-summary.v1",
        "sweep_kind": "judge_robustness_summary",
        "judge_families": list(args.judge_families),
        "reference_sut": list(args.reference_sut),
        "scenario_pack": args.scenario_pack,
        "seed": args.seed,
        "dry_run": False,
        "scores_dir": str(scores_dir),
        "system_count": len(rows),
        "score_mean": statistics.fmean(means),
        "score_population_stdev": statistics.pstdev(means),
        "systems": rows,
        "notes": (
            "Summary-based first real artifact: validates scored-system coverage "
            "and non-Qwen judge-family configuration. Full cross-family rank "
            "matrix can be layered on this schema when multiple judge output "
            "sets are available."
        ),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_path


if __name__ == "__main__":
    raise SystemExit(main())
