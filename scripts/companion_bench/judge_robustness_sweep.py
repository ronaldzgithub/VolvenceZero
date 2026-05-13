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
        sys.stderr.write(
            "P5 #48 SHADOW: real judge sweep not yet wired (depends on "
            "scripts/companion_bench/score_reference_systems.py SUT "
            "orchestration + N judge family API keys). Use --dry-run.\n"
        )
        return 2
    out_path = _emit_placeholder_artifact(args)
    print(f"wrote SHADOW placeholder: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
