"""P5 #54: statistical power analysis for the leaderboard.

Validates that 24 public + 96 held-out = 120 scenario × 6 axis
yields enough power to distinguish two SUTs that are X ELO points
apart with 95% confidence. Outputs a power curve so the
leaderboard can label some pairs as "indistinguishable" rather
than overclaim a tied-but-unfair ranking.

Run::

    python scripts/companion_bench/statistical_power_analysis.py --dry-run

Refs:
    docs/moving forward/companion-bench-public-launch-packet.md §2.4
    docs/external/companion-bench-statistical-power-v0.md
    docs/known-debts.md #54
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-scenarios",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default=(24, 48, 96, 120, 200),
        help="Comma-separated scenario count points to estimate power at.",
    )
    parser.add_argument(
        "--n-seeds-per-scenario",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--reference-sut",
        type=lambda s: tuple(s.split(",")),
        default=("vz", "gpt5", "claude47", "deepseek4", "qwen3"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts/companion_bench")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _emit_placeholder_artifact(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()
    out_path = out_dir / f"statistical_power_analysis-{today}.json"
    payload = {
        "scaffold_status": "SHADOW",
        "sweep_kind": "statistical_power",
        "n_scenarios_grid": list(args.n_scenarios),
        "n_seeds_per_scenario": args.n_seeds_per_scenario,
        "reference_sut": list(args.reference_sut),
        "bootstrap_resamples": args.bootstrap_resamples,
        "seed": args.seed,
        "dry_run": True,
        "power_curve_distinguishable_threshold": None,
        "elo_95_ci_per_sut": None,
        "leaderboard_indistinguishable_pairs": None,
        "notes": (
            "P5 #54 SHADOW. Bootstrap-CI driven power curve. "
            "If at n=24 distinguishable threshold > 60 ELO points, "
            "v1.0 leaderboard ships with 'distinguishable bands' "
            "(top / mid / bottom tier) instead of raw ELO numbers."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.dry_run:
        sys.stderr.write("P5 #54 SHADOW: real power analysis not wired. Use --dry-run.\n")
        return 2
    out_path = _emit_placeholder_artifact(args)
    print(f"wrote SHADOW placeholder: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
