"""P5 #56: estimate quarterly running cost for Companion Bench.

Reads ``CostTracker._DEFAULT_PRICES`` + the actual token usage
recorded by past sweep runs (#48 / #52 / #53 / #54 +
``score_reference_systems.py``) and projects the next-quarter
cost so the leaderboard maintenance budget stays explicit.

Run::

    python scripts/companion_bench/estimate_quarterly_cost.py --dry-run

Output:
    artifacts/companion_bench/quarterly_cost_estimate-<date>.md

Refs:
    docs/moving forward/companion-bench-public-launch-packet.md §3
    docs/external/companion-bench-cost-model-v0.md
    docs/known-debts.md #56
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--past-runs-dir",
        default="artifacts/companion_bench",
        help="Directory containing past sweep / reference-run JSON artifacts.",
    )
    parser.add_argument("--quarters", type=int, default=8)
    parser.add_argument(
        "--reference-models",
        type=lambda s: tuple(s.split(",")),
        default=("gpt5", "claude47", "deepseek4", "qwen3", "gemini25"),
    )
    parser.add_argument("--scenarios-per-quarter", type=int, default=120)
    parser.add_argument("--output-dir", default="artifacts/companion_bench")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _emit_placeholder_artifact(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()
    out_path = out_dir / f"quarterly_cost_estimate-{today}.md"

    body = f"""# Companion Bench Quarterly Cost Estimate

> Status: SHADOW placeholder (dry-run)
> Generated: {today}

## Inputs

- Past runs dir: `{args.past_runs_dir}`
- Quarters projected: {args.quarters}
- Reference models: {", ".join(args.reference_models)}
- Scenarios per quarter: {args.scenarios_per_quarter}

## Output (placeholder)

Real estimation walks past sweep artifacts and `CostTracker._DEFAULT_PRICES`
to project per-quarter token cost. Until P5 #48 / #52 sweeps run for real,
this artifact is a placeholder; numbers will populate once at least one
real sweep emits true token usage.

## Phase A 6-month total budget envelope (from packet §7.2)

- API sweep evidence (#48 + #52 + #53 + #54): ~$8-11k USD
- Reference SUT release-tier run (#32 sub-track 1): ~$5-15k USD
- Buffer: ~12-23%
- **Total**: ~$13.3-26.5k USD ≈ ¥93-186k RMB

## Notes

- Submitter-side cost guidance: see `docs/external/companion-bench-cost-model-v0.md`
- Trusted-runner mode: VZ may charge $X / submission for held-out runs
  (see `docs/external/companion-bench-trusted-runner-protocol.md`)
"""
    out_path.write_text(body, encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.dry_run:
        sys.stderr.write(
            "P5 #56 SHADOW: real cost estimation depends on past sweep "
            "artifacts existing in --past-runs-dir. Use --dry-run.\n"
        )
        return 2
    out_path = _emit_placeholder_artifact(args)
    print(f"wrote SHADOW placeholder: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
