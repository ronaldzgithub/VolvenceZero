"""P5 #52: 6-axis weight + A6 cap calibration sweep.

Validates that the current weights (0.10/0.15/0.25/0.20/0.10/0.20)
+ A6_CAP_THRESHOLD=60 are robust to ±0.05 axis-weight perturbation
+ A6 cap ∈ {50, 55, 60, 65, 70} sweep. Outputs the 105-config
sensitivity matrix + ranking stability summary used by the public
calibration report.

This is the second public-launch credibility check (alongside #48
judge robustness): the first question RFC reviewers will ask is
"why these weights, why this A6 cap" — this script answers with a
sensitivity matrix.

Run::

    python scripts/companion_bench/calibration_sweep.py --dry-run

Refs:
    docs/moving forward/companion-bench-public-launch-packet.md §2.2
    docs/external/companion-bench-calibration-report-v0.md
    docs/known-debts.md #52
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
        "--axis-weight-step",
        type=float,
        default=0.05,
        help="±step around current axis weights (default 0.05).",
    )
    parser.add_argument(
        "--a6-caps",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default=(50, 55, 60, 65, 70),
        help="Comma-separated A6 cap values to sweep.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts/companion_bench")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _emit_placeholder_artifact(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()
    out_path = out_dir / f"calibration_sweep-{today}.json"
    n_axis_configs = 6 * 3  # 6 axes × {-step, 0, +step}
    n_total_configs = n_axis_configs * len(args.a6_caps)
    payload = {
        "scaffold_status": "SHADOW",
        "sweep_kind": "calibration",
        "axis_weight_step": args.axis_weight_step,
        "a6_caps": list(args.a6_caps),
        "n_total_configs": n_total_configs,
        "seed": args.seed,
        "dry_run": True,
        "current_weights": {
            "A1": 0.10,
            "A2": 0.15,
            "A3": 0.25,
            "A4": 0.20,
            "A5": 0.10,
            "A6": 0.20,
        },
        "current_a6_cap": 60,
        "ranking_stability_under_perturbation": None,
        "sensitivity_matrix": None,
        "notes": (
            f"P5 #52 SHADOW. Real sweep runs {n_total_configs} configs × "
            "5 reference SUT × 24 scenarios. Per-config ranking compared "
            "against baseline current_weights. Output drives "
            "docs/external/companion-bench-calibration-report-v0.md "
            "+ adds WEIGHTS_VERSION docstring to aggregator.py."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.dry_run:
        sys.stderr.write(
            "P5 #52 SHADOW: real calibration sweep not yet wired. "
            "Use --dry-run.\n"
        )
        return 2
    out_path = _emit_placeholder_artifact(args)
    print(f"wrote SHADOW placeholder: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
