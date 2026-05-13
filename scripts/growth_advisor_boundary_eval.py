"""P2 #64 G-A: per-boundary trigger rate baseline + reviewer GT precision/recall.

Loads ``data/growth_advisor_boundary_eval/<profile>/scenarios.jsonl``,
runs each user_turn through the growth-advisor lifeform with the
profile compiled, captures which boundary policies trigger, computes:

* per-boundary trigger rate (must stay in [5%, 50%] band)
* per-boundary precision / recall vs reviewer GT
* per-archetype boundary distribution

ACTIVE pass criteria (from packet G-A + commercialisation §4.2 P2
kill criteria):

* per-boundary trigger rate ∈ [0.05, 0.50]
* per-boundary precision >= 0.7
* per-boundary recall >= 0.6

SHADOW: real lifeform run depends on growth-advisor wheel
``compile`` + boundary policy enforcer + (optionally)
LLMArchetypeClassifier (debt #66). Until ACTIVE, ``--dry-run``
emits placeholder.

Run::

    python scripts/growth_advisor_boundary_eval.py --dry-run

Refs:
    docs/moving forward/growth-advisor-pilot-packet.md §2.1 G-A
    docs/specs/growth-advisor-boundary-baseline.md
    docs/known-debts.md #64
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path


_BOUNDARIES = (
    "bp-no-hard-sell",
    "bp-no-overclaim",
    "bp-no-flooding",
    "bp-no-judgmental",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-id", default="cheng-laoshi")
    parser.add_argument(
        "--gt-root",
        default="packages/lifeform-domain-growth-advisor/data/growth_advisor_boundary_eval",
    )
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--output-dir", default="artifacts/growth_advisor_eval")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _emit_placeholder_artifact(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()
    out_path = out_dir / f"boundary_baseline_{today}.json"

    gt_path = Path(args.gt_root) / args.profile_id.replace("-", "_") / "scenarios.jsonl"
    if not gt_path.exists():
        gt_path = gt_path.with_suffix(".jsonl.example")
    gt_count = 0
    if gt_path.exists():
        gt_count = sum(
            1 for line in gt_path.read_text(encoding="utf-8").splitlines() if line.strip()
        )

    payload = {
        "scaffold_status": "SHADOW",
        "report_kind": "boundary_baseline",
        "profile_id": args.profile_id,
        "n_scenarios": gt_count,
        "n_seeds": args.n_seeds,
        "dry_run": True,
        "per_boundary": {
            b: {
                "trigger_rate": None,
                "precision": None,
                "recall": None,
                "f1": None,
            }
            for b in _BOUNDARIES
        },
        "kill_criteria_band": [0.05, 0.50],
        "sla_thresholds": {
            "precision_min": 0.70,
            "recall_min": 0.60,
        },
        "notes": (
            "P2 #64 G-A SHADOW. Real eval requires growth-advisor lifeform "
            "+ boundary enforcer + (optional) LLMArchetypeClassifier. "
            "ACTIVE pass: trigger_rate ∈ [0.05, 0.50] AND precision ≥ 0.70 AND recall ≥ 0.60."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.dry_run:
        sys.stderr.write(
            "P2 #64 G-A SHADOW: real boundary eval not wired. Use --dry-run.\n"
        )
        return 2
    out_path = _emit_placeholder_artifact(args)
    print(f"wrote SHADOW placeholder: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
