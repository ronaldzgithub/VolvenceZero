"""P2 #68 G-A: drives ablation evidence (4 conditions × per-boundary delta).

Validates that each ``GrowthAdvisorDrivePrior`` (trust_building /
empathy_response / restraint_against_pitch / kb_share) **causally**
moves the corresponding boundary trigger rate, not just nominally
compiles into an owner.

4 conditions:

* full
* no-restraint (drop ``restraint_against_pitch_drive``)
* no-empathy (drop ``empathy_response_drive``)
* no-trust (drop ``trust_building_drive``)

Per packet G-A SLA: ``no-restraint`` must reduce ``bp-no-hard-sell``
trigger rate by ≥ 30% relative (causal evidence the drive matters).

SHADOW: run the same fixture as boundary_eval.py × 4 condition
variants. Until ACTIVE, ``--dry-run`` emits placeholder.

Refs:
    docs/moving forward/growth-advisor-pilot-packet.md §2.1 G-A
    docs/specs/growth-advisor-drive-ablation-evidence.md
    docs/known-debts.md #68
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path


_CONDITIONS = ("full", "no-restraint", "no-empathy", "no-trust")
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
    out_path = out_dir / f"drive_ablation_{today}.json"

    payload = {
        "scaffold_status": "SHADOW",
        "report_kind": "drive_ablation",
        "profile_id": args.profile_id,
        "conditions": list(_CONDITIONS),
        "n_seeds": args.n_seeds,
        "dry_run": True,
        "per_condition": {
            condition: {
                "boundary_trigger_rates": {b: None for b in _BOUNDARIES},
                "regime_distribution": None,
                "response_style_proxy": None,
            }
            for condition in _CONDITIONS
        },
        "ablation_check_no_restraint_vs_full": {
            "bp-no-hard-sell_relative_decrease": None,
            "sla_min_relative_decrease": 0.30,
        },
        "notes": (
            "P2 #68 SHADOW. Real ablation: dataclasses.replace(profile, "
            "drive_priors=...) per condition × same fixture × output "
            "comparison. ACTIVE pass: no-restraint ↓ bp-no-hard-sell ≥ 30%."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.dry_run:
        sys.stderr.write(
            "P2 #68 SHADOW: real drive ablation not wired. Use --dry-run.\n"
        )
        return 2
    out_path = _emit_placeholder_artifact(args)
    print(f"wrote SHADOW placeholder: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
