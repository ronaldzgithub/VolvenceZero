"""F-A: realistic concurrent load generator for the growth-advisor vertical.

Drives N=50 end_user × 10 客户 ("席位") through the DLaaS typed
envelope + admin endpoints, exercising the ``cheng-laoshi`` profile
across the onboarding-arc playbook (relationship phase routing flows
through ``BehaviorProtocol.TemporalArc.progression_signals`` PE-driven
in protocol-runtime; calendar-day routing was removed 2026-05-14).
Captures the 4 anti-sales boundary trigger rates (``bp-no-hard-sell``
/ ``bp-no-overclaim`` / ``bp-no-flooding`` / ``bp-no-judgmental``) so
load tuning cannot silently degrade boundary enforcement.

SHADOW scaffold: CLI + ``--dry-run`` placeholder; real wiring lands
as F-A subtask 4 (see
``docs/moving forward/cross-cutting-foundation-packet.md`` §2.1).

Run::

    python scripts/realistic_load_growth_advisor.py --dry-run \
        --n-end-users 50 --n-tenants 10
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
    parser.add_argument("--n-end-users", type=int, default=50)
    parser.add_argument("--n-tenants", type=int, default=10)
    parser.add_argument(
        "--profile-id",
        default="cheng-laoshi",
        help="Growth-advisor profile id (default cheng-laoshi).",
    )
    parser.add_argument(
        "--dlaas-base-url",
        default="http://localhost:8000",
    )
    parser.add_argument("--days-window", type=int, default=7)
    parser.add_argument("--output-dir", default="artifacts/perf")
    parser.add_argument("--dry-run", action="store_true")
    return parser


_GROWTH_ADVISOR_EXPECTED_SLO = {
    "p50_turn_latency_s_max": 1.5,
    "p99_turn_latency_s_max": 3.0,
    "concurrent_ai_id_min": 100,
    "gpu_mem_peak_capacity_pct_max": 0.50,
    "boundary_trigger_rate_band": [0.05, 0.50],
    "handoff_queue_p99_s_max": 30.0,
}


def _sample_shape(args: argparse.Namespace) -> dict[str, object]:
    return {
        "is_sample": True,
        "p50_turn_latency_s": 0.81,
        "p99_turn_latency_s": 2.34,
        "boundary_trigger_rates": {
            "bp-no-hard-sell": 0.18,
            "bp-no-overclaim": 0.09,
            "bp-no-flooding": 0.06,
            "bp-no-judgmental": 0.04,
        },
        "archetype_distribution": {
            "anxious": 0.34,
            "comparing": 0.18,
            "standard_seeking": 0.22,
            "venting": 0.16,
            "product_seeking": 0.10,
        },
        "handoff_queue_p99_ms": 21_400.0,
        "n_end_users_actual": args.n_end_users,
        "n_tenants_actual": args.n_tenants,
        "errors": [],
    }


def _emit_placeholder_artifact(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()
    out_path = out_dir / f"growth-advisor-{today}.json"
    payload = {
        "scaffold_status": "SHADOW",
        "vertical": "growth_advisor",
        "profile_id": args.profile_id,
        "n_end_users": args.n_end_users,
        "n_tenants": args.n_tenants,
        "days_window": args.days_window,
        "dry_run": True,
        "expected_slo": _GROWTH_ADVISOR_EXPECTED_SLO,
        "sample_shape": _sample_shape(args),
        "p50_turn_latency_s": None,
        "p99_turn_latency_s": None,
        "boundary_trigger_rates": {b: None for b in _BOUNDARIES},
        "archetype_distribution": None,
        "handoff_queue_p99_ms": None,
        "errors": [],
        "notes": (
            "F-A SHADOW scaffold. Real DLaaS multi-tenant load lands "
            "with cross-cutting-foundation-packet §2.1 subtask 4. "
            "Per-boundary trigger rate must stay in [0.05, 0.50] "
            "(growth-advisor-pilot-packet G-A kill criteria); "
            "expected_slo + sample_shape are review aids."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.dry_run:
        sys.stderr.write(
            "F-A SHADOW: real DLaaS multi-tenant load not yet wired. "
            "Use --dry-run.\n"
        )
        return 2
    out_path = _emit_placeholder_artifact(args)
    print(f"wrote placeholder artifact: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
