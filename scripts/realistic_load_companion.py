"""F-A: realistic concurrent load generator for the companion vertical.

Drives N concurrent ``LifeformSession`` instances via the closed-alpha
HTTP service (``/v1/sessions/...``) using prompts derived from the
``companion-bench`` 6 family × 4 scenario user simulator. Outputs
per-session latency / GPU mem / owner-snapshot dispatch telemetry to
``artifacts/perf/companion-<date>.json``.

This is a **SHADOW scaffold**: the CLI is wired and ``--dry-run`` emits
a placeholder artifact, but the actual HTTP load generation lands as
F-A subtask 4 (see
``docs/moving forward/cross-cutting-foundation-packet.md`` §2.1).

Run::

    python scripts/realistic_load_companion.py --dry-run \
        --n-sessions 20 --duration-min 30
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
        "--n-sessions",
        type=int,
        default=20,
        help="Concurrent LifeformSession count (default 20).",
    )
    parser.add_argument(
        "--duration-min",
        type=int,
        default=30,
        help="Wall-clock minutes to sustain load (default 30).",
    )
    parser.add_argument(
        "--alpha-base-url",
        default="http://localhost:8000",
        help="Closed-alpha service base URL.",
    )
    parser.add_argument(
        "--scenario-pack",
        default="companion-bench-public-24",
        help="Scenario pack identifier (default 24 public companion-bench).",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/perf",
        help="Directory to write the JSON artifact into.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Emit a placeholder artifact without driving real load.",
    )
    return parser


def _emit_placeholder_artifact(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()
    out_path = out_dir / f"companion-{today}.json"
    payload = {
        "scaffold_status": "SHADOW",
        "vertical": "companion",
        "n_sessions": args.n_sessions,
        "duration_min": args.duration_min,
        "alpha_base_url": args.alpha_base_url,
        "scenario_pack": args.scenario_pack,
        "dry_run": True,
        "p50_turn_latency_s": None,
        "p99_turn_latency_s": None,
        "owner_snapshot_dispatch_p50_ms": None,
        "gpu_mem_peak_mb": None,
        "errors": [],
        "notes": (
            "F-A SHADOW scaffold. Real load generation lands with "
            "cross-cutting-foundation-packet §2.1 subtask 4."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.dry_run:
        sys.stderr.write(
            "F-A SHADOW: real load generation not yet wired. "
            "Re-run with --dry-run to emit placeholder artifact.\n"
        )
        return 2
    out_path = _emit_placeholder_artifact(args)
    print(f"wrote placeholder artifact: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
