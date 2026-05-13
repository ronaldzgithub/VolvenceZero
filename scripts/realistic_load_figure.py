"""F-A: realistic concurrent load generator for the figure vertical.

Drives N=10 ``ai_id`` × M=50 turn against the DLaaS service
(``/dlaas/adopt`` + ``/dlaas/interactions``) using mixed in-corpus +
out-of-scope (OOS) Einstein questions. Captures L3 citation rate +
L4 refusal rate so any latency tuning that *also* degrades L3 / L4
SLA is caught here, not in production.

SHADOW scaffold: CLI + ``--dry-run`` placeholder; real load wiring
lands as F-A subtask 4 (see
``docs/moving forward/cross-cutting-foundation-packet.md`` §2.1).

Run::

    python scripts/realistic_load_figure.py --dry-run \
        --n-ai-id 10 --turns-per-id 50
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-ai-id", type=int, default=10)
    parser.add_argument("--turns-per-id", type=int, default=50)
    parser.add_argument(
        "--figure-id",
        default="einstein",
        help="Figure bundle id (default einstein).",
    )
    parser.add_argument(
        "--dlaas-base-url",
        default="http://localhost:8000",
    )
    parser.add_argument(
        "--in-corpus-ratio",
        type=float,
        default=0.6,
        help="Fraction of turns sampled from in-corpus questions; rest OOS.",
    )
    parser.add_argument("--output-dir", default="artifacts/perf")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _emit_placeholder_artifact(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()
    out_path = out_dir / f"figure-{today}.json"
    payload = {
        "scaffold_status": "SHADOW",
        "vertical": "figure",
        "figure_id": args.figure_id,
        "n_ai_id": args.n_ai_id,
        "turns_per_id": args.turns_per_id,
        "in_corpus_ratio": args.in_corpus_ratio,
        "dry_run": True,
        "p50_turn_latency_s": None,
        "p99_turn_latency_s": None,
        "l3_citation_rate": None,
        "l4_refusal_rate": None,
        "lora_swap_overhead_p50_ms": None,
        "errors": [],
        "notes": (
            "F-A SHADOW scaffold. Real DLaaS load lands with "
            "cross-cutting-foundation-packet §2.1 subtask 4. "
            "L3/L4 rates are tracked here so latency tuning cannot "
            "silently regress them."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.dry_run:
        sys.stderr.write(
            "F-A SHADOW: real DLaaS load not yet wired. Use --dry-run.\n"
        )
        return 2
    out_path = _emit_placeholder_artifact(args)
    print(f"wrote placeholder artifact: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
