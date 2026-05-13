"""P5 #57: trusted-runner mode for held-out scenario submissions.

Submission modes (per ``docs/external/companion-bench-trusted-runner-protocol.md``):

1. **self-hosted**: submitter runs scenarios locally; only
   ``public-24`` are accessible. Result lands on public leaderboard.
2. **trusted-runner**: VZ runs scenarios for the submitter on
   held-out scenarios. Submitter provides:
   * OpenAI-compat ``base_url`` + ``api_key`` (encrypted at rest)
   * model card + system prompt
   VZ runs, computes scores, deletes the raw transcripts, returns
   only the verdict / per-axis scores. Result lands on the held-out
   leaderboard.

This script is the trusted-runner driver. Skeleton only; real
encryption / queue / scheduling / billing land in #57 ACTIVE.

Run::

    python scripts/companion_bench/trusted_runner.py --dry-run \
        --submission-id sub-001 --mode trusted

Refs:
    docs/moving forward/companion-bench-public-launch-packet.md §2.7
    docs/external/companion-bench-trusted-runner-protocol.md
    docs/external/companion-bench-heldout-leak-protocol.md
    docs/known-debts.md #57
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-id", required=True)
    parser.add_argument(
        "--mode",
        choices=("self-hosted", "trusted"),
        required=True,
    )
    parser.add_argument(
        "--manifest",
        help="Path to submission manifest YAML (for self-hosted) or "
        "encrypted credential bundle (for trusted).",
    )
    parser.add_argument("--scenario-pack", default="public-24")
    parser.add_argument("--output-dir", default="artifacts/companion_bench")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _emit_placeholder_artifact(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()
    out_path = out_dir / f"trusted_runner-{args.submission_id}-{today}.json"
    payload = {
        "scaffold_status": "SHADOW",
        "submission_id": args.submission_id,
        "mode": args.mode,
        "scenario_pack": args.scenario_pack,
        "dry_run": True,
        "verdict": None,
        "per_axis_scores": None,
        "estimated_cost_usd": None,
        "transcripts_deleted_per_protocol": None,
        "notes": (
            "P5 #57 SHADOW. trusted mode requires encrypted credential "
            "handling + queue + per-submission billing — all land in "
            "#57 ACTIVE. self-hosted mode delegates to existing "
            "scripts/companion_bench/run_real_submission.py."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.dry_run:
        sys.stderr.write(
            "P5 #57 SHADOW: trusted-runner mode not yet wired. "
            "Use --dry-run for placeholder.\n"
        )
        return 2
    out_path = _emit_placeholder_artifact(args)
    print(f"wrote SHADOW placeholder: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
