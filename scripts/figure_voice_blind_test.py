"""P1 #60: voice blind test (L1 perceptibility).

Generates N=30 Q→A pairs across 3 conditions (raw substrate / raw
+ bundle / raw + bundle + LoRA), shuffles them, runs an N=20 human
evaluator panel (Likert 1-5 "sounds like Einstein"), computes
Cronbach's α + per-condition mean.

ACTIVE pass criteria (from packet §2.3):

* L1+L3+L4 (bundle) vs raw: > 0.5 Likert delta (perceptible)
* L1+L2+L3+L4 (bundle+LoRA) vs L1+L3+L4: > 0.3 Likert delta
* Cronbach's α >= 0.7 (rater consistency)

SHADOW: this script is a workflow scaffold; it generates question
samples from corpus + emits a CSV the human panel fills + after
collection, computes scores. Until N=20 panel runs, ``--dry-run``
emits placeholder.

Run::

    python scripts/figure_voice_blind_test.py --dry-run --figure-id einstein

Refs:
    docs/moving forward/figure-evidence-packet.md §2.3
    docs/specs/figure-voice-blind-test-protocol.md
    docs/known-debts.md #60
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-id", default="einstein")
    parser.add_argument("--n-segments", type=int, default=30)
    parser.add_argument("--n-evaluators", type=int, default=20)
    parser.add_argument(
        "--bundle-id",
        default="figure-bundle:einstein:29eacd226a7cdfd0",
    )
    parser.add_argument(
        "--conditions",
        type=lambda s: tuple(s.split(",")),
        default=("raw", "bundle", "bundle+lora"),
    )
    parser.add_argument("--output-dir", default="artifacts/figure_eval")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _emit_placeholder_artifact(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()
    out_path = out_dir / f"voice-blind-test-{args.figure_id}-{today}.json"
    payload = {
        "scaffold_status": "SHADOW",
        "report_kind": "voice_blind_test",
        "figure_id": args.figure_id,
        "bundle_id": args.bundle_id,
        "conditions": list(args.conditions),
        "n_segments": args.n_segments,
        "n_evaluators": args.n_evaluators,
        "dry_run": True,
        "per_condition_likert_mean": None,
        "per_condition_likert_95ci": None,
        "cronbach_alpha": None,
        "raw_vs_bundle_delta": None,
        "bundle_vs_bundle_lora_delta": None,
        "sla_thresholds": {
            "raw_vs_bundle_delta_min": 0.5,
            "bundle_vs_bundle_lora_delta_min": 0.3,
            "cronbach_alpha_min": 0.7,
        },
        "notes": (
            "P1 #60 SHADOW. Workflow: (1) generate 30 Q→A across 3 conditions "
            "→ emit CSV; (2) recruit N=20 evaluators, distribute CSV; "
            "(3) collect responses, compute scores. Each evaluator: "
            "30 segments × ~30 sec × 5 sec rating = ~17.5 min × ¥100 = "
            "¥2k total budget. SLA: see thresholds above."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.dry_run:
        sys.stderr.write(
            "P1 #60 SHADOW: voice blind test workflow not wired. Use --dry-run.\n"
        )
        return 2
    out_path = _emit_placeholder_artifact(args)
    print(f"wrote SHADOW placeholder: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
