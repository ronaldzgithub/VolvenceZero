"""P1 #58: refusal accuracy eval (false-refuse / false-answer rate).

Loads ``data/figure_refusal_gt/<figure_id>/{in_scope,out_of_scope}.jsonl``,
runs each question through the figure synthesizer (with the bundle
attached), records whether the system refused or answered, computes:

* false_refuse_rate = (refused | in-scope) / (in-scope total)
* false_answer_rate = (answered | out-of-scope) / (out-of-scope total)
* per-domain breakdown
* 95% Wilson CI for each rate

Output JSON conforms to ``RefusalEvalReport`` shape (defined inline
here; promoted to typed dataclass in #58 ACTIVE).

SHADOW scaffold: real synthesizer wire-up depends on
``LifeformLLMResponseSynthesizer`` + verification.persona ablation
runtime. Until ACTIVE, ``--dry-run`` emits placeholder.

Run::

    python scripts/figure_refusal_eval.py --dry-run \
        --figure-id einstein

Refs:
    docs/moving forward/figure-evidence-packet.md §2.1
    docs/specs/figure-refusal-gt-protocol.md
    docs/known-debts.md #58
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
    parser.add_argument(
        "--gt-root",
        default="packages/lifeform-domain-figure/data/figure_refusal_gt",
    )
    parser.add_argument(
        "--bundle-id",
        default="figure-bundle:einstein:29eacd226a7cdfd0",
    )
    parser.add_argument(
        "--substrate",
        default="synthetic",
        choices=("synthetic", "qwen-1.5b", "llama-3.1-8b"),
    )
    parser.add_argument("--output-dir", default="artifacts/figure_eval")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _emit_placeholder_artifact(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()
    out_path = out_dir / f"refusal-eval-{args.figure_id}-{today}.json"

    in_scope_path = Path(args.gt_root) / args.figure_id / "in_scope.jsonl"
    if not in_scope_path.exists():
        in_scope_path = Path(args.gt_root) / args.figure_id / "in_scope.jsonl.example"
    oos_path = Path(args.gt_root) / args.figure_id / "out_of_scope.jsonl"
    if not oos_path.exists():
        oos_path = Path(args.gt_root) / args.figure_id / "out_of_scope.jsonl.example"

    in_scope = _load_jsonl(in_scope_path)
    oos = _load_jsonl(oos_path)

    payload = {
        "scaffold_status": "SHADOW",
        "report_kind": "refusal_eval",
        "figure_id": args.figure_id,
        "bundle_id": args.bundle_id,
        "substrate": args.substrate,
        "gt_in_scope_count": len(in_scope),
        "gt_out_of_scope_count": len(oos),
        "dry_run": True,
        "false_refuse_rate": None,
        "false_refuse_rate_95ci": None,
        "false_answer_rate": None,
        "false_answer_rate_95ci": None,
        "per_domain_breakdown": None,
        "sla_thresholds": {
            "false_refuse_rate_max": 0.10,
            "false_answer_rate_max": 0.05,
        },
        "notes": (
            "P1 #58 SHADOW. Real eval requires LifeformLLMResponseSynthesizer "
            "+ figure bundle + Qwen substrate. Placeholder shows GT counts "
            "loaded from .example files. ACTIVE pass criteria: "
            "false_refuse_rate <= 0.10 AND false_answer_rate <= 0.05."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.dry_run:
        sys.stderr.write(
            "P1 #58 SHADOW: real refusal eval not wired (depends on "
            "verification.persona ablation runtime + Qwen). Use --dry-run.\n"
        )
        return 2
    out_path = _emit_placeholder_artifact(args)
    print(f"wrote SHADOW placeholder: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
