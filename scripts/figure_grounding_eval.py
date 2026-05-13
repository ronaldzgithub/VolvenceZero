"""P1 #59: grounding faithfulness eval (citation actually supports claim).

Loads ``data/figure_grounding_gt/<figure_id>/assertions.jsonl``,
runs each question through the figure synthesizer with the bundle,
captures the resulting :class:`EvidencePointer` set, computes:

* evidence_faithfulness = (pointer truly supports assertion) /
  (assertions made)
* unsupported_assertion_rate = (substantive claims with no pointer
  / wrong pointer) / total

ACTIVE pass criteria (from packet §2.2):

* evidence_faithfulness >= 0.95
* unsupported_assertion_rate <= 0.05

SHADOW scaffold; depends on real Qwen substrate + retrieval index +
GroundedDecoder verify_with_pointers (Wave A debt #24 closure).

Run::

    python scripts/figure_grounding_eval.py --dry-run --figure-id einstein

Refs:
    docs/moving forward/figure-evidence-packet.md §2.2
    docs/specs/figure-grounding-gt-protocol.md
    docs/known-debts.md #59
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
        default="packages/lifeform-domain-figure/data/figure_grounding_gt",
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
    out_path = out_dir / f"grounding-eval-{args.figure_id}-{today}.json"

    gt_path = Path(args.gt_root) / args.figure_id / "assertions.jsonl"
    if not gt_path.exists():
        gt_path = Path(args.gt_root) / args.figure_id / "assertions.jsonl.example"
    assertions = _load_jsonl(gt_path)

    payload = {
        "scaffold_status": "SHADOW",
        "report_kind": "grounding_eval",
        "figure_id": args.figure_id,
        "bundle_id": args.bundle_id,
        "substrate": args.substrate,
        "gt_assertion_count": len(assertions),
        "dry_run": True,
        "evidence_faithfulness": None,
        "evidence_faithfulness_95ci": None,
        "unsupported_assertion_rate": None,
        "unsupported_assertion_rate_95ci": None,
        "sla_thresholds": {
            "evidence_faithfulness_min": 0.95,
            "unsupported_assertion_rate_max": 0.05,
        },
        "notes": (
            "P1 #59 SHADOW. Real eval requires real Qwen + GroundedDecoder + "
            "retrieval_index. Placeholder shows GT count loaded. "
            "ACTIVE pass: faithfulness >= 0.95 AND unsupported <= 0.05."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.dry_run:
        sys.stderr.write(
            "P1 #59 SHADOW: real grounding eval not wired. Use --dry-run.\n"
        )
        return 2
    out_path = _emit_placeholder_artifact(args)
    print(f"wrote SHADOW placeholder: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
