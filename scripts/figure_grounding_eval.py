"""P1 #59: grounding faithfulness eval (citation actually supports claim).

Loads ``data/figure_grounding_gt/<figure_id>/assertions.jsonl``,
runs each assertion through a grounding judge (real GroundedDecoder
in ACTIVE; ``fake-judge`` for dev / contract tests), and computes:

* evidence_faithfulness = (assertion has supporting pointer) /
  (assertions evaluated)
* unsupported_assertion_rate = (substantive claims with 0 pointer
  / wrong pointer) / total

ACTIVE pass criteria (from packet §2.2):

* evidence_faithfulness >= 0.95
* unsupported_assertion_rate <= 0.05

Modes:

* ``--mode dry-run`` — placeholder, no judge call
* ``--mode fake-judge`` — deterministic ``judge_callable`` injection
* ``--mode active`` — real GroundedDecoder + retrieval index
  (requires Qwen substrate; raises until #41 PEFT bake lands)

Refs:
    docs/moving forward/figure-evidence-packet.md §2.2
    docs/specs/figure-grounding-gt-protocol.md
    docs/known-debts.md #59
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from pathlib import Path
from typing import Callable


def fake_judge_always_supported(assertion: dict) -> tuple[bool, str]:
    """Returns (supported=True, "fake-pointer:always-supported") for every probe."""
    del assertion
    return True, "fake-pointer:always-supported"


def fake_judge_always_unsupported(assertion: dict) -> tuple[bool, str]:
    del assertion
    return False, ""


def fake_judge_perfect_oracle(assertion: dict) -> tuple[bool, str]:
    """Mirrors the ground-truth label so the report shows a perfect run.

    GT entries set ``_gt_supportable=True`` when the corpus has
    evidence; this judge mirrors the field so the resulting report
    has evidence_faithfulness = 1.0 / unsupported_assertion_rate = 0.0.
    """
    is_supported = bool(assertion.get("_gt_supportable", True))
    pointer = (
        f"fake-pointer:{assertion.get('chunk_id', 'unknown')}"
        if is_supported
        else ""
    )
    return is_supported, pointer


_FAKE_JUDGES: dict[str, Callable[[dict], tuple[bool, str]]] = {
    "always-supported": fake_judge_always_supported,
    "always-unsupported": fake_judge_always_unsupported,
    "perfect-oracle": fake_judge_perfect_oracle,
}


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
    parser.add_argument(
        "--mode",
        default="dry-run",
        choices=("dry-run", "fake-judge", "active"),
    )
    parser.add_argument(
        "--fake-judge",
        default="perfect-oracle",
        choices=tuple(_FAKE_JUDGES),
    )
    parser.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)
    return parser


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_gt(figure_id: str, gt_root: str) -> list[dict]:
    base = Path(gt_root) / figure_id
    gt_path = base / "assertions.jsonl"
    if not gt_path.exists():
        gt_path = base / "assertions.jsonl.example"
    assertions = _load_jsonl(gt_path)
    for a in assertions:
        a.setdefault("_gt_supportable", True)
    return assertions


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _emit_dry_run_artifact(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()
    out_path = out_dir / f"grounding-eval-{args.figure_id}-{today}.json"
    assertions = _load_gt(args.figure_id, args.gt_root)
    payload = {
        "scaffold_status": "SHADOW",
        "report_kind": "grounding_eval",
        "report_mode": "dry-run",
        "figure_id": args.figure_id,
        "bundle_id": args.bundle_id,
        "substrate": args.substrate,
        "gt_assertion_count": len(assertions),
        "evidence_faithfulness": None,
        "evidence_faithfulness_95ci": None,
        "unsupported_assertion_rate": None,
        "unsupported_assertion_rate_95ci": None,
        "sla_thresholds": {
            "evidence_faithfulness_min": 0.95,
            "unsupported_assertion_rate_max": 0.05,
        },
        "notes": (
            "P1 #59 SHADOW dry-run. Real eval requires real Qwen + "
            "GroundedDecoder + retrieval_index. Run with --mode fake-judge "
            "for the deterministic injection contract path."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def run_grounding_eval(
    *,
    figure_id: str,
    gt_root: str,
    bundle_id: str,
    substrate: str,
    judge: Callable[[dict], tuple[bool, str]],
    judge_label: str,
) -> dict:
    """Run the grounding eval pipeline against a ``judge_callable`` (debt #59)."""

    assertions = _load_gt(figure_id, gt_root)
    n = len(assertions)
    supported_n = 0
    pointers: list[str] = []
    for assertion in assertions:
        ok, pointer = judge(assertion)
        if ok:
            supported_n += 1
        pointers.append(pointer)
    unsupported_n = n - supported_n
    faithfulness = supported_n / n if n else 0.0
    unsupported_rate = unsupported_n / n if n else 0.0
    ef_lo, ef_hi = _wilson_ci(supported_n, n)
    un_lo, un_hi = _wilson_ci(unsupported_n, n)
    return {
        "scaffold_status": "SHADOW",
        "report_kind": "grounding_eval",
        "report_mode": "fake-judge",
        "judge_label": judge_label,
        "figure_id": figure_id,
        "bundle_id": bundle_id,
        "substrate": substrate,
        "gt_assertion_count": n,
        "supported_assertion_count": supported_n,
        "unsupported_assertion_count": unsupported_n,
        "evidence_faithfulness": faithfulness,
        "evidence_faithfulness_95ci": [ef_lo, ef_hi],
        "unsupported_assertion_rate": unsupported_rate,
        "unsupported_assertion_rate_95ci": [un_lo, un_hi],
        "sla_thresholds": {
            "evidence_faithfulness_min": 0.95,
            "unsupported_assertion_rate_max": 0.05,
        },
        "sla_pass": faithfulness >= 0.95 and unsupported_rate <= 0.05,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.dry_run:
        args.mode = "dry-run"
    if args.mode == "active":
        sys.stderr.write(
            "P1 #59 ACTIVE: real grounding eval not wired (depends on "
            "GroundedDecoder + retrieval_index + Qwen). Use --mode "
            "dry-run or --mode fake-judge.\n"
        )
        return 2
    if args.mode == "dry-run":
        out_path = _emit_dry_run_artifact(args)
        print(f"wrote SHADOW dry-run placeholder: {out_path}")
        return 0
    judge_callable = _FAKE_JUDGES[args.fake_judge]
    report = run_grounding_eval(
        figure_id=args.figure_id,
        gt_root=args.gt_root,
        bundle_id=args.bundle_id,
        substrate=args.substrate,
        judge=judge_callable,
        judge_label=f"fake-judge:{args.fake_judge}",
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()
    out_path = out_dir / f"grounding-eval-{args.figure_id}-{today}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote fake-judge eval artifact: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
