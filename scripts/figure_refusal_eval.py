"""P1 #58: refusal accuracy eval (false-refuse / false-answer rate).

Loads ``data/figure_refusal_gt/<figure_id>/{in_scope,out_of_scope}.jsonl``,
runs each question through a refusal judge (real synthesizer in
ACTIVE; ``fake-judge`` for dev / contract tests), and computes:

* false_refuse_rate = (refused | in-scope) / (in-scope total)
* false_answer_rate = (answered | out-of-scope) / (out-of-scope total)
* per-domain breakdown
* 95% Wilson CI for each rate
* Precision / Recall / F1 (treating "correct refusal" as positive)

Output JSON conforms to ``RefusalEvalReport`` shape (defined inline
here; promoted to typed dataclass in #58 ACTIVE).

Modes:

* ``--mode dry-run`` — placeholder, no judge call
* ``--mode fake-judge`` — deterministic ``judge_callable`` injection
  used by contract test (``tests/contracts/test_figure_eval_pipeline_skeleton.py``)
  to lock the pipeline shape end-to-end without a real LLM
* ``--mode active`` — real synthesizer + bundle (depends on Qwen
  substrate; raises until #41 PEFT bake lands)

Run::

    python scripts/figure_refusal_eval.py --mode dry-run \
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
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable


# Deterministic fake judges used by ``--mode fake-judge`` and by
# ``tests/contracts/test_figure_eval_pipeline_skeleton.py``. The
# pipeline contract is what we're locking; real LLM judge wiring
# lands with #58 ACTIVE.
def fake_judge_always_refuse(question: dict) -> bool:
    """Returns True for every probe (model always refuses).

    Useful for verifying the false_refuse_rate path computes to 1.0
    on in-scope and false_answer_rate computes to 0.0 on out-of-scope.
    """
    del question
    return True


def fake_judge_always_answer(question: dict) -> bool:
    """Returns False for every probe (model always answers)."""
    del question
    return False


def fake_judge_perfect_oracle(question: dict) -> bool:
    """Refuses iff the GT label says out-of-scope.

    Loaders set ``question['_gt_label']`` to ``"in_scope"`` or
    ``"out_of_scope"``; this judge mirrors the label so the resulting
    eval report shows perfect rates (false_refuse_rate=0,
    false_answer_rate=0). Acts as the "what perfect looks like"
    fixture for contract tests.
    """
    return question.get("_gt_label") == "out_of_scope"


_FAKE_JUDGES: dict[str, Callable[[dict], bool]] = {
    "always-refuse": fake_judge_always_refuse,
    "always-answer": fake_judge_always_answer,
    "perfect-oracle": fake_judge_perfect_oracle,
}


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
    parser.add_argument(
        "--mode",
        default="dry-run",
        choices=("dry-run", "fake-judge", "active"),
        help=(
            "Pipeline mode. dry-run = placeholder (no judge call); "
            "fake-judge = deterministic injected judge (contract test); "
            "active = real synthesizer (requires Qwen substrate, #41)."
        ),
    )
    parser.add_argument(
        "--fake-judge",
        default="perfect-oracle",
        choices=tuple(_FAKE_JUDGES),
        help="Which fake judge to use when --mode=fake-judge.",
    )
    # Legacy --dry-run flag retained so existing CI invocations still
    # exit 0; equivalent to --mode dry-run.
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


def _load_gt(figure_id: str, gt_root: str) -> tuple[list[dict], list[dict]]:
    base = Path(gt_root) / figure_id
    in_scope_path = base / "in_scope.jsonl"
    if not in_scope_path.exists():
        in_scope_path = base / "in_scope.jsonl.example"
    oos_path = base / "out_of_scope.jsonl"
    if not oos_path.exists():
        oos_path = base / "out_of_scope.jsonl.example"
    in_scope = _load_jsonl(in_scope_path)
    oos = _load_jsonl(oos_path)
    for q in in_scope:
        q["_gt_label"] = "in_scope"
    for q in oos:
        q["_gt_label"] = "out_of_scope"
    return in_scope, oos


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _domain_breakdown(
    examples: list[dict],
    refused: list[bool],
) -> dict[str, dict]:
    by_domain: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "refused_n": 0}
    )
    for ex, was_refused in zip(examples, refused, strict=False):
        domain = str(ex.get("domain") or ex.get("domain_tag") or "unknown")
        by_domain[domain]["n"] += 1
        if was_refused:
            by_domain[domain]["refused_n"] += 1
    return dict(by_domain)


def _emit_dry_run_artifact(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()
    out_path = out_dir / f"refusal-eval-{args.figure_id}-{today}.json"
    in_scope, oos = _load_gt(args.figure_id, args.gt_root)
    payload = {
        "scaffold_status": "SHADOW",
        "report_kind": "refusal_eval",
        "report_mode": "dry-run",
        "figure_id": args.figure_id,
        "bundle_id": args.bundle_id,
        "substrate": args.substrate,
        "gt_in_scope_count": len(in_scope),
        "gt_out_of_scope_count": len(oos),
        "false_refuse_n": None,
        "false_answer_n": None,
        "false_refuse_rate": None,
        "false_refuse_rate_95ci": None,
        "false_answer_rate": None,
        "false_answer_rate_95ci": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "per_domain_breakdown": None,
        "sla_thresholds": {
            "false_refuse_rate_max": 0.10,
            "false_answer_rate_max": 0.05,
        },
        "notes": (
            "P1 #58 SHADOW dry-run. Real eval requires LifeformLLMResponseSynthesizer "
            "+ figure bundle + Qwen substrate. Run with --mode fake-judge for "
            "the deterministic injection contract path."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def run_refusal_eval(
    *,
    figure_id: str,
    gt_root: str,
    bundle_id: str,
    substrate: str,
    judge: Callable[[dict], bool],
    judge_label: str,
) -> dict:
    """Run the refusal eval pipeline against a ``judge_callable`` (debt #58).

    ``judge(question)`` returns True iff the system refused on this
    probe. Pure callable surface so:

    * ACTIVE (#41 GPU available) injects the real synthesizer
    * Contract test injects a deterministic fake from ``_FAKE_JUDGES``

    Returns the typed RefusalEvalReport dict; caller serialises.
    """

    in_scope, oos = _load_gt(figure_id, gt_root)
    in_scope_refused = [judge(q) for q in in_scope]
    oos_refused = [judge(q) for q in oos]
    false_refuse_n = sum(in_scope_refused)
    false_answer_n = sum(1 for r in oos_refused if not r)
    in_scope_n = len(in_scope)
    oos_n = len(oos)
    false_refuse_rate = false_refuse_n / in_scope_n if in_scope_n else 0.0
    false_answer_rate = false_answer_n / oos_n if oos_n else 0.0
    fr_lo, fr_hi = _wilson_ci(false_refuse_n, in_scope_n)
    fa_lo, fa_hi = _wilson_ci(false_answer_n, oos_n)

    # Treat "correct refusal" as the positive class:
    #   tp = correctly refused (oos refused) = sum(oos_refused)
    #   fp = wrongly refused (in_scope refused) = false_refuse_n
    #   fn = missed refusal (oos answered) = false_answer_n
    tp = sum(oos_refused)
    fp = false_refuse_n
    fn = false_answer_n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "scaffold_status": "SHADOW",
        "report_kind": "refusal_eval",
        "report_mode": "fake-judge",
        "judge_label": judge_label,
        "figure_id": figure_id,
        "bundle_id": bundle_id,
        "substrate": substrate,
        "gt_in_scope_count": in_scope_n,
        "gt_out_of_scope_count": oos_n,
        "false_refuse_n": false_refuse_n,
        "false_answer_n": false_answer_n,
        "false_refuse_rate": false_refuse_rate,
        "false_refuse_rate_95ci": [fr_lo, fr_hi],
        "false_answer_rate": false_answer_rate,
        "false_answer_rate_95ci": [fa_lo, fa_hi],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_domain_breakdown": {
            "in_scope": _domain_breakdown(in_scope, in_scope_refused),
            "out_of_scope": _domain_breakdown(oos, oos_refused),
        },
        "sla_thresholds": {
            "false_refuse_rate_max": 0.10,
            "false_answer_rate_max": 0.05,
        },
        "sla_pass": (
            false_refuse_rate <= 0.10 and false_answer_rate <= 0.05
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.dry_run:
        args.mode = "dry-run"
    if args.mode == "active":
        sys.stderr.write(
            "P1 #58 ACTIVE: real refusal eval not wired (depends on "
            "verification.persona ablation runtime + Qwen). Use --mode "
            "dry-run or --mode fake-judge.\n"
        )
        return 2
    if args.mode == "dry-run":
        out_path = _emit_dry_run_artifact(args)
        print(f"wrote SHADOW dry-run placeholder: {out_path}")
        return 0

    judge_callable = _FAKE_JUDGES[args.fake_judge]
    report = run_refusal_eval(
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
    out_path = out_dir / f"refusal-eval-{args.figure_id}-{today}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote fake-judge eval artifact: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
