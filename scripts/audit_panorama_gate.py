"""Audit the panorama participation gate against the structural corpus.

Reports ceiling breaches (the gate opened a decision panorama into a turn
that did not want one) and floor misses (it stayed quiet when the
situation warranted structure) as two separate numbers. They are never
summed: the two failures cost very different things, and one combined
score hides exactly the asymmetry the gate is designed around.

Exit code is part of the contract. ``--max-ceiling-breaches`` /
``--max-floor-misses`` default to ``None`` (report only, always exit 0)
so the baseline run cannot be mistaken for a passing gate; CI passes
explicit budgets once a gate is expected to hold them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from volvence_zero.regime.panorama_audit import (
    DECISION_FEATURES,
    GateAuditReport,
    ablation_report,
    audit_gate,
    feature_correlations,
    v1_gate,
    v2_gate,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_REPORT_DIR = Path("research/panorama_gate")

_GATES = {
    "v1": (v1_gate, "v1-readout"),
    "v2": (v2_gate, "v2-decision-structure"),
}


def _repo_path(path: Path) -> Path:
    resolved = path if path.is_absolute() else _REPO_ROOT / path
    resolved.relative_to(_REPO_ROOT)
    return resolved


def _print_report(report: GateAuditReport) -> None:
    data = report.to_dict()
    print(f"gate: {data['gate_name']}  cases: {data['total_cases']}")
    print(f"  ceiling breaches (opened when unwanted): {data['ceiling_breaches']}")
    print(f"  floor misses    (stayed quiet when needed): {data['floor_misses']}")
    for family in data["families"]:
        print(
            f"  [{family['family']:8}] n={family['total']} "
            f"ceiling={family['ceiling_breaches']} floor={family['floor_misses']}"
        )
    print()
    for case in data["cases"]:
        marker = {"ok": "  ", "over": "!!", "under": "..."}.get(case["verdict"], "??")
        print(
            f"  {marker} {case['verdict']:6} {case['case_id']:34} "
            f"expected[{case['expected_min']},{case['expected_max']}] "
            f"actual={case['actual']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the panorama participation gate against the corpus"
    )
    parser.add_argument(
        "--gate",
        choices=sorted(_GATES) + ["all"],
        default="all",
        help="Which gate implementation to audit.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=_DEFAULT_REPORT_DIR,
        help="Directory for the JSON evidence artifacts.",
    )
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument(
        "--max-ceiling-breaches",
        type=int,
        default=None,
        help="Fail (exit 1) when a gate exceeds this many ceiling breaches.",
    )
    parser.add_argument(
        "--max-floor-misses",
        type=int,
        default=None,
        help="Fail (exit 1) when a gate exceeds this many floor misses.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help=(
            "Also run the adversarial probes: per-feature ablation and "
            "pairwise collinearity. Without these, 'the gate passes the "
            "corpus' says nothing — both were written by the same hand."
        ),
    )
    args = parser.parse_args()

    names = sorted(_GATES) if args.gate == "all" else [args.gate]
    status = 0
    for name in names:
        gate_fn, gate_label = _GATES[name]
        report = audit_gate(gate_fn, gate_name=gate_label)
        _print_report(report)
        if not args.no_write:
            out_dir = _repo_path(args.report_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"panorama_gate_audit.{name}.json"
            out_path.write_text(report.to_json() + "\n", encoding="utf-8")
            print(f"  report: {out_path.relative_to(_REPO_ROOT)}")
        if (
            args.max_ceiling_breaches is not None
            and report.over_count > args.max_ceiling_breaches
        ):
            print(
                f"  BLOCK: {gate_label} ceiling breaches "
                f"{report.over_count} > {args.max_ceiling_breaches}"
            )
            status = 1
        if (
            args.max_floor_misses is not None
            and report.under_count > args.max_floor_misses
        ):
            print(
                f"  BLOCK: {gate_label} floor misses "
                f"{report.under_count} > {args.max_floor_misses}"
            )
            status = 1
        print()

    if args.probe:
        probe: dict[str, object] = {"ablation": [], "correlations": []}
        print("adversarial probes")
        print("  ablation — pin one feature to 1.0 so the gate is blind to it;")
        print("  a feature no case notices is decoration, not a condition.")
        for feature in DECISION_FEATURES:
            report = ablation_report(feature)
            caught = [
                case.case_id for case in report.cases if case.verdict != "ok"
            ]
            print(
                f"    blind to {feature:22} ceiling={report.over_count} "
                f"floor={report.under_count} caught_by={caught or 'NOTHING'}"
            )
            probe["ablation"].append(  # type: ignore[union-attr]
                {
                    "feature": feature,
                    "ceiling_breaches": report.over_count,
                    "floor_misses": report.under_count,
                    "caught_by": caught,
                }
            )
        print("  collinearity — a geometric mean over duplicated axes is not")
        print("  a four-way conjunction, it is one axis raised to a power.")
        for left, right, value in feature_correlations():
            print(f"    {left:22} x {right:22} r={value:+.3f}")
            probe["correlations"].append(  # type: ignore[union-attr]
                {"left": left, "right": right, "pearson_r": value}
            )
        if not args.no_write:
            out_dir = _repo_path(args.report_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "panorama_gate_probes.json"
            out_path.write_text(
                json.dumps(probe, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"  report: {out_path.relative_to(_REPO_ROOT)}")
        print()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
