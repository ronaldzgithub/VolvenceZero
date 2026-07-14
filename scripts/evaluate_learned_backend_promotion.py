#!/usr/bin/env python3
"""Evaluate learned backend ACTIVE candidacy from an evidence JSON artifact.

This script never flips defaults. It is the operator-facing wrapper around
``volvence_zero.agent.learned_active_gate``: read an evidence artifact, emit a
promotion report, and fail loudly when the artifact lacks the required gate
payload.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from volvence_zero.agent.learned_active_gate import (
    LearnedActiveEvidence,
    LearnedBackendComponent,
    evaluate_learned_active_candidate,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
        help="Evidence artifact JSON containing learned_active_gate.verdicts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/learned_backend_promotion/report.json"),
    )
    return parser


def _evidence_from_payload(component: str, payload: dict[str, object]) -> LearnedActiveEvidence:
    return LearnedActiveEvidence(
        component=LearnedBackendComponent(component),
        real_trace_turns=int(payload["real_trace_turns"]),
        validation_delta=float(payload["validation_delta"]),
        strict_eta_gate_passed=bool(payload["strict_eta_gate_passed"]),
        pe_off_control_direction_correct=bool(payload["pe_off_control_direction_correct"]),
        eta_off_control_direction_correct=bool(payload["eta_off_control_direction_correct"]),
        rollback_drill_passed=bool(payload["rollback_drill_passed"]),
        latency_slo_ok=bool(payload["latency_slo_ok"]),
        safety_gate_ok=bool(payload["safety_gate_ok"]),
        prior_runtime_active=bool(payload.get("prior_runtime_active", False)),
        prior_ssl_active=bool(payload.get("prior_ssl_active", False)),
        internal_rl_no_reward_leakage=bool(
            payload.get("internal_rl_no_reward_leakage", True)
        ),
        cms_retention_non_degrading=bool(
            payload.get("cms_retention_non_degrading", True)
        ),
        cms_absorption_improved=bool(payload.get("cms_absorption_improved", True)),
    )


def _candidate_payload_from_soak(verdict: dict[str, object]) -> dict[str, object]:
    """Convert legacy soak verdict rows into full gate evidence rows.

    Synthetic soak rows intentionally lack real-trace promotion evidence. We
    preserve that honestly by filling the missing promotion-only gates with
    conservative ``False``/``0`` values, yielding BLOCKED verdicts.
    """

    return {
        "real_trace_turns": 0,
        "validation_delta": 0.0,
        "strict_eta_gate_passed": False,
        "pe_off_control_direction_correct": False,
        "eta_off_control_direction_correct": False,
        "rollback_drill_passed": False,
        "latency_slo_ok": bool(verdict.get("latency_slo_ok", False)),
        "safety_gate_ok": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    gate = artifact.get("learned_active_gate")
    if not isinstance(gate, dict):
        raise SystemExit("artifact missing learned_active_gate object")
    rows = gate.get("evidence") or gate.get("verdicts")
    if not isinstance(rows, list):
        raise SystemExit("learned_active_gate must contain evidence or verdicts list")

    reports: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit("learned_active_gate rows must be objects")
        component = str(row["component"])
        evidence_payload = row if "real_trace_turns" in row else _candidate_payload_from_soak(row)
        evidence = _evidence_from_payload(component, evidence_payload)
        verdict = evaluate_learned_active_candidate(evidence)
        reports.append(
            {
                "component": verdict.component.value,
                "eligible": verdict.eligible,
                "missing_gates": list(verdict.missing_gates),
                "description": verdict.description,
                "recommended_env": (
                    f"VZ_{verdict.component.value.upper()}=active"
                    if verdict.eligible
                    else ""
                ),
            }
        )

    payload = {
        "schema_version": "learned-backend-promotion-report.v1",
        "source_artifact": str(args.artifact),
        "reports": reports,
        "all_eligible": all(report["eligible"] for report in reports),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote promotion report: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
