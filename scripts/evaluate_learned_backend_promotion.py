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
    ValidationDeltaV2Readout,
    evaluate_learned_active_chain,
)


def _v2_readout_from_payload(
    payload: dict[str, object] | None,
) -> ValidationDeltaV2Readout | None:
    """Rebuild the frozen v2 readout from an artifact's JSON block."""

    if payload is None:
        return None
    return ValidationDeltaV2Readout(
        window_filled=bool(payload["window_filled"]),
        informative_axes=tuple(str(axis) for axis in payload["informative_axes"]),
        excluded_axes=tuple(str(axis) for axis in payload["excluded_axes"]),
        per_axis_relative_improvement=tuple(
            (str(axis), float(value))
            for axis, value in payload["per_axis_relative_improvement"]
        ),
        min_relative_improvement=float(payload["min_relative_improvement"]),
        blocking_reasons=tuple(
            str(reason) for reason in payload["blocking_reasons"]
        ),
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


def _evidence_from_payload(
    component: str,
    payload: dict[str, object],
    *,
    validation_delta_v2: ValidationDeltaV2Readout | None = None,
) -> LearnedActiveEvidence:
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
        validation_delta_v2=validation_delta_v2,
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


def _active_components_from_gate(
    gate: dict[str, object],
    evidence_rows: list[LearnedActiveEvidence],
) -> tuple[str, ...]:
    declared = gate.get("active_components")
    if declared is not None:
        if not isinstance(declared, list):
            raise SystemExit("learned_active_gate.active_components must be a list")
        return tuple(str(component) for component in declared)

    runtime_bits = {evidence.prior_runtime_active for evidence in evidence_rows}
    ssl_bits = {evidence.prior_ssl_active for evidence in evidence_rows}
    if len(runtime_bits) != 1 or len(ssl_bits) != 1:
        raise SystemExit(
            "legacy prior-active fields disagree across learned backend evidence rows"
        )
    runtime_active = runtime_bits.pop()
    ssl_active = ssl_bits.pop()
    if ssl_active and not runtime_active:
        raise SystemExit("legacy prior_ssl_active requires prior_runtime_active")
    active: list[str] = []
    if runtime_active:
        active.append(LearnedBackendComponent.TEMPORAL_RUNTIME.value)
    if ssl_active:
        active.append(LearnedBackendComponent.TEMPORAL_SSL.value)
    return tuple(active)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    gate = artifact.get("learned_active_gate")
    if not isinstance(gate, dict):
        raise SystemExit("artifact missing learned_active_gate object")
    rows = gate.get("evidence") or gate.get("verdicts")
    if not isinstance(rows, list):
        raise SystemExit("learned_active_gate must contain evidence or verdicts list")

    # Observation-window versioning (threshold pre-registration): the
    # artifact declares which validation gate its window was opened under.
    # Pre-v2 artifacts carry no marker and are judged under v1 forever.
    gate_version = str(gate.get("validation_gate_version", "v1"))
    v2_readout = _v2_readout_from_payload(gate.get("validation_delta_v2"))
    if gate_version == "v2" and v2_readout is None:
        raise SystemExit(
            "artifact declares validation_gate_version=v2 but lacks the "
            "validation_delta_v2 readout block"
        )

    evidence_rows: list[LearnedActiveEvidence] = []
    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit("learned_active_gate rows must be objects")
        component = str(row["component"])
        evidence_payload = row if "real_trace_turns" in row else _candidate_payload_from_soak(row)
        evidence_rows.append(
            _evidence_from_payload(
                component, evidence_payload, validation_delta_v2=v2_readout
            )
        )

    active_components = _active_components_from_gate(gate, evidence_rows)
    chain = evaluate_learned_active_chain(
        evidence_rows,
        active_components=active_components,
        validation_gate_version=gate_version,
    )

    reports: list[dict[str, object]] = []
    for verdict in chain.terminal_reports:
        if verdict.component in chain.active_components:
            stage_status = "active"
        elif verdict.component is chain.next_component:
            stage_status = (
                "next_candidate"
                if chain.next_component_eligible
                else "next_candidate_blocked"
            )
        else:
            stage_status = "queued_after_predecessor"
        reports.append(
            {
                "component": verdict.component.value,
                "eligible": verdict.eligible,
                "validation_gate_version": verdict.validation_gate_version,
                "missing_gates": list(verdict.missing_gates),
                "description": verdict.description,
                "stage_status": stage_status,
                "recommended_env": (
                    f"VZ_{verdict.component.value.upper()}=active"
                    if (
                        verdict.component is chain.next_component
                        and chain.next_component_eligible
                    )
                    else ""
                ),
            }
        )

    payload = {
        "schema_version": "learned-backend-promotion-report.v2",
        "source_artifact": str(args.artifact),
        "validation_gate_version": gate_version,
        "reports": reports,
        # Backward-compatible name: all four components pass the isolated
        # terminal-candidate gate. It does not authorize a four-field flip.
        "all_eligible": chain.terminal_candidate_ready,
        "terminal_candidate_ready": chain.terminal_candidate_ready,
        "production_terminal_ready": chain.production_terminal_ready,
        "staged_gate": {
            "active_components": [
                component.value for component in chain.active_components
            ],
            "next_component": (
                chain.next_component.value
                if chain.next_component is not None
                else None
            ),
            "next_component_eligible": chain.next_component_eligible,
            "blocking_reasons": list(chain.blocking_reasons),
            "description": chain.description,
        },
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
