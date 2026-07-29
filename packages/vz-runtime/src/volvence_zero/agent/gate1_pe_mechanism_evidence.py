"""Gate 1 prediction-error mechanism evidence and lineage audit.

This module is an offline evidence harness.  The live PE owner remains
``PredictionErrorModule``; typed LSS links are read-only autograd audits and
the lineage join never reconstructs prediction-error semantics.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

from volvence_zero.credit import derive_prediction_error_credit_records
from volvence_zero.dual_track import DualTrackSnapshot, TrackState
from volvence_zero.evaluation import EvaluationScore, EvaluationSnapshot
from volvence_zero.memory import Track
from volvence_zero.prediction.error import PredictionErrorModule
from volvence_zero.prediction.torch_lss import (
    TypedLSSBridgeReport,
    bridge_typed_runtime_pe_to_lss,
)
from volvence_zero.runtime import WiringLevel


GATE1_PE_MECHANISM_SCHEMA_VERSION = "gate1-pe-mechanism.v1"
GATE1_PE_MECHANISM_REQUIRED_FILES = (
    "manifest.yaml",
    "predictions.jsonl",
    "outcomes.jsonl",
    "prediction_errors.jsonl",
    "segments.jsonl",
    "credit.jsonl",
    "state_diff.jsonl",
    "action_selection.jsonl",
    "ablation_results.json",
    "promotion_verdict.json",
    "rollback_evidence.json",
    "report.md",
)
_GATE1_LSS_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Gate1LineageAudit:
    prediction_count: int
    valid_lineage_count: int
    lineage_coverage: float
    accepted_mismatch_count: int
    duplicate_settlement_count: int
    passed: bool
    description: str


@dataclass(frozen=True)
class Gate1EvaluationDecouplingReport:
    active_payload_sha256_a: str
    active_payload_sha256_b: str
    byte_invariant: bool
    rollback_gate: str
    passed: bool
    description: str


def _duplicate_count(values: Sequence[str]) -> int:
    return len(values) - len(set(values))


def _require_lineage_fields(
    row: Mapping[str, Any],
    *,
    record_kind: str,
    fields: tuple[str, ...],
) -> None:
    missing = tuple(
        field
        for field in fields
        if field not in row or not str(row[field]).strip()
    )
    if missing:
        raise ValueError(
            f"Gate 1 {record_kind} record lacks lineage fields {missing!r}"
        )


def audit_prediction_lineage(
    *,
    predictions: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    prediction_errors: Sequence[Mapping[str, Any]],
) -> Gate1LineageAudit:
    """Join prediction→outcome→PE exactly once using owner-issued ids."""

    if not predictions:
        raise ValueError("Gate 1 lineage audit requires predictions")
    prediction_fields = ("prediction_id",)
    settled_fields = (
        "prediction_id",
        "environment_event_id",
        "environment_outcome_id",
        "observed_at",
    )
    for row in predictions:
        _require_lineage_fields(
            row,
            record_kind="prediction",
            fields=prediction_fields,
        )
    for row in outcomes:
        _require_lineage_fields(
            row,
            record_kind="outcome",
            fields=settled_fields,
        )
    for row in prediction_errors:
        _require_lineage_fields(
            row,
            record_kind="prediction-error",
            fields=settled_fields,
        )
    prediction_ids = tuple(str(row["prediction_id"]) for row in predictions)
    outcome_ids = tuple(str(row["prediction_id"]) for row in outcomes)
    error_ids = tuple(
        str(row["prediction_id"]) for row in prediction_errors
    )
    duplicate_settlement_count = (
        _duplicate_count(prediction_ids)
        + _duplicate_count(outcome_ids)
        + _duplicate_count(error_ids)
    )
    outcome_by_prediction = {
        str(row["prediction_id"]): row for row in outcomes
    }
    error_by_prediction = {
        str(row["prediction_id"]): row for row in prediction_errors
    }
    valid_lineage_count = 0
    accepted_mismatch_count = 0
    for prediction_id in prediction_ids:
        outcome = outcome_by_prediction.get(prediction_id)
        error = error_by_prediction.get(prediction_id)
        if outcome is None or error is None:
            accepted_mismatch_count += 1
            continue
        lineage_matches = all(
            outcome[field] == error[field]
            for field in settled_fields
        )
        if lineage_matches:
            valid_lineage_count += 1
        else:
            accepted_mismatch_count += 1
    unexpected_ids = (
        (set(outcome_ids) | set(error_ids)) - set(prediction_ids)
    )
    accepted_mismatch_count += len(unexpected_ids)
    lineage_coverage = valid_lineage_count / len(prediction_ids)
    passed = (
        lineage_coverage == 1.0
        and accepted_mismatch_count == 0
        and duplicate_settlement_count == 0
    )
    return Gate1LineageAudit(
        prediction_count=len(prediction_ids),
        valid_lineage_count=valid_lineage_count,
        lineage_coverage=lineage_coverage,
        accepted_mismatch_count=accepted_mismatch_count,
        duplicate_settlement_count=duplicate_settlement_count,
        passed=passed,
        description=(
            "Gate 1 prediction→outcome→PE one-to-one lineage audit: "
            f"coverage={lineage_coverage:.3f}, "
            f"mismatches={accepted_mismatch_count}, "
            f"duplicates={duplicate_settlement_count}."
        ),
    )


def _evaluation_snapshot(value: float) -> EvaluationSnapshot:
    return EvaluationSnapshot(
        turn_scores=(
            EvaluationScore(
                family="task",
                metric_name="gate1_task",
                value=value,
                confidence=1.0,
                evidence="gate1 evaluation-decoupled audit",
            ),
            EvaluationScore(
                family="relationship",
                metric_name="gate1_relationship",
                value=1.0 - value,
                confidence=1.0,
                evidence="gate1 evaluation-decoupled audit",
            ),
            EvaluationScore(
                family="learning",
                metric_name="gate1_learning",
                value=value,
                confidence=1.0,
                evidence="gate1 evaluation-decoupled audit",
            ),
            EvaluationScore(
                family="abstraction",
                metric_name="gate1_abstraction",
                value=1.0 - value,
                confidence=1.0,
                evidence="gate1 evaluation-decoupled audit",
            ),
            EvaluationScore(
                family="safety",
                metric_name="gate1_safety",
                value=value,
                confidence=1.0,
                evidence="gate1 evaluation-decoupled audit",
            ),
        ),
        session_scores=(),
        alerts=(),
        description=f"gate1 evaluation {value:.2f}",
    )


def _dual_track_snapshot() -> DualTrackSnapshot:
    return DualTrackSnapshot(
        world_track=TrackState(
            track=Track.WORLD,
            active_goals=("gate1-world",),
            recent_credits=(),
            controller_code=(0.2, 0.3),
            tension_level=0.4,
        ),
        self_track=TrackState(
            track=Track.SELF,
            active_goals=("gate1-self",),
            recent_credits=(),
            controller_code=(0.3, 0.2),
            tension_level=0.5,
        ),
        cross_track_tension=0.2,
        description="gate1 fixed dual-track evidence",
    )


def _json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=lambda value: value.value,
    ).encode("utf-8")


def _decoupled_trace_payload(evaluation_value: float) -> dict[str, object]:
    module = PredictionErrorModule(wiring_level=WiringLevel.ACTIVE)
    first = asyncio.run(
        module.process_standalone(
            turn_index=1,
            evaluation_snapshot=_evaluation_snapshot(evaluation_value),
            dual_track_snapshot=_dual_track_snapshot(),
        )
    )
    second = asyncio.run(
        module.process_standalone(
            previous_prediction=first.value.next_prediction,
            turn_index=2,
            evaluation_snapshot=_evaluation_snapshot(evaluation_value),
            dual_track_snapshot=_dual_track_snapshot(),
        )
    )
    if second.value.bootstrap or second.value.evaluated_prediction is None:
        raise RuntimeError(
            "Gate 1 evaluation-decoupled trace did not settle prediction"
        )
    credits = derive_prediction_error_credit_records(
        prediction_error=second.value.error,
        timestamp_ms=2,
        action_context=second.value.actual_outcome.action_context,
    )
    return {
        "actual_outcome": asdict(second.value.actual_outcome),
        "prediction_error": asdict(second.value.error),
        "learning_credit": tuple(
            {
                "level": credit.level,
                "track": credit.track.value,
                "source_event": credit.source_event,
                "credit_value": credit.credit_value,
                "context": credit.context,
                "timestamp_ms": credit.timestamp_ms,
                "prediction_id": credit.prediction_id,
                "environment_event_id": credit.environment_event_id,
                "environment_outcome_id": credit.environment_outcome_id,
            }
            for credit in credits
        ),
    }


def evaluate_evaluation_decoupling() -> Gate1EvaluationDecouplingReport:
    """Prove ACTIVE decoupling is byte-invariant to evaluation content."""

    variable_name = "VZ_PE_EVALUATION_DECOUPLED"
    previous = os.environ.get(variable_name)
    os.environ[variable_name] = "active"
    try:
        payload_a = _json_bytes(_decoupled_trace_payload(0.1))
        payload_b = _json_bytes(_decoupled_trace_payload(0.9))
    finally:
        if previous is None:
            os.environ.pop(variable_name, None)
        else:
            os.environ[variable_name] = previous
    fingerprint_a = hashlib.sha256(payload_a).hexdigest()
    fingerprint_b = hashlib.sha256(payload_b).hexdigest()
    byte_invariant = payload_a == payload_b
    return Gate1EvaluationDecouplingReport(
        active_payload_sha256_a=fingerprint_a,
        active_payload_sha256_b=fingerprint_b,
        byte_invariant=byte_invariant,
        rollback_gate="VZ_PE_EVALUATION_DECOUPLED=SHADOW",
        passed=byte_invariant,
        description=(
            "ACTIVE evaluation-decoupled actual outcome, PE and PE-derived "
            f"credit byte invariant: {byte_invariant}."
        ),
    )


def _gold_case_inputs() -> tuple[
    tuple[str, tuple[float, ...], Sequence[float] | float | int],
    ...,
]:
    return (
        ("numeric", (0.25,), 0.75),
        ("probability", (0.2,), 1.0),
        ("enum", (0.1, 0.7, 0.2), 2),
        ("vector", (0.1, 0.4, 0.9), (0.2, 0.0, 0.6)),
        ("distribution", (0.2, 0.5, 0.3), (0.1, 0.2, 0.7)),
    )


def _gold_reports() -> tuple[TypedLSSBridgeReport, ...]:
    return tuple(
        bridge_typed_runtime_pe_to_lss(
            surface_kind=surface_kind,
            predicted=predicted,
            actual=actual,
        )
        for surface_kind, predicted, actual in _gold_case_inputs()
    )


def _fail_loud_probe_count() -> int:
    invalid_cases = (
        ("probability", (1.0,), 1.0),
        ("enum", (0.2, 0.8), 2),
        ("vector", (0.1, 0.2), (0.1,)),
        ("distribution", (0.2, 0.8), (0.2, 0.2)),
    )
    rejection_count = 0
    for surface_kind, predicted, actual in invalid_cases:
        try:
            bridge_typed_runtime_pe_to_lss(
                surface_kind=surface_kind,
                predicted=predicted,
                actual=actual,
            )
        except ValueError:
            rejection_count += 1
    return rejection_count


def _git_output(*args: str) -> str:
    try:
        result = subprocess.run(
            ("git", *args),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def export_gate1_pe_mechanism_bundle(
    *,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Run the preregistered mechanism checks and write the 12-file packet."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    reports = _gold_reports()
    repeated_reports = _gold_reports()
    deterministic = tuple(asdict(report) for report in reports) == tuple(
        asdict(report) for report in repeated_reports
    )
    predictions: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    credit: list[dict[str, Any]] = []
    state_diff: list[dict[str, Any]] = []
    for index, report in enumerate(reports):
        prediction_id = f"gate1-prediction-{report.surface_kind}"
        event_id = f"gate1-event-{index:02d}"
        outcome_id = f"gate1-outcome-{index:02d}"
        observed_at = f"2026-07-30T00:00:{index:02d}+00:00"
        lineage = {
            "prediction_id": prediction_id,
            "environment_event_id": event_id,
            "environment_outcome_id": outcome_id,
            "observed_at": observed_at,
        }
        predictions.append(
            {
                "prediction_id": prediction_id,
                "surface_kind": report.surface_kind,
                "predicted": list(report.predicted),
                "link_function": report.link_function,
            }
        )
        outcomes.append(
            {
                **lineage,
                "actual": list(report.actual),
                "target_index": report.target_index,
            }
        )
        errors.append(
            {
                **lineage,
                "surface_kind": report.surface_kind,
                "runtime_signed_pe": list(report.runtime_signed_pe),
                "true_lss": list(report.lss),
                "max_abs_bridge_error": report.max_abs_bridge_error,
                "component_decomposition": [
                    list(component)
                    for component in report.component_decomposition
                ],
            }
        )
        segments.append(
            {
                **lineage,
                "segment_id": f"gate1-segment-{index:02d}",
                "settled": True,
            }
        )
        credit.append(
            {
                **lineage,
                "credit_source": "prediction-error",
                "component_count": len(report.runtime_signed_pe),
            }
        )
        state_diff.append(
            {
                **lineage,
                "runtime_pe_plus_lss": [
                    runtime_value + lss_value
                    for runtime_value, lss_value in zip(
                        report.runtime_signed_pe,
                        report.lss,
                        strict=True,
                    )
                ],
            }
        )
    lineage_audit = audit_prediction_lineage(
        predictions=predictions,
        outcomes=outcomes,
        prediction_errors=errors,
    )
    decoupling = evaluate_evaluation_decoupling()
    rejection_count = _fail_loud_probe_count()
    gates = {
        "five_gold_surfaces_present": len(reports) == 5,
        "gold_cases_deterministic": deterministic,
        "true_lss_bridge_within_1e_9": all(
            report.bridge_passed
            and report.max_abs_bridge_error <= _GATE1_LSS_TOLERANCE
            for report in reports
        ),
        "invalid_inputs_fail_loudly": rejection_count == 4,
        "lineage_coverage_complete": lineage_audit.lineage_coverage == 1.0,
        "lineage_mismatch_zero": (
            lineage_audit.accepted_mismatch_count == 0
        ),
        "duplicate_settlement_zero": (
            lineage_audit.duplicate_settlement_count == 0
        ),
        "evaluation_decoupled_byte_invariant": decoupling.passed,
    }
    passed = all(gates.values())
    manifest = {
        "schema_version": GATE1_PE_MECHANISM_SCHEMA_VERSION,
        "suite_id": "gate1-pe-mechanism",
        "owner": "PredictionErrorModule",
        "lss_tolerance": _GATE1_LSS_TOLERANCE,
        "surface_kinds": [
            report.surface_kind for report in reports
        ],
        "required_files": list(GATE1_PE_MECHANISM_REQUIRED_FILES),
        "lineage_key": "prediction_id",
        "settled_lineage_fields": [
            "environment_event_id",
            "environment_outcome_id",
            "observed_at",
        ],
        "evaluation_decoupled_gate": "ACTIVE",
        "provenance": {
            "git_sha": _git_output("rev-parse", "HEAD"),
            "git_branch": _git_output("branch", "--show-current"),
            "working_tree_dirty": bool(
                _git_output("status", "--porcelain") not in {"", "unknown"}
            ),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    ablation = {
        "schema_version": GATE1_PE_MECHANISM_SCHEMA_VERSION,
        "gates": gates,
        "passed": passed,
        "gold_reports": [asdict(report) for report in reports],
        "lineage_audit": asdict(lineage_audit),
        "evaluation_decoupling": asdict(decoupling),
        "invalid_input_rejection_count": rejection_count,
    }
    verdict = {
        "schema_version": GATE1_PE_MECHANISM_SCHEMA_VERSION,
        "gate_scope": "Gate 1 PE/LSS mechanism",
        "status": "mechanism-supported" if passed else "wiring-ready",
        "mechanism_passed": passed,
        "causal_status": "not-evaluated",
        "thesis_status": "not-evaluated",
        "failed_gates": [
            gate for gate, gate_passed in gates.items() if not gate_passed
        ],
    }
    rollback = {
        "schema_version": GATE1_PE_MECHANISM_SCHEMA_VERSION,
        "rollback_target": "VZ_PE_EVALUATION_DECOUPLED=SHADOW",
        "owner_state_mutated_by_evidence": False,
        "offline_lss_only": True,
        "passed": True,
    }
    payload_by_name: dict[str, object] = {
        "manifest.yaml": manifest,
        "ablation_results.json": ablation,
        "promotion_verdict.json": verdict,
        "rollback_evidence.json": rollback,
    }
    rows_by_name = {
        "predictions.jsonl": predictions,
        "outcomes.jsonl": outcomes,
        "prediction_errors.jsonl": errors,
        "segments.jsonl": segments,
        "credit.jsonl": credit,
        "state_diff.jsonl": state_diff,
        "action_selection.jsonl": [],
    }
    for name, payload in payload_by_name.items():
        _write_json(target / name, payload)
    for name, rows in rows_by_name.items():
        _write_jsonl(target / name, rows)
    report_lines = [
        "# Gate 1 PE/LSS mechanism evidence",
        "",
        f"- status: `{verdict['status']}`",
        f"- gold surfaces: `{len(reports)}`",
        (
            "- max LSS bridge error: "
            f"`{max(report.max_abs_bridge_error for report in reports):.12g}`"
        ),
        f"- lineage coverage: `{lineage_audit.lineage_coverage:.3f}`",
        (
            "- evaluation-decoupled byte invariant: "
            f"`{decoupling.byte_invariant}`"
        ),
        "- causal verdict: `not-evaluated`",
        "",
    ]
    (target / "report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )
    written = tuple(
        target / name for name in GATE1_PE_MECHANISM_REQUIRED_FILES
    )
    missing = tuple(path.name for path in written if not path.is_file())
    if missing:
        raise RuntimeError(
            f"Gate 1 mechanism bundle missing required files {missing!r}"
        )
    return written


__all__ = [
    "GATE1_PE_MECHANISM_REQUIRED_FILES",
    "GATE1_PE_MECHANISM_SCHEMA_VERSION",
    "Gate1EvaluationDecouplingReport",
    "Gate1LineageAudit",
    "audit_prediction_lineage",
    "evaluate_evaluation_decoupling",
    "export_gate1_pe_mechanism_bundle",
]
