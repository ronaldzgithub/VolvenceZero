#!/usr/bin/env python3
"""Loop-external OFFLINE gate adjudicator for Forge runtime proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import jsonschema


REPO_ROOT = Path(__file__).resolve().parents[1]
for _source_root in sorted((REPO_ROOT / "packages").glob("*/src")):
    sys.path.insert(0, str(_source_root))

from volvence_zero.credit import (  # noqa: E402
    GateDecision,
    ModificationGate,
    ModificationProposal,
    evaluate_gate_reasons,
)
from volvence_zero.evaluation import EvaluationScore, EvaluationSnapshot  # noqa: E402


class GateAdjudicationError(RuntimeError):
    """Raised when gate inputs are missing, stale, or internally inconsistent."""


def build_gate_decision(
    proposal_dir: Path,
    *,
    validation_report_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    proposal_dir = proposal_dir.resolve()
    patch_path = proposal_dir / "patch.diff"
    manifesto_path = proposal_dir / "manifesto.json"
    validation_path = (validation_report_path or proposal_dir / "validation.json").resolve()
    patch = _read_text(patch_path)
    manifesto_text = _read_text(manifesto_path)
    validation_text = _read_text(validation_path)
    manifesto = _read_object(manifesto_path)
    validation = _read_object(validation_path)
    if validation.get("status") != "PASS":
        raise GateAdjudicationError("OFFLINE adjudication requires a PASS validation report")
    if validation.get("proposal_id") != manifesto.get("proposal_id"):
        raise GateAdjudicationError("Validation report belongs to a different proposal")
    patch_sha = _sha256(patch)
    manifesto_sha = _sha256(manifesto_text)
    validation_sha = _sha256(validation_text)
    if validation.get("patch_sha256") != patch_sha:
        raise GateAdjudicationError("Patch changed after validation")
    if validation.get("manifesto_sha256") != manifesto_sha:
        raise GateAdjudicationError("Manifesto changed after validation")
    evidence = validation.get("runtime_gate_evidence")
    if not isinstance(evidence, dict):
        raise GateAdjudicationError("Validation report lacks runtime_gate_evidence")
    target = manifesto.get("target")
    if not isinstance(target, str) or evidence.get("target") != target:
        raise GateAdjudicationError("Runtime gate evidence target does not match manifesto")
    baseline_rate = _bounded_rate(evidence, "baseline_pass_rate")
    candidate_rate = _bounded_rate(evidence, "candidate_pass_rate")
    validation_delta = _finite_number(evidence, "validation_delta")
    if abs(validation_delta - (candidate_rate - baseline_rate)) > 1e-9:
        raise GateAdjudicationError("validation_delta does not match candidate minus baseline pass rate")
    capacity_cost = _finite_number(evidence, "capacity_cost")
    if capacity_cost != 0.1:
        raise GateAdjudicationError("Runtime asset capacity_cost must remain preregistered at 0.1")
    contract_integrity = _required_boolean(evidence, "contract_integrity")
    rollback_resilience = _required_boolean(evidence, "rollback_resilience")
    evaluation = EvaluationSnapshot(
        turn_scores=(),
        session_scores=(
            EvaluationScore(
                family="learning",
                metric_name="runtime_validation_delta",
                value=validation_delta,
                confidence=1.0,
                evidence="frozen scenario test-suite baseline/candidate comparison",
            ),
            EvaluationScore(
                family="safety",
                metric_name="contract_integrity",
                value=1.0 if contract_integrity else 0.0,
                confidence=1.0,
                evidence="frozen evaluator path and single-target proposal hash binding",
            ),
            EvaluationScore(
                family="safety",
                metric_name="rollback_resilience",
                value=1.0 if rollback_resilience else 0.0,
                confidence=1.0,
                evidence="byte-identical reverse-patch sandbox drill",
            ),
        ),
        alerts=(),
        structured_alerts=(),
        description="Forge runtime semantic asset OFFLINE promotion evidence.",
    )
    proposal = ModificationProposal(
        target=target,
        desired_gate=ModificationGate.OFFLINE,
        old_value_hash=str(manifesto.get("target_preimage_sha256", "")),
        new_value_hash=patch_sha,
        justification="Promote one reviewed runtime semantic asset after frozen-suite validation.",
        is_reversible=True,
        validation_delta=validation_delta,
        capacity_cost=capacity_cost,
        rollback_evidence=(
            str(manifesto.get("rollback", {}).get("command", ""))
            if rollback_resilience
            else ""
        ),
    )
    reasons = evaluate_gate_reasons(proposal=proposal, evaluation_snapshot=evaluation)
    decision = GateDecision.BLOCK if reasons else GateDecision.ALLOW
    payload = {
        "schema_version": "forge-gate-decision.v1",
        "proposal_id": manifesto["proposal_id"],
        "target": target,
        "decision": decision.value.upper(),
        "reasons": list(reasons),
        "desired_gate": ModificationGate.OFFLINE.value,
        "inputs": {
            "patch_sha256": patch_sha,
            "manifesto_sha256": manifesto_sha,
            "validation_sha256": validation_sha,
        },
        "metrics": {
            "baseline_pass_rate": baseline_rate,
            "candidate_pass_rate": candidate_rate,
            "validation_delta": validation_delta,
            "capacity_cost": capacity_cost,
            "contract_integrity": 1.0 if contract_integrity else 0.0,
            "rollback_resilience": 1.0 if rollback_resilience else 0.0,
        },
        "authority": "volvence_zero.credit.gate.evaluate_gate_reasons",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    schema = _read_object(REPO_ROOT / "forge" / "schemas" / "gate_decision.schema.json")
    try:
        jsonschema.Draft202012Validator(schema).validate(payload)
    except jsonschema.ValidationError as exc:
        raise GateAdjudicationError(f"Generated gate decision violates schema: {exc.message}") from exc
    destination = (output_path or proposal_dir / "gate_decision.json").resolve()
    _atomic_write(destination, payload)
    return destination


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, UnicodeDecodeError) as exc:
        raise GateAdjudicationError(f"Cannot read gate input {path}: {exc}") from exc


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        raise GateAdjudicationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateAdjudicationError(f"Expected JSON object in {path}")
    return value


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _finite_number(raw: dict[str, Any], key: str) -> float:
    value = raw.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise GateAdjudicationError(f"runtime_gate_evidence.{key} must be numeric")
    numeric = float(value)
    if numeric != numeric or numeric in {float("inf"), float("-inf")}:
        raise GateAdjudicationError(f"runtime_gate_evidence.{key} must be finite")
    return numeric


def _bounded_rate(raw: dict[str, Any], key: str) -> float:
    value = _finite_number(raw, key)
    if not 0.0 <= value <= 1.0:
        raise GateAdjudicationError(f"runtime_gate_evidence.{key} must be in [0, 1]")
    return value


def _required_boolean(raw: dict[str, Any], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise GateAdjudicationError(f"runtime_gate_evidence.{key} must be boolean")
    return value


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError as exc:
        if temporary.exists():
            temporary.unlink()
        raise GateAdjudicationError(f"Cannot write gate decision {path}: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adjudicate a Forge runtime proposal through ModificationGate.OFFLINE"
    )
    parser.add_argument("proposal_dir", type=Path)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        destination = build_gate_decision(
            args.proposal_dir,
            validation_report_path=args.validation_report,
            output_path=args.output,
        )
    except GateAdjudicationError as exc:
        print(f"forge gate: {exc}", file=sys.stderr)
        return 2
    decision = _read_object(destination)["decision"]
    print(f"{decision}: {destination}")
    return 0 if decision == "ALLOW" else 2


if __name__ == "__main__":
    raise SystemExit(main())
