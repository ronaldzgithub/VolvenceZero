"""Bounded Pareto selection and explicit STOP decisions for Forge proposals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import ForgeConfig
from .foundation import (
    ForgeError,
    SchemaStore,
    atomic_write_json,
    read_json,
    sha256_text,
    utc_now,
)


@dataclass(frozen=True)
class OptimizerResult:
    decision: str
    report_path: Path
    selected_proposal_ids: tuple[str, ...]


def select_pareto_candidates(
    *,
    config: ForgeConfig,
    proposals_root: Path,
    output_path: Path | None = None,
    selection_limit_per_component: int = 1,
) -> OptimizerResult:
    if selection_limit_per_component <= 0:
        raise ForgeError("selection_limit_per_component must be positive")
    root = proposals_root.expanduser().resolve()
    if not root.is_dir():
        raise ForgeError(f"proposal population root is not a directory: {root}")
    proposal_dirs = tuple(sorted(path for path in root.iterdir() if path.is_dir()))
    schema_store = SchemaStore(config.paths.forge_root / "schemas")
    candidates = tuple(
        _load_candidate(config=config, schema_store=schema_store, proposal_dir=path)
        for path in proposal_dirs
    )
    groups: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        groups.setdefault(str(candidate["component"]), []).append(candidate)

    frontiers: list[dict[str, object]] = []
    selected: list[str] = []
    for component in sorted(groups):
        eligible = [candidate for candidate in groups[component] if candidate["eligible"]]
        frontier = [
            candidate
            for candidate in eligible
            if not any(
                _dominates(other["metrics"], candidate["metrics"])
                for other in eligible
                if other is not candidate
            )
        ]
        ranked = sorted(frontier, key=_ranking_key)
        component_selected = [
            str(candidate["proposal_id"])
            for candidate in ranked[:selection_limit_per_component]
        ]
        selected.extend(component_selected)
        frontiers.append(
            {
                "component": component,
                "eligible_proposal_ids": sorted(
                    str(candidate["proposal_id"]) for candidate in eligible
                ),
                "pareto_front_ids": sorted(
                    str(candidate["proposal_id"]) for candidate in frontier
                ),
                "selected_ids": component_selected,
            }
        )

    stop_reasons: list[str] = []
    if not proposal_dirs:
        stop_reasons.append("population is empty")
    elif not selected:
        stop_reasons.append("no candidate has PASS validation and required gate ALLOW evidence")
    decision = "SELECT" if selected else "STOP"
    payload = {
        "schema_version": "forge-optimizer-decision.v1",
        "decision": decision,
        "candidates": list(candidates),
        "component_frontiers": frontiers,
        "selected_proposal_ids": selected,
        "stop_reasons": stop_reasons,
        "created_at": utc_now(),
    }
    schema_store.validate(payload, "optimizer_decision.schema.json")
    destination = (output_path or root.parent / "optimizer_decision.json").resolve()
    atomic_write_json(destination, payload)
    return OptimizerResult(
        decision=decision,
        report_path=destination,
        selected_proposal_ids=tuple(selected),
    )


def _load_candidate(
    *,
    config: ForgeConfig,
    schema_store: SchemaStore,
    proposal_dir: Path,
) -> dict[str, Any]:
    patch_path = proposal_dir / "patch.diff"
    manifesto_path = proposal_dir / "manifesto.json"
    validation_path = proposal_dir / "validation.json"
    try:
        patch = patch_path.read_text(encoding="utf-8")
        manifesto_text = manifesto_path.read_text(encoding="utf-8")
        validation_text = validation_path.read_text(encoding="utf-8")
    except (FileNotFoundError, UnicodeDecodeError) as exc:
        raise ForgeError(f"incomplete optimizer candidate {proposal_dir}: {exc}") from exc
    manifesto = read_json(manifesto_path)
    validation = read_json(validation_path)
    schema_store.validate(manifesto, "proposal_manifesto.schema.json")
    schema_store.validate(validation, "validation_report.schema.json")
    proposal_id = manifesto["proposal_id"]
    if proposal_dir.name != proposal_id:
        raise ForgeError(
            f"proposal directory {proposal_dir.name!r} does not match manifesto {proposal_id!r}"
        )
    if validation["proposal_id"] != proposal_id:
        raise ForgeError(f"validation report belongs to a different proposal: {proposal_id}")
    patch_sha = sha256_text(patch)
    manifesto_sha = sha256_text(manifesto_text)
    validation_sha = sha256_text(validation_text)
    if validation["patch_sha256"] != patch_sha:
        raise ForgeError(f"optimizer candidate patch changed after validation: {proposal_id}")
    if validation["manifesto_sha256"] != manifesto_sha:
        raise ForgeError(f"optimizer candidate manifesto changed after validation: {proposal_id}")
    target = manifesto["target"]
    entry = config.editable_entry_for(target)
    if entry is None:
        raise ForgeError(f"optimizer candidate target is no longer editable: {target}")

    reasons: list[str] = []
    if validation["status"] != "PASS":
        reasons.append("validation status is not PASS")
    runtime_evidence = validation.get("runtime_gate_evidence")
    if isinstance(runtime_evidence, dict):
        validation_delta = float(runtime_evidence["validation_delta"])
        capacity_cost = float(runtime_evidence["capacity_cost"])
    else:
        validation_delta = 0.0
        capacity_cost = 0.1

    gate_sha: str | None = None
    if entry.requires_offline_gate:
        gate_path = proposal_dir / "gate_decision.json"
        try:
            gate_text = gate_path.read_text(encoding="utf-8")
        except (FileNotFoundError, UnicodeDecodeError):
            reasons.append("required OFFLINE gate decision is missing")
        else:
            gate = read_json(gate_path)
            schema_store.validate(gate, "gate_decision.schema.json")
            gate_sha = sha256_text(gate_text)
            if gate["proposal_id"] != proposal_id or gate["target"] != target:
                raise ForgeError(f"gate decision identity mismatch for {proposal_id}")
            expected_inputs = {
                "patch_sha256": patch_sha,
                "manifesto_sha256": manifesto_sha,
                "validation_sha256": validation_sha,
            }
            if gate["inputs"] != expected_inputs:
                raise ForgeError(f"gate decision inputs are stale for {proposal_id}")
            if gate["decision"] != "ALLOW":
                reasons.append("OFFLINE gate decision is not ALLOW")

    added_lines = sum(
        1
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    if added_lines <= 0:
        raise ForgeError(f"optimizer candidate has no added lines: {proposal_id}")
    risk_count = len(manifesto["at_risk_regressions"])
    return {
        "proposal_id": proposal_id,
        "component": entry.component,
        "target": target,
        "eligible": not reasons,
        "blocking_reasons": reasons,
        "metrics": {
            "validation_delta": validation_delta,
            "capacity_cost": capacity_cost,
            "added_lines": added_lines,
            "risk_count": risk_count,
        },
        "inputs": {
            "patch_sha256": patch_sha,
            "manifesto_sha256": manifesto_sha,
            "validation_sha256": validation_sha,
            "gate_sha256": gate_sha,
        },
    }


def _dominates(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    comparisons = (
        float(left["validation_delta"]) >= float(right["validation_delta"]),
        float(left["capacity_cost"]) <= float(right["capacity_cost"]),
        int(left["added_lines"]) <= int(right["added_lines"]),
        int(left["risk_count"]) <= int(right["risk_count"]),
    )
    strict = (
        float(left["validation_delta"]) > float(right["validation_delta"])
        or float(left["capacity_cost"]) < float(right["capacity_cost"])
        or int(left["added_lines"]) < int(right["added_lines"])
        or int(left["risk_count"]) < int(right["risk_count"])
    )
    return all(comparisons) and strict


def _ranking_key(candidate: Mapping[str, object]) -> tuple[float, float, int, int, str]:
    metrics = candidate["metrics"]
    assert isinstance(metrics, Mapping)
    return (
        -float(metrics["validation_delta"]),
        float(metrics["capacity_cost"]),
        int(metrics["added_lines"]),
        int(metrics["risk_count"]),
        str(candidate["proposal_id"]),
    )


__all__ = ["OptimizerResult", "select_pareto_candidates"]
