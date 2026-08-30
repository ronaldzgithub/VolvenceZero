"""Public, self-contained v2 Foundry simulation seam; it never invokes Praxist."""

from __future__ import annotations
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from .foundation import (
    ForgeError,
    SchemaStore,
    canonical_json,
    read_json,
    sha256_bytes,
    sha256_text,
)

_SCHEMA = "foundry_research_handoff_v2.schema.json"
_RUNNER = "volvence_deterministic_simulation"
_SCOPE = "research_runner_start"
_PROTOCOL = "volvence-deterministic-simulation.v1"


class FoundrySimulationError(ForgeError):
    pass


@dataclass(frozen=True)
class SimulationHandoffResult:
    handoff_path: Path
    handoff_id: str
    schema_sha256: str


def _p() -> Path:
    return Path(__file__).resolve().parents[2] / "schemas" / _SCHEMA


def _sha(v: dict[str, Any]) -> str:
    return sha256_text(canonical_json(v))


def _id(prefix: str, v: dict[str, Any], field: str) -> dict[str, Any]:
    r = dict(v)
    r[field] = f"{prefix}:{sha256_text(canonical_json({k: x for k, x in r.items() if k not in {field, 'created_at'}}))}"
    return r


def _ref(v: dict[str, Any], field: str) -> dict[str, str]:
    return {"artifact_id": v[field], "sha256": _sha(v)}


def _v(v: dict[str, Any], version: str) -> None:
    if v.get("schema_version") != version:
        raise FoundrySimulationError(f"expected {version}")
    SchemaStore(_p().parent).validate(v, _SCHEMA)


def _authority() -> dict[str, bool]:
    return {
        "network_allowed": False,
        "external_actions_allowed": False,
        "simulation_only": True,
        "field_claimed": False,
        "revenue_claimed": False,
        "profit_claimed": False,
        "adoption_claimed": False,
        "active_claimed": False,
        "volvence_promotion_eligible": False,
        "modification_gate_applicable": False,
        "runtime_wiring_applicable": False,
        "foundry_approve_allowed": False,
        "foundry_reconcile_allowed": False,
        "foundry_runner_start_allowed": False,
    }


def _perms() -> dict[str, list[str]]:
    return {
        "allowed_operations": ["import_simulation_handoff"],
        "prohibited_operations": [
            "approve_research_request",
            "reconcile_research_control",
            "start_research_runner",
            "import_volvence_candidate",
            "modify_runtime_wiring",
        ],
    }


def create_simulation_request(*, intent: dict[str, Any], created_at: str) -> dict[str, Any]:
    _v(intent, "foundry-research-lab-intent.v2")
    _verify_intent_identity(intent)
    return _id(
        "research-runner-request",
        {
            "schema_version": "forge-external-research-simulation-request.v2",
            "request_id": "",
            "intent": intent,
            "objective": intent["objective"],
            "subject_refs": intent["subject_refs"],
            "evidence_class": "simulation",
            "budget_policy": intent["budget_policy"],
            "stop_policy": intent["stop_policy"],
            "acceptance_policy": intent["acceptance_policy"],
            "result_policy": intent["result_policy"],
            "authority": _authority(),
            "created_at": created_at,
        },
        "request_id",
    )


def approve_simulation_request(
    *, request: dict[str, Any], reviewed_by: str, reason: str, created_at: str
) -> dict[str, Any]:
    _v(request, "forge-external-research-simulation-request.v2")
    if not reviewed_by.strip() or not reason.strip():
        raise FoundrySimulationError("A0 needs named reviewer and reason")
    return _id(
        "research-runner-approval",
        {
            "schema_version": "forge-research-runner-approval.v2",
            "approval_id": "",
            "request_id": request["request_id"],
            "request_sha256": _sha(request),
            "scope": _SCOPE,
            "decision": "APPROVE",
            "review": {"reviewed_by": reviewed_by, "reason": reason},
            "created_at": created_at,
        },
        "approval_id",
    )


def run_deterministic_simulation(
    *, request: dict[str, Any], approval: dict[str, Any], created_at: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    _v(request, "forge-external-research-simulation-request.v2")
    _v(approval, "forge-research-runner-approval.v2")
    if approval["request_id"] != request["request_id"] or approval["request_sha256"] != _sha(request):
        raise FoundrySimulationError("A0 does not bind request")
    runner = {
        "runner_id": _RUNNER,
        "protocol_id": _PROTOCOL,
        "protocol_version": 1,
        "seed": int(sha256_text(request["request_id"])[:8], 16),
        "network_allowed": False,
        "external_actions_allowed": False,
    }
    result = _id(
        "research-runner-result",
        {
            "schema_version": "volvence-deterministic-simulation-result.v1",
            "result_id": "",
            "request_id": request["request_id"],
            "runner": runner,
            "evidence_class": "simulation",
            "field_claimed": False,
            "revenue_claimed": False,
            "profit_claimed": False,
            "adoption_claimed": False,
            "active_claimed": False,
            "outcome": {
                "status": "deterministic_simulation_completed",
                "objective_sha256": sha256_text(request["objective"]),
            },
            "created_at": created_at,
        },
        "result_id",
    )
    completion = _id(
        "research-runner-completion",
        {
            "schema_version": "forge-research-run-completion.v2",
            "completion_id": "",
            "request_id": request["request_id"],
            "approval_id": approval["approval_id"],
            "runner": runner,
            "state": "RUN_COMPLETED",
            "result_id": result["result_id"],
            "result_sha256": _sha(result),
            "created_at": created_at,
        },
        "completion_id",
    )
    return completion, result


def seal_simulation_handoff(
    *,
    intent: dict[str, Any],
    request: dict[str, Any],
    approval: dict[str, Any],
    completion: dict[str, Any],
    result: dict[str, Any],
    created_at: str,
) -> dict[str, Any]:
    for x, n in (
        (intent, "foundry-research-lab-intent.v2"),
        (request, "forge-external-research-simulation-request.v2"),
        (approval, "forge-research-runner-approval.v2"),
        (completion, "forge-research-run-completion.v2"),
        (result, "volvence-deterministic-simulation-result.v1"),
    ):
        _v(x, n)
    _verify_intent_identity(intent)
    if not _cross_bound(intent, request, approval, completion, result):
        raise FoundrySimulationError("cross-boundary chain")
    chain = {
        "request": _ref(request, "request_id"),
        "approval": _ref(approval, "approval_id"),
        "run_completion": _ref(completion, "completion_id"),
        "result": _ref(result, "result_id"),
    }
    chain["chain_sha256"] = _sha(chain)
    h = _id(
        "external-research-handoff",
        {
            "schema_version": "forge-foundry-research-handoff.v2",
            "handoff_id": "",
            "contract": {
                "contract_version": "foundry-research-lab-seam.v2",
                "schema_sha256": sha256_bytes(_p().read_bytes()),
                "producer_owner": "volvence_labs.research_lab",
                "consumer_domain": "foundry",
            },
            "intent": intent,
            "request": request,
            "approval": approval,
            "run_completion": completion,
            "result": result,
            "hash_chain": chain,
            "consumer_permissions": _perms(),
            "authority": _authority(),
            "created_at": created_at,
        },
        "handoff_id",
    )
    _v(h, "forge-foundry-research-handoff.v2")
    return h


def write_immutable(path: Path, payload: dict[str, Any]) -> None:
    s = canonical_json(payload) + "\n"
    encoded = s.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != encoded:
        raise FoundrySimulationError(f"immutable artifact conflicts: {path}")
    if not path.exists():
        path.write_bytes(encoded)


def verify_simulation_handoff(path: Path) -> dict[str, Any]:
    h = read_json(path)
    _v(h, "forge-foundry-research-handoff.v2")
    _verify_intent_identity(h["intent"])
    expected_chain = {
        "request": _ref(h["request"], "request_id"),
        "approval": _ref(h["approval"], "approval_id"),
        "run_completion": _ref(h["run_completion"], "completion_id"),
        "result": _ref(h["result"], "result_id"),
    }
    expected_chain["chain_sha256"] = _sha(expected_chain)
    if h["hash_chain"] != expected_chain:
        raise FoundrySimulationError("hash chain does not close")
    expected_contract = {
        "contract_version": "foundry-research-lab-seam.v2",
        "schema_sha256": sha256_bytes(_p().read_bytes()),
        "producer_owner": "volvence_labs.research_lab",
        "consumer_domain": "foundry",
    }
    if h["contract"] != expected_contract or h["consumer_permissions"] != _perms() or h["authority"] != _authority():
        raise FoundrySimulationError("contract, permissions, or authority drifted")
    if h["handoff_id"] != _id("external-research-handoff", h, "handoff_id")["handoff_id"]:
        raise FoundrySimulationError("handoff id mismatch")
    if h != seal_simulation_handoff(
        intent=h["intent"],
        request=h["request"],
        approval=h["approval"],
        completion=h["run_completion"],
        result=h["result"],
        created_at=h["created_at"],
    ):
        raise FoundrySimulationError("handoff replay mismatch")
    return h


def _verify_intent_identity(intent: dict[str, Any]) -> None:
    body = {key: value for key, value in intent.items() if key not in {"intent_id", "created_at"}}
    expected = "rli2_" + sha256_text(canonical_json(body))[:16]
    if intent["intent_id"] != expected:
        raise FoundrySimulationError("Foundry Intent v2 identity does not close")


def _cross_bound(
    intent: dict[str, Any],
    request: dict[str, Any],
    approval: dict[str, Any],
    completion: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    projection = ("objective", "subject_refs", "budget_policy", "stop_policy", "acceptance_policy", "result_policy")
    return (
        request["intent"] == intent
        and all(request[field] == intent[field] for field in projection)
        and approval["request_id"] == request["request_id"]
        and approval["request_sha256"] == _sha(request)
        and approval["scope"] == _SCOPE
        and approval["decision"] == "APPROVE"
        and completion["request_id"] == request["request_id"]
        and completion["approval_id"] == approval["approval_id"]
        and completion["result_id"] == result["result_id"]
        and completion["result_sha256"] == _sha(result)
        and completion["runner"] == result["runner"]
        and completion["state"] == "RUN_COMPLETED"
        and result["request_id"] == request["request_id"]
        and result["runner"]["runner_id"] == _RUNNER
        and all(
            result[key] is False
            for key in ("field_claimed", "revenue_claimed", "profit_claimed", "adoption_claimed", "active_claimed")
        )
    )


def generate_simulation_handoff(
    *,
    intent_path: Path,
    output_dir: Path,
    approved_by: str,
    approval_reason: str,
    created_at: str = "2026-08-30T00:00:00Z",
) -> SimulationHandoffResult:
    """Convenience fixture producer; production callers should use the four stage functions."""
    intent = read_json(intent_path)
    request = create_simulation_request(intent=intent, created_at=created_at)
    approval = approve_simulation_request(
        request=request, reviewed_by=approved_by, reason=approval_reason, created_at=created_at
    )
    completion, result = run_deterministic_simulation(request=request, approval=approval, created_at=created_at)
    handoff = seal_simulation_handoff(
        intent=intent, request=request, approval=approval, completion=completion, result=result, created_at=created_at
    )
    for name, value in (
        ("request.json", request),
        ("approval.json", approval),
        ("run_completion.json", completion),
        ("result.json", result),
        ("handoff.json", handoff),
    ):
        write_immutable(output_dir / name, value)
    verify_simulation_handoff(output_dir / "handoff.json")
    return SimulationHandoffResult(
        output_dir / "handoff.json",
        handoff["handoff_id"],
        sha256_bytes(_p().read_bytes()),
    )
