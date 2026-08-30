from pathlib import Path
import copy
import hashlib

import pytest

from volvence_forge.foundation import SchemaContractError, canonical_json, read_json, sha256_text
from volvence_forge.foundry_research_handoff_v2 import (
    FoundrySimulationError,
    create_simulation_request,
    approve_simulation_request,
    run_deterministic_simulation,
    seal_simulation_handoff,
    verify_simulation_handoff,
    write_immutable,
)


def test_v2_chain_is_a0_bound_and_self_contained(tmp_path: Path) -> None:
    intent = {
        "schema_version": "foundry-research-lab-intent.v2",
        "intent_id": "",
        "objective": "Test the runner-agnostic Foundry M5 public handoff seam.",
        "subject_refs": [{"subject_id": "subject", "sha256": "0" * 64}],
        "budget_policy": {"value": "zero external spend"},
        "stop_policy": {"value": "one deterministic completion"},
        "acceptance_policy": {"value": "import-only"},
        "result_policy": {"value": "simulation proposal only"},
        "created_at": "2026-08-30T00:00:00Z",
    }
    intent["intent_id"] = (
        "rli2_"
        + sha256_text(
            canonical_json({key: value for key, value in intent.items() if key not in {"intent_id", "created_at"}})
        )[:16]
    )
    request = create_simulation_request(intent=intent, created_at=intent["created_at"])
    approval = approve_simulation_request(
        request=request,
        reviewed_by="Named A0 Reviewer",
        reason="Approve exact runner start",
        created_at=intent["created_at"],
    )
    completion, result = run_deterministic_simulation(
        request=request, approval=approval, created_at=intent["created_at"]
    )
    handoff = seal_simulation_handoff(
        intent=intent,
        request=request,
        approval=approval,
        completion=completion,
        result=result,
        created_at=intent["created_at"],
    )
    path = tmp_path / "handoff.json"
    write_immutable(path, handoff)
    assert verify_simulation_handoff(path)["approval"]["scope"] == "research_runner_start"


def test_checked_in_sample_binds_real_subject_bytes() -> None:
    contract = Path(__file__).resolve().parents[1] / "contracts" / "foundry_research_lab_seam" / "v2"
    subject = contract / "subject.sample.json"
    intent = read_json(contract / "intent.example.json")
    handoff = contract / "simulation_handoff.sample" / "handoff.json"
    digest = hashlib.sha256(subject.read_bytes()).hexdigest()

    assert digest != "0" * 64
    assert intent["subject_refs"] == [
        {"subject_id": "foundry_m5_contract_sample", "sha256": digest}
    ]
    assert verify_simulation_handoff(handoff)["intent"] == intent


@pytest.mark.parametrize(
    "path,value",
    [
        (("intent", "objective"), "tampered"),
        (("intent", "intent_id"), "rli2_0000000000000000"),
        (("request", "objective"), "tampered"),
        (("request", "request_id"), "research-runner-request:" + "0" * 64),
        (("approval", "decision"), "REJECT"),
        (("approval", "scope"), "praxist_research_start"),
        (("approval", "request_sha256"), "0" * 64),
        (("approval", "review", "reviewed_by"), ""),
        (("run_completion", "state"), "RUNNING"),
        (("run_completion", "request_id"), "research-runner-request:" + "0" * 64),
        (("run_completion", "approval_id"), "research-runner-approval:" + "0" * 64),
        (("run_completion", "result_sha256"), "0" * 64),
        (("result", "request_id"), "research-runner-request:" + "0" * 64),
        (("result", "field_claimed"), True),
        (("hash_chain", "chain_sha256"), "0" * 64),
        (("hash_chain", "request", "sha256"), "0" * 64),
        (("contract", "schema_sha256"), "0" * 64),
        (("contract", "producer_owner"), "foundry"),
        (("consumer_permissions", "allowed_operations"), []),
        (("authority", "network_allowed"), True),
    ],
)
def test_v2_verifier_rejects_each_tampered_boundary(tmp_path: Path, path: tuple[str, ...], value: object) -> None:
    intent = _valid_intent()
    request = create_simulation_request(intent=intent, created_at=intent["created_at"])
    approval = approve_simulation_request(
        request=request, reviewed_by="A0", reason="exact", created_at=intent["created_at"]
    )
    completion, result = run_deterministic_simulation(
        request=request, approval=approval, created_at=intent["created_at"]
    )
    handoff = seal_simulation_handoff(
        intent=intent,
        request=request,
        approval=approval,
        completion=completion,
        result=result,
        created_at=intent["created_at"],
    )
    mutated = copy.deepcopy(handoff)
    target = mutated
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    candidate = tmp_path / "tampered.json"
    candidate.write_text(canonical_json(mutated), encoding="utf-8")
    with pytest.raises((FoundrySimulationError, SchemaContractError)):
        verify_simulation_handoff(candidate)


def _valid_intent() -> dict[str, object]:
    intent = {
        "schema_version": "foundry-research-lab-intent.v2",
        "intent_id": "",
        "objective": "Test the runner-agnostic Foundry M5 public handoff seam.",
        "subject_refs": [{"subject_id": "subject", "sha256": "0" * 64}],
        "budget_policy": {"value": "zero external spend"},
        "stop_policy": {"value": "one completion"},
        "acceptance_policy": {"value": "import-only"},
        "result_policy": {"value": "simulation only"},
        "created_at": "2026-08-30T00:00:00Z",
    }
    intent["intent_id"] = (
        "rli2_"
        + sha256_text(
            canonical_json({key: value for key, value in intent.items() if key not in {"intent_id", "created_at"}})
        )[:16]
    )
    return intent
