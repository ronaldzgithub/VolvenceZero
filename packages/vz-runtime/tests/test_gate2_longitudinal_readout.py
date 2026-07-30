from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from volvence_zero.agent.gate2_longitudinal_readout import (
    GATE2_LONGITUDINAL_INPUT_SCHEMA_VERSION,
    GATE2_LONGITUDINAL_OUTCOME_SCHEMA_VERSION,
    Gate2LongitudinalReadoutContract,
    assess_gate2_longitudinal_readout_admission,
    load_and_assess_gate2_longitudinal_readout,
)


def _record(index: int) -> dict[str, object]:
    return {
        "transition_id": f"t-{index}",
        "settled": True,
        "substrate": {
            "capture_source": "real",
            "fallback_active": False,
            "mutation_applied": False,
        },
        "longitudinal": {
            "consumer_session_boundary_interval": 10,
        },
    }


def _input(
    transition_id: str,
    contract: Gate2LongitudinalReadoutContract,
) -> dict[str, object]:
    return {
        "schema_version": GATE2_LONGITUDINAL_INPUT_SCHEMA_VERSION,
        "transition_id": transition_id,
        "selector_fingerprint": contract.selector_fingerprint,
        "control_basis_fingerprint": contract.control_basis_fingerprint,
        "state_features": [0.1, -0.2],
        "capture_source": "real",
        "fallback_active": False,
        "substrate_mutation_applied": False,
    }


def _outcome(
    transition_id: str,
    contract: Gate2LongitudinalReadoutContract,
) -> dict[str, object]:
    return {
        "schema_version": GATE2_LONGITUDINAL_OUTCOME_SCHEMA_VERSION,
        "transition_id": transition_id,
        "selector_fingerprint": contract.selector_fingerprint,
        "control_basis_fingerprint": contract.control_basis_fingerprint,
        "selected_action_index": 1,
        "selected_realized_delta": 0.03,
        "zero_realized_delta": 0.0,
        "permutation_null_mean": 0.005,
        "outcome_chain": (
            "isolated-residual-forward->realized-continuation-nll"
            "->prediction-error->action-credit"
        ),
        "source_fixed_outcome_reused": False,
    }


def _contract() -> Gate2LongitudinalReadoutContract:
    return Gate2LongitudinalReadoutContract(
        selector_fingerprint=hashlib.sha256(b"selector").hexdigest(),
        control_basis_fingerprint=hashlib.sha256(b"basis").hexdigest(),
        selector_input_dim=2,
        selector_action_count=2,
        min_settled_transition_count=5,
        min_consumer_session_count=2,
    )


def test_admission_fails_closed_when_source_lacks_readout_companions() -> None:
    contract = _contract()
    records = tuple(_record(index) for index in range(20))
    result = assess_gate2_longitudinal_readout_admission(
        records_by_seed={1: records},
        readout_inputs_by_seed={1: {}},
        matched_outcomes_by_seed={1: {}},
        selector_lineage={"selector_fingerprint": contract.selector_fingerprint},
        contract=contract,
    )

    assert result["source_admitted"] is True
    assert result["missing_readout_input_count"] == 20
    assert result["missing_matched_outcome_count"] == 20
    assert result["admission_status"] == "capture-required"
    assert result["readout_ready"] is False
    assert result["validation_delta_computed"] is False
    assert result["promotion_allowed"] is False


def test_admission_accepts_complete_one_to_one_readout_contract() -> None:
    contract = _contract()
    records = tuple(_record(index) for index in range(20))
    inputs = {
        str(record["transition_id"]): _input(
            str(record["transition_id"]),
            contract,
        )
        for record in records
    }
    outcomes = {
        str(record["transition_id"]): _outcome(
            str(record["transition_id"]),
            contract,
        )
        for record in records
    }
    result = assess_gate2_longitudinal_readout_admission(
        records_by_seed={1: records},
        readout_inputs_by_seed={1: inputs},
        matched_outcomes_by_seed={1: outcomes},
        selector_lineage={"selector_fingerprint": contract.selector_fingerprint},
        contract=contract,
    )

    assert result["source_admitted"] is True
    assert result["readout_ready"] is True
    assert result["admission_status"] == "readout-ready"
    assert result["selector_executed"] is False
    assert result["substrate_control_applied"] is False


def test_admission_rejects_fixed_source_outcome_as_selector_outcome() -> None:
    contract = _contract()
    records = tuple(_record(index) for index in range(20))
    inputs = {
        str(record["transition_id"]): _input(
            str(record["transition_id"]),
            contract,
        )
        for record in records
    }
    outcomes = {
        str(record["transition_id"]): _outcome(
            str(record["transition_id"]),
            contract,
        )
        for record in records
    }
    outcomes["t-0"]["source_fixed_outcome_reused"] = True

    with pytest.raises(ValueError, match="reused the fixed source result"):
        assess_gate2_longitudinal_readout_admission(
            records_by_seed={1: records},
            readout_inputs_by_seed={1: inputs},
            matched_outcomes_by_seed={1: outcomes},
            selector_lineage={
                "selector_fingerprint": contract.selector_fingerprint
            },
            contract=contract,
        )


def test_contract_is_json_serializable() -> None:
    payload = json.dumps(_contract().__dict__, sort_keys=True)
    assert "selector_input_dim" in payload


def test_repository_source_requires_fullwidth_readout_capture() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    assessment, _hashes = load_and_assess_gate2_longitudinal_readout(
        source_root=(
            repository_root
            / "artifacts"
            / "gate11_longitudinal_source_v2_20260730"
        ),
        selector_artifact_path=(
            repository_root
            / "artifacts"
            / "eta_gate2_v36_recent_k1_development_20260730"
            / "selector_artifact.json"
        ),
    )

    assert assessment["source_admitted"] is True
    assert assessment["total_transition_count"] == 1530
    assert assessment["missing_readout_input_count"] == 1530
    assert assessment["missing_matched_outcome_count"] == 1530
    assert assessment["admission_status"] == "capture-required"
    assert assessment["longitudinal_verdict"] == "not-supported"
