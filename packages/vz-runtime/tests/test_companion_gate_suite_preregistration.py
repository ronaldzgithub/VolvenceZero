from __future__ import annotations

from pathlib import Path

import pytest

from lifeform_service.companion_evidence_profile import (
    resolve_companion_evidence_profile,
)
from volvence_zero.agent.companion_gate_suite_evidence import (
    GATE_ARM_SCHEDULES,
)
from volvence_zero.agent.companion_gate_suite_preregistration import (
    build_companion_gate_suite_preregistration,
    validate_companion_gate_suite_preregistration,
    write_companion_gate_suite_preregistration,
)


def _models() -> tuple[dict[str, object], dict[str, object]]:
    sut = {
        "model_id": "local/sut",
        "model_family": "smollm",
        "weights_sha256": "a" * 64,
        "local_files_only": True,
        "frozen": True,
        "max_new_tokens": 96,
    }
    simulator = {
        "model_id": "local/simulator",
        "model_family": "qwen",
        "weights_sha256": "b" * 64,
        "local_files_only": True,
        "frozen": True,
        "max_new_tokens": 12,
        "temperature": 0.0,
        "top_p": 1.0,
        "rendering_contract": "typed deterministic renderer",
    }
    return sut, simulator


def _profiles(gate_id: int) -> dict[str, dict[str, object]]:
    return {arm: resolve_companion_evidence_profile(arm).intervention_contract() for arm in GATE_ARM_SCHEDULES[gate_id]}


@pytest.mark.parametrize("gate_id", [4, 5, 6, 7, 9, 10])
def test_preregistration_freezes_exact_gate_matrix(gate_id: int) -> None:
    root = Path(__file__).parents[3]
    sut, simulator = _models()
    payload = build_companion_gate_suite_preregistration(
        gate_id=gate_id,
        repo_root=root,
        created_at_unix_ms=1_800_000_000_000,
        execution_device="mps",
        sut_model=sut,
        simulator_model=simulator,
        profile_contracts=_profiles(gate_id),
    )

    validate_companion_gate_suite_preregistration(
        payload,
        repo_root=root,
        expected_profile_contracts=_profiles(gate_id),
    )
    formal = payload["formal_run"]
    assert formal["pair_count"] == 18
    assert formal["run_count"] == 18 * len(GATE_ARM_SCHEDULES[gate_id])
    source = payload["execution_source_snapshot"]
    assert source["file_count"] > 1_000
    assert len(source["tree_sha256"]) == 64
    assert payload["authorization"]["production_promotion_authorized"] is False


def test_preregistration_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "prereg.json"
    write_companion_gate_suite_preregistration(output_path=path, payload={"frozen": True})
    write_companion_gate_suite_preregistration(output_path=path, payload={"frozen": True})
    with pytest.raises(ValueError, match="immutable"):
        write_companion_gate_suite_preregistration(output_path=path, payload={"frozen": False})
