from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from volvence_zero.agent.seven_day_companion_preregistration import (
    build_seven_day_companion_preregistration,
    seven_day_source_attestation_contract,
    validate_seven_day_companion_preregistration,
    write_seven_day_companion_preregistration,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _payload() -> dict[str, object]:
    return build_seven_day_companion_preregistration(
        repo_root=REPO_ROOT,
        created_at_unix_ms=1_786_032_000_000,
    )


def _character_runtime_stack() -> dict[str, object]:
    artifact = REPO_ROOT / "pyproject.toml"
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    nested_artifact = REPO_ROOT / "docs/prd.md"
    nested_digest = hashlib.sha256(nested_artifact.read_bytes()).hexdigest()
    return {
        "mode": "base+common-adapter+character-package",
        "vertical": "zhang_wuji",
        "selected_character_id": "zhang-wuji",
        "wiring_level": "active",
        "sut_model_family": "qwen",
        "sut_max_new_tokens": 96,
        "common_adapter": {
            "locator": "pyproject.toml",
            "sha256": digest,
            "bundle_id": "common-bundle-v1",
            "common_adapter_version": "common-v1",
            "compatibility_fingerprint": "compat-v1",
            "base_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "base_model_weights_sha256": "1" * 64,
        },
        "character_manifests": [
            {
                "locator": "pyproject.toml",
                "sha256": digest,
                "package_id": "character-manifest-v1",
                "character_id": "zhang-wuji",
                "prefix_package_id": "character-prefix-v1",
                "artifact_files": [
                    {
                        "locator": "docs/prd.md",
                        "sha256": nested_digest,
                    }
                ],
            }
        ],
    }


def test_preregistration_freezes_six_scenarios_six_arms_and_real_models() -> None:
    payload = _payload()
    assert len(payload["scenario_ids"]) == 6
    assert payload["formal_run"]["run_count"] == 36
    assert payload["formal_run"]["session_count"] == 252
    assert payload["formal_run"]["exchange_count"] == 1260
    assert payload["source_requirements"]["deterministic_fake_allowed_in_formal"] is False
    assert payload["authorization"]["production_promotion_authorized"] is False
    source_snapshot = payload["execution_source_snapshot"]
    assert source_snapshot["file_count"] > len(payload["code_manifest"])
    assert len(source_snapshot["tree_sha256"]) == 64
    assert "packages/*/src" in source_snapshot["roots"]
    assert "scripts/run_gate811_simulated_capture.py" in source_snapshot["roots"]
    assert "scripts/companion_test_plan_common.py" in source_snapshot["roots"]
    assert "scripts/run_seven_day_companion_test_plan.py" in source_snapshot["roots"]
    assert "scripts/run_seven_day_companion_test_plan.py" in payload["code_manifest"]
    validate_seven_day_companion_preregistration(
        payload,
        repo_root=REPO_ROOT,
    )


def test_preregistration_drift_fails_loudly() -> None:
    payload = deepcopy(_payload())
    payload["minimum_effects"]["callback_hit_rate_gain"] = 0.0
    with pytest.raises(ValueError, match="drift"):
        validate_seven_day_companion_preregistration(
            payload,
            repo_root=REPO_ROOT,
        )


def test_v2_preregistration_freezes_active_base_adapter_character_stack() -> None:
    payload = build_seven_day_companion_preregistration(
        repo_root=REPO_ROOT,
        created_at_unix_ms=1_786_032_000_000,
        runtime_stack=_character_runtime_stack(),
    )

    assert payload["schema_version"] == "seven-day-companion-simulated.v2"
    assert payload["formal_models"]["sut"]["model_id"] == ("Qwen/Qwen2.5-1.5B-Instruct")
    assert payload["formal_models"]["simulator"]["model_family"] == "smollm"
    assert "docs/prd.md" in payload["execution_source_snapshot"]["roots"]
    source = seven_day_source_attestation_contract(payload)
    assert source["common_adapter_bundle_id"] == "common-bundle-v1"
    assert source["character_id"] == "zhang-wuji"
    assert source["character_wiring_level"] == "active"
    assert len(source["model_and_adapter_fingerprint"]) == 64
    validate_seven_day_companion_preregistration(
        payload,
        repo_root=REPO_ROOT,
    )


def test_execution_source_tree_drift_fails_loudly() -> None:
    payload = deepcopy(_payload())
    payload["execution_source_snapshot"]["tree_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="drift"):
        validate_seven_day_companion_preregistration(
            payload,
            repo_root=REPO_ROOT,
        )


def test_preregistration_is_immutable_once_written(tmp_path: Path) -> None:
    path = tmp_path / "prereg.json"
    digest = write_seven_day_companion_preregistration(
        payload=_payload(),
        output_path=path,
    )
    assert len(digest) == 64
    with pytest.raises(FileExistsError, match="immutable"):
        write_seven_day_companion_preregistration(
            payload=_payload(),
            output_path=path,
        )
