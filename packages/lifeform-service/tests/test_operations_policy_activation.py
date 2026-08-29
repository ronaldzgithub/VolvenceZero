from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from lifeform_domain_operations import OperationsBrainController
from lifeform_service.app import create_app
from lifeform_service.operations_brain_routes import operations_brain_controller
from lifeform_service.operations_policy_activation import (
    OPERATIONS_BRAIN_ENVIRONMENT_FIELD,
    OPERATIONS_BRAIN_WIRING_FIELD,
    OPERATIONS_POLICY_ACTIVATION_RECEIPT_ID_FIELD,
    OPERATIONS_POLICY_BUNDLE_DIR_FIELD,
    OperationsPolicyActivationError,
    build_operations_brain_controller_from_env,
    load_operations_policy_activation_bundle,
)
from lifeform_service.verticals import _try_operations
from volvence_zero.runtime import WiringLevel


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_BUNDLE_DIR = _REPOSITORY_ROOT / "artifacts" / "operations_brain" / "operations_policy_gate_20260830"
_RECEIPT_ID = "operations-policy-activation:f13c21df7c5c9ebeacac935c9aa69b198e57b3d5047512cec8e671cb9f94d161"


def _active_environment(bundle_dir: Path = _BUNDLE_DIR) -> dict[str, str]:
    return {
        OPERATIONS_BRAIN_ENVIRONMENT_FIELD: "staging",
        OPERATIONS_BRAIN_WIRING_FIELD: "active",
        OPERATIONS_POLICY_BUNDLE_DIR_FIELD: str(bundle_dir),
        OPERATIONS_POLICY_ACTIVATION_RECEIPT_ID_FIELD: _RECEIPT_ID,
    }


def _copy_bundle(tmp_path: Path) -> Path:
    destination = tmp_path / "operations-policy-bundle"
    shutil.copytree(_BUNDLE_DIR, destination)
    return destination


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_loads_exact_reviewed_operations_activation_bundle() -> None:
    bundle = load_operations_policy_activation_bundle(
        _BUNDLE_DIR,
        expected_activation_receipt_id=_RECEIPT_ID,
    )

    assert bundle.activation_receipt.activation_receipt_id == _RECEIPT_ID
    assert bundle.activation_receipt.activation_scope == "autocompany_staging"
    assert bundle.candidate_checkpoint.update_count == 720
    assert bundle.benchmark_report.evidence_scope == "deterministic_simulation"
    assert bundle.benchmark_report.production_default_changed is False


def test_rejects_raw_artifact_digest_drift(tmp_path: Path) -> None:
    bundle_dir = _copy_bundle(tmp_path)
    with (bundle_dir / "scenario_set.json").open("ab") as stream:
        stream.write(b"\n")

    with pytest.raises(
        OperationsPolicyActivationError,
        match="raw SHA-256 mismatch: scenario_set.json",
    ):
        load_operations_policy_activation_bundle(
            bundle_dir,
            expected_activation_receipt_id=_RECEIPT_ID,
        )


def test_manifest_digest_cannot_authorize_changed_protocol(tmp_path: Path) -> None:
    bundle_dir = _copy_bundle(tmp_path)
    scenario_path = bundle_dir / "scenario_set.json"
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    scenario["utility_contract"]["correct_candidate"] = 0.99
    _write_json(scenario_path, scenario)

    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["file_sha256"]["scenario_set.json"] = hashlib.sha256(scenario_path.read_bytes()).hexdigest()
    _write_json(manifest_path, manifest)

    with pytest.raises(
        OperationsPolicyActivationError,
        match="scenario set differs from the reviewed protocol",
    ):
        load_operations_policy_activation_bundle(
            bundle_dir,
            expected_activation_receipt_id=_RECEIPT_ID,
        )


def test_active_startup_requires_staging_exact_bundle_and_receipt_pin() -> None:
    controller = build_operations_brain_controller_from_env(_active_environment())
    assert isinstance(controller, OperationsBrainController)

    with pytest.raises(
        OperationsPolicyActivationError,
        match="limited to the staging environment",
    ):
        build_operations_brain_controller_from_env(
            {
                **_active_environment(),
                OPERATIONS_BRAIN_ENVIRONMENT_FIELD: "production",
            }
        )

    missing_pin = _active_environment()
    del missing_pin[OPERATIONS_POLICY_ACTIVATION_RECEIPT_ID_FIELD]
    with pytest.raises(
        OperationsPolicyActivationError,
        match=OPERATIONS_POLICY_ACTIVATION_RECEIPT_ID_FIELD,
    ):
        build_operations_brain_controller_from_env(missing_pin)

    with pytest.raises(
        OperationsPolicyActivationError,
        match="does not match the deployment pin",
    ):
        build_operations_brain_controller_from_env(
            {
                **_active_environment(),
                OPERATIONS_POLICY_ACTIVATION_RECEIPT_ID_FIELD: ("operations-policy-activation:" + "0" * 64),
            }
        )


@pytest.mark.parametrize("wiring", ["disabled", "shadow"])
def test_non_active_startup_never_requires_a_bundle(wiring: str) -> None:
    controller = build_operations_brain_controller_from_env(
        {
            OPERATIONS_BRAIN_ENVIRONMENT_FIELD: "production",
            OPERATIONS_BRAIN_WIRING_FIELD: wiring,
        }
    )
    assert isinstance(controller, OperationsBrainController)


def test_app_uses_explicit_operations_controller_without_reading_env(
    monkeypatch,
) -> None:
    monkeypatch.setenv(OPERATIONS_BRAIN_ENVIRONMENT_FIELD, "invalid")
    controller = OperationsBrainController()
    vertical = _try_operations()
    assert vertical is not None

    app = create_app(
        vertical=vertical,
        operations_brain_controller=controller,
    )

    assert operations_brain_controller(app) is controller


def test_factory_passes_verified_active_lineage_to_controller(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _CapturedController:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "lifeform_service.operations_policy_activation.OperationsBrainController",
        _CapturedController,
    )

    build_operations_brain_controller_from_env(_active_environment())

    assert captured["policy_wiring_level"] is WiringLevel.ACTIVE
    assert captured["policy_checkpoint_seed"].update_count == 720
    assert captured["activation_receipt"].activation_receipt_id == _RECEIPT_ID
