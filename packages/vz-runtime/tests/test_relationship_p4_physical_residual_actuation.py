from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import pytest

from volvence_zero.agent import relationship_p4_physical_residual_actuation as lane
from volvence_zero.steering_contracts import SteeringArtifactBundle
from volvence_zero.substrate import (
    FeatureSignal,
    OpenWeightRuntimeCapture,
    ResidualActivation,
    ResidualControlApplication,
    SubstrateSnapshot,
    SurfaceKind,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_FIT_ROOT = _REPOSITORY_ROOT / "artifacts" / "relationship_lab" / "p4_windows_cuda_steering_fit_qwen25_15b_20260822"
_CAMPAIGN_MANIFEST = (
    _REPOSITORY_ROOT
    / "artifacts"
    / "relationship_lab"
    / "p4_windows_cuda_steering_fit_qwen25_15b_20260822_campaign_manifest.json"
)


@dataclass(frozen=True)
class _FakeAttestation:
    payload: dict[str, Any]

    @property
    def attestation_id(self) -> str:
        return self.payload["attestation_id"]

    def to_payload(self) -> dict[str, Any]:
        return {key: value for key, value in self.payload.items() if key != "attestation_id"}


class _FakeStrictRuntime:
    model_id = lane._MODEL_ID
    is_frozen = True
    fallback_active = False
    capture_source = "real"
    fail_on_truncation = True

    def __init__(
        self,
        *,
        bundle: SteeringArtifactBundle,
        attestation: dict[str, Any],
    ) -> None:
        self._bundle = bundle
        self._execution_attestation = _FakeAttestation(attestation)
        self.capture_calls = 0
        self.direct_calls = 0
        self.generate_calls = 0

    @property
    def loaded_base_model_weights_sha256(self) -> str:
        return lane._MODEL_WEIGHTS_SHA256

    @property
    def execution_attestation(self) -> _FakeAttestation:
        return self._execution_attestation

    def capture(self, *, source_text: str) -> OpenWeightRuntimeCapture:
        self.capture_calls += 1
        reader = self._bundle.reader
        label = next(item for item in reader.class_labels if f"Objective: {item}." in source_text)
        label_index = reader.class_labels.index(label)
        residual = tuple(
            reader.feature_mean[index] + reader.feature_scale[index] * reader.weights[index][label_index] * 100.0
            for index in range(reader.residual_width)
        )
        return OpenWeightRuntimeCapture(
            token_logits=(0.7, 0.3),
            feature_surface=(
                FeatureSignal(
                    name="fake_strict_capture",
                    values=(1.0,),
                    source="test-only-fake-strict-runtime",
                ),
            ),
            residual_activations=(
                ResidualActivation(
                    layer_index=lane._LAYER_INDEX,
                    activation=residual,
                    step=0,
                ),
            ),
            residual_sequence=(),
            description="test-only deterministic strict capture",
        )

    def apply_direct_residual_delta(
        self,
        *,
        source_text: str,
        substrate_snapshot: SubstrateSnapshot,
        layer_index: int,
        residual_delta: tuple[float, ...],
    ) -> ResidualControlApplication:
        del source_text
        self.direct_calls += 1
        assert layer_index == lane._LAYER_INDEX
        source = substrate_snapshot.residual_activations[0].activation
        output = tuple(value + delta for value, delta in zip(source, residual_delta, strict=True))
        norm = math.sqrt(math.fsum(value * value for value in residual_delta))
        mean_abs = sum(abs(value) for value in residual_delta) / len(residual_delta)
        effect = (
            min(1.0, norm),
            min(1.0, mean_abs),
            min(1.0, abs(math.fsum(residual_delta)) / len(residual_delta)),
        )
        applied = SubstrateSnapshot(
            model_id=self.model_id,
            is_frozen=True,
            surface_kind=SurfaceKind.RESIDUAL_STREAM,
            token_logits=substrate_snapshot.token_logits,
            feature_surface=substrate_snapshot.feature_surface,
            residual_activations=(
                ResidualActivation(
                    layer_index=layer_index,
                    activation=output,
                    step=0,
                ),
            ),
            residual_sequence=(),
            unavailable_fields=(),
            description="test-only direct residual application",
        )
        return ResidualControlApplication(
            applied_snapshot=applied,
            downstream_effect=effect,
            control_energy=mean_abs,
            backend_name=f"transformers-direct-steering:{self.model_id}",
            description="test-only canonical direct hook stand-in",
        )

    def generate(self, **kwargs: object) -> object:
        del kwargs
        self.generate_calls += 1
        raise AssertionError("physical-actuation preflight must not generate")


class _NoEffectFakeStrictRuntime(_FakeStrictRuntime):
    def apply_direct_residual_delta(
        self,
        *,
        source_text: str,
        substrate_snapshot: SubstrateSnapshot,
        layer_index: int,
        residual_delta: tuple[float, ...],
    ) -> ResidualControlApplication:
        application = super().apply_direct_residual_delta(
            source_text=source_text,
            substrate_snapshot=substrate_snapshot,
            layer_index=layer_index,
            residual_delta=residual_delta,
        )
        if not any(value != 0.0 for value in residual_delta):
            return application
        return ResidualControlApplication(
            applied_snapshot=substrate_snapshot,
            downstream_effect=(0.0, 0.0, 0.0),
            control_energy=application.control_energy,
            backend_name=application.backend_name,
            description="test-only suppressed nonzero physical effect",
        )


@pytest.fixture(scope="module")
def authenticated_input() -> lane._AuthenticatedFitInput:
    return lane._authenticate_fit_input(
        input_fit_root=_FIT_ROOT,
        campaign_manifest_path=_CAMPAIGN_MANIFEST,
    )


@pytest.fixture(scope="module")
def physical_artifact(
    tmp_path_factory: pytest.TempPathFactory,
    authenticated_input: lane._AuthenticatedFitInput,
) -> Any:
    output = tmp_path_factory.mktemp("p4-physical") / "artifact"
    runtime = _FakeStrictRuntime(
        bundle=authenticated_input.bundle,
        attestation=dict(authenticated_input.execution_attestation),
    )
    patcher = pytest.MonkeyPatch()
    patcher.setattr(lane, "_build_strict_runtime", lambda: runtime)
    try:
        result = lane.run_relationship_p4_physical_residual_actuation(
            output_dir=output,
            input_fit_root=_FIT_ROOT,
            campaign_manifest_path=_CAMPAIGN_MANIFEST,
        )
        yield output, result, runtime
    finally:
        patcher.undo()


def test_protocol_freezes_balanced_unique_heldout_prompt_set_without_torch() -> None:
    protocol = lane.load_relationship_p4_physical_actuation_protocol()
    prompts = lane._build_frozen_prompt_set(protocol)

    assert len(prompts) == 68
    assert len({row.prompt for row in prompts}) == 68
    assert protocol.prompt_set_sha256 == lane._sha256_bytes(lane._canonical_bytes([asdict(row) for row in prompts]))
    counts = {label: sum(row.expected_subgoal_class == label for row in prompts) for label in lane._CLASS_LABELS}
    assert counts == lane._DATASET_RECIPE["class_quotas"]
    assert "torch" not in sys.modules


def test_complete_fit_root_and_campaign_are_authenticated_without_gpu(
    authenticated_input: lane._AuthenticatedFitInput,
) -> None:
    assert authenticated_input.fit_manifest["artifact_id"] == lane._INPUT_ARTIFACT_ID
    assert authenticated_input.campaign_manifest["campaign_id"] == lane._INPUT_CAMPAIGN_ID
    assert authenticated_input.bundle.sensor_off_executor is not None
    assert "torch" not in sys.modules


def test_standalone_bundle_and_campaign_hash_drift_are_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="complete P4.6-fit artifact root"):
        lane._authenticate_fit_input(
            input_fit_root=_FIT_ROOT / "steering_artifact_bundle.json",
            campaign_manifest_path=_CAMPAIGN_MANIFEST,
        )

    campaign_copy = tmp_path / "campaign.json"
    campaign_copy.write_bytes(_CAMPAIGN_MANIFEST.read_bytes() + b" ")
    with pytest.raises(ValueError, match="campaign manifest raw SHA-256"):
        lane._authenticate_fit_input(
            input_fit_root=_FIT_ROOT,
            campaign_manifest_path=campaign_copy,
        )


def test_four_arm_run_uses_canonical_noop_direct_hook_and_publishes_pass(
    physical_artifact: tuple[
        Path,
        lane.RelationshipP4PhysicalActuationRunResult,
        _FakeStrictRuntime,
    ],
) -> None:
    output, result, runtime = physical_artifact

    assert result.preflight_passed is True
    assert result.verdict.endswith("passed_development_only")
    assert runtime.capture_calls == 136
    assert runtime.direct_calls == 408
    assert runtime.generate_calls == 0
    report = _read_json(output / lane._REPORT_FILE)
    assert report["metrics"]["runtime_forward_invocation_count"] == 544
    assert report["metrics"]["conditional_code_row_count"] == 8
    assert report["metrics"]["sensor_off_code_row_count"] == 1
    assert all(report["checks"].values())
    assert report["evidence_firewall"] == lane._EVIDENCE_FIREWALL
    assert report["execution"]["applied_gpu_residual_provenance"].endswith("not_bundle_derivable")
    assert "not a pure sensor ablation" in report["control_interpretation"]

    receipt_path = next((output / lane._RECEIPT_DIRECTORY).glob("*.json"))
    receipt = _read_json(receipt_path)
    arms = {row["arm"]: row for row in receipt["arms"]}
    noop = arms["strict_noop"]
    assert noop["executor_invoked"] is True
    assert noop["direct_hook_invoked"] is True
    assert noop["canonical_executor_shadow_hook_executed"] is False
    assert noop["action"] == "noop"
    assert noop["application_mode"] == "shadow-noop"
    assert noop["output_residual"]["sha256"] == arms["raw_no_intervention"]["output_residual"]["sha256"]
    assert [row["runtime_api"] for row in receipt["runtime_forward_invocation_ledger"]] == [
        "runtime.capture",
        "runtime.apply_direct_residual_delta",
        "runtime.apply_direct_residual_delta",
        "runtime.apply_direct_residual_delta",
    ]


def test_offline_validator_recomputes_artifact_without_torch(
    physical_artifact: tuple[
        Path,
        lane.RelationshipP4PhysicalActuationRunResult,
        _FakeStrictRuntime,
    ],
) -> None:
    output, original, _ = physical_artifact
    validated = lane.validate_relationship_p4_physical_residual_actuation(
        output_dir=output,
        input_fit_root=_FIT_ROOT,
        campaign_manifest_path=_CAMPAIGN_MANIFEST,
    )

    assert validated == original
    assert "torch" not in sys.modules


def test_offline_validator_rejects_receipt_tampering(
    tmp_path: Path,
    physical_artifact: tuple[
        Path,
        lane.RelationshipP4PhysicalActuationRunResult,
        _FakeStrictRuntime,
    ],
) -> None:
    source, _, _ = physical_artifact
    copied = tmp_path / "tampered"
    shutil.copytree(source, copied)
    receipt_path = next((copied / lane._RECEIPT_DIRECTORY).glob("*.json"))
    receipt_path.write_bytes(receipt_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="byte_count|sha256"):
        lane.validate_relationship_p4_physical_residual_actuation(
            output_dir=copied,
            input_fit_root=_FIT_ROOT,
            campaign_manifest_path=_CAMPAIGN_MANIFEST,
        )


def test_offline_validator_rejects_coherently_resealed_delta_and_output(
    tmp_path: Path,
    physical_artifact: tuple[
        Path,
        lane.RelationshipP4PhysicalActuationRunResult,
        _FakeStrictRuntime,
    ],
) -> None:
    source, _, _ = physical_artifact
    copied = tmp_path / "coherent-delta-reseal"
    shutil.copytree(source, copied)
    receipt_paths = tuple((copied / lane._RECEIPT_DIRECTORY).glob("*.json"))
    first = _read_json(receipt_paths[0])
    sample_id = first["sample_id"]
    matching_paths = [path for path in receipt_paths if _read_json(path)["sample_id"] == sample_id]
    assert len(matching_paths) == 2

    for receipt_path in matching_paths:
        receipt = _read_json(receipt_path)
        arms = {row["arm"]: row for row in receipt["arms"]}
        raw, _ = lane._decode_vector(
            arms["raw_no_intervention"]["output_residual"],
            label="test raw",
        )
        delta, _ = lane._decode_vector(
            arms["conditional_always_on"]["delta"],
            label="test conditional delta",
        )
        replacement = tuple(value * 0.5 for value in delta)
        replacement_norm = lane._vector_norm(replacement)
        replacement_energy = sum(abs(value) for value in replacement) / len(replacement)
        conditional = arms["conditional_always_on"]
        conditional["delta"] = lane._encode_vector(replacement)
        conditional["output_residual"] = lane._encode_vector(
            tuple(
                source_value + delta_value
                for source_value, delta_value in zip(
                    raw,
                    replacement,
                    strict=True,
                )
            )
        )
        conditional["control_norm"] = replacement_norm
        conditional["control_energy"] = replacement_energy
        conditional["downstream_effect"] = [
            min(1.0, replacement_norm),
            min(1.0, replacement_energy),
            min(1.0, abs(math.fsum(replacement)) / len(replacement)),
        ]
        _reseal_receipt(copied, receipt_path, receipt)

    with pytest.raises(
        ValueError,
        match="offline owner-recomputed conditional_always_on",
    ):
        lane.validate_relationship_p4_physical_residual_actuation(
            output_dir=copied,
            input_fit_root=_FIT_ROOT,
            campaign_manifest_path=_CAMPAIGN_MANIFEST,
        )


def test_offline_validator_rejects_coherently_resealed_invocation_ledger(
    tmp_path: Path,
    physical_artifact: tuple[
        Path,
        lane.RelationshipP4PhysicalActuationRunResult,
        _FakeStrictRuntime,
    ],
) -> None:
    source, _, _ = physical_artifact
    copied = tmp_path / "coherent-ledger-reseal"
    shutil.copytree(source, copied)
    receipt_path = next((copied / lane._RECEIPT_DIRECTORY).glob("*.json"))
    receipt = _read_json(receipt_path)
    receipt["runtime_forward_invocation_ledger"][1]["runtime_api"] = "runtime.capture"
    _reseal_receipt(copied, receipt_path, receipt)

    with pytest.raises(ValueError, match="runtime forward invocation ledger"):
        lane.validate_relationship_p4_physical_residual_actuation(
            output_dir=copied,
            input_fit_root=_FIT_ROOT,
            campaign_manifest_path=_CAMPAIGN_MANIFEST,
        )


def test_safe_threshold_failure_publishes_stop_without_retuning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authenticated_input: lane._AuthenticatedFitInput,
) -> None:
    output = tmp_path / "failed-stop"
    runtime = _NoEffectFakeStrictRuntime(
        bundle=authenticated_input.bundle,
        attestation=dict(authenticated_input.execution_attestation),
    )
    monkeypatch.setattr(lane, "_build_strict_runtime", lambda: runtime)

    result = lane.run_relationship_p4_physical_residual_actuation(
        output_dir=output,
        input_fit_root=_FIT_ROOT,
        campaign_manifest_path=_CAMPAIGN_MANIFEST,
    )

    assert output.is_dir()
    assert result.preflight_passed is False
    assert result.verdict == "failed_stop_no_retuning"
    assert runtime.capture_calls + runtime.direct_calls == 544
    report = _read_json(output / lane._REPORT_FILE)
    assert report["failure_retuning_performed"] is False
    assert report["evidence_firewall"]["failure_retuning_performed"] is False
    assert report["verdict"] == "failed_stop_no_retuning"


def test_run_is_create_only_before_runtime_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    runtime_called = False

    def forbidden_runtime() -> object:
        nonlocal runtime_called
        runtime_called = True
        raise AssertionError("runtime must not be constructed")

    monkeypatch.setattr(lane, "_build_strict_runtime", forbidden_runtime)
    with pytest.raises(FileExistsError, match="create-only"):
        lane.run_relationship_p4_physical_residual_actuation(
            output_dir=output,
            input_fit_root=_FIT_ROOT,
            campaign_manifest_path=_CAMPAIGN_MANIFEST,
        )
    assert runtime_called is False


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _reseal_receipt(
    artifact_root: Path,
    receipt_path: Path,
    receipt: dict[str, Any],
) -> Path:
    old_relative_path = receipt_path.relative_to(artifact_root).as_posix()
    receipt_core = {key: value for key, value in receipt.items() if key != "receipt_id"}
    receipt["receipt_id"] = lane._sha256_bytes(lane._canonical_bytes(receipt_core))
    payload = lane._canonical_bytes(receipt)
    new_relative_path = f"{lane._RECEIPT_DIRECTORY}/{receipt['receipt_id']}.json"
    new_path = artifact_root / new_relative_path
    receipt_path.unlink()
    new_path.write_bytes(payload)

    manifest_path = artifact_root / lane._MANIFEST_FILE
    manifest = _read_json(manifest_path)
    matching = [row for row in manifest["files"] if row["path"] == old_relative_path]
    assert len(matching) == 1
    matching[0].update(
        {
            "path": new_relative_path,
            "byte_count": len(payload),
            "sha256": lane._sha256_bytes(payload),
        }
    )
    manifest["files"] = sorted(
        manifest["files"],
        key=lambda row: row["path"],
    )
    manifest_core = {key: value for key, value in manifest.items() if key != "artifact_id"}
    manifest["artifact_id"] = lane._sha256_bytes(lane._canonical_bytes(manifest_core))
    manifest_path.write_bytes(lane._canonical_bytes(manifest))
    return new_path
