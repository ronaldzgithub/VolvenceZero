"""Development-only Windows/CUDA physical residual-actuation preflight.

The unique owner in this module is a thin ``vz-runtime`` evidence
orchestrator.  It authenticates the complete P4.6-fit artifact root before a
GPU runtime is constructed, then exercises the existing substrate capture,
steering sensor, steering executor, and direct residual hook without adding a
runtime slot or product wiring.

The offline validator deliberately keeps all GPU/runtime imports lazy.  It
authenticates the frozen input again, decodes every retained float vector, and
recomputes the stopping checks, report, manifest, and verdict without importing
``torch`` or probing CUDA.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import pathlib
import shutil
import struct
import tempfile
from typing import Any, Callable, Mapping, Sequence


P4_PHYSICAL_ACTUATION_PROTOCOL_SCHEMA_VERSION = "relationship-p4-windows-cuda-physical-residual-actuation.v1"
P4_PHYSICAL_ACTUATION_RECEIPT_SCHEMA_VERSION = "relationship-p4-physical-residual-actuation-receipt.v1"
P4_PHYSICAL_ACTUATION_REPORT_SCHEMA_VERSION = "relationship-p4-physical-residual-actuation-report.v1"
P4_PHYSICAL_ACTUATION_MANIFEST_SCHEMA_VERSION = "relationship-p4-physical-residual-actuation-manifest.v1"

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
_REPOSITORY_ROOT = _PACKAGE_ROOT.parents[4]
_DEFAULT_PROTOCOL_PATH = (
    _PACKAGE_ROOT / "protocols" / "relationship_p4_windows_cuda_physical_residual_actuation_v1.json"
)
_REPORT_FILE = "physical_residual_actuation_report.json"
_ATTESTATION_FILE = "live_execution_attestation.json"
_MANIFEST_FILE = "manifest.json"
_RECEIPT_DIRECTORY = "receipts"
_SOURCE_HASH_MODE = "utf8_lf_canonical_v1"
_VECTOR_ENCODING = "float64-big-endian-base64.v1"

_INPUT_ARTIFACT_ID = "57b59b269ecc5cf3f15abf3e16c3a8a03a9e9c74dd8025647203bdd16edcbe04"
_INPUT_BUNDLE_SHA256 = "6d6047bdd1e3996df906f5606fb9bf4e9caa9a2f57b6b3f4f288e96df1573249"
_INPUT_MANIFEST_SHA256 = "bd40a8103fcfd36a8b3fc78f04409c240b132775bbaf0c7a63c9389688d61415"
_INPUT_CAMPAIGN_ID = "67760e78bce6c11df862b3cfdca1a20c5e7f2a0114ce9c312af4bc9d8414ef31"
_INPUT_CAMPAIGN_MANIFEST_SHA256 = "41ed7a2d3f4e7cf3d3abbdc25f5d68be20173f38912d3a1caa9688b47640007b"
_INPUT_FIT_PROTOCOL_ID = "b6e1d79a5087945c4be6bf8d4f1e2f6535cdbedeca56ae0bd1af53a56f3def8d"
_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
_MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
_MODEL_WEIGHTS_SHA256 = "fb8c44c48b8359fdd306cdc5f473d7c04d88955013f0dd8549f266e248194da4"
_EXECUTION_ASSETS_SHA256 = "bbb5446f8d802b437c2fc7e2cefcdabb996bbd4bc657fe155ea015d30a841bb0"
_EXECUTION_PROFILE_ID = "3be84d866afbda07cf80dee277d89cdc0e366ce545bf7e97f015cf8afcbfe21a"
_EXECUTION_ATTESTATION_ID = "9a33a698b95d923d6a4e82b64471213d529b0cbbf6a30ca24644860211e6dde1"
_LAYER_INDEX = 20
_RESIDUAL_WIDTH = 1536
_STEERING_RANK = 8
_CONTROL_NORM_CAP_RATIO = 0.25
_CLASS_LABELS = (
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "black",
    "white",
)
_ARM_ORDER = (
    "raw_no_intervention",
    "strict_noop",
    "conditional_always_on",
    "sensor_off_always_on",
)
_PROMPT_COUNT = 68
_REPEAT_COUNT = 2
_RECEIPT_COUNT = _PROMPT_COUNT * _REPEAT_COUNT
_ARM_EVALUATION_COUNT = _RECEIPT_COUNT * len(_ARM_ORDER)
_RUNTIME_FORWARD_INVOCATION_COUNT = _RECEIPT_COUNT * 4

_RUNTIME_FORWARD_INVOCATION_APIS = {
    "raw_no_intervention": "runtime.capture",
    "strict_noop": "runtime.apply_direct_residual_delta",
    "conditional_always_on": "runtime.apply_direct_residual_delta",
    "sensor_off_always_on": "runtime.apply_direct_residual_delta",
}
_RUNTIME_FORWARD_INVOCATION_SEMANTICS = (
    "Each ledger row is appended by the vz-runtime orchestrator only after the "
    "named forward-bearing runtime API returns successfully. The pinned strict "
    "backend and adapter sources establish one frozen-model forward per listed "
    "API return; the ledger is an observed invocation count, not a hardware "
    "performance counter."
)

_PROFILE_FACTS = {
    "preset_name": "windows-cuda-cudnn-sdpa-cached-strict.v1",
    "platform_system": "Windows",
    "device_type": "cuda",
    "attention_implementation": "sdpa",
    "sdpa_backend": "cudnn",
    "sdpa_backend_policy": "exclusive-cudnn",
    "sdpa_backend_exclusive": True,
    "generation_use_cache": True,
    "generation_capture_strategy": "first-full-prompt-set-once",
    "capture_failure_mode": "raise",
    "context_window_tokens": 32768,
    "local_files_only": True,
    "fallback_mode": "deny",
    "fail_on_truncation": True,
    "model_dtype": "bfloat16",
    "require_verified_model_revision": True,
    "require_model_weights_sha256": True,
    "require_execution_assets_sha256": True,
    "require_generation_chat_template": True,
}

_DATASET_RECIPE = {
    "seed": 20260822,
    "objective_count": 8,
    "corridor_count": 2,
    "extra_edge_probability": 0.35,
    "train_route_count": 128,
    "heldout_route_count": 128,
    "train_lengths": [3, 4],
    "heldout_lengths": [3, 4],
    "prompt_count": _PROMPT_COUNT,
    "repeat_count": _REPEAT_COUNT,
    "class_labels": list(_CLASS_LABELS),
    "class_quotas": {
        "red": 9,
        "blue": 9,
        "green": 9,
        "yellow": 9,
        "orange": 8,
        "purple": 8,
        "black": 8,
        "white": 8,
    },
    "selection": "first_unique_heldout_row_per_class_in_owner_order",
    "prompt_surface": ("case_source_plus_subgoal_revealed_text_plus_action_suffix"),
}

_ARM_SEMANTICS = {
    "raw_no_intervention": {
        "sensor_invoked": True,
        "executor_invoked": False,
        "direct_hook_invoked": False,
        "action": "raw",
        "executor_artifact": "none",
    },
    "strict_noop": {
        "sensor_invoked": True,
        "executor_invoked": True,
        "direct_hook_invoked": True,
        "action": "noop",
        "executor_artifact": "conditional",
    },
    "conditional_always_on": {
        "sensor_invoked": True,
        "executor_invoked": True,
        "direct_hook_invoked": True,
        "action": "steer",
        "executor_artifact": "conditional",
    },
    "sensor_off_always_on": {
        "sensor_invoked": True,
        "executor_invoked": True,
        "direct_hook_invoked": True,
        "action": "steer",
        "executor_artifact": "matched_sensor_off",
    },
}

_STOPPING_CHECK_NAMES = (
    "prompt_set_unique_and_class_balanced",
    "raw_noop_output_residual_exact",
    "strict_noop_delta_and_effect_zero",
    "strict_noop_canonical_executor_and_direct_hook",
    "conditional_delta_finite_nonzero_capped",
    "conditional_observed_applied_residual_differs",
    "sensor_off_delta_finite_nonzero_capped",
    "sensor_off_observed_applied_residual_differs",
    "conditional_sensor_off_observed_hashes_differ",
    "repeat_outputs_exact",
    "conditional_uses_at_least_two_code_rows",
    "sensor_off_condition_code_rows_identical",
    "runtime_forward_invocation_ledger_exact",
)

_CONTROL_INTERPRETATION = (
    "The sensor-off arm consumes the same frozen substrate snapshot and the "
    "same published belief, but applies the separately fitted matched-capacity "
    "unconditional executor. It is not a pure sensor ablation because its "
    "U/V factors were fitted separately from the conditional executor."
)
_CLAIM_BOUNDARY = (
    "This development-only preflight can establish only that the exact "
    "Qwen2.5-1.5B frozen runtime runner recorded successful bounded layer-20 "
    "direct-hook invocations and distinct post-hook residual observations from "
    "the authenticated P4.6-fit reader/executors on a frozen synthetic ETA "
    "proxy set. The offline validator recomputes owner beliefs and requested "
    "deltas, but the applied GPU residual is not bundle-derivable and is "
    "authenticated only as an observation in the pinned content-addressed run "
    "receipt. It performs no user-visible generation and makes no relationship-"
    "semantic, PE-learning, 32K-context, independent-subject, long-"
    "companionship, multi-session, production ACTIVE, complete Appendable, "
    "Readable, Learnable, Steerable, or four-axis claim."
)

_EVIDENCE_FIREWALL = {
    "synthetic_proxy_only": True,
    "relationship_semantics_claimed": False,
    "user_visible_generation_performed": False,
    "prediction_error_learning_performed": False,
    "long_context_32k_exercised": False,
    "independent_subject_count": 0,
    "long_companionship_exercised": False,
    "multi_session_exercised": False,
    "product_wiring_changed": False,
    "production_active_authorized": False,
    "formal_evidence_authorized": False,
    "four_capability_claim_authorized": False,
    "applied_gpu_residual_bundle_derivable_claimed": False,
    "failure_retuning_performed": False,
}

_CRITICAL_SOURCE_PATHS = (
    "packages/companion-standard/src/companion_standard/kernel.py",
    "packages/vz-contracts/src/volvence_zero/runtime/kernel.py",
    "packages/vz-contracts/src/volvence_zero/steering_contracts.py",
    "packages/vz-cognition/src/volvence_zero/steering_sensor.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_conditional_steering_screen.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_conflict_instrument.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_proof_benchmark.py",
    "packages/vz-runtime/src/volvence_zero/agent/relationship_p4_physical_residual_actuation.py",
    "packages/vz-runtime/src/volvence_zero/agent/relationship_p4_steering_artifact_fit.py",
    "packages/vz-substrate/src/volvence_zero/steering_executor.py",
    "packages/vz-substrate/src/volvence_zero/substrate/adapter.py",
    "packages/vz-substrate/src/volvence_zero/substrate/residual_backend.py",
    "packages/vz-substrate/src/volvence_zero/substrate/residual_contracts.py",
    "scripts/run_relationship_lab_p4_physical_residual_actuation.py",
)


@dataclass(frozen=True)
class PhysicalActuationPrompt:
    sample_id: str
    case_id: str
    step_index: int
    expected_subgoal_class: str
    prompt: str
    prompt_sha256: str


@dataclass(frozen=True)
class RelationshipP4PhysicalActuationProtocol:
    protocol_id: str
    prompt_set_sha256: str
    source_sha256: tuple[tuple[str, str], ...]
    source_hash_mode: str = _SOURCE_HASH_MODE
    claim_boundary: str = _CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        _require_sha256(self.protocol_id, "protocol_id")
        _require_sha256(self.prompt_set_sha256, "prompt_set_sha256")
        if self.source_hash_mode != _SOURCE_HASH_MODE:
            raise ValueError("P4 physical-actuation source hash mode drift")
        if tuple(path for path, _ in self.source_sha256) != _CRITICAL_SOURCE_PATHS:
            raise ValueError("P4 physical-actuation critical source path drift")
        for path, digest in self.source_sha256:
            _require_relative_posix_path(path, "critical source path")
            _require_sha256(digest, f"critical source {path}")
        if self.claim_boundary != _CLAIM_BOUNDARY:
            raise ValueError("P4 physical-actuation claim boundary drift")

    def source_sha256_payload(self) -> dict[str, str]:
        return dict(self.source_sha256)


@dataclass(frozen=True)
class RelationshipP4PhysicalActuationRunResult:
    artifact_id: str
    protocol_id: str
    input_fit_artifact_id: str
    execution_attestation_id: str
    preflight_passed: bool
    verdict: str
    output_dir: pathlib.Path


@dataclass(frozen=True)
class _AuthenticatedFitInput:
    bundle: Any
    execution_attestation: Mapping[str, Any]
    fit_manifest: Mapping[str, Any]
    campaign_manifest: Mapping[str, Any]


def relationship_p4_physical_actuation_protocol_path() -> pathlib.Path:
    return _DEFAULT_PROTOCOL_PATH


def load_relationship_p4_physical_actuation_protocol(
    path: pathlib.Path | None = None,
) -> RelationshipP4PhysicalActuationProtocol:
    protocol_path = pathlib.Path(path or _DEFAULT_PROTOCOL_PATH)
    raw = _load_json_object(protocol_path)
    _validate_protocol_payload(raw)
    source_sha256 = _require_mapping(raw["source_sha256"], "source_sha256")
    return RelationshipP4PhysicalActuationProtocol(
        protocol_id=_sha256_bytes(_canonical_bytes(raw)),
        prompt_set_sha256=_require_sha256_value(
            raw["dataset"]["prompt_set_sha256"],
            "dataset.prompt_set_sha256",
        ),
        source_sha256=tuple(
            (
                source_path,
                _require_sha256_value(
                    source_sha256[source_path],
                    f"source_sha256.{source_path}",
                ),
            )
            for source_path in _CRITICAL_SOURCE_PATHS
        ),
        source_hash_mode=_require_text_value(raw["source_hash_mode"], "source_hash_mode"),
        claim_boundary=_require_text_value(raw["claim_boundary"], "claim_boundary"),
    )


def run_relationship_p4_physical_residual_actuation(
    *,
    output_dir: pathlib.Path,
    input_fit_root: pathlib.Path,
    campaign_manifest_path: pathlib.Path,
    protocol_path: pathlib.Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> RelationshipP4PhysicalActuationRunResult:
    """Run and atomically publish one frozen physical-actuation preflight."""

    output = pathlib.Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"P4 physical-actuation output is create-only and already exists: {output}")
    protocol = load_relationship_p4_physical_actuation_protocol(protocol_path)
    _verify_critical_sources(protocol)
    prompts = _build_frozen_prompt_set(protocol)
    authenticated = _authenticate_fit_input(
        input_fit_root=pathlib.Path(input_fit_root).resolve(),
        campaign_manifest_path=pathlib.Path(campaign_manifest_path).resolve(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = pathlib.Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.building."))
    published = False
    try:
        runtime = _build_strict_runtime()
        live_attestation = _attest_live_runtime(
            runtime=runtime,
            authenticated=authenticated,
        )
        receipts = asyncio.run(
            _execute_physical_actuation_preflight(
                runtime=runtime,
                bundle=authenticated.bundle,
                protocol=protocol,
                prompts=prompts,
                progress=progress,
            )
        )
        derived = _derive_metrics_and_checks(
            protocol=protocol,
            prompts=prompts,
            bundle=authenticated.bundle,
            receipts=receipts,
        )
        checks = derived["checks"]
        preflight_passed = all(checks[name] for name in _STOPPING_CHECK_NAMES)
        verdict = _verdict(preflight_passed)
        receipt_payload_bytes: dict[str, bytes] = {}
        for receipt in receipts:
            relative_path = f"{_RECEIPT_DIRECTORY}/{receipt['receipt_id']}.json"
            payload = _canonical_bytes(receipt)
            receipt_payload_bytes[relative_path] = payload
            _write_create_bytes(temporary / relative_path, payload)
        attestation_bytes = _canonical_bytes(live_attestation)
        _write_create_bytes(temporary / _ATTESTATION_FILE, attestation_bytes)
        report = _build_report_payload(
            protocol=protocol,
            authenticated=authenticated,
            derived=derived,
            live_attestation=live_attestation,
            preflight_passed=preflight_passed,
            verdict=verdict,
        )
        report_bytes = _canonical_bytes(report)
        _write_create_bytes(temporary / _REPORT_FILE, report_bytes)
        payload_bytes = {
            **receipt_payload_bytes,
            _ATTESTATION_FILE: attestation_bytes,
            _REPORT_FILE: report_bytes,
        }
        manifest_core = _build_manifest_core(
            protocol=protocol,
            checks=checks,
            preflight_passed=preflight_passed,
            verdict=verdict,
            payload_bytes=payload_bytes,
        )
        artifact_id = _sha256_bytes(_canonical_bytes(manifest_core))
        _write_create_bytes(
            temporary / _MANIFEST_FILE,
            _canonical_bytes({**manifest_core, "artifact_id": artifact_id}),
        )
        _validate_physical_actuation_artifact(
            output_dir=temporary,
            input_fit_root=pathlib.Path(input_fit_root).resolve(),
            campaign_manifest_path=pathlib.Path(campaign_manifest_path).resolve(),
            protocol=protocol,
            authenticated=authenticated,
        )
        if output.exists():
            raise FileExistsError("P4 physical-actuation output appeared during publication")
        temporary.rename(output)
        published = True
        return RelationshipP4PhysicalActuationRunResult(
            artifact_id=artifact_id,
            protocol_id=protocol.protocol_id,
            input_fit_artifact_id=_INPUT_ARTIFACT_ID,
            execution_attestation_id=_EXECUTION_ATTESTATION_ID,
            preflight_passed=preflight_passed,
            verdict=verdict,
            output_dir=output,
        )
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)


def validate_relationship_p4_physical_residual_actuation(
    *,
    output_dir: pathlib.Path,
    input_fit_root: pathlib.Path,
    campaign_manifest_path: pathlib.Path,
    protocol_path: pathlib.Path | None = None,
) -> RelationshipP4PhysicalActuationRunResult:
    """GPU/torch-free validation of an existing preflight root."""

    protocol = load_relationship_p4_physical_actuation_protocol(protocol_path)
    _verify_critical_sources(protocol)
    authenticated = _authenticate_fit_input(
        input_fit_root=pathlib.Path(input_fit_root).resolve(),
        campaign_manifest_path=pathlib.Path(campaign_manifest_path).resolve(),
    )
    return _validate_physical_actuation_artifact(
        output_dir=pathlib.Path(output_dir).resolve(),
        input_fit_root=pathlib.Path(input_fit_root).resolve(),
        campaign_manifest_path=pathlib.Path(campaign_manifest_path).resolve(),
        protocol=protocol,
        authenticated=authenticated,
    )


def _build_strict_runtime() -> Any:
    from volvence_zero.substrate import (
        WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1,
        build_transformers_runtime_with_fallback,
    )

    profile = WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1
    if profile.profile_id != _EXECUTION_PROFILE_ID:
        raise ValueError("public strict execution profile ID drift")
    _require_literal(profile.to_payload(), _PROFILE_FACTS, "strict profile")
    return build_transformers_runtime_with_fallback(
        model_id=_MODEL_ID,
        model_source=None,
        device="cuda",
        layer_indices=(_LAYER_INDEX,),
        activation_width=_RESIDUAL_WIDTH,
        max_length=_PROFILE_FACTS["context_window_tokens"],
        fail_on_truncation=True,
        local_files_only=True,
        fallback_mode="deny",
        runtime_mode="strict-local",
        model_dtype="bfloat16",
        expected_model_weights_sha256=_MODEL_WEIGHTS_SHA256,
        execution_profile=profile,
        verified_model_revision=_MODEL_REVISION,
        expected_execution_assets_sha256=_EXECUTION_ASSETS_SHA256,
    )


def _attest_live_runtime(*, runtime: Any, authenticated: _AuthenticatedFitInput) -> dict[str, Any]:
    if runtime.model_id != _MODEL_ID or runtime.is_frozen is not True:
        raise ValueError("live strict runtime model/frozen lineage mismatch")
    if runtime.loaded_base_model_weights_sha256 != _MODEL_WEIGHTS_SHA256:
        raise ValueError("live strict runtime weights digest mismatch")
    if runtime.fallback_active is not False or runtime.capture_source != "real":
        raise ValueError("live strict runtime entered a fallback capture path")
    if runtime.fail_on_truncation is not True:
        raise ValueError("live strict runtime disabled fail-on-truncation")
    attestation = runtime.execution_attestation
    if attestation is None:
        raise RuntimeError("live strict runtime did not publish an attestation")
    payload = {**attestation.to_payload(), "attestation_id": attestation.attestation_id}
    _require_literal(
        payload,
        authenticated.execution_attestation,
        "live execution attestation versus authenticated fit attestation",
    )
    _validate_attestation_lineage(payload)
    return payload


async def _execute_physical_actuation_preflight(
    *,
    runtime: Any,
    bundle: Any,
    protocol: RelationshipP4PhysicalActuationProtocol,
    prompts: tuple[PhysicalActuationPrompt, ...],
    progress: Callable[[str], None] | None,
) -> tuple[dict[str, Any], ...]:
    from volvence_zero.runtime import RuntimePlaceholderValue, WiringLevel
    from volvence_zero.steering_contracts import SteeringGateAction
    from volvence_zero.steering_executor import SteeringExecutorModule
    from volvence_zero.steering_sensor import SteeringSensorModule
    from volvence_zero.substrate import OpenWeightResidualStreamSubstrateAdapter

    if bundle.sensor_off_executor is None:
        raise ValueError("authenticated fit bundle lacks sensor-off executor")
    adapter = OpenWeightResidualStreamSubstrateAdapter(runtime=runtime)
    sensor = SteeringSensorModule(
        artifact=bundle.reader,
        wiring_level=WiringLevel.SHADOW,
    )
    gate_placeholder = RuntimePlaceholderValue(
        reason="gate_disabled_for_frozen_gate_off_preflight",
        expected_slot="steering_gate_decision",
        produced_by="relationship_p4_physical_residual_actuation",
        detail=(
            "Explicit SHADOW gate-off arm; action is frozen by each executor "
            "constructor and no gate policy is learned or invoked."
        ),
    )
    receipts: list[dict[str, Any]] = []
    completed = 0
    for prompt in prompts:
        for repeat_index in range(_REPEAT_COUNT):
            sensor.reset_history()
            substrate = await adapter.capture(source_text=prompt.prompt)
            base_residual = _extract_residual(
                substrate,
                layer_index=_LAYER_INDEX,
                residual_width=_RESIDUAL_WIDTH,
            )
            invocation_ledger = [
                _runtime_forward_invocation_entry(
                    sample_id=prompt.sample_id,
                    repeat_index=repeat_index,
                    ordinal=0,
                    arm="raw_no_intervention",
                )
            ]
            belief_snapshot = await sensor.process_standalone(substrate=substrate)
            belief = belief_snapshot.value
            noop = await _execute_one_arm(
                runtime=runtime,
                substrate=substrate,
                belief=belief,
                artifact=bundle.executor,
                source_text=prompt.prompt,
                action=SteeringGateAction.NOOP,
                gate_placeholder=gate_placeholder,
                wiring_level=WiringLevel.SHADOW,
                executor_type=SteeringExecutorModule,
            )
            invocation_ledger.append(
                _runtime_forward_invocation_entry(
                    sample_id=prompt.sample_id,
                    repeat_index=repeat_index,
                    ordinal=1,
                    arm="strict_noop",
                )
            )
            conditional = await _execute_one_arm(
                runtime=runtime,
                substrate=substrate,
                belief=belief,
                artifact=bundle.executor,
                source_text=prompt.prompt,
                action=SteeringGateAction.STEER,
                gate_placeholder=gate_placeholder,
                wiring_level=WiringLevel.SHADOW,
                executor_type=SteeringExecutorModule,
            )
            invocation_ledger.append(
                _runtime_forward_invocation_entry(
                    sample_id=prompt.sample_id,
                    repeat_index=repeat_index,
                    ordinal=2,
                    arm="conditional_always_on",
                )
            )
            sensor_off = await _execute_one_arm(
                runtime=runtime,
                substrate=substrate,
                belief=belief,
                artifact=bundle.sensor_off_executor,
                source_text=prompt.prompt,
                action=SteeringGateAction.STEER,
                gate_placeholder=gate_placeholder,
                wiring_level=WiringLevel.SHADOW,
                executor_type=SteeringExecutorModule,
            )
            invocation_ledger.append(
                _runtime_forward_invocation_entry(
                    sample_id=prompt.sample_id,
                    repeat_index=repeat_index,
                    ordinal=3,
                    arm="sensor_off_always_on",
                )
            )
            receipt_core = {
                "schema_version": P4_PHYSICAL_ACTUATION_RECEIPT_SCHEMA_VERSION,
                "protocol_id": protocol.protocol_id,
                "prompt_set_sha256": protocol.prompt_set_sha256,
                "input_fit_artifact_id": _INPUT_ARTIFACT_ID,
                "execution_attestation_id": _EXECUTION_ATTESTATION_ID,
                "sample_id": prompt.sample_id,
                "case_id": prompt.case_id,
                "step_index": prompt.step_index,
                "expected_subgoal_class": prompt.expected_subgoal_class,
                "prompt": prompt.prompt,
                "prompt_sha256": prompt.prompt_sha256,
                "repeat_index": repeat_index,
                "sensor_reset_before_capture": True,
                "base_token_logits": list(substrate.token_logits),
                "belief": asdict(belief),
                "runtime_forward_invocation_ledger": invocation_ledger,
                "arms": [
                    _raw_arm_payload(base_residual),
                    noop,
                    conditional,
                    sensor_off,
                ],
            }
            receipt_id = _sha256_bytes(_canonical_bytes(receipt_core))
            receipts.append({**receipt_core, "receipt_id": receipt_id})
            completed += 1
            if progress is not None and (completed == _RECEIPT_COUNT or completed % 8 == 0):
                progress(
                    "P4 physical residual actuation: "
                    f"{completed}/{_RECEIPT_COUNT} receipts, "
                    f"{completed * 4}/{_RUNTIME_FORWARD_INVOCATION_COUNT} "
                    "observed forward-bearing runtime API returns"
                )
    return tuple(receipts)


def _runtime_forward_invocation_entry(
    *,
    sample_id: str,
    repeat_index: int,
    ordinal: int,
    arm: str,
) -> dict[str, Any]:
    if arm not in _ARM_ORDER:
        raise ValueError(f"unknown runtime invocation arm {arm!r}")
    return {
        "ordinal": ordinal,
        "sample_id": sample_id,
        "repeat_index": repeat_index,
        "arm": arm,
        "runtime_api": _RUNTIME_FORWARD_INVOCATION_APIS[arm],
        "successful_return_observed": True,
    }


async def _execute_one_arm(
    *,
    runtime: Any,
    substrate: Any,
    belief: Any,
    artifact: Any,
    source_text: str,
    action: Any,
    gate_placeholder: Any,
    wiring_level: Any,
    executor_type: Any,
) -> dict[str, Any]:
    executor = executor_type(
        artifact=artifact,
        runtime=None,
        source_text=source_text,
        apply_shadow_hook=False,
        sensor_off_artifact=None,
        ungated_action=action,
        wiring_level=wiring_level,
    )
    intervention_snapshot = await executor.process_standalone(
        substrate=substrate,
        belief=belief,
        gate=gate_placeholder,
    )
    intervention = intervention_snapshot.value
    if intervention.shadow_hook_executed:
        raise ValueError(
            "canonical executor unexpectedly ran its own hook; the preflight "
            "requires exactly one explicit direct apply per intervention arm"
        )
    application = runtime.apply_direct_residual_delta(
        source_text=source_text,
        substrate_snapshot=substrate,
        layer_index=intervention.layer_index,
        residual_delta=intervention.residual_delta,
    )
    output_residual = _extract_residual(
        application.applied_snapshot,
        layer_index=_LAYER_INDEX,
        residual_width=_RESIDUAL_WIDTH,
    )
    artifact_kind = (
        "matched_sensor_off" if artifact.artifact_id.startswith("steering-executor-sensor-off:") else "conditional"
    )
    arm_name = (
        "strict_noop"
        if action.value == "noop"
        else ("sensor_off_always_on" if artifact_kind == "matched_sensor_off" else "conditional_always_on")
    )
    selected_code = artifact.condition_codes[belief.belief_index]
    return {
        "arm": arm_name,
        "sensor_invoked": True,
        "executor_invoked": True,
        "direct_hook_invoked": True,
        "canonical_executor_shadow_hook_executed": False,
        "wiring_level": "shadow",
        "action": intervention.action.value,
        "executor_artifact_kind": artifact_kind,
        "executor_artifact_id": artifact.artifact_id,
        "reader_artifact_id": intervention.reader_artifact_id,
        "belief_index": belief.belief_index,
        "selected_condition_code_sha256": _vector_sha256(selected_code),
        "application_mode": intervention.application_mode,
        "source_residual_norm": intervention.residual_norm,
        "control_norm": intervention.control_norm,
        "control_norm_cap": intervention.control_norm_cap,
        "zero_code_noop": intervention.zero_code_noop,
        "delta": _encode_vector(intervention.residual_delta),
        "output_residual": _encode_vector(output_residual),
        "runtime_backend": application.backend_name,
        "downstream_effect": list(application.downstream_effect),
        "control_energy": application.control_energy,
    }


def _raw_arm_payload(base_residual: tuple[float, ...]) -> dict[str, Any]:
    return {
        "arm": "raw_no_intervention",
        "sensor_invoked": True,
        "executor_invoked": False,
        "direct_hook_invoked": False,
        "canonical_executor_shadow_hook_executed": False,
        "wiring_level": "shadow",
        "action": "raw",
        "executor_artifact_kind": "none",
        "executor_artifact_id": "",
        "reader_artifact_id": "",
        "belief_index": -1,
        "selected_condition_code_sha256": "",
        "application_mode": "raw-base-capture",
        "source_residual_norm": _vector_norm(base_residual),
        "control_norm": 0.0,
        "control_norm_cap": 0.0,
        "zero_code_noop": False,
        "delta": None,
        "output_residual": _encode_vector(base_residual),
        "runtime_backend": "not-applied",
        "downstream_effect": [0.0, 0.0, 0.0],
        "control_energy": 0.0,
    }


def _authenticate_fit_input(
    *, input_fit_root: pathlib.Path, campaign_manifest_path: pathlib.Path
) -> _AuthenticatedFitInput:
    if not input_fit_root.is_dir():
        raise ValueError(
            "P4 physical actuation requires the complete P4.6-fit artifact root; standalone bundle paths are forbidden"
        )
    from volvence_zero.agent.relationship_p4_steering_artifact_fit import (
        validate_relationship_p4_steering_artifact_fit,
    )

    validation = validate_relationship_p4_steering_artifact_fit(output_dir=input_fit_root)
    if (
        validation.artifact_id != _INPUT_ARTIFACT_ID
        or validation.protocol_id != _INPUT_FIT_PROTOCOL_ID
        or validation.execution_attestation_id != _EXECUTION_ATTESTATION_ID
        or validation.prerequisite_passed is not True
    ):
        raise ValueError("P4.6-fit validator result does not match the frozen canonical input")
    fit_manifest_path = input_fit_root / "manifest.json"
    bundle_path = input_fit_root / "steering_artifact_bundle.json"
    attestation_path = input_fit_root / "execution_attestation.json"
    if _sha256_file(fit_manifest_path) != _INPUT_MANIFEST_SHA256:
        raise ValueError("P4.6-fit manifest raw SHA-256 mismatch")
    if _sha256_file(bundle_path) != _INPUT_BUNDLE_SHA256:
        raise ValueError("P4.6-fit bundle raw SHA-256 mismatch")
    fit_manifest = _load_json_object(fit_manifest_path)
    _require_literal(
        fit_manifest.get("artifact_id"),
        _INPUT_ARTIFACT_ID,
        "P4.6-fit artifact_id",
    )
    _require_literal(
        fit_manifest.get("protocol_id"),
        _INPUT_FIT_PROTOCOL_ID,
        "P4.6-fit protocol_id",
    )
    campaign_raw_sha256 = _sha256_file(campaign_manifest_path)
    if campaign_raw_sha256 != _INPUT_CAMPAIGN_MANIFEST_SHA256:
        raise ValueError("P4.6-fit campaign manifest raw SHA-256 mismatch")
    campaign = _load_json_object(campaign_manifest_path)
    _require_literal(
        campaign.get("campaign_id"),
        _INPUT_CAMPAIGN_ID,
        "P4.6-fit campaign_id",
    )
    _require_literal(
        campaign.get("protocol_id"),
        _INPUT_FIT_PROTOCOL_ID,
        "P4.6-fit campaign protocol_id",
    )
    _require_literal(
        campaign.get("execution_profile_id"),
        _EXECUTION_PROFILE_ID,
        "P4.6-fit campaign execution profile",
    )
    attempts = _require_list(campaign.get("attempts"), "campaign attempts")
    if len(attempts) != 2:
        raise ValueError("P4.6-fit campaign must retain exactly two attempts")
    adjudicable = _require_mapping(attempts[1], "campaign attempts[1]")
    for key, expected in (
        ("fit_artifact_id", _INPUT_ARTIFACT_ID),
        ("fit_manifest_sha256", _INPUT_MANIFEST_SHA256),
        ("bundle_sha256", _INPUT_BUNDLE_SHA256),
        ("execution_attestation_id", _EXECUTION_ATTESTATION_ID),
    ):
        _require_literal(adjudicable.get(key), expected, f"campaign {key}")
    attestation = _load_json_object(attestation_path)
    _validate_attestation_lineage(attestation)
    from volvence_zero.steering_contracts import SteeringArtifactBundle

    bundle = SteeringArtifactBundle.from_json(bundle_path.read_text(encoding="utf-8"))
    _validate_bundle_lineage(bundle)
    return _AuthenticatedFitInput(
        bundle=bundle,
        execution_attestation=attestation,
        fit_manifest=fit_manifest,
        campaign_manifest=campaign,
    )


def _validate_bundle_lineage(bundle: Any) -> None:
    if bundle.reader.model_id != _MODEL_ID:
        raise ValueError("P4.6-fit reader model ID mismatch")
    if bundle.reader.model_weights_sha256 != _MODEL_WEIGHTS_SHA256:
        raise ValueError("P4.6-fit reader weights digest mismatch")
    if (
        bundle.reader.layer_index != _LAYER_INDEX
        or bundle.reader.residual_width != _RESIDUAL_WIDTH
        or bundle.reader.class_labels != _CLASS_LABELS
    ):
        raise ValueError("P4.6-fit reader geometry/class lineage mismatch")
    executor = bundle.executor
    sensor_off = bundle.sensor_off_executor
    if sensor_off is None:
        raise ValueError("P4.6-fit bundle lacks matched sensor-off executor")
    for name, candidate in (
        ("conditional", executor),
        ("sensor-off", sensor_off),
    ):
        if (
            candidate.model_id != _MODEL_ID
            or candidate.model_weights_sha256 != _MODEL_WEIGHTS_SHA256
            or candidate.reader_artifact_id != bundle.reader.artifact_id
            or candidate.layer_index != _LAYER_INDEX
            or candidate.residual_width != _RESIDUAL_WIDTH
            or candidate.rank != _STEERING_RANK
            or candidate.class_labels != _CLASS_LABELS
            or not math.isclose(
                candidate.control_norm_cap_ratio,
                _CONTROL_NORM_CAP_RATIO,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError(f"P4.6-fit {name} executor lineage mismatch")
    if len(set(sensor_off.condition_codes)) != 1:
        raise ValueError("P4.6-fit sensor-off condition-code rows are not identical")


def _validate_attestation_lineage(attestation: Mapping[str, Any]) -> None:
    expected = {
        "attestation_id": _EXECUTION_ATTESTATION_ID,
        "profile_id": _EXECUTION_PROFILE_ID,
        "model_id": _MODEL_ID,
        "model_revision": _MODEL_REVISION,
        "model_weights_sha256": _MODEL_WEIGHTS_SHA256,
        "execution_assets_sha256": _EXECUTION_ASSETS_SHA256,
        "hidden_size": _RESIDUAL_WIDTH,
        "hook_layer_indices": [_LAYER_INDEX],
        "context_window_tokens": 32768,
        "model_max_position_embeddings": 32768,
        "runtime_origin": "hf-local",
        "platform_system": "Windows",
        "device": "cuda",
        "model_dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "sdpa_backend": "cudnn",
        "sdpa_backend_policy": "exclusive-cudnn",
        "sdpa_backend_exclusive": True,
        "local_files_only": True,
        "fallback_mode": "deny",
        "fail_on_truncation": True,
    }
    for key, value in expected.items():
        _require_literal(attestation.get(key), value, f"attestation.{key}")


def _build_frozen_prompt_set(
    protocol: RelationshipP4PhysicalActuationProtocol,
) -> tuple[PhysicalActuationPrompt, ...]:
    from volvence_zero.agent.eta_conditional_steering_screen import (
        ACTION_PROMPT_SUFFIX,
    )
    from volvence_zero.agent.eta_conflict_instrument import (
        build_conflict_junction_rows,
    )
    from volvence_zero.agent.eta_proof_benchmark import generate_eta_proof_corpus

    corpus = generate_eta_proof_corpus(
        seed=_DATASET_RECIPE["seed"],
        objective_count=_DATASET_RECIPE["objective_count"],
        corridor_count=_DATASET_RECIPE["corridor_count"],
        extra_edge_probability=_DATASET_RECIPE["extra_edge_probability"],
        train_route_count=_DATASET_RECIPE["train_route_count"],
        heldout_route_count=_DATASET_RECIPE["heldout_route_count"],
        train_lengths=tuple(_DATASET_RECIPE["train_lengths"]),
        heldout_lengths=tuple(_DATASET_RECIPE["heldout_lengths"]),
    )
    labels = tuple(location.location_id for location in corpus.environment.objective_locations())
    if labels != _CLASS_LABELS:
        raise ValueError("frozen ETA objective-class order drift")
    case_sources = {case.case_id: case.source_text for case in corpus.heldout_cases}
    rows = build_conflict_junction_rows(corpus, split="heldout")
    quotas = _DATASET_RECIPE["class_quotas"]
    selected_counts = {label: 0 for label in _CLASS_LABELS}
    selected: list[PhysicalActuationPrompt] = []
    prompt_values: set[str] = set()
    for row in rows:
        label = row.active_subgoal
        if label not in selected_counts or selected_counts[label] >= quotas[label]:
            continue
        source = case_sources[row.case_id]
        prompt = f"Task context: {source}. {row.subgoal_revealed_text}{ACTION_PROMPT_SUFFIX}"
        if prompt in prompt_values:
            continue
        prompt_sha256 = _sha256_bytes(prompt.encode("utf-8"))
        selected_counts[label] += 1
        sample_id = f"p46-physical-{len(selected):03d}-{prompt_sha256[:12]}"
        selected.append(
            PhysicalActuationPrompt(
                sample_id=sample_id,
                case_id=row.case_id,
                step_index=row.step_index,
                expected_subgoal_class=label,
                prompt=prompt,
                prompt_sha256=prompt_sha256,
            )
        )
        prompt_values.add(prompt)
        if len(selected) == _PROMPT_COUNT:
            break
    if len(selected) != _PROMPT_COUNT:
        raise RuntimeError("frozen ETA heldout corpus could not supply 68 unique balanced prompts")
    if selected_counts != quotas:
        raise RuntimeError(f"frozen ETA heldout prompt class quotas drifted: {selected_counts!r} != {quotas!r}")
    payload = [asdict(row) for row in selected]
    prompt_set_sha256 = _sha256_bytes(_canonical_bytes(payload))
    if prompt_set_sha256 != protocol.prompt_set_sha256:
        raise ValueError(
            "frozen ETA physical-actuation prompt-set SHA-256 drift: "
            f"{prompt_set_sha256} != {protocol.prompt_set_sha256}"
        )
    return tuple(selected)


def _derive_metrics_and_checks(
    *,
    protocol: RelationshipP4PhysicalActuationProtocol,
    prompts: tuple[PhysicalActuationPrompt, ...],
    bundle: Any,
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(receipts) != _RECEIPT_COUNT:
        raise ValueError(f"physical-actuation receipt count drift: {len(receipts)}")
    expected_by_coordinate = {
        (prompt.sample_id, repeat): prompt for prompt in prompts for repeat in range(_REPEAT_COUNT)
    }
    observed_coordinates: set[tuple[str, int]] = set()
    decoded_rows: list[dict[str, Any]] = []
    for receipt in receipts:
        decoded = _validate_and_decode_receipt(
            receipt=receipt,
            protocol=protocol,
            expected_by_coordinate=expected_by_coordinate,
            bundle=bundle,
        )
        coordinate = (decoded["sample_id"], decoded["repeat_index"])
        if coordinate in observed_coordinates:
            raise ValueError(f"duplicate physical-actuation receipt {coordinate!r}")
        observed_coordinates.add(coordinate)
        decoded_rows.append(decoded)
    if observed_coordinates != set(expected_by_coordinate):
        raise ValueError("physical-actuation receipt coordinate coverage drift")
    _recompute_owner_outputs_from_receipts(
        bundle=bundle,
        decoded_rows=decoded_rows,
    )

    class_counts = {
        label: sum(1 for prompt in prompts if prompt.expected_subgoal_class == label) for label in _CLASS_LABELS
    }
    prompt_set_check = (
        len(prompts) == _PROMPT_COUNT
        and len({prompt.prompt for prompt in prompts}) == _PROMPT_COUNT
        and class_counts == _DATASET_RECIPE["class_quotas"]
    )
    raw_noop_exact = True
    noop_zero = True
    noop_canonical = True
    conditional_nonzero_capped = True
    conditional_observed_effect = True
    sensor_nonzero_capped = True
    sensor_observed_effect = True
    conditional_sensor_observed_differ = True
    conditional_code_hashes: set[str] = set()
    sensor_code_hashes: set[str] = set()
    for row in decoded_rows:
        arms = row["arms"]
        raw = arms["raw_no_intervention"]
        noop = arms["strict_noop"]
        conditional = arms["conditional_always_on"]
        sensor_off = arms["sensor_off_always_on"]
        raw_bytes = raw["output_bytes"]
        noop_delta = noop["delta"]
        conditional_delta = conditional["delta"]
        sensor_delta = sensor_off["delta"]
        raw_noop_exact = raw_noop_exact and (noop["output_bytes"] == raw_bytes)
        noop_zero = noop_zero and (
            noop_delta == tuple(0.0 for _ in range(_RESIDUAL_WIDTH))
            and noop["control_norm"] == 0.0
            and noop["control_energy"] == 0.0
            and tuple(noop["downstream_effect"]) == (0.0, 0.0, 0.0)
            and noop["zero_code_noop"] is True
        )
        noop_canonical = noop_canonical and (
            noop["executor_invoked"] is True
            and noop["direct_hook_invoked"] is True
            and noop["canonical_executor_shadow_hook_executed"] is False
            and noop["application_mode"] == "shadow-noop"
        )
        conditional_nonzero_capped = conditional_nonzero_capped and (
            _finite_nonzero_capped_delta(conditional, conditional_delta)
        )
        sensor_nonzero_capped = sensor_nonzero_capped and (_finite_nonzero_capped_delta(sensor_off, sensor_delta))
        conditional_observed_effect = conditional_observed_effect and (
            conditional["direct_hook_invoked"] is True and conditional["output_bytes"] != raw_bytes
        )
        sensor_observed_effect = sensor_observed_effect and (
            sensor_off["direct_hook_invoked"] is True and sensor_off["output_bytes"] != raw_bytes
        )
        conditional_sensor_observed_differ = conditional_sensor_observed_differ and (
            conditional["output_bytes"] != sensor_off["output_bytes"]
            and _vector_bytes(conditional_delta) != _vector_bytes(sensor_delta)
        )
        conditional_code_hashes.add(conditional["selected_condition_code_sha256"])
        sensor_code_hashes.add(sensor_off["selected_condition_code_sha256"])

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in decoded_rows:
        grouped.setdefault(row["sample_id"], []).append(row)
    repeat_exact = all(
        len(rows) == _REPEAT_COUNT and _repeat_projection(rows[0]) == _repeat_projection(rows[1])
        for rows in grouped.values()
    )
    sensor_artifact_code_hashes = {_vector_sha256(row) for row in bundle.sensor_off_executor.condition_codes}
    checks = {
        "prompt_set_unique_and_class_balanced": prompt_set_check,
        "raw_noop_output_residual_exact": raw_noop_exact,
        "strict_noop_delta_and_effect_zero": noop_zero,
        "strict_noop_canonical_executor_and_direct_hook": noop_canonical,
        "conditional_delta_finite_nonzero_capped": conditional_nonzero_capped,
        "conditional_observed_applied_residual_differs": conditional_observed_effect,
        "sensor_off_delta_finite_nonzero_capped": sensor_nonzero_capped,
        "sensor_off_observed_applied_residual_differs": sensor_observed_effect,
        "conditional_sensor_off_observed_hashes_differ": conditional_sensor_observed_differ,
        "repeat_outputs_exact": repeat_exact,
        "conditional_uses_at_least_two_code_rows": (len(conditional_code_hashes) >= 2),
        "sensor_off_condition_code_rows_identical": (
            len(sensor_artifact_code_hashes) == 1
            and len(sensor_code_hashes) == 1
            and sensor_code_hashes == sensor_artifact_code_hashes
        ),
        "runtime_forward_invocation_ledger_exact": (
            sum(len(row["runtime_forward_invocation_ledger"]) for row in decoded_rows)
            == _RUNTIME_FORWARD_INVOCATION_COUNT
            and len(decoded_rows) * len(_ARM_ORDER) == _ARM_EVALUATION_COUNT
        ),
    }
    return {
        "prompt_count": len(prompts),
        "repeat_count": _REPEAT_COUNT,
        "receipt_count": len(decoded_rows),
        "arm_evaluation_count": len(decoded_rows) * len(_ARM_ORDER),
        "runtime_forward_invocation_count": sum(len(row["runtime_forward_invocation_ledger"]) for row in decoded_rows),
        "class_counts": class_counts,
        "fresh_belief_class_count": len({row["belief"]["fresh_belief_label"] for row in decoded_rows}),
        "conditional_code_row_count": len(conditional_code_hashes),
        "sensor_off_code_row_count": len(sensor_code_hashes),
        "checks": checks,
    }


def _validate_and_decode_receipt(
    *,
    receipt: Mapping[str, Any],
    protocol: RelationshipP4PhysicalActuationProtocol,
    expected_by_coordinate: Mapping[tuple[str, int], PhysicalActuationPrompt],
    bundle: Any,
) -> dict[str, Any]:
    required_keys = {
        "schema_version",
        "protocol_id",
        "prompt_set_sha256",
        "input_fit_artifact_id",
        "execution_attestation_id",
        "sample_id",
        "case_id",
        "step_index",
        "expected_subgoal_class",
        "prompt",
        "prompt_sha256",
        "repeat_index",
        "sensor_reset_before_capture",
        "base_token_logits",
        "belief",
        "runtime_forward_invocation_ledger",
        "arms",
        "receipt_id",
    }
    _require_exact_keys(receipt, required_keys, "physical-actuation receipt")
    _require_literal(
        receipt["schema_version"],
        P4_PHYSICAL_ACTUATION_RECEIPT_SCHEMA_VERSION,
        "receipt schema_version",
    )
    for key, expected in (
        ("protocol_id", protocol.protocol_id),
        ("prompt_set_sha256", protocol.prompt_set_sha256),
        ("input_fit_artifact_id", _INPUT_ARTIFACT_ID),
        ("execution_attestation_id", _EXECUTION_ATTESTATION_ID),
        ("sensor_reset_before_capture", True),
    ):
        _require_literal(receipt[key], expected, f"receipt {key}")
    receipt_core = {key: value for key, value in receipt.items() if key != "receipt_id"}
    expected_receipt_id = _sha256_bytes(_canonical_bytes(receipt_core))
    _require_literal(receipt["receipt_id"], expected_receipt_id, "receipt receipt_id")
    sample_id = _require_text_value(receipt["sample_id"], "receipt sample_id")
    repeat_index = _require_int_value(receipt["repeat_index"], "receipt repeat_index")
    coordinate = (sample_id, repeat_index)
    expected_prompt = expected_by_coordinate.get(coordinate)
    if expected_prompt is None:
        raise ValueError(f"unknown receipt coordinate {coordinate!r}")
    expected_prompt_payload = {
        "case_id": expected_prompt.case_id,
        "step_index": expected_prompt.step_index,
        "expected_subgoal_class": expected_prompt.expected_subgoal_class,
        "prompt": expected_prompt.prompt,
        "prompt_sha256": expected_prompt.prompt_sha256,
    }
    for key, expected in expected_prompt_payload.items():
        _require_literal(receipt[key], expected, f"receipt {key}")
    base_token_logits = _decode_token_logits(
        receipt["base_token_logits"],
        label="receipt base_token_logits",
    )
    invocation_ledger = _require_list(
        receipt["runtime_forward_invocation_ledger"],
        "receipt runtime_forward_invocation_ledger",
    )
    expected_ledger = [
        _runtime_forward_invocation_entry(
            sample_id=sample_id,
            repeat_index=repeat_index,
            ordinal=ordinal,
            arm=arm,
        )
        for ordinal, arm in enumerate(_ARM_ORDER)
    ]
    _require_literal(
        invocation_ledger,
        expected_ledger,
        "receipt runtime forward invocation ledger",
    )
    belief = _require_mapping(receipt["belief"], "receipt belief")
    if (
        belief.get("reader_artifact_id") != bundle.reader.artifact_id
        or belief.get("source_model_id") != _MODEL_ID
        or belief.get("source_layer_index") != _LAYER_INDEX
    ):
        raise ValueError("receipt belief lineage mismatch")
    belief_index = _require_int_value(belief.get("belief_index"), "receipt belief.belief_index")
    if not 0 <= belief_index < len(_CLASS_LABELS):
        raise ValueError("receipt belief index outside frozen class rows")
    _require_literal(
        belief.get("belief_label"),
        _CLASS_LABELS[belief_index],
        "receipt belief label/index",
    )
    arms_raw = _require_list(receipt["arms"], "receipt arms")
    if [arm.get("arm") for arm in arms_raw if isinstance(arm, Mapping)] != list(_ARM_ORDER):
        raise ValueError("receipt arm order/coverage drift")
    arms: dict[str, dict[str, Any]] = {}
    for raw_arm in arms_raw:
        arm = _require_mapping(raw_arm, "receipt arm")
        arm_name = _require_text_value(arm.get("arm"), "receipt arm name")
        _validate_arm_shape_and_lineage(
            arm=arm,
            arm_name=arm_name,
            belief_index=belief_index,
            bundle=bundle,
        )
        output_vector, output_bytes = _decode_vector(
            arm["output_residual"],
            label=f"receipt arm {arm_name} output_residual",
        )
        if arm["delta"] is None:
            delta = None
        else:
            delta, _ = _decode_vector(arm["delta"], label=f"receipt arm {arm_name} delta")
        decoded_arm = dict(arm)
        decoded_arm["output_residual"] = output_vector
        decoded_arm["output_bytes"] = output_bytes
        decoded_arm["delta"] = delta
        arms[arm_name] = decoded_arm
    base_norm = _vector_norm(arms["raw_no_intervention"]["output_residual"])
    if not math.isclose(
        _require_number_value(
            arms["raw_no_intervention"]["source_residual_norm"],
            "raw_no_intervention source_residual_norm",
        ),
        base_norm,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("raw_no_intervention source residual norm drift")
    for arm_name in _ARM_ORDER[1:]:
        arm = arms[arm_name]
        if not math.isclose(
            _require_number_value(arm["source_residual_norm"], f"{arm_name} source_residual_norm"),
            base_norm,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{arm_name} source residual norm drift")
        delta = arm["delta"]
        if delta is None:
            raise ValueError(f"{arm_name} is missing executor delta")
        delta_norm = _vector_norm(delta)
        if not math.isclose(
            _require_number_value(arm["control_norm"], f"{arm_name} control_norm"),
            delta_norm,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{arm_name} control norm does not match retained delta")
        expected_cap = base_norm * _CONTROL_NORM_CAP_RATIO
        if not math.isclose(
            _require_number_value(arm["control_norm_cap"], f"{arm_name} control_norm_cap"),
            expected_cap,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{arm_name} control norm cap drift")
    return {
        "sample_id": sample_id,
        "repeat_index": repeat_index,
        "prompt": receipt["prompt"],
        "base_token_logits": base_token_logits,
        "belief": dict(belief),
        "runtime_forward_invocation_ledger": tuple(invocation_ledger),
        "arms": arms,
    }


def _recompute_owner_outputs_from_receipts(
    *,
    bundle: Any,
    decoded_rows: Sequence[Mapping[str, Any]],
) -> None:
    asyncio.run(
        _recompute_owner_outputs_from_receipts_async(
            bundle=bundle,
            decoded_rows=decoded_rows,
        )
    )


async def _recompute_owner_outputs_from_receipts_async(
    *,
    bundle: Any,
    decoded_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Re-run the public sensor/executor owners without a model or GPU."""

    from volvence_zero.runtime import RuntimePlaceholderValue, WiringLevel
    from volvence_zero.steering_contracts import SteeringGateAction
    from volvence_zero.steering_executor import SteeringExecutorModule
    from volvence_zero.steering_sensor import SteeringSensorModule
    from volvence_zero.substrate import (
        ResidualActivation,
        SubstrateSnapshot,
        SurfaceKind,
    )

    if bundle.sensor_off_executor is None:
        raise ValueError("authenticated fit bundle lacks sensor-off executor")
    sensor = SteeringSensorModule(
        artifact=bundle.reader,
        wiring_level=WiringLevel.SHADOW,
    )
    gate_placeholder = RuntimePlaceholderValue(
        reason="gate_disabled_for_frozen_gate_off_preflight_offline_recompute",
        expected_slot="steering_gate_decision",
        produced_by="relationship_p4_physical_residual_actuation",
        detail=("GPU-free owner recomputation of the frozen explicit SHADOW gate-off arms."),
    )
    for row in decoded_rows:
        sample_id = row["sample_id"]
        repeat_index = row["repeat_index"]
        coordinate = f"{sample_id}/repeat-{repeat_index}"
        raw = row["arms"]["raw_no_intervention"]
        raw_residual = raw["output_residual"]
        substrate = SubstrateSnapshot(
            model_id=_MODEL_ID,
            is_frozen=True,
            surface_kind=SurfaceKind.RESIDUAL_STREAM,
            token_logits=row["base_token_logits"],
            feature_surface=(),
            residual_activations=(
                ResidualActivation(
                    layer_index=_LAYER_INDEX,
                    activation=raw_residual,
                    step=0,
                ),
            ),
            residual_sequence=(),
            unavailable_fields=(),
            description="GPU-free reconstructed P4 physical-actuation base snapshot",
        )
        sensor.reset_history()
        recomputed_belief = (await sensor.process_standalone(substrate=substrate)).value
        if (
            recomputed_belief.belief_index != recomputed_belief.fresh_belief_index
            or recomputed_belief.belief_label != recomputed_belief.fresh_belief_label
            or recomputed_belief.belief_disagrees_fresh is not False
        ):
            raise ValueError(f"offline reset-history owner belief is not fresh at {coordinate}")
        _require_literal(
            row["belief"],
            asdict(recomputed_belief),
            f"offline owner-recomputed belief at {coordinate}",
        )

        expected_arms = (
            ("strict_noop", bundle.executor, SteeringGateAction.NOOP),
            (
                "conditional_always_on",
                bundle.executor,
                SteeringGateAction.STEER,
            ),
            (
                "sensor_off_always_on",
                bundle.sensor_off_executor,
                SteeringGateAction.STEER,
            ),
        )
        for arm_name, artifact, action in expected_arms:
            executor = SteeringExecutorModule(
                artifact=artifact,
                runtime=None,
                source_text=row["prompt"],
                apply_shadow_hook=False,
                sensor_off_artifact=None,
                ungated_action=action,
                wiring_level=WiringLevel.SHADOW,
            )
            recomputed = (
                await executor.process_standalone(
                    substrate=substrate,
                    belief=recomputed_belief,
                    gate=gate_placeholder,
                )
            ).value
            observed = row["arms"][arm_name]
            expected_values = {
                "action": recomputed.action.value,
                "executor_artifact_id": recomputed.executor_artifact_id,
                "reader_artifact_id": recomputed.reader_artifact_id,
                "belief_index": recomputed_belief.belief_index,
                "selected_condition_code_sha256": _vector_sha256(
                    artifact.condition_codes[recomputed_belief.belief_index]
                ),
                "application_mode": recomputed.application_mode,
                "source_residual_norm": recomputed.residual_norm,
                "control_norm": recomputed.control_norm,
                "control_norm_cap": recomputed.control_norm_cap,
                "zero_code_noop": recomputed.zero_code_noop,
            }
            for key, expected in expected_values.items():
                _require_literal(
                    observed[key],
                    expected,
                    f"offline owner-recomputed {arm_name}.{key} at {coordinate}",
                )
            observed_delta = observed["delta"]
            if observed_delta is None:
                raise ValueError(f"offline owner-recomputed {arm_name} lost its delta at {coordinate}")
            if _vector_bytes(observed_delta) != _vector_bytes(recomputed.residual_delta):
                raise ValueError(f"offline owner-recomputed {arm_name}.delta drift at {coordinate}")
            expected_energy = sum(abs(value) for value in recomputed.residual_delta) / len(recomputed.residual_delta)
            _require_literal(
                observed["control_energy"],
                expected_energy,
                f"offline owner-recomputed {arm_name}.control_energy at {coordinate}",
            )


def _validate_arm_shape_and_lineage(*, arm: Mapping[str, Any], arm_name: str, belief_index: int, bundle: Any) -> None:
    required_keys = {
        "arm",
        "sensor_invoked",
        "executor_invoked",
        "direct_hook_invoked",
        "canonical_executor_shadow_hook_executed",
        "wiring_level",
        "action",
        "executor_artifact_kind",
        "executor_artifact_id",
        "reader_artifact_id",
        "belief_index",
        "selected_condition_code_sha256",
        "application_mode",
        "source_residual_norm",
        "control_norm",
        "control_norm_cap",
        "zero_code_noop",
        "delta",
        "output_residual",
        "runtime_backend",
        "downstream_effect",
        "control_energy",
    }
    _require_exact_keys(arm, required_keys, f"receipt arm {arm_name}")
    if arm_name not in _ARM_ORDER:
        raise ValueError(f"unknown receipt arm {arm_name!r}")
    semantics = _ARM_SEMANTICS[arm_name]
    for key in (
        "sensor_invoked",
        "executor_invoked",
        "direct_hook_invoked",
        "action",
    ):
        _require_literal(arm[key], semantics[key], f"{arm_name}.{key}")
    _require_literal(
        arm["executor_artifact_kind"],
        semantics["executor_artifact"],
        f"{arm_name}.executor_artifact_kind",
    )
    _require_literal(arm["wiring_level"], "shadow", f"{arm_name}.wiring_level")
    _require_literal(
        arm["canonical_executor_shadow_hook_executed"],
        False,
        f"{arm_name}.canonical_executor_shadow_hook_executed",
    )
    effects = _require_list(arm["downstream_effect"], f"{arm_name}.downstream_effect")
    if len(effects) != 3 or any(
        not math.isfinite(_require_number_value(value, f"{arm_name}.effect")) for value in effects
    ):
        raise ValueError(f"{arm_name} downstream effect must contain 3 finite values")
    for key in (
        "source_residual_norm",
        "control_norm",
        "control_norm_cap",
        "control_energy",
    ):
        value = _require_number_value(arm[key], f"{arm_name}.{key}")
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{arm_name}.{key} must be finite/non-negative")
    if arm_name == "raw_no_intervention":
        raw_expected = {
            "executor_artifact_id": "",
            "reader_artifact_id": "",
            "belief_index": -1,
            "selected_condition_code_sha256": "",
            "application_mode": "raw-base-capture",
            "control_norm": 0.0,
            "control_norm_cap": 0.0,
            "zero_code_noop": False,
            "delta": None,
            "runtime_backend": "not-applied",
            "control_energy": 0.0,
        }
        for key, expected in raw_expected.items():
            _require_literal(arm[key], expected, f"raw.{key}")
        return
    artifact = bundle.sensor_off_executor if arm_name == "sensor_off_always_on" else bundle.executor
    if artifact is None:
        raise ValueError("sensor-off artifact disappeared")
    for key, expected in (
        ("executor_artifact_id", artifact.artifact_id),
        ("reader_artifact_id", bundle.reader.artifact_id),
        ("belief_index", belief_index),
        (
            "selected_condition_code_sha256",
            _vector_sha256(artifact.condition_codes[belief_index]),
        ),
    ):
        _require_literal(arm[key], expected, f"{arm_name}.{key}")
    _require_text_value(arm["runtime_backend"], f"{arm_name}.runtime_backend")
    if not arm["runtime_backend"].startswith("transformers-direct-steering:"):
        raise ValueError(f"{arm_name} did not use canonical direct backend")
    expected_mode = "shadow-noop" if arm_name == "strict_noop" else "shadow-compute-only"
    _require_literal(arm["application_mode"], expected_mode, f"{arm_name}.application_mode")
    _require_literal(
        arm["zero_code_noop"],
        arm_name == "strict_noop",
        f"{arm_name}.zero_code_noop",
    )


def _finite_nonzero_capped_delta(arm: Mapping[str, Any], delta: tuple[float, ...]) -> bool:
    norm = _vector_norm(delta)
    return (
        all(math.isfinite(value) for value in delta)
        and norm > 0.0
        and arm["control_norm"] > 0.0
        and math.isclose(
            norm,
            arm["control_norm"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        and arm["control_norm"] <= arm["control_norm_cap"] + 1e-8
        and arm["control_norm_cap"] > 0.0
    )


def _repeat_projection(row: Mapping[str, Any]) -> bytes:
    arms = row["arms"]
    projection = {
        "base_token_logits": row["base_token_logits"],
        "belief": row["belief"],
        "arms": {
            name: {key: value for key, value in arms[name].items() if key not in {"output_bytes"}}
            for name in _ARM_ORDER
        },
    }
    return _canonical_bytes(projection)


def _build_report_payload(
    *,
    protocol: RelationshipP4PhysicalActuationProtocol,
    authenticated: _AuthenticatedFitInput,
    derived: Mapping[str, Any],
    live_attestation: Mapping[str, Any],
    preflight_passed: bool,
    verdict: str,
) -> dict[str, Any]:
    del authenticated
    return {
        "schema_version": P4_PHYSICAL_ACTUATION_REPORT_SCHEMA_VERSION,
        "protocol_id": protocol.protocol_id,
        "prompt_set_sha256": protocol.prompt_set_sha256,
        "source_hash_mode": protocol.source_hash_mode,
        "source_sha256": protocol.source_sha256_payload(),
        "owner": {
            "wheel": "vz-runtime",
            "orchestrator": ("volvence_zero.agent.relationship_p4_physical_residual_actuation"),
            "new_runtime_slot_added": False,
        },
        "authenticated_input": _authenticated_input_payload(),
        "live_execution_attestation_id": live_attestation["attestation_id"],
        "execution": {
            "wiring_level": "shadow",
            "generation_performed": False,
            "capture_tokenization": "raw_tokenizer",
            "capture_use_cache": False,
            "exclusive_cudnn_sdpa_context": True,
            "fallback_allowed": False,
            "same_process_capture_sensor_executor_direct_apply": True,
            "strict_noop_explicit_zero_delta_direct_apply": True,
            "arm_order": list(_ARM_ORDER),
            "runtime_forward_invocation_semantics": (_RUNTIME_FORWARD_INVOCATION_SEMANTICS),
            "physical_readout": ("content-addressed-observed-latest-token-layer-20-residual-f64.v1"),
            "applied_gpu_residual_provenance": ("pinned_content_addressed_run_receipt_not_bundle_derivable"),
        },
        "dataset": {
            **_DATASET_RECIPE,
            "prompt_set_sha256": protocol.prompt_set_sha256,
            "synthetic_heldout_only": True,
        },
        "metrics": {
            key: derived[key]
            for key in (
                "prompt_count",
                "repeat_count",
                "receipt_count",
                "arm_evaluation_count",
                "runtime_forward_invocation_count",
                "class_counts",
                "fresh_belief_class_count",
                "conditional_code_row_count",
                "sensor_off_code_row_count",
            )
        },
        "checks": dict(derived["checks"]),
        "control_interpretation": _CONTROL_INTERPRETATION,
        "evidence_firewall": dict(_EVIDENCE_FIREWALL),
        "preflight_passed": preflight_passed,
        "failure_retuning_performed": False,
        "verdict": verdict,
        "claim_boundary": protocol.claim_boundary,
    }


def _build_manifest_core(
    *,
    protocol: RelationshipP4PhysicalActuationProtocol,
    checks: Mapping[str, Any],
    preflight_passed: bool,
    verdict: str,
    payload_bytes: Mapping[str, bytes],
) -> dict[str, Any]:
    return {
        "schema_version": P4_PHYSICAL_ACTUATION_MANIFEST_SCHEMA_VERSION,
        "protocol_id": protocol.protocol_id,
        "prompt_set_sha256": protocol.prompt_set_sha256,
        "source_hash_mode": protocol.source_hash_mode,
        "source_sha256": protocol.source_sha256_payload(),
        "authenticated_input": _authenticated_input_payload(),
        "execution_attestation_id": _EXECUTION_ATTESTATION_ID,
        "receipt_count": _RECEIPT_COUNT,
        "arm_evaluation_count": _ARM_EVALUATION_COUNT,
        "runtime_forward_invocation_count": (_RUNTIME_FORWARD_INVOCATION_COUNT),
        "runtime_forward_invocation_semantics": (_RUNTIME_FORWARD_INVOCATION_SEMANTICS),
        "files": [
            {
                "path": path,
                "byte_count": len(payload),
                "sha256": _sha256_bytes(payload),
            }
            for path, payload in sorted(payload_bytes.items())
        ],
        "checks": dict(checks),
        "preflight_passed": preflight_passed,
        "failure_retuning_performed": False,
        "verdict": verdict,
        "formal_evidence_authorized": False,
        "production_active_authorized": False,
        "four_capability_claim_authorized": False,
        "applied_gpu_residual_bundle_derivable_claimed": False,
    }


def _authenticated_input_payload() -> dict[str, Any]:
    return {
        "complete_fit_root_required": True,
        "standalone_bundle_allowed": False,
        "fit_artifact_id": _INPUT_ARTIFACT_ID,
        "fit_protocol_id": _INPUT_FIT_PROTOCOL_ID,
        "fit_bundle_sha256": _INPUT_BUNDLE_SHA256,
        "fit_manifest_sha256": _INPUT_MANIFEST_SHA256,
        "fit_campaign_id": _INPUT_CAMPAIGN_ID,
        "fit_campaign_manifest_sha256": _INPUT_CAMPAIGN_MANIFEST_SHA256,
        "model_id": _MODEL_ID,
        "verified_revision": _MODEL_REVISION,
        "model_weights_sha256": _MODEL_WEIGHTS_SHA256,
        "execution_assets_sha256": _EXECUTION_ASSETS_SHA256,
        "execution_profile_id": _EXECUTION_PROFILE_ID,
        "execution_attestation_id": _EXECUTION_ATTESTATION_ID,
        "layer_index": _LAYER_INDEX,
        "residual_width": _RESIDUAL_WIDTH,
        "steering_rank": _STEERING_RANK,
        "control_norm_cap_ratio": _CONTROL_NORM_CAP_RATIO,
    }


def _validate_physical_actuation_artifact(
    *,
    output_dir: pathlib.Path,
    input_fit_root: pathlib.Path,
    campaign_manifest_path: pathlib.Path,
    protocol: RelationshipP4PhysicalActuationProtocol,
    authenticated: _AuthenticatedFitInput,
) -> RelationshipP4PhysicalActuationRunResult:
    del input_fit_root, campaign_manifest_path
    if not output_dir.is_dir():
        raise FileNotFoundError(f"physical-actuation output root is not a directory: {output_dir}")
    manifest_path = output_dir / _MANIFEST_FILE
    manifest = _load_json_object(manifest_path)
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "protocol_id",
            "prompt_set_sha256",
            "source_hash_mode",
            "source_sha256",
            "authenticated_input",
            "execution_attestation_id",
            "receipt_count",
            "arm_evaluation_count",
            "runtime_forward_invocation_count",
            "runtime_forward_invocation_semantics",
            "files",
            "checks",
            "preflight_passed",
            "failure_retuning_performed",
            "verdict",
            "formal_evidence_authorized",
            "production_active_authorized",
            "four_capability_claim_authorized",
            "applied_gpu_residual_bundle_derivable_claimed",
            "artifact_id",
        },
        "physical-actuation manifest",
    )
    for key, expected in (
        ("schema_version", P4_PHYSICAL_ACTUATION_MANIFEST_SCHEMA_VERSION),
        ("protocol_id", protocol.protocol_id),
        ("prompt_set_sha256", protocol.prompt_set_sha256),
        ("source_hash_mode", protocol.source_hash_mode),
        ("source_sha256", protocol.source_sha256_payload()),
        ("authenticated_input", _authenticated_input_payload()),
        ("execution_attestation_id", _EXECUTION_ATTESTATION_ID),
        ("receipt_count", _RECEIPT_COUNT),
        ("arm_evaluation_count", _ARM_EVALUATION_COUNT),
        (
            "runtime_forward_invocation_count",
            _RUNTIME_FORWARD_INVOCATION_COUNT,
        ),
        (
            "runtime_forward_invocation_semantics",
            _RUNTIME_FORWARD_INVOCATION_SEMANTICS,
        ),
        ("failure_retuning_performed", False),
        ("formal_evidence_authorized", False),
        ("production_active_authorized", False),
        ("four_capability_claim_authorized", False),
        ("applied_gpu_residual_bundle_derivable_claimed", False),
    ):
        _require_literal(manifest[key], expected, f"manifest {key}")
    manifest_core = {key: value for key, value in manifest.items() if key != "artifact_id"}
    expected_artifact_id = _sha256_bytes(_canonical_bytes(manifest_core))
    _require_literal(manifest["artifact_id"], expected_artifact_id, "manifest artifact_id")
    file_rows = _require_list(manifest["files"], "manifest files")
    if len(file_rows) != _RECEIPT_COUNT + 2:
        raise ValueError("manifest payload file count drift")
    seen_paths: set[str] = set()
    receipts: list[dict[str, Any]] = []
    for row in file_rows:
        entry = _require_mapping(row, "manifest file")
        _require_exact_keys(entry, {"path", "byte_count", "sha256"}, "manifest file")
        relative_path = _require_relative_posix_path(entry["path"], "manifest file path")
        if relative_path in seen_paths:
            raise ValueError(f"duplicate manifest file path {relative_path!r}")
        seen_paths.add(relative_path)
        payload_path = output_dir / pathlib.PurePosixPath(relative_path)
        payload = payload_path.read_bytes()
        _require_literal(entry["byte_count"], len(payload), f"{relative_path} byte_count")
        _require_literal(entry["sha256"], _sha256_bytes(payload), f"{relative_path} sha256")
        if relative_path.startswith(f"{_RECEIPT_DIRECTORY}/"):
            receipt = _load_json_object(payload_path)
            receipt_id = _require_sha256_value(
                receipt.get("receipt_id"),
                f"{relative_path} receipt_id",
            )
            _require_literal(
                relative_path,
                f"{_RECEIPT_DIRECTORY}/{receipt_id}.json",
                "content-addressed receipt path",
            )
            receipts.append(receipt)
    actual_files = {path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file()}
    expected_files = {entry["path"] for entry in file_rows} | {_MANIFEST_FILE}
    if not {_REPORT_FILE, _ATTESTATION_FILE}.issubset(seen_paths):
        raise ValueError("manifest is missing report or live attestation")
    if actual_files != expected_files:
        raise ValueError(
            "physical-actuation output file set drift: "
            f"actual={sorted(actual_files)!r}, expected={sorted(expected_files)!r}"
        )
    live_attestation = _load_json_object(output_dir / _ATTESTATION_FILE)
    _require_literal(
        live_attestation,
        authenticated.execution_attestation,
        "published live attestation versus fit attestation",
    )
    _validate_attestation_lineage(live_attestation)
    prompts = _build_frozen_prompt_set(protocol)
    derived = _derive_metrics_and_checks(
        protocol=protocol,
        prompts=prompts,
        bundle=authenticated.bundle,
        receipts=receipts,
    )
    checks = derived["checks"]
    preflight_passed = all(checks[name] for name in _STOPPING_CHECK_NAMES)
    verdict = _verdict(preflight_passed)
    expected_report = _build_report_payload(
        protocol=protocol,
        authenticated=authenticated,
        derived=derived,
        live_attestation=live_attestation,
        preflight_passed=preflight_passed,
        verdict=verdict,
    )
    report = _load_json_object(output_dir / _REPORT_FILE)
    _require_literal(report, expected_report, "physical-actuation report")
    _require_literal(manifest["checks"], checks, "manifest checks")
    _require_literal(
        manifest["preflight_passed"],
        preflight_passed,
        "manifest preflight_passed",
    )
    _require_literal(manifest["verdict"], verdict, "manifest verdict")
    expected_manifest_core = _build_manifest_core(
        protocol=protocol,
        checks=checks,
        preflight_passed=preflight_passed,
        verdict=verdict,
        payload_bytes={
            entry["path"]: (output_dir / pathlib.PurePosixPath(entry["path"])).read_bytes() for entry in file_rows
        },
    )
    _require_literal(manifest_core, expected_manifest_core, "manifest core")
    return RelationshipP4PhysicalActuationRunResult(
        artifact_id=expected_artifact_id,
        protocol_id=protocol.protocol_id,
        input_fit_artifact_id=_INPUT_ARTIFACT_ID,
        execution_attestation_id=_EXECUTION_ATTESTATION_ID,
        preflight_passed=preflight_passed,
        verdict=verdict,
        output_dir=output_dir,
    )


def _validate_protocol_payload(raw: Mapping[str, Any]) -> None:
    required_keys = {
        "schema_version",
        "owner",
        "authenticated_input",
        "execution_profile",
        "dataset",
        "arms",
        "runtime_forward_invocation_contract",
        "stopping_checks",
        "source_hash_mode",
        "source_sha256",
        "output_contract",
        "evidence_firewall",
        "control_interpretation",
        "claim_boundary",
    }
    _require_exact_keys(raw, required_keys, "physical-actuation protocol")
    _require_literal(
        raw["schema_version"],
        P4_PHYSICAL_ACTUATION_PROTOCOL_SCHEMA_VERSION,
        "protocol schema_version",
    )
    _require_literal(
        raw["owner"],
        {
            "wheel": "vz-runtime",
            "orchestrator": ("volvence_zero.agent.relationship_p4_physical_residual_actuation"),
            "mode": "development_only_shadow_physical_actuation_preflight",
            "new_runtime_slot_added": False,
        },
        "protocol owner",
    )
    _require_literal(
        raw["authenticated_input"],
        _authenticated_input_payload(),
        "protocol authenticated_input",
    )
    expected_execution = {
        "profile_id": _EXECUTION_PROFILE_ID,
        **_PROFILE_FACTS,
        "runtime_mode": "strict-local",
        "model_id": _MODEL_ID,
        "verified_revision": _MODEL_REVISION,
        "model_weights_sha256": _MODEL_WEIGHTS_SHA256,
        "execution_assets_sha256": _EXECUTION_ASSETS_SHA256,
        "execution_attestation_id": _EXECUTION_ATTESTATION_ID,
        "layer_index": _LAYER_INDEX,
        "residual_width": _RESIDUAL_WIDTH,
        "capture_use_cache": False,
        "direct_apply_use_cache": False,
        "generation_performed": False,
    }
    _require_literal(raw["execution_profile"], expected_execution, "protocol execution_profile")
    dataset = _require_mapping(raw["dataset"], "protocol dataset")
    _require_exact_keys(
        dataset,
        set(_DATASET_RECIPE) | {"prompt_set_sha256", "synthetic_heldout_only"},
        "protocol dataset",
    )
    for key, expected in _DATASET_RECIPE.items():
        _require_literal(dataset[key], expected, f"protocol dataset.{key}")
    _require_literal(dataset["synthetic_heldout_only"], True, "protocol dataset heldout")
    _require_sha256_value(dataset["prompt_set_sha256"], "protocol dataset.prompt_set_sha256")
    _require_literal(raw["arms"], _ARM_SEMANTICS, "protocol arms")
    _require_literal(
        raw["runtime_forward_invocation_contract"],
        {
            "count_name": "runtime_forward_invocation_count",
            "expected_count": _RUNTIME_FORWARD_INVOCATION_COUNT,
            "successful_return_observed": True,
            "per_receipt_order": list(_ARM_ORDER),
            "runtime_api_by_arm": dict(_RUNTIME_FORWARD_INVOCATION_APIS),
            "one_model_forward_per_api_return_established_by_pinned_source": True,
            "semantics": _RUNTIME_FORWARD_INVOCATION_SEMANTICS,
        },
        "protocol runtime_forward_invocation_contract",
    )
    _require_literal(
        raw["stopping_checks"],
        list(_STOPPING_CHECK_NAMES),
        "protocol stopping_checks",
    )
    _require_literal(raw["source_hash_mode"], _SOURCE_HASH_MODE, "protocol source_hash_mode")
    source_sha256 = _require_mapping(raw["source_sha256"], "source_sha256")
    if tuple(source_sha256) != _CRITICAL_SOURCE_PATHS:
        raise ValueError("physical-actuation protocol source path/order drift")
    for source_path in _CRITICAL_SOURCE_PATHS:
        _require_sha256_value(source_sha256[source_path], f"source_sha256.{source_path}")
    _require_literal(
        raw["output_contract"],
        {
            "create_only": True,
            "atomic_directory_publish": True,
            "content_addressed": True,
            "per_sample_receipts": True,
            "offline_validator_loads_gpu": False,
            "offline_validator_imports_torch": False,
            "offline_validator_recomputes_vectors_metrics_checks_verdict": True,
            "offline_validator_recomputes_owner_belief_and_deltas": True,
            "applied_gpu_residual_bundle_derivable": False,
            "failed_threshold_verdict": "failed_stop_no_retuning",
            "contract_violation_behavior": "fail_loudly_without_publication",
        },
        "protocol output_contract",
    )
    _require_literal(raw["evidence_firewall"], _EVIDENCE_FIREWALL, "protocol evidence_firewall")
    _require_literal(
        raw["control_interpretation"],
        _CONTROL_INTERPRETATION,
        "protocol control_interpretation",
    )
    _require_literal(raw["claim_boundary"], _CLAIM_BOUNDARY, "protocol claim_boundary")


def _verdict(preflight_passed: bool) -> str:
    return (
        "synthetic_proxy_physical_residual_actuation_passed_development_only"
        if preflight_passed
        else "failed_stop_no_retuning"
    )


def _extract_residual(substrate: Any, *, layer_index: int, residual_width: int) -> tuple[float, ...]:
    if substrate.model_id != _MODEL_ID or substrate.is_frozen is not True:
        raise ValueError("substrate snapshot model/frozen lineage drift")
    activations = tuple(row.activation for row in substrate.residual_activations if row.layer_index == layer_index)
    if len(activations) != 1 or len(activations[0]) != residual_width:
        raise ValueError(f"physical-actuation capture requires exactly one full-width residual at layer {layer_index}")
    values = tuple(float(value) for value in activations[0])
    if not all(math.isfinite(value) for value in values):
        raise ValueError("physical-actuation residual contains non-finite values")
    if _vector_norm(values) <= 0.0:
        raise ValueError("physical-actuation residual norm must be positive")
    return values


def _encode_vector(values: Sequence[float]) -> dict[str, Any]:
    vector = tuple(float(value) for value in values)
    if len(vector) != _RESIDUAL_WIDTH:
        raise ValueError(f"retained physical-actuation vector width must be {_RESIDUAL_WIDTH}")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("retained physical-actuation vector must be finite")
    payload = _vector_bytes(vector)
    return {
        "encoding": _VECTOR_ENCODING,
        "length": len(vector),
        "sha256": _sha256_bytes(payload),
        "data": base64.b64encode(payload).decode("ascii"),
    }


def _decode_token_logits(payload: object, *, label: str) -> tuple[float, ...]:
    values = _require_list(payload, label)
    if not values:
        raise ValueError(f"{label} must be nonempty")
    decoded = tuple(_require_number_value(value, f"{label}[{index}]") for index, value in enumerate(values))
    if any(not math.isfinite(value) or value < 0.0 for value in decoded):
        raise ValueError(f"{label} must contain finite non-negative values")
    return decoded


def _decode_vector(payload: object, *, label: str) -> tuple[tuple[float, ...], bytes]:
    row = _require_mapping(payload, label)
    _require_exact_keys(row, {"encoding", "length", "sha256", "data"}, label)
    _require_literal(row["encoding"], _VECTOR_ENCODING, f"{label}.encoding")
    _require_literal(row["length"], _RESIDUAL_WIDTH, f"{label}.length")
    digest = _require_sha256_value(row["sha256"], f"{label}.sha256")
    data = _require_text_value(row["data"], f"{label}.data")
    try:
        raw = base64.b64decode(data.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ValueError(f"{label}.data is not strict base64") from exc
    expected_bytes = _RESIDUAL_WIDTH * 8
    if len(raw) != expected_bytes:
        raise ValueError(f"{label} decoded byte length drift: {len(raw)} != {expected_bytes}")
    if _sha256_bytes(raw) != digest:
        raise ValueError(f"{label} decoded SHA-256 mismatch")
    vector = tuple(struct.unpack(f"!{_RESIDUAL_WIDTH}d", raw))
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{label} decoded non-finite values")
    return vector, raw


def _vector_bytes(values: Sequence[float]) -> bytes:
    vector = tuple(float(value) for value in values)
    return struct.pack(f"!{len(vector)}d", *vector)


def _vector_sha256(values: Sequence[float]) -> str:
    return _sha256_bytes(_vector_bytes(values))


def _vector_norm(values: Sequence[float]) -> float:
    return math.sqrt(math.fsum(value * value for value in values))


def _verify_critical_sources(
    protocol: RelationshipP4PhysicalActuationProtocol,
) -> None:
    for relative_path, expected_sha256 in protocol.source_sha256:
        path = _REPOSITORY_ROOT / pathlib.PurePosixPath(relative_path)
        actual_sha256 = _source_text_sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "P4 physical-actuation critical source SHA-256 mismatch: "
                f"{relative_path}: {actual_sha256} != {expected_sha256}"
            )


def _source_text_sha256(path: pathlib.Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"critical source must not contain UTF-8 BOM: {path}")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"critical source is not strict UTF-8: {path}") from exc
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return _sha256_bytes(normalized.encode("utf-8"))


def _canonical_bytes(value: object, *, newline: bool = True) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _write_create_bytes(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _load_json_object(path: pathlib.Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FileNotFoundError(f"required JSON artifact is unavailable: {path}") from exc
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid strict UTF-8 JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact root must be an object: {path}")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _require_literal(actual: object, expected: object, label: str) -> None:
    if actual != expected or type(actual) is not type(expected):
        raise ValueError(f"{label} drift: {actual!r} != {expected!r}")


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} key drift: missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}"
        )


def _require_text_value(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def _require_relative_posix_path(value: object, label: str) -> str:
    text = _require_text_value(value, label)
    pure = pathlib.PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts or "\\" in text:
        raise ValueError(f"{label} must be a safe relative POSIX path")
    if pure.as_posix() != text:
        raise ValueError(f"{label} is not canonical POSIX text")
    return text


def _require_int_value(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an exact integer")
    return value


def _require_number_value(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _require_sha256_value(value: object, label: str) -> str:
    text = _require_text_value(value, label)
    _require_sha256(text, label)
    return text


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or value.lower() != value or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FileNotFoundError(f"required artifact file is unavailable: {path}") from exc
    return digest.hexdigest()


__all__ = (
    "P4_PHYSICAL_ACTUATION_MANIFEST_SCHEMA_VERSION",
    "P4_PHYSICAL_ACTUATION_PROTOCOL_SCHEMA_VERSION",
    "P4_PHYSICAL_ACTUATION_RECEIPT_SCHEMA_VERSION",
    "P4_PHYSICAL_ACTUATION_REPORT_SCHEMA_VERSION",
    "PhysicalActuationPrompt",
    "RelationshipP4PhysicalActuationProtocol",
    "RelationshipP4PhysicalActuationRunResult",
    "load_relationship_p4_physical_actuation_protocol",
    "relationship_p4_physical_actuation_protocol_path",
    "run_relationship_p4_physical_residual_actuation",
    "validate_relationship_p4_physical_residual_actuation",
)
