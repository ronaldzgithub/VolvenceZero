"""Create-only Windows/CUDA steering-artifact fit prerequisite for P4.

This module is deliberately a thin evidence orchestrator.  The reader and
executor mathematics remain owned by
``vz-runtime.agent.steering_artifact_training.fit_steering_artifact_bundle``;
this lane only freezes one Qwen2.5-1.5B recipe, constructs the canonical strict
substrate runtime, and publishes tamper-evident offline evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
import pathlib
import shutil
import tempfile
from typing import Any, Callable, Mapping


P4_STEERING_FIT_PROTOCOL_SCHEMA_VERSION = (
    "relationship-p4-windows-cuda-steering-fit-qwen25-15b.v1"
)
P4_STEERING_FIT_REPORT_SCHEMA_VERSION = (
    "relationship-p4-windows-cuda-steering-fit-report.v1"
)
P4_STEERING_FIT_MANIFEST_SCHEMA_VERSION = (
    "relationship-p4-windows-cuda-steering-fit-manifest.v1"
)

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
_REPOSITORY_ROOT = _PACKAGE_ROOT.parents[4]
_DEFAULT_PROTOCOL_PATH = (
    _PACKAGE_ROOT
    / "protocols"
    / "relationship_p4_windows_cuda_steering_fit_qwen25_15b_v1.json"
)
_BUNDLE_FILE = "steering_artifact_bundle.json"
_REPORT_FILE = "steering_artifact_fit_report.json"
_ATTESTATION_FILE = "execution_attestation.json"
_MANIFEST_FILE = "manifest.json"
_REQUIRED_FILES = (
    _BUNDLE_FILE,
    _REPORT_FILE,
    _ATTESTATION_FILE,
    _MANIFEST_FILE,
)
_PAYLOAD_FILES = (_BUNDLE_FILE, _REPORT_FILE, _ATTESTATION_FILE)
_SOURCE_HASH_MODE = "utf8_lf_canonical_v1"
_DERIVED_NLL_REL_TOLERANCE = 0.0
_DERIVED_NLL_ABS_TOLERANCE = 1e-12
_CRITICAL_SOURCE_PATHS = (
    "packages/vz-contracts/src/volvence_zero/steering_contracts.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_conditional_steering_screen.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_conflict_instrument.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_proof_benchmark.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_rate_distortion_evidence.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_read_steer_prereq.py",
    "packages/vz-runtime/src/volvence_zero/agent/relationship_p4_steering_artifact_fit.py",
    "packages/vz-runtime/src/volvence_zero/agent/steering_artifact_training.py",
    "packages/vz-substrate/src/volvence_zero/substrate/residual_backend.py",
    "packages/vz-substrate/src/volvence_zero/substrate/residual_contracts.py",
    "packages/vz-substrate/src/volvence_zero/substrate/steered_action_scoring.py",
    "scripts/run_relationship_lab_p4_steering_artifact_fit.py",
)

_OWNER_WHEEL = "vz-runtime"
_OWNER_CALLABLE = (
    "volvence_zero.agent.steering_artifact_training."
    "fit_steering_artifact_bundle"
)
_ORCHESTRATOR = (
    "volvence_zero.agent.relationship_p4_steering_artifact_fit"
)
_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
_MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
_MODEL_WEIGHTS_SHA256 = (
    "fb8c44c48b8359fdd306cdc5f473d7c04d88955013f0dd8549f266e248194da4"
)
_EXECUTION_ASSETS_SHA256 = (
    "bbb5446f8d802b437c2fc7e2cefcdabb996bbd4bc657fe155ea015d30a841bb0"
)
_CLAIM_BOUNDARY = (
    "This create-only Windows/CUDA prerequisite fits a fresh Qwen2.5-1.5B "
    "model-bound reader, conditional executor, and matched sensor-off "
    "executor by delegating once to the existing vz-runtime owner. It may "
    "establish only that a new artifact passes the frozen ETA proxy "
    "thresholds on the exact verified model and strict execution profile. "
    "It does not reuse or relabel the missing historical C3 weights, does "
    "not train a dialogue gate, does not generate independent-subject or "
    "long-companion evidence, does not authorize product ACTIVE, and does "
    "not prove complete Readable, Steerable, Learnable, Appendable, or the "
    "four-capability system claim. Any failed stopping threshold is "
    "recorded unchanged and stops this recipe without retuning. A missing or "
    "non-unconditional sensor-off executor is quarantined from the published "
    "bundle so an invalid bundle never escapes as a consumable artifact."
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

_FIT_CONFIGURATION = {
    "injection_layer_index": 20,
    "residual_width": 1536,
    "steering_rank": 8,
    "conditional_executor_updates": 80,
    "sensor_off_executor_updates": 80,
    "executor_learning_rate": 0.01,
    "batch_size": 32,
    "reader_ridge_lambda": 10.0,
    "control_norm_cap_ratio": 0.25,
    "seed": 0,
    "expected_train_row_count": 307,
    "expected_heldout_row_count": 165,
    "corpus": {
        "seed": 20260802,
        "objective_count": 8,
        "corridor_count": 2,
        "extra_edge_probability": 0.35,
        "train_route_count": 64,
        "heldout_route_count": 24,
        "train_lengths": [2, 3],
        "heldout_lengths": [3, 4],
    },
}

_STOPPING_THRESHOLDS = {
    "reader_heldout_accuracy_min": 0.8,
    "heldout_gain_vs_noop_nll_strictly_positive": True,
    "heldout_conditional_advantage_nll_strictly_positive": True,
    "substrate_trainable_parameter_count": 0,
    "free_bias_present": False,
    "zero_code_strict_noop": True,
    "sensor_off_condition_code_rows_identical": True,
    "failure_action": (
        "stop_publish_quarantined_failure_without_retuning"
    ),
}

_OUTPUT_CONTRACT = {
    "create_only": True,
    "immutable": True,
    "content_addressed": True,
    "required_files": list(_REQUIRED_FILES),
    "validate_existing_loads_gpu": False,
    "validate_existing_rehashes_every_payload_file": True,
    "derived_nll_consistency": {
        "relative_tolerance": _DERIVED_NLL_REL_TOLERANCE,
        "absolute_tolerance": _DERIVED_NLL_ABS_TOLERANCE,
        "gain_vs_noop_formula": (
            "heldout_noop_nll-heldout_online_steer_nll"
        ),
        "conditional_advantage_formula": (
            "heldout_sensor_off_nll-heldout_online_steer_nll"
        ),
    },
    "sensor_off_mismatch_publication": (
        "omit_invalid_optional_executor_and_publish_failed_root"
    ),
    "quarantine_flag": "sensor_off_executor_quarantined",
}

_INSTRUMENTAL_FIT_EXECUTION = {
    "input_tokenization": "raw_tokenizer",
    "tokenizer_truncation_argument": False,
    "fail_on_truncation": True,
    "max_length": 32768,
    "use_cache": False,
    "prefix_cache": True,
    "prefix_cache_semantics": (
        "lower_stack_hidden_replay_not_generation_kv_cache"
    ),
    "exclusive_cudnn_sdpa_context": True,
    "generation_attestation_applies_to_fit_operation": False,
}

_EVIDENCE_FIREWALL = {
    "old_c3_bundle_reused": False,
    "old_c3_lineage_relabelled": False,
    "dialogue_domain_output_observed": False,
    "evaluation_feedback_to_fit": False,
    "gate_learning_performed": False,
    "product_wiring_changed": False,
    "standalone_bundle_consumption_allowed": False,
    "complete_artifact_root_required": True,
    "production_active_authorized": False,
    "formal_evidence_authorized": False,
}

_ATTESTATION_KEYS = {
    "schema_version",
    "profile_id",
    "preset_name",
    "model_id",
    "model_revision",
    "model_weights_sha256",
    "execution_assets_sha256",
    "runtime_origin",
    "platform_system",
    "platform_release",
    "device",
    "device_name",
    "python_version",
    "torch_version",
    "transformers_version",
    "cuda_version",
    "cudnn_version",
    "device_compute_capability",
    "attention_implementation",
    "sdpa_backend",
    "sdpa_backend_policy",
    "sdpa_backend_exclusive",
    "generation_use_cache",
    "require_generation_chat_template",
    "generation_capture_strategy",
    "capture_failure_mode",
    "context_window_tokens",
    "local_files_only",
    "fallback_mode",
    "fail_on_truncation",
    "model_dtype",
    "hidden_size",
    "model_max_position_embeddings",
    "hook_layer_indices",
    "attestation_id",
}

_OWNER_REPORT_KEYS = {
    "train_row_count",
    "heldout_row_count",
    "reader_heldout_accuracy",
    "heldout_noop_nll",
    "heldout_online_steer_nll",
    "heldout_sensor_off_nll",
    "heldout_gain_vs_noop_nll",
    "heldout_conditional_advantage_nll",
    "reader_ridge_lambda",
    "executor_updates",
    "executor_learning_rate",
    "steering_rank",
    "seed",
    "control_norm_cap_ratio",
    "free_bias_present",
    "zero_code_strict_noop",
    "substrate_trainable_parameter_count",
    "reader_executor_frozen_for_dialogue",
    "description",
}


@dataclass(frozen=True)
class RelationshipP4SteeringFitProtocol:
    """Frozen scalar view of the content-addressed prerequisite protocol."""

    protocol_id: str
    profile_id: str
    source_sha256: tuple[tuple[str, str], ...]
    source_hash_mode: str = _SOURCE_HASH_MODE
    model_id: str = _MODEL_ID
    verified_revision: str = _MODEL_REVISION
    model_weights_sha256: str = _MODEL_WEIGHTS_SHA256
    execution_assets_sha256: str = _EXECUTION_ASSETS_SHA256
    injection_layer_index: int = 20
    residual_width: int = 1536
    steering_rank: int = 8
    conditional_executor_updates: int = 80
    sensor_off_executor_updates: int = 80
    executor_learning_rate: float = 0.01
    batch_size: int = 32
    reader_ridge_lambda: float = 10.0
    control_norm_cap_ratio: float = 0.25
    fit_seed: int = 0
    expected_train_row_count: int = 307
    expected_heldout_row_count: int = 165
    corpus_seed: int = 20260802
    objective_count: int = 8
    corridor_count: int = 2
    extra_edge_probability: float = 0.35
    train_route_count: int = 64
    heldout_route_count: int = 24
    train_lengths: tuple[int, ...] = (2, 3)
    heldout_lengths: tuple[int, ...] = (3, 4)
    claim_boundary: str = _CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        _require_sha256(self.protocol_id, "protocol_id")
        _require_sha256(self.profile_id, "profile_id")
        _require_sha256(self.model_weights_sha256, "model_weights_sha256")
        _require_sha256(
            self.execution_assets_sha256,
            "execution_assets_sha256",
        )
        if tuple(path for path, _ in self.source_sha256) != _CRITICAL_SOURCE_PATHS:
            raise ValueError("P4 steering-fit critical source path set drift")
        if self.source_hash_mode != _SOURCE_HASH_MODE:
            raise ValueError("P4 steering-fit source hash mode drift")
        for path, digest in self.source_sha256:
            _require_relative_posix_path(path, "critical source path")
            _require_sha256(digest, f"critical source {path}")
        if self.verified_revision != _MODEL_REVISION:
            raise ValueError("P4 steering-fit model revision drift")
        if self.claim_boundary != _CLAIM_BOUNDARY:
            raise ValueError("P4 steering-fit claim boundary drift")

    def fit_configuration_payload(self) -> dict[str, object]:
        return json.loads(json.dumps(_FIT_CONFIGURATION))

    def stopping_thresholds_payload(self) -> dict[str, object]:
        return dict(_STOPPING_THRESHOLDS)

    def source_sha256_payload(self) -> dict[str, str]:
        return dict(self.source_sha256)


@dataclass(frozen=True)
class RelationshipP4SteeringFitRunResult:
    artifact_id: str
    protocol_id: str
    bundle_id: str
    execution_attestation_id: str
    prerequisite_passed: bool
    verdict: str
    output_dir: pathlib.Path


def relationship_p4_steering_fit_protocol_path() -> pathlib.Path:
    return _DEFAULT_PROTOCOL_PATH


def load_relationship_p4_steering_fit_protocol(
    path: pathlib.Path | None = None,
) -> RelationshipP4SteeringFitProtocol:
    protocol_path = pathlib.Path(path or _DEFAULT_PROTOCOL_PATH)
    raw = _load_json_object(protocol_path)
    _validate_protocol_payload(raw)
    execution = _require_mapping(raw["execution_profile"], "execution_profile")
    fit = _require_mapping(raw["artifact_fit"], "artifact_fit")
    corpus = _require_mapping(fit["corpus"], "artifact_fit.corpus")
    source_sha256 = _require_mapping(raw["source_sha256"], "source_sha256")
    return RelationshipP4SteeringFitProtocol(
        protocol_id=_sha256_bytes(_canonical_bytes(raw)),
        profile_id=_require_sha256_value(
            execution["profile_id"],
            "execution_profile.profile_id",
        ),
        source_sha256=tuple(
            (
                path,
                _require_sha256_value(
                    source_sha256[path],
                    f"source_sha256.{path}",
                ),
            )
            for path in _CRITICAL_SOURCE_PATHS
        ),
        source_hash_mode=_require_text_value(
            raw["source_hash_mode"],
            "source_hash_mode",
        ),
        model_id=_MODEL_ID,
        verified_revision=_MODEL_REVISION,
        model_weights_sha256=_MODEL_WEIGHTS_SHA256,
        execution_assets_sha256=_EXECUTION_ASSETS_SHA256,
        injection_layer_index=_require_int_value(
            fit["injection_layer_index"],
            "artifact_fit.injection_layer_index",
        ),
        residual_width=_require_int_value(
            fit["residual_width"],
            "artifact_fit.residual_width",
        ),
        steering_rank=_require_int_value(
            fit["steering_rank"],
            "artifact_fit.steering_rank",
        ),
        conditional_executor_updates=_require_int_value(
            fit["conditional_executor_updates"],
            "artifact_fit.conditional_executor_updates",
        ),
        sensor_off_executor_updates=_require_int_value(
            fit["sensor_off_executor_updates"],
            "artifact_fit.sensor_off_executor_updates",
        ),
        executor_learning_rate=_require_float_value(
            fit["executor_learning_rate"],
            "artifact_fit.executor_learning_rate",
        ),
        batch_size=_require_int_value(
            fit["batch_size"],
            "artifact_fit.batch_size",
        ),
        reader_ridge_lambda=_require_float_value(
            fit["reader_ridge_lambda"],
            "artifact_fit.reader_ridge_lambda",
        ),
        control_norm_cap_ratio=_require_float_value(
            fit["control_norm_cap_ratio"],
            "artifact_fit.control_norm_cap_ratio",
        ),
        fit_seed=_require_int_value(fit["seed"], "artifact_fit.seed"),
        expected_train_row_count=_require_int_value(
            fit["expected_train_row_count"],
            "artifact_fit.expected_train_row_count",
        ),
        expected_heldout_row_count=_require_int_value(
            fit["expected_heldout_row_count"],
            "artifact_fit.expected_heldout_row_count",
        ),
        corpus_seed=_require_int_value(
            corpus["seed"],
            "artifact_fit.corpus.seed",
        ),
        objective_count=_require_int_value(
            corpus["objective_count"],
            "artifact_fit.corpus.objective_count",
        ),
        corridor_count=_require_int_value(
            corpus["corridor_count"],
            "artifact_fit.corpus.corridor_count",
        ),
        extra_edge_probability=_require_float_value(
            corpus["extra_edge_probability"],
            "artifact_fit.corpus.extra_edge_probability",
        ),
        train_route_count=_require_int_value(
            corpus["train_route_count"],
            "artifact_fit.corpus.train_route_count",
        ),
        heldout_route_count=_require_int_value(
            corpus["heldout_route_count"],
            "artifact_fit.corpus.heldout_route_count",
        ),
        train_lengths=tuple(
            _require_int_value(value, "artifact_fit.corpus.train_lengths")
            for value in _require_list(
                corpus["train_lengths"],
                "artifact_fit.corpus.train_lengths",
            )
        ),
        heldout_lengths=tuple(
            _require_int_value(value, "artifact_fit.corpus.heldout_lengths")
            for value in _require_list(
                corpus["heldout_lengths"],
                "artifact_fit.corpus.heldout_lengths",
            )
        ),
        claim_boundary=_require_text_value(
            raw["claim_boundary"],
            "claim_boundary",
        ),
    )


def run_relationship_p4_steering_artifact_fit(
    *,
    output_dir: pathlib.Path,
    protocol_path: pathlib.Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> RelationshipP4SteeringFitRunResult:
    """Fit once and atomically publish one new model-bound evidence root."""

    output = pathlib.Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(
            f"P4 steering-fit output is create-only and already exists: {output}"
        )
    protocol = load_relationship_p4_steering_fit_protocol(protocol_path)
    _verify_critical_sources(protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = pathlib.Path(
        tempfile.mkdtemp(
            dir=output.parent,
            prefix=f".{output.name}.building.",
        )
    )
    published = False
    try:
        runtime = _build_strict_runtime(protocol)
        attestation = runtime.execution_attestation
        if attestation is None:
            raise RuntimeError(
                "strict P4 steering-fit runtime did not publish an execution "
                "attestation"
            )
        attestation_payload = {
            **attestation.to_payload(),
            "attestation_id": attestation.attestation_id,
        }
        _validate_execution_attestation_payload(
            attestation_payload,
            protocol=protocol,
        )
        fit_result = _fit_with_runtime(
            protocol=protocol,
            runtime=runtime,
            progress=progress,
        )
        bundle, sensor_off_quarantine = (
            _prepare_owner_bundle_for_publication(
                fit_result.bundle,
                protocol=protocol,
            )
        )
        owner_report = asdict(fit_result.report)
        checks = _validate_bundle_and_owner_report(
            bundle=bundle,
            owner_report=owner_report,
            protocol=protocol,
            sensor_off_quarantine=sensor_off_quarantine,
        )
        prerequisite_passed = all(checks.values())
        owner_prerequisite_passed = all(
            value
            for name, value in checks.items()
            if name != "sensor_off_condition_code_rows_identical"
        )
        if (
            fit_result.report.prerequisite_passed
            is not owner_prerequisite_passed
        ):
            raise ValueError(
                "vz-runtime owner prerequisite verdict differs from the "
                "owner-report portion of the frozen P4 steering-fit gate"
            )
        verdict = _verdict(prerequisite_passed)
        bundle_bytes = (bundle.to_json() + "\n").encode("utf-8")
        bundle_sha256 = _sha256_bytes(bundle_bytes)
        report_payload = _build_report_payload(
            protocol=protocol,
            bundle_id=bundle.bundle_id,
            bundle_sha256=bundle_sha256,
            execution_attestation_id=attestation.attestation_id,
            owner_report=owner_report,
            checks=checks,
            prerequisite_passed=prerequisite_passed,
            verdict=verdict,
            sensor_off_quarantine=sensor_off_quarantine,
            numpy_version=_numpy_version(),
        )
        report_bytes = _canonical_bytes(report_payload)
        attestation_bytes = _canonical_bytes(attestation_payload)
        payload_bytes = {
            _BUNDLE_FILE: bundle_bytes,
            _REPORT_FILE: report_bytes,
            _ATTESTATION_FILE: attestation_bytes,
        }
        for name, payload in payload_bytes.items():
            _write_create_bytes(temporary / name, payload)
        manifest_core = _build_manifest_core(
            protocol=protocol,
            bundle_id=bundle.bundle_id,
            execution_attestation_id=attestation.attestation_id,
            prerequisite_passed=prerequisite_passed,
            verdict=verdict,
            sensor_off_quarantine=sensor_off_quarantine,
            payload_bytes=payload_bytes,
        )
        artifact_id = _sha256_bytes(_canonical_bytes(manifest_core))
        _write_create_bytes(
            temporary / _MANIFEST_FILE,
            _canonical_bytes({**manifest_core, "artifact_id": artifact_id}),
        )
        _validate_relationship_p4_steering_fit_artifact(
            output_dir=temporary,
            protocol=protocol,
        )
        if output.exists():
            raise FileExistsError(
                "P4 steering-fit output appeared during create-only publication"
            )
        temporary.rename(output)
        published = True
        return RelationshipP4SteeringFitRunResult(
            artifact_id=artifact_id,
            protocol_id=protocol.protocol_id,
            bundle_id=bundle.bundle_id,
            execution_attestation_id=attestation.attestation_id,
            prerequisite_passed=prerequisite_passed,
            verdict=verdict,
            output_dir=output,
        )
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)


def validate_relationship_p4_steering_artifact_fit(
    *,
    output_dir: pathlib.Path,
    protocol_path: pathlib.Path | None = None,
) -> RelationshipP4SteeringFitRunResult:
    """Validate an existing artifact using only files and frozen contracts.

    This path deliberately never constructs a substrate runtime, imports
    ``torch``, probes CUDA, or invokes the fit owner.
    """

    protocol = load_relationship_p4_steering_fit_protocol(protocol_path)
    _verify_critical_sources(protocol)
    return _validate_relationship_p4_steering_fit_artifact(
        output_dir=pathlib.Path(output_dir).resolve(),
        protocol=protocol,
    )


def _build_strict_runtime(protocol: RelationshipP4SteeringFitProtocol) -> Any:
    from volvence_zero.substrate import (
        WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1,
        build_transformers_runtime_with_fallback,
    )

    profile = WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1
    if profile.profile_id != protocol.profile_id:
        raise ValueError(
            "public strict execution profile differs from the frozen P4 "
            "steering-fit protocol"
        )
    _require_literal(
        profile.to_payload(),
        _PROFILE_FACTS,
        "public strict execution profile",
    )
    return build_transformers_runtime_with_fallback(
        model_id=protocol.model_id,
        model_source=None,
        device="cuda",
        layer_indices=(protocol.injection_layer_index,),
        activation_width=protocol.residual_width,
        max_length=_PROFILE_FACTS["context_window_tokens"],
        fail_on_truncation=True,
        local_files_only=True,
        fallback_mode="deny",
        runtime_mode="strict-local",
        model_dtype="bfloat16",
        expected_model_weights_sha256=protocol.model_weights_sha256,
        execution_profile=profile,
        verified_model_revision=protocol.verified_revision,
        expected_execution_assets_sha256=(
            protocol.execution_assets_sha256
        ),
    )


def _fit_with_runtime(
    *,
    protocol: RelationshipP4SteeringFitProtocol,
    runtime: Any,
    progress: Callable[[str], None] | None,
) -> Any:
    from volvence_zero.agent.eta_conditional_steering_screen import (
        ACTION_PROMPT_SUFFIX,
    )
    from volvence_zero.agent.eta_conflict_instrument import (
        build_conflict_junction_rows,
    )
    from volvence_zero.agent.eta_proof_benchmark import (
        generate_eta_proof_corpus,
    )
    from volvence_zero.agent.eta_rate_distortion_evidence import (
        _action_options,
    )
    from volvence_zero.agent.steering_artifact_training import (
        fit_steering_artifact_bundle,
    )

    if runtime.fail_on_truncation is not True:
        raise ValueError(
            "P4 steering fit requires full tokenization with "
            "fail_on_truncation=True"
        )

    corpus = generate_eta_proof_corpus(
        seed=protocol.corpus_seed,
        objective_count=protocol.objective_count,
        corridor_count=protocol.corridor_count,
        extra_edge_probability=protocol.extra_edge_probability,
        train_route_count=protocol.train_route_count,
        heldout_route_count=protocol.heldout_route_count,
        train_lengths=protocol.train_lengths,
        heldout_lengths=protocol.heldout_lengths,
    )
    probe_rows = build_conflict_junction_rows(corpus, split="train")
    scorer = runtime.build_steered_action_scorer(
        action_options=_action_options(corpus.environment),
        injection_layer_index=protocol.injection_layer_index,
        prompt_suffix="",
        max_length=_PROFILE_FACTS["context_window_tokens"],
        control_norm_ratio=protocol.control_norm_cap_ratio,
        probe_texts=tuple(
            row.observation_text + ACTION_PROMPT_SUFFIX
            for row in probe_rows[:16]
        ),
        joint_training=False,
        prefix_cache=True,
    )
    return fit_steering_artifact_bundle(
        corpus=corpus,
        runtime=runtime,
        scorer=scorer,
        model_weights_sha256=protocol.model_weights_sha256,
        source_preregistration_sha256=protocol.protocol_id,
        injection_layer_index=protocol.injection_layer_index,
        residual_width=protocol.residual_width,
        steering_rank=protocol.steering_rank,
        executor_updates=protocol.conditional_executor_updates,
        executor_learning_rate=protocol.executor_learning_rate,
        reader_ridge_lambda=protocol.reader_ridge_lambda,
        batch_size=protocol.batch_size,
        seed=protocol.fit_seed,
        control_norm_cap_ratio=protocol.control_norm_cap_ratio,
        progress=progress,
    )


def _validate_protocol_payload(raw: Mapping[str, Any]) -> None:
    _require_exact_keys(
        raw,
        {
            "schema_version",
            "owner",
            "model",
            "execution_profile",
            "artifact_fit",
            "source_hash_mode",
            "source_sha256",
            "instrumental_fit_execution",
            "stopping_thresholds",
            "output_contract",
            "evidence_firewall",
            "claim_boundary",
        },
        "P4 steering-fit protocol",
    )
    if raw["schema_version"] != P4_STEERING_FIT_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("P4 steering-fit protocol schema drift")
    _require_literal(
        raw["owner"],
        {
            "wheel": _OWNER_WHEEL,
            "callable": _OWNER_CALLABLE,
            "orchestrator": _ORCHESTRATOR,
            "mode": "offline_rare_heavy_fit_prerequisite",
        },
        "P4 steering-fit owner",
    )
    _require_literal(
        raw["model"],
        {
            "model_id": _MODEL_ID,
            "verified_revision": _MODEL_REVISION,
            "model_weights_sha256": _MODEL_WEIGHTS_SHA256,
            "execution_assets_sha256": _EXECUTION_ASSETS_SHA256,
            "model_source": (
                "logical_model_id_resolved_from_verified_hf_cache"
            ),
            "local_snapshot_path_recorded": False,
        },
        "P4 steering-fit model",
    )
    execution = _require_mapping(raw["execution_profile"], "execution_profile")
    _require_exact_keys(
        execution,
        {"profile_id", "runtime_mode", *_PROFILE_FACTS},
        "P4 steering-fit execution_profile",
    )
    _require_sha256_value(execution["profile_id"], "execution_profile.profile_id")
    if execution["runtime_mode"] != "strict-local":
        raise ValueError("P4 steering-fit runtime mode drift")
    _require_literal(
        {name: execution[name] for name in _PROFILE_FACTS},
        _PROFILE_FACTS,
        "P4 steering-fit execution profile facts",
    )
    _require_literal(
        raw["artifact_fit"],
        _FIT_CONFIGURATION,
        "P4 steering-fit configuration",
    )
    if raw["source_hash_mode"] != _SOURCE_HASH_MODE:
        raise ValueError("P4 steering-fit source hash mode drift")
    source_sha256 = _require_mapping(raw["source_sha256"], "source_sha256")
    _require_exact_keys(
        source_sha256,
        set(_CRITICAL_SOURCE_PATHS),
        "P4 steering-fit critical source map",
    )
    for path in _CRITICAL_SOURCE_PATHS:
        _require_relative_posix_path(path, "critical source path")
        _require_sha256_value(
            source_sha256[path],
            f"source_sha256.{path}",
        )
    _require_literal(
        raw["instrumental_fit_execution"],
        _INSTRUMENTAL_FIT_EXECUTION,
        "P4 steering-fit instrumental execution contract",
    )
    _require_literal(
        raw["stopping_thresholds"],
        _STOPPING_THRESHOLDS,
        "P4 steering-fit stopping thresholds",
    )
    _require_literal(
        raw["output_contract"],
        _OUTPUT_CONTRACT,
        "P4 steering-fit output contract",
    )
    _require_literal(
        raw["evidence_firewall"],
        _EVIDENCE_FIREWALL,
        "P4 steering-fit evidence firewall",
    )
    if raw["claim_boundary"] != _CLAIM_BOUNDARY:
        raise ValueError("P4 steering-fit claim boundary drift")


def _build_report_payload(
    *,
    protocol: RelationshipP4SteeringFitProtocol,
    bundle_id: str,
    bundle_sha256: str,
    execution_attestation_id: str,
    owner_report: Mapping[str, Any],
    checks: Mapping[str, bool],
    prerequisite_passed: bool,
    verdict: str,
    sensor_off_quarantine: Mapping[str, Any] | None,
    numpy_version: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": P4_STEERING_FIT_REPORT_SCHEMA_VERSION,
        "protocol_id": protocol.protocol_id,
        "owner": {
            "wheel": _OWNER_WHEEL,
            "callable": _OWNER_CALLABLE,
        },
        "model": {
            "model_id": protocol.model_id,
            "verified_revision": protocol.verified_revision,
            "model_weights_sha256": protocol.model_weights_sha256,
            "execution_assets_sha256": protocol.execution_assets_sha256,
        },
        "execution_profile_id": protocol.profile_id,
        "execution_attestation_id": execution_attestation_id,
        "bundle_id": bundle_id,
        "bundle_sha256": bundle_sha256,
        "source_sha256": protocol.source_sha256_payload(),
        "source_hash_mode": protocol.source_hash_mode,
        "numpy_version": numpy_version,
        "fit_configuration": protocol.fit_configuration_payload(),
        "instrumental_fit_execution": dict(_INSTRUMENTAL_FIT_EXECUTION),
        "stopping_thresholds": protocol.stopping_thresholds_payload(),
        "owner_report": dict(owner_report),
        "checks": dict(checks),
        "prerequisite_passed": prerequisite_passed,
        "verdict": verdict,
        "failure_retuning_performed": False,
        "evidence_firewall": dict(_EVIDENCE_FIREWALL),
        "claim_boundary": protocol.claim_boundary,
    }
    if sensor_off_quarantine is not None:
        payload["sensor_off_quarantine"] = dict(sensor_off_quarantine)
    return payload


def _build_manifest_core(
    *,
    protocol: RelationshipP4SteeringFitProtocol,
    bundle_id: str,
    execution_attestation_id: str,
    prerequisite_passed: bool,
    verdict: str,
    sensor_off_quarantine: Mapping[str, Any] | None,
    payload_bytes: Mapping[str, bytes],
) -> dict[str, object]:
    if set(payload_bytes) != set(_PAYLOAD_FILES):
        raise ValueError("P4 steering-fit manifest payload file set drift")
    payload: dict[str, object] = {
        "schema_version": P4_STEERING_FIT_MANIFEST_SCHEMA_VERSION,
        "protocol_id": protocol.protocol_id,
        "bundle_id": bundle_id,
        "execution_attestation_id": execution_attestation_id,
        "source_sha256": protocol.source_sha256_payload(),
        "source_hash_mode": protocol.source_hash_mode,
        "prerequisite_passed": prerequisite_passed,
        "verdict": verdict,
        "formal_evidence_authorized": False,
        "production_active_authorized": False,
        "files": [
            {
                "path": name,
                "sha256": _sha256_bytes(payload_bytes[name]),
                "byte_count": len(payload_bytes[name]),
            }
            for name in _PAYLOAD_FILES
        ],
    }
    if sensor_off_quarantine is not None:
        payload["sensor_off_executor_quarantined"] = True
        payload["sensor_off_quarantine_sha256"] = _sha256_bytes(
            _canonical_bytes(sensor_off_quarantine, newline=False)
        )
    return payload


def _validate_relationship_p4_steering_fit_artifact(
    *,
    output_dir: pathlib.Path,
    protocol: RelationshipP4SteeringFitProtocol,
) -> RelationshipP4SteeringFitRunResult:
    output = pathlib.Path(output_dir)
    if not output.is_dir():
        raise FileNotFoundError(f"P4 steering-fit artifact root is missing: {output}")
    entries = tuple(output.iterdir())
    if any(entry.is_symlink() for entry in entries):
        raise ValueError("P4 steering-fit artifact files cannot be symlinks")
    if {entry.name for entry in entries} != set(_REQUIRED_FILES) or any(
        not entry.is_file() for entry in entries
    ):
        raise ValueError("P4 steering-fit artifact file set drift")

    manifest = _load_json_object(output / _MANIFEST_FILE)
    manifest_keys = {
        "schema_version",
        "protocol_id",
        "bundle_id",
        "execution_attestation_id",
        "source_sha256",
        "source_hash_mode",
        "prerequisite_passed",
        "verdict",
        "formal_evidence_authorized",
        "production_active_authorized",
        "files",
        "artifact_id",
    }
    manifest_has_quarantine = (
        "sensor_off_executor_quarantined" in manifest
        or "sensor_off_quarantine_sha256" in manifest
    )
    if manifest_has_quarantine:
        manifest_keys.update(
            {
                "sensor_off_executor_quarantined",
                "sensor_off_quarantine_sha256",
            }
        )
    _require_exact_keys(
        manifest,
        manifest_keys,
        "P4 steering-fit manifest",
    )
    if manifest["schema_version"] != P4_STEERING_FIT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("P4 steering-fit manifest schema drift")
    if manifest["protocol_id"] != protocol.protocol_id:
        raise ValueError("P4 steering-fit manifest protocol lineage drift")
    _require_literal(
        manifest["source_sha256"],
        protocol.source_sha256_payload(),
        "P4 steering-fit manifest source lineage",
    )
    if manifest["source_hash_mode"] != protocol.source_hash_mode:
        raise ValueError("P4 steering-fit manifest source hash mode drift")
    if manifest_has_quarantine:
        if manifest["sensor_off_executor_quarantined"] is not True:
            raise ValueError("P4 steering-fit manifest quarantine flag drift")
        _require_sha256_value(
            manifest["sensor_off_quarantine_sha256"],
            "manifest.sensor_off_quarantine_sha256",
        )
    artifact_id = _require_sha256_value(
        manifest["artifact_id"],
        "manifest.artifact_id",
    )
    manifest_core = dict(manifest)
    del manifest_core["artifact_id"]
    if artifact_id != _sha256_bytes(_canonical_bytes(manifest_core)):
        raise ValueError("P4 steering-fit manifest artifact_id drift")
    if (
        manifest["formal_evidence_authorized"] is not False
        or manifest["production_active_authorized"] is not False
    ):
        raise ValueError("P4 steering-fit manifest evidence firewall is open")
    file_records = _require_list(manifest["files"], "manifest.files")
    if len(file_records) != len(_PAYLOAD_FILES):
        raise ValueError("P4 steering-fit manifest file count drift")
    record_by_path: dict[str, Mapping[str, Any]] = {}
    for index, raw_record in enumerate(file_records):
        record = _require_mapping(raw_record, f"manifest.files[{index}]")
        _require_exact_keys(
            record,
            {"path", "sha256", "byte_count"},
            f"manifest.files[{index}]",
        )
        name = _require_text_value(record["path"], "manifest file path")
        if (
            name not in _PAYLOAD_FILES
            or pathlib.PurePosixPath(name).name != name
            or name in record_by_path
        ):
            raise ValueError("P4 steering-fit manifest file path drift")
        expected_sha = _require_sha256_value(
            record["sha256"],
            f"manifest file {name} sha256",
        )
        expected_size = _require_int_value(
            record["byte_count"],
            f"manifest file {name} byte_count",
        )
        payload = (output / name).read_bytes()
        if len(payload) != expected_size or _sha256_bytes(payload) != expected_sha:
            raise ValueError(f"P4 steering-fit payload hash drift: {name}")
        record_by_path[name] = record
    if tuple(record_by_path) != _PAYLOAD_FILES:
        raise ValueError("P4 steering-fit manifest file order drift")

    attestation = _load_json_object(output / _ATTESTATION_FILE)
    _validate_execution_attestation_payload(attestation, protocol=protocol)
    report = _load_json_object(output / _REPORT_FILE)
    bundle = _load_and_validate_bundle(
        output / _BUNDLE_FILE,
        protocol=protocol,
    )
    checks = _validate_report_payload(
        report,
        protocol=protocol,
        bundle=bundle,
        bundle_sha256=_sha256_file(output / _BUNDLE_FILE),
        execution_attestation_id=_require_text_value(
            attestation["attestation_id"],
            "execution attestation id",
        ),
    )
    report_has_quarantine = "sensor_off_quarantine" in report
    if manifest_has_quarantine is not report_has_quarantine:
        raise ValueError("P4 steering-fit quarantine manifest/report drift")
    if report_has_quarantine:
        quarantine = _require_mapping(
            report["sensor_off_quarantine"],
            "report.sensor_off_quarantine",
        )
        if manifest["sensor_off_quarantine_sha256"] != _sha256_bytes(
            _canonical_bytes(quarantine, newline=False)
        ):
            raise ValueError("P4 steering-fit quarantine evidence hash drift")
    prerequisite_passed = all(checks.values())
    verdict = _verdict(prerequisite_passed)
    if (
        manifest["bundle_id"] != bundle.bundle_id
        or manifest["execution_attestation_id"]
        != attestation["attestation_id"]
        or manifest["prerequisite_passed"] is not prerequisite_passed
        or manifest["verdict"] != verdict
    ):
        raise ValueError("P4 steering-fit manifest/report lineage drift")
    return RelationshipP4SteeringFitRunResult(
        artifact_id=artifact_id,
        protocol_id=protocol.protocol_id,
        bundle_id=bundle.bundle_id,
        execution_attestation_id=str(attestation["attestation_id"]),
        prerequisite_passed=prerequisite_passed,
        verdict=verdict,
        output_dir=output.resolve(),
    )


def _load_and_validate_bundle(
    path: pathlib.Path,
    *,
    protocol: RelationshipP4SteeringFitProtocol,
) -> Any:
    from volvence_zero.steering_contracts import SteeringArtifactBundle

    bundle = SteeringArtifactBundle.from_json(path.read_text(encoding="utf-8"))
    _validate_bundle_lineage(bundle, protocol=protocol)
    return bundle


def _prepare_owner_bundle_for_publication(
    bundle: Any,
    *,
    protocol: RelationshipP4SteeringFitProtocol,
) -> tuple[Any, dict[str, object] | None]:
    """Turn an owner candidate into a contract-valid published bundle.

    The owner candidate is the only place where a valid executor can expose
    non-unconditional sensor-off rows.  Such rows are stopping evidence, not
    an executable artifact: preserve them in the report, omit that optional
    executor from the published bundle, then round-trip the safe bundle
    through the public contract before any bytes are written.
    """

    from volvence_zero.steering_contracts import SteeringArtifactBundle

    if not isinstance(bundle, SteeringArtifactBundle):
        raise TypeError("P4 steering-fit owner returned the wrong bundle type")
    sensor_off = bundle.sensor_off_executor
    if sensor_off is not None:
        replace(sensor_off)
    _validate_bundle_lineage(bundle, protocol=protocol)
    quarantine: dict[str, object] | None = None
    publication_bundle = bundle
    if not _sensor_off_condition_code_rows_identical(bundle):
        quarantine = _build_sensor_off_quarantine_evidence(
            bundle=bundle,
            protocol=protocol,
        )
        publication_bundle = replace(bundle, sensor_off_executor=None)
    safe_bundle = SteeringArtifactBundle.from_json(publication_bundle.to_json())
    _validate_bundle_lineage(safe_bundle, protocol=protocol)
    return safe_bundle, quarantine


def _validate_bundle_lineage(
    bundle: Any,
    *,
    protocol: RelationshipP4SteeringFitProtocol,
) -> bool:
    prefix = f"{protocol.protocol_id[:12]}:{protocol.model_weights_sha256[:12]}"
    expected_ids = {
        "bundle": f"steering-dialogue-shadow:{prefix}",
        "reader": f"steering-reader:{prefix}",
        "executor": f"steering-executor:{prefix}",
        "sensor_off": f"steering-executor-sensor-off:{prefix}",
        "gate": f"steering-gate-shadow-collector:{prefix}",
    }
    sensor_off = bundle.sensor_off_executor
    if (
        bundle.bundle_id != expected_ids["bundle"]
        or bundle.reader.artifact_id != expected_ids["reader"]
        or bundle.executor.artifact_id != expected_ids["executor"]
        or bundle.gate.artifact_id != expected_ids["gate"]
    ):
        raise ValueError("P4 steering-fit bundle uses a noncanonical new lineage")
    if (
        sensor_off is not None
        and sensor_off.artifact_id != expected_ids["sensor_off"]
    ):
        raise ValueError("P4 steering-fit sensor-off artifact lineage drift")
    artifacts = (bundle.reader, bundle.executor)
    if sensor_off is not None:
        artifacts += (sensor_off,)
    for artifact in artifacts:
        if (
            artifact.model_id != protocol.model_id
            or artifact.model_weights_sha256 != protocol.model_weights_sha256
            or artifact.source_preregistration_sha256 != protocol.protocol_id
            or artifact.layer_index != protocol.injection_layer_index
            or artifact.residual_width != protocol.residual_width
        ):
            raise ValueError("P4 steering-fit bundle model/geometry lineage drift")
    if bundle.gate.source_preregistration_sha256 != protocol.protocol_id:
        raise ValueError("P4 steering-fit gate preregistration lineage drift")
    executors = (bundle.executor,)
    if sensor_off is not None:
        executors += (sensor_off,)
    for artifact in executors:
        if (
            artifact.rank != protocol.steering_rank
            or artifact.control_norm_cap_ratio
            != protocol.control_norm_cap_ratio
            or artifact.free_bias_present
            or not artifact.zero_code_strict_noop
        ):
            raise ValueError("P4 steering-fit executor safety invariant drift")
    if sensor_off is not None and (
        sensor_off.reader_artifact_id != bundle.reader.artifact_id
        or sensor_off.class_labels != bundle.reader.class_labels
    ):
        raise ValueError("P4 steering-fit sensor-off reader/class lineage drift")
    return True


def _sensor_off_condition_code_rows_identical(bundle: Any) -> bool:
    sensor_off = bundle.sensor_off_executor
    if sensor_off is None:
        return False
    rows = sensor_off.condition_codes
    return bool(rows) and all(row == rows[0] for row in rows[1:])


def _build_sensor_off_quarantine_evidence(
    *,
    bundle: Any,
    protocol: RelationshipP4SteeringFitProtocol,
) -> dict[str, object]:
    sensor_off = bundle.sensor_off_executor
    if sensor_off is None:
        reason = "sensor_off_executor_missing"
        artifact_id: str | None = None
        condition_codes: list[list[float]] = []
        condition_code_rank = 0
    else:
        reason = "condition_code_rows_not_identical"
        artifact_id = sensor_off.artifact_id
        condition_codes = [list(row) for row in sensor_off.condition_codes]
        condition_code_rank = protocol.steering_rank
    return {
        "schema_version": "relationship-p4-sensor-off-quarantine.v1",
        "quarantined": True,
        "reason": reason,
        "executor_artifact_id": artifact_id,
        "condition_code_row_count": len(condition_codes),
        "condition_code_rank": condition_code_rank,
        "condition_codes": condition_codes,
        "condition_codes_sha256": _sha256_bytes(
            _canonical_bytes(condition_codes, newline=False)
        ),
        "published_bundle_sensor_off_executor_present": False,
    }


def _validate_sensor_off_quarantine_evidence(
    evidence: Mapping[str, Any],
    *,
    bundle: Any,
    protocol: RelationshipP4SteeringFitProtocol,
) -> None:
    _require_exact_keys(
        evidence,
        {
            "schema_version",
            "quarantined",
            "reason",
            "executor_artifact_id",
            "condition_code_row_count",
            "condition_code_rank",
            "condition_codes",
            "condition_codes_sha256",
            "published_bundle_sensor_off_executor_present",
        },
        "P4 steering-fit sensor-off quarantine evidence",
    )
    if (
        evidence["schema_version"]
        != "relationship-p4-sensor-off-quarantine.v1"
    ):
        raise ValueError("P4 steering-fit quarantine schema drift")
    if (
        evidence["quarantined"] is not True
        or evidence["published_bundle_sensor_off_executor_present"] is not False
        or bundle.sensor_off_executor is not None
    ):
        raise ValueError("P4 steering-fit quarantine did not isolate sensor-off")
    reason = _require_text_value(evidence["reason"], "quarantine.reason")
    raw_codes = _require_list(
        evidence["condition_codes"],
        "quarantine.condition_codes",
    )
    expected_digest = _require_sha256_value(
        evidence["condition_codes_sha256"],
        "quarantine.condition_codes_sha256",
    )
    if expected_digest != _sha256_bytes(
        _canonical_bytes(raw_codes, newline=False)
    ):
        raise ValueError("P4 steering-fit quarantine condition-code hash drift")
    row_count = _require_int_value(
        evidence["condition_code_row_count"],
        "quarantine.condition_code_row_count",
    )
    rank = _require_int_value(
        evidence["condition_code_rank"],
        "quarantine.condition_code_rank",
    )
    if row_count != len(raw_codes):
        raise ValueError("P4 steering-fit quarantine row-count drift")
    if reason == "sensor_off_executor_missing":
        if (
            evidence["executor_artifact_id"] is not None
            or row_count != 0
            or rank != 0
            or raw_codes
        ):
            raise ValueError("P4 steering-fit missing sensor-off evidence drift")
        return
    if reason != "condition_code_rows_not_identical":
        raise ValueError("P4 steering-fit quarantine reason drift")
    prefix = f"{protocol.protocol_id[:12]}:{protocol.model_weights_sha256[:12]}"
    if evidence["executor_artifact_id"] != (
        f"steering-executor-sensor-off:{prefix}"
    ):
        raise ValueError("P4 steering-fit quarantined executor lineage drift")
    if (
        row_count != len(bundle.reader.class_labels)
        or rank != protocol.steering_rank
        or not raw_codes
    ):
        raise ValueError("P4 steering-fit quarantined condition-code geometry drift")
    rows: list[tuple[float, ...]] = []
    for row_index, raw_row in enumerate(raw_codes):
        row = _require_list(
            raw_row,
            f"quarantine.condition_codes[{row_index}]",
        )
        if len(row) != rank:
            raise ValueError(
                "P4 steering-fit quarantined condition-code rank drift"
            )
        rows.append(
            tuple(
                _require_float_value(
                    value,
                    f"quarantine.condition_codes[{row_index}]",
                )
                for value in row
            )
        )
    if all(row == rows[0] for row in rows[1:]):
        raise ValueError(
            "P4 steering-fit quarantine evidence does not establish the "
            "sensor-off stopping failure"
        )


def _validate_bundle_and_owner_report(
    *,
    bundle: Any,
    owner_report: Mapping[str, Any],
    protocol: RelationshipP4SteeringFitProtocol,
    sensor_off_quarantine: Mapping[str, Any] | None = None,
) -> dict[str, bool]:
    _validate_bundle_lineage(bundle, protocol=protocol)
    _validate_owner_report_shape(owner_report, protocol=protocol)
    if sensor_off_quarantine is None:
        if bundle.sensor_off_executor is None:
            raise ValueError(
                "P4 steering-fit published bundle lacks sensor-off executor "
                "without independently checkable quarantine evidence"
            )
        sensor_off_rows_identical = (
            _sensor_off_condition_code_rows_identical(bundle)
        )
    else:
        _validate_sensor_off_quarantine_evidence(
            sensor_off_quarantine,
            bundle=bundle,
            protocol=protocol,
        )
        sensor_off_rows_identical = False
    checks = {
        "reader_heldout_accuracy": (
            float(owner_report["reader_heldout_accuracy"])
            >= float(_STOPPING_THRESHOLDS["reader_heldout_accuracy_min"])
        ),
        "gain_vs_noop_strictly_positive": (
            float(owner_report["heldout_gain_vs_noop_nll"]) > 0.0
        ),
        "conditional_advantage_strictly_positive": (
            float(owner_report["heldout_conditional_advantage_nll"]) > 0.0
        ),
        "substrate_frozen": (
            owner_report["substrate_trainable_parameter_count"] == 0
        ),
        "reader_executor_frozen_for_dialogue": (
            owner_report["reader_executor_frozen_for_dialogue"] is True
        ),
        "no_free_bias": owner_report["free_bias_present"] is False,
        "strict_zero_code_noop": (
            owner_report["zero_code_strict_noop"] is True
        ),
        "sensor_off_condition_code_rows_identical": (
            sensor_off_rows_identical
        ),
    }
    if any(type(value) is not bool for value in checks.values()):
        raise TypeError("P4 steering-fit stopping checks must be exact bools")
    return checks


def _validate_owner_report_shape(
    owner_report: Mapping[str, Any],
    *,
    protocol: RelationshipP4SteeringFitProtocol,
) -> None:
    _require_exact_keys(
        owner_report,
        _OWNER_REPORT_KEYS,
        "P4 steering-fit owner report",
    )
    train_row_count = _require_int_value(
        owner_report["train_row_count"],
        "owner_report.train_row_count",
    )
    heldout_row_count = _require_int_value(
        owner_report["heldout_row_count"],
        "owner_report.heldout_row_count",
    )
    if (
        train_row_count != protocol.expected_train_row_count
        or heldout_row_count != protocol.expected_heldout_row_count
    ):
        raise ValueError("P4 steering-fit canonical corpus row-count drift")
    for name in (
        "reader_heldout_accuracy",
        "heldout_noop_nll",
        "heldout_online_steer_nll",
        "heldout_sensor_off_nll",
        "heldout_gain_vs_noop_nll",
        "heldout_conditional_advantage_nll",
        "reader_ridge_lambda",
        "executor_learning_rate",
    ):
        _require_number_value(owner_report[name], f"owner_report.{name}")
    derived_metrics = {
        "heldout_gain_vs_noop_nll": (
            float(owner_report["heldout_noop_nll"])
            - float(owner_report["heldout_online_steer_nll"])
        ),
        "heldout_conditional_advantage_nll": (
            float(owner_report["heldout_sensor_off_nll"])
            - float(owner_report["heldout_online_steer_nll"])
        ),
    }
    for name, expected in derived_metrics.items():
        if not math.isclose(
            float(owner_report[name]),
            expected,
            rel_tol=_DERIVED_NLL_REL_TOLERANCE,
            abs_tol=_DERIVED_NLL_ABS_TOLERANCE,
        ):
            raise ValueError(
                f"P4 steering-fit derived NLL metric drift: {name}"
            )
    if not 0.0 <= float(owner_report["reader_heldout_accuracy"]) <= 1.0:
        raise ValueError("P4 steering-fit reader accuracy is outside [0, 1]")
    if (
        float(owner_report["reader_ridge_lambda"])
        != protocol.reader_ridge_lambda
        or _require_int_value(
            owner_report["executor_updates"],
            "owner_report.executor_updates",
        )
        != protocol.conditional_executor_updates
        or float(owner_report["executor_learning_rate"])
        != protocol.executor_learning_rate
        or _require_int_value(
            owner_report["steering_rank"],
            "owner_report.steering_rank",
        )
        != protocol.steering_rank
        or _require_int_value(
            owner_report["seed"],
            "owner_report.seed",
        )
        != protocol.fit_seed
        or _require_float_value(
            owner_report["control_norm_cap_ratio"],
            "owner_report.control_norm_cap_ratio",
        )
        != protocol.control_norm_cap_ratio
    ):
        raise ValueError("P4 steering-fit owner hyperparameter report drift")
    for name in (
        "free_bias_present",
        "zero_code_strict_noop",
        "reader_executor_frozen_for_dialogue",
    ):
        _require_bool_value(owner_report[name], f"owner_report.{name}")
    if (
        _require_int_value(
            owner_report["substrate_trainable_parameter_count"],
            "owner_report.substrate_trainable_parameter_count",
        )
        < 0
    ):
        raise ValueError("P4 steering-fit trainable count cannot be negative")
    _require_text_value(owner_report["description"], "owner_report.description")


def _validate_report_payload(
    report: Mapping[str, Any],
    *,
    protocol: RelationshipP4SteeringFitProtocol,
    bundle: Any,
    bundle_sha256: str,
    execution_attestation_id: str,
) -> dict[str, bool]:
    report_keys = {
        "schema_version",
        "protocol_id",
        "owner",
        "model",
        "execution_profile_id",
        "execution_attestation_id",
        "bundle_id",
        "bundle_sha256",
        "source_sha256",
        "source_hash_mode",
        "numpy_version",
        "fit_configuration",
        "instrumental_fit_execution",
        "stopping_thresholds",
        "owner_report",
        "checks",
        "prerequisite_passed",
        "verdict",
        "failure_retuning_performed",
        "evidence_firewall",
        "claim_boundary",
    }
    report_has_quarantine = "sensor_off_quarantine" in report
    if report_has_quarantine:
        report_keys.add("sensor_off_quarantine")
    _require_exact_keys(
        report,
        report_keys,
        "P4 steering-fit report",
    )
    if report["schema_version"] != P4_STEERING_FIT_REPORT_SCHEMA_VERSION:
        raise ValueError("P4 steering-fit report schema drift")
    _require_literal(
        report["owner"],
        {"wheel": _OWNER_WHEEL, "callable": _OWNER_CALLABLE},
        "P4 steering-fit report owner",
    )
    _require_literal(
        report["model"],
        {
            "model_id": protocol.model_id,
            "verified_revision": protocol.verified_revision,
            "model_weights_sha256": protocol.model_weights_sha256,
            "execution_assets_sha256": protocol.execution_assets_sha256,
        },
        "P4 steering-fit report model",
    )
    if (
        report["protocol_id"] != protocol.protocol_id
        or report["execution_profile_id"] != protocol.profile_id
        or report["execution_attestation_id"] != execution_attestation_id
        or report["bundle_id"] != bundle.bundle_id
        or report["bundle_sha256"] != bundle_sha256
    ):
        raise ValueError("P4 steering-fit report lineage drift")
    _require_literal(
        report["source_sha256"],
        protocol.source_sha256_payload(),
        "P4 steering-fit report source lineage",
    )
    if report["source_hash_mode"] != protocol.source_hash_mode:
        raise ValueError("P4 steering-fit report source hash mode drift")
    _require_text_value(report["numpy_version"], "report.numpy_version")
    _require_literal(
        report["fit_configuration"],
        protocol.fit_configuration_payload(),
        "P4 steering-fit report configuration",
    )
    _require_literal(
        report["instrumental_fit_execution"],
        _INSTRUMENTAL_FIT_EXECUTION,
        "P4 steering-fit instrumental execution attestation",
    )
    _require_literal(
        report["stopping_thresholds"],
        protocol.stopping_thresholds_payload(),
        "P4 steering-fit report thresholds",
    )
    _require_literal(
        report["evidence_firewall"],
        _EVIDENCE_FIREWALL,
        "P4 steering-fit report firewall",
    )
    if (
        report["failure_retuning_performed"] is not False
        or report["claim_boundary"] != protocol.claim_boundary
    ):
        raise ValueError("P4 steering-fit report claim/retuning boundary drift")
    owner_report = _require_mapping(report["owner_report"], "report.owner_report")
    sensor_off_quarantine = None
    if report_has_quarantine:
        sensor_off_quarantine = _require_mapping(
            report["sensor_off_quarantine"],
            "report.sensor_off_quarantine",
        )
    expected_checks = _validate_bundle_and_owner_report(
        bundle=bundle,
        owner_report=owner_report,
        protocol=protocol,
        sensor_off_quarantine=sensor_off_quarantine,
    )
    _require_literal(
        report["checks"],
        expected_checks,
        "P4 steering-fit report derived checks",
    )
    prerequisite_passed = all(expected_checks.values())
    if (
        report["prerequisite_passed"] is not prerequisite_passed
        or report["verdict"] != _verdict(prerequisite_passed)
    ):
        raise ValueError("P4 steering-fit report verdict drift")
    return expected_checks


def _validate_execution_attestation_payload(
    payload: Mapping[str, Any],
    *,
    protocol: RelationshipP4SteeringFitProtocol,
) -> None:
    _require_exact_keys(
        payload,
        _ATTESTATION_KEYS,
        "P4 steering-fit execution attestation",
    )
    canonical = dict(payload)
    attestation_id = _require_sha256_value(
        canonical.pop("attestation_id"),
        "execution attestation id",
    )
    if attestation_id != _sha256_bytes(_canonical_bytes(canonical, newline=False)):
        raise ValueError("P4 steering-fit execution attestation id drift")
    expected_direct = {
        "schema_version": "transformers-execution-attestation.v1",
        "profile_id": protocol.profile_id,
        "preset_name": _PROFILE_FACTS["preset_name"],
        "model_id": protocol.model_id,
        "model_revision": protocol.verified_revision,
        "model_weights_sha256": protocol.model_weights_sha256,
        "execution_assets_sha256": protocol.execution_assets_sha256,
        "runtime_origin": "hf-local",
        "platform_system": "Windows",
        "attention_implementation": _PROFILE_FACTS[
            "attention_implementation"
        ],
        "sdpa_backend": _PROFILE_FACTS["sdpa_backend"],
        "sdpa_backend_policy": _PROFILE_FACTS["sdpa_backend_policy"],
        "sdpa_backend_exclusive": True,
        "generation_use_cache": True,
        "require_generation_chat_template": True,
        "generation_capture_strategy": _PROFILE_FACTS[
            "generation_capture_strategy"
        ],
        "capture_failure_mode": "raise",
        "context_window_tokens": 32768,
        "local_files_only": True,
        "fallback_mode": "deny",
        "fail_on_truncation": True,
        "model_dtype": "bfloat16",
        "hidden_size": protocol.residual_width,
        "model_max_position_embeddings": 32768,
        "hook_layer_indices": [protocol.injection_layer_index],
    }
    for name, expected in expected_direct.items():
        _require_literal(
            payload[name],
            expected,
            f"execution attestation.{name}",
        )
    for name in (
        "platform_release",
        "device_name",
        "python_version",
        "torch_version",
        "transformers_version",
        "cuda_version",
    ):
        _require_text_value(payload[name], f"execution attestation.{name}")
    device = _require_text_value(payload["device"], "execution attestation.device")
    if device != "cuda" and not (
        device.startswith("cuda:") and device.removeprefix("cuda:").isdigit()
    ):
        raise ValueError("P4 steering-fit attestation device is not CUDA")
    if _require_int_value(
        payload["cudnn_version"],
        "execution attestation.cudnn_version",
    ) <= 0:
        raise ValueError("P4 steering-fit attestation cuDNN version is invalid")
    capability = _require_list(
        payload["device_compute_capability"],
        "execution attestation.device_compute_capability",
    )
    if len(capability) != 2 or any(
        _require_int_value(value, "execution attestation compute capability") < 0
        for value in capability
    ):
        raise ValueError("P4 steering-fit compute capability is invalid")


def _verdict(prerequisite_passed: bool) -> str:
    return (
        "fresh_qwen25_15b_steering_artifact_fit_passed_development_only"
        if prerequisite_passed
        else "fresh_qwen25_15b_steering_artifact_fit_failed_stop_no_retuning"
    )


def _verify_critical_sources(
    protocol: RelationshipP4SteeringFitProtocol,
) -> None:
    repository_root = _REPOSITORY_ROOT.resolve()
    for relative, expected in protocol.source_sha256:
        _require_relative_posix_path(relative, "critical source path")
        candidate = repository_root.joinpath(
            *pathlib.PurePosixPath(relative).parts
        )
        if candidate.is_symlink():
            raise ValueError(f"critical source cannot be a symlink: {relative}")
        source = candidate.resolve()
        try:
            source.relative_to(repository_root)
        except ValueError as exc:
            raise ValueError(
                f"critical source escapes repository root: {relative}"
            ) from exc
        if not source.is_file():
            raise FileNotFoundError(
                f"critical source is missing or is a symlink: {relative}"
            )
        actual = _source_text_sha256(source)
        if actual != expected:
            raise ValueError(
                f"critical source SHA-256 drift before P4 steering fit: {relative}"
            )


def _numpy_version() -> str:
    from importlib.metadata import version

    value = version("numpy")
    return _require_text_value(value, "numpy version")


def _canonical_bytes(value: object, *, newline: bool = True) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if newline:
        text += "\n"
    return text.encode("utf-8")


def _write_create_bytes(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _load_json_object(path: pathlib.Path) -> dict[str, Any]:
    try:
        text = pathlib.Path(path).read_bytes().decode("utf-8")
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except UnicodeDecodeError as exc:
        raise ValueError(f"JSON artifact is not UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON artifact is invalid: {path}") from exc
    if type(payload) is not dict:
        raise ValueError(f"JSON artifact root must be an object: {path}")
    return payload


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_literal(actual: object, expected: object, label: str) -> None:
    if type(actual) is not type(expected):
        raise TypeError(f"{label} type drift")
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        _require_exact_keys(actual, set(expected), label)
        for key, value in expected.items():
            _require_literal(actual[key], value, f"{label}.{key}")
        return
    if isinstance(expected, list):
        assert isinstance(actual, list)
        if len(actual) != len(expected):
            raise ValueError(f"{label} length drift")
        for index, value in enumerate(expected):
            _require_literal(actual[index], value, f"{label}[{index}]")
        return
    if actual != expected:
        raise ValueError(f"{label} value drift")


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be an exact object")
    return value


def _require_list(value: object, label: str) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{label} must be an exact list")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys drifted")


def _require_text_value(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def _require_relative_posix_path(value: object, label: str) -> str:
    text = _require_text_value(value, label)
    pure = pathlib.PurePosixPath(text)
    if (
        pure.is_absolute()
        or "\\" in text
        or str(pure) != text
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return text


def _require_int_value(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact int")
    return value


def _require_float_value(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{label} must be a finite exact float")
    return value


def _require_number_value(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise TypeError(f"{label} must be a finite number")
    return float(value)


def _require_bool_value(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be an exact bool")
    return value


def _require_sha256(value: str, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _require_sha256_value(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be text")
    _require_sha256(value, label)
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_text_sha256(path: pathlib.Path) -> str:
    payload = pathlib.Path(path).read_bytes()
    if payload.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"critical source must not carry a UTF-8 BOM: {path}")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"critical source must be strict UTF-8: {path}") from exc
    canonical_lf = text.replace("\r\n", "\n").replace("\r", "\n")
    return _sha256_bytes(canonical_lf.encode("utf-8"))


__all__ = (
    "P4_STEERING_FIT_MANIFEST_SCHEMA_VERSION",
    "P4_STEERING_FIT_PROTOCOL_SCHEMA_VERSION",
    "P4_STEERING_FIT_REPORT_SCHEMA_VERSION",
    "RelationshipP4SteeringFitProtocol",
    "RelationshipP4SteeringFitRunResult",
    "load_relationship_p4_steering_fit_protocol",
    "relationship_p4_steering_fit_protocol_path",
    "run_relationship_p4_steering_artifact_fit",
    "validate_relationship_p4_steering_artifact_fit",
)
