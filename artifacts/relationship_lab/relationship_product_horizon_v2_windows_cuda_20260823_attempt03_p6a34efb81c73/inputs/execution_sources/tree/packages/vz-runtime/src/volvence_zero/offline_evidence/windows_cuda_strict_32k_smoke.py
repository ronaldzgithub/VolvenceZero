"""Content-addressed Windows/CUDA strict 32767+1 engineering smoke.

The substrate remains the sole owner of model execution, chat-template token
budgets, hooks, and residual capture.  This module is a thin offline evidence
orchestrator: it freezes one exact public ``generate`` call, publishes only a
bounded summary, and provides a validator that never imports torch or CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import pathlib
from typing import Any, Callable, Mapping


STRICT_32K_PROTOCOL_SCHEMA_VERSION = "windows-cuda-strict-32k-smoke.v1"
STRICT_32K_REPORT_SCHEMA_VERSION = "windows-cuda-strict-32k-smoke-report.v1"
STRICT_32K_MANIFEST_SCHEMA_VERSION = "windows-cuda-strict-32k-smoke-manifest.v1"
STRICT_32K_LAUNCH_SCHEMA_VERSION = "windows-cuda-strict-32k-smoke-launch.v1"
STRICT_32K_COMPLETION_SCHEMA_VERSION = "windows-cuda-strict-32k-smoke-completion.v1"

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
_REPOSITORY_ROOT = _PACKAGE_ROOT.parents[4]
_DEFAULT_PROTOCOL_PATH = _PACKAGE_ROOT / "protocols" / "windows_cuda_strict_32k_smoke_v1.json"
_LAUNCH_FILE = "launch_receipt.json"
_ATTESTATION_FILE = "execution_attestation.json"
_REPORT_FILE = "strict_32k_smoke_report.json"
_MANIFEST_FILE = "manifest.json"
_COMPLETION_FILE = "completion_receipt.json"
_PAYLOAD_FILES = (_LAUNCH_FILE, _ATTESTATION_FILE, _REPORT_FILE)
_REQUIRED_FILES = (*_PAYLOAD_FILES, _MANIFEST_FILE, _COMPLETION_FILE)
_SOURCE_HASH_MODE = "utf8_lf_canonical_v1"

_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
_MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
_MODEL_WEIGHTS_SHA256 = "fb8c44c48b8359fdd306cdc5f473d7c04d88955013f0dd8549f266e248194da4"
_EXECUTION_ASSETS_SHA256 = "bbb5446f8d802b437c2fc7e2cefcdabb996bbd4bc657fe155ea015d30a841bb0"
_PROFILE_ID = "3be84d866afbda07cf80dee277d89cdc0e366ce545bf7e97f015cf8afcbfe21a"
_EXECUTION_ATTESTATION_ID = "9a33a698b95d923d6a4e82b64471213d529b0cbbf6a30ca24644860211e6dde1"
_LAYER_INDEX = 20
_ACTIVATION_WIDTH = 1536
_CONTEXT_WINDOW_TOKENS = 32768
_INPUT_TOKEN_COUNT = 32767
_MAX_NEW_TOKENS = 1
_PROMPT_UNIT = "<|im_start|>"
_PROMPT_UNIT_REPEAT_COUNT = 32754
_RENDERED_PROMPT_SHA256 = "2bae362c6e83f091aa96b1902a573c99de9adc53bf996661a2f0a750d25f38b0"
_RENDERED_PROMPT_BYTE_COUNT = 393128
_CAPTURE_AUDIT_SCHEMA_VERSION = "strict-capture-audit-summary.v1"

_CRITICAL_SOURCE_PATHS = (
    "packages/vz-substrate/src/volvence_zero/substrate/__init__.py",
    "packages/vz-substrate/src/volvence_zero/substrate/adapter.py",
    "packages/vz-substrate/src/volvence_zero/substrate/common_adapter_bundle.py",
    "packages/vz-substrate/src/volvence_zero/substrate/residual_backend.py",
    "packages/vz-substrate/src/volvence_zero/substrate/residual_contracts.py",
    "packages/vz-substrate/src/volvence_zero/substrate/residual_helpers.py",
    "packages/vz-substrate/src/volvence_zero/substrate/residual_interfaces.py",
    "packages/vz-substrate/src/volvence_zero/substrate/strict_capture_audit.py",
    "packages/vz-runtime/src/volvence_zero/offline_evidence/windows_cuda_strict_32k_smoke.py",
    "scripts/run_windows_cuda_strict_32k_smoke.py",
)

_OWNER_FACTS = {
    "execution_owner_wheel": "vz-substrate",
    "execution_factory": ("volvence_zero.substrate.build_transformers_runtime_with_fallback"),
    "capture_audit_owner": ("volvence_zero.substrate.strict_capture_audit.audit_strict_capture"),
    "orchestrator_wheel": "vz-runtime",
    "orchestrator": ("volvence_zero.offline_evidence.windows_cuda_strict_32k_smoke"),
    "mode": "offline_engineering_diagnostic",
}
_MODEL_FACTS = {
    "model_id": _MODEL_ID,
    "verified_revision": _MODEL_REVISION,
    "model_weights_sha256": _MODEL_WEIGHTS_SHA256,
    "execution_assets_sha256": _EXECUTION_ASSETS_SHA256,
    "model_source": "logical_model_id_resolved_from_verified_hf_cache",
    "local_snapshot_path_recorded": False,
}
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
    "context_window_tokens": _CONTEXT_WINDOW_TOKENS,
    "local_files_only": True,
    "fallback_mode": "deny",
    "fail_on_truncation": True,
    "model_dtype": "bfloat16",
    "require_verified_model_revision": True,
    "require_model_weights_sha256": True,
    "require_execution_assets_sha256": True,
    "require_generation_chat_template": True,
}
_PROTOCOL_PROFILE_FACTS = {
    "profile_id": _PROFILE_ID,
    "expected_execution_attestation_id": _EXECUTION_ATTESTATION_ID,
    **_PROFILE_FACTS,
}
_DIAGNOSTIC_FACTS = {
    "layer_index": _LAYER_INDEX,
    "activation_width": _ACTIVATION_WIDTH,
    "attempt_budget": 1,
    "retry_budget": 0,
    "attempt_budget_scope": "per_frozen_output_root",
    "retry_enforcement_owner": "outer_host_campaign",
    "outer_attempt_lease_required": True,
    "prompt_recipe": {
        "template": "qwen25_chat_template_v1",
        "messages": [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content_recipe": "repeat_unit_without_separator",
                "unit": _PROMPT_UNIT,
                "repeat_count": _PROMPT_UNIT_REPEAT_COUNT,
            },
        ],
        "search_or_calibration_permitted": False,
        "expected_rendered_prompt_sha256": _RENDERED_PROMPT_SHA256,
        "expected_rendered_prompt_byte_count": _RENDERED_PROMPT_BYTE_COUNT,
    },
    "generation_call": {
        "prompt": "",
        "system_context": "",
        "max_new_tokens": _MAX_NEW_TOKENS,
        "temperature": 0.0,
        "capture_residuals": True,
        "call_count": 1,
    },
    "expected_context_budget": {
        "schema_version": "generation-context-budget-attestation.v1",
        "input_mode": "chat-template",
        "input_token_count": _INPUT_TOKEN_COUNT,
        "prefix_slot_count": 0,
        "effective_max_new_tokens": _MAX_NEW_TOKENS,
        "combined_token_count": _CONTEXT_WINDOW_TOKENS,
        "context_window_tokens": _CONTEXT_WINDOW_TOKENS,
        "remaining_token_count": 0,
    },
    "expected_capture": {
        "audit_summary_schema_version": _CAPTURE_AUDIT_SCHEMA_VERSION,
        "residual_sequence_length": _INPUT_TOKEN_COUNT,
        "layer_indices": [_LAYER_INDEX],
        "activation_width": _ACTIVATION_WIDTH,
        "latest_matches_sequence_exact": True,
        "hook_layer_coverage": 1.0,
        "hook_fire_rate": 1.0,
        "token_step_coverage": 1.0,
        "residual_sequence_present": 1.0,
        "fallback_active": 0.0,
    },
    "failure_action": "failed_diagnostic_stop_no_retry",
}
_OUTPUT_CONTRACT = {
    "create_only": True,
    "immutable": True,
    "content_addressed": True,
    "atomic_directory_publication": False,
    "launch_receipt_fsync_before_runtime_construction": True,
    "incomplete_attempt_root_never_deleted_by_runner": True,
    "completion_not_before_launch": True,
    "attempt_budget_scope": "per_frozen_output_root",
    "retry_enforcement_owner": "outer_host_campaign",
    "outer_attempt_lease_required": True,
    "required_files": list(_REQUIRED_FILES),
    "validate_existing_loads_torch_or_cuda": False,
    "full_residual_sequence_persisted": False,
    "missing_complete_root_is_pass": False,
}
_EVIDENCE_FIREWALL = {
    "external_append_only_anchor_present": False,
    "standalone_artifact_proves_physical_execution": False,
    "transitive_local_source_closure_pinned": False,
    "host_stability_proven": False,
    "long_context_information_utilization_proven": False,
    "appendable_proven": False,
    "readable_proven": False,
    "learnable_proven": False,
    "steerable_proven": False,
    "four_capability_claim_authorized": False,
    "independent_subject_evidence": False,
    "long_companion_multi_session_evidence": False,
    "production_active_authorized": False,
    "formal_evidence_authorized": False,
}
_CLAIM_BOUNDARY = (
    "When bound to an independently preregistered outer host-campaign "
    "attempt lease and append-only launch/completion receipts, this create-only "
    "artifact may establish only that the reviewed entrypoint files launched a "
    "Windows/CUDA substrate which completed one 32767 chat-template input plus "
    "one cached generated token with first-full-prompt residual capture. The "
    "current ten-file source set does not pin the complete transitive local "
    "import closure, so even an outer-bound observation is not authorized as "
    "exact-source physical evidence until a later protocol closes that gap. "
    "Standing alone it establishes only local file, hash, and lineage "
    "consistency, not physical execution. "
    "It does not establish host stability, long-context information use, "
    "independent-subject or long-companion evidence, production ACTIVE, "
    "Appendable, Readable, Learnable, Steerable, or the four-capability "
    "system claim. The local one-attempt/no-retry rule is scoped to one frozen "
    "output root; the outer host campaign owns cross-root lease consumption and "
    "prohibits replacement attempts or PASS selection. A completed failed "
    "observation is preserved. A runtime exception leaves an incomplete root "
    "that the runner never deletes. An OS hard crash may retain the already "
    "fsynced launch root, but the independent outer receipt remains authoritative; "
    "neither case can be relabelled as a completed diagnostic."
)

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
_CAPTURE_CHECK_KEYS = (
    "capture_present",
    "residual_sequence_length_exact",
    "residual_step_continuity_exact",
    "capture_layer_exact",
    "capture_width_exact",
    "latest_capture_matches_sequence_exact",
    "capture_values_all_finite",
    "top_logits_finite_nonempty",
    "hook_layer_coverage_exact",
    "hook_fire_rate_exact",
    "token_step_coverage_exact",
    "residual_sequence_present_exact",
    "fallback_inactive",
)
_CHECK_KEYS = (
    "execution_attestation_exact",
    "generation_execution_lineage_exact",
    "rendered_prompt_hash_exact",
    "input_token_count_exact",
    "context_budget_exact",
    "generated_token_count_exact",
    *_CAPTURE_CHECK_KEYS,
    "no_conditioning_or_steering_applied",
)


@dataclass(frozen=True)
class WindowsCudaStrict32KSmokeProtocol:
    """Frozen lineage for one exact public strict-generation call."""

    protocol_id: str
    protocol_raw_sha256: str
    profile_id: str
    expected_execution_attestation_id: str
    source_sha256: tuple[tuple[str, str], ...]
    source_hash_mode: str = _SOURCE_HASH_MODE

    def __post_init__(self) -> None:
        for value, label in (
            (self.protocol_id, "protocol_id"),
            (self.protocol_raw_sha256, "protocol_raw_sha256"),
            (self.profile_id, "profile_id"),
            (
                self.expected_execution_attestation_id,
                "expected_execution_attestation_id",
            ),
        ):
            _require_sha256(value, label)
        if self.profile_id != _PROFILE_ID:
            raise ValueError("strict 32K profile ID drift")
        if self.expected_execution_attestation_id != _EXECUTION_ATTESTATION_ID:
            raise ValueError("strict 32K execution attestation ID drift")
        if self.source_hash_mode != _SOURCE_HASH_MODE:
            raise ValueError("strict 32K source hash mode drift")
        if tuple(path for path, _ in self.source_sha256) != _CRITICAL_SOURCE_PATHS:
            raise ValueError("strict 32K critical source path set drift")
        for path, digest in self.source_sha256:
            _require_relative_posix_path(path, "critical source path")
            _require_sha256(digest, f"critical source {path}")

    def source_sha256_payload(self) -> dict[str, str]:
        return dict(self.source_sha256)


@dataclass(frozen=True)
class WindowsCudaStrict32KSmokeResult:
    artifact_id: str
    attempt_id: str
    outer_attempt_lease_id: str
    protocol_id: str
    execution_attestation_id: str
    passed: bool
    verdict: str
    output_dir: pathlib.Path


def strict_32k_smoke_protocol_path() -> pathlib.Path:
    return _DEFAULT_PROTOCOL_PATH


def load_windows_cuda_strict_32k_smoke_protocol(
    path: pathlib.Path | None = None,
) -> WindowsCudaStrict32KSmokeProtocol:
    protocol_path = pathlib.Path(path or _DEFAULT_PROTOCOL_PATH)
    raw_bytes = protocol_path.read_bytes()
    raw = _load_json_object_bytes(raw_bytes, label=str(protocol_path))
    _validate_protocol_payload(raw)
    profile = _require_mapping(raw["execution_profile"], "execution_profile")
    source_sha256 = _require_mapping(raw["source_sha256"], "source_sha256")
    return WindowsCudaStrict32KSmokeProtocol(
        protocol_id=_sha256_bytes(_canonical_bytes(raw)),
        protocol_raw_sha256=_sha256_bytes(raw_bytes),
        profile_id=_require_sha256_value(profile["profile_id"], "execution_profile.profile_id"),
        expected_execution_attestation_id=_require_sha256_value(
            profile["expected_execution_attestation_id"],
            "execution_profile.expected_execution_attestation_id",
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
    )


def _build_launch_receipt(
    protocol: WindowsCudaStrict32KSmokeProtocol,
    *,
    outer_attempt_lease_id: str,
) -> dict[str, object]:
    _require_sha256(outer_attempt_lease_id, "outer_attempt_lease_id")
    core = {
        "schema_version": STRICT_32K_LAUNCH_SCHEMA_VERSION,
        "protocol_id": protocol.protocol_id,
        "protocol_raw_sha256": protocol.protocol_raw_sha256,
        "source_hash_mode": protocol.source_hash_mode,
        "source_sha256": protocol.source_sha256_payload(),
        "attempt_budget": 1,
        "retry_budget": 0,
        "attempt_budget_scope": "per_frozen_output_root",
        "retry_enforcement_owner": "outer_host_campaign",
        "outer_attempt_lease_id": outer_attempt_lease_id,
        "process_id": os.getpid(),
        "started_at_utc": _utc_now_text(),
    }
    return {
        **core,
        "attempt_id": _sha256_bytes(_canonical_bytes(core)),
    }


def _build_completion_receipt(
    *,
    attempt_id: str,
    outer_attempt_lease_id: str,
    artifact_id: str,
    protocol_id: str,
    execution_attestation_id: str,
    passed: bool,
    verdict: str,
) -> dict[str, object]:
    core = {
        "schema_version": STRICT_32K_COMPLETION_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "outer_attempt_lease_id": outer_attempt_lease_id,
        "artifact_id": artifact_id,
        "protocol_id": protocol_id,
        "execution_attestation_id": execution_attestation_id,
        "passed": passed,
        "verdict": verdict,
        "completed_at_utc": _utc_now_text(),
    }
    return {
        **core,
        "completion_id": _sha256_bytes(_canonical_bytes(core)),
    }


def run_windows_cuda_strict_32k_smoke(
    *,
    output_dir: pathlib.Path,
    outer_attempt_lease_id: str,
    protocol_path: pathlib.Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> WindowsCudaStrict32KSmokeResult:
    """Fsync one launch receipt, execute once, and publish if complete."""

    output = pathlib.Path(output_dir).absolute()
    _require_sha256(outer_attempt_lease_id, "outer_attempt_lease_id")
    if output.exists():
        raise FileExistsError(f"strict 32K smoke output is create-only and exists: {output}")
    protocol = load_windows_cuda_strict_32k_smoke_protocol(protocol_path)
    _verify_critical_sources(protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    launch_payload = _build_launch_receipt(
        protocol,
        outer_attempt_lease_id=outer_attempt_lease_id,
    )
    attempt_id = _require_sha256_value(launch_payload["attempt_id"], "launch.attempt_id")
    launch_bytes = _canonical_bytes(launch_payload)
    _write_create_bytes(output / _LAUNCH_FILE, launch_bytes)
    if progress is not None:
        progress(f"sealed fsynced attempt launch {attempt_id}")

    runtime = _build_strict_runtime(protocol)
    attestation = runtime.execution_attestation
    if attestation is None:
        raise RuntimeError("strict 32K runtime omitted execution attestation")
    attestation_payload = {
        **attestation.to_payload(),
        "attestation_id": attestation.attestation_id,
    }
    _validate_execution_attestation_payload(attestation_payload, protocol=protocol)
    if progress is not None:
        progress("executing the sole frozen 32767+1 generation call")
    result = runtime.generate(
        prompt="",
        system_context="",
        chat_messages=_build_frozen_chat_messages(),
        max_new_tokens=_MAX_NEW_TOKENS,
        temperature=0.0,
        capture_residuals=True,
    )
    observation, checks = _summarize_generation(
        result=result,
        expected_execution_attestation_id=(protocol.expected_execution_attestation_id),
    )
    passed = all(checks.values())
    verdict = "passed_exact_strict_32767_plus_1_engineering_diagnostic" if passed else "failed_diagnostic_stop_no_retry"
    report_payload = {
        "schema_version": STRICT_32K_REPORT_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "outer_attempt_lease_id": outer_attempt_lease_id,
        "protocol_id": protocol.protocol_id,
        "protocol_raw_sha256": protocol.protocol_raw_sha256,
        "source_hash_mode": protocol.source_hash_mode,
        "source_sha256": protocol.source_sha256_payload(),
        "execution_attestation_id": attestation.attestation_id,
        "generation_call": dict(_DIAGNOSTIC_FACTS["generation_call"]),
        "observation": observation,
        "checks": checks,
        "passed": passed,
        "verdict": verdict,
        "evidence_firewall": dict(_EVIDENCE_FIREWALL),
        "claim_boundary": _CLAIM_BOUNDARY,
    }
    payload_bytes = {
        _LAUNCH_FILE: launch_bytes,
        _ATTESTATION_FILE: _canonical_bytes(attestation_payload),
        _REPORT_FILE: _canonical_bytes(report_payload),
    }
    for name in (_ATTESTATION_FILE, _REPORT_FILE):
        _write_create_bytes(output / name, payload_bytes[name])
    manifest_core = _build_manifest_core(
        protocol=protocol,
        attempt_id=attempt_id,
        outer_attempt_lease_id=outer_attempt_lease_id,
        execution_attestation_id=attestation.attestation_id,
        passed=passed,
        verdict=verdict,
        payload_bytes=payload_bytes,
    )
    artifact_id = _sha256_bytes(_canonical_bytes(manifest_core))
    _write_create_bytes(
        output / _MANIFEST_FILE,
        _canonical_bytes({**manifest_core, "artifact_id": artifact_id}),
    )
    completion_payload = _build_completion_receipt(
        attempt_id=attempt_id,
        outer_attempt_lease_id=outer_attempt_lease_id,
        artifact_id=artifact_id,
        protocol_id=protocol.protocol_id,
        execution_attestation_id=attestation.attestation_id,
        passed=passed,
        verdict=verdict,
    )
    _write_create_bytes(
        output / _COMPLETION_FILE,
        _canonical_bytes(completion_payload),
    )
    return _validate_artifact(
        output_dir=output,
        protocol=protocol,
        expected_outer_attempt_lease_id=outer_attempt_lease_id,
    )


def validate_windows_cuda_strict_32k_smoke(
    *,
    output_dir: pathlib.Path,
    expected_outer_attempt_lease_id: str,
    protocol_path: pathlib.Path | None = None,
) -> WindowsCudaStrict32KSmokeResult:
    """Revalidate files and lineage without importing substrate, torch, or CUDA."""

    _require_sha256(
        expected_outer_attempt_lease_id,
        "expected_outer_attempt_lease_id",
    )
    protocol = load_windows_cuda_strict_32k_smoke_protocol(protocol_path)
    _verify_critical_sources(protocol)
    return _validate_artifact(
        output_dir=pathlib.Path(output_dir).absolute(),
        protocol=protocol,
        expected_outer_attempt_lease_id=expected_outer_attempt_lease_id,
    )


def _build_strict_runtime(protocol: WindowsCudaStrict32KSmokeProtocol) -> Any:
    from volvence_zero.substrate import (
        WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1,
        build_transformers_runtime_with_fallback,
    )

    profile = WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1
    if profile.profile_id != protocol.profile_id:
        raise ValueError("public strict execution profile ID drift")
    _require_literal(profile.to_payload(), _PROFILE_FACTS, "public strict execution profile")
    return build_transformers_runtime_with_fallback(
        model_id=_MODEL_ID,
        model_source=None,
        device="cuda",
        layer_indices=(_LAYER_INDEX,),
        activation_width=_ACTIVATION_WIDTH,
        max_length=_CONTEXT_WINDOW_TOKENS,
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


def _build_frozen_chat_messages() -> tuple[tuple[str, str], ...]:
    return (
        ("system", ""),
        ("user", _PROMPT_UNIT * _PROMPT_UNIT_REPEAT_COUNT),
    )


def _summarize_generation(
    *, result: Any, expected_execution_attestation_id: str
) -> tuple[dict[str, object], dict[str, bool]]:
    generated_text = _require_exact_text(result.text, "GenerationResult.text")
    token_count = _require_int(result.token_count, "GenerationResult.token_count")
    input_token_count = _require_int(result.input_token_count, "GenerationResult.input_token_count")
    source_sha256 = _require_sha256_value(result.source_sha256, "GenerationResult.source_sha256")
    execution_attestation_id = _require_sha256_value(
        result.execution_attestation_id,
        "GenerationResult.execution_attestation_id",
    )
    budget = result.context_budget
    budget_payload: dict[str, object] | None = None
    budget_exact = False
    if budget is not None:
        budget_payload = {
            "schema_version": _require_text_value(budget.schema_version, "context_budget.schema_version"),
            "execution_attestation_id": _require_sha256_value(
                budget.execution_attestation_id,
                "context_budget.execution_attestation_id",
            ),
            "input_mode": _require_text_value(budget.input_mode, "context_budget.input_mode"),
            "input_token_count": _require_int(budget.input_token_count, "context_budget.input_token_count"),
            "prefix_slot_count": _require_int(budget.prefix_slot_count, "context_budget.prefix_slot_count"),
            "effective_max_new_tokens": _require_int(
                budget.effective_max_new_tokens,
                "context_budget.effective_max_new_tokens",
            ),
            "combined_token_count": _require_int(
                budget.combined_token_count,
                "context_budget.combined_token_count",
            ),
            "context_window_tokens": _require_int(
                budget.context_window_tokens,
                "context_budget.context_window_tokens",
            ),
            "remaining_token_count": _require_int(
                budget.remaining_token_count,
                "context_budget.remaining_token_count",
            ),
        }
        expected_budget = {
            **_DIAGNOSTIC_FACTS["expected_context_budget"],
            "execution_attestation_id": expected_execution_attestation_id,
        }
        budget_exact = budget_payload == expected_budget

    capture_summary, capture_checks = _summarize_capture(result.capture)
    flags = {
        "personal_conditioning_applied": _require_bool(
            result.personal_conditioning_applied,
            "GenerationResult.personal_conditioning_applied",
        ),
        "conditioning_bank_carrier_count": len(
            _require_exact_tuple(
                result.conditioning_bank_carriers_applied,
                "GenerationResult.conditioning_bank_carriers_applied",
            )
        ),
        "character_prefix_applied": _require_bool(
            result.character_prefix_applied,
            "GenerationResult.character_prefix_applied",
        ),
        "character_residual_applied": _require_bool(
            result.character_residual_applied,
            "GenerationResult.character_residual_applied",
        ),
        "steering_intervention_applied": _require_bool(
            result.steering_intervention_applied,
            "GenerationResult.steering_intervention_applied",
        ),
    }
    no_intervention = not any(
        (
            flags["personal_conditioning_applied"],
            flags["conditioning_bank_carrier_count"] != 0,
            flags["character_prefix_applied"],
            flags["character_residual_applied"],
            flags["steering_intervention_applied"],
        )
    )
    checks = {
        "execution_attestation_exact": (execution_attestation_id == expected_execution_attestation_id),
        "generation_execution_lineage_exact": (
            budget_payload is not None and budget_payload["execution_attestation_id"] == execution_attestation_id
        ),
        "rendered_prompt_hash_exact": source_sha256 == _RENDERED_PROMPT_SHA256,
        "input_token_count_exact": input_token_count == _INPUT_TOKEN_COUNT,
        "context_budget_exact": budget_exact,
        "generated_token_count_exact": token_count == _MAX_NEW_TOKENS,
        **capture_checks,
        "no_conditioning_or_steering_applied": no_intervention,
    }
    if tuple(checks) != _CHECK_KEYS:
        raise AssertionError("strict 32K check order/set drift")
    observation = {
        "generated_text_sha256": _sha256_bytes(generated_text.encode("utf-8")),
        "generated_text_byte_count": len(generated_text.encode("utf-8")),
        "generated_token_count": token_count,
        "input_token_count": input_token_count,
        "rendered_prompt_sha256": source_sha256,
        "context_budget": budget_payload,
        "capture": capture_summary,
        "application_flags": flags,
    }
    return observation, checks


def _summarize_capture(
    capture: Any | None,
) -> tuple[dict[str, object] | None, dict[str, bool]]:
    if capture is None:
        return None, _capture_checks_from_summary(None)
    from volvence_zero.substrate.strict_capture_audit import audit_strict_capture

    owner_summary = audit_strict_capture(
        capture,
        expected_layer_index=_LAYER_INDEX,
        expected_activation_width=_ACTIVATION_WIDTH,
    )
    summary = owner_summary.to_payload()
    if summary["schema_version"] != _CAPTURE_AUDIT_SCHEMA_VERSION:
        raise ValueError("strict capture owner summary schema drift")
    return summary, _capture_checks_from_summary(summary)


def _capture_checks_from_summary(
    summary: Mapping[str, Any] | None,
) -> dict[str, bool]:
    if summary is None:
        return {name: False for name in _CAPTURE_CHECK_KEYS}
    feature_values = _require_mapping(
        summary["selected_feature_values"],
        "capture.selected_feature_values",
    )
    expected_capture = _DIAGNOSTIC_FACTS["expected_capture"]
    checks = {
        "capture_present": True,
        "residual_sequence_length_exact": (
            summary["residual_sequence_length"] == expected_capture["residual_sequence_length"]
        ),
        "residual_step_continuity_exact": (summary["residual_step_continuity_exact"] is True),
        "capture_layer_exact": summary["capture_layer_exact"] is True,
        "capture_width_exact": (
            summary["capture_width_exact"] is True
            and summary["latest_activation_width"] == expected_capture["activation_width"]
        ),
        "latest_capture_matches_sequence_exact": (
            summary["latest_matches_sequence_exact"] is expected_capture["latest_matches_sequence_exact"]
        ),
        "capture_values_all_finite": (
            summary["capture_values_all_finite"] is True
            and summary["residual_activation_value_count"]
            == expected_capture["residual_sequence_length"] * expected_capture["activation_width"]
            and summary["finite_residual_activation_value_count"] == summary["residual_activation_value_count"]
        ),
        "top_logits_finite_nonempty": (
            summary["top_logits_finite_nonempty"] is True and summary["top_logit_count"] > 0
        ),
        "hook_layer_coverage_exact": (feature_values["hook_layer_coverage"] == expected_capture["hook_layer_coverage"]),
        "hook_fire_rate_exact": (feature_values["hook_fire_rate"] == expected_capture["hook_fire_rate"]),
        "token_step_coverage_exact": (feature_values["token_step_coverage"] == expected_capture["token_step_coverage"]),
        "residual_sequence_present_exact": (
            feature_values["residual_sequence_present"] == expected_capture["residual_sequence_present"]
        ),
        "fallback_inactive": (feature_values["fallback_active"] == expected_capture["fallback_active"]),
    }
    if tuple(checks) != _CAPTURE_CHECK_KEYS:
        raise AssertionError("strict 32K capture check order/set drift")
    return checks


def _build_manifest_core(
    *,
    protocol: WindowsCudaStrict32KSmokeProtocol,
    attempt_id: str,
    outer_attempt_lease_id: str,
    execution_attestation_id: str,
    passed: bool,
    verdict: str,
    payload_bytes: Mapping[str, bytes],
) -> dict[str, object]:
    if tuple(payload_bytes) != _PAYLOAD_FILES:
        raise ValueError("strict 32K manifest payload file set drift")
    return {
        "schema_version": STRICT_32K_MANIFEST_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "outer_attempt_lease_id": outer_attempt_lease_id,
        "protocol_id": protocol.protocol_id,
        "protocol_raw_sha256": protocol.protocol_raw_sha256,
        "source_hash_mode": protocol.source_hash_mode,
        "source_sha256": protocol.source_sha256_payload(),
        "execution_attestation_id": execution_attestation_id,
        "passed": passed,
        "verdict": verdict,
        "files": [
            {
                "path": name,
                "byte_count": len(payload),
                "sha256": _sha256_bytes(payload),
            }
            for name, payload in payload_bytes.items()
        ],
        "evidence_firewall": dict(_EVIDENCE_FIREWALL),
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def _validate_launch_receipt(
    launch: Mapping[str, Any],
    *,
    protocol: WindowsCudaStrict32KSmokeProtocol,
    expected_outer_attempt_lease_id: str,
) -> tuple[str, datetime]:
    _require_exact_keys(
        launch,
        {
            "schema_version",
            "protocol_id",
            "protocol_raw_sha256",
            "source_hash_mode",
            "source_sha256",
            "attempt_budget",
            "retry_budget",
            "attempt_budget_scope",
            "retry_enforcement_owner",
            "outer_attempt_lease_id",
            "process_id",
            "started_at_utc",
            "attempt_id",
        },
        "strict 32K launch receipt",
    )
    if launch["schema_version"] != STRICT_32K_LAUNCH_SCHEMA_VERSION:
        raise ValueError("strict 32K launch receipt schema drift")
    _validate_protocol_source_lineage(
        launch,
        protocol=protocol,
        label="launch receipt",
    )
    _require_literal(launch["attempt_budget"], 1, "launch.attempt_budget")
    _require_literal(launch["retry_budget"], 0, "launch.retry_budget")
    _require_literal(
        launch["attempt_budget_scope"],
        "per_frozen_output_root",
        "launch.attempt_budget_scope",
    )
    _require_literal(
        launch["retry_enforcement_owner"],
        "outer_host_campaign",
        "launch.retry_enforcement_owner",
    )
    outer_attempt_lease_id = _require_sha256_value(
        launch["outer_attempt_lease_id"],
        "launch.outer_attempt_lease_id",
    )
    if outer_attempt_lease_id != expected_outer_attempt_lease_id:
        raise ValueError("strict 32K outer attempt lease drift")
    if _require_int(launch["process_id"], "launch.process_id") <= 0:
        raise ValueError("strict 32K launch process_id must be positive")
    started_at_utc = _require_utc_timestamp(
        launch["started_at_utc"],
        "launch.started_at_utc",
    )
    attempt_id = _require_sha256_value(
        launch["attempt_id"],
        "launch.attempt_id",
    )
    core = dict(launch)
    del core["attempt_id"]
    if attempt_id != _sha256_bytes(_canonical_bytes(core)):
        raise ValueError("strict 32K launch attempt_id drift")
    return attempt_id, started_at_utc


def _validate_completion_receipt(
    completion: Mapping[str, Any],
    *,
    protocol: WindowsCudaStrict32KSmokeProtocol,
    expected_attempt_id: str,
    expected_outer_attempt_lease_id: str,
    expected_artifact_id: str,
    expected_execution_attestation_id: str,
    expected_passed: bool,
    expected_verdict: str,
    expected_started_at_utc: datetime,
) -> None:
    _require_exact_keys(
        completion,
        {
            "schema_version",
            "attempt_id",
            "outer_attempt_lease_id",
            "artifact_id",
            "protocol_id",
            "execution_attestation_id",
            "passed",
            "verdict",
            "completed_at_utc",
            "completion_id",
        },
        "strict 32K completion receipt",
    )
    if completion["schema_version"] != STRICT_32K_COMPLETION_SCHEMA_VERSION:
        raise ValueError("strict 32K completion receipt schema drift")
    values = (
        (
            _require_sha256_value(completion["attempt_id"], "completion.attempt_id"),
            expected_attempt_id,
            "attempt_id",
        ),
        (
            _require_sha256_value(
                completion["outer_attempt_lease_id"],
                "completion.outer_attempt_lease_id",
            ),
            expected_outer_attempt_lease_id,
            "outer_attempt_lease_id",
        ),
        (
            _require_sha256_value(completion["artifact_id"], "completion.artifact_id"),
            expected_artifact_id,
            "artifact_id",
        ),
        (
            _require_sha256_value(completion["protocol_id"], "completion.protocol_id"),
            protocol.protocol_id,
            "protocol_id",
        ),
        (
            _require_sha256_value(
                completion["execution_attestation_id"],
                "completion.execution_attestation_id",
            ),
            expected_execution_attestation_id,
            "execution_attestation_id",
        ),
    )
    for actual, expected, label in values:
        if actual != expected:
            raise ValueError(f"strict 32K completion {label} drift")
    if _require_bool(completion["passed"], "completion.passed") is not expected_passed:
        raise ValueError("strict 32K completion passed drift")
    if _require_text_value(completion["verdict"], "completion.verdict") != expected_verdict:
        raise ValueError("strict 32K completion verdict drift")
    completed_at_utc = _require_utc_timestamp(
        completion["completed_at_utc"],
        "completion.completed_at_utc",
    )
    if completed_at_utc < expected_started_at_utc:
        raise ValueError("strict 32K completion predates launch")
    completion_id = _require_sha256_value(
        completion["completion_id"],
        "completion.completion_id",
    )
    core = dict(completion)
    del core["completion_id"]
    if completion_id != _sha256_bytes(_canonical_bytes(core)):
        raise ValueError("strict 32K completion completion_id drift")


def _validate_artifact(
    *,
    output_dir: pathlib.Path,
    protocol: WindowsCudaStrict32KSmokeProtocol,
    expected_outer_attempt_lease_id: str,
) -> WindowsCudaStrict32KSmokeResult:
    output = pathlib.Path(output_dir).absolute()
    if not output.is_dir() or output.is_symlink():
        raise FileNotFoundError(f"strict 32K artifact root is missing or invalid: {output}")
    actual_files = tuple(sorted(path.name for path in output.iterdir()))
    if actual_files != tuple(sorted(_REQUIRED_FILES)):
        raise ValueError("strict 32K artifact root file set drift")
    launch = _load_regular_json_object(
        output / _LAUNCH_FILE,
        label="strict 32K launch receipt",
    )
    attempt_id, started_at_utc = _validate_launch_receipt(
        launch,
        protocol=protocol,
        expected_outer_attempt_lease_id=expected_outer_attempt_lease_id,
    )
    manifest = _load_regular_json_object(
        output / _MANIFEST_FILE,
        label="strict 32K manifest",
    )
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "attempt_id",
            "outer_attempt_lease_id",
            "protocol_id",
            "protocol_raw_sha256",
            "source_hash_mode",
            "source_sha256",
            "execution_attestation_id",
            "passed",
            "verdict",
            "files",
            "evidence_firewall",
            "claim_boundary",
            "artifact_id",
        },
        "strict 32K manifest",
    )
    if manifest["schema_version"] != STRICT_32K_MANIFEST_SCHEMA_VERSION:
        raise ValueError("strict 32K manifest schema drift")
    _validate_common_lineage(manifest, protocol=protocol, label="manifest")
    manifest_attempt_id = _require_sha256_value(
        manifest["attempt_id"],
        "manifest.attempt_id",
    )
    manifest_outer_attempt_lease_id = _require_sha256_value(
        manifest["outer_attempt_lease_id"],
        "manifest.outer_attempt_lease_id",
    )
    if manifest_attempt_id != attempt_id or manifest_outer_attempt_lease_id != expected_outer_attempt_lease_id:
        raise ValueError("strict 32K launch/manifest attempt lineage drift")
    artifact_id = _require_sha256_value(manifest["artifact_id"], "manifest.artifact_id")
    manifest_core = dict(manifest)
    del manifest_core["artifact_id"]
    if artifact_id != _sha256_bytes(_canonical_bytes(manifest_core)):
        raise ValueError("strict 32K manifest artifact_id drift")
    file_records = _require_list(manifest["files"], "manifest.files")
    if len(file_records) != len(_PAYLOAD_FILES):
        raise ValueError("strict 32K manifest file record count drift")
    names: list[str] = []
    for index, raw_record in enumerate(file_records):
        record = _require_mapping(raw_record, f"manifest.files[{index}]")
        _require_exact_keys(record, {"path", "byte_count", "sha256"}, "manifest file record")
        name = _require_text_value(record["path"], "manifest file path")
        if name not in _PAYLOAD_FILES or pathlib.PurePath(name).name != name:
            raise ValueError("strict 32K manifest file path drift")
        payload_path = output / name
        if not payload_path.is_file() or payload_path.is_symlink():
            raise ValueError(f"strict 32K payload is missing or a symlink: {name}")
        payload = payload_path.read_bytes()
        if _require_int(record["byte_count"], "file byte_count") != len(payload):
            raise ValueError(f"strict 32K payload byte count drift: {name}")
        if _require_sha256_value(record["sha256"], "file sha256") != _sha256_bytes(payload):
            raise ValueError(f"strict 32K payload SHA-256 drift: {name}")
        names.append(name)
    if tuple(names) != _PAYLOAD_FILES:
        raise ValueError("strict 32K manifest file order drift")

    attestation = _load_regular_json_object(
        output / _ATTESTATION_FILE,
        label="strict 32K execution attestation",
    )
    _validate_execution_attestation_payload(attestation, protocol=protocol)
    report = _load_regular_json_object(
        output / _REPORT_FILE,
        label="strict 32K report",
    )
    _validate_report_payload(report, protocol=protocol)
    execution_attestation_id = _require_sha256_value(
        manifest["execution_attestation_id"],
        "manifest.execution_attestation_id",
    )
    passed = _require_bool(manifest["passed"], "manifest.passed")
    verdict = _require_text_value(manifest["verdict"], "manifest.verdict")
    report_attempt_id = _require_sha256_value(
        report["attempt_id"],
        "report.attempt_id",
    )
    report_outer_attempt_lease_id = _require_sha256_value(
        report["outer_attempt_lease_id"],
        "report.outer_attempt_lease_id",
    )
    if (
        execution_attestation_id != attestation["attestation_id"]
        or execution_attestation_id != report["execution_attestation_id"]
        or attempt_id != report_attempt_id
        or expected_outer_attempt_lease_id != report_outer_attempt_lease_id
        or passed is not report["passed"]
        or verdict != report["verdict"]
    ):
        raise ValueError("strict 32K manifest/report/attestation lineage drift")
    completion = _load_regular_json_object(
        output / _COMPLETION_FILE,
        label="strict 32K completion receipt",
    )
    _validate_completion_receipt(
        completion,
        protocol=protocol,
        expected_attempt_id=attempt_id,
        expected_outer_attempt_lease_id=expected_outer_attempt_lease_id,
        expected_artifact_id=artifact_id,
        expected_execution_attestation_id=execution_attestation_id,
        expected_passed=passed,
        expected_verdict=verdict,
        expected_started_at_utc=started_at_utc,
    )
    return WindowsCudaStrict32KSmokeResult(
        artifact_id=artifact_id,
        attempt_id=attempt_id,
        outer_attempt_lease_id=expected_outer_attempt_lease_id,
        protocol_id=protocol.protocol_id,
        execution_attestation_id=execution_attestation_id,
        passed=passed,
        verdict=verdict,
        output_dir=output.resolve(),
    )


def _validate_report_payload(report: Mapping[str, Any], *, protocol: WindowsCudaStrict32KSmokeProtocol) -> None:
    _require_exact_keys(
        report,
        {
            "schema_version",
            "attempt_id",
            "outer_attempt_lease_id",
            "protocol_id",
            "protocol_raw_sha256",
            "source_hash_mode",
            "source_sha256",
            "execution_attestation_id",
            "generation_call",
            "observation",
            "checks",
            "passed",
            "verdict",
            "evidence_firewall",
            "claim_boundary",
        },
        "strict 32K report",
    )
    if report["schema_version"] != STRICT_32K_REPORT_SCHEMA_VERSION:
        raise ValueError("strict 32K report schema drift")
    _require_sha256_value(report["attempt_id"], "report.attempt_id")
    _require_sha256_value(
        report["outer_attempt_lease_id"],
        "report.outer_attempt_lease_id",
    )
    _validate_common_lineage(report, protocol=protocol, label="report")
    _require_literal(
        report["generation_call"],
        _DIAGNOSTIC_FACTS["generation_call"],
        "report.generation_call",
    )
    _validate_observation(_require_mapping(report["observation"], "report.observation"))
    checks = _require_mapping(report["checks"], "report.checks")
    if set(checks) != set(_CHECK_KEYS):
        raise ValueError("strict 32K report check set drift")
    if any(type(value) is not bool for value in checks.values()):
        raise TypeError("strict 32K report checks must be exact bools")
    _require_literal(
        checks,
        _recompute_report_checks(report, protocol=protocol),
        "report.checks",
    )
    passed = _require_bool(report["passed"], "report.passed")
    if passed is not all(checks.values()):
        raise ValueError("strict 32K report verdict/check drift")
    expected_verdict = (
        "passed_exact_strict_32767_plus_1_engineering_diagnostic" if passed else "failed_diagnostic_stop_no_retry"
    )
    if report["verdict"] != expected_verdict:
        raise ValueError("strict 32K report verdict drift")


def _validate_observation(observation: Mapping[str, Any]) -> None:
    _require_exact_keys(
        observation,
        {
            "generated_text_sha256",
            "generated_text_byte_count",
            "generated_token_count",
            "input_token_count",
            "rendered_prompt_sha256",
            "context_budget",
            "capture",
            "application_flags",
        },
        "strict 32K observation",
    )
    _require_sha256_value(observation["generated_text_sha256"], "generated_text_sha256")
    _require_sha256_value(observation["rendered_prompt_sha256"], "rendered_prompt_sha256")
    for name in (
        "generated_text_byte_count",
        "generated_token_count",
        "input_token_count",
    ):
        if _require_int(observation[name], f"observation.{name}") < 0:
            raise ValueError(f"observation.{name} must be non-negative")
    budget = observation["context_budget"]
    if budget is not None:
        budget_mapping = _require_mapping(budget, "observation.context_budget")
        _require_exact_keys(
            budget_mapping,
            {
                "schema_version",
                "execution_attestation_id",
                "input_mode",
                "input_token_count",
                "prefix_slot_count",
                "effective_max_new_tokens",
                "combined_token_count",
                "context_window_tokens",
                "remaining_token_count",
            },
            "observation.context_budget",
        )
        _require_sha256_value(
            budget_mapping["execution_attestation_id"],
            "context_budget.execution_attestation_id",
        )
        _require_text_value(budget_mapping["schema_version"], "context_budget.schema_version")
        _require_text_value(budget_mapping["input_mode"], "context_budget.input_mode")
        for name in (
            "input_token_count",
            "prefix_slot_count",
            "effective_max_new_tokens",
            "combined_token_count",
            "context_window_tokens",
            "remaining_token_count",
        ):
            if _require_int(budget_mapping[name], f"context_budget.{name}") < 0:
                raise ValueError(f"context_budget.{name} must be non-negative")
    capture = observation["capture"]
    if capture is not None:
        capture_mapping = _require_mapping(capture, "observation.capture")
        _require_exact_keys(
            capture_mapping,
            {
                "schema_version",
                "residual_sequence_length",
                "residual_step_continuity_exact",
                "capture_layer_exact",
                "capture_width_exact",
                "residual_activation_value_count",
                "finite_residual_activation_value_count",
                "capture_values_all_finite",
                "residual_sequence_sha256",
                "latest_activation_width",
                "latest_activation_sha256",
                "latest_matches_sequence_exact",
                "top_logit_count",
                "top_logits_finite_nonempty",
                "top_logits_sha256",
                "selected_feature_values",
                "description_sha256",
            },
            "observation.capture",
        )
        if capture_mapping["schema_version"] != _CAPTURE_AUDIT_SCHEMA_VERSION:
            raise ValueError("strict capture audit summary schema drift")
        for name in (
            "residual_sequence_sha256",
            "latest_activation_sha256",
            "top_logits_sha256",
            "description_sha256",
        ):
            _require_sha256_value(capture_mapping[name], f"capture.{name}")
        if any(isinstance(value, (list, tuple)) for value in capture_mapping.values()):
            raise ValueError("strict 32K report persisted unbounded capture data")
        for name in (
            "residual_step_continuity_exact",
            "capture_layer_exact",
            "capture_width_exact",
            "capture_values_all_finite",
            "latest_matches_sequence_exact",
            "top_logits_finite_nonempty",
        ):
            _require_bool(capture_mapping[name], f"capture.{name}")
        for name in (
            "residual_sequence_length",
            "residual_activation_value_count",
            "finite_residual_activation_value_count",
            "latest_activation_width",
            "top_logit_count",
        ):
            if _require_int(capture_mapping[name], f"capture.{name}") < 0:
                raise ValueError(f"capture.{name} must be non-negative")
        feature_values = _require_mapping(
            capture_mapping["selected_feature_values"],
            "capture.selected_feature_values",
        )
        _require_exact_keys(
            feature_values,
            {
                "hook_layer_coverage",
                "hook_fire_rate",
                "token_step_coverage",
                "residual_sequence_present",
                "fallback_active",
            },
            "capture.selected_feature_values",
        )
        for name, value in feature_values.items():
            if value is not None and not math.isfinite(
                _require_number(value, f"capture.selected_feature_values.{name}")
            ):
                raise ValueError(f"capture feature {name} must be finite or null")
    flags = _require_mapping(observation["application_flags"], "observation.application_flags")
    _require_exact_keys(
        flags,
        {
            "personal_conditioning_applied",
            "conditioning_bank_carrier_count",
            "character_prefix_applied",
            "character_residual_applied",
            "steering_intervention_applied",
        },
        "observation.application_flags",
    )
    if (
        _require_int(
            flags["conditioning_bank_carrier_count"],
            "conditioning_bank_carrier_count",
        )
        < 0
    ):
        raise ValueError("conditioning_bank_carrier_count must be non-negative")
    for name in (
        "personal_conditioning_applied",
        "character_prefix_applied",
        "character_residual_applied",
        "steering_intervention_applied",
    ):
        _require_bool(flags[name], f"application_flags.{name}")


def _recompute_report_checks(
    report: Mapping[str, Any],
    *,
    protocol: WindowsCudaStrict32KSmokeProtocol,
) -> dict[str, bool]:
    observation = _require_mapping(report["observation"], "report.observation")
    budget_raw = observation["context_budget"]
    budget = _require_mapping(budget_raw, "observation.context_budget") if budget_raw is not None else None
    expected_budget = {
        **_DIAGNOSTIC_FACTS["expected_context_budget"],
        "execution_attestation_id": protocol.expected_execution_attestation_id,
    }
    budget_exact = False
    if budget is not None:
        budget_exact = budget == expected_budget
    capture_raw = observation["capture"]
    capture = _require_mapping(capture_raw, "observation.capture") if capture_raw is not None else None
    capture_checks = _capture_checks_from_summary(capture)
    flags = _require_mapping(observation["application_flags"], "observation.application_flags")
    no_intervention = not any(
        (
            flags["personal_conditioning_applied"],
            flags["conditioning_bank_carrier_count"] != 0,
            flags["character_prefix_applied"],
            flags["character_residual_applied"],
            flags["steering_intervention_applied"],
        )
    )
    execution_attestation_id = report["execution_attestation_id"]
    checks = {
        "execution_attestation_exact": (execution_attestation_id == protocol.expected_execution_attestation_id),
        "generation_execution_lineage_exact": (
            budget is not None and budget["execution_attestation_id"] == execution_attestation_id
        ),
        "rendered_prompt_hash_exact": (observation["rendered_prompt_sha256"] == _RENDERED_PROMPT_SHA256),
        "input_token_count_exact": (observation["input_token_count"] == _INPUT_TOKEN_COUNT),
        "context_budget_exact": budget_exact,
        "generated_token_count_exact": (observation["generated_token_count"] == _MAX_NEW_TOKENS),
        **capture_checks,
        "no_conditioning_or_steering_applied": no_intervention,
    }
    if tuple(checks) != _CHECK_KEYS:
        raise AssertionError("strict 32K recomputed check order/set drift")
    return checks


def _validate_execution_attestation_payload(
    attestation: Mapping[str, Any],
    *,
    protocol: WindowsCudaStrict32KSmokeProtocol,
) -> None:
    _require_exact_keys(attestation, _ATTESTATION_KEYS, "execution attestation")
    attestation_id = _require_sha256_value(attestation["attestation_id"], "attestation.attestation_id")
    canonical = dict(attestation)
    del canonical["attestation_id"]
    if attestation_id != _sha256_bytes(_canonical_bytes(canonical, newline=False)):
        raise ValueError("execution attestation ID drift")
    if attestation_id != protocol.expected_execution_attestation_id:
        raise ValueError("execution attestation differs from frozen protocol")
    expected = {
        "profile_id": _PROFILE_ID,
        "preset_name": _PROFILE_FACTS["preset_name"],
        "model_id": _MODEL_ID,
        "model_revision": _MODEL_REVISION,
        "model_weights_sha256": _MODEL_WEIGHTS_SHA256,
        "execution_assets_sha256": _EXECUTION_ASSETS_SHA256,
        "runtime_origin": "hf-local",
        "platform_system": "Windows",
        "attention_implementation": "sdpa",
        "sdpa_backend": "cudnn",
        "sdpa_backend_policy": "exclusive-cudnn",
        "sdpa_backend_exclusive": True,
        "generation_use_cache": True,
        "require_generation_chat_template": True,
        "generation_capture_strategy": "first-full-prompt-set-once",
        "capture_failure_mode": "raise",
        "context_window_tokens": _CONTEXT_WINDOW_TOKENS,
        "local_files_only": True,
        "fallback_mode": "deny",
        "fail_on_truncation": True,
        "model_dtype": "bfloat16",
        "hidden_size": _ACTIVATION_WIDTH,
        "model_max_position_embeddings": _CONTEXT_WINDOW_TOKENS,
        "hook_layer_indices": [_LAYER_INDEX],
    }
    for name, value in expected.items():
        _require_literal(attestation[name], value, f"attestation.{name}")
    device = _require_text_value(attestation["device"], "attestation.device")
    if device != "cuda" and not (device.startswith("cuda:") and device.removeprefix("cuda:").isdigit()):
        raise ValueError("execution attestation device is not CUDA")


def _validate_common_lineage(
    payload: Mapping[str, Any],
    *,
    protocol: WindowsCudaStrict32KSmokeProtocol,
    label: str,
) -> None:
    _validate_protocol_source_lineage(
        payload,
        protocol=protocol,
        label=label,
    )
    _require_literal(
        payload["evidence_firewall"],
        _EVIDENCE_FIREWALL,
        f"{label}.evidence_firewall",
    )
    if payload["claim_boundary"] != _CLAIM_BOUNDARY:
        raise ValueError(f"strict 32K {label} claim boundary drift")


def _validate_protocol_source_lineage(
    payload: Mapping[str, Any],
    *,
    protocol: WindowsCudaStrict32KSmokeProtocol,
    label: str,
) -> None:
    if (
        payload["protocol_id"] != protocol.protocol_id
        or payload["protocol_raw_sha256"] != protocol.protocol_raw_sha256
        or payload["source_hash_mode"] != protocol.source_hash_mode
    ):
        raise ValueError(f"strict 32K {label} protocol/source lineage drift")
    _require_literal(
        payload["source_sha256"],
        protocol.source_sha256_payload(),
        f"{label}.source_sha256",
    )


def _validate_protocol_payload(raw: Mapping[str, Any]) -> None:
    _require_exact_keys(
        raw,
        {
            "schema_version",
            "owner",
            "model",
            "execution_profile",
            "diagnostic",
            "source_hash_mode",
            "source_sha256",
            "output_contract",
            "evidence_firewall",
            "claim_boundary",
        },
        "strict 32K protocol",
    )
    if raw["schema_version"] != STRICT_32K_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("strict 32K protocol schema drift")
    _require_literal(raw["owner"], _OWNER_FACTS, "protocol.owner")
    _require_literal(raw["model"], _MODEL_FACTS, "protocol.model")
    _require_literal(
        raw["execution_profile"],
        _PROTOCOL_PROFILE_FACTS,
        "protocol.execution_profile",
    )
    _require_literal(raw["diagnostic"], _DIAGNOSTIC_FACTS, "protocol.diagnostic")
    if raw["source_hash_mode"] != _SOURCE_HASH_MODE:
        raise ValueError("strict 32K protocol source hash mode drift")
    source_sha256 = _require_mapping(raw["source_sha256"], "source_sha256")
    if tuple(source_sha256) != _CRITICAL_SOURCE_PATHS:
        raise ValueError("strict 32K protocol source path order/set drift")
    for path in _CRITICAL_SOURCE_PATHS:
        _require_sha256_value(source_sha256[path], f"source_sha256.{path}")
    _require_literal(raw["output_contract"], _OUTPUT_CONTRACT, "protocol.output_contract")
    _require_literal(
        raw["evidence_firewall"],
        _EVIDENCE_FIREWALL,
        "protocol.evidence_firewall",
    )
    if raw["claim_boundary"] != _CLAIM_BOUNDARY:
        raise ValueError("strict 32K protocol claim boundary drift")


def _verify_critical_sources(protocol: WindowsCudaStrict32KSmokeProtocol) -> None:
    repository_root = _REPOSITORY_ROOT.resolve()
    for relative, expected in protocol.source_sha256:
        source = repository_root.joinpath(*pathlib.PurePosixPath(relative).parts)
        try:
            source.resolve().relative_to(repository_root)
        except ValueError as exc:
            raise ValueError(f"critical source escapes repository: {relative}") from exc
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"critical source is missing/symlink: {relative}")
        if _source_text_sha256(source) != expected:
            raise ValueError(f"critical source SHA-256 drift: {relative}")


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


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _write_create_bytes(path: pathlib.Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _load_json_object(path: pathlib.Path) -> dict[str, Any]:
    return _load_json_object_bytes(path.read_bytes(), label=str(path))


def _load_regular_json_object(
    path: pathlib.Path,
    *,
    label: str,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing or a symlink")
    return _load_json_object(path)


def _load_json_object_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
        value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except UnicodeDecodeError as exc:
        raise ValueError(f"JSON is not UTF-8: {label}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON is invalid: {label}") from exc
    if type(value) is not dict:
        raise TypeError(f"JSON root must be an exact object: {label}")
    return value


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


def _require_exact_tuple(value: object, label: str) -> tuple[Any, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be an exact tuple")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys drifted")


def _require_exact_text(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be exact text")
    return value


def _require_text_value(value: object, label: str) -> str:
    text = _require_exact_text(value, label)
    if not text.strip():
        raise ValueError(f"{label} must be nonempty text")
    return text


def _require_utc_timestamp(value: object, label: str) -> datetime:
    text = _require_text_value(value, label)
    if not text.endswith("Z"):
        raise ValueError(f"{label} must be canonical UTC text")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} must be canonical UTC text") from exc
    canonical = parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if parsed.tzinfo != timezone.utc or text != canonical:
        raise ValueError(f"{label} must be canonical UTC text")
    return parsed


def _require_relative_posix_path(value: object, label: str) -> str:
    text = _require_text_value(value, label)
    pure = pathlib.PurePosixPath(text)
    if pure.is_absolute() or "\\" in text or str(pure) != text or any(part in ("", ".", "..") for part in pure.parts):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return text


def _require_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact int")
    return value


def _require_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be an exact bool")
    return value


def _require_number(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{label} must be an exact number")
    return float(value)


def _require_sha256(value: str, label: str) -> None:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _require_sha256_value(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be text")
    _require_sha256(value, label)
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_text_sha256(path: pathlib.Path) -> str:
    payload = path.read_bytes()
    if payload.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"critical source carries UTF-8 BOM: {path}")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"critical source is not UTF-8: {path}") from exc
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return _sha256_bytes(canonical.encode("utf-8"))


__all__ = (
    "STRICT_32K_COMPLETION_SCHEMA_VERSION",
    "STRICT_32K_LAUNCH_SCHEMA_VERSION",
    "STRICT_32K_MANIFEST_SCHEMA_VERSION",
    "STRICT_32K_PROTOCOL_SCHEMA_VERSION",
    "STRICT_32K_REPORT_SCHEMA_VERSION",
    "WindowsCudaStrict32KSmokeProtocol",
    "WindowsCudaStrict32KSmokeResult",
    "load_windows_cuda_strict_32k_smoke_protocol",
    "run_windows_cuda_strict_32k_smoke",
    "strict_32k_smoke_protocol_path",
    "validate_windows_cuda_strict_32k_smoke",
)
