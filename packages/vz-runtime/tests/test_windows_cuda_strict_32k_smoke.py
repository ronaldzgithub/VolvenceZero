from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import json
import pathlib
import subprocess
import sys
from typing import Any

import pytest

from volvence_zero.offline_evidence import windows_cuda_strict_32k_smoke as lane


_OUTER_ATTEMPT_LEASE_ID = hashlib.sha256(b"strict-32k-test-outer-attempt-lease").hexdigest()
_OTHER_OUTER_ATTEMPT_LEASE_ID = hashlib.sha256(b"strict-32k-test-other-outer-attempt-lease").hexdigest()
_ATTESTATION_PAYLOAD: dict[str, Any] = {
    "attention_implementation": "sdpa",
    "attestation_id": "9a33a698b95d923d6a4e82b64471213d529b0cbbf6a30ca24644860211e6dde1",
    "capture_failure_mode": "raise",
    "context_window_tokens": 32768,
    "cuda_version": "12.6",
    "cudnn_version": 91002,
    "device": "cuda",
    "device_compute_capability": [8, 9],
    "device_name": "NVIDIA GeForce RTX 4090",
    "execution_assets_sha256": "bbb5446f8d802b437c2fc7e2cefcdabb996bbd4bc657fe155ea015d30a841bb0",
    "fail_on_truncation": True,
    "fallback_mode": "deny",
    "generation_capture_strategy": "first-full-prompt-set-once",
    "generation_use_cache": True,
    "hidden_size": 1536,
    "hook_layer_indices": [20],
    "local_files_only": True,
    "model_dtype": "bfloat16",
    "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
    "model_max_position_embeddings": 32768,
    "model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "model_weights_sha256": "fb8c44c48b8359fdd306cdc5f473d7c04d88955013f0dd8549f266e248194da4",
    "platform_release": "10",
    "platform_system": "Windows",
    "preset_name": "windows-cuda-cudnn-sdpa-cached-strict.v1",
    "profile_id": "3be84d866afbda07cf80dee277d89cdc0e366ce545bf7e97f015cf8afcbfe21a",
    "python_version": "3.11.15",
    "require_generation_chat_template": True,
    "runtime_origin": "hf-local",
    "schema_version": "transformers-execution-attestation.v1",
    "sdpa_backend": "cudnn",
    "sdpa_backend_exclusive": True,
    "sdpa_backend_policy": "exclusive-cudnn",
    "torch_version": "2.12.0+cu126",
    "transformers_version": "5.9.0",
}
_SMALL_INPUT_TOKEN_COUNT = 2


@dataclass(frozen=True)
class _FakeAttestation:
    payload: dict[str, Any]

    @property
    def attestation_id(self) -> str:
        return self.payload["attestation_id"]

    def to_payload(self) -> dict[str, Any]:
        return {key: value for key, value in self.payload.items() if key != "attestation_id"}


@dataclass(frozen=True)
class _FrozenContextBudget:
    schema_version: str
    execution_attestation_id: str
    input_mode: str
    input_token_count: int
    prefix_slot_count: int
    effective_max_new_tokens: int
    combined_token_count: int
    context_window_tokens: int
    remaining_token_count: int


@dataclass(frozen=True)
class _FrozenGenerationResult:
    text: str
    token_count: int
    input_token_count: int
    source_sha256: str
    execution_attestation_id: str
    context_budget: _FrozenContextBudget
    capture: object | None
    personal_conditioning_applied: bool
    conditioning_bank_carriers_applied: tuple[tuple[str, str], ...]
    character_prefix_applied: bool
    character_residual_applied: bool
    steering_intervention_applied: bool


class _FakeStrictRuntime:
    def __init__(self, *, failure_kind: str | None = None) -> None:
        self.execution_attestation = _FakeAttestation(dict(_ATTESTATION_PAYLOAD))
        self.failure_kind = failure_kind
        self.generate_calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> object:
        self.generate_calls.append(kwargs)
        assert len(self.generate_calls) == 1, "the diagnostic must never retry"
        assert kwargs == {
            "prompt": "",
            "system_context": "",
            "chat_messages": (
                ("system", ""),
                ("user", "<|im_start|>" * 32754),
            ),
            "max_new_tokens": 1,
            "temperature": 0.0,
            "capture_residuals": True,
        }
        if self.failure_kind == "runtime-exception":
            raise RuntimeError("synthetic strict runtime failure")
        return _fake_generation_result(failure_kind=self.failure_kind)


def _fake_generation_result(*, failure_kind: str | None) -> _FrozenGenerationResult:
    input_token_count = lane._INPUT_TOKEN_COUNT
    budget_input_tokens = input_token_count + (failure_kind == "budget")
    budget_schema = (
        "generation-context-budget-attestation.v2"
        if failure_kind == "budget-schema"
        else "generation-context-budget-attestation.v1"
    )
    budget = _FrozenContextBudget(
        schema_version=budget_schema,
        execution_attestation_id=_ATTESTATION_PAYLOAD["attestation_id"],
        input_mode="chat-template",
        input_token_count=budget_input_tokens,
        prefix_slot_count=0,
        effective_max_new_tokens=1,
        combined_token_count=budget_input_tokens + 1,
        context_window_tokens=32768,
        remaining_token_count=32768 - budget_input_tokens - 1,
    )
    capture = None if failure_kind == "capture" else _small_capture()
    if failure_kind == "latest-mismatch":
        capture = replace(
            capture,
            residual_activations=(capture.residual_sequence[0].residual_activations),
        )
    return _FrozenGenerationResult(
        text="x",
        token_count=1,
        input_token_count=input_token_count,
        source_sha256=("2bae362c6e83f091aa96b1902a573c99de9adc53bf996661a2f0a750d25f38b0"),
        execution_attestation_id=_ATTESTATION_PAYLOAD["attestation_id"],
        context_budget=budget,
        capture=capture,
        personal_conditioning_applied=False,
        conditioning_bank_carriers_applied=(),
        character_prefix_applied=False,
        character_residual_applied=False,
        steering_intervention_applied=False,
    )


def _small_capture() -> object:
    from volvence_zero.substrate import (
        FeatureSignal,
        OpenWeightRuntimeCapture,
        ResidualActivation,
        ResidualSequenceStep,
    )

    sequence: list[ResidualSequenceStep] = []
    for step_index in range(lane._INPUT_TOKEN_COUNT):
        activation = ResidualActivation(
            layer_index=20,
            activation=(float(step_index),) * lane._ACTIVATION_WIDTH,
            step=step_index,
        )
        sequence.append(
            ResidualSequenceStep(
                step=step_index,
                token=f"token-{step_index}",
                feature_surface=(),
                residual_activations=(activation,),
                description="small test-only strict capture step",
            )
        )
    features = tuple(
        FeatureSignal(
            name=name,
            values=(value,),
            source="strict-32k-test-owner-fixture",
        )
        for name, value in (
            ("hook_layer_coverage", 1.0),
            ("hook_fire_rate", 1.0),
            ("token_step_coverage", 1.0),
            ("residual_sequence_present", 1.0),
            ("fallback_active", 0.0),
        )
    )
    latest_activations = sequence[-1].residual_activations if sequence else ()
    return OpenWeightRuntimeCapture(
        residual_sequence=tuple(sequence),
        residual_activations=latest_activations,
        token_logits=(0.75, -0.25),
        feature_surface=features,
        description="small test-only strict capture",
    )


def _patch_small_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> lane.WindowsCudaStrict32KSmokeProtocol:
    protocol = lane.load_windows_cuda_strict_32k_smoke_protocol()
    facts = copy.deepcopy(lane._DIAGNOSTIC_FACTS)
    facts["expected_context_budget"] = {
        "schema_version": "generation-context-budget-attestation.v1",
        "input_mode": "chat-template",
        "input_token_count": _SMALL_INPUT_TOKEN_COUNT,
        "prefix_slot_count": 0,
        "effective_max_new_tokens": 1,
        "combined_token_count": _SMALL_INPUT_TOKEN_COUNT + 1,
        "context_window_tokens": 32768,
        "remaining_token_count": 32768 - _SMALL_INPUT_TOKEN_COUNT - 1,
    }
    facts["expected_capture"]["residual_sequence_length"] = _SMALL_INPUT_TOKEN_COUNT
    monkeypatch.setattr(lane, "_INPUT_TOKEN_COUNT", _SMALL_INPUT_TOKEN_COUNT)
    monkeypatch.setattr(lane, "_DIAGNOSTIC_FACTS", facts)
    monkeypatch.setattr(
        lane,
        "load_windows_cuda_strict_32k_smoke_protocol",
        lambda path=None: protocol,
    )
    return protocol


def _install_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure_kind: str | None = None,
) -> _FakeStrictRuntime:
    runtime = _FakeStrictRuntime(failure_kind=failure_kind)
    monkeypatch.setattr(lane, "_build_strict_runtime", lambda protocol: runtime)
    return runtime


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_canonical_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_bytes(_canonical_bytes(payload))


def _write_protocol(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _reseal_manifest_and_completion(output: pathlib.Path) -> None:
    manifest_path = output / lane._MANIFEST_FILE
    manifest = _read_json(manifest_path)
    manifest["files"] = [
        {
            "path": name,
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name in lane._PAYLOAD_FILES
        for payload in [(output / name).read_bytes()]
    ]
    manifest_core = dict(manifest)
    del manifest_core["artifact_id"]
    artifact_id = hashlib.sha256(_canonical_bytes(manifest_core)).hexdigest()
    manifest["artifact_id"] = artifact_id
    _write_canonical_json(manifest_path, manifest)

    completion_path = output / lane._COMPLETION_FILE
    completion = _read_json(completion_path)
    completion["artifact_id"] = artifact_id
    completion_core = dict(completion)
    del completion_core["completion_id"]
    completion["completion_id"] = hashlib.sha256(_canonical_bytes(completion_core)).hexdigest()
    _write_canonical_json(completion_path, completion)


def _run_small_completed_artifact(
    *,
    output: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[lane.WindowsCudaStrict32KSmokeResult, _FakeStrictRuntime]:
    _patch_small_diagnostic(monkeypatch)
    runtime = _install_runtime(monkeypatch)
    result = lane.run_windows_cuda_strict_32k_smoke(
        output_dir=output,
        outer_attempt_lease_id=_OUTER_ATTEMPT_LEASE_ID,
    )
    return result, runtime


def test_protocol_freezes_exact_model_source_set_prompt_and_attempt_contract() -> None:
    protocol_path = lane.strict_32k_smoke_protocol_path()
    protocol = lane.load_windows_cuda_strict_32k_smoke_protocol()
    payload = _read_json(protocol_path)

    assert protocol.protocol_raw_sha256 == hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    assert protocol.profile_id == "3be84d866afbda07cf80dee277d89cdc0e366ce545bf7e97f015cf8afcbfe21a"
    assert protocol.expected_execution_attestation_id == (
        "9a33a698b95d923d6a4e82b64471213d529b0cbbf6a30ca24644860211e6dde1"
    )
    assert payload["model"] == {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "verified_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "model_weights_sha256": "fb8c44c48b8359fdd306cdc5f473d7c04d88955013f0dd8549f266e248194da4",
        "execution_assets_sha256": "bbb5446f8d802b437c2fc7e2cefcdabb996bbd4bc657fe155ea015d30a841bb0",
        "model_source": "logical_model_id_resolved_from_verified_hf_cache",
        "local_snapshot_path_recorded": False,
    }
    expected_sources = (
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
    assert tuple(payload["source_sha256"]) == expected_sources
    assert tuple(path for path, _ in protocol.source_sha256) == expected_sources
    assert payload["diagnostic"]["attempt_budget"] == 1
    assert payload["diagnostic"]["retry_budget"] == 0
    assert payload["diagnostic"]["attempt_budget_scope"] == ("per_frozen_output_root")
    assert payload["diagnostic"]["retry_enforcement_owner"] == ("outer_host_campaign")
    assert payload["diagnostic"]["outer_attempt_lease_required"] is True
    assert payload["diagnostic"]["prompt_recipe"] == {
        "template": "qwen25_chat_template_v1",
        "messages": [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content_recipe": "repeat_unit_without_separator",
                "unit": "<|im_start|>",
                "repeat_count": 32754,
            },
        ],
        "search_or_calibration_permitted": False,
        "expected_rendered_prompt_sha256": ("2bae362c6e83f091aa96b1902a573c99de9adc53bf996661a2f0a750d25f38b0"),
        "expected_rendered_prompt_byte_count": 393128,
    }
    assert payload["diagnostic"]["expected_context_budget"] == {
        "schema_version": "generation-context-budget-attestation.v1",
        "input_mode": "chat-template",
        "input_token_count": 32767,
        "prefix_slot_count": 0,
        "effective_max_new_tokens": 1,
        "combined_token_count": 32768,
        "context_window_tokens": 32768,
        "remaining_token_count": 0,
    }
    assert payload["diagnostic"]["expected_capture"]["latest_matches_sequence_exact"] is True
    assert payload["output_contract"]["required_files"] == list(lane._REQUIRED_FILES)
    assert payload["output_contract"]["launch_receipt_fsync_before_runtime_construction"] is True
    assert payload["output_contract"]["incomplete_attempt_root_never_deleted_by_runner"] is True
    assert payload["output_contract"]["completion_not_before_launch"] is True
    assert payload["output_contract"]["outer_attempt_lease_required"] is True
    assert payload["evidence_firewall"]["external_append_only_anchor_present"] is False
    lane._verify_critical_sources(protocol)


def test_public_strict_factory_receives_every_exact_frozen_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from volvence_zero import substrate

    protocol = lane.load_windows_cuda_strict_32k_smoke_protocol()
    calls: list[dict[str, object]] = []
    sentinel = object()

    def fake_factory(**kwargs: object) -> object:
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(
        substrate,
        "build_transformers_runtime_with_fallback",
        fake_factory,
    )

    assert lane._build_strict_runtime(protocol) is sentinel
    assert calls == [
        {
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "model_source": None,
            "device": "cuda",
            "layer_indices": (20,),
            "activation_width": 1536,
            "max_length": 32768,
            "fail_on_truncation": True,
            "local_files_only": True,
            "fallback_mode": "deny",
            "runtime_mode": "strict-local",
            "model_dtype": "bfloat16",
            "expected_model_weights_sha256": ("fb8c44c48b8359fdd306cdc5f473d7c04d88955013f0dd8549f266e248194da4"),
            "execution_profile": (substrate.WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1),
            "verified_model_revision": ("989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
            "expected_execution_assets_sha256": ("bbb5446f8d802b437c2fc7e2cefcdabb996bbd4bc657fe155ea015d30a841bb0"),
        }
    ]


def test_single_call_pass_seals_launch_report_manifest_and_completion_lineage(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "strict-pass"
    result, runtime = _run_small_completed_artifact(
        output=output,
        monkeypatch=monkeypatch,
    )

    assert len(runtime.generate_calls) == 1
    assert result.passed is True
    assert result.verdict == ("passed_exact_strict_32767_plus_1_engineering_diagnostic")
    assert result.outer_attempt_lease_id == _OUTER_ATTEMPT_LEASE_ID
    assert tuple(sorted(path.name for path in output.iterdir())) == tuple(sorted(lane._REQUIRED_FILES))
    launch = _read_json(output / lane._LAUNCH_FILE)
    manifest = _read_json(output / lane._MANIFEST_FILE)
    report = _read_json(output / lane._REPORT_FILE)
    completion = _read_json(output / lane._COMPLETION_FILE)

    assert {
        launch["attempt_id"],
        report["attempt_id"],
        manifest["attempt_id"],
        completion["attempt_id"],
        result.attempt_id,
    } == {result.attempt_id}
    assert {
        launch["outer_attempt_lease_id"],
        report["outer_attempt_lease_id"],
        manifest["outer_attempt_lease_id"],
        completion["outer_attempt_lease_id"],
        result.outer_attempt_lease_id,
    } == {_OUTER_ATTEMPT_LEASE_ID}
    assert result.artifact_id == manifest["artifact_id"] == completion["artifact_id"]
    assert completion["protocol_id"] == result.protocol_id
    assert completion["execution_attestation_id"] == result.execution_attestation_id
    assert completion["passed"] is result.passed
    assert completion["verdict"] == result.verdict
    assert tuple(record["path"] for record in manifest["files"]) == lane._PAYLOAD_FILES
    capture = report["observation"]["capture"]
    assert capture["residual_sequence_length"] == _SMALL_INPUT_TOKEN_COUNT
    assert capture["latest_matches_sequence_exact"] is True
    assert "residual_sequence" not in capture
    assert report["observation"]["context_budget"]["schema_version"] == ("generation-context-budget-attestation.v1")
    assert all(report["checks"].values())
    assert (
        lane.validate_windows_cuda_strict_32k_smoke(
            output_dir=output,
            expected_outer_attempt_lease_id=_OUTER_ATTEMPT_LEASE_ID,
        )
        == result
    )

    with pytest.raises(FileExistsError, match="create-only"):
        lane.run_windows_cuda_strict_32k_smoke(
            output_dir=output,
            outer_attempt_lease_id=_OUTER_ATTEMPT_LEASE_ID,
        )
    assert len(runtime.generate_calls) == 1


def test_runtime_exception_keeps_fsynced_launch_and_forbids_same_root_rerun(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _install_runtime(monkeypatch, failure_kind="runtime-exception")
    output = tmp_path / "strict-incomplete"

    with pytest.raises(RuntimeError, match="synthetic strict runtime failure"):
        lane.run_windows_cuda_strict_32k_smoke(
            output_dir=output,
            outer_attempt_lease_id=_OUTER_ATTEMPT_LEASE_ID,
        )

    assert len(runtime.generate_calls) == 1
    assert output.is_dir()
    assert tuple(path.name for path in output.iterdir()) == (lane._LAUNCH_FILE,)
    launch = _read_json(output / lane._LAUNCH_FILE)
    assert launch["outer_attempt_lease_id"] == _OUTER_ATTEMPT_LEASE_ID
    assert len(launch["attempt_id"]) == 64
    with pytest.raises(ValueError, match="file set drift"):
        lane.validate_windows_cuda_strict_32k_smoke(
            output_dir=output,
            expected_outer_attempt_lease_id=_OUTER_ATTEMPT_LEASE_ID,
        )
    with pytest.raises(FileExistsError, match="create-only"):
        lane.run_windows_cuda_strict_32k_smoke(
            output_dir=output,
            outer_attempt_lease_id=_OUTER_ATTEMPT_LEASE_ID,
        )
    assert len(runtime.generate_calls) == 1


@pytest.mark.parametrize(
    "failure_kind",
    ("capture", "budget", "budget-schema", "latest-mismatch"),
)
def test_completed_failure_is_published_once_without_retry(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    if failure_kind != "capture":
        _patch_small_diagnostic(monkeypatch)
    runtime = _install_runtime(monkeypatch, failure_kind=failure_kind)
    output = tmp_path / failure_kind

    result = lane.run_windows_cuda_strict_32k_smoke(
        output_dir=output,
        outer_attempt_lease_id=_OUTER_ATTEMPT_LEASE_ID,
    )

    assert len(runtime.generate_calls) == 1
    assert result.passed is False
    assert result.verdict == "failed_diagnostic_stop_no_retry"
    report = _read_json(output / lane._REPORT_FILE)
    assert report["passed"] is False
    assert report["verdict"] == "failed_diagnostic_stop_no_retry"
    budget = report["observation"]["context_budget"]
    if failure_kind == "capture":
        assert report["checks"]["capture_present"] is False
        assert report["checks"]["context_budget_exact"] is True
        assert budget["input_token_count"] == 32767
        assert budget["combined_token_count"] == 32768
    elif failure_kind == "budget":
        assert report["checks"]["context_budget_exact"] is False
        assert budget["schema_version"] == ("generation-context-budget-attestation.v1")
    elif failure_kind == "budget-schema":
        assert report["checks"]["context_budget_exact"] is False
        assert budget["schema_version"] == ("generation-context-budget-attestation.v2")
    else:
        assert report["checks"]["latest_capture_matches_sequence_exact"] is False
        assert report["observation"]["capture"]["latest_matches_sequence_exact"] is False
    assert (
        lane.validate_windows_cuda_strict_32k_smoke(
            output_dir=output,
            expected_outer_attempt_lease_id=_OUTER_ATTEMPT_LEASE_ID,
        )
        == result
    )


def test_cold_child_validator_does_not_import_substrate_or_torch(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _install_runtime(monkeypatch, failure_kind="capture")
    output = tmp_path / "cold-offline"
    original = lane.run_windows_cuda_strict_32k_smoke(
        output_dir=output,
        outer_attempt_lease_id=_OUTER_ATTEMPT_LEASE_ID,
    )
    assert len(runtime.generate_calls) == 1

    runtime_src = pathlib.Path(__file__).resolve().parents[1] / "src"
    child_source = """
import json
import pathlib
import sys

sys.path.insert(0, sys.argv[1])

class _ForbiddenImportBlocker:
    @staticmethod
    def find_spec(fullname, path=None, target=None):
        if (
            fullname == "torch"
            or fullname.startswith("torch.")
            or fullname == "volvence_zero.substrate"
            or fullname.startswith("volvence_zero.substrate.")
        ):
            raise RuntimeError("offline validator attempted forbidden import: " + fullname)
        return None

sys.meta_path.insert(0, _ForbiddenImportBlocker())
from volvence_zero.offline_evidence import windows_cuda_strict_32k_smoke as lane

result = lane.validate_windows_cuda_strict_32k_smoke(
    output_dir=pathlib.Path(sys.argv[2]),
    expected_outer_attempt_lease_id=sys.argv[3],
)
forbidden = sorted(
    name
    for name in sys.modules
    if name == "torch"
    or name.startswith("torch.")
    or name == "volvence_zero.substrate"
    or name.startswith("volvence_zero.substrate.")
)
if forbidden:
    raise RuntimeError("offline validator imported forbidden modules: " + repr(forbidden))
print(json.dumps({
    "artifact_id": result.artifact_id,
    "attempt_id": result.attempt_id,
    "outer_attempt_lease_id": result.outer_attempt_lease_id,
    "passed": result.passed,
}, sort_keys=True))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            child_source,
            str(runtime_src),
            str(output),
            _OUTER_ATTEMPT_LEASE_ID,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    child_result = json.loads(completed.stdout)
    assert child_result == {
        "artifact_id": original.artifact_id,
        "attempt_id": original.attempt_id,
        "outer_attempt_lease_id": _OUTER_ATTEMPT_LEASE_ID,
        "passed": False,
    }


@pytest.mark.parametrize(
    "tamper_kind",
    ("launch", "completion", "time-order", "lease"),
)
def test_validator_rejects_launch_completion_and_outer_lease_tamper(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_kind: str,
) -> None:
    output = tmp_path / tamper_kind
    _run_small_completed_artifact(output=output, monkeypatch=monkeypatch)

    expected_lease = _OUTER_ATTEMPT_LEASE_ID
    if tamper_kind == "launch":
        launch_path = output / lane._LAUNCH_FILE
        launch = _read_json(launch_path)
        launch["process_id"] += 1
        _write_canonical_json(launch_path, launch)
        expected_error = "launch attempt_id drift"
    elif tamper_kind == "completion":
        completion_path = output / lane._COMPLETION_FILE
        completion = _read_json(completion_path)
        completion["completion_id"] = "0" * 64
        _write_canonical_json(completion_path, completion)
        expected_error = "completion completion_id drift"
    elif tamper_kind == "time-order":
        completion_path = output / lane._COMPLETION_FILE
        completion = _read_json(completion_path)
        completion["completed_at_utc"] = "1970-01-01T00:00:00.000000Z"
        completion_core = dict(completion)
        del completion_core["completion_id"]
        completion["completion_id"] = hashlib.sha256(_canonical_bytes(completion_core)).hexdigest()
        _write_canonical_json(completion_path, completion)
        expected_error = "completion predates launch"
    else:
        expected_lease = _OTHER_OUTER_ATTEMPT_LEASE_ID
        expected_error = "outer attempt lease drift"

    with pytest.raises(ValueError, match=expected_error):
        lane.validate_windows_cuda_strict_32k_smoke(
            output_dir=output,
            expected_outer_attempt_lease_id=expected_lease,
        )


def test_context_budget_schema_drift_fails_after_consistent_local_reseal(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "budget-schema-reseal"
    _run_small_completed_artifact(output=output, monkeypatch=monkeypatch)
    report_path = output / lane._REPORT_FILE
    report = _read_json(report_path)
    report["observation"]["context_budget"]["schema_version"] = "generation-context-budget-attestation.v2"
    _write_canonical_json(report_path, report)
    _reseal_manifest_and_completion(output)

    with pytest.raises(ValueError, match="context_budget_exact value drift"):
        lane.validate_windows_cuda_strict_32k_smoke(
            output_dir=output,
            expected_outer_attempt_lease_id=_OUTER_ATTEMPT_LEASE_ID,
        )


@pytest.mark.parametrize("tamper_kind", ("payload", "file-set"))
def test_offline_validator_rejects_payload_tamper_and_file_set_drift(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_kind: str,
) -> None:
    output = tmp_path / tamper_kind
    _run_small_completed_artifact(output=output, monkeypatch=monkeypatch)

    if tamper_kind == "payload":
        report_path = output / lane._REPORT_FILE
        report_path.write_bytes(report_path.read_bytes() + b" ")
        expected_error = "payload SHA-256 drift"
    else:
        (output / "unexpected.json").write_text("{}\n", encoding="utf-8")
        expected_error = "file set drift"

    with pytest.raises(ValueError, match=expected_error):
        lane.validate_windows_cuda_strict_32k_smoke(
            output_dir=output,
            expected_outer_attempt_lease_id=_OUTER_ATTEMPT_LEASE_ID,
        )


def test_protocol_rejects_source_set_and_source_hash_drift(
    tmp_path: pathlib.Path,
) -> None:
    payload = _read_json(lane.strict_32k_smoke_protocol_path())
    missing_source = copy.deepcopy(payload)
    missing_source["source_sha256"].pop(next(reversed(missing_source["source_sha256"])))
    missing_path = tmp_path / "missing-source.json"
    _write_protocol(missing_path, missing_source)

    with pytest.raises(ValueError, match="source path order/set drift"):
        lane.load_windows_cuda_strict_32k_smoke_protocol(missing_path)

    drifted_source = copy.deepcopy(payload)
    first_source = next(iter(drifted_source["source_sha256"]))
    drifted_source["source_sha256"][first_source] = "0" * 64
    drifted_path = tmp_path / "source-drift.json"
    _write_protocol(drifted_path, drifted_source)

    with pytest.raises(ValueError, match="critical source SHA-256 drift"):
        lane.validate_windows_cuda_strict_32k_smoke(
            output_dir=tmp_path / "missing-artifact",
            expected_outer_attempt_lease_id=_OUTER_ATTEMPT_LEASE_ID,
            protocol_path=drifted_path,
        )
