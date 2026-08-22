from __future__ import annotations

import json
import pathlib
import shutil
from types import SimpleNamespace

import pytest

from volvence_zero.agent import relationship_p4_steering_artifact_fit as fit_lane
from volvence_zero.agent.steering_artifact_training import (
    SteeringArtifactFitReport,
    SteeringArtifactFitResult,
)
from volvence_zero.steering_contracts import (
    STEERING_ARTIFACT_BUNDLE_SCHEMA_VERSION,
    STEERING_EXECUTOR_ARTIFACT_SCHEMA_VERSION,
    STEERING_GATE_ARTIFACT_SCHEMA_VERSION,
    STEERING_READER_ARTIFACT_SCHEMA_VERSION,
    SteeringArtifactBundle,
    SteeringExecutorArtifact,
    SteeringGateArtifact,
    SteeringReaderArtifact,
)


def test_protocol_freezes_new_qwen15b_model_and_no_retuning_recipe() -> None:
    protocol = fit_lane.load_relationship_p4_steering_fit_protocol()

    assert protocol.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert protocol.verified_revision == (
        "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    )
    assert protocol.model_weights_sha256 == (
        "fb8c44c48b8359fdd306cdc5f473d7c04d88955013f0dd8549f266e248194da4"
    )
    assert protocol.execution_assets_sha256 == (
        "bbb5446f8d802b437c2fc7e2cefcdabb996bbd4bc657fe155ea015d30a841bb0"
    )
    assert protocol.injection_layer_index == 20
    assert protocol.residual_width == 1536
    assert protocol.steering_rank == 8
    assert protocol.conditional_executor_updates == 80
    assert protocol.sensor_off_executor_updates == 80
    assert protocol.executor_learning_rate == 0.01
    assert protocol.batch_size == 32
    assert protocol.reader_ridge_lambda == 10.0
    assert protocol.control_norm_cap_ratio == 0.25
    assert protocol.fit_seed == 0
    assert protocol.expected_train_row_count == 307
    assert protocol.expected_heldout_row_count == 165
    assert protocol.corpus_seed == 20260802
    assert protocol.train_route_count == 64
    assert protocol.heldout_route_count == 24
    assert protocol.train_lengths == (2, 3)
    assert protocol.heldout_lengths == (3, 4)
    assert protocol.source_hash_mode == "utf8_lf_canonical_v1"
    assert "without retuning" in protocol.claim_boundary
    assert all(
        not pathlib.PurePosixPath(path).is_absolute() and "\\" not in path
        for path, _ in protocol.source_sha256
    )
    payload = _read_json(fit_lane.relationship_p4_steering_fit_protocol_path())
    assert payload["instrumental_fit_execution"] == {
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
    assert payload["output_contract"]["derived_nll_consistency"] == {
        "relative_tolerance": 0.0,
        "absolute_tolerance": 1e-12,
        "gain_vs_noop_formula": (
            "heldout_noop_nll-heldout_online_steer_nll"
        ),
        "conditional_advantage_formula": (
            "heldout_sensor_off_nll-heldout_online_steer_nll"
        ),
    }


def test_strict_runtime_factory_uses_logical_id_and_all_frozen_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from volvence_zero import substrate

    protocol = fit_lane.load_relationship_p4_steering_fit_protocol()
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

    assert fit_lane._build_strict_runtime(protocol) is sentinel
    assert calls == [
        {
            "model_id": protocol.model_id,
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
            "expected_model_weights_sha256": (
                protocol.model_weights_sha256
            ),
            "execution_profile": (
                substrate.WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1
            ),
            "verified_model_revision": protocol.verified_revision,
            "expected_execution_assets_sha256": (
                protocol.execution_assets_sha256
            ),
        }
    ]


def test_thin_orchestrator_delegates_exact_recipe_to_runtime_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from volvence_zero.agent import eta_conflict_instrument
    from volvence_zero.agent import eta_proof_benchmark
    from volvence_zero.agent import eta_rate_distortion_evidence
    from volvence_zero.agent import steering_artifact_training

    protocol = fit_lane.load_relationship_p4_steering_fit_protocol()
    corpus = SimpleNamespace(environment=object())
    corpus_calls: list[dict[str, object]] = []
    fit_calls: list[dict[str, object]] = []
    scorer_calls: list[dict[str, object]] = []
    expected_result = object()

    def generate_corpus(**kwargs: object) -> object:
        corpus_calls.append(kwargs)
        return corpus

    def fit_owner(**kwargs: object) -> object:
        fit_calls.append(kwargs)
        return expected_result

    class Runtime:
        fail_on_truncation = True

        def build_steered_action_scorer(self, **kwargs: object) -> object:
            scorer_calls.append(kwargs)
            return "frozen-scorer"

    monkeypatch.setattr(
        eta_proof_benchmark,
        "generate_eta_proof_corpus",
        generate_corpus,
    )
    monkeypatch.setattr(
        eta_conflict_instrument,
        "build_conflict_junction_rows",
        lambda source, *, split: (
            SimpleNamespace(observation_text="probe"),
        ),
    )
    monkeypatch.setattr(
        eta_rate_distortion_evidence,
        "_action_options",
        lambda environment: ("action-a", "action-b"),
    )
    monkeypatch.setattr(
        steering_artifact_training,
        "fit_steering_artifact_bundle",
        fit_owner,
    )

    def progress(message: str) -> None:
        del message

    runtime = Runtime()
    assert (
        fit_lane._fit_with_runtime(
            protocol=protocol,
            runtime=runtime,
            progress=progress,
        )
        is expected_result
    )
    assert corpus_calls == [
        {
            "seed": 20260802,
            "objective_count": 8,
            "corridor_count": 2,
            "extra_edge_probability": 0.35,
            "train_route_count": 64,
            "heldout_route_count": 24,
            "train_lengths": (2, 3),
            "heldout_lengths": (3, 4),
        }
    ]
    assert scorer_calls == [
        {
            "action_options": ("action-a", "action-b"),
            "injection_layer_index": 20,
            "prompt_suffix": "",
            "max_length": 32768,
            "control_norm_ratio": 0.25,
            "probe_texts": ("probe\nNext move:",),
            "joint_training": False,
            "prefix_cache": True,
        }
    ]
    assert len(fit_calls) == 1
    assert fit_calls[0]["runtime"] is runtime
    fit_call_without_runtime = dict(fit_calls[0])
    del fit_call_without_runtime["runtime"]
    assert fit_call_without_runtime == {
        "corpus": corpus,
        "scorer": "frozen-scorer",
        "model_weights_sha256": protocol.model_weights_sha256,
        "source_preregistration_sha256": protocol.protocol_id,
        "injection_layer_index": 20,
        "residual_width": 1536,
        "steering_rank": 8,
        "executor_updates": 80,
        "executor_learning_rate": 0.01,
        "reader_ridge_lambda": 10.0,
        "batch_size": 32,
        "seed": 0,
        "control_norm_cap_ratio": 0.25,
        "progress": progress,
    }


@pytest.mark.parametrize("seed", (True, -1, 0.0))
def test_fit_owner_rejects_noncanonical_seed(seed: object) -> None:
    from volvence_zero.agent.steering_artifact_training import (
        fit_steering_artifact_bundle,
    )

    with pytest.raises(ValueError, match="seed"):
        fit_steering_artifact_bundle(
            corpus=object(),
            runtime=SimpleNamespace(model_id="unused"),
            scorer=object(),
            model_weights_sha256="a" * 64,
            source_preregistration_sha256="b" * 64,
            seed=seed,
        )


@pytest.mark.parametrize("ratio", (True, 0, 0.0, float("inf"), 2.1))
def test_fit_owner_rejects_noncanonical_control_norm_ratio(
    ratio: object,
) -> None:
    from volvence_zero.agent.steering_artifact_training import (
        fit_steering_artifact_bundle,
    )

    with pytest.raises(ValueError, match="control_norm_cap_ratio"):
        fit_steering_artifact_bundle(
            corpus=object(),
            runtime=SimpleNamespace(model_id="unused"),
            scorer=object(),
            model_weights_sha256="a" * 64,
            source_preregistration_sha256="b" * 64,
            control_norm_cap_ratio=ratio,
        )


def test_fit_owner_rejects_scorer_control_norm_ratio_drift() -> None:
    from volvence_zero.agent.steering_artifact_training import (
        fit_steering_artifact_bundle,
    )

    scorer = SimpleNamespace(
        trainable_parameters=lambda: (),
        probe_hidden_norm=4.0,
        control_norm_cap=2.0,
    )
    with pytest.raises(ValueError, match="scorer control-norm ratio"):
        fit_steering_artifact_bundle(
            corpus=object(),
            runtime=SimpleNamespace(model_id="unused"),
            scorer=scorer,
            model_weights_sha256="a" * 64,
            source_preregistration_sha256="b" * 64,
            control_norm_cap_ratio=0.25,
        )


@pytest.fixture
def fit_artifact(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[pathlib.Path, object]:
    protocol = fit_lane.load_relationship_p4_steering_fit_protocol()
    output = tmp_path / "fresh steering fit"
    fake_runtime = SimpleNamespace(
        execution_attestation=_FakeAttestation(protocol),
    )
    monkeypatch.setattr(
        fit_lane,
        "_build_strict_runtime",
        lambda actual: fake_runtime,
    )
    monkeypatch.setattr(
        fit_lane,
        "_fit_with_runtime",
        lambda **kwargs: _fit_result(protocol, passed=True),
    )
    result = fit_lane.run_relationship_p4_steering_artifact_fit(
        output_dir=output,
    )
    return output, result


@pytest.fixture
def quarantined_fit_artifact(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[pathlib.Path, object]:
    protocol = fit_lane.load_relationship_p4_steering_fit_protocol()
    output = tmp_path / "quarantined steering fit"
    fit_result = _fit_result(protocol, passed=True)
    sensor_off = fit_result.bundle.sensor_off_executor
    assert sensor_off is not None
    object.__setattr__(
        sensor_off,
        "condition_codes",
        ((0.3,) * protocol.steering_rank, (0.4,) * protocol.steering_rank),
    )
    monkeypatch.setattr(
        fit_lane,
        "_build_strict_runtime",
        lambda actual: SimpleNamespace(
            execution_attestation=_FakeAttestation(protocol),
        ),
    )
    monkeypatch.setattr(
        fit_lane,
        "_fit_with_runtime",
        lambda **kwargs: fit_result,
    )
    result = fit_lane.run_relationship_p4_steering_artifact_fit(
        output_dir=output,
    )
    return output, result


def test_run_publishes_four_create_only_content_addressed_files(
    fit_artifact: tuple[pathlib.Path, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, result = fit_artifact

    assert result.prerequisite_passed is True
    assert {item.name for item in output.iterdir()} == {
        "steering_artifact_bundle.json",
        "steering_artifact_fit_report.json",
        "execution_attestation.json",
        "manifest.json",
    }
    manifest = _read_json(output / "manifest.json")
    assert manifest["artifact_id"] == result.artifact_id
    assert manifest["formal_evidence_authorized"] is False
    assert manifest["production_active_authorized"] is False
    report = _read_json(output / "steering_artifact_fit_report.json")
    protocol = fit_lane.load_relationship_p4_steering_fit_protocol()
    assert manifest["source_sha256"] == protocol.source_sha256_payload()
    assert report["source_sha256"] == protocol.source_sha256_payload()
    assert manifest["source_hash_mode"] == "utf8_lf_canonical_v1"
    assert report["source_hash_mode"] == "utf8_lf_canonical_v1"
    assert "sensor_off_executor_quarantined" not in manifest
    assert "sensor_off_quarantine" not in report
    assert isinstance(report["numpy_version"], str)
    assert report["numpy_version"]
    assert (
        report["evidence_firewall"][
            "standalone_bundle_consumption_allowed"
        ]
        is False
    )
    assert report["evidence_firewall"]["complete_artifact_root_required"] is True
    assert b"\r\n" not in (output / "manifest.json").read_bytes()

    monkeypatch.setattr(
        fit_lane,
        "_build_strict_runtime",
        lambda protocol: pytest.fail("create-only rerun loaded the GPU"),
    )
    with pytest.raises(FileExistsError, match="create-only"):
        fit_lane.run_relationship_p4_steering_artifact_fit(output_dir=output)


def test_validate_existing_relocates_and_never_builds_gpu_runtime(
    fit_artifact: tuple[pathlib.Path, object],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, original = fit_artifact
    relocated = tmp_path / "relocated artifact with spaces"
    shutil.copytree(output, relocated)
    monkeypatch.setattr(
        fit_lane,
        "_build_strict_runtime",
        lambda protocol: pytest.fail("offline validator loaded the GPU"),
    )
    monkeypatch.setattr(
        fit_lane,
        "_fit_with_runtime",
        lambda **kwargs: pytest.fail("offline validator invoked fitting"),
    )

    validated = fit_lane.validate_relationship_p4_steering_artifact_fit(
        output_dir=relocated,
    )

    assert validated.artifact_id == original.artifact_id
    assert validated.output_dir == relocated.resolve()


def test_validate_existing_rejects_payload_hash_tampering(
    fit_artifact: tuple[pathlib.Path, object],
) -> None:
    output, _ = fit_artifact
    bundle_path = output / "steering_artifact_bundle.json"
    bundle_path.write_bytes(bundle_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="payload hash drift"):
        fit_lane.validate_relationship_p4_steering_artifact_fit(
            output_dir=output,
        )


def test_validate_existing_rejects_resealed_derived_metric_tampering(
    fit_artifact: tuple[pathlib.Path, object],
) -> None:
    output, _ = fit_artifact
    report_path = output / "steering_artifact_fit_report.json"
    report = _read_json(report_path)
    report["owner_report"]["reader_heldout_accuracy"] = 0.1
    report_path.write_bytes(fit_lane._canonical_bytes(report))
    _reseal_manifest(output)

    with pytest.raises(ValueError, match="derived checks"):
        fit_lane.validate_relationship_p4_steering_artifact_fit(
            output_dir=output,
        )


@pytest.mark.parametrize(
    "metric",
    (
        "heldout_gain_vs_noop_nll",
        "heldout_conditional_advantage_nll",
    ),
)
def test_validate_existing_recomputes_resealed_derived_nll_metrics(
    fit_artifact: tuple[pathlib.Path, object],
    metric: str,
) -> None:
    output, _ = fit_artifact
    report_path = output / "steering_artifact_fit_report.json"
    report = _read_json(report_path)
    report["owner_report"][metric] += 0.01
    report_path.write_bytes(fit_lane._canonical_bytes(report))
    _reseal_manifest(output)

    with pytest.raises(ValueError, match=f"derived NLL metric drift: {metric}"):
        fit_lane.validate_relationship_p4_steering_artifact_fit(
            output_dir=output,
        )


def test_sensor_off_row_mismatch_publishes_quarantined_failed_root(
    quarantined_fit_artifact: tuple[pathlib.Path, object],
) -> None:
    output, result = quarantined_fit_artifact

    assert result.prerequisite_passed is False
    assert result.verdict.endswith("failed_stop_no_retuning")
    bundle = _read_json(output / "steering_artifact_bundle.json")
    assert bundle["sensor_off_executor"] is None
    report = _read_json(output / "steering_artifact_fit_report.json")
    assert (
        report["checks"]["sensor_off_condition_code_rows_identical"]
        is False
    )
    quarantine = report["sensor_off_quarantine"]
    assert quarantine["quarantined"] is True
    assert quarantine["reason"] == "condition_code_rows_not_identical"
    assert quarantine["condition_codes"][0] != quarantine["condition_codes"][1]
    manifest = _read_json(output / "manifest.json")
    assert manifest["sensor_off_executor_quarantined"] is True
    validated = fit_lane.validate_relationship_p4_steering_artifact_fit(
        output_dir=output,
    )
    assert validated.prerequisite_passed is False


def test_validate_existing_rejects_resealed_quarantine_evidence_tampering(
    quarantined_fit_artifact: tuple[pathlib.Path, object],
) -> None:
    output, _ = quarantined_fit_artifact
    report_path = output / "steering_artifact_fit_report.json"
    report = _read_json(report_path)
    quarantine = report["sensor_off_quarantine"]
    quarantine["condition_codes"][1] = list(quarantine["condition_codes"][0])
    quarantine["condition_codes_sha256"] = fit_lane._sha256_bytes(
        fit_lane._canonical_bytes(
            quarantine["condition_codes"],
            newline=False,
        )
    )
    report_path.write_bytes(fit_lane._canonical_bytes(report))
    _reseal_manifest(output)

    with pytest.raises(ValueError, match="does not establish"):
        fit_lane.validate_relationship_p4_steering_artifact_fit(
            output_dir=output,
        )


def test_validate_existing_rejects_resealed_quarantine_flag_tampering(
    quarantined_fit_artifact: tuple[pathlib.Path, object],
) -> None:
    output, _ = quarantined_fit_artifact
    manifest_path = output / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["sensor_off_executor_quarantined"] = False
    del manifest["artifact_id"]
    manifest["artifact_id"] = fit_lane._sha256_bytes(
        fit_lane._canonical_bytes(manifest)
    )
    manifest_path.write_bytes(fit_lane._canonical_bytes(manifest))

    with pytest.raises(ValueError, match="quarantine flag drift"):
        fit_lane.validate_relationship_p4_steering_artifact_fit(
            output_dir=output,
        )


def test_validate_existing_rejects_resealed_attestation_revision(
    fit_artifact: tuple[pathlib.Path, object],
) -> None:
    output, _ = fit_artifact
    attestation_path = output / "execution_attestation.json"
    report_path = output / "steering_artifact_fit_report.json"
    attestation = _read_json(attestation_path)
    attestation["model_revision"] = "0" * 40
    unsigned = dict(attestation)
    del unsigned["attestation_id"]
    attestation["attestation_id"] = fit_lane._sha256_bytes(
        fit_lane._canonical_bytes(unsigned, newline=False)
    )
    attestation_path.write_bytes(fit_lane._canonical_bytes(attestation))
    report = _read_json(report_path)
    report["execution_attestation_id"] = attestation["attestation_id"]
    report_path.write_bytes(fit_lane._canonical_bytes(report))
    _reseal_manifest(
        output,
        execution_attestation_id=attestation["attestation_id"],
    )

    with pytest.raises(ValueError, match="model_revision"):
        fit_lane.validate_relationship_p4_steering_artifact_fit(
            output_dir=output,
        )


def test_failed_threshold_is_published_without_retuning(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = fit_lane.load_relationship_p4_steering_fit_protocol()
    output = tmp_path / "failed fit"
    monkeypatch.setattr(
        fit_lane,
        "_build_strict_runtime",
        lambda actual: SimpleNamespace(
            execution_attestation=_FakeAttestation(protocol),
        ),
    )
    monkeypatch.setattr(
        fit_lane,
        "_fit_with_runtime",
        lambda **kwargs: _fit_result(protocol, passed=False),
    )

    result = fit_lane.run_relationship_p4_steering_artifact_fit(
        output_dir=output,
    )

    assert result.prerequisite_passed is False
    assert result.verdict.endswith("failed_stop_no_retuning")
    report = _read_json(output / "steering_artifact_fit_report.json")
    assert report["failure_retuning_performed"] is False
    assert report["instrumental_fit_execution"]["use_cache"] is False
    assert (
        report["instrumental_fit_execution"][
            "exclusive_cudnn_sdpa_context"
        ]
        is True
    )
    assert report["owner_report"]["reader_heldout_accuracy"] == 0.79
    assert fit_lane.validate_relationship_p4_steering_artifact_fit(
        output_dir=output,
    ).prerequisite_passed is False


def test_protocol_rejects_bool_in_place_of_exact_context_integer(
    tmp_path: pathlib.Path,
) -> None:
    payload = _read_json(fit_lane.relationship_p4_steering_fit_protocol_path())
    payload["execution_profile"]["context_window_tokens"] = True
    path = tmp_path / "tampered-protocol.json"
    path.write_bytes(fit_lane._canonical_bytes(payload))

    with pytest.raises((TypeError, ValueError), match="context_window_tokens"):
        fit_lane.load_relationship_p4_steering_fit_protocol(path)


def test_validate_existing_rejects_critical_source_drift_before_gpu(
    fit_artifact: tuple[pathlib.Path, object],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _ = fit_artifact
    payload = _read_json(fit_lane.relationship_p4_steering_fit_protocol_path())
    source_path = next(iter(payload["source_sha256"]))
    payload["source_sha256"][source_path] = "f" * 64
    protocol_path = tmp_path / "source-drift-protocol.json"
    protocol_path.write_bytes(fit_lane._canonical_bytes(payload))
    monkeypatch.setattr(
        fit_lane,
        "_build_strict_runtime",
        lambda protocol: pytest.fail("source drift loaded the GPU"),
    )

    with pytest.raises(ValueError, match="critical source SHA-256 drift"):
        fit_lane.validate_relationship_p4_steering_artifact_fit(
            output_dir=output,
            protocol_path=protocol_path,
        )


def test_critical_source_hash_canonicalizes_line_endings(
    tmp_path: pathlib.Path,
) -> None:
    lf_path = tmp_path / "lf.py"
    crlf_path = tmp_path / "crlf.py"
    cr_path = tmp_path / "cr.py"
    lf_path.write_bytes("alpha = 'β'\nomega = 2\n".encode())
    crlf_path.write_bytes("alpha = 'β'\r\nomega = 2\r\n".encode())
    cr_path.write_bytes("alpha = 'β'\romega = 2\r".encode())

    assert fit_lane._source_text_sha256(lf_path) == (
        fit_lane._source_text_sha256(crlf_path)
    )
    assert fit_lane._source_text_sha256(lf_path) == (
        fit_lane._source_text_sha256(cr_path)
    )


def test_critical_source_hash_rejects_utf8_bom(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "bom.py"
    path.write_bytes(b"\xef\xbb\xbfvalue = 1\n")

    with pytest.raises(ValueError, match="UTF-8 BOM"):
        fit_lane._source_text_sha256(path)


class _FakeAttestation:
    def __init__(self, protocol: object) -> None:
        self._payload = {
            "schema_version": "transformers-execution-attestation.v1",
            "profile_id": protocol.profile_id,
            "preset_name": "windows-cuda-cudnn-sdpa-cached-strict.v1",
            "model_id": protocol.model_id,
            "model_revision": protocol.verified_revision,
            "model_weights_sha256": protocol.model_weights_sha256,
            "execution_assets_sha256": protocol.execution_assets_sha256,
            "runtime_origin": "hf-local",
            "platform_system": "Windows",
            "platform_release": "11",
            "device": "cuda",
            "device_name": "test-cuda",
            "python_version": "3.11",
            "torch_version": "test",
            "transformers_version": "test",
            "cuda_version": "12.6",
            "cudnn_version": 91002,
            "device_compute_capability": [8, 9],
            "attention_implementation": "sdpa",
            "sdpa_backend": "cudnn",
            "sdpa_backend_policy": "exclusive-cudnn",
            "sdpa_backend_exclusive": True,
            "generation_use_cache": True,
            "require_generation_chat_template": True,
            "generation_capture_strategy": "first-full-prompt-set-once",
            "capture_failure_mode": "raise",
            "context_window_tokens": 32768,
            "local_files_only": True,
            "fallback_mode": "deny",
            "fail_on_truncation": True,
            "model_dtype": "bfloat16",
            "hidden_size": 1536,
            "model_max_position_embeddings": 32768,
            "hook_layer_indices": [20],
        }

    def to_payload(self) -> dict[str, object]:
        return dict(self._payload)

    @property
    def attestation_id(self) -> str:
        return fit_lane._sha256_bytes(
            fit_lane._canonical_bytes(self._payload, newline=False)
        )


def _fit_result(protocol: object, *, passed: bool) -> SteeringArtifactFitResult:
    accuracy = 0.90 if passed else 0.79
    return SteeringArtifactFitResult(
        bundle=_bundle(protocol),
        report=SteeringArtifactFitReport(
            train_row_count=307,
            heldout_row_count=165,
            reader_heldout_accuracy=accuracy,
            heldout_noop_nll=1.0,
            heldout_online_steer_nll=0.8,
            heldout_sensor_off_nll=0.9,
            heldout_gain_vs_noop_nll=0.2,
            heldout_conditional_advantage_nll=0.1,
            reader_ridge_lambda=10.0,
            executor_updates=80,
            executor_learning_rate=0.01,
            steering_rank=8,
            seed=0,
            control_norm_cap_ratio=0.25,
            free_bias_present=False,
            zero_code_strict_noop=True,
            substrate_trainable_parameter_count=0,
            reader_executor_frozen_for_dialogue=True,
            description="frozen test owner report",
        ),
    )


def _bundle(protocol: object) -> SteeringArtifactBundle:
    labels = ("condition-a", "condition-b")
    width = 1536
    rank = 8
    prefix = f"{protocol.protocol_id[:12]}:{protocol.model_weights_sha256[:12]}"
    reader = SteeringReaderArtifact(
        schema_version=STEERING_READER_ARTIFACT_SCHEMA_VERSION,
        artifact_id=f"steering-reader:{prefix}",
        model_id=protocol.model_id,
        model_weights_sha256=protocol.model_weights_sha256,
        source_preregistration_sha256=protocol.protocol_id,
        layer_index=20,
        residual_width=width,
        class_labels=labels,
        weights=tuple((0.0, 0.0) for _ in range(width)),
        feature_mean=(0.0,) * width,
        feature_scale=(1.0,) * width,
        ridge_lambda=10.0,
        description="test reader",
    )
    common = {
        "schema_version": STEERING_EXECUTOR_ARTIFACT_SCHEMA_VERSION,
        "model_id": protocol.model_id,
        "model_weights_sha256": protocol.model_weights_sha256,
        "source_preregistration_sha256": protocol.protocol_id,
        "reader_artifact_id": reader.artifact_id,
        "layer_index": 20,
        "residual_width": width,
        "rank": rank,
        "class_labels": labels,
        "u_factors": tuple((0.0,) * rank for _ in range(width)),
        "v_factors": tuple((0.0,) * rank for _ in range(width)),
        "control_norm_cap_ratio": 0.25,
        "free_bias_present": False,
        "zero_code_strict_noop": True,
    }
    executor = SteeringExecutorArtifact(
        **common,
        artifact_id=f"steering-executor:{prefix}",
        condition_codes=((0.1,) * rank, (0.2,) * rank),
        description="test conditional executor",
    )
    sensor_off = SteeringExecutorArtifact(
        **common,
        artifact_id=f"steering-executor-sensor-off:{prefix}",
        condition_codes=((0.3,) * rank,) * len(labels),
        description="test unconditional executor",
    )
    gate = SteeringGateArtifact(
        schema_version=STEERING_GATE_ARTIFACT_SCHEMA_VERSION,
        artifact_id=f"steering-gate-shadow-collector:{prefix}",
        source_preregistration_sha256=protocol.protocol_id,
        feature_names=(
            "belief_margin",
            "fresh_margin",
            "belief_disagrees_fresh",
            "base_action_entropy",
            "prediction_error_magnitude",
            "staleness_proxy",
        ),
        weights=((0.0, 0.0),) * 6,
        bias=(-4.0, 4.0),
        policy_version=1,
        description="test shadow gate",
    )
    return SteeringArtifactBundle(
        schema_version=STEERING_ARTIFACT_BUNDLE_SCHEMA_VERSION,
        bundle_id=f"steering-dialogue-shadow:{prefix}",
        reader=reader,
        executor=executor,
        gate=gate,
        sensor_off_executor=sensor_off,
        description="fresh test bundle",
    )


def _read_json(path: pathlib.Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _reseal_manifest(
    output: pathlib.Path,
    *,
    execution_attestation_id: str | None = None,
) -> None:
    manifest_path = output / "manifest.json"
    manifest = _read_json(manifest_path)
    if execution_attestation_id is not None:
        manifest["execution_attestation_id"] = execution_attestation_id
    if "sensor_off_quarantine_sha256" in manifest:
        report = _read_json(output / "steering_artifact_fit_report.json")
        manifest["sensor_off_quarantine_sha256"] = fit_lane._sha256_bytes(
            fit_lane._canonical_bytes(
                report["sensor_off_quarantine"],
                newline=False,
            )
        )
    for record in manifest["files"]:
        path = output / record["path"]
        payload = path.read_bytes()
        record["sha256"] = fit_lane._sha256_bytes(payload)
        record["byte_count"] = len(payload)
    del manifest["artifact_id"]
    manifest["artifact_id"] = fit_lane._sha256_bytes(
        fit_lane._canonical_bytes(manifest)
    )
    manifest_path.write_bytes(fit_lane._canonical_bytes(manifest))
