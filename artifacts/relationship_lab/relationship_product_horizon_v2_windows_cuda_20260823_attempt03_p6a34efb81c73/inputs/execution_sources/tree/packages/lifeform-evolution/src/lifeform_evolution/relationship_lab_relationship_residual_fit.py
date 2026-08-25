"""Create-only relationship-domain residual fit runner and offline validator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import pathlib
import shutil
import tempfile
from typing import Any

from lifeform_domain_emogpt.lab.relationship_residual_fit_corpus import (
    RelationshipResidualFitProtocol,
    load_relationship_residual_fit_protocol,
)
from volvence_zero.agent.named_action_steering_artifact_training import (
    NamedActionSteeringCorpus,
    fit_named_action_steering_artifact_bundle,
    named_action_fit_lineage_sha256,
)
from volvence_zero.steering_contracts import SteeringArtifactBundle


RELATIONSHIP_RESIDUAL_FIT_RUN_SCHEMA_VERSION = (
    "relationship-residual-named-action-fit-run.v1"
)
_CORPUS_FILE = "relationship_residual_fit_corpus.json"
_BUNDLE_FILE = "steering_artifact_bundle.json"
_REPORT_FILE = "relationship_residual_fit_report.json"
_ATTESTATION_FILE = "execution_attestation.json"
_MANIFEST_FILE = "manifest.json"
_PAYLOAD_FILES = (
    _CORPUS_FILE,
    _BUNDLE_FILE,
    _REPORT_FILE,
    _ATTESTATION_FILE,
)


@dataclass(frozen=True)
class RelationshipResidualFitRunResult:
    artifact_id: str
    protocol_id: str
    corpus_id: str
    fit_lineage_sha256: str
    bundle_id: str
    prerequisite_passed: bool
    verdict: str
    output_dir: pathlib.Path


def run_relationship_residual_fit(
    *,
    corpus_path: pathlib.Path,
    output_dir: pathlib.Path,
    protocol: RelationshipResidualFitProtocol | None = None,
    progress: Any | None = None,
) -> RelationshipResidualFitRunResult:
    """Run the frozen CUDA recipe once and atomically publish its artifact."""

    active_protocol = protocol or load_relationship_residual_fit_protocol()
    corpus = _load_corpus(corpus_path, protocol=active_protocol)
    output = pathlib.Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(
            f"relationship residual fit output is create-only: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = pathlib.Path(
        tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.building.")
    )
    published = False
    try:
        runtime, scorer = _build_runtime_and_scorer(
            protocol=active_protocol,
            corpus=corpus,
        )
        attestation = runtime.execution_attestation
        if attestation is None:
            raise RuntimeError(
                "relationship residual strict runtime did not publish attestation"
            )
        attestation_payload = {
            **attestation.to_payload(),
            "attestation_id": attestation.attestation_id,
        }
        _validate_attestation(attestation_payload, protocol=active_protocol)
        if runtime.model_id != active_protocol.model_id:
            raise ValueError("relationship residual runtime model ID drift")
        fit_lineage_sha256 = named_action_fit_lineage_sha256(corpus)
        fit = fit_named_action_steering_artifact_bundle(
            corpus=corpus,
            runtime=runtime,
            scorer=scorer,
            model_weights_sha256=active_protocol.model_weights_sha256,
            source_preregistration_sha256=fit_lineage_sha256,
            injection_layer_index=active_protocol.injection_layer_index,
            residual_width=active_protocol.residual_width,
            steering_rank=active_protocol.steering_rank,
            executor_updates=active_protocol.conditional_executor_updates,
            executor_learning_rate=active_protocol.executor_learning_rate,
            reader_ridge_lambda=active_protocol.reader_ridge_lambda,
            batch_size=active_protocol.batch_size,
            seed=active_protocol.seed,
            control_norm_cap_ratio=active_protocol.control_norm_cap_ratio,
            progress=progress,
        )
        prerequisite_passed = fit.report.prerequisite_passed
        verdict = _verdict(prerequisite_passed)
        report = {
            "schema_version": RELATIONSHIP_RESIDUAL_FIT_RUN_SCHEMA_VERSION,
            "protocol_id": active_protocol.protocol_sha256,
            "corpus_id": corpus.corpus_id,
            "fit_lineage_sha256": fit_lineage_sha256,
            "bundle_id": fit.bundle.bundle_id,
            "owner_report": asdict(fit.report),
            "prerequisite_passed": prerequisite_passed,
            "verdict": verdict,
            "claim_flags": {
                "relationship_typed_action_residual_prerequisite": (
                    prerequisite_passed
                ),
                "raw_model_strict_json_generation_proven": False,
                "user_visible_relationship_reply_changed": False,
                "long_horizon_product_effect_proven": False,
                "production_active_authorized": False,
                "complete_steerable_proven": False,
                "four_able_complete": False,
            },
            "claim_boundary": active_protocol.claim_boundary,
        }
        payload_bytes = {
            _CORPUS_FILE: _canonical_bytes(corpus.to_payload()),
            _BUNDLE_FILE: (fit.bundle.to_json() + "\n").encode("utf-8"),
            _REPORT_FILE: _canonical_bytes(report),
            _ATTESTATION_FILE: _canonical_bytes(attestation_payload),
        }
        for name, payload in payload_bytes.items():
            _write_create(temporary / name, payload)
        manifest_core = {
            "schema_version": RELATIONSHIP_RESIDUAL_FIT_RUN_SCHEMA_VERSION,
            "protocol_id": active_protocol.protocol_sha256,
            "corpus_id": corpus.corpus_id,
            "fit_lineage_sha256": fit_lineage_sha256,
            "bundle_id": fit.bundle.bundle_id,
            "prerequisite_passed": prerequisite_passed,
            "verdict": verdict,
            "payload_files": {
                name: {
                    "byte_count": len(payload),
                    "sha256": _sha256_bytes(payload),
                }
                for name, payload in payload_bytes.items()
            },
            "formal_evidence_authorized": False,
            "product_wiring_changed": False,
        }
        artifact_id = _sha256_bytes(_canonical_bytes(manifest_core, newline=False))
        _write_create(
            temporary / _MANIFEST_FILE,
            _canonical_bytes({**manifest_core, "artifact_id": artifact_id}),
        )
        _validate_existing(
            output_dir=temporary,
            protocol=active_protocol,
        )
        if output.exists():
            raise FileExistsError(
                "relationship residual fit output appeared during publication"
            )
        temporary.rename(output)
        published = True
        return RelationshipResidualFitRunResult(
            artifact_id=artifact_id,
            protocol_id=active_protocol.protocol_sha256,
            corpus_id=corpus.corpus_id,
            fit_lineage_sha256=fit_lineage_sha256,
            bundle_id=fit.bundle.bundle_id,
            prerequisite_passed=prerequisite_passed,
            verdict=verdict,
            output_dir=output,
        )
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)


def validate_relationship_residual_fit(
    *,
    output_dir: pathlib.Path,
    protocol: RelationshipResidualFitProtocol | None = None,
) -> RelationshipResidualFitRunResult:
    """Validate existing bytes without constructing a model or importing torch."""

    active_protocol = protocol or load_relationship_residual_fit_protocol()
    return _validate_existing(
        output_dir=pathlib.Path(output_dir).resolve(),
        protocol=active_protocol,
    )


def _build_runtime_and_scorer(
    *,
    protocol: RelationshipResidualFitProtocol,
    corpus: NamedActionSteeringCorpus,
) -> tuple[Any, Any]:
    from volvence_zero.substrate import (
        WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1,
        SteeredActionOption,
        build_transformers_runtime_with_fallback,
    )

    profile = WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1
    runtime = build_transformers_runtime_with_fallback(
        model_id=protocol.model_id,
        model_source=None,
        device="cuda",
        layer_indices=(protocol.injection_layer_index,),
        activation_width=protocol.residual_width,
        max_length=32768,
        fail_on_truncation=True,
        local_files_only=True,
        fallback_mode="deny",
        runtime_mode="strict-local",
        model_dtype="bfloat16",
        expected_model_weights_sha256=protocol.model_weights_sha256,
        execution_profile=profile,
        verified_model_revision=protocol.model_revision,
        expected_execution_assets_sha256=protocol.execution_assets_sha256,
    )
    scorer = runtime.build_steered_action_scorer(
        action_options=tuple(
            SteeredActionOption(action_id=value, surface_text=value)
            for value in corpus.action_ids
        ),
        injection_layer_index=protocol.injection_layer_index,
        prompt_suffix="",
        max_length=32768,
        control_norm_ratio=protocol.control_norm_cap_ratio,
        probe_texts=tuple(row.action_text for row in corpus.train_rows[:16]),
        joint_training=False,
        prefix_cache=True,
    )
    return runtime, scorer


def _load_corpus(
    path: pathlib.Path,
    *,
    protocol: RelationshipResidualFitProtocol,
) -> NamedActionSteeringCorpus:
    raw = _load_json(pathlib.Path(path))
    corpus = NamedActionSteeringCorpus.from_payload(raw)
    if corpus.source_protocol_sha256 != protocol.protocol_sha256:
        raise ValueError("relationship residual corpus/protocol lineage mismatch")
    if corpus.action_ids != protocol.action_ids:
        raise ValueError("relationship residual corpus action surface drift")
    return corpus


def _validate_existing(
    *,
    output_dir: pathlib.Path,
    protocol: RelationshipResidualFitProtocol,
) -> RelationshipResidualFitRunResult:
    output = pathlib.Path(output_dir)
    if not output.is_dir():
        raise FileNotFoundError(f"relationship residual fit root is missing: {output}")
    if {path.name for path in output.iterdir()} != {
        *_PAYLOAD_FILES,
        _MANIFEST_FILE,
    }:
        raise ValueError("relationship residual fit root file set drift")
    payload_bytes = {name: (output / name).read_bytes() for name in _PAYLOAD_FILES}
    manifest = _load_json(output / _MANIFEST_FILE)
    report = _load_json(output / _REPORT_FILE)
    attestation = _load_json(output / _ATTESTATION_FILE)
    corpus = NamedActionSteeringCorpus.from_payload(
        _load_json(output / _CORPUS_FILE)
    )
    bundle = SteeringArtifactBundle.from_json(
        payload_bytes[_BUNDLE_FILE].decode("utf-8")
    )
    if corpus.source_protocol_sha256 != protocol.protocol_sha256:
        raise ValueError("relationship residual output corpus lineage drift")
    fit_lineage = named_action_fit_lineage_sha256(corpus)
    expected_manifest_keys = {
        "schema_version",
        "protocol_id",
        "corpus_id",
        "fit_lineage_sha256",
        "bundle_id",
        "prerequisite_passed",
        "verdict",
        "payload_files",
        "formal_evidence_authorized",
        "product_wiring_changed",
        "artifact_id",
    }
    if set(manifest) != expected_manifest_keys:
        raise ValueError("relationship residual manifest shape drift")
    for key, expected in (
        ("schema_version", RELATIONSHIP_RESIDUAL_FIT_RUN_SCHEMA_VERSION),
        ("protocol_id", protocol.protocol_sha256),
        ("corpus_id", corpus.corpus_id),
        ("fit_lineage_sha256", fit_lineage),
        ("bundle_id", bundle.bundle_id),
        ("formal_evidence_authorized", False),
        ("product_wiring_changed", False),
    ):
        if manifest[key] != expected:
            raise ValueError(f"relationship residual manifest {key} drift")
    ledger = manifest["payload_files"]
    if not isinstance(ledger, Mapping) or set(ledger) != set(_PAYLOAD_FILES):
        raise ValueError("relationship residual payload ledger drift")
    for name, payload in payload_bytes.items():
        entry = ledger[name]
        if not isinstance(entry, Mapping) or entry != {
            "byte_count": len(payload),
            "sha256": _sha256_bytes(payload),
        }:
            raise ValueError(f"relationship residual payload hash drift: {name}")
    manifest_core = dict(manifest)
    artifact_id = manifest_core.pop("artifact_id")
    _require_sha256(artifact_id, "artifact_id")
    if artifact_id != _sha256_bytes(
        _canonical_bytes(manifest_core, newline=False)
    ):
        raise ValueError("relationship residual artifact ID drift")

    _validate_bundle(
        bundle=bundle,
        protocol=protocol,
        corpus=corpus,
        fit_lineage=fit_lineage,
    )
    prerequisite_passed = _validate_report(
        report=report,
        protocol=protocol,
        corpus=corpus,
        bundle=bundle,
        fit_lineage=fit_lineage,
    )
    _validate_attestation(attestation, protocol=protocol)
    verdict = _verdict(prerequisite_passed)
    if (
        manifest["prerequisite_passed"] is not prerequisite_passed
        or manifest["verdict"] != verdict
    ):
        raise ValueError("relationship residual manifest verdict drift")
    return RelationshipResidualFitRunResult(
        artifact_id=artifact_id,
        protocol_id=protocol.protocol_sha256,
        corpus_id=corpus.corpus_id,
        fit_lineage_sha256=fit_lineage,
        bundle_id=bundle.bundle_id,
        prerequisite_passed=prerequisite_passed,
        verdict=verdict,
        output_dir=output.resolve(),
    )


def _validate_bundle(
    *,
    bundle: SteeringArtifactBundle,
    protocol: RelationshipResidualFitProtocol,
    corpus: NamedActionSteeringCorpus,
    fit_lineage: str,
) -> None:
    reader = bundle.reader
    if (
        reader.model_id != protocol.model_id
        or reader.model_weights_sha256 != protocol.model_weights_sha256
        or reader.source_preregistration_sha256 != fit_lineage
        or reader.layer_index != protocol.injection_layer_index
        or reader.residual_width != protocol.residual_width
        or reader.class_labels != corpus.class_labels
    ):
        raise ValueError("relationship residual reader lineage drift")
    sensor_off = bundle.sensor_off_executor
    if sensor_off is None:
        raise ValueError("relationship residual bundle lacks sensor-off executor")
    for candidate in (bundle.executor, sensor_off):
        if (
            candidate.model_id != reader.model_id
            or candidate.model_weights_sha256 != reader.model_weights_sha256
            or candidate.source_preregistration_sha256 != fit_lineage
            or candidate.reader_artifact_id != reader.artifact_id
            or candidate.layer_index != protocol.injection_layer_index
            or candidate.residual_width != protocol.residual_width
            or candidate.rank != protocol.steering_rank
            or candidate.class_labels != corpus.class_labels
            or candidate.control_norm_cap_ratio != protocol.control_norm_cap_ratio
            or candidate.free_bias_present
            or not candidate.zero_code_strict_noop
        ):
            raise ValueError("relationship residual executor lineage/safety drift")
    if len(set(sensor_off.condition_codes)) != 1:
        raise ValueError("relationship residual sensor-off executor is conditional")


def _validate_report(
    *,
    report: Mapping[str, object],
    protocol: RelationshipResidualFitProtocol,
    corpus: NamedActionSteeringCorpus,
    bundle: SteeringArtifactBundle,
    fit_lineage: str,
) -> bool:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "corpus_id",
        "fit_lineage_sha256",
        "bundle_id",
        "owner_report",
        "prerequisite_passed",
        "verdict",
        "claim_flags",
        "claim_boundary",
    }
    if set(report) != expected_keys:
        raise ValueError("relationship residual report shape drift")
    for key, expected in (
        ("schema_version", RELATIONSHIP_RESIDUAL_FIT_RUN_SCHEMA_VERSION),
        ("protocol_id", protocol.protocol_sha256),
        ("corpus_id", corpus.corpus_id),
        ("fit_lineage_sha256", fit_lineage),
        ("bundle_id", bundle.bundle_id),
        ("claim_boundary", protocol.claim_boundary),
    ):
        if report[key] != expected:
            raise ValueError(f"relationship residual report {key} drift")
    owner = report["owner_report"]
    if not isinstance(owner, Mapping):
        raise TypeError("relationship residual owner_report must be a mapping")
    if (
        owner["train_row_count"] != len(corpus.train_rows)
        or owner["heldout_row_count"] != len(corpus.heldout_rows)
        or owner["executor_updates"] != protocol.conditional_executor_updates
        or owner["executor_learning_rate"] != protocol.executor_learning_rate
        or owner["reader_ridge_lambda"] != protocol.reader_ridge_lambda
        or owner["steering_rank"] != protocol.steering_rank
        or owner["seed"] != protocol.seed
        or owner["control_norm_cap_ratio"] != protocol.control_norm_cap_ratio
        or owner["free_bias_present"] is not False
        or owner["zero_code_strict_noop"] is not True
        or owner["substrate_trainable_parameter_count"] != 0
        or owner["reader_executor_frozen_for_dialogue"] is not True
    ):
        raise ValueError("relationship residual owner report contract drift")
    noop = _finite_number(owner["heldout_noop_nll"], "heldout_noop_nll")
    steer = _finite_number(
        owner["heldout_online_steer_nll"],
        "heldout_online_steer_nll",
    )
    sensor_off = _finite_number(
        owner["heldout_sensor_off_nll"],
        "heldout_sensor_off_nll",
    )
    reader_accuracy = _finite_number(
        owner["reader_heldout_accuracy"],
        "reader_heldout_accuracy",
    )
    gain = _finite_number(
        owner["heldout_gain_vs_noop_nll"],
        "heldout_gain_vs_noop_nll",
    )
    conditional = _finite_number(
        owner["heldout_conditional_advantage_nll"],
        "heldout_conditional_advantage_nll",
    )
    if not math.isclose(gain, noop - steer, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("relationship residual gain derivation drift")
    if not math.isclose(
        conditional,
        sensor_off - steer,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("relationship residual conditional derivation drift")
    prerequisite = reader_accuracy >= 0.8 and gain > 0.0 and conditional > 0.0
    flags = report["claim_flags"]
    if not isinstance(flags, Mapping) or flags != {
        "relationship_typed_action_residual_prerequisite": prerequisite,
        "raw_model_strict_json_generation_proven": False,
        "user_visible_relationship_reply_changed": False,
        "long_horizon_product_effect_proven": False,
        "production_active_authorized": False,
        "complete_steerable_proven": False,
        "four_able_complete": False,
    }:
        raise ValueError("relationship residual claim flags drift")
    if (
        report["prerequisite_passed"] is not prerequisite
        or report["verdict"] != _verdict(prerequisite)
    ):
        raise ValueError("relationship residual report verdict drift")
    return prerequisite


def _validate_attestation(
    payload: Mapping[str, object],
    *,
    protocol: RelationshipResidualFitProtocol,
) -> None:
    canonical = dict(payload)
    attestation_id = canonical.pop("attestation_id", None)
    if not isinstance(attestation_id, str):
        raise ValueError("relationship residual attestation ID is missing")
    _require_sha256(attestation_id, "execution attestation id")
    if attestation_id != _sha256_bytes(_canonical_bytes(canonical, newline=False)):
        raise ValueError("relationship residual attestation ID drift")
    expected = {
        "schema_version": "transformers-execution-attestation.v1",
        "model_id": protocol.model_id,
        "model_revision": protocol.model_revision,
        "model_weights_sha256": protocol.model_weights_sha256,
        "execution_assets_sha256": protocol.execution_assets_sha256,
        "platform_system": "Windows",
        "local_files_only": True,
        "fallback_mode": "deny",
        "fail_on_truncation": True,
        "model_dtype": "bfloat16",
        "hidden_size": protocol.residual_width,
        "hook_layer_indices": [protocol.injection_layer_index],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"relationship residual attestation {key} drift")
    device = payload.get("device")
    if not isinstance(device, str) or not (
        device == "cuda"
        or (device.startswith("cuda:") and device.removeprefix("cuda:").isdigit())
    ):
        raise ValueError("relationship residual attestation is not CUDA")


def _verdict(prerequisite_passed: bool) -> str:
    return (
        "relationship_typed_action_residual_prerequisite_passed_development_only"
        if prerequisite_passed
        else "relationship_typed_action_residual_prerequisite_failed_stop_no_retuning"
    )


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _load_json(path: pathlib.Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicates)
    if not isinstance(raw, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return raw


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_bytes(value: object, *, newline: bool = True) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _write_create(path: pathlib.Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


__all__ = (
    "RELATIONSHIP_RESIDUAL_FIT_RUN_SCHEMA_VERSION",
    "RelationshipResidualFitRunResult",
    "run_relationship_residual_fit",
    "validate_relationship_residual_fit",
)
