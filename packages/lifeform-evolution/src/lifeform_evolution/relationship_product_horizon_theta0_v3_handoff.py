"""Artifact-specific typed handoff for the accepted theta0-v3 attempt03.

The historical protocol keeps its exact-HEAD validator unchanged.  This owner
first binds the already accepted detached validation receipt, then performs a
separate current-code compatibility replay of all 112 roots / 896 credits.  A
condensed authorization is constructed only from the resulting full typed
federation; the persisted theta JSON and compact transition IDs are never a
substitute for that graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
from typing import Mapping

import lifeform_domain_emogpt.lab.contracts as contracts_owner
import lifeform_domain_emogpt.lab.relationship_product_horizon_source_v4 as source_owner
import lifeform_domain_emogpt.lab.relationship_product_pulse as pulse_owner
import lifeform_domain_emogpt.relationship_action_contracts as action_contracts_owner
import lifeform_domain_emogpt.relationship_action_gate_v2 as gate_owner
import lifeform_domain_emogpt.relationship_condition_reader as reader_owner
from lifeform_domain_emogpt.lab.contracts import canonical_json, sha256_json
from lifeform_domain_emogpt.lab.relationship_product_pulse import (
    RelationshipProductPulseAuthorization,
    RelationshipProductV2CondensedTheta0FrozenPulseAuthorization,
)
from lifeform_evolution import (
    relationship_product_horizon_theta0_v3_bootstrap as bootstrap_owner,
)
import lifeform_evolution.relationship_lab_product_model_adapters as model_adapters_owner
import lifeform_evolution.relationship_product_horizon_source_admission as source_admission_owner
import lifeform_evolution.relationship_product_horizon_theta0_calibration as calibration_owner
import volvence_zero.credit as credit_owner
import volvence_zero.dialogue_external_outcome as dialogue_external_outcome_owner
import volvence_zero.dialogue_trace as dialogue_trace_owner
import volvence_zero.memory as memory_owner
import volvence_zero.owner_hydration as owner_hydration_owner
import volvence_zero.runtime as runtime_owner
import volvence_zero.semantic_state as semantic_state_owner
import volvence_zero.social as social_owner
import volvence_zero.social_cognition as social_cognition_owner
import volvence_zero.substrate as substrate_owner
import volvence_zero.temporal as temporal_owner
import volvence_zero.temporal_types as temporal_types_owner


THETA0_V3_HANDOFF_PROTOCOL_SCHEMA_VERSION = "relationship-product-horizon-theta0-v3-handoff-protocol.v1"
THETA0_V3_HANDOFF_MANIFEST_SCHEMA_VERSION = "relationship-product-horizon-theta0-v3-handoff-manifest.v1"

_PROTOCOL_FILENAME = "relationship_product_horizon_theta0_v3_handoff_v1.json"
_AUTHORIZATION_FILENAME = "theta0_authorization.json"
_MANIFEST_FILENAME = "manifest.json"
_EXPECTED_OUTPUT_FILES = frozenset({"protocol.json", _AUTHORIZATION_FILENAME, _MANIFEST_FILENAME})
_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[4]
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_REPORT_ARTIFACT_PREFIX = "relationship-product-horizon-theta0-v3-bootstrap-validation-pass-report-sha256:"

_CURRENT_EXECUTION_CLOSURE = (
    "packages",
    "scripts/run_relationship_product_horizon_theta0_v3_handoff.py",
)
_CURRENT_EXECUTION_CLOSURE_OBJECT_TYPES = {
    "packages": "tree",
    "scripts/run_relationship_product_horizon_theta0_v3_handoff.py": "blob",
}
_REPOSITORY_PACKAGE_TOP_LEVELS = frozenset(
    {
        "companion_ref_harness",
        "companion_standard",
        "lifeform_core",
        "lifeform_domain_emogpt",
        "lifeform_evolution",
        "volvence_zero",
    }
)

_EXPECTED_AUTHORIZATION_CONTRACT = {
    "authorization_kind": "RelationshipProductV2CondensedTheta0FrozenPulseAuthorization",
    "allowed_policy_artifact_version": 2,
    "source_transition_disposition": "apply",
    "evaluation_transition_disposition": None,
    "cold_checkpoint_update_count": 0,
    "cold_checkpoint_informative_update_count": 0,
    "cold_checkpoint_processed_credit_count": 0,
    "cold_checkpoint_weights_equal_learned_theta0": True,
    "full_federated_components_required_at_construction_and_payload_replay": True,
}

_EXPECTED_MATERIALIZATION_CONTRACT = {
    "expected_files": ["manifest.json", "protocol.json", _AUTHORIZATION_FILENAME],
    "create_only": True,
    "manifest_written_last": True,
    "current_implementation_commit_required_at_materialization": True,
    "current_execution_git_object_closure_reobserved_before_replay_and_manifest": True,
    "validate_existing_external_protocol_and_artifact_ids_required": True,
    "validate_existing_read_only": True,
    "model_output_count": 0,
    "cuda_execution_count": 0,
    "reader_fit_count": 0,
    "compatibility_replay_onboarding_write_count": 448,
    "compatibility_replay_reader_inference_count": 896,
    "compatibility_replay_precomputed_embedding_lookup_count": 896,
    "compatibility_replay_forecast_publication_count": 896,
    "compatibility_replay_source_settlement_count": 896,
    "compatibility_replay_prediction_error_derivation_count": 896,
    "compatibility_replay_credit_derivation_count": 896,
    "compatibility_replay_parent_apply_update_count": 896,
    "compatibility_replay_parent_apply_informative_update_count": 839,
    "compatibility_replay_parent_apply_cap_hit_count": 0,
    "compatibility_replay_parent_withhold_update_count": 0,
    "new_campaign_decision_count": 0,
    "new_scientific_outcome_observation_count": 0,
    "new_scientific_prediction_error_observation_count": 0,
    "new_scientific_credit_observation_count": 0,
    "authorization_evaluation_gate_update_count": 0,
}

_EXPECTED_CLAIMS = {
    "historical_acceptance_receipt_bound": True,
    "artifact_specific_current_compatibility_replay_passed": True,
    "current_full_typed_federation_rehydrated": True,
    "condensed_theta0_authorization_materialized": True,
    "theta_handoff_materialized": True,
    "historical_validate_existing_reexecuted": False,
    "general_backward_compatibility": False,
    "source_v5_bound": False,
    "reader_qualified": False,
    "condition_reader_qualified": False,
    "geometric_reachability_established": False,
    "credit_achievability_established": False,
    "treatment_reachability_admitted": False,
    "campaign_protocol_frozen": False,
    "campaign_execution_authorized": False,
    "appendable_effect": False,
    "readable_effect": False,
    "learnable_effect": False,
    "steerable_effect": False,
    "formal_evidence_authorized": False,
    "unseen_single_axis_evidence": False,
    "integrated_horizon_authorized": False,
    "four_able_complete": False,
    "human_sample_claimed": False,
    "production_active": False,
}


@dataclass(frozen=True)
class RelationshipProductHorizonTheta0V3HandoffProtocol:
    payload: Mapping[str, object]
    protocol_id: str
    raw_sha256: str
    raw_bytes: int

    @property
    def theta0_input(self) -> Mapping[str, object]:
        return _mapping(self.payload["theta0_input"], "theta0_input")

    @property
    def historical_acceptance(self) -> Mapping[str, object]:
        return _mapping(self.payload["historical_acceptance"], "historical_acceptance")

    @property
    def authorization_contract(self) -> Mapping[str, object]:
        return _mapping(self.payload["authorization_contract"], "authorization_contract")

    @property
    def materialization_contract(self) -> Mapping[str, object]:
        return _mapping(self.payload["materialization_contract"], "materialization_contract")


@dataclass(frozen=True)
class _CurrentImplementationReceipt:
    implementation_git_commit: str
    owned_git_object_ids: tuple[tuple[str, str, str], ...]

    @property
    def owned_git_object_ids_sha256(self) -> str:
        return sha256_json(
            [
                {
                    "path": path,
                    "git_object_type": object_type,
                    "git_object_id": object_id,
                }
                for path, object_type, object_id in self.owned_git_object_ids
            ]
        )


@dataclass(frozen=True)
class _HistoricalAcceptanceReceipt:
    report_artifact_id: str
    report_raw_sha256: str
    theta_manifest: Mapping[str, object]


@dataclass(frozen=True)
class RelationshipProductHorizonTheta0V3HandoffBundle:
    """Validated compact authorization; loading always replays the full graph."""

    manifest: Mapping[str, object]
    theta0_authorization: RelationshipProductV2CondensedTheta0FrozenPulseAuthorization


def relationship_product_horizon_theta0_v3_handoff_protocol_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent / "protocols" / _PROTOCOL_FILENAME


def load_relationship_product_horizon_theta0_v3_handoff_protocol(
    path: pathlib.Path | None = None,
) -> RelationshipProductHorizonTheta0V3HandoffProtocol:
    source = pathlib.Path(path or relationship_product_horizon_theta0_v3_handoff_protocol_path())
    raw = _read_regular(source)
    payload = _parse_json(raw, source=str(source))
    _exact_keys(
        payload,
        {
            "schema_version",
            "evidence_tier",
            "owner",
            "purpose",
            "theta0_input",
            "historical_acceptance",
            "compatibility_replay",
            "authorization_contract",
            "materialization_contract",
            "current_execution_closure",
            "claims_ceiling",
            "claim_boundary",
        },
        source="theta0 v3 handoff protocol",
    )
    if payload["schema_version"] != THETA0_V3_HANDOFF_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("theta0 v3 handoff protocol schema drifted")
    if payload["evidence_tier"] != "development":
        raise ValueError("theta0 v3 handoff evidence tier drifted")
    if payload["owner"] != ("lifeform_evolution.relationship_product_horizon_theta0_v3_handoff"):
        raise ValueError("theta0 v3 handoff owner drifted")
    if payload["purpose"] != (
        "bind_accepted_attempt03_and_replay_full_typed_federation_under_current_"
        "code_before_condensed_theta0_authorization"
    ):
        raise ValueError("theta0 v3 handoff purpose drifted")
    _require_exact_value(
        payload["current_execution_closure"],
        list(_CURRENT_EXECUTION_CLOSURE),
        source="theta0 v3 handoff current execution closure",
    )
    _require_exact_value(
        _mapping(payload["claims_ceiling"], "claims_ceiling"),
        _EXPECTED_CLAIMS,
        source="theta0 v3 handoff claim ceiling",
    )
    theta = _mapping(payload["theta0_input"], "theta0_input")
    _exact_keys(
        theta,
        {
            "protocol_id",
            "protocol_raw_sha256",
            "artifact_id",
            "implementation_git_commit",
            "implementation_owned_blob_ids_sha256",
            "required_status",
            "published_theta0_artifact_id",
            "federated_collection_id",
            "pulse_matched_transitions_id",
            "gate_matched_transitions_id",
            "apply_receipt_id",
            "withhold_receipt_id",
            "root_count",
            "credit_count",
            "apply_update_count",
            "apply_informative_update_count",
            "apply_cap_hit_count",
            "withhold_update_count",
            "file_inventory",
        },
        source="theta0_input",
    )
    if theta["protocol_id"] != bootstrap_owner.THETA0_V3_BOOTSTRAP_PROTOCOL_ID:
        raise ValueError("theta0 v3 handoff upstream protocol drifted")
    if theta["protocol_raw_sha256"] != (bootstrap_owner.THETA0_V3_BOOTSTRAP_PROTOCOL_RAW_SHA256):
        raise ValueError("theta0 v3 handoff upstream protocol bytes drifted")
    _digest(theta["artifact_id"], "theta0_input.artifact_id")
    _git_commit(theta["implementation_git_commit"])
    _digest(
        theta["implementation_owned_blob_ids_sha256"],
        "theta0_input.implementation_owned_blob_ids_sha256",
    )
    for key in (
        "required_status",
        "published_theta0_artifact_id",
        "federated_collection_id",
        "pulse_matched_transitions_id",
        "gate_matched_transitions_id",
        "apply_receipt_id",
        "withhold_receipt_id",
    ):
        _text(theta[key], f"theta0_input.{key}")
    for key, expected in (
        ("root_count", 112),
        ("credit_count", 896),
        ("apply_update_count", 896),
        ("apply_informative_update_count", 839),
        ("apply_cap_hit_count", 0),
        ("withhold_update_count", 0),
    ):
        _require_exact_value(theta[key], expected, source=f"theta0_input.{key}")
    inventory = _file_inventory_contract(theta["file_inventory"])
    if set(inventory) != {
        "federated_transition_bundle.json",
        "manifest.json",
        "parent_schedule.json",
        "protocol.json",
        "theta0_artifact.json",
        "theta0_v3_trace.jsonl",
    }:
        raise ValueError("theta0 v3 handoff upstream inventory drifted")
    acceptance = _mapping(payload["historical_acceptance"], "historical_acceptance")
    _exact_keys(
        acceptance,
        {
            "report_schema_version",
            "report_artifact_id",
            "report_raw_bytes",
            "report_raw_sha256",
            "attempt_id",
            "materialization_completed",
            "external_expected_protocol_and_artifact_ids_supplied",
            "validate_existing_passed",
            "filesystem_fingerprint_unchanged",
            "scientific_terminal",
        },
        source="historical_acceptance",
    )
    for key in ("report_schema_version", "report_artifact_id", "attempt_id"):
        _text(acceptance[key], f"historical_acceptance.{key}")
    _require_exact_value(
        acceptance["report_raw_bytes"],
        9124,
        source="historical_acceptance.report_raw_bytes",
    )
    _digest(acceptance["report_raw_sha256"], "historical_acceptance.report_raw_sha256")
    for key in (
        "materialization_completed",
        "external_expected_protocol_and_artifact_ids_supplied",
        "validate_existing_passed",
        "filesystem_fingerprint_unchanged",
    ):
        _require_exact_value(acceptance[key], True, source=f"historical_acceptance.{key}")
    _require_exact_value(
        acceptance["scientific_terminal"],
        False,
        source="historical_acceptance.scientific_terminal",
    )
    replay = _mapping(payload["compatibility_replay"], "compatibility_replay")
    required_replay = {
        "historical_validate_existing_reexecuted": False,
        "historical_acceptance_receipt_must_be_bound_before_replay": True,
        "artifact_specific_current_compatibility": True,
        "general_backward_compatibility": False,
        "current_full_typed_federation_rehydration_required": True,
        "all_six_persisted_files_byte_exact_required": True,
        "before_after_no_write_fingerprint_required": True,
        "theta_json_only_handoff_forbidden": True,
        "compact_projection_bypass_forbidden": True,
        "pickle_or_untyped_deserialization_forbidden": True,
    }
    _require_exact_value(
        replay,
        required_replay,
        source="theta0 v3 handoff compatibility contract",
    )
    authorization = _mapping(payload["authorization_contract"], "authorization_contract")
    _require_exact_value(
        authorization,
        _EXPECTED_AUTHORIZATION_CONTRACT,
        source="theta0 v3 handoff authorization contract",
    )
    materialization = _mapping(payload["materialization_contract"], "materialization_contract")
    _require_exact_value(
        materialization,
        _EXPECTED_MATERIALIZATION_CONTRACT,
        source="theta0 v3 handoff materialization contract",
    )
    _text(payload["claim_boundary"], "claim_boundary")
    return RelationshipProductHorizonTheta0V3HandoffProtocol(
        payload=payload,
        protocol_id=sha256_json(payload),
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        raw_bytes=len(raw),
    )


def materialize_relationship_product_horizon_theta0_v3_handoff(
    *,
    source_v4_admission_root: pathlib.Path,
    development_reader_root: pathlib.Path,
    theta0_root: pathlib.Path,
    historical_validation_report_path: pathlib.Path,
    output_dir: pathlib.Path,
    implementation_git_commit: str,
) -> Mapping[str, object]:
    """Create one compact authorization after the atomic full-graph gate."""

    protocol = load_relationship_product_horizon_theta0_v3_handoff_protocol()
    output = pathlib.Path(output_dir).resolve()
    inputs = (
        pathlib.Path(source_v4_admission_root).resolve(),
        pathlib.Path(development_reader_root).resolve(),
        pathlib.Path(theta0_root).resolve(),
        pathlib.Path(historical_validation_report_path).resolve(),
    )
    _require_disjoint_output(output=output, inputs=inputs)
    if output.exists():
        raise FileExistsError(f"theta0 v3 handoff output already exists: {output}")
    authorization, _full_transitions, acceptance, implementation = _replay_and_authorize(
        protocol=protocol,
        source_v4_admission_root=inputs[0],
        development_reader_root=inputs[1],
        theta0_root=inputs[2],
        historical_validation_report_path=inputs[3],
        implementation_git_commit=implementation_git_commit,
        require_current_head=True,
    )
    output.mkdir(parents=True, exist_ok=False)
    _write_create_only(
        output / "protocol.json",
        _read_regular(relationship_product_horizon_theta0_v3_handoff_protocol_path()),
    )
    _write_create_only(
        output / _AUTHORIZATION_FILENAME,
        _artifact_bytes(authorization.to_payload()),
    )
    implementation_after = _verify_current_execution_closure(
        protocol=protocol,
        implementation_git_commit=implementation_git_commit,
        require_current_head=True,
    )
    if implementation_after != implementation:
        raise RuntimeError("theta0 v3 handoff execution closure changed before manifest")
    manifest = _build_manifest(
        root=output,
        protocol=protocol,
        implementation=implementation,
        acceptance=acceptance,
        authorization=authorization,
    )
    manifest_raw = _artifact_bytes(manifest)
    _write_create_only(output / _MANIFEST_FILENAME, manifest_raw)
    if set(_filesystem_fingerprint(output)) != _EXPECTED_OUTPUT_FILES:
        raise RuntimeError("theta0 v3 handoff post-materialization inventory drifted")
    if _read_regular(output / _MANIFEST_FILENAME) != manifest_raw:
        raise RuntimeError("theta0 v3 handoff manifest reopen drifted")
    return manifest


def validate_relationship_product_horizon_theta0_v3_handoff(
    *,
    source_v4_admission_root: pathlib.Path,
    development_reader_root: pathlib.Path,
    theta0_root: pathlib.Path,
    historical_validation_report_path: pathlib.Path,
    output_dir: pathlib.Path,
    expected_protocol_id: str,
    expected_artifact_id: str,
) -> Mapping[str, object]:
    return _validate_persisted_handoff(
        protocol=load_relationship_product_horizon_theta0_v3_handoff_protocol(),
        source_v4_admission_root=pathlib.Path(source_v4_admission_root).resolve(),
        development_reader_root=pathlib.Path(development_reader_root).resolve(),
        theta0_root=pathlib.Path(theta0_root).resolve(),
        historical_validation_report_path=pathlib.Path(historical_validation_report_path).resolve(),
        output_dir=pathlib.Path(output_dir).resolve(),
        expected_protocol_id=expected_protocol_id,
        expected_artifact_id=expected_artifact_id,
    ).manifest


def load_relationship_product_horizon_theta0_v3_handoff(
    *,
    source_v4_admission_root: pathlib.Path,
    development_reader_root: pathlib.Path,
    theta0_root: pathlib.Path,
    historical_validation_report_path: pathlib.Path,
    output_dir: pathlib.Path,
    expected_protocol_id: str,
    expected_artifact_id: str,
) -> RelationshipProductHorizonTheta0V3HandoffBundle:
    """Load only through receipt + closure + full-federation replay."""

    return _validate_persisted_handoff(
        protocol=load_relationship_product_horizon_theta0_v3_handoff_protocol(),
        source_v4_admission_root=pathlib.Path(source_v4_admission_root).resolve(),
        development_reader_root=pathlib.Path(development_reader_root).resolve(),
        theta0_root=pathlib.Path(theta0_root).resolve(),
        historical_validation_report_path=pathlib.Path(historical_validation_report_path).resolve(),
        output_dir=pathlib.Path(output_dir).resolve(),
        expected_protocol_id=expected_protocol_id,
        expected_artifact_id=expected_artifact_id,
    )


def _validate_persisted_handoff(
    *,
    protocol: RelationshipProductHorizonTheta0V3HandoffProtocol,
    source_v4_admission_root: pathlib.Path,
    development_reader_root: pathlib.Path,
    theta0_root: pathlib.Path,
    historical_validation_report_path: pathlib.Path,
    output_dir: pathlib.Path,
    expected_protocol_id: str,
    expected_artifact_id: str,
) -> RelationshipProductHorizonTheta0V3HandoffBundle:
    external_protocol = _digest(expected_protocol_id, "expected_protocol_id")
    external_artifact = _digest(expected_artifact_id, "expected_artifact_id")
    if protocol.protocol_id != external_protocol:
        raise ValueError("theta0 v3 handoff external protocol identity drifted")
    root = pathlib.Path(output_dir).resolve()
    before = _filesystem_fingerprint(root)
    if set(before) != _EXPECTED_OUTPUT_FILES:
        raise ValueError("theta0 v3 handoff output inventory drifted")
    protocol_raw = _read_regular(root / "protocol.json")
    if protocol_raw != _read_regular(relationship_product_horizon_theta0_v3_handoff_protocol_path()):
        raise ValueError("theta0 v3 handoff persisted protocol bytes drifted")
    manifest_raw = _read_regular(root / _MANIFEST_FILENAME)
    manifest = _parse_json(manifest_raw, source="theta0 v3 handoff manifest")
    if manifest_raw != _artifact_bytes(manifest):
        raise ValueError("theta0 v3 handoff manifest is not canonical")
    if (
        manifest.get("protocol_id") != external_protocol
        or manifest.get("artifact_id") != external_artifact
        or manifest.get("artifact_id")
        != sha256_json({key: value for key, value in manifest.items() if key != "artifact_id"})
    ):
        raise ValueError("theta0 v3 handoff manifest identity drifted")
    implementation_commit = _git_commit(manifest.get("implementation_git_commit"))
    authorization, full_transitions, acceptance, implementation = _replay_and_authorize(
        protocol=protocol,
        source_v4_admission_root=source_v4_admission_root,
        development_reader_root=development_reader_root,
        theta0_root=theta0_root,
        historical_validation_report_path=historical_validation_report_path,
        implementation_git_commit=implementation_commit,
        require_current_head=False,
    )
    authorization_raw = _read_regular(root / _AUTHORIZATION_FILENAME)
    authorization_payload = _parse_json(authorization_raw, source="theta0 v3 handoff authorization")
    if authorization_raw != _artifact_bytes(authorization_payload):
        raise ValueError("theta0 v3 handoff authorization is not canonical")
    replayed_authorization = RelationshipProductV2CondensedTheta0FrozenPulseAuthorization.from_payload(
        authorization_payload,
        pulse_authorization=authorization.pulse_authorization,
        learned_theta0_artifact=authorization.learned_theta0_artifact,
        source_federated_matched_transitions=full_transitions,
    )
    if replayed_authorization != authorization:
        raise ValueError("theta0 v3 handoff authorization typed replay drifted")
    expected_manifest = _build_manifest(
        root=root,
        protocol=protocol,
        implementation=implementation,
        acceptance=acceptance,
        authorization=authorization,
    )
    if manifest != expected_manifest or manifest_raw != _artifact_bytes(expected_manifest):
        raise ValueError("theta0 v3 handoff manifest content drifted")
    after = _filesystem_fingerprint(root)
    if after != before:
        raise RuntimeError("theta0 v3 handoff validate-existing modified output")
    return RelationshipProductHorizonTheta0V3HandoffBundle(
        manifest=manifest,
        theta0_authorization=authorization,
    )


def _replay_and_authorize(
    *,
    protocol: RelationshipProductHorizonTheta0V3HandoffProtocol,
    source_v4_admission_root: pathlib.Path,
    development_reader_root: pathlib.Path,
    theta0_root: pathlib.Path,
    historical_validation_report_path: pathlib.Path,
    implementation_git_commit: str,
    require_current_head: bool,
) -> tuple[
    RelationshipProductV2CondensedTheta0FrozenPulseAuthorization,
    pulse_owner.RelationshipProductV2FederatedMatchedGateTransitions,
    _HistoricalAcceptanceReceipt,
    _CurrentImplementationReceipt,
]:
    acceptance = _validate_historical_acceptance(
        protocol=protocol,
        theta0_root=theta0_root,
        historical_validation_report_path=historical_validation_report_path,
    )
    implementation = _verify_current_execution_closure(
        protocol=protocol,
        implementation_git_commit=implementation_git_commit,
        require_current_head=require_current_head,
    )
    bundle = bootstrap_owner._replay_relationship_product_horizon_theta0_v3_bundle_for_cross_commit_handoff(
        source_v4_admission_root=source_v4_admission_root,
        reader_root=development_reader_root,
        output_dir=theta0_root,
        expected_protocol_id=_text(protocol.theta0_input["protocol_id"], "protocol_id"),
        expected_artifact_id=_digest(protocol.theta0_input["artifact_id"], "artifact_id"),
    )
    theta = protocol.theta0_input
    applied = bundle.matched_transitions.applied
    withheld = bundle.matched_transitions.withheld
    observed = (
        bundle.manifest["artifact_id"],
        bundle.theta0_artifact.artifact_id,
        bundle.federated_collection.collection_id,
        bundle.matched_transitions.transitions_id,
        bundle.matched_transitions.gate_matched_transitions.transitions_id,
        applied.gate_receipt.receipt_id,
        withheld.gate_receipt.receipt_id,
        len(applied.batch.credits),
        applied.gate_receipt.update_count_delta,
        applied.gate_receipt.informative_update_count_delta,
        applied.gate_receipt.cap_hit_count,
        withheld.gate_receipt.update_count_delta,
    )
    expected = (
        theta["artifact_id"],
        theta["published_theta0_artifact_id"],
        theta["federated_collection_id"],
        theta["pulse_matched_transitions_id"],
        theta["gate_matched_transitions_id"],
        theta["apply_receipt_id"],
        theta["withhold_receipt_id"],
        theta["credit_count"],
        theta["apply_update_count"],
        theta["apply_informative_update_count"],
        theta["apply_cap_hit_count"],
        theta["withhold_update_count"],
    )
    if observed != expected:
        raise ValueError("theta0 v3 handoff full federation terminal drifted")
    pulse_authorization = RelationshipProductPulseAuthorization(
        authorization_id=(f"relationship-product-horizon-theta0-v3-handoff-pulse-sha256:{protocol.protocol_id}"),
        allowed_policy_artifact_id=bundle.theta0_artifact.artifact_id,
        allowed_policy_artifact_version=2,
    )
    authorization = RelationshipProductV2CondensedTheta0FrozenPulseAuthorization(
        pulse_authorization=pulse_authorization,
        learned_theta0_artifact=bundle.theta0_artifact,
        source_federated_matched_transitions=bundle.matched_transitions,
    )
    cold = authorization.frozen_policy.checkpoint
    if (
        cold.update_count != 0
        or cold.informative_update_count != 0
        or cold.processed_credit_ids != ()
        or cold.weights != bundle.theta0_artifact.weights
        or authorization.to_payload()["evaluation_transition_disposition"] is not None
    ):
        raise ValueError("theta0 v3 handoff authorization is not an exact cold policy")
    replayed = RelationshipProductV2CondensedTheta0FrozenPulseAuthorization.from_payload(
        authorization.to_payload(),
        pulse_authorization=pulse_authorization,
        learned_theta0_artifact=bundle.theta0_artifact,
        source_federated_matched_transitions=bundle.matched_transitions,
    )
    if replayed != authorization:
        raise ValueError("theta0 v3 handoff in-memory authorization replay drifted")
    implementation_after = _verify_current_execution_closure(
        protocol=protocol,
        implementation_git_commit=implementation_git_commit,
        require_current_head=require_current_head,
    )
    if implementation_after != implementation:
        raise RuntimeError("theta0 v3 handoff current implementation changed during replay")
    return authorization, bundle.matched_transitions, acceptance, implementation


def _validate_historical_acceptance(
    *,
    protocol: RelationshipProductHorizonTheta0V3HandoffProtocol,
    theta0_root: pathlib.Path,
    historical_validation_report_path: pathlib.Path,
) -> _HistoricalAcceptanceReceipt:
    theta = protocol.theta0_input
    root = pathlib.Path(theta0_root).resolve()
    observed_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    expected_inventory = _file_inventory_contract(theta["file_inventory"])
    if observed_files != set(expected_inventory):
        raise ValueError("theta0 v3 handoff historical root inventory drifted")
    for relative, pin in expected_inventory.items():
        raw = _read_regular(root / relative)
        if len(raw) != pin["raw_bytes"] or hashlib.sha256(raw).hexdigest() != pin["raw_sha256"]:
            raise ValueError(f"theta0 v3 handoff historical file drifted: {relative}")
    theta_manifest = _parse_json(_read_regular(root / "manifest.json"), source="theta0 v3 historical manifest")
    for manifest_key, protocol_key in (
        ("protocol_id", "protocol_id"),
        ("protocol_raw_sha256", "protocol_raw_sha256"),
        ("artifact_id", "artifact_id"),
        ("implementation_git_commit", "implementation_git_commit"),
        (
            "implementation_owned_blob_ids_sha256",
            "implementation_owned_blob_ids_sha256",
        ),
        ("status", "required_status"),
        ("published_theta0_artifact_id", "published_theta0_artifact_id"),
        ("federated_collection_id", "federated_collection_id"),
        ("pulse_matched_transitions_id", "pulse_matched_transitions_id"),
        ("gate_matched_transitions_id", "gate_matched_transitions_id"),
        ("apply_receipt_id", "apply_receipt_id"),
        ("withhold_receipt_id", "withhold_receipt_id"),
        ("completed_root_count", "root_count"),
        ("credit_count", "credit_count"),
        ("apply_update_count_delta", "apply_update_count"),
        (
            "apply_informative_update_count_delta",
            "apply_informative_update_count",
        ),
        ("apply_cap_hit_count", "apply_cap_hit_count"),
        ("withhold_update_count_delta", "withhold_update_count"),
    ):
        if theta_manifest.get(manifest_key) != theta[protocol_key]:
            raise ValueError(f"theta0 v3 historical manifest {manifest_key} drifted")
    acceptance_pin = protocol.historical_acceptance
    report_raw = _read_regular(pathlib.Path(historical_validation_report_path))
    if (
        len(report_raw) != acceptance_pin["report_raw_bytes"]
        or hashlib.sha256(report_raw).hexdigest() != acceptance_pin["report_raw_sha256"]
    ):
        raise ValueError("theta0 v3 historical acceptance report bytes drifted")
    report = _parse_json(report_raw, source="theta0 v3 historical acceptance report")
    report_for_id = dict(report)
    report_for_id["report_artifact_id"] = None
    expected_report_id = f"{_REPORT_ARTIFACT_PREFIX}{sha256_json(report_for_id)}"
    if report.get("report_artifact_id") != expected_report_id:
        raise ValueError("theta0 v3 historical acceptance self identity drifted")
    for report_key, expected_value in (
        ("report_schema_version", acceptance_pin["report_schema_version"]),
        ("report_artifact_id", acceptance_pin["report_artifact_id"]),
        ("attempt_id", acceptance_pin["attempt_id"]),
        ("protocol_id", theta["protocol_id"]),
        ("protocol_raw_sha256", theta["protocol_raw_sha256"]),
        ("implementation_git_commit", theta["implementation_git_commit"]),
        (
            "implementation_owned_blob_ids_sha256",
            theta["implementation_owned_blob_ids_sha256"],
        ),
    ):
        if report.get(report_key) != expected_value:
            raise ValueError(f"theta0 v3 historical acceptance {report_key} drifted")
    acceptance = _mapping(report["acceptance"], "acceptance")
    if (
        acceptance.get("accepted_artifact_id") != theta["artifact_id"]
        or acceptance.get("validate_existing_passed") is not True
        or acceptance.get("development_mechanism_terminal_accepted") is not True
        or acceptance.get("effect_scientific_terminal") is not False
    ):
        raise ValueError("theta0 v3 historical acceptance verdict drifted")
    validation_events = report.get("validation_events")
    if not isinstance(validation_events, list) or len(validation_events) != 2:
        raise ValueError("theta0 v3 historical validation event count drifted")
    exact_event = _mapping(validation_events[1], "validation_events[1]")
    if (
        exact_event.get("phase") != "detached_exact_implementation_commit_full_replay"
        or exact_event.get("implementation_head_exact") is not True
        or exact_event.get("implementation_clean_scope_exact") is not True
        or exact_event.get("full_artifact_replay_completed") is not True
        or exact_event.get("passed") is not True
        or exact_event.get("file_fingerprints_unchanged_during_validation") is not True
    ):
        raise ValueError("theta0 v3 historical exact-commit validation event drifted")
    report_inventory = _file_inventory_contract(report.get("root_inventory"))
    if report_inventory != expected_inventory:
        raise ValueError("theta0 v3 historical acceptance root inventory drifted")
    claim_ceiling = _mapping(report["claim_ceiling"], "claim_ceiling")
    if any(value not in (False, 0) for value in claim_ceiling.values()):
        raise ValueError("theta0 v3 historical acceptance claim ceiling inflated")
    return _HistoricalAcceptanceReceipt(
        report_artifact_id=_text(report["report_artifact_id"], "report_artifact_id"),
        report_raw_sha256=hashlib.sha256(report_raw).hexdigest(),
        theta_manifest=theta_manifest,
    )


def _verify_current_execution_closure(
    *,
    protocol: RelationshipProductHorizonTheta0V3HandoffProtocol,
    implementation_git_commit: str,
    require_current_head: bool,
) -> _CurrentImplementationReceipt:
    commit = _git_commit(implementation_git_commit)
    repository = pathlib.Path(_run_git("rev-parse", "--show-toplevel").stdout.strip()).resolve(strict=True)
    if os.path.normcase(str(repository)) != os.path.normcase(str(_REPOSITORY_ROOT.resolve(strict=True))):
        raise ValueError("theta0 v3 handoff repository identity drifted")
    resolved = _run_git("rev-parse", "--verify", f"{commit}^{{commit}}").stdout.strip()
    if resolved != commit:
        raise ValueError("theta0 v3 handoff implementation commit does not exist")
    if require_current_head and _run_git("rev-parse", "HEAD").stdout.strip() != commit:
        raise ValueError("theta0 v3 handoff implementation commit does not match HEAD")
    closure = tuple(protocol.payload["current_execution_closure"])
    if closure != _CURRENT_EXECUTION_CLOSURE:
        raise ValueError("theta0 v3 handoff execution closure drifted")
    _verify_module_origins()
    clean = _run_git("diff", "--quiet", commit, "--", *closure, check=False)
    if clean.returncode != 0:
        if clean.returncode == 1:
            raise ValueError("theta0 v3 handoff execution closure differs from commit")
        raise RuntimeError("theta0 v3 handoff execution closure diff failed")
    untracked = _run_git("ls-files", "--others", "--exclude-standard", "--", *closure).stdout
    if untracked:
        raise ValueError("theta0 v3 handoff execution closure has untracked files")
    owned_git_object_ids: list[tuple[str, str, str]] = []
    for path in closure:
        expected_type = _CURRENT_EXECUTION_CLOSURE_OBJECT_TYPES[path]
        observed_type = _run_git("cat-file", "-t", f"{commit}:{path}").stdout.strip()
        if observed_type != expected_type:
            raise ValueError(f"theta0 v3 handoff execution object type drifted: {path}")
        object_id = _run_git("rev-parse", f"{commit}:{path}").stdout.strip()
        if not object_id:
            raise ValueError("theta0 v3 handoff execution git object identity is empty")
        owned_git_object_ids.append((path, observed_type, object_id))
    return _CurrentImplementationReceipt(
        implementation_git_commit=commit,
        owned_git_object_ids=tuple(owned_git_object_ids),
    )


def _verify_module_origins() -> None:
    expected = {
        contracts_owner: "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab/contracts.py",
        source_owner: (
            "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab/relationship_product_horizon_source_v4.py"
        ),
        pulse_owner: ("packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab/relationship_product_pulse.py"),
        action_contracts_owner: (
            "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/relationship_action_contracts.py"
        ),
        gate_owner: ("packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/relationship_action_gate_v2.py"),
        reader_owner: ("packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/relationship_condition_reader.py"),
        bootstrap_owner: (
            "packages/lifeform-evolution/src/lifeform_evolution/relationship_product_horizon_theta0_v3_bootstrap.py"
        ),
        calibration_owner: (
            "packages/lifeform-evolution/src/lifeform_evolution/relationship_product_horizon_theta0_calibration.py"
        ),
        source_admission_owner: (
            "packages/lifeform-evolution/src/lifeform_evolution/relationship_product_horizon_source_admission.py"
        ),
        model_adapters_owner: (
            "packages/lifeform-evolution/src/lifeform_evolution/relationship_lab_product_model_adapters.py"
        ),
        credit_owner: "packages/vz-cognition/src/volvence_zero/credit/__init__.py",
        dialogue_external_outcome_owner: ("packages/vz-runtime/src/volvence_zero/dialogue_external_outcome.py"),
        dialogue_trace_owner: "packages/vz-contracts/src/volvence_zero/dialogue_trace.py",
        memory_owner: "packages/vz-memory/src/volvence_zero/memory/__init__.py",
        owner_hydration_owner: "packages/vz-contracts/src/volvence_zero/owner_hydration.py",
        runtime_owner: "packages/vz-contracts/src/volvence_zero/runtime/__init__.py",
        semantic_state_owner: "packages/vz-cognition/src/volvence_zero/semantic_state/__init__.py",
        social_owner: "packages/vz-cognition/src/volvence_zero/social/__init__.py",
        social_cognition_owner: "packages/vz-contracts/src/volvence_zero/social_cognition.py",
        substrate_owner: "packages/vz-substrate/src/volvence_zero/substrate/__init__.py",
        temporal_owner: "packages/vz-temporal/src/volvence_zero/temporal/__init__.py",
        temporal_types_owner: "packages/vz-contracts/src/volvence_zero/temporal_types.py",
    }
    for module, relative in expected.items():
        module_path = pathlib.Path(module.__file__).resolve(strict=True)
        expected_path = (_REPOSITORY_ROOT / relative).resolve(strict=True)
        if os.path.normcase(str(module_path)) != os.path.normcase(str(expected_path)):
            raise ValueError(f"theta0 v3 handoff module origin drifted: {relative}")
    expected_self = (
        _REPOSITORY_ROOT / "packages/lifeform-evolution/src/lifeform_evolution/"
        "relationship_product_horizon_theta0_v3_handoff.py"
    ).resolve(strict=True)
    if os.path.normcase(str(pathlib.Path(__file__).resolve(strict=True))) != os.path.normcase(str(expected_self)):
        raise ValueError("theta0 v3 handoff owner module origin drifted")
    packages_root = (_REPOSITORY_ROOT / "packages").resolve(strict=True)
    for name, module in tuple(sys.modules.items()):
        if name.partition(".")[0] not in _REPOSITORY_PACKAGE_TOP_LEVELS:
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is not None:
            resolved_file = pathlib.Path(module_file).resolve(strict=True)
            if not resolved_file.is_relative_to(packages_root):
                raise ValueError(f"theta0 v3 handoff repository module escaped packages tree: {name}")
            continue
        module_path = getattr(module, "__path__", None)
        if module_path is None:
            raise ValueError(f"theta0 v3 handoff repository module has no origin: {name}")
        resolved_paths = tuple(pathlib.Path(item).resolve(strict=True) for item in module_path)
        if not resolved_paths or any(not item.is_relative_to(packages_root) for item in resolved_paths):
            raise ValueError(f"theta0 v3 handoff namespace module escaped packages tree: {name}")


def _build_manifest(
    *,
    root: pathlib.Path,
    protocol: RelationshipProductHorizonTheta0V3HandoffProtocol,
    implementation: _CurrentImplementationReceipt,
    acceptance: _HistoricalAcceptanceReceipt,
    authorization: RelationshipProductV2CondensedTheta0FrozenPulseAuthorization,
) -> dict[str, object]:
    files = []
    for relative in sorted(_EXPECTED_OUTPUT_FILES - {_MANIFEST_FILENAME}):
        raw = _read_regular(root / relative)
        files.append(
            {
                "path": relative,
                "raw_bytes": len(raw),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    theta = protocol.theta0_input
    core: dict[str, object] = {
        "schema_version": THETA0_V3_HANDOFF_MANIFEST_SCHEMA_VERSION,
        "protocol_id": protocol.protocol_id,
        "protocol_raw_sha256": protocol.raw_sha256,
        "implementation_git_commit": implementation.implementation_git_commit,
        "implementation_owned_git_object_ids": [
            {
                "path": path,
                "git_object_type": object_type,
                "git_object_id": object_id,
            }
            for path, object_type, object_id in implementation.owned_git_object_ids
        ],
        "implementation_owned_git_object_ids_sha256": implementation.owned_git_object_ids_sha256,
        "historical_theta0_protocol_id": theta["protocol_id"],
        "historical_theta0_artifact_id": theta["artifact_id"],
        "historical_theta0_implementation_git_commit": theta["implementation_git_commit"],
        "historical_acceptance_report_artifact_id": acceptance.report_artifact_id,
        "historical_acceptance_report_raw_sha256": acceptance.report_raw_sha256,
        "historical_acceptance_receipt_bound": True,
        "historical_validate_existing_reexecuted": False,
        "artifact_specific_current_compatibility_replay_passed": True,
        "general_backward_compatibility": False,
        "current_full_typed_federation_rehydrated": True,
        "current_execution_git_object_closure_reobserved_before_replay_and_manifest": True,
        "source_v5_bound": False,
        "theta0_artifact_id": authorization.learned_theta0_artifact.artifact_id,
        "theta0_authorization_id": authorization.authorization_id,
        "source_pulse_federated_matched_transitions_id": authorization.to_payload()[
            "source_pulse_federated_matched_transitions_id"
        ],
        "source_gate_federated_matched_transitions_id": authorization.to_payload()[
            "source_gate_federated_matched_transitions_id"
        ],
        "source_apply_receipt_id": authorization.to_payload()["source_apply_receipt_id"],
        "source_transition_disposition": "apply",
        "evaluation_transition_disposition": None,
        "cold_checkpoint_update_count": authorization.frozen_policy.checkpoint.update_count,
        "cold_checkpoint_informative_update_count": (authorization.frozen_policy.checkpoint.informative_update_count),
        "cold_checkpoint_processed_credit_count": len(authorization.frozen_policy.checkpoint.processed_credit_ids),
        "model_output_count": 0,
        "cuda_execution_count": 0,
        "reader_fit_count": 0,
        "compatibility_replay_onboarding_write_count": 448,
        "compatibility_replay_reader_inference_count": 896,
        "compatibility_replay_precomputed_embedding_lookup_count": 896,
        "compatibility_replay_forecast_publication_count": 896,
        "compatibility_replay_source_settlement_count": 896,
        "compatibility_replay_prediction_error_derivation_count": 896,
        "compatibility_replay_credit_derivation_count": 896,
        "compatibility_replay_parent_apply_update_count": theta["apply_update_count"],
        "compatibility_replay_parent_apply_informative_update_count": theta["apply_informative_update_count"],
        "compatibility_replay_parent_apply_cap_hit_count": theta["apply_cap_hit_count"],
        "compatibility_replay_parent_withhold_update_count": theta["withhold_update_count"],
        "new_campaign_decision_count": 0,
        "new_scientific_outcome_observation_count": 0,
        "new_scientific_prediction_error_observation_count": 0,
        "new_scientific_credit_observation_count": 0,
        "authorization_evaluation_gate_update_count": 0,
        "files": files,
        "status": "theta0_v3_typed_cross_commit_handoff_materialized_effect_not_tested",
        "claims": protocol.payload["claims_ceiling"],
        "claim_boundary": protocol.payload["claim_boundary"],
    }
    return {"artifact_id": sha256_json(core), **core}


def _require_disjoint_output(*, output: pathlib.Path, inputs: tuple[pathlib.Path, ...]) -> None:
    output_text = os.path.normcase(str(output.resolve(strict=False)))
    for item in inputs:
        input_text = os.path.normcase(str(item.resolve(strict=True)))
        try:
            common = os.path.commonpath((output_text, input_text))
        except ValueError:
            continue
        if common in {output_text, input_text}:
            raise ValueError("theta0 v3 handoff output must be disjoint from inputs")


def _filesystem_fingerprint(root: pathlib.Path) -> dict[str, tuple[int, int, str]]:
    if not root.is_dir():
        raise FileNotFoundError(f"theta0 v3 handoff output does not exist: {root}")
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            hashlib.sha256(_read_regular(path)).hexdigest(),
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def _file_inventory_contract(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, list):
        raise TypeError("file inventory must be an exact list")
    result: dict[str, dict[str, object]] = {}
    for index, item in enumerate(value):
        raw = _mapping(item, f"file_inventory[{index}]")
        _exact_keys(raw, {"path", "raw_bytes", "raw_sha256"}, source="file pin")
        path = _text(raw["path"], "file path")
        if path in result or pathlib.PurePosixPath(path).name != path:
            raise ValueError("file inventory path is duplicate or non-local")
        raw_bytes = raw["raw_bytes"]
        if type(raw_bytes) is not int or raw_bytes < 1:
            raise ValueError("file inventory raw_bytes must be positive")
        result[path] = {
            "raw_bytes": raw_bytes,
            "raw_sha256": _digest(raw["raw_sha256"], "file raw_sha256"),
        }
    return result


def _run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=_REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError("theta0 v3 handoff could not execute git") from exc
    if check and completed.returncode != 0:
        raise ValueError(
            f"theta0 v3 handoff git lineage check failed: args={args!r}; stderr={completed.stderr.strip()!r}"
        )
    return completed


def _read_regular(path: pathlib.Path) -> bytes:
    source = pathlib.Path(path)
    if source.is_symlink():
        raise ValueError(f"theta0 v3 handoff rejects symlink input: {source}")
    details = source.stat()
    if not stat.S_ISREG(details.st_mode):
        raise ValueError(f"theta0 v3 handoff input is not a regular file: {source}")
    raw = source.read_bytes()
    if raw.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ValueError(f"theta0 v3 handoff rejects unresolved LFS pointer: {source}")
    return raw


def _write_create_only(path: pathlib.Path, raw: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _artifact_bytes(payload: Mapping[str, object]) -> bytes:
    return (canonical_json(payload) + "\n").encode("utf-8")


def _parse_json(raw: bytes, *, source: str) -> dict[str, object]:
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_object_pairs_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source} is not strict UTF-8 JSON") from exc
    if type(payload) is not dict:
        raise TypeError(f"{source} must be an exact object")
    return payload


def _object_pairs_no_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _mapping(value: object, source: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{source} must be an exact object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], *, source: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{source} keys drifted")


def _require_exact_value(value: object, expected: object, *, source: str) -> None:
    """Reject JSON drift including Python's bool/int equality aliasing."""

    if type(value) is not type(expected):
        raise ValueError(f"{source} drifted")
    if type(expected) is dict:
        actual_mapping = _mapping(value, source)
        expected_mapping = _mapping(expected, f"{source} expected")
        if set(actual_mapping) != set(expected_mapping):
            raise ValueError(f"{source} drifted")
        for key in expected_mapping:
            _require_exact_value(
                actual_mapping[key],
                expected_mapping[key],
                source=f"{source}.{key}",
            )
        return
    if type(expected) is list:
        actual_list = value
        expected_list = expected
        if len(actual_list) != len(expected_list):
            raise ValueError(f"{source} drifted")
        for index, (actual_item, expected_item) in enumerate(zip(actual_list, expected_list, strict=True)):
            _require_exact_value(
                actual_item,
                expected_item,
                source=f"{source}[{index}]",
            )
        return
    if value != expected:
        raise ValueError(f"{source} drifted")


def _text(value: object, source: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{source} must be a non-empty exact string")
    return value


def _digest(value: object, source: str) -> str:
    text = _text(value, source)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{source} must be lowercase sha256")
    return text


def _git_commit(value: object) -> str:
    text = _text(value, "implementation_git_commit")
    if _GIT_COMMIT.fullmatch(text) is None:
        raise ValueError("implementation_git_commit must be lowercase 40-hex")
    return text


__all__ = [
    "RelationshipProductHorizonTheta0V3HandoffBundle",
    "RelationshipProductHorizonTheta0V3HandoffProtocol",
    "THETA0_V3_HANDOFF_MANIFEST_SCHEMA_VERSION",
    "THETA0_V3_HANDOFF_PROTOCOL_SCHEMA_VERSION",
    "load_relationship_product_horizon_theta0_v3_handoff",
    "load_relationship_product_horizon_theta0_v3_handoff_protocol",
    "materialize_relationship_product_horizon_theta0_v3_handoff",
    "relationship_product_horizon_theta0_v3_handoff_protocol_path",
    "validate_relationship_product_horizon_theta0_v3_handoff",
]
