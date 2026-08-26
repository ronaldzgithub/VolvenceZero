"""Development-only source-v3 bootstrap for a nonzero relationship gate theta0.

The public join is label-free.  During calibration one legacy zero-initialized
gate is carried across eight independently reset owner roots.  Every public
preaction trace line is flushed before the existing reactive environment owner
is allowed to settle the actually delivered action.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
import pathlib
import re
from typing import Mapping, Protocol

from lifeform_domain_emogpt.lab.contracts import canonical_json, sha256_json
from lifeform_domain_emogpt.lab.relationship_product_pilot_source_v2 import (
    build_relationship_product_pilot_environment,
    build_relationship_product_pilot_evaluator_bundle,
    build_relationship_product_pilot_public_view,
    load_relationship_product_pilot_source_protocol,
    relationship_product_pilot_source_protocol_path,
)
from lifeform_domain_emogpt.lab.relationship_product_pulse import (
    RelationshipProductOnboardingInput,
    RelationshipProductPreActionRequest,
    RelationshipProductPulseAuthorization,
    RelationshipProductSettlementInput,
    append_relationship_product_onboarding,
    prepare_relationship_product_preaction,
    settle_relationship_product_pulse,
)
from lifeform_domain_emogpt.relationship_action_contracts import (
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
    RelationshipAction,
)
from lifeform_domain_emogpt.relationship_action_gate import (
    RELATIONSHIP_ACTION_GATE_FEATURE_ORDER,
    RELATIONSHIP_ACTION_GATE_THRESHOLD_RULE,
    RelationshipActionGate,
    RelationshipActionGateArtifact,
    RelationshipActionGateCheckpoint,
    RelationshipActionGateMode,
    RelationshipActionGateTheta0Artifact,
)
from lifeform_domain_emogpt.relationship_condition_reader import (
    FrozenLinearRelationshipConditionReaderArtifact,
    FrozenLinearRelationshipConditionReaderRuntime,
    FrozenLinearRelationshipPreferenceForecastRuntime,
)
from lifeform_evolution.relationship_lab_product_model_adapters import (
    PrecomputedPublicEmbeddingTable,
    PrecomputedPublicSemanticEmbedder,
    load_precomputed_public_embedding_table,
)
from lifeform_evolution.relationship_product_horizon_development_reader import (
    validate_relationship_product_horizon_development_reader,
)
from lifeform_evolution.relationship_product_source_admission import (
    SOURCE_ADMISSION_MANIFEST_SCHEMA_VERSION,
    SOURCE_ADMISSION_PROTOCOL_SCHEMA_VERSION,
    SOURCE_ADMISSION_ROOT_MANIFEST_SCHEMA_VERSION,
)
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
)
from volvence_zero.social import social_record_store_persistence_sha256
from volvence_zero.social import PreferenceActionForecastRequest
from volvence_zero.social_cognition import (
    PreferenceActionOutcomeEvidence,
    preference_action_forecast_to_payload,
)
from volvence_zero.substrate import SubstrateSnapshot, SurfaceKind
from volvence_zero.temporal_types import TemporalActionAdvisoryStatus


THETA0_CALIBRATION_PROTOCOL_SCHEMA_VERSION = (
    "relationship-product-horizon-theta0-calibration-protocol.v1"
)
THETA0_PUBLIC_JOIN_SCHEMA_VERSION = (
    "relationship-product-horizon-theta0-public-join.v1"
)
THETA0_TRACE_SCHEMA_VERSION = (
    "relationship-product-horizon-theta0-calibration-trace.v1"
)
THETA0_MANIFEST_SCHEMA_VERSION = (
    "relationship-product-horizon-theta0-calibration-manifest.v1"
)

_PROTOCOL_FILENAME = "relationship_product_horizon_theta0_calibration_v1.json"
_TRACE_FILENAME = "calibration_trace.jsonl"
_BASE_OUTPUT_FILES = frozenset(
    {"protocol.json", "public_join.json", _TRACE_FILENAME, "manifest.json"}
)
_SUCCESS_OUTPUT_FILES = frozenset({*_BASE_OUTPUT_FILES, "theta0_artifact.json"})
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_INTERLOCUTOR_ID = "primary"


@dataclass(frozen=True)
class RelationshipProductHorizonTheta0CalibrationProtocol:
    payload: Mapping[str, object]
    raw_bytes: bytes
    protocol_id: str
    raw_sha256: str


@dataclass(frozen=True)
class _Dependencies:
    protocol: RelationshipProductHorizonTheta0CalibrationProtocol
    public_view: object
    source_owner: object
    preflight_corpus: Mapping[str, object]
    table: PrecomputedPublicEmbeddingTable
    reader_artifact: FrozenLinearRelationshipConditionReaderArtifact
    forecast_runtime: FrozenLinearRelationshipPreferenceForecastRuntime


@dataclass(frozen=True)
class _SafeEnvironmentOutcome:
    environment_subject_id: str
    selected_action_id: str
    typed_outcome_id: str
    rendered_user_reaction: str
    environment_evidence_ref: str
    environment_version: str


@dataclass(frozen=True)
class _CalibrationReplay:
    final_checkpoint: RelationshipActionGateCheckpoint
    credit_ids: tuple[str, ...]
    root_mapping: tuple[Mapping[str, object], ...]
    terminal_status: str


class _TraceSink(Protocol):
    def append(self, payload: Mapping[str, object]) -> None: ...


class _MemoryTraceSink:
    def __init__(self) -> None:
        self._chunks: list[bytes] = []

    def append(self, payload: Mapping[str, object]) -> None:
        self._chunks.append(_canonical_bytes(payload))

    @property
    def raw_bytes(self) -> bytes:
        return b"".join(self._chunks)


class _FsyncTraceSink(_MemoryTraceSink):
    def __init__(self, path: pathlib.Path) -> None:
        super().__init__()
        self._handle = path.open("xb")

    def append(self, payload: Mapping[str, object]) -> None:
        raw = _canonical_bytes(payload)
        self._handle.write(raw)
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._chunks.append(raw)

    def close(self) -> None:
        self._handle.close()


class _EnvironmentScope:
    """Keep evaluator truth behind one post-preaction environment facade."""

    def __init__(self, *, dependencies: _Dependencies) -> None:
        source_pin = _mapping(
            dependencies.protocol.payload["source_v3_admission"],
            "source_v3_admission",
        )
        evaluator = build_relationship_product_pilot_evaluator_bundle(
            dependencies.source_owner
        )
        if evaluator.protocol_sha256 != source_pin["source_protocol_id"]:
            raise ValueError("environment source protocol identity drifted")
        if evaluator.sealed_bundle_sha256 != source_pin[
            "sealed_evaluator_bundle_sha256"
        ]:
            raise ValueError("environment sealed identity drifted")

        by_world: dict[str, list[object]] = {}
        for decision in evaluator.decision_sessions:
            by_world.setdefault(decision.world_clone_id, []).append(decision)
        public_subjects = dependencies.public_view.subjects
        mapping: list[Mapping[str, object]] = []
        decisions: dict[tuple[str, str], object] = {}
        environments: dict[str, object] = {}
        for root_index, subject in enumerate(public_subjects):
            matched = by_world.get(subject.world_clone_id, [])
            subject_ids = {item.subject_id for item in matched}
            if len(matched) != 24 or len(subject_ids) != 1:
                raise ValueError("world_clone_id did not uniquely join one evaluator root")
            evaluator_subject_id = next(iter(subject_ids))
            mapping.append(
                {
                    "root_sequence_index": root_index,
                    "subject_scope": subject.subject_scope,
                    "world_clone_id": subject.world_clone_id,
                    "environment_subject_id": evaluator_subject_id,
                }
            )
            environments[evaluator_subject_id] = (
                build_relationship_product_pilot_environment(
                    evaluator,
                    subject_id=evaluator_subject_id,
                )
            )
            for decision in matched:
                key = (subject.world_clone_id, decision.decision_id)
                if key in decisions:
                    raise ValueError("environment decision identity is not unique")
                decisions[key] = decision
        if len(decisions) != 192 or len(mapping) != 8:
            raise ValueError("environment root/decision inventory drifted")
        self._mapping = tuple(mapping)
        self._decisions = decisions
        self._environments = environments

    @property
    def root_mapping(self) -> tuple[Mapping[str, object], ...]:
        return self._mapping

    def settle(
        self,
        *,
        public_subject: object,
        public_session: object,
        selected_action_id: str,
    ) -> _SafeEnvironmentOutcome:
        key = (public_subject.world_clone_id, public_session.decision_id)
        decision = self._decisions.get(key)
        if decision is None:
            raise ValueError("public decision has no exact environment owner row")
        if (
            decision.session_id != public_session.session_id
            or decision.decision_index != public_session.decision_index
            or decision.domain_id != public_session.domain_id
        ):
            raise ValueError("public/environment decision lineage drifted")
        action = RelationshipAction(selected_action_id)
        environment = self._environments[decision.subject_id]
        outcome = environment.settle(
            scene_id=decision.scene_id,
            decision_id=decision.decision_id,
            action=action,
            seed=decision.environment_seed,
        )
        if outcome.selected_action.value != selected_action_id:
            raise ValueError("environment settled a different action")
        return _SafeEnvironmentOutcome(
            environment_subject_id=decision.subject_id,
            selected_action_id=outcome.selected_action.value,
            typed_outcome_id=outcome.typed_outcome.value,
            rendered_user_reaction=outcome.rendered_user_reaction,
            environment_evidence_ref=outcome.environment_evidence_ref,
            environment_version=outcome.environment_version,
        )


def relationship_product_horizon_theta0_calibration_protocol_path() -> pathlib.Path:
    return pathlib.Path(__file__).with_name("protocols") / _PROTOCOL_FILENAME


def load_relationship_product_horizon_theta0_calibration_protocol(
    path: pathlib.Path | None = None,
) -> RelationshipProductHorizonTheta0CalibrationProtocol:
    protocol_path = pathlib.Path(
        path or relationship_product_horizon_theta0_calibration_protocol_path()
    )
    raw = protocol_path.read_bytes()
    payload = _parse_json_bytes(raw, source="theta0 calibration protocol")
    _exact_keys(
        payload,
        {
            "schema_version",
            "evidence_tier",
            "owner",
            "purpose",
            "source_v3_admission",
            "preflight_public_corpus",
            "development_reader",
            "gate",
            "topology",
            "causal_firewall",
            "terminal_gates",
            "claims",
            "claim_boundary",
        },
        "theta0 calibration protocol",
    )
    if payload["schema_version"] != THETA0_CALIBRATION_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("theta0 calibration protocol schema drifted")
    if payload["evidence_tier"] != "development":
        raise ValueError("theta0 calibration evidence tier drifted")
    if payload["owner"] != (
        "lifeform_evolution.relationship_product_horizon_theta0_calibration"
    ):
        raise ValueError("theta0 calibration owner drifted")
    if payload["purpose"] != "source_v3_evaluation_outside_theta0_calibration":
        raise ValueError("theta0 calibration purpose drifted")
    _validate_protocol_contract(payload)
    return RelationshipProductHorizonTheta0CalibrationProtocol(
        payload=payload,
        raw_bytes=raw,
        protocol_id=sha256_json(payload),
        raw_sha256=_sha256_bytes(raw),
    )


def _validate_protocol_contract(payload: Mapping[str, object]) -> None:
    source = _mapping(payload["source_v3_admission"], "source_v3_admission")
    preflight = _mapping(
        payload["preflight_public_corpus"],
        "preflight_public_corpus",
    )
    reader = _mapping(payload["development_reader"], "development_reader")
    gate = _mapping(payload["gate"], "gate")
    topology = _mapping(payload["topology"], "topology")
    firewall = _mapping(payload["causal_firewall"], "causal_firewall")
    terminal = _mapping(payload["terminal_gates"], "terminal_gates")
    claims = _mapping(payload["claims"], "claims")
    for container, expected, source_name in (
        (
            source,
            {
                "protocol_id",
                "artifact_id",
                "manifest_raw_sha256",
                "replay_materialization_artifact_id",
                "replay_manifest_raw_sha256",
                "admission_protocol_raw_sha256",
                "source_protocol_id",
                "source_protocol_raw_sha256",
                "public_plan_sha256",
                "public_plan_raw_sha256",
                "sealed_evaluator_bundle_sha256",
                "implementation_git_commit",
            },
            "source_v3_admission",
        ),
        (
            preflight,
            {
                "protocol_id",
                "manifest_artifact_id",
                "manifest_raw_sha256",
                "public_corpus_artifact_id",
                "public_corpus_raw_sha256",
                "challenge_text_count",
                "challenge_label_files_consumed",
                "group_split_files_consumed",
            },
            "preflight_public_corpus",
        ),
        (
            reader,
            {
                "protocol_id",
                "artifact_id",
                "manifest_raw_sha256",
                "embedding_table_artifact_id",
                "embedding_table_raw_sha256",
                "reader_artifact_id",
                "reader_artifact_raw_sha256",
                "implementation_git_commit",
                "condition_reader_qualified",
            },
            "development_reader",
        ),
        (
            gate,
            {
                "artifact_id",
                "artifact_version",
                "mode",
                "random_seed",
                "learning_rate_hex",
                "max_abs_parameter_hex",
                "feature_order",
                "threshold_rule",
                "owner_replay_loader_git_commit",
            },
            "gate",
        ),
        (
            topology,
            {
                "root_count",
                "onboarding_per_root",
                "decision_per_root",
                "decision_count",
                "source_order",
                "owner_reset_each_root",
                "single_global_gate_carried_across_roots",
                "per_root_gate_reset_forbidden",
                "parameter_averaging_forbidden",
                "global_sequence_index",
                "credit_timestamp_formula",
                "credit_timestamp_strictly_increasing",
            },
            "topology",
        ),
        (
            firewall,
            {
                "preaction_trace_fsync_before_environment_settle",
                "actual_action_owner",
                "environment_owner_api",
                "public_to_evaluator_root_join",
                "environment_safe_projection_fields",
                "hidden_fields_forbidden_from_reader_gate_credit",
                "evaluation_or_judge_feedback_to_learning",
                "oracle_input_to_learning",
                "forced_action",
                "os_process_secrecy_claim",
            },
            "causal_firewall",
        ),
        (
            terminal,
            {
                "gate_update_count",
                "processed_credit_id_count",
                "unique_credit_id_count",
                "pending_decision_count",
                "one_credit_apply_per_decision",
                "nonzero_parameter_required",
                "cold_theta0_update_count",
                "cold_theta0_processed_credit_id_count",
                "cold_theta0_pending_decision_count",
                "all_zero_terminal",
                "successful_terminal",
                "scientific_retry_with_different_order_or_seed_forbidden",
            },
            "terminal_gates",
        ),
        (
            claims,
            {
                "calibration_execution_authorized",
                "development_public_exact_join_may_be_derived",
                "development_theta0_may_be_materialized",
                "reader_qualified",
                "campaign_execution_authorized",
                "formal_evidence_authorized",
                "integrated_horizon_authorized",
                "appendable_effect",
                "readable_effect",
                "learnable_effect",
                "steerable_effect",
                "four_able_complete",
                "human_validation_complete",
                "production_active",
                "model_output_count",
                "cuda_execution_count",
            },
            "claims",
        ),
    ):
        _exact_keys(container, expected, source_name)
    for container, fields in (
        (
            source,
            (
                "protocol_id",
                "artifact_id",
                "manifest_raw_sha256",
                "replay_materialization_artifact_id",
                "replay_manifest_raw_sha256",
                "admission_protocol_raw_sha256",
                "source_protocol_id",
                "source_protocol_raw_sha256",
                "public_plan_sha256",
                "public_plan_raw_sha256",
                "sealed_evaluator_bundle_sha256",
            ),
        ),
        (
            preflight,
            (
                "protocol_id",
                "manifest_artifact_id",
                "manifest_raw_sha256",
                "public_corpus_artifact_id",
                "public_corpus_raw_sha256",
            ),
        ),
        (
            reader,
            (
                "protocol_id",
                "artifact_id",
                "manifest_raw_sha256",
                "embedding_table_artifact_id",
                "embedding_table_raw_sha256",
                "reader_artifact_id",
                "reader_artifact_raw_sha256",
            ),
        ),
    ):
        for field_name in fields:
            _digest(container[field_name], field_name)
    if (
        _integer(preflight["challenge_text_count"], "challenge_text_count") != 224
        or preflight["challenge_label_files_consumed"] != []
        or preflight["group_split_files_consumed"] != []
    ):
        raise ValueError("theta0 public join boundary drifted")
    if _boolean(
        reader["condition_reader_qualified"],
        "condition_reader_qualified",
    ):
        raise ValueError("development reader cannot be promoted by calibration")
    _git_commit(source["implementation_git_commit"])
    _git_commit(reader["implementation_git_commit"])
    _git_commit(gate["owner_replay_loader_git_commit"])
    artifact = RelationshipActionGateArtifact()
    if (
        gate["artifact_id"] != artifact.artifact_id
        or _integer(gate["artifact_version"], "gate.artifact_version")
        != artifact.artifact_version
        or gate["mode"] != RelationshipActionGateMode.LEARNED.value
        or gate["random_seed"] != "relationship-action-random-control-v1"
        or gate["learning_rate_hex"] != artifact.learning_rate.hex()
        or gate["max_abs_parameter_hex"] != artifact.max_abs_parameter.hex()
        or tuple(gate["feature_order"]) != RELATIONSHIP_ACTION_GATE_FEATURE_ORDER
        or gate["threshold_rule"] != RELATIONSHIP_ACTION_GATE_THRESHOLD_RULE
    ):
        raise ValueError("theta0 calibration gate contract drifted")
    expected_topology = (8, 4, 24, 192, True, True, True, True, True)
    observed_topology = (
        _integer(topology["root_count"], "topology.root_count"),
        _integer(
            topology["onboarding_per_root"],
            "topology.onboarding_per_root",
        ),
        _integer(topology["decision_per_root"], "topology.decision_per_root"),
        _integer(topology["decision_count"], "topology.decision_count"),
        _boolean(
            topology["owner_reset_each_root"],
            "topology.owner_reset_each_root",
        ),
        _boolean(
            topology["single_global_gate_carried_across_roots"],
            "topology.single_global_gate_carried_across_roots",
        ),
        _boolean(
            topology["per_root_gate_reset_forbidden"],
            "topology.per_root_gate_reset_forbidden",
        ),
        _boolean(
            topology["parameter_averaging_forbidden"],
            "topology.parameter_averaging_forbidden",
        ),
        _boolean(
            topology["credit_timestamp_strictly_increasing"],
            "topology.credit_timestamp_strictly_increasing",
        ),
    )
    if observed_topology != expected_topology:
        raise ValueError("theta0 calibration topology drifted")
    if (
        topology["source_order"]
        != "public_subject_array_then_onboarding_0_to_3_then_decision_0_to_23"
        or topology["global_sequence_index"] != "0_to_191"
        or topology["credit_timestamp_formula"]
        != "root_sequence_index_times_52_plus_5_plus_2_times_decision_index"
    ):
        raise ValueError("theta0 calibration sequence contract drifted")
    safe_fields = (
        "environment_subject_id",
        "selected_action_id",
        "typed_outcome_id",
        "rendered_user_reaction",
        "environment_evidence_ref",
        "environment_version",
    )
    hidden_fields = (
        "condition_id",
        "policy_id",
        "preferred_action_id",
        "environment_seed",
        "outcome_distribution",
        "deterministic_draw",
        "evaluation_score",
        "judge_feedback",
    )
    if (
        not _boolean(
            firewall["preaction_trace_fsync_before_environment_settle"],
            "causal_firewall.preaction_trace_fsync_before_environment_settle",
        )
        or firewall["actual_action_owner"]
        != "temporal_snapshot.value.active_abstract_action"
        or firewall["environment_owner_api"]
        != "ReactiveRelationshipEnvironment.settle"
        or firewall["public_to_evaluator_root_join"] != "unique_world_clone_id"
        or tuple(_list(firewall["environment_safe_projection_fields"], "safe fields"))
        != safe_fields
        or tuple(
            _list(
                firewall["hidden_fields_forbidden_from_reader_gate_credit"],
                "hidden fields",
            )
        )
        != hidden_fields
        or _boolean(
            firewall["evaluation_or_judge_feedback_to_learning"],
            "causal_firewall.evaluation_or_judge_feedback_to_learning",
        )
        or _boolean(
            firewall["oracle_input_to_learning"],
            "causal_firewall.oracle_input_to_learning",
        )
        or _boolean(
            firewall["forced_action"],
            "causal_firewall.forced_action",
        )
        or _boolean(
            firewall["os_process_secrecy_claim"],
            "causal_firewall.os_process_secrecy_claim",
        )
    ):
        raise ValueError("theta0 causal firewall drifted")
    terminal_count_fields = (
        "gate_update_count",
        "processed_credit_id_count",
        "unique_credit_id_count",
        "pending_decision_count",
        "cold_theta0_update_count",
        "cold_theta0_processed_credit_id_count",
        "cold_theta0_pending_decision_count",
    )
    if tuple(
        _integer(terminal[field], f"terminal_gates.{field}")
        for field in terminal_count_fields
    ) != (192, 192, 192, 0, 0, 0, 0):
        raise ValueError("theta0 terminal counts drifted")
    if (
        not _boolean(
            terminal["nonzero_parameter_required"],
            "terminal_gates.nonzero_parameter_required",
        )
        or not _boolean(
            terminal["one_credit_apply_per_decision"],
            "terminal_gates.one_credit_apply_per_decision",
        )
        or terminal["all_zero_terminal"]
        != "calibration_completed_no_nonzero_theta0"
        or terminal["successful_terminal"]
        != "development_theta0_materialized_effect_not_tested"
        or not _boolean(
            terminal["scientific_retry_with_different_order_or_seed_forbidden"],
            "terminal_gates.scientific_retry_with_different_order_or_seed_forbidden",
        )
    ):
        raise ValueError("theta0 terminal honesty boundary drifted")
    allowed_true = {
        "calibration_execution_authorized",
        "development_public_exact_join_may_be_derived",
        "development_theta0_may_be_materialized",
    }
    for key in allowed_true:
        if not _boolean(claims[key], f"claims.{key}"):
            raise ValueError(f"theta0 protocol claim must remain true: {key}")
    if {key for key, value in claims.items() if value is True} != allowed_true:
        raise ValueError("theta0 protocol claim ceiling drifted")
    if (
        _integer(claims["model_output_count"], "claims.model_output_count") != 0
        or _integer(claims["cuda_execution_count"], "claims.cuda_execution_count")
        != 0
    ):
        raise ValueError("theta0 protocol model/CUDA count drifted")
    for key, value in claims.items():
        if key in allowed_true or key in {"model_output_count", "cuda_execution_count"}:
            continue
        if value is not False:
            raise ValueError(f"theta0 protocol claim must remain false: {key}")
    _text(payload["claim_boundary"], "claim_boundary")


def _validate_source_admission_public_envelope(
    *,
    manifest: Mapping[str, object],
    source_pin: Mapping[str, object],
) -> None:
    """Validate the admitted public envelope without opening sealed replay files."""

    _exact_keys(
        manifest,
        {
            "artifact_id",
            "schema_version",
            "protocol_id",
            "implementation_git_commit",
            "comparison_raw_sha256",
            "materialization_artifact_id",
            "subject_count",
            "onboarding_session_count",
            "decision_count",
            "action_counterfactual_commitment_count",
            "files",
            "status",
            "claims",
            "claim_boundary",
        },
        "source-v3 admission manifest",
    )
    if (
        manifest["schema_version"] != SOURCE_ADMISSION_MANIFEST_SCHEMA_VERSION
        or manifest["protocol_id"] != source_pin["protocol_id"]
        or manifest["artifact_id"] != source_pin["artifact_id"]
        or manifest["materialization_artifact_id"]
        != source_pin["replay_materialization_artifact_id"]
        or manifest["implementation_git_commit"]
        != source_pin["implementation_git_commit"]
        or manifest["status"] != "campaign_input_admitted_execution_not_authorized"
        or tuple(
            manifest[field]
            for field in (
                "subject_count",
                "onboarding_session_count",
                "decision_count",
                "action_counterfactual_commitment_count",
            )
        )
        != (8, 32, 192, 576)
    ):
        raise ValueError("source-v3 admission public envelope drifted")
    if manifest["artifact_id"] != sha256_json(
        {key: value for key, value in manifest.items() if key != "artifact_id"}
    ):
        raise ValueError("source-v3 admission manifest content identity drifted")
    claims = _mapping(manifest["claims"], "source-v3 admission claims")
    if {key for key, value in claims.items() if value is True} != {
        "campaign_input_admitted"
    } or claims.get("model_output_count") != 0:
        raise ValueError("source-v3 admission claim ceiling drifted")
    files = _file_entry_map(manifest["files"], "source-v3 admission files")
    expected_raw = {
        "replay_a/manifest.json": source_pin["replay_manifest_raw_sha256"],
        "replay_a/protocol.json": source_pin["admission_protocol_raw_sha256"],
        "replay_a/public/source_plan.json": source_pin["public_plan_raw_sha256"],
    }
    if set(files) != {
        "comparison.json",
        "replay_a/manifest.json",
        "replay_a/protocol.json",
        "replay_a/public/source_plan.json",
        "replay_a/sealed/action_counterfactual_commitments.json",
        "replay_a/sealed/evaluator_bundle.json",
        "replay_b/manifest.json",
        "replay_b/protocol.json",
        "replay_b/public/source_plan.json",
        "replay_b/sealed/action_counterfactual_commitments.json",
        "replay_b/sealed/evaluator_bundle.json",
    }:
        raise ValueError("source-v3 admission file envelope drifted")
    for relative, expected in expected_raw.items():
        if files[relative]["raw_sha256"] != expected:
            raise ValueError(f"source-v3 admission file pin drifted: {relative}")


def _validate_source_replay_public_envelope(
    *,
    manifest: Mapping[str, object],
    source_pin: Mapping[str, object],
) -> None:
    _exact_keys(
        manifest,
        {
            "artifact_id",
            "schema_version",
            "protocol_id",
            "source_protocol_sha256",
            "public_plan_sha256",
            "sealed_bundle_sha256",
            "subject_count",
            "onboarding_session_count",
            "decision_count",
            "action_counterfactual_commitment_count",
            "files",
            "claims",
        },
        "source-v3 replay manifest",
    )
    if (
        manifest["schema_version"] != SOURCE_ADMISSION_ROOT_MANIFEST_SCHEMA_VERSION
        or manifest["artifact_id"]
        != source_pin["replay_materialization_artifact_id"]
        or manifest["protocol_id"] != source_pin["protocol_id"]
        or manifest["source_protocol_sha256"] != source_pin["source_protocol_id"]
        or manifest["public_plan_sha256"] != source_pin["public_plan_sha256"]
        or manifest["sealed_bundle_sha256"]
        != source_pin["sealed_evaluator_bundle_sha256"]
        or tuple(
            manifest[field]
            for field in (
                "subject_count",
                "onboarding_session_count",
                "decision_count",
                "action_counterfactual_commitment_count",
            )
        )
        != (8, 32, 192, 576)
    ):
        raise ValueError("source-v3 replay public envelope drifted")
    if manifest["artifact_id"] != sha256_json(
        {key: value for key, value in manifest.items() if key != "artifact_id"}
    ):
        raise ValueError("source-v3 replay manifest content identity drifted")
    claims = _mapping(manifest["claims"], "source-v3 replay claims")
    if {key for key, value in claims.items() if value is True} != {
        "materialization_complete"
    } or claims.get("model_output_count") != 0:
        raise ValueError("source-v3 replay claim ceiling drifted")
    files = _file_entry_map(manifest["files"], "source-v3 replay files")
    if set(files) != {
        "protocol.json",
        "public/source_plan.json",
        "sealed/action_counterfactual_commitments.json",
        "sealed/evaluator_bundle.json",
    }:
        raise ValueError("source-v3 replay file envelope drifted")
    if (
        files["protocol.json"]["raw_sha256"]
        != source_pin["admission_protocol_raw_sha256"]
        or files["public/source_plan.json"]["raw_sha256"]
        != source_pin["public_plan_raw_sha256"]
    ):
        raise ValueError("source-v3 replay public file pins drifted")


def _load_dependencies(
    *,
    source_v3_admission_root: pathlib.Path,
    preflight_root: pathlib.Path,
    reader_root: pathlib.Path,
    source_v4_admission_root: pathlib.Path,
) -> _Dependencies:
    protocol = load_relationship_product_horizon_theta0_calibration_protocol()
    source_pin = _mapping(protocol.payload["source_v3_admission"], "source pin")
    source_manifest_raw = _require_raw_sha(
        source_v3_admission_root / "manifest.json",
        source_pin["manifest_raw_sha256"],
        "source-v3 admission manifest",
    )
    source_manifest = _parse_json_bytes(
        source_manifest_raw,
        source="source-v3 admission manifest",
    )
    _validate_source_admission_public_envelope(
        manifest=source_manifest,
        source_pin=source_pin,
    )
    replay_manifest_raw = _require_raw_sha(
        source_v3_admission_root / "replay_a" / "manifest.json",
        source_pin["replay_manifest_raw_sha256"],
        "source-v3 replay manifest",
    )
    replay_manifest = _parse_json_bytes(
        replay_manifest_raw,
        source="source-v3 replay manifest",
    )
    _validate_source_replay_public_envelope(
        manifest=replay_manifest,
        source_pin=source_pin,
    )
    admission_protocol_raw = _require_raw_sha(
        source_v3_admission_root / "replay_a" / "protocol.json",
        source_pin["admission_protocol_raw_sha256"],
        "source-v3 admission protocol",
    )
    admission_protocol = _parse_json_bytes(
        admission_protocol_raw,
        source="source-v3 admission protocol",
    )
    if (
        admission_protocol["schema_version"]
        != SOURCE_ADMISSION_PROTOCOL_SCHEMA_VERSION
        or sha256_json(admission_protocol) != source_pin["protocol_id"]
    ):
        raise ValueError("source-v3 admission protocol identity drifted")
    admitted_source_pin = _mapping(
        admission_protocol["source"],
        "source-v3 admission protocol source",
    )
    if (
        admitted_source_pin["protocol_sha256"] != source_pin["source_protocol_id"]
        or admitted_source_pin["protocol_raw_sha256"]
        != source_pin["source_protocol_raw_sha256"]
        or admitted_source_pin["public_plan_sha256"]
        != source_pin["public_plan_sha256"]
        or admitted_source_pin["sealed_bundle_sha256"]
        != source_pin["sealed_evaluator_bundle_sha256"]
    ):
        raise ValueError("source-v3 admission protocol source pins drifted")
    admitted_public_raw = _require_raw_sha(
        source_v3_admission_root / "replay_a" / "public" / "source_plan.json",
        source_pin["public_plan_raw_sha256"],
        "source-v3 public plan",
    )

    source_protocol_path = relationship_product_pilot_source_protocol_path()
    _require_raw_sha(
        source_protocol_path,
        source_pin["source_protocol_raw_sha256"],
        "source-v3 owner protocol",
    )
    source_owner = load_relationship_product_pilot_source_protocol(
        source_protocol_path
    )
    if source_owner.protocol_sha256 != source_pin["source_protocol_id"]:
        raise ValueError("source-v3 owner protocol identity drifted")
    public_view = build_relationship_product_pilot_public_view(source_owner)
    if public_view.public_plan_sha256 != source_pin["public_plan_sha256"]:
        raise ValueError("source-v3 public plan identity drifted")
    if admitted_public_raw != _canonical_bytes(public_view.to_sut_payload()):
        raise ValueError("source-v3 owner public plan is not byte-exact admitted input")

    preflight_pin = _mapping(
        protocol.payload["preflight_public_corpus"],
        "preflight pin",
    )
    preflight_manifest_raw = _require_raw_sha(
        preflight_root / "manifest.json",
        preflight_pin["manifest_raw_sha256"],
        "preflight manifest",
    )
    preflight_manifest = _parse_json_bytes(
        preflight_manifest_raw,
        source="preflight manifest",
    )
    if (
        preflight_manifest["protocol_id"] != preflight_pin["protocol_id"]
        or preflight_manifest["artifact_id"]
        != preflight_pin["manifest_artifact_id"]
    ):
        raise ValueError("preflight identity drifted")
    corpus_raw = _require_raw_sha(
        preflight_root / "public" / "public_corpus.json",
        preflight_pin["public_corpus_raw_sha256"],
        "preflight public corpus",
    )
    corpus = _parse_json_bytes(corpus_raw, source="preflight public corpus")
    if (
        corpus["protocol_id"] != preflight_pin["protocol_id"]
        or corpus["artifact_id"] != preflight_pin["public_corpus_artifact_id"]
    ):
        raise ValueError("preflight public corpus identity drifted")

    reader_pin = _mapping(protocol.payload["development_reader"], "reader pin")
    reader_manifest = validate_relationship_product_horizon_development_reader(
        preflight_root=preflight_root,
        source_v4_admission_root=source_v4_admission_root,
        output_dir=reader_root,
        expected_protocol_id=_text(reader_pin["protocol_id"], "reader protocol_id"),
        expected_artifact_id=_text(reader_pin["artifact_id"], "reader artifact_id"),
    )
    if reader_manifest["artifact_id"] != reader_pin["artifact_id"]:
        raise ValueError("development reader bundle identity drifted")
    _require_raw_sha(
        reader_root / "manifest.json",
        reader_pin["manifest_raw_sha256"],
        "development reader manifest",
    )
    table_path = reader_root / "embedding_table.json"
    _require_raw_sha(
        table_path,
        reader_pin["embedding_table_raw_sha256"],
        "development embedding table",
    )
    table = load_precomputed_public_embedding_table(table_path)
    if table.artifact_id != reader_pin["embedding_table_artifact_id"]:
        raise ValueError("development embedding table identity drifted")
    reader_path = reader_root / "reader_artifact.json"
    reader_raw = _require_raw_sha(
        reader_path,
        reader_pin["reader_artifact_raw_sha256"],
        "development reader artifact",
    )
    reader_artifact = FrozenLinearRelationshipConditionReaderArtifact.from_json(
        reader_raw.decode("utf-8")
    )
    if reader_artifact.artifact_id != reader_pin["reader_artifact_id"]:
        raise ValueError("development reader artifact identity drifted")
    semantic = PrecomputedPublicSemanticEmbedder(table)
    reader_runtime = FrozenLinearRelationshipConditionReaderRuntime(
        artifact=reader_artifact,
        embedder=semantic,
    )
    return _Dependencies(
        protocol=protocol,
        public_view=public_view,
        source_owner=source_owner,
        preflight_corpus=corpus,
        table=table,
        reader_artifact=reader_artifact,
        forecast_runtime=FrozenLinearRelationshipPreferenceForecastRuntime(
            reader=reader_runtime
        ),
    )


def _build_public_join(dependencies: _Dependencies) -> Mapping[str, object]:
    challenge = _list(
        dependencies.preflight_corpus["challenge_inputs"],
        "challenge_inputs",
    )
    if len(challenge) != 224:
        raise ValueError("preflight challenge count drifted")
    challenge_by_digest: dict[str, tuple[int, Mapping[str, object]]] = {}
    for index, raw in enumerate(challenge):
        item = _mapping(raw, f"challenge_inputs[{index}]")
        _exact_keys(item, {"item_id", "text", "text_sha256"}, "challenge input")
        text = _text(item["text"], "challenge text")
        digest = _digest(item["text_sha256"], "challenge text_sha256")
        if digest != _sha256_text(text):
            raise ValueError("challenge text sha256 drifted")
        if digest in challenge_by_digest:
            raise ValueError("challenge text digest is not unique")
        challenge_by_digest[digest] = (index, item)

    table_by_digest = {record.text_sha256: record for record in dependencies.table.records}
    if len(table_by_digest) != len(dependencies.table.records):
        raise ValueError("development embedding table text digest is not unique")
    rows: list[Mapping[str, object]] = []
    seen_source: set[str] = set()
    for root_index, subject in enumerate(dependencies.public_view.subjects):
        surfaces = (
            *(("onboarding", item.session_index, None, item.session_id, item.user_utterance)
              for item in subject.onboarding_sessions),
            *(("decision", item.decision_index, item.decision_id, item.session_id, item.current_input)
              for item in subject.decision_sessions),
        )
        for kind, local_index, decision_id, session_id, text in surfaces:
            digest = _sha256_text(text)
            if digest in seen_source:
                raise ValueError("source-v3 public reader text is not unique")
            seen_source.add(digest)
            challenge_match = challenge_by_digest.get(digest)
            if challenge_match is None or challenge_match[1]["text"] != text:
                raise ValueError("source-v3 text has no exact preflight challenge join")
            table_record = table_by_digest.get(digest)
            if table_record is None or table_record.text != text:
                raise ValueError("source-v3 text has no exact embedding table record")
            challenge_index, challenge_item = challenge_match
            core: dict[str, object] = {
                "source_sequence_index": len(rows),
                "root_sequence_index": root_index,
                "subject_scope": subject.subject_scope,
                "world_clone_id": subject.world_clone_id,
                "kind": kind,
                "source_local_index": local_index,
                "session_id": session_id,
                "decision_id": decision_id,
                "utf8_text_sha256": digest,
                "challenge_input_index": challenge_index,
                "challenge_item_id": challenge_item["item_id"],
                "embedding_record_artifact_id": table_record.artifact_id,
            }
            rows.append({"join_row_id": sha256_json(core), **core})
    if len(rows) != 224 or set(challenge_by_digest) != seen_source:
        raise ValueError("source/preflight 224-row join is not a bijection")
    challenge_order = tuple(row["challenge_input_index"] for row in rows)
    if len(set(challenge_order)) != 224 or set(challenge_order) != set(range(224)):
        raise ValueError("source/preflight join permutation is incomplete")
    core = {
        "schema_version": THETA0_PUBLIC_JOIN_SCHEMA_VERSION,
        "protocol_id": dependencies.protocol.protocol_id,
        "source_protocol_id": dependencies.source_owner.protocol_sha256,
        "source_public_plan_sha256": dependencies.public_view.public_plan_sha256,
        "preflight_public_corpus_artifact_id": dependencies.preflight_corpus[
            "artifact_id"
        ],
        "embedding_table_artifact_id": dependencies.table.artifact_id,
        "reader_artifact_id": dependencies.reader_artifact.artifact_id,
        "row_count": 224,
        "challenge_label_file_read_count": 0,
        "group_split_file_read_count": 0,
        "source_order_sha256": sha256_json(
            [row["join_row_id"] for row in rows]
        ),
        "source_to_challenge_permutation_sha256": sha256_json(
            list(challenge_order)
        ),
        "rows": rows,
    }
    return {"artifact_id": sha256_json(core), **core}


async def _run_calibration(
    *,
    dependencies: _Dependencies,
    public_join: Mapping[str, object],
    sink: _TraceSink,
) -> _CalibrationReplay:
    join_by_session = {
        row["session_id"]: row
        for row in _list(public_join["rows"], "public join rows")
    }
    if len(join_by_session) != 224:
        raise ValueError("theta0 public join session identity is not unique")
    sink.append(
        {
            "schema_version": THETA0_TRACE_SCHEMA_VERSION,
            "record_type": "header",
            "protocol_id": dependencies.protocol.protocol_id,
            "public_join_artifact_id": public_join["artifact_id"],
            "source_protocol_id": dependencies.source_owner.protocol_sha256,
            "reader_artifact_id": dependencies.reader_artifact.artifact_id,
            "embedding_table_artifact_id": dependencies.table.artifact_id,
            "gate_artifact_id": "relationship-action-gate-zero-init",
            "gate_artifact_version": 1,
            "gate_random_seed": "relationship-action-random-control-v1",
            "owner_reset_each_root": True,
            "single_global_gate_carried_across_roots": True,
            "preaction_trace_fsync_before_environment_settle": True,
            "environment_scope_created": False,
            "model_output_count": 0,
            "cuda_execution_count": 0,
        }
    )
    authorization = RelationshipProductPulseAuthorization(
        authorization_id=(
            "relationship-product-horizon-theta0-calibration:"
            f"{dependencies.protocol.protocol_id}"
        ),
        allowed_policy_artifact_id="relationship-action-gate-zero-init",
        allowed_policy_artifact_version=1,
    )
    gate_checkpoint: RelationshipActionGateCheckpoint | None = None
    initial_checkpoint = RelationshipActionGate().export_checkpoint()
    environment_scope: _EnvironmentScope | None = None
    credit_ids: list[str] = []
    previous_credit_timestamp = -1

    for root_index, subject in enumerate(dependencies.public_view.subjects):
        owner_persistence = None
        sink.append(
            {
                "schema_version": THETA0_TRACE_SCHEMA_VERSION,
                "record_type": "root_start",
                "root_sequence_index": root_index,
                "subject_scope": subject.subject_scope,
                "world_clone_id": subject.world_clone_id,
                "owner_reset": True,
                "gate_checkpoint_content_sha256": (
                    initial_checkpoint.content_sha256
                    if gate_checkpoint is None
                    else gate_checkpoint.content_sha256
                ),
                "gate_update_count": 0 if gate_checkpoint is None else gate_checkpoint.update_count,
            }
        )
        for onboarding in subject.onboarding_sessions:
            join_row = join_by_session[onboarding.session_id]
            appended = await append_relationship_product_onboarding(
                owner_persistence_snapshot=owner_persistence,
                onboarding=RelationshipProductOnboardingInput(
                    session_id=onboarding.session_id,
                    session_index=onboarding.session_index,
                    turn_index=onboarding.session_index,
                    public_observation=onboarding.user_utterance,
                    action_id=onboarding.assistant_action_id,
                    observed_outcome_id=onboarding.observed_outcome_id,
                    reaction_summary=onboarding.rendered_user_reaction,
                    evidence_ref=(
                        "public-onboarding:"
                        f"{sha256_json(onboarding.to_sut_payload())}"
                    ),
                ),
            )
            owner_persistence = appended.owner_persistence_snapshot
            sink.append(
                {
                    "schema_version": THETA0_TRACE_SCHEMA_VERSION,
                    "record_type": "onboarding",
                    "root_sequence_index": root_index,
                    "session_index": onboarding.session_index,
                    "session_id": onboarding.session_id,
                    "join_row_id": join_row["join_row_id"],
                    "owner_persistence_sha256": (
                        social_record_store_persistence_sha256(owner_persistence)
                    ),
                }
            )
        if owner_persistence is None:
            raise RuntimeError("theta0 calibration root has no onboarding state")

        for decision in subject.decision_sessions:
            global_sequence = root_index * 24 + decision.decision_index
            action_turn = 4 + decision.decision_index * 2
            credit_timestamp = root_index * 52 + 5 + 2 * decision.decision_index
            if credit_timestamp <= previous_credit_timestamp:
                raise RuntimeError("theta0 credit timestamp is not strictly increasing")
            previous_credit_timestamp = credit_timestamp
            join_row = join_by_session[decision.session_id]
            owner_before_sha = social_record_store_persistence_sha256(
                owner_persistence
            )
            request = RelationshipProductPreActionRequest(
                session_id=decision.session_id,
                forecast_request=PreferenceActionForecastRequest(
                    decision_id=decision.decision_id,
                    interlocutor_id=_INTERLOCUTOR_ID,
                    current_observation=decision.current_input,
                    observation_ref=(
                        "public-decision:"
                        f"{sha256_json(decision.to_sut_payload())}"
                    ),
                    candidate_action_ids=tuple(
                        action.value for action in RELATIONSHIP_ACTIONS
                    ),
                    outcome_ids=tuple(outcome.value for outcome in RELATIONSHIP_OUTCOMES),
                    turn_index=action_turn,
                    session_scope=subject.subject_scope,
                ),
                outcome_turn_index=action_turn + 1,
            )
            preaction = await prepare_relationship_product_preaction(
                request=request,
                owner_persistence_snapshot=owner_persistence,
                gate_checkpoint=gate_checkpoint,
                forecast_runtime=dependencies.forecast_runtime,
                gate_mode=RelationshipActionGateMode.LEARNED,
                authorization=authorization,
                substrate_snapshot=_placeholder_substrate(),
            )
            temporal = preaction.temporal_snapshot.value
            if temporal.action_advisory_status is not TemporalActionAdvisoryStatus.APPLIED:
                raise RuntimeError("theta0 calibration temporal action was not applied")
            selected_action_id = temporal.active_abstract_action
            if selected_action_id != preaction.gate_decision.selected_action_id:
                raise RuntimeError("temporal action differs from frozen gate decision")
            if len(preaction.gate_checkpoint_before.pending_decisions) != 0:
                raise RuntimeError("theta0 preaction began with pending gate state")
            if len(preaction.gate_checkpoint_after.pending_decisions) != 1:
                raise RuntimeError("theta0 preaction did not create one pending decision")
            if preaction.gate_checkpoint_after.update_count != global_sequence:
                raise RuntimeError("theta0 preaction global gate update count drifted")
            preaction_owner_sha = social_record_store_persistence_sha256(
                preaction.owner_persistence_snapshot
            )
            sink.append(
                {
                    "schema_version": THETA0_TRACE_SCHEMA_VERSION,
                    "record_type": "preaction",
                    "global_sequence_index": global_sequence,
                    "root_sequence_index": root_index,
                    "decision_index": decision.decision_index,
                    "session_id": decision.session_id,
                    "decision_id": decision.decision_id,
                    "join_row_id": join_row["join_row_id"],
                    "owner_prestate_sha256": owner_before_sha,
                    "owner_preaction_persistence_sha256": preaction_owner_sha,
                    "forecast": preference_action_forecast_to_payload(
                        preaction.forecast
                    ),
                    "forecast_sha256": sha256_json(
                        preference_action_forecast_to_payload(preaction.forecast)
                    ),
                    "gate_decision": preaction.gate_decision.to_payload(),
                    "gate_checkpoint_before_content_sha256": (
                        preaction.gate_checkpoint_before.content_sha256
                    ),
                    "gate_checkpoint_after_content_sha256": (
                        preaction.gate_checkpoint_after.content_sha256
                    ),
                    "gate_update_count_before": (
                        preaction.gate_checkpoint_before.update_count
                    ),
                    "gate_update_count_after_preaction": (
                        preaction.gate_checkpoint_after.update_count
                    ),
                    "gate_pending_count_before": 0,
                    "gate_pending_count_after_preaction": 1,
                    "delivered_action_id": selected_action_id,
                    "temporal_action_advisory_status": (
                        temporal.action_advisory_status.value
                    ),
                    "environment_settle_called": False,
                }
            )

            if environment_scope is None:
                environment_scope = _EnvironmentScope(dependencies=dependencies)
            environment_outcome = environment_scope.settle(
                public_subject=subject,
                public_session=decision,
                selected_action_id=selected_action_id,
            )
            settlement_input = _settlement_input(
                subject_scope=subject.subject_scope,
                decision=decision,
                forecast_id=preaction.forecast.forecast_id,
                selected_action_id=selected_action_id,
                environment_outcome=environment_outcome,
                action_turn=action_turn,
                credit_timestamp=credit_timestamp,
            )
            settled = await settle_relationship_product_pulse(
                preaction=preaction,
                settlement_input=settlement_input,
            )
            if settled.gate_update is None or not settled.credit_applied_to_gate:
                raise RuntimeError("theta0 PE credit was not applied exactly once")
            if settled.gate_checkpoint.update_count != global_sequence + 1:
                raise RuntimeError("theta0 gate update count did not advance by one")
            if settled.gate_checkpoint.pending_decisions:
                raise RuntimeError("theta0 settlement left a pending gate decision")
            if (
                settled.credit.prediction_id != preaction.forecast.forecast_id
                or settled.credit.abstract_action_id != selected_action_id
                or settled.credit.timestamp_ms != credit_timestamp
            ):
                raise RuntimeError("theta0 credit/action/forecast lineage drifted")
            if settled.credit.record_id in credit_ids:
                raise RuntimeError("theta0 credit record id was reused")
            credit_ids.append(settled.credit.record_id)
            gate_checkpoint = settled.gate_checkpoint
            owner_persistence = settled.owner_persistence_snapshot
            sink.append(
                {
                    "schema_version": THETA0_TRACE_SCHEMA_VERSION,
                    "record_type": "postaction",
                    "global_sequence_index": global_sequence,
                    "root_sequence_index": root_index,
                    "decision_index": decision.decision_index,
                    "session_id": decision.session_id,
                    "decision_id": decision.decision_id,
                    "environment": {
                        "environment_subject_id": (
                            environment_outcome.environment_subject_id
                        ),
                        "selected_action_id": selected_action_id,
                        "typed_outcome_id": environment_outcome.typed_outcome_id,
                        "rendered_user_reaction_sha256": _sha256_text(
                            environment_outcome.rendered_user_reaction
                        ),
                        "environment_evidence_ref": (
                            environment_outcome.environment_evidence_ref
                        ),
                        "environment_version": (
                            environment_outcome.environment_version
                        ),
                    },
                    "settlement_id": settled.settlement.settlement_id,
                    "social_prediction_error": _social_pe_payload(
                        settled.social_prediction_error_snapshot.value
                    ),
                    "credit": _credit_payload(settled.credit),
                    "gate_update": {
                        "credit_record_id": settled.gate_update.credit_record_id,
                        "forecast_id": settled.gate_update.forecast_id,
                        "selected_action_id": settled.gate_update.selected_action_id,
                        "credit_value_hex": settled.gate_update.credit_value.hex(),
                        "old_state_sha256": settled.gate_update.old_state_sha256,
                        "new_state_sha256": settled.gate_update.new_state_sha256,
                        "update_count": settled.gate_update.update_count,
                    },
                    "gate_checkpoint_content_sha256": gate_checkpoint.content_sha256,
                    "gate_update_count": gate_checkpoint.update_count,
                    "gate_pending_count": 0,
                    "owner_poststate_sha256": (
                        social_record_store_persistence_sha256(owner_persistence)
                    ),
                    "credit_applied_to_gate": True,
                    "evaluation_or_judge_feedback_received": False,
                }
            )

    if gate_checkpoint is None or environment_scope is None:
        raise RuntimeError("theta0 calibration produced no decisions")
    if (
        gate_checkpoint.update_count != 192
        or len(gate_checkpoint.processed_credit_ids) != 192
        or len(set(credit_ids)) != 192
        or gate_checkpoint.pending_decisions
        or set(gate_checkpoint.processed_credit_ids) != set(credit_ids)
    ):
        raise RuntimeError("theta0 terminal gate counts did not close")
    nonzero = any(
        value != 0.0 for value in (*gate_checkpoint.weights, gate_checkpoint.bias)
    )
    terminal = (
        "development_theta0_materialized_effect_not_tested"
        if nonzero
        else "calibration_completed_no_nonzero_theta0"
    )
    root_mapping = environment_scope.root_mapping
    sink.append(
        {
            "schema_version": THETA0_TRACE_SCHEMA_VERSION,
            "record_type": "terminal",
            "decision_count": 192,
            "credit_count": len(credit_ids),
            "unique_credit_count": len(set(credit_ids)),
            "credit_ids_sha256": sha256_json(credit_ids),
            "root_mapping": list(root_mapping),
            "root_mapping_sha256": sha256_json(list(root_mapping)),
            "final_checkpoint": gate_checkpoint.to_payload(),
            "final_checkpoint_content_sha256": gate_checkpoint.content_sha256,
            "final_parameters_nonzero": nonzero,
            "terminal_status": terminal,
            "model_output_count": 0,
            "cuda_execution_count": 0,
        }
    )
    return _CalibrationReplay(
        final_checkpoint=gate_checkpoint,
        credit_ids=tuple(credit_ids),
        root_mapping=root_mapping,
        terminal_status=terminal,
    )


def _settlement_input(
    *,
    subject_scope: str,
    decision: object,
    forecast_id: str,
    selected_action_id: str,
    environment_outcome: _SafeEnvironmentOutcome,
    action_turn: int,
    credit_timestamp: int,
) -> RelationshipProductSettlementInput:
    evidence_id = f"relationship-product-outcome:{decision.decision_id}"
    external = DialogueExternalOutcomeEvidence(
        evidence_id=evidence_id,
        turn_index=action_turn + 1,
        kind=next(
            item
            for item in RELATIONSHIP_OUTCOMES
            if item.value == environment_outcome.typed_outcome_id
        ),
        source=DialogueExternalOutcomeEvidenceSource.ENVIRONMENT,
        confidence=1.0,
        evidence_ref=environment_outcome.environment_evidence_ref,
        description=environment_outcome.rendered_user_reaction,
        session_scope=subject_scope,
        action_turn_index=action_turn,
        forecast_id=forecast_id,
        decision_id=decision.decision_id,
        action_id=selected_action_id,
    )
    owner = PreferenceActionOutcomeEvidence(
        evidence_id=evidence_id,
        interlocutor_id=_INTERLOCUTOR_ID,
        observation_summary=decision.current_input,
        action_id=selected_action_id,
        observed_outcome_id=environment_outcome.typed_outcome_id,
        reaction_summary=environment_outcome.rendered_user_reaction,
        source_turn=action_turn + 1,
        evidence_refs=(environment_outcome.environment_evidence_ref,),
    )
    return RelationshipProductSettlementInput(
        external_outcome=external,
        owner_outcome_evidence=owner,
        credit_timestamp_ms=credit_timestamp,
        apply_credit_to_gate=True,
    )


def _credit_payload(credit: object) -> Mapping[str, object]:
    return {
        "record_id": credit.record_id,
        "level": credit.level,
        "track": credit.track.value,
        "source_event": credit.source_event,
        "credit_value_hex": credit.credit_value.hex(),
        "context_sha256": _sha256_text(credit.context),
        "timestamp_ms": credit.timestamp_ms,
        "prediction_id": credit.prediction_id,
        "environment_event_id": credit.environment_event_id,
        "environment_outcome_id": credit.environment_outcome_id,
        "segment_id": credit.segment_id,
        "abstract_action_id": credit.abstract_action_id,
        "conditioning_bank_set": list(credit.conditioning_bank_set),
        "conditioning_bank_fingerprints": [
            list(pair) for pair in credit.conditioning_bank_fingerprints
        ],
    }


def _social_pe_payload(snapshot: object) -> Mapping[str, object]:
    return {
        "description": snapshot.description,
        "errors": [
            {
                "error_id": item.error_id,
                "prediction_id": item.prediction_id,
                "kind": item.kind.value,
                "outcome": item.outcome.value,
                "magnitude_hex": item.magnitude.hex(),
                "owner": item.owner,
                "scope_kind": item.scope_kind.value,
                "scope_id": item.scope_id,
                "evidence": list(item.evidence),
            }
            for item in snapshot.errors
        ],
    }


def _placeholder_substrate() -> SubstrateSnapshot:
    return SubstrateSnapshot(
        model_id="relationship-product-horizon-theta0-placeholder",
        is_frozen=True,
        surface_kind=SurfaceKind.PLACEHOLDER,
        token_logits=(),
        feature_surface=(),
        residual_activations=(),
        residual_sequence=(),
        unavailable_fields=(),
        description="development theta0 calibration typed action surface",
    )


def materialize_relationship_product_horizon_theta0_calibration(
    *,
    source_v3_admission_root: pathlib.Path,
    preflight_root: pathlib.Path,
    reader_root: pathlib.Path,
    source_v4_admission_root: pathlib.Path,
    output_dir: pathlib.Path,
    implementation_git_commit: str,
) -> Mapping[str, object]:
    commit = _git_commit(implementation_git_commit)
    root = pathlib.Path(output_dir)
    if root.exists():
        raise FileExistsError(f"theta0 calibration root is create-only: {root}")
    dependencies = _load_dependencies(
        source_v3_admission_root=pathlib.Path(source_v3_admission_root),
        preflight_root=pathlib.Path(preflight_root),
        reader_root=pathlib.Path(reader_root),
        source_v4_admission_root=pathlib.Path(source_v4_admission_root),
    )
    public_join = _build_public_join(dependencies)
    root.mkdir(parents=True, exist_ok=False)
    _write_create_only(root / "protocol.json", dependencies.protocol.raw_bytes)
    _write_create_only(root / "public_join.json", _canonical_bytes(public_join))
    sink = _FsyncTraceSink(root / _TRACE_FILENAME)
    try:
        replay = asyncio.run(
            _run_calibration(
                dependencies=dependencies,
                public_join=public_join,
                sink=sink,
            )
        )
    finally:
        sink.close()
    trace_raw = sink.raw_bytes
    if _read_regular(root / "protocol.json") != dependencies.protocol.raw_bytes:
        raise RuntimeError("theta0 persisted protocol bytes drifted before manifest")
    if _read_regular(root / "public_join.json") != _canonical_bytes(public_join):
        raise RuntimeError("theta0 persisted public join drifted before manifest")
    if _read_regular(root / _TRACE_FILENAME) != trace_raw:
        raise RuntimeError("theta0 persisted trace drifted before manifest")
    trace_artifact_id = (
        "relationship-product-horizon-theta0-calibration-trace-sha256:"
        f"{_sha256_bytes(trace_raw)}"
    )
    theta0: RelationshipActionGateTheta0Artifact | None = None
    if replay.terminal_status == "development_theta0_materialized_effect_not_tested":
        theta0 = RelationshipActionGateTheta0Artifact.create(
            source_checkpoint=replay.final_checkpoint,
            learning_rate=0.25,
            max_abs_parameter=4.0,
            source_batch_artifact_id=trace_artifact_id,
        )
        theta0.validate_source_checkpoint(replay.final_checkpoint)
        cold = RelationshipActionGate.from_theta0(theta0).export_checkpoint()
        if (
            cold.update_count != 0
            or cold.processed_credit_ids
            or cold.pending_decisions
            or cold.weights != replay.final_checkpoint.weights
            or cold.bias != replay.final_checkpoint.bias
        ):
            raise RuntimeError("materialized theta0 did not restore as exact cold state")
        _write_create_only(
            root / "theta0_artifact.json",
            _canonical_bytes(theta0.to_payload()),
        )
        if _read_regular(root / "theta0_artifact.json") != _canonical_bytes(
            theta0.to_payload()
        ):
            raise RuntimeError("theta0 persisted artifact drifted before manifest")
    manifest = _build_manifest(
        root=root,
        dependencies=dependencies,
        public_join=public_join,
        replay=replay,
        trace_artifact_id=trace_artifact_id,
        theta0=theta0,
        implementation_git_commit=commit,
    )
    _write_create_only(root / "manifest.json", _canonical_bytes(manifest))
    return manifest


def validate_relationship_product_horizon_theta0_calibration(
    *,
    source_v3_admission_root: pathlib.Path,
    preflight_root: pathlib.Path,
    reader_root: pathlib.Path,
    source_v4_admission_root: pathlib.Path,
    output_dir: pathlib.Path,
    expected_protocol_id: str,
    expected_artifact_id: str,
) -> Mapping[str, object]:
    external_protocol_id = _digest(expected_protocol_id, "expected_protocol_id")
    external_artifact_id = _digest(expected_artifact_id, "expected_artifact_id")
    root = pathlib.Path(output_dir)
    manifest_raw = _read_regular(root / "manifest.json")
    manifest = _parse_json_bytes(manifest_raw, source="theta0 manifest")
    if manifest_raw != _canonical_bytes(manifest):
        raise ValueError("theta0 manifest must use canonical bytes")
    if manifest["protocol_id"] != external_protocol_id:
        raise ValueError("theta0 external expected protocol identity drifted")
    if manifest["artifact_id"] != external_artifact_id:
        raise ValueError("theta0 external expected artifact identity drifted")
    dependencies = _load_dependencies(
        source_v3_admission_root=pathlib.Path(source_v3_admission_root),
        preflight_root=pathlib.Path(preflight_root),
        reader_root=pathlib.Path(reader_root),
        source_v4_admission_root=pathlib.Path(source_v4_admission_root),
    )
    if dependencies.protocol.protocol_id != external_protocol_id:
        raise ValueError("theta0 packaged protocol identity drifted")
    if _read_regular(root / "protocol.json") != dependencies.protocol.raw_bytes:
        raise ValueError("theta0 persisted protocol bytes drifted")
    public_join = _build_public_join(dependencies)
    if _read_regular(root / "public_join.json") != _canonical_bytes(public_join):
        raise ValueError("theta0 public exact join bytes drifted")
    sink = _MemoryTraceSink()
    replay = asyncio.run(
        _run_calibration(
            dependencies=dependencies,
            public_join=public_join,
            sink=sink,
        )
    )
    trace_raw = _read_regular(root / _TRACE_FILENAME)
    if trace_raw != sink.raw_bytes:
        raise ValueError("theta0 calibration trace bytes drifted")
    trace_artifact_id = (
        "relationship-product-horizon-theta0-calibration-trace-sha256:"
        f"{_sha256_bytes(trace_raw)}"
    )
    theta0: RelationshipActionGateTheta0Artifact | None = None
    if replay.terminal_status == "development_theta0_materialized_effect_not_tested":
        theta0 = RelationshipActionGateTheta0Artifact.create(
            source_checkpoint=replay.final_checkpoint,
            learning_rate=0.25,
            max_abs_parameter=4.0,
            source_batch_artifact_id=trace_artifact_id,
        )
        theta0_raw = _read_regular(root / "theta0_artifact.json")
        if theta0_raw != _canonical_bytes(theta0.to_payload()):
            raise ValueError("theta0 artifact bytes drifted")
        restored = RelationshipActionGateTheta0Artifact.from_payload(
            _parse_json_bytes(theta0_raw, source="theta0 artifact")
        )
        restored.validate_source_checkpoint(replay.final_checkpoint)
        cold = RelationshipActionGate.from_theta0(restored).export_checkpoint()
        if (
            cold.update_count != 0
            or cold.processed_credit_ids
            or cold.pending_decisions
        ):
            raise ValueError("theta0 cold replay state drifted")
    expected_files = _SUCCESS_OUTPUT_FILES if theta0 is not None else _BASE_OUTPUT_FILES
    if _regular_file_inventory(root) != expected_files:
        raise ValueError("theta0 calibration output file inventory drifted")
    expected_manifest = _build_manifest(
        root=root,
        dependencies=dependencies,
        public_join=public_join,
        replay=replay,
        trace_artifact_id=trace_artifact_id,
        theta0=theta0,
        implementation_git_commit=_git_commit(manifest["implementation_git_commit"]),
    )
    if manifest != expected_manifest:
        raise ValueError("theta0 calibration manifest content drifted")
    if manifest_raw != _canonical_bytes(expected_manifest):
        raise ValueError("theta0 calibration manifest bytes drifted")
    if manifest["artifact_id"] != external_artifact_id:
        raise ValueError("theta0 calibration artifact identity drifted")
    return manifest


def _build_manifest(
    *,
    root: pathlib.Path,
    dependencies: _Dependencies,
    public_join: Mapping[str, object],
    replay: _CalibrationReplay,
    trace_artifact_id: str,
    theta0: RelationshipActionGateTheta0Artifact | None,
    implementation_git_commit: str,
) -> Mapping[str, object]:
    paths = ["protocol.json", "public_join.json", _TRACE_FILENAME]
    if theta0 is not None:
        paths.append("theta0_artifact.json")
    files = []
    for relative in paths:
        raw = _read_regular(root / relative)
        files.append(
            {
                "path": relative,
                "raw_bytes": len(raw),
                "raw_sha256": _sha256_bytes(raw),
            }
        )
    success = theta0 is not None
    claims = {
        "development_public_exact_join": True,
        "calibration_completed": True,
        "development_theta0_materialized": success,
        "reader_qualified": False,
        "campaign_execution_authorized": False,
        "formal_evidence_authorized": False,
        "integrated_horizon_authorized": False,
        "appendable_effect": False,
        "readable_effect": False,
        "learnable_effect": False,
        "steerable_effect": False,
        "four_able_complete": False,
        "human_validation_complete": False,
        "production_active": False,
        "model_output_count": 0,
        "cuda_execution_count": 0,
    }
    core = {
        "schema_version": THETA0_MANIFEST_SCHEMA_VERSION,
        "protocol_id": dependencies.protocol.protocol_id,
        "protocol_raw_sha256": dependencies.protocol.raw_sha256,
        "implementation_git_commit": implementation_git_commit,
        "source_v3_admission_artifact_id": dependencies.protocol.payload[
            "source_v3_admission"
        ]["artifact_id"],
        "development_reader_artifact_id": dependencies.protocol.payload[
            "development_reader"
        ]["artifact_id"],
        "public_join_artifact_id": public_join["artifact_id"],
        "calibration_trace_artifact_id": trace_artifact_id,
        "theta0_artifact_id": None if theta0 is None else theta0.artifact_id,
        "final_checkpoint_content_sha256": (
            replay.final_checkpoint.content_sha256
        ),
        "root_count": 8,
        "onboarding_count": 32,
        "decision_count": 192,
        "gate_update_count": replay.final_checkpoint.update_count,
        "processed_credit_id_count": len(
            replay.final_checkpoint.processed_credit_ids
        ),
        "unique_credit_id_count": len(set(replay.credit_ids)),
        "pending_decision_count": len(replay.final_checkpoint.pending_decisions),
        "root_mapping_sha256": sha256_json(list(replay.root_mapping)),
        "preaction_trace_fsync_before_environment_settle": True,
        "challenge_label_file_read_count": 0,
        "group_split_file_read_count": 0,
        "admitted_sealed_file_runtime_read_count": 0,
        "files": files,
        "status": replay.terminal_status,
        "claims": claims,
        "claim_boundary": dependencies.protocol.payload["claim_boundary"],
    }
    return {"artifact_id": sha256_json(core), **core}


def _canonical_bytes(payload: object) -> bytes:
    return (canonical_json(payload) + "\n").encode("utf-8")


def _parse_json_bytes(raw: bytes, *, source: str) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{source} contains duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(token: str) -> object:
        raise ValueError(f"{source} contains non-finite JSON constant: {token}")

    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source} must be canonical UTF-8 JSON") from exc
    return _mapping(parsed, source)


def _write_create_only(path: pathlib.Path, raw: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _read_regular(path: pathlib.Path) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"theta0 calibration file must be regular: {path}")
    return path.read_bytes()


def _require_raw_sha(path: pathlib.Path, expected: object, source: str) -> bytes:
    raw = _read_regular(path)
    if _sha256_bytes(raw) != _digest(expected, f"{source} raw_sha256"):
        raise ValueError(f"{source} raw bytes drifted")
    return raw


def _regular_file_inventory(root: pathlib.Path) -> frozenset[str]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("theta0 calibration root must be a regular directory")
    files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError("theta0 calibration root contains non-regular entry")
        files.add(path.relative_to(root).as_posix())
    return frozenset(files)


def _file_entry_map(
    value: object,
    source: str,
) -> Mapping[str, Mapping[str, object]]:
    entries = _list(value, source)
    result: dict[str, Mapping[str, object]] = {}
    for index, raw in enumerate(entries):
        entry = _mapping(raw, f"{source}[{index}]")
        _exact_keys(
            entry,
            {"path", "raw_bytes", "raw_sha256"},
            f"{source}[{index}]",
        )
        relative = _text(entry["path"], f"{source}[{index}].path")
        raw_bytes = entry["raw_bytes"]
        if isinstance(raw_bytes, bool) or not isinstance(raw_bytes, int) or raw_bytes < 1:
            raise ValueError(f"{source}[{index}].raw_bytes must be a positive integer")
        _digest(entry["raw_sha256"], f"{source}[{index}].raw_sha256")
        if relative in result:
            raise ValueError(f"{source} contains duplicate path: {relative}")
        result[relative] = entry
    return result


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mapping(value: object, source: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{source} must be an object with string keys")
    return value


def _list(value: object, source: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{source} must be an array")
    return value


def _exact_keys(
    payload: Mapping[str, object],
    expected: set[str],
    source: str,
) -> None:
    missing = sorted(expected - set(payload))
    extra = sorted(set(payload) - expected)
    if missing or extra:
        raise ValueError(
            f"{source} fields do not match schema; missing={missing}, extra={extra}"
        )


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _boolean(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _digest(value: object, field_name: str) -> str:
    text = _text(value, field_name)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return text


def _git_commit(value: object) -> str:
    text = _text(value, "implementation_git_commit")
    if _GIT_COMMIT.fullmatch(text) is None:
        raise ValueError("implementation_git_commit must be 40 lowercase hex")
    return text


__all__ = [
    "THETA0_CALIBRATION_PROTOCOL_SCHEMA_VERSION",
    "THETA0_MANIFEST_SCHEMA_VERSION",
    "THETA0_PUBLIC_JOIN_SCHEMA_VERSION",
    "THETA0_TRACE_SCHEMA_VERSION",
    "RelationshipProductHorizonTheta0CalibrationProtocol",
    "load_relationship_product_horizon_theta0_calibration_protocol",
    "materialize_relationship_product_horizon_theta0_calibration",
    "relationship_product_horizon_theta0_calibration_protocol_path",
    "validate_relationship_product_horizon_theta0_calibration",
]
