"""Relationship Lab P1j: one-shot v4 consumer qualification.

P1j consumes exactly one P1i frozen ordinary-Qwen consumer and the P1h
evaluator-only v4 split.  It freezes the complete v4 context surface before
the first Qwen output, persists each readout before attaching evaluator truth,
and publishes one terminal development verdict.  Qualification feedback never
revises the consumer or enters Volvence memory, PE, credit, reward, controller,
steering, or runtime state.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable

from lifeform_domain_emogpt.lab import (
    RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME,
    RELATIONSHIP_TRANSFER_V4_PACKAGE_NAME,
    RelationshipAction,
    RelationshipConsumerSplitBundle,
    RelationshipDatasetSplit,
    RelationshipTransferDataset,
    canonical_json,
    sha256_json,
)
from lifeform_evolution.relationship_lab_contexts import (
    RelationshipP1Arm,
    RelationshipP1ContextBundle,
)
from lifeform_evolution.relationship_lab_packet1 import (
    ContextualRelationshipActionPolicy,
    RelationshipP1Decision,
    relationship_p1_completion_to_decision,
)
from lifeform_evolution.relationship_lab_packet1b import (
    RelationshipEvidenceReadout,
    parse_relationship_evidence_scores,
)
from lifeform_evolution.relationship_lab_packet1i import (
    RelationshipP1iFrozenConsumerProtocol,
    load_relationship_p1i_candidate_prompt,
    relationship_p1i_decision_from_record_payload,
    relationship_p1i_decision_record_payload,
    relationship_p1i_readout_completion,
    relationship_p1i_readout_from_record_payload,
    relationship_p1i_readout_record_payload,
    render_relationship_p1i_candidate_request,
)


RELATIONSHIP_P1J_PROTOCOL_SCHEMA_VERSION = "relationship-p1j-protocol.v1"
RELATIONSHIP_P1J_CHECKPOINT_SCHEMA_VERSION = "relationship-p1j-checkpoint.v1"
RELATIONSHIP_P1J_REPORT_SCHEMA_VERSION = "relationship-p1j-report.v1"
RELATIONSHIP_P1J_PREPARED_NEXT_ACTION = "execute_frozen_one_shot_v4_qualification"

_EVALUATED_ARMS = (
    RelationshipP1Arm.PROMPT_STEELMAN,
    RelationshipP1Arm.RAG_STEELMAN,
    RelationshipP1Arm.STRUCTURED_STATE,
)
_HEX_DIGITS = frozenset("0123456789abcdef")
_PROTOCOL_CLAIM_BOUNDARY = (
    "P1j freezes the complete development-only v4 model-input surface for one "
    "already-frozen P1i ordinary-Qwen consumer before the first v4 Qwen output. "
    "It does not revise the consumer, open formal hidden test or P2, write "
    "Volvence learning/control state, prove Volvence advantage, human "
    "readability, product value, or a complete four-capability claim."
)
_REPORT_CLAIM_BOUNDARY = (
    "P1j reports one terminal development qualification result for the frozen "
    "ordinary-Qwen consumer on the process-isolated v4 split. The result cannot "
    "revise that consumer and is not formal held-out evidence, Volvence "
    "advantage, Readable/Learnable/Steerable evidence, product evidence, or a "
    "complete four-capability claim."
)


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_sha256(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _HEX_DIGITS for char in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase sha256 digest")
    return value


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    return float(value)


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be boolean")
    return value


def _require_timestamp(value: object, field_name: str) -> str:
    text = _require_text(value, field_name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return text


def _parsed_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _require_object(
    value: object,
    expected_fields: set[str],
    *,
    field_name: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError(f"{field_name} fields do not match schema")
    return value


def _atomic_write_text(path: pathlib.Path, content: str) -> None:
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        handle.flush()
        temporary = pathlib.Path(handle.name)
    temporary.replace(target)


@dataclass(frozen=True)
class RelationshipP1jRecordKey:
    arm: RelationshipP1Arm
    scene_id: str
    mirror_pair_id: str
    seed: int

    def __post_init__(self) -> None:
        if self.arm not in _EVALUATED_ARMS:
            raise ValueError("P1j record arm is not qualified")
        if not self.scene_id.strip() or not self.mirror_pair_id.strip():
            raise ValueError("P1j record identity is empty")
        if self.seed < 0:
            raise ValueError("P1j record seed must be non-negative")

    def to_payload(self) -> dict[str, object]:
        return {
            "arm": self.arm.value,
            "scene_id": self.scene_id,
            "mirror_pair_id": self.mirror_pair_id,
            "seed": self.seed,
        }

    @classmethod
    def from_payload(cls, value: object) -> "RelationshipP1jRecordKey":
        raw = _require_object(
            value,
            {"arm", "scene_id", "mirror_pair_id", "seed"},
            field_name="P1j record key",
        )
        return cls(
            arm=RelationshipP1Arm(_require_text(raw["arm"], "P1j record arm")),
            scene_id=_require_text(raw["scene_id"], "P1j record scene"),
            mirror_pair_id=_require_text(
                raw["mirror_pair_id"], "P1j record mirror pair"
            ),
            seed=_require_int(raw["seed"], "P1j record seed"),
        )


def relationship_p1j_record_plan(
    *,
    dataset: RelationshipTransferDataset,
    seed_schedule: tuple[int, ...],
) -> tuple[RelationshipP1jRecordKey, ...]:
    if dataset.package_name != RELATIONSHIP_TRANSFER_V4_PACKAGE_NAME:
        raise ValueError("P1j record plan requires relationship_transfer_v4")
    plan: list[RelationshipP1jRecordKey] = []
    for mirror_pair_id, members in dataset.mirrored_pairs():
        for seed in seed_schedule:
            for arm in _EVALUATED_ARMS:
                for observation, _dynamic in members:
                    plan.append(
                        RelationshipP1jRecordKey(
                            arm=arm,
                            scene_id=observation.scene_id,
                            mirror_pair_id=mirror_pair_id,
                            seed=seed,
                        )
                    )
    frozen = tuple(plan)
    identities = tuple(
        (item.arm, item.scene_id, item.mirror_pair_id, item.seed) for item in frozen
    )
    if not frozen or len(set(identities)) != len(frozen):
        raise ValueError("P1j record plan must be non-empty and unique")
    return frozen


def relationship_p1j_record_plan_sha256(
    plan: tuple[RelationshipP1jRecordKey, ...],
) -> str:
    return sha256_json(tuple(item.to_payload() for item in plan))


def relationship_p1j_context_surface_sha256(
    *,
    bundle: RelationshipP1ContextBundle,
) -> str:
    rows = tuple(
        sorted(
            (
                context.scene_id,
                context.background_depth,
                context.arm.value,
                context.context_sha256,
            )
            for context in bundle.contexts
        )
    )
    if not rows:
        raise ValueError("P1j qualification context surface must be non-empty")
    return sha256_json(
        {
            "dataset_fingerprint": bundle.dataset_fingerprint,
            "background_depths": list(bundle.background_depths),
            "background_templates_sha256": bundle.background_templates_sha256,
            "rag_config_sha256": bundle.rag_config_sha256,
            "dataset_role": "unseen_qualification_only",
            "contexts": rows,
        }
    )


@dataclass(frozen=True)
class RelationshipP1jQualificationProtocol:
    frozen_at_iso: str
    consumer_protocol_id: str
    calibration_report_artifact_id: str
    consumer_split_contract_id: str
    qualification_package_name: str
    qualification_dataset_fingerprint: str
    context_manifest_artifact_id: str
    qualification_context_surface_sha256: str
    background_template_package_name: str
    background_templates_sha256: str
    rag_config_sha256: str
    selected_candidate_id: str
    selected_candidate_artifact_id: str
    selected_pipeline_sha256: str
    record_plan_sha256: str
    qualification_observation_count: int
    planned_qwen_output_count: int
    evaluated_arms: tuple[str, ...]
    seed_schedule: tuple[int, ...]
    consumer_frozen_before_qualification_inputs: bool
    qualification_public_inputs_materialized_before_freeze: int
    qualification_qwen_outputs_observed_before_freeze: int
    one_shot: bool
    qualification_feedback_to_consumer: bool
    consumer_revision_after_qualification: bool
    evaluation_feedback_to_pe_credit_reward_or_steering: bool
    formal_hidden_test_opened: bool
    p2_enabled: bool
    next_action: str
    claim_boundary: str
    schema_version: str = RELATIONSHIP_P1J_PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1J_PROTOCOL_SCHEMA_VERSION:
            raise ValueError("P1j protocol schema_version mismatch")
        _require_timestamp(self.frozen_at_iso, "P1j protocol frozen_at_iso")
        for field_name, value in (
            ("consumer_protocol_id", self.consumer_protocol_id),
            ("calibration_report_artifact_id", self.calibration_report_artifact_id),
            ("consumer_split_contract_id", self.consumer_split_contract_id),
            (
                "qualification_dataset_fingerprint",
                self.qualification_dataset_fingerprint,
            ),
            ("context_manifest_artifact_id", self.context_manifest_artifact_id),
            (
                "qualification_context_surface_sha256",
                self.qualification_context_surface_sha256,
            ),
            ("background_templates_sha256", self.background_templates_sha256),
            ("rag_config_sha256", self.rag_config_sha256),
            ("selected_candidate_artifact_id", self.selected_candidate_artifact_id),
            ("selected_pipeline_sha256", self.selected_pipeline_sha256),
            ("record_plan_sha256", self.record_plan_sha256),
        ):
            _require_sha256(value, f"P1j protocol {field_name}")
        if self.qualification_package_name != RELATIONSHIP_TRANSFER_V4_PACKAGE_NAME:
            raise ValueError("P1j protocol qualification package mismatch")
        if self.background_template_package_name != (
            RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME
        ):
            raise ValueError("P1j must reuse the consumer-frozen background templates")
        if self.evaluated_arms != tuple(arm.value for arm in _EVALUATED_ARMS):
            raise ValueError("P1j evaluated arms mismatch")
        if not self.seed_schedule or len(set(self.seed_schedule)) != len(
            self.seed_schedule
        ):
            raise ValueError("P1j seed schedule must be non-empty and unique")
        if self.qualification_observation_count != 24:
            raise ValueError("P1j requires exactly 24 v4 observations")
        if self.planned_qwen_output_count != (
            self.qualification_observation_count
            * len(self.evaluated_arms)
            * len(self.seed_schedule)
        ):
            raise ValueError("P1j planned Qwen output count mismatch")
        if (
            not self.consumer_frozen_before_qualification_inputs
            or self.qualification_public_inputs_materialized_before_freeze
            != self.qualification_observation_count
            or self.qualification_qwen_outputs_observed_before_freeze != 0
            or not self.one_shot
        ):
            raise ValueError("P1j one-shot freeze order is invalid")
        if any(
            (
                self.qualification_feedback_to_consumer,
                self.consumer_revision_after_qualification,
                self.evaluation_feedback_to_pe_credit_reward_or_steering,
                self.formal_hidden_test_opened,
                self.p2_enabled,
            )
        ):
            raise ValueError("P1j cannot open feedback, formal hidden test, or P2")
        if self.next_action != RELATIONSHIP_P1J_PREPARED_NEXT_ACTION:
            raise ValueError("P1j prepared next action mismatch")
        if self.claim_boundary != _PROTOCOL_CLAIM_BOUNDARY:
            raise ValueError("P1j protocol claim boundary mismatch")

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "frozen_at_iso": self.frozen_at_iso,
            "source_lineage": {
                "consumer_protocol_id": self.consumer_protocol_id,
                "calibration_report_artifact_id": (
                    self.calibration_report_artifact_id
                ),
                "consumer_split_contract_id": self.consumer_split_contract_id,
                "qualification_package_name": self.qualification_package_name,
                "qualification_dataset_fingerprint": (
                    self.qualification_dataset_fingerprint
                ),
            },
            "frozen_context": {
                "context_manifest_artifact_id": self.context_manifest_artifact_id,
                "qualification_context_surface_sha256": (
                    self.qualification_context_surface_sha256
                ),
                "background_template_package_name": (
                    self.background_template_package_name
                ),
                "background_templates_sha256": self.background_templates_sha256,
                "rag_config_sha256": self.rag_config_sha256,
            },
            "frozen_consumer": {
                "selected_candidate_id": self.selected_candidate_id,
                "selected_candidate_artifact_id": (
                    self.selected_candidate_artifact_id
                ),
                "selected_pipeline_sha256": self.selected_pipeline_sha256,
            },
            "execution_plan": {
                "record_plan_sha256": self.record_plan_sha256,
                "qualification_observation_count": (
                    self.qualification_observation_count
                ),
                "planned_qwen_output_count": self.planned_qwen_output_count,
                "evaluated_arms": list(self.evaluated_arms),
                "seed_schedule": list(self.seed_schedule),
            },
            "experiment_guards": {
                "consumer_frozen_before_qualification_inputs": (
                    self.consumer_frozen_before_qualification_inputs
                ),
                "qualification_public_inputs_materialized_before_freeze": (
                    self.qualification_public_inputs_materialized_before_freeze
                ),
                "qualification_qwen_outputs_observed_before_freeze": (
                    self.qualification_qwen_outputs_observed_before_freeze
                ),
                "one_shot": self.one_shot,
                "qualification_feedback_to_consumer": (
                    self.qualification_feedback_to_consumer
                ),
                "consumer_revision_after_qualification": (
                    self.consumer_revision_after_qualification
                ),
                "evaluation_feedback_to_pe_credit_reward_or_steering": (
                    self.evaluation_feedback_to_pe_credit_reward_or_steering
                ),
                "formal_hidden_test_opened": self.formal_hidden_test_opened,
                "p2_enabled": self.p2_enabled,
            },
            "next_action": self.next_action,
            "claim_boundary": self.claim_boundary,
        }

    @property
    def protocol_id(self) -> str:
        return sha256_json(self._canonical_payload())

    def to_json(self) -> str:
        payload = self._canonical_payload()
        payload["protocol_id"] = self.protocol_id
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, encoded: str) -> "RelationshipP1jQualificationProtocol":
        try:
            raw_value = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ValueError("P1j protocol is not valid JSON") from exc
        raw = _require_object(
            raw_value,
            {
                "schema_version",
                "frozen_at_iso",
                "source_lineage",
                "frozen_context",
                "frozen_consumer",
                "execution_plan",
                "experiment_guards",
                "next_action",
                "claim_boundary",
                "protocol_id",
            },
            field_name="P1j protocol",
        )
        source = _require_object(
            raw["source_lineage"],
            {
                "consumer_protocol_id",
                "calibration_report_artifact_id",
                "consumer_split_contract_id",
                "qualification_package_name",
                "qualification_dataset_fingerprint",
            },
            field_name="P1j source lineage",
        )
        context = _require_object(
            raw["frozen_context"],
            {
                "context_manifest_artifact_id",
                "qualification_context_surface_sha256",
                "background_template_package_name",
                "background_templates_sha256",
                "rag_config_sha256",
            },
            field_name="P1j frozen context",
        )
        consumer = _require_object(
            raw["frozen_consumer"],
            {
                "selected_candidate_id",
                "selected_candidate_artifact_id",
                "selected_pipeline_sha256",
            },
            field_name="P1j frozen consumer",
        )
        execution = _require_object(
            raw["execution_plan"],
            {
                "record_plan_sha256",
                "qualification_observation_count",
                "planned_qwen_output_count",
                "evaluated_arms",
                "seed_schedule",
            },
            field_name="P1j execution plan",
        )
        guards = _require_object(
            raw["experiment_guards"],
            {
                "consumer_frozen_before_qualification_inputs",
                "qualification_public_inputs_materialized_before_freeze",
                "qualification_qwen_outputs_observed_before_freeze",
                "one_shot",
                "qualification_feedback_to_consumer",
                "consumer_revision_after_qualification",
                "evaluation_feedback_to_pe_credit_reward_or_steering",
                "formal_hidden_test_opened",
                "p2_enabled",
            },
            field_name="P1j experiment guards",
        )
        arms = execution["evaluated_arms"]
        seeds = execution["seed_schedule"]
        if not isinstance(arms, list) or any(not isinstance(item, str) for item in arms):
            raise ValueError("P1j evaluated arms must be a string array")
        if not isinstance(seeds, list) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in seeds
        ):
            raise ValueError("P1j seed schedule must be an integer array")
        protocol = cls(
            schema_version=_require_text(raw["schema_version"], "P1j schema"),
            frozen_at_iso=_require_timestamp(
                raw["frozen_at_iso"], "P1j frozen timestamp"
            ),
            consumer_protocol_id=_require_sha256(
                source["consumer_protocol_id"], "P1j consumer protocol"
            ),
            calibration_report_artifact_id=_require_sha256(
                source["calibration_report_artifact_id"],
                "P1j calibration report",
            ),
            consumer_split_contract_id=_require_sha256(
                source["consumer_split_contract_id"], "P1j split contract"
            ),
            qualification_package_name=_require_text(
                source["qualification_package_name"], "P1j package"
            ),
            qualification_dataset_fingerprint=_require_sha256(
                source["qualification_dataset_fingerprint"],
                "P1j qualification fingerprint",
            ),
            context_manifest_artifact_id=_require_sha256(
                context["context_manifest_artifact_id"], "P1j context manifest"
            ),
            qualification_context_surface_sha256=_require_sha256(
                context["qualification_context_surface_sha256"],
                "P1j context surface",
            ),
            background_template_package_name=_require_text(
                context["background_template_package_name"],
                "P1j template package",
            ),
            background_templates_sha256=_require_sha256(
                context["background_templates_sha256"], "P1j templates"
            ),
            rag_config_sha256=_require_sha256(
                context["rag_config_sha256"], "P1j RAG config"
            ),
            selected_candidate_id=_require_text(
                consumer["selected_candidate_id"], "P1j candidate"
            ),
            selected_candidate_artifact_id=_require_sha256(
                consumer["selected_candidate_artifact_id"],
                "P1j candidate artifact",
            ),
            selected_pipeline_sha256=_require_sha256(
                consumer["selected_pipeline_sha256"], "P1j pipeline"
            ),
            record_plan_sha256=_require_sha256(
                execution["record_plan_sha256"], "P1j record plan"
            ),
            qualification_observation_count=_require_int(
                execution["qualification_observation_count"],
                "P1j observation count",
            ),
            planned_qwen_output_count=_require_int(
                execution["planned_qwen_output_count"], "P1j output count"
            ),
            evaluated_arms=tuple(arms),
            seed_schedule=tuple(seeds),
            consumer_frozen_before_qualification_inputs=_require_bool(
                guards["consumer_frozen_before_qualification_inputs"],
                "P1j consumer freeze guard",
            ),
            qualification_public_inputs_materialized_before_freeze=_require_int(
                guards["qualification_public_inputs_materialized_before_freeze"],
                "P1j materialized inputs",
            ),
            qualification_qwen_outputs_observed_before_freeze=_require_int(
                guards["qualification_qwen_outputs_observed_before_freeze"],
                "P1j outputs before freeze",
            ),
            one_shot=_require_bool(guards["one_shot"], "P1j one-shot guard"),
            qualification_feedback_to_consumer=_require_bool(
                guards["qualification_feedback_to_consumer"],
                "P1j consumer feedback",
            ),
            consumer_revision_after_qualification=_require_bool(
                guards["consumer_revision_after_qualification"],
                "P1j consumer revision",
            ),
            evaluation_feedback_to_pe_credit_reward_or_steering=_require_bool(
                guards["evaluation_feedback_to_pe_credit_reward_or_steering"],
                "P1j learning feedback",
            ),
            formal_hidden_test_opened=_require_bool(
                guards["formal_hidden_test_opened"], "P1j formal guard"
            ),
            p2_enabled=_require_bool(guards["p2_enabled"], "P1j P2 guard"),
            next_action=_require_text(raw["next_action"], "P1j next action"),
            claim_boundary=_require_text(
                raw["claim_boundary"], "P1j claim boundary"
            ),
        )
        if protocol.protocol_id != _require_sha256(
            raw["protocol_id"], "P1j protocol id"
        ):
            raise ValueError("P1j protocol id mismatch")
        return protocol


def freeze_relationship_p1j_protocol(
    *,
    consumer: RelationshipP1iFrozenConsumerProtocol,
    split_bundle: RelationshipConsumerSplitBundle,
    contexts: RelationshipP1ContextBundle,
    context_manifest_artifact_id: str,
    frozen_at_iso: str | None = None,
) -> RelationshipP1jQualificationProtocol:
    dataset = split_bundle.qualification_dataset
    plan = relationship_p1j_record_plan(
        dataset=dataset,
        seed_schedule=consumer.seed_schedule,
    )
    protocol = RelationshipP1jQualificationProtocol(
        frozen_at_iso=(
            frozen_at_iso
            or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        ),
        consumer_protocol_id=consumer.protocol_id,
        calibration_report_artifact_id=consumer.calibration_report_artifact_id,
        consumer_split_contract_id=split_bundle.contract.contract_sha256,
        qualification_package_name=dataset.package_name,
        qualification_dataset_fingerprint=dataset.dataset_fingerprint,
        context_manifest_artifact_id=context_manifest_artifact_id,
        qualification_context_surface_sha256=(
            relationship_p1j_context_surface_sha256(bundle=contexts)
        ),
        background_template_package_name=consumer.training_package_name,
        background_templates_sha256=contexts.background_templates_sha256,
        rag_config_sha256=contexts.rag_config_sha256,
        selected_candidate_id=consumer.selected_candidate.candidate_id,
        selected_candidate_artifact_id=consumer.selected_candidate_artifact_id,
        selected_pipeline_sha256=consumer.selected_pipeline_sha256,
        record_plan_sha256=relationship_p1j_record_plan_sha256(plan),
        qualification_observation_count=len(dataset.observations),
        planned_qwen_output_count=len(plan),
        evaluated_arms=tuple(arm.value for arm in _EVALUATED_ARMS),
        seed_schedule=consumer.seed_schedule,
        consumer_frozen_before_qualification_inputs=True,
        qualification_public_inputs_materialized_before_freeze=len(
            dataset.observations
        ),
        qualification_qwen_outputs_observed_before_freeze=0,
        one_shot=True,
        qualification_feedback_to_consumer=False,
        consumer_revision_after_qualification=False,
        evaluation_feedback_to_pe_credit_reward_or_steering=False,
        formal_hidden_test_opened=False,
        p2_enabled=False,
        next_action=RELATIONSHIP_P1J_PREPARED_NEXT_ACTION,
        claim_boundary=_PROTOCOL_CLAIM_BOUNDARY,
    )
    validate_relationship_p1j_protocol_lineage(
        protocol,
        consumer=consumer,
        split_bundle=split_bundle,
        contexts=contexts,
        context_manifest_artifact_id=context_manifest_artifact_id,
    )
    return protocol


def validate_relationship_p1j_protocol_lineage(
    protocol: RelationshipP1jQualificationProtocol,
    *,
    consumer: RelationshipP1iFrozenConsumerProtocol,
    split_bundle: RelationshipConsumerSplitBundle,
    contexts: RelationshipP1ContextBundle,
    context_manifest_artifact_id: str,
) -> None:
    dataset = split_bundle.qualification_dataset
    plan = relationship_p1j_record_plan(
        dataset=dataset,
        seed_schedule=consumer.seed_schedule,
    )
    expected = {
        "consumer_protocol_id": consumer.protocol_id,
        "calibration_report_artifact_id": consumer.calibration_report_artifact_id,
        "consumer_split_contract_id": split_bundle.contract.contract_sha256,
        "qualification_package_name": consumer.qualification_package_name,
        "qualification_dataset_fingerprint": (
            consumer.qualification_dataset_fingerprint
        ),
        "context_manifest_artifact_id": context_manifest_artifact_id,
        "qualification_context_surface_sha256": (
            relationship_p1j_context_surface_sha256(bundle=contexts)
        ),
        "background_template_package_name": consumer.training_package_name,
        "background_templates_sha256": consumer.background_templates_sha256,
        "rag_config_sha256": consumer.rag_config_sha256,
        "selected_candidate_id": consumer.selected_candidate.candidate_id,
        "selected_candidate_artifact_id": consumer.selected_candidate_artifact_id,
        "selected_pipeline_sha256": consumer.selected_pipeline_sha256,
        "record_plan_sha256": relationship_p1j_record_plan_sha256(plan),
        "qualification_observation_count": len(dataset.observations),
        "planned_qwen_output_count": len(plan),
        "evaluated_arms": tuple(arm.value for arm in _EVALUATED_ARMS),
        "seed_schedule": consumer.seed_schedule,
    }
    actual = vars(protocol)
    drift = sorted(name for name, value in expected.items() if actual[name] != value)
    if drift:
        raise ValueError(f"P1j protocol lineage mismatch: {drift}")
    if (
        dataset.dataset_fingerprint != consumer.qualification_dataset_fingerprint
        or dataset.package_name != consumer.qualification_package_name
        or contexts.dataset_fingerprint != dataset.dataset_fingerprint
    ):
        raise ValueError("P1j qualification dataset lineage mismatch")
    if _parsed_timestamp(protocol.frozen_at_iso) < _parsed_timestamp(
        consumer.frozen_at_iso
    ):
        raise ValueError("P1j protocol cannot predate the frozen consumer")
    load_relationship_p1i_candidate_prompt(consumer.selected_candidate)


def write_relationship_p1j_protocol(
    protocol: RelationshipP1jQualificationProtocol,
    path: pathlib.Path,
) -> pathlib.Path:
    target = pathlib.Path(path)
    if target.exists():
        raise FileExistsError(f"P1j protocol already exists: {target}")
    _atomic_write_text(target, protocol.to_json())
    return target


def load_relationship_p1j_protocol(
    path: pathlib.Path,
) -> RelationshipP1jQualificationProtocol:
    target = pathlib.Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"P1j protocol is missing: {target}")
    return RelationshipP1jQualificationProtocol.from_json(
        target.read_text(encoding="utf-8")
    )


@dataclass(frozen=True)
class RelationshipP1jCheckpoint:
    qualification_protocol_id: str
    consumer_protocol_id: str
    dataset_fingerprint: str
    context_manifest_artifact_id: str
    qualification_context_surface_sha256: str
    selected_candidate_id: str
    selected_pipeline_sha256: str
    model_id: str
    weights_sha256: str
    generation_config_sha256: str
    planned_record_keys: tuple[RelationshipP1jRecordKey, ...]
    schema_version: str = RELATIONSHIP_P1J_CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1J_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("P1j checkpoint schema_version mismatch")
        for field_name, value in (
            ("qualification_protocol_id", self.qualification_protocol_id),
            ("consumer_protocol_id", self.consumer_protocol_id),
            ("dataset_fingerprint", self.dataset_fingerprint),
            ("context_manifest_artifact_id", self.context_manifest_artifact_id),
            (
                "qualification_context_surface_sha256",
                self.qualification_context_surface_sha256,
            ),
            ("selected_pipeline_sha256", self.selected_pipeline_sha256),
            ("weights_sha256", self.weights_sha256),
            ("generation_config_sha256", self.generation_config_sha256),
        ):
            _require_sha256(value, f"P1j checkpoint {field_name}")
        identities = tuple(
            (item.arm, item.scene_id, item.seed) for item in self.planned_record_keys
        )
        if not identities or len(set(identities)) != len(identities):
            raise ValueError("P1j checkpoint record plan must be non-empty and unique")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "qualification_protocol_id": self.qualification_protocol_id,
            "consumer_protocol_id": self.consumer_protocol_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "context_manifest_artifact_id": self.context_manifest_artifact_id,
            "qualification_context_surface_sha256": (
                self.qualification_context_surface_sha256
            ),
            "selected_candidate_id": self.selected_candidate_id,
            "selected_pipeline_sha256": self.selected_pipeline_sha256,
            "model_lineage": {
                "model_id": self.model_id,
                "weights_sha256": self.weights_sha256,
                "generation_config_sha256": self.generation_config_sha256,
            },
            "planned_records": [
                {"index": index, **item.to_payload()}
                for index, item in enumerate(self.planned_record_keys)
            ],
        }

    @property
    def checkpoint_id(self) -> str:
        return sha256_json(self._identity_payload())

    def to_json(self) -> str:
        payload = self._identity_payload()
        payload["checkpoint_id"] = self.checkpoint_id
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, encoded: str) -> "RelationshipP1jCheckpoint":
        try:
            raw_value = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ValueError("P1j checkpoint is not valid JSON") from exc
        raw = _require_object(
            raw_value,
            {
                "schema_version",
                "qualification_protocol_id",
                "consumer_protocol_id",
                "dataset_fingerprint",
                "context_manifest_artifact_id",
                "qualification_context_surface_sha256",
                "selected_candidate_id",
                "selected_pipeline_sha256",
                "model_lineage",
                "planned_records",
                "checkpoint_id",
            },
            field_name="P1j checkpoint",
        )
        model = _require_object(
            raw["model_lineage"],
            {"model_id", "weights_sha256", "generation_config_sha256"},
            field_name="P1j checkpoint model lineage",
        )
        records = raw["planned_records"]
        if not isinstance(records, list) or not records:
            raise ValueError("P1j checkpoint records must be non-empty")
        keys: list[RelationshipP1jRecordKey] = []
        for expected_index, value in enumerate(records):
            item = _require_object(
                value,
                {"index", "arm", "scene_id", "mirror_pair_id", "seed"},
                field_name=f"P1j checkpoint record {expected_index}",
            )
            if _require_int(item["index"], "P1j checkpoint index") != (
                expected_index
            ):
                raise ValueError("P1j checkpoint indices must be contiguous")
            keys.append(
                RelationshipP1jRecordKey.from_payload(
                    {name: item[name] for name in item if name != "index"}
                )
            )
        checkpoint = cls(
            schema_version=_require_text(raw["schema_version"], "P1j checkpoint schema"),
            qualification_protocol_id=_require_sha256(
                raw["qualification_protocol_id"], "P1j checkpoint protocol"
            ),
            consumer_protocol_id=_require_sha256(
                raw["consumer_protocol_id"], "P1j checkpoint consumer"
            ),
            dataset_fingerprint=_require_sha256(
                raw["dataset_fingerprint"], "P1j checkpoint dataset"
            ),
            context_manifest_artifact_id=_require_sha256(
                raw["context_manifest_artifact_id"],
                "P1j checkpoint context manifest",
            ),
            qualification_context_surface_sha256=_require_sha256(
                raw["qualification_context_surface_sha256"],
                "P1j checkpoint context surface",
            ),
            selected_candidate_id=_require_text(
                raw["selected_candidate_id"], "P1j checkpoint candidate"
            ),
            selected_pipeline_sha256=_require_sha256(
                raw["selected_pipeline_sha256"], "P1j checkpoint pipeline"
            ),
            model_id=_require_text(model["model_id"], "P1j checkpoint model"),
            weights_sha256=_require_sha256(
                model["weights_sha256"], "P1j checkpoint weights"
            ),
            generation_config_sha256=_require_sha256(
                model["generation_config_sha256"],
                "P1j checkpoint generation config",
            ),
            planned_record_keys=tuple(keys),
        )
        if checkpoint.checkpoint_id != _require_sha256(
            raw["checkpoint_id"], "P1j checkpoint id"
        ):
            raise ValueError("P1j checkpoint id mismatch")
        return checkpoint


def build_relationship_p1j_checkpoint(
    *,
    protocol: RelationshipP1jQualificationProtocol,
    consumer: RelationshipP1iFrozenConsumerProtocol,
    split_bundle: RelationshipConsumerSplitBundle,
) -> RelationshipP1jCheckpoint:
    return RelationshipP1jCheckpoint(
        qualification_protocol_id=protocol.protocol_id,
        consumer_protocol_id=consumer.protocol_id,
        dataset_fingerprint=split_bundle.qualification_dataset.dataset_fingerprint,
        context_manifest_artifact_id=protocol.context_manifest_artifact_id,
        qualification_context_surface_sha256=(
            protocol.qualification_context_surface_sha256
        ),
        selected_candidate_id=consumer.selected_candidate.candidate_id,
        selected_pipeline_sha256=consumer.selected_pipeline_sha256,
        model_id=consumer.model_id,
        weights_sha256=consumer.expected_weights_sha256,
        generation_config_sha256=consumer.expected_generation_config_sha256,
        planned_record_keys=relationship_p1j_record_plan(
            dataset=split_bundle.qualification_dataset,
            seed_schedule=consumer.seed_schedule,
        ),
    )


def write_relationship_p1j_checkpoint(
    *,
    checkpoint: RelationshipP1jCheckpoint,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    path = pathlib.Path(output_dir) / "checkpoint.json"
    if path.exists():
        raise FileExistsError(f"P1j checkpoint already exists: {path}")
    _atomic_write_text(path, checkpoint.to_json())
    return path


def load_relationship_p1j_checkpoint(
    output_dir: pathlib.Path,
) -> RelationshipP1jCheckpoint:
    path = pathlib.Path(output_dir) / "checkpoint.json"
    if not path.is_file():
        raise FileNotFoundError(f"P1j checkpoint is missing: {path}")
    return RelationshipP1jCheckpoint.from_json(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class RelationshipP1jProgress:
    checkpoint: RelationshipP1jCheckpoint
    readouts: tuple[RelationshipEvidenceReadout, ...]
    decisions: tuple[RelationshipP1Decision, ...]

    def __post_init__(self) -> None:
        if len(self.decisions) > len(self.readouts) or (
            len(self.readouts) - len(self.decisions) > 1
        ):
            raise ValueError("P1j progress readout/decision counts are invalid")
        readout_keys = tuple(
            (item.arm, item.scene_id, item.seed) for item in self.readouts
        )
        decision_keys = tuple(
            (item.arm, item.scene_id, item.seed) for item in self.decisions
        )
        planned = tuple(
            (item.arm, item.scene_id, item.seed)
            for item in self.checkpoint.planned_record_keys
        )
        if readout_keys != planned[: len(readout_keys)]:
            raise ValueError("P1j readouts are not a contiguous planned prefix")
        if decision_keys != planned[: len(decision_keys)]:
            raise ValueError("P1j decisions are not a contiguous planned prefix")

    @property
    def is_complete(self) -> bool:
        expected = len(self.checkpoint.planned_record_keys)
        return len(self.readouts) == len(self.decisions) == expected


def _record_path(output_dir: pathlib.Path, index: int, kind: str) -> pathlib.Path:
    return pathlib.Path(output_dir) / "records" / f"{index:04d}.{kind}.json"


def _validate_record_directory_shape(
    output_dir: pathlib.Path,
    *,
    planned_record_count: int,
) -> None:
    records_dir = pathlib.Path(output_dir) / "records"
    if not records_dir.exists():
        return
    if not records_dir.is_dir():
        raise ValueError("P1j records path must be a directory")
    expected_names = {
        f"{index:04d}.{kind}.json"
        for index in range(planned_record_count)
        for kind in ("readout", "decision")
    }
    unexpected = sorted(
        entry.name
        for entry in records_dir.iterdir()
        if entry.name not in expected_names
        or not entry.is_file()
        or entry.is_symlink()
    )
    if unexpected:
        raise ValueError(
            "P1j records directory contains entries outside the frozen plan: "
            + ", ".join(unexpected)
        )


def persist_relationship_p1j_readout(
    *,
    checkpoint: RelationshipP1jCheckpoint,
    output_dir: pathlib.Path,
    index: int,
    readout: RelationshipEvidenceReadout,
) -> pathlib.Path:
    if index < 0 or index >= len(checkpoint.planned_record_keys):
        raise IndexError("P1j readout index is outside the record plan")
    key = checkpoint.planned_record_keys[index]
    if (readout.arm, readout.scene_id, readout.seed) != (
        key.arm,
        key.scene_id,
        key.seed,
    ):
        raise ValueError("P1j readout key diverges from record plan")
    path = _record_path(output_dir, index, "readout")
    if path.exists():
        raise FileExistsError(f"P1j readout already exists: {path}")
    payload = relationship_p1i_readout_record_payload(
        candidate_id=checkpoint.selected_candidate_id,
        readout=readout,
    )
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return path


def persist_relationship_p1j_decision(
    *,
    checkpoint: RelationshipP1jCheckpoint,
    output_dir: pathlib.Path,
    index: int,
    decision: RelationshipP1Decision,
) -> pathlib.Path:
    if index < 0 or index >= len(checkpoint.planned_record_keys):
        raise IndexError("P1j decision index is outside the record plan")
    readout_path = _record_path(output_dir, index, "readout")
    if not readout_path.is_file():
        raise FileNotFoundError("P1j decision cannot precede its durable readout")
    key = checkpoint.planned_record_keys[index]
    if (decision.arm, decision.scene_id, decision.seed) != (
        key.arm,
        key.scene_id,
        key.seed,
    ):
        raise ValueError("P1j decision key diverges from record plan")
    path = _record_path(output_dir, index, "decision")
    if path.exists():
        raise FileExistsError(f"P1j decision already exists: {path}")
    payload = relationship_p1i_decision_record_payload(
        candidate_id=checkpoint.selected_candidate_id,
        decision=decision,
    )
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return path


def load_relationship_p1j_progress(
    output_dir: pathlib.Path,
) -> RelationshipP1jProgress:
    checkpoint = load_relationship_p1j_checkpoint(output_dir)
    _validate_record_directory_shape(
        output_dir,
        planned_record_count=len(checkpoint.planned_record_keys),
    )
    readouts: list[RelationshipEvidenceReadout] = []
    decisions: list[RelationshipP1Decision] = []
    missing_seen = False
    for index in range(len(checkpoint.planned_record_keys)):
        readout_path = _record_path(output_dir, index, "readout")
        decision_path = _record_path(output_dir, index, "decision")
        if readout_path.is_file():
            if missing_seen:
                raise ValueError("P1j readout files contain a non-contiguous gap")
            readouts.append(
                relationship_p1i_readout_from_record_payload(
                    json.loads(readout_path.read_text(encoding="utf-8")),
                    expected_candidate_id=checkpoint.selected_candidate_id,
                )
            )
        else:
            missing_seen = True
        if decision_path.is_file():
            if not readout_path.is_file() or len(decisions) != index:
                raise ValueError("P1j decision files contain a gap or orphan")
            decisions.append(
                relationship_p1i_decision_from_record_payload(
                    json.loads(decision_path.read_text(encoding="utf-8")),
                    expected_candidate_id=checkpoint.selected_candidate_id,
                )
            )
    return RelationshipP1jProgress(
        checkpoint=checkpoint,
        readouts=tuple(readouts),
        decisions=tuple(decisions),
    )


def _expected_decision(
    *,
    readout: RelationshipEvidenceReadout,
    key: RelationshipP1jRecordKey,
    consumer: RelationshipP1iFrozenConsumerProtocol,
    dataset: RelationshipTransferDataset,
) -> RelationshipP1Decision:
    dynamic = dataset.dynamic_for_scene(key.scene_id)
    if dynamic.split is not RelationshipDatasetSplit.HELDOUT:
        raise ValueError("P1j qualification truth must remain on heldout split")
    return relationship_p1_completion_to_decision(
        completion=relationship_p1i_readout_completion(readout),
        arm=key.arm,
        scene_id=key.scene_id,
        mirror_pair_id=key.mirror_pair_id,
        split=dynamic.split,
        seed=key.seed,
        current_input_sha256=readout.current_input_sha256,
        context_sha256=readout.context_sha256,
        arm_prompt_sha256=consumer.selected_pipeline_sha256,
        expected_action_id=dynamic.preferred_action,
        model_id=consumer.model_id,
    )


def validate_relationship_p1j_progress(
    progress: RelationshipP1jProgress,
    *,
    protocol: RelationshipP1jQualificationProtocol,
    consumer: RelationshipP1iFrozenConsumerProtocol,
    split_bundle: RelationshipConsumerSplitBundle,
    contexts: RelationshipP1ContextBundle,
) -> None:
    expected_checkpoint = build_relationship_p1j_checkpoint(
        protocol=protocol,
        consumer=consumer,
        split_bundle=split_bundle,
    )
    if progress.checkpoint != expected_checkpoint:
        raise ValueError("P1j checkpoint diverges from frozen lineage")
    dataset = split_bundle.qualification_dataset
    observations = {item.scene_id: item for item in dataset.observations}
    for index, readout in enumerate(progress.readouts):
        key = progress.checkpoint.planned_record_keys[index]
        observation = observations[key.scene_id]
        context = contexts.context(scene_id=key.scene_id, arm=key.arm)
        expected_input_hash = hashlib.sha256(
            observation.current_input.encode("utf-8")
        ).hexdigest()
        if (
            readout.current_input_sha256 != expected_input_hash
            or readout.context_sha256 != context.context_sha256
            or readout.model_id != consumer.model_id
            or readout.weights_sha256 != consumer.expected_weights_sha256
            or readout.generation_config_sha256
            != consumer.expected_generation_config_sha256
            or readout.prompt_sha256 != consumer.selected_candidate.prompt_sha256
            or readout.request_template_sha256
            != consumer.selected_candidate.request_template_sha256
            or readout.schema_sha256
            != consumer.selected_candidate.readout_schema_sha256
        ):
            raise ValueError("P1j readout diverges from frozen consumer/input lineage")
    for index, decision in enumerate(progress.decisions):
        expected = _expected_decision(
            readout=progress.readouts[index],
            key=progress.checkpoint.planned_record_keys[index],
            consumer=consumer,
            dataset=dataset,
        )
        if decision != expected:
            raise ValueError("P1j decision diverges from evaluator truth")


@dataclass(frozen=True)
class RelationshipP1jExecution:
    readouts: tuple[RelationshipEvidenceReadout, ...]
    decisions: tuple[RelationshipP1Decision, ...]
    new_qwen_outputs: int
    planned_qwen_outputs: int

    @property
    def complete(self) -> bool:
        return len(self.readouts) == len(self.decisions) == self.planned_qwen_outputs


def execute_relationship_p1j_qualification(
    policy: ContextualRelationshipActionPolicy,
    *,
    protocol: RelationshipP1jQualificationProtocol,
    consumer: RelationshipP1iFrozenConsumerProtocol,
    split_bundle: RelationshipConsumerSplitBundle,
    contexts: RelationshipP1ContextBundle,
    existing_progress: RelationshipP1jProgress,
    max_new_readouts: int | None = None,
    readout_observer: Callable[[int, RelationshipEvidenceReadout], None]
    | None = None,
    decision_observer: Callable[[int, RelationshipP1Decision], None]
    | None = None,
) -> RelationshipP1jExecution:
    if max_new_readouts is not None and max_new_readouts < 0:
        raise ValueError("P1j max_new_readouts must be non-negative")
    validate_relationship_p1j_progress(
        existing_progress,
        protocol=protocol,
        consumer=consumer,
        split_bundle=split_bundle,
        contexts=contexts,
    )
    if (
        policy.model_id != consumer.model_id
        or policy.weights_sha256 != consumer.expected_weights_sha256
        or policy.generation_config_sha256
        != consumer.expected_generation_config_sha256
    ):
        raise ValueError("P1j policy diverges from frozen consumer substrate")
    prompt = load_relationship_p1i_candidate_prompt(consumer.selected_candidate)
    dataset = split_bundle.qualification_dataset
    observations = {item.scene_id: item for item in dataset.observations}
    readouts = list(existing_progress.readouts)
    decisions = list(existing_progress.decisions)
    new_outputs = 0
    for index, key in enumerate(existing_progress.checkpoint.planned_record_keys):
        if index < len(readouts):
            readout = readouts[index]
        else:
            if max_new_readouts is not None and new_outputs >= max_new_readouts:
                break
            observation = observations[key.scene_id]
            context = contexts.context(scene_id=key.scene_id, arm=key.arm)
            completion = policy.choose_from_messages(
                messages=(
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": render_relationship_p1i_candidate_request(
                            candidate=consumer.selected_candidate,
                            context_text=context.context_text,
                            current_input=observation.current_input,
                        ),
                    },
                ),
                seed=key.seed,
            )
            stay_score, space_score = parse_relationship_evidence_scores(
                completion.raw_output
            )
            readout = RelationshipEvidenceReadout(
                arm=key.arm,
                scene_id=key.scene_id,
                seed=key.seed,
                current_input_sha256=hashlib.sha256(
                    observation.current_input.encode("utf-8")
                ).hexdigest(),
                context_sha256=context.context_sha256,
                model_id=policy.model_id,
                weights_sha256=policy.weights_sha256,
                generation_config_sha256=policy.generation_config_sha256,
                prompt_sha256=consumer.selected_candidate.prompt_sha256,
                request_template_sha256=(
                    consumer.selected_candidate.request_template_sha256
                ),
                schema_sha256=consumer.selected_candidate.readout_schema_sha256,
                raw_output=completion.raw_output,
                stay_score=stay_score,
                space_score=space_score,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
            )
            if readout_observer is not None:
                readout_observer(index, readout)
            readouts.append(readout)
            new_outputs += 1
        if index < len(decisions):
            continue
        decision = _expected_decision(
            readout=readout,
            key=key,
            consumer=consumer,
            dataset=dataset,
        )
        if decision_observer is not None:
            decision_observer(index, decision)
        decisions.append(decision)
    result = RelationshipP1jExecution(
        readouts=tuple(readouts),
        decisions=tuple(decisions),
        new_qwen_outputs=new_outputs,
        planned_qwen_outputs=len(existing_progress.checkpoint.planned_record_keys),
    )
    validate_relationship_p1j_progress(
        RelationshipP1jProgress(
            checkpoint=existing_progress.checkpoint,
            readouts=result.readouts,
            decisions=result.decisions,
        ),
        protocol=protocol,
        consumer=consumer,
        split_bundle=split_bundle,
        contexts=contexts,
    )
    return result


@dataclass(frozen=True)
class RelationshipP1jArmMetric:
    arm: str
    decisions: int
    valid_decisions: int
    valid_rate: float
    correct_decisions: int
    accuracy: float
    pair_groups: int
    valid_pair_groups: int
    pair_flip_rate: float
    readouts: int
    valid_readouts: int
    prompt_tokens_total: int
    completion_tokens_total: int

    def __post_init__(self) -> None:
        if self.arm not in tuple(arm.value for arm in _EVALUATED_ARMS):
            raise ValueError("P1j metric arm is unknown")
        if self.decisions <= 0 or self.readouts != self.decisions:
            raise ValueError("P1j metric decision/readout count mismatch")
        if not 0.0 <= self.valid_rate <= 1.0 or not 0.0 <= self.accuracy <= 1.0:
            raise ValueError("P1j metric rate is outside [0, 1]")
        if not 0.0 <= self.pair_flip_rate <= 1.0:
            raise ValueError("P1j pair flip rate is outside [0, 1]")

    def to_payload(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "decisions": self.decisions,
            "valid_decisions": self.valid_decisions,
            "valid_rate": self.valid_rate,
            "correct_decisions": self.correct_decisions,
            "accuracy": self.accuracy,
            "pair_groups": self.pair_groups,
            "valid_pair_groups": self.valid_pair_groups,
            "pair_flip_rate": self.pair_flip_rate,
            "readouts": self.readouts,
            "valid_readouts": self.valid_readouts,
            "prompt_tokens_total": self.prompt_tokens_total,
            "completion_tokens_total": self.completion_tokens_total,
        }

    @classmethod
    def from_payload(cls, value: object) -> "RelationshipP1jArmMetric":
        fields = {
            "arm",
            "decisions",
            "valid_decisions",
            "valid_rate",
            "correct_decisions",
            "accuracy",
            "pair_groups",
            "valid_pair_groups",
            "pair_flip_rate",
            "readouts",
            "valid_readouts",
            "prompt_tokens_total",
            "completion_tokens_total",
        }
        raw = _require_object(value, fields, field_name="P1j arm metric")
        return cls(
            arm=_require_text(raw["arm"], "P1j metric arm"),
            decisions=_require_int(raw["decisions"], "P1j metric decisions"),
            valid_decisions=_require_int(
                raw["valid_decisions"], "P1j metric valid decisions"
            ),
            valid_rate=_require_number(raw["valid_rate"], "P1j metric valid rate"),
            correct_decisions=_require_int(
                raw["correct_decisions"], "P1j metric correct decisions"
            ),
            accuracy=_require_number(raw["accuracy"], "P1j metric accuracy"),
            pair_groups=_require_int(
                raw["pair_groups"], "P1j metric pair groups"
            ),
            valid_pair_groups=_require_int(
                raw["valid_pair_groups"], "P1j metric valid pair groups"
            ),
            pair_flip_rate=_require_number(
                raw["pair_flip_rate"], "P1j metric pair flip"
            ),
            readouts=_require_int(raw["readouts"], "P1j metric readouts"),
            valid_readouts=_require_int(
                raw["valid_readouts"], "P1j metric valid readouts"
            ),
            prompt_tokens_total=_require_int(
                raw["prompt_tokens_total"], "P1j metric prompt tokens"
            ),
            completion_tokens_total=_require_int(
                raw["completion_tokens_total"], "P1j metric completion tokens"
            ),
        )


def _arm_metric(
    *,
    arm: RelationshipP1Arm,
    decisions: tuple[RelationshipP1Decision, ...],
    readouts: tuple[RelationshipEvidenceReadout, ...],
) -> RelationshipP1jArmMetric:
    selected_decisions = tuple(item for item in decisions if item.arm is arm)
    by_key = {(item.arm, item.scene_id, item.seed): item for item in readouts}
    selected_readouts = tuple(
        by_key[(item.arm, item.scene_id, item.seed)] for item in selected_decisions
    )
    if not selected_decisions:
        raise ValueError("P1j arm metric requires decisions")
    groups: dict[tuple[str, int], list[RelationshipP1Decision]] = {}
    for item in selected_decisions:
        groups.setdefault((item.mirror_pair_id, item.seed), []).append(item)
    valid_groups = 0
    flip_groups = 0
    for group in groups.values():
        if len(group) != 2:
            raise ValueError("P1j mirrored metric group must contain two decisions")
        if all(item.chosen_action_id is not None for item in group):
            valid_groups += 1
            flip_groups += int(
                {item.chosen_action_id for item in group}
                == {
                    RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
                    RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
                }
            )
    valid = sum(int(item.valid) for item in selected_decisions)
    correct = sum(int(item.correct) for item in selected_decisions)
    return RelationshipP1jArmMetric(
        arm=arm.value,
        decisions=len(selected_decisions),
        valid_decisions=valid,
        valid_rate=valid / len(selected_decisions),
        correct_decisions=correct,
        accuracy=correct / len(selected_decisions),
        pair_groups=len(groups),
        valid_pair_groups=valid_groups,
        pair_flip_rate=flip_groups / valid_groups if valid_groups else 0.0,
        readouts=len(selected_readouts),
        valid_readouts=sum(int(item.valid) for item in selected_readouts),
        prompt_tokens_total=sum(item.prompt_tokens for item in selected_readouts),
        completion_tokens_total=sum(
            item.completion_tokens for item in selected_readouts
        ),
    )


class RelationshipP1jVerdict(str, Enum):
    QUALIFIED = "consumer_qualified_on_v4_development"
    DATASET_SATURATED = "v4_qualification_dataset_saturated"
    MACHINERY_REGRESSION = "qualification_machinery_regression"
    UNDERQUALIFIED = "consumer_failed_v4_qualification"


_NEXT_ACTIONS = {
    RelationshipP1jVerdict.QUALIFIED: "freeze_formal_comparison_prereg_candidate",
    RelationshipP1jVerdict.DATASET_SATURATED: (
        "stop_and_version_future_evidence_lane_without_consumer_feedback"
    ),
    RelationshipP1jVerdict.MACHINERY_REGRESSION: (
        "stop_qualification_lane_preserve_failed_attempt"
    ),
    RelationshipP1jVerdict.UNDERQUALIFIED: (
        "stop_consumer_lane_preserve_unseen_failure"
    ),
}


def _verdict_from_metrics(
    *,
    metrics: tuple[RelationshipP1jArmMetric, ...],
    consumer: RelationshipP1iFrozenConsumerProtocol,
) -> RelationshipP1jVerdict:
    by_arm = {item.arm: item for item in metrics}
    if tuple(by_arm) != tuple(arm.value for arm in _EVALUATED_ARMS):
        raise ValueError("P1j qualification metrics are incomplete or unordered")
    if any(
        item.valid_rate != consumer.required_valid_rate
        or item.valid_readouts != item.readouts
        for item in metrics
    ):
        return RelationshipP1jVerdict.MACHINERY_REGRESSION
    primary = tuple(by_arm[arm] for arm in consumer.primary_qualification_arms)
    if any(item.accuracy > consumer.maximum_accuracy for item in primary):
        return RelationshipP1jVerdict.DATASET_SATURATED
    primary_qualified = all(
        consumer.minimum_accuracy <= item.accuracy <= consumer.maximum_accuracy
        and item.pair_flip_rate >= consumer.minimum_pair_flip_rate
        for item in primary
    )
    structured = by_arm[RelationshipP1Arm.STRUCTURED_STATE.value]
    if (
        primary_qualified
        and structured.pair_flip_rate
        >= consumer.structured_state_minimum_pair_flip_rate
    ):
        return RelationshipP1jVerdict.QUALIFIED
    return RelationshipP1jVerdict.UNDERQUALIFIED


@dataclass(frozen=True)
class RelationshipP1jQualificationReport:
    created_at_iso: str
    qualification_protocol_id: str
    consumer_protocol_id: str
    consumer_split_contract_id: str
    qualification_dataset_fingerprint: str
    context_manifest_artifact_id: str
    qualification_context_surface_sha256: str
    selected_candidate_id: str
    selected_pipeline_sha256: str
    model_id: str
    weights_sha256: str
    generation_config_sha256: str
    readout_ledger_sha256: str
    decision_ledger_sha256: str
    arm_metrics: tuple[RelationshipP1jArmMetric, ...]
    qualification_input_count: int
    qualification_qwen_output_count: int
    qualification_feedback_to_consumer: bool
    consumer_revision_after_qualification: bool
    evaluation_feedback_to_pe_credit_reward_or_steering: bool
    formal_hidden_test_opened: bool
    p2_enabled: bool
    verdict: RelationshipP1jVerdict
    claim_boundary: str = _REPORT_CLAIM_BOUNDARY
    schema_version: str = RELATIONSHIP_P1J_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1J_REPORT_SCHEMA_VERSION:
            raise ValueError("P1j report schema_version mismatch")
        _require_timestamp(self.created_at_iso, "P1j report timestamp")
        for field_name, value in (
            ("qualification_protocol_id", self.qualification_protocol_id),
            ("consumer_protocol_id", self.consumer_protocol_id),
            ("consumer_split_contract_id", self.consumer_split_contract_id),
            (
                "qualification_dataset_fingerprint",
                self.qualification_dataset_fingerprint,
            ),
            ("context_manifest_artifact_id", self.context_manifest_artifact_id),
            (
                "qualification_context_surface_sha256",
                self.qualification_context_surface_sha256,
            ),
            ("selected_pipeline_sha256", self.selected_pipeline_sha256),
            ("weights_sha256", self.weights_sha256),
            ("generation_config_sha256", self.generation_config_sha256),
            ("readout_ledger_sha256", self.readout_ledger_sha256),
            ("decision_ledger_sha256", self.decision_ledger_sha256),
        ):
            _require_sha256(value, f"P1j report {field_name}")
        if tuple(item.arm for item in self.arm_metrics) != tuple(
            arm.value for arm in _EVALUATED_ARMS
        ):
            raise ValueError("P1j report metrics are incomplete or unordered")
        if self.qualification_input_count != 24:
            raise ValueError("P1j report qualification input count mismatch")
        if self.qualification_qwen_output_count != sum(
            item.readouts for item in self.arm_metrics
        ):
            raise ValueError("P1j report Qwen output count mismatch")
        if any(
            (
                self.qualification_feedback_to_consumer,
                self.consumer_revision_after_qualification,
                self.evaluation_feedback_to_pe_credit_reward_or_steering,
                self.formal_hidden_test_opened,
                self.p2_enabled,
            )
        ):
            raise ValueError("P1j report cannot open feedback, formal, or P2")
        if self.claim_boundary != _REPORT_CLAIM_BOUNDARY:
            raise ValueError("P1j report claim boundary mismatch")

    @property
    def next_action(self) -> str:
        return _NEXT_ACTIONS[self.verdict]

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at_iso": self.created_at_iso,
            "qualification_protocol_id": self.qualification_protocol_id,
            "consumer_protocol_id": self.consumer_protocol_id,
            "consumer_split_contract_id": self.consumer_split_contract_id,
            "qualification_dataset_fingerprint": (
                self.qualification_dataset_fingerprint
            ),
            "context_manifest_artifact_id": self.context_manifest_artifact_id,
            "qualification_context_surface_sha256": (
                self.qualification_context_surface_sha256
            ),
            "selected_candidate_id": self.selected_candidate_id,
            "selected_pipeline_sha256": self.selected_pipeline_sha256,
            "model_lineage": {
                "model_id": self.model_id,
                "weights_sha256": self.weights_sha256,
                "generation_config_sha256": self.generation_config_sha256,
            },
            "readout_ledger_sha256": self.readout_ledger_sha256,
            "decision_ledger_sha256": self.decision_ledger_sha256,
            "arm_metrics": [item.to_payload() for item in self.arm_metrics],
            "qualification_input_count": self.qualification_input_count,
            "qualification_qwen_output_count": self.qualification_qwen_output_count,
            "experiment_guards": {
                "qualification_feedback_to_consumer": (
                    self.qualification_feedback_to_consumer
                ),
                "consumer_revision_after_qualification": (
                    self.consumer_revision_after_qualification
                ),
                "evaluation_feedback_to_pe_credit_reward_or_steering": (
                    self.evaluation_feedback_to_pe_credit_reward_or_steering
                ),
                "formal_hidden_test_opened": self.formal_hidden_test_opened,
                "p2_enabled": self.p2_enabled,
            },
            "verdict": self.verdict.value,
            "next_action": self.next_action,
            "claim_boundary": self.claim_boundary,
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self._canonical_payload())

    def to_json(self) -> str:
        payload = self._canonical_payload()
        payload["artifact_id"] = self.artifact_id
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, encoded: str) -> "RelationshipP1jQualificationReport":
        try:
            raw_value = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ValueError("P1j report is not valid JSON") from exc
        raw = _require_object(
            raw_value,
            {
                "schema_version",
                "created_at_iso",
                "qualification_protocol_id",
                "consumer_protocol_id",
                "consumer_split_contract_id",
                "qualification_dataset_fingerprint",
                "context_manifest_artifact_id",
                "qualification_context_surface_sha256",
                "selected_candidate_id",
                "selected_pipeline_sha256",
                "model_lineage",
                "readout_ledger_sha256",
                "decision_ledger_sha256",
                "arm_metrics",
                "qualification_input_count",
                "qualification_qwen_output_count",
                "experiment_guards",
                "verdict",
                "next_action",
                "claim_boundary",
                "artifact_id",
            },
            field_name="P1j report",
        )
        model = _require_object(
            raw["model_lineage"],
            {"model_id", "weights_sha256", "generation_config_sha256"},
            field_name="P1j report model lineage",
        )
        guards = _require_object(
            raw["experiment_guards"],
            {
                "qualification_feedback_to_consumer",
                "consumer_revision_after_qualification",
                "evaluation_feedback_to_pe_credit_reward_or_steering",
                "formal_hidden_test_opened",
                "p2_enabled",
            },
            field_name="P1j report guards",
        )
        raw_metrics = raw["arm_metrics"]
        if not isinstance(raw_metrics, list):
            raise ValueError("P1j report metrics must be an array")
        report = cls(
            schema_version=_require_text(raw["schema_version"], "P1j report schema"),
            created_at_iso=_require_timestamp(
                raw["created_at_iso"], "P1j report timestamp"
            ),
            qualification_protocol_id=_require_sha256(
                raw["qualification_protocol_id"], "P1j report protocol"
            ),
            consumer_protocol_id=_require_sha256(
                raw["consumer_protocol_id"], "P1j report consumer"
            ),
            consumer_split_contract_id=_require_sha256(
                raw["consumer_split_contract_id"], "P1j report split contract"
            ),
            qualification_dataset_fingerprint=_require_sha256(
                raw["qualification_dataset_fingerprint"], "P1j report dataset"
            ),
            context_manifest_artifact_id=_require_sha256(
                raw["context_manifest_artifact_id"],
                "P1j report context manifest",
            ),
            qualification_context_surface_sha256=_require_sha256(
                raw["qualification_context_surface_sha256"],
                "P1j report context surface",
            ),
            selected_candidate_id=_require_text(
                raw["selected_candidate_id"], "P1j report candidate"
            ),
            selected_pipeline_sha256=_require_sha256(
                raw["selected_pipeline_sha256"], "P1j report pipeline"
            ),
            model_id=_require_text(model["model_id"], "P1j report model"),
            weights_sha256=_require_sha256(
                model["weights_sha256"], "P1j report weights"
            ),
            generation_config_sha256=_require_sha256(
                model["generation_config_sha256"], "P1j report generation"
            ),
            readout_ledger_sha256=_require_sha256(
                raw["readout_ledger_sha256"], "P1j report readout ledger"
            ),
            decision_ledger_sha256=_require_sha256(
                raw["decision_ledger_sha256"], "P1j report decision ledger"
            ),
            arm_metrics=tuple(
                RelationshipP1jArmMetric.from_payload(item) for item in raw_metrics
            ),
            qualification_input_count=_require_int(
                raw["qualification_input_count"], "P1j report input count"
            ),
            qualification_qwen_output_count=_require_int(
                raw["qualification_qwen_output_count"], "P1j report output count"
            ),
            qualification_feedback_to_consumer=_require_bool(
                guards["qualification_feedback_to_consumer"],
                "P1j report consumer feedback",
            ),
            consumer_revision_after_qualification=_require_bool(
                guards["consumer_revision_after_qualification"],
                "P1j report consumer revision",
            ),
            evaluation_feedback_to_pe_credit_reward_or_steering=_require_bool(
                guards["evaluation_feedback_to_pe_credit_reward_or_steering"],
                "P1j report learning feedback",
            ),
            formal_hidden_test_opened=_require_bool(
                guards["formal_hidden_test_opened"], "P1j report formal guard"
            ),
            p2_enabled=_require_bool(guards["p2_enabled"], "P1j report P2 guard"),
            verdict=RelationshipP1jVerdict(
                _require_text(raw["verdict"], "P1j report verdict")
            ),
            claim_boundary=_require_text(
                raw["claim_boundary"], "P1j report claim boundary"
            ),
        )
        if raw["next_action"] != report.next_action:
            raise ValueError("P1j report next action mismatch")
        if _require_sha256(raw["artifact_id"], "P1j report artifact") != (
            report.artifact_id
        ):
            raise ValueError("P1j report artifact id mismatch")
        return report


def _readout_ledger(
    *,
    candidate_id: str,
    readouts: tuple[RelationshipEvidenceReadout, ...],
) -> str:
    return "".join(
        canonical_json(
            relationship_p1i_readout_record_payload(
                candidate_id=candidate_id,
                readout=item,
            )
        )
        + "\n"
        for item in readouts
    )


def _decision_ledger(
    *,
    candidate_id: str,
    decisions: tuple[RelationshipP1Decision, ...],
) -> str:
    return "".join(
        canonical_json(
            relationship_p1i_decision_record_payload(
                candidate_id=candidate_id,
                decision=item,
            )
        )
        + "\n"
        for item in decisions
    )


def assess_relationship_p1j_qualification(
    *,
    protocol: RelationshipP1jQualificationProtocol,
    consumer: RelationshipP1iFrozenConsumerProtocol,
    split_bundle: RelationshipConsumerSplitBundle,
    progress: RelationshipP1jProgress,
    created_at_iso: str | None = None,
) -> RelationshipP1jQualificationReport:
    if not progress.is_complete:
        raise ValueError("P1j cannot assess an incomplete one-shot attempt")
    if progress.checkpoint.qualification_protocol_id != protocol.protocol_id:
        raise ValueError("P1j progress protocol lineage mismatch")
    metrics = tuple(
        _arm_metric(
            arm=arm,
            decisions=progress.decisions,
            readouts=progress.readouts,
        )
        for arm in _EVALUATED_ARMS
    )
    readout_ledger = _readout_ledger(
        candidate_id=consumer.selected_candidate.candidate_id,
        readouts=progress.readouts,
    )
    decision_ledger = _decision_ledger(
        candidate_id=consumer.selected_candidate.candidate_id,
        decisions=progress.decisions,
    )
    report = RelationshipP1jQualificationReport(
        created_at_iso=(
            created_at_iso
            or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        ),
        qualification_protocol_id=protocol.protocol_id,
        consumer_protocol_id=consumer.protocol_id,
        consumer_split_contract_id=split_bundle.contract.contract_sha256,
        qualification_dataset_fingerprint=(
            split_bundle.qualification_dataset.dataset_fingerprint
        ),
        context_manifest_artifact_id=protocol.context_manifest_artifact_id,
        qualification_context_surface_sha256=(
            protocol.qualification_context_surface_sha256
        ),
        selected_candidate_id=consumer.selected_candidate.candidate_id,
        selected_pipeline_sha256=consumer.selected_pipeline_sha256,
        model_id=consumer.model_id,
        weights_sha256=consumer.expected_weights_sha256,
        generation_config_sha256=consumer.expected_generation_config_sha256,
        readout_ledger_sha256=hashlib.sha256(
            readout_ledger.encode("utf-8")
        ).hexdigest(),
        decision_ledger_sha256=hashlib.sha256(
            decision_ledger.encode("utf-8")
        ).hexdigest(),
        arm_metrics=metrics,
        qualification_input_count=len(split_bundle.qualification_dataset.observations),
        qualification_qwen_output_count=len(progress.readouts),
        qualification_feedback_to_consumer=False,
        consumer_revision_after_qualification=False,
        evaluation_feedback_to_pe_credit_reward_or_steering=False,
        formal_hidden_test_opened=False,
        p2_enabled=False,
        verdict=_verdict_from_metrics(metrics=metrics, consumer=consumer),
    )
    validate_relationship_p1j_report_lineage(
        report,
        protocol=protocol,
        consumer=consumer,
        split_bundle=split_bundle,
    )
    return report


def validate_relationship_p1j_report_lineage(
    report: RelationshipP1jQualificationReport,
    *,
    protocol: RelationshipP1jQualificationProtocol,
    consumer: RelationshipP1iFrozenConsumerProtocol,
    split_bundle: RelationshipConsumerSplitBundle,
) -> None:
    expected = {
        "qualification_protocol_id": protocol.protocol_id,
        "consumer_protocol_id": consumer.protocol_id,
        "consumer_split_contract_id": split_bundle.contract.contract_sha256,
        "qualification_dataset_fingerprint": (
            split_bundle.qualification_dataset.dataset_fingerprint
        ),
        "context_manifest_artifact_id": protocol.context_manifest_artifact_id,
        "qualification_context_surface_sha256": (
            protocol.qualification_context_surface_sha256
        ),
        "selected_candidate_id": consumer.selected_candidate.candidate_id,
        "selected_pipeline_sha256": consumer.selected_pipeline_sha256,
        "model_id": consumer.model_id,
        "weights_sha256": consumer.expected_weights_sha256,
        "generation_config_sha256": consumer.expected_generation_config_sha256,
    }
    actual = vars(report)
    drift = sorted(name for name, value in expected.items() if actual[name] != value)
    if drift:
        raise ValueError(f"P1j report lineage mismatch: {drift}")
    if report.verdict is not _verdict_from_metrics(
        metrics=report.arm_metrics,
        consumer=consumer,
    ):
        raise ValueError("P1j report verdict diverges from frozen gate")
    if _parsed_timestamp(report.created_at_iso) < _parsed_timestamp(
        protocol.frozen_at_iso
    ):
        raise ValueError("P1j report cannot predate its protocol")


def write_relationship_p1j_report(
    *,
    report: RelationshipP1jQualificationReport,
    progress: RelationshipP1jProgress,
    output_dir: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    if not progress.is_complete:
        raise ValueError("P1j cannot write report for incomplete progress")
    root = pathlib.Path(output_dir)
    readout_text = _readout_ledger(
        candidate_id=progress.checkpoint.selected_candidate_id,
        readouts=progress.readouts,
    )
    decision_text = _decision_ledger(
        candidate_id=progress.checkpoint.selected_candidate_id,
        decisions=progress.decisions,
    )
    if hashlib.sha256(readout_text.encode("utf-8")).hexdigest() != (
        report.readout_ledger_sha256
    ):
        raise ValueError("P1j readout ledger hash mismatch before write")
    if hashlib.sha256(decision_text.encode("utf-8")).hexdigest() != (
        report.decision_ledger_sha256
    ):
        raise ValueError("P1j decision ledger hash mismatch before write")
    paths = (
        root / "readouts.jsonl",
        root / "decisions.jsonl",
        root / "packet1j_report.json",
        root / "packet1j_report.md",
    )
    if any(path.exists() for path in paths):
        raise FileExistsError("P1j terminal artifact already exists")
    _atomic_write_text(paths[0], readout_text)
    _atomic_write_text(paths[1], decision_text)
    _atomic_write_text(paths[2], report.to_json())
    lines = [
        "# Relationship Lab P1j one-shot v4 qualification",
        "",
        f"- Report artifact: `{report.artifact_id}`",
        f"- Qualification protocol: `{report.qualification_protocol_id}`",
        f"- Frozen consumer: `{report.consumer_protocol_id}`",
        f"- Selected candidate: `{report.selected_candidate_id}`",
        f"- v4 Qwen outputs: `{report.qualification_qwen_output_count}`",
        f"- Verdict: `{report.verdict.value}`",
        f"- Next action: `{report.next_action}`",
        "",
        "## Arm metrics",
        "",
        "| Arm | Valid | Accuracy | Pair flip |",
        "|---|---:|---:|---:|",
        *(
            f"| {item.arm} | {item.valid_decisions}/{item.decisions} | "
            f"{item.accuracy:.3f} | {item.pair_flip_rate:.3f} |"
            for item in report.arm_metrics
        ),
        "",
        "## Claim boundary",
        "",
        report.claim_boundary,
        "",
    ]
    _atomic_write_text(paths[3], "\n".join(lines))
    return paths[2], paths[3]


def load_relationship_p1j_report(
    path: pathlib.Path,
) -> RelationshipP1jQualificationReport:
    target = pathlib.Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"P1j report is missing: {target}")
    return RelationshipP1jQualificationReport.from_json(
        target.read_text(encoding="utf-8")
    )


def validate_relationship_p1j_terminal_files(
    *,
    report: RelationshipP1jQualificationReport,
    progress: RelationshipP1jProgress,
    output_dir: pathlib.Path,
) -> None:
    root = pathlib.Path(output_dir)
    expected_readouts = _readout_ledger(
        candidate_id=progress.checkpoint.selected_candidate_id,
        readouts=progress.readouts,
    )
    expected_decisions = _decision_ledger(
        candidate_id=progress.checkpoint.selected_candidate_id,
        decisions=progress.decisions,
    )
    if (root / "readouts.jsonl").read_text(encoding="utf-8") != expected_readouts:
        raise ValueError("P1j terminal readout ledger bytes mismatch")
    if (root / "decisions.jsonl").read_text(encoding="utf-8") != expected_decisions:
        raise ValueError("P1j terminal decision ledger bytes mismatch")
    loaded = load_relationship_p1j_report(root / "packet1j_report.json")
    if loaded != report:
        raise ValueError("P1j terminal report round-trip mismatch")


__all__ = [
    "RELATIONSHIP_P1J_CHECKPOINT_SCHEMA_VERSION",
    "RELATIONSHIP_P1J_PREPARED_NEXT_ACTION",
    "RELATIONSHIP_P1J_PROTOCOL_SCHEMA_VERSION",
    "RELATIONSHIP_P1J_REPORT_SCHEMA_VERSION",
    "RelationshipP1jArmMetric",
    "RelationshipP1jCheckpoint",
    "RelationshipP1jExecution",
    "RelationshipP1jProgress",
    "RelationshipP1jQualificationProtocol",
    "RelationshipP1jQualificationReport",
    "RelationshipP1jRecordKey",
    "RelationshipP1jVerdict",
    "assess_relationship_p1j_qualification",
    "build_relationship_p1j_checkpoint",
    "execute_relationship_p1j_qualification",
    "freeze_relationship_p1j_protocol",
    "load_relationship_p1j_checkpoint",
    "load_relationship_p1j_progress",
    "load_relationship_p1j_protocol",
    "load_relationship_p1j_report",
    "persist_relationship_p1j_decision",
    "persist_relationship_p1j_readout",
    "relationship_p1j_context_surface_sha256",
    "relationship_p1j_record_plan",
    "relationship_p1j_record_plan_sha256",
    "validate_relationship_p1j_progress",
    "validate_relationship_p1j_protocol_lineage",
    "validate_relationship_p1j_report_lineage",
    "validate_relationship_p1j_terminal_files",
    "write_relationship_p1j_checkpoint",
    "write_relationship_p1j_protocol",
    "write_relationship_p1j_report",
]
