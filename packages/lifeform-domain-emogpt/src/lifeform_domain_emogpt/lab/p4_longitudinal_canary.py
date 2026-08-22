"""P4.1 long-horizon canary plus the P4.2 preference-mutation drill.

The public plan is built from the already-seen ``relationship_transfer_v3``
fixture.  A system under test receives only incremental public sessions.  The
evaluator keeps scene identity, preferred action, and reactive-environment
seeds in a separate bundle.  The only ACTIVE authorization in this module is
for a typed relationship action consumed by the offline reactive environment;
it cannot authorize expression or a product runtime.

No model call lives here.  The protocol freezes steelman full-history and RAG
arms, but their execution remains blocked by the P1k/P1m evidence sequence.
The executable development lane therefore demonstrates the owner -> forecast
-> gate -> self_temporal -> reactive outcome -> PE -> credit mechanics and a
typed no-op control without claiming model or product superiority.

P4.2 independently exercises owner-authored correction/redaction, persistence
v3, forecast invalidation, and redaction tombstones. It uses no evaluator truth
or model output and does not alter the P4.1 effect report.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import pathlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from volvence_zero.credit import derive_preference_action_forecast_credit_records
from volvence_zero.dialogue_external_outcome import DialogueExternalOutcomeModule
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
)
from volvence_zero.memory import Track
from volvence_zero.runtime import WiringLevel
from volvence_zero.semantic_state import (
    SemanticProposal,
    SemanticProposalBatch,
    SemanticProposalOperation,
    SemanticProposalRuntime,
)
from volvence_zero.social import (
    PreferenceAboutOtherModule,
    PreferenceActionForecastProposal,
    PreferenceActionForecastRequest,
    PreferenceActionForecastRuntime,
    SocialPredictionErrorModule,
    SocialRecordStore,
)
from volvence_zero.social_cognition import (
    OtherMindRecord,
    PreferenceActionOutcomeEvidence,
    PreferenceActionOutcomeMutation,
    PreferenceActionOutcomeMutationOperation,
    PreferenceAboutOtherSnapshot,
    RelationshipConditionReadout,
    preference_action_outcome_evidence_sha256,
    preference_action_outcome_mutation_sha256,
)
from volvence_zero.substrate import SubstrateSnapshot, SurfaceKind
from volvence_zero.temporal import PlaceholderTemporalPolicy, TrackTemporalModule
from volvence_zero.temporal_types import (
    TemporalActionAdvisoryProposal,
    TemporalActionAdvisoryStatus,
)

from lifeform_domain_emogpt.lab.contracts import (
    PreActionRelationshipDecision,
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
    RelationshipAction,
    sha256_json,
)
from lifeform_domain_emogpt.lab.dataset import (
    RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME,
    RelationshipObservation,
    RelationshipTransferDataset,
    load_relationship_transfer_dataset,
)
from lifeform_domain_emogpt.lab.environment import ReactiveRelationshipEnvironment
from lifeform_domain_emogpt.relationship_action_gate import (
    RELATIONSHIP_ACTION_CREDIT_LEVEL,
    RelationshipActionGate,
    RelationshipActionGateCheckpoint,
    RelationshipActionGateDecision,
    RelationshipActionGateMode,
    temporal_action_advisory_from_gate_decision,
)
from lifeform_domain_emogpt.relationship_forecast import (
    BoundedRelationshipPreferenceForecastRuntime,
)


P4_LONGITUDINAL_CANARY_SCHEMA_VERSION = "relationship-p4-longitudinal-canary.v1"
P4_LONGITUDINAL_CANARY_AUTHORIZATION_SCHEMA_VERSION = "relationship-p4-lab-active-authorization.v1"
P4_LONGITUDINAL_CANARY_PREPARATION_SCHEMA_VERSION = "relationship-p4-longitudinal-canary-preparation.v1"
P4_LONGITUDINAL_CANARY_REPORT_SCHEMA_VERSION = "relationship-p4-longitudinal-canary-report.v1"
P4_LONGITUDINAL_CANARY_ARM_PREACTION_SCHEMA_VERSION = "relationship-p4-canary-arm-preaction.v1"
P4_PREFERENCE_MUTATION_DRILL_SCHEMA_VERSION = "relationship-p4-preference-mutation-drill.v1"
P4_LONGITUDINAL_CANARY_SCOPE = "relationship-p4-longitudinal-canary-lab.v1"

_PREFERENCE_SLOT = "preference_about_other"
_INTERLOCUTOR_ID = "primary"
_DEVELOPMENT_SUBJECT_COUNT = 2
_ONBOARDING_SESSIONS = 4
_DECISION_SESSIONS = 8
_FORMAL_MINIMUM_SUBJECTS = 20
_FORMAL_MINIMUM_CONTEXT_TOKENS = 32_768
_ACTION_IDS = tuple(action.value for action in RELATIONSHIP_ACTIONS)
_OUTCOME_IDS = tuple(outcome.value for outcome in RELATIONSHIP_OUTCOMES)
_POSITIVE_OUTCOMES = frozenset(
    {
        DialogueExternalOutcomeKind.HELPED.value,
        DialogueExternalOutcomeKind.FELT_HEARD.value,
    }
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "scene_id",
        "preferred_action",
        "expected_action",
        "policy_id",
        "condition_id",
        "dynamic_id",
        "generator_truth",
        "future_outcome",
        "environment_seed",
    }
)


class P4LongitudinalCanaryArm(str, Enum):
    QWEN_STEELMAN_FULL_HISTORY = "qwen_steelman_full_history"
    QWEN_STEELMAN_SELECTIVE_RAG = "qwen_steelman_selective_rag"
    VOLVENCE_CLOSED_LOOP = "volvence_closed_loop"
    VOLVENCE_TYPED_NOOP_CONTROL = "volvence_typed_noop_control"


_REQUIRED_ARMS = tuple(arm.value for arm in P4LongitudinalCanaryArm)


@dataclass(frozen=True)
class P4LongitudinalCanarySubjectSpec:
    subject_id: str
    onboarding_source_trajectory_sha256: str
    decision_source_trajectory_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.subject_id, "subject_id")
        _require_sha256(
            self.onboarding_source_trajectory_sha256,
            "onboarding_source_trajectory_sha256",
        )
        if len(self.decision_source_trajectory_sha256) != _DECISION_SESSIONS:
            raise ValueError("P4.1 fixture subject requires eight decision sources")
        for digest in self.decision_source_trajectory_sha256:
            _require_sha256(digest, "decision_source_trajectory_sha256")


@dataclass(frozen=True)
class RelationshipP4LongitudinalCanaryContract:
    protocol_sha256: str
    source_package_name: str
    source_dataset_fingerprint: str
    evidence_role: str
    story_title: str
    story_plain_language_claim: str
    subject_specs: tuple[P4LongitudinalCanarySubjectSpec, ...]
    phase_ids: tuple[str, ...]
    reactive_seed_namespace: str
    required_arms: tuple[str, ...]
    formal_minimum_independent_subjects: int
    formal_minimum_context_tokens: int
    authorization_scope: str
    allowed_policy_artifact_id: str
    allowed_policy_artifact_version: int
    claim_boundary: str
    schema_version: str = P4_LONGITUDINAL_CANARY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != P4_LONGITUDINAL_CANARY_SCHEMA_VERSION:
            raise ValueError("P4.1 protocol schema version mismatch")
        _require_sha256(self.protocol_sha256, "protocol_sha256")
        _require_sha256(
            self.source_dataset_fingerprint,
            "source_dataset_fingerprint",
        )
        if self.source_package_name != RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME:
            raise ValueError("P4.1 development fixture must remain on seen v3")
        if self.evidence_role != "seen_engineering_fixture_only":
            raise ValueError("P4.1 development evidence role drifted")
        for field_name, value in (
            ("story_title", self.story_title),
            ("story_plain_language_claim", self.story_plain_language_claim),
            ("allowed_policy_artifact_id", self.allowed_policy_artifact_id),
            ("claim_boundary", self.claim_boundary),
        ):
            _require_text(value, field_name)
        if len(self.subject_specs) != _DEVELOPMENT_SUBJECT_COUNT:
            raise ValueError("P4.1 development fixture requires exactly two subjects")
        subject_ids = tuple(item.subject_id for item in self.subject_specs)
        if len(set(subject_ids)) != len(subject_ids):
            raise ValueError("P4.1 subject ids must be unique")
        if len(self.phase_ids) != _DECISION_SESSIONS:
            raise ValueError("P4.1 requires eight ordered phase ids")
        _require_unique_texts(self.phase_ids, "phase_ids")
        if self.reactive_seed_namespace != "relationship-p4-canary-reactive-seeds-v1":
            raise ValueError("P4.1 reactive seed namespace drifted")
        if self.required_arms != _REQUIRED_ARMS:
            raise ValueError("P4.1 strong-baseline arm order drifted")
        if self.formal_minimum_independent_subjects != _FORMAL_MINIMUM_SUBJECTS:
            raise ValueError("P4.1 formal subject floor drifted")
        if self.formal_minimum_context_tokens != _FORMAL_MINIMUM_CONTEXT_TOKENS:
            raise ValueError("P4.1 long-context floor drifted")
        if self.authorization_scope != P4_LONGITUDINAL_CANARY_SCOPE:
            raise ValueError("P4.1 Lab ACTIVE scope drifted")
        if self.allowed_policy_artifact_version != 1:
            raise ValueError("P4.1 gate artifact version drifted")


@dataclass(frozen=True)
class P4CanaryOnboardingSession:
    subject_id: str
    session_id: str
    session_index: int
    event_id: str
    observation_summary: str
    action_id: str
    observed_outcome_id: str
    reaction_summary: str
    observation_ref: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("subject_id", self.subject_id),
            ("session_id", self.session_id),
            ("event_id", self.event_id),
            ("observation_summary", self.observation_summary),
            ("action_id", self.action_id),
            ("observed_outcome_id", self.observed_outcome_id),
            ("reaction_summary", self.reaction_summary),
            ("observation_ref", self.observation_ref),
        ):
            _require_text(value, field_name)
        if not 0 <= self.session_index < _ONBOARDING_SESSIONS:
            raise ValueError("onboarding session index is outside [0, 4)")
        if self.action_id not in _ACTION_IDS:
            raise ValueError("onboarding action is outside the frozen surface")
        if self.observed_outcome_id not in _OUTCOME_IDS:
            raise ValueError("onboarding outcome is outside the frozen surface")

    def to_owner_evidence(self) -> PreferenceActionOutcomeEvidence:
        return PreferenceActionOutcomeEvidence(
            evidence_id=self.event_id,
            interlocutor_id=_INTERLOCUTOR_ID,
            observation_summary=self.observation_summary,
            action_id=self.action_id,
            observed_outcome_id=self.observed_outcome_id,
            reaction_summary=self.reaction_summary,
            source_turn=self.session_index,
            evidence_refs=(self.observation_ref,),
        )

    def to_sut_payload(self) -> dict[str, object]:
        return {
            "schema_version": "relationship-p4-canary-onboarding-session.v1",
            "session_id": self.session_id,
            "session_index": self.session_index,
            "event_id": self.event_id,
            "observation_summary": self.observation_summary,
            "action_id": self.action_id,
            "observed_outcome_id": self.observed_outcome_id,
            "reaction_summary": self.reaction_summary,
            "observation_ref": self.observation_ref,
        }


@dataclass(frozen=True)
class P4CanaryDecisionSession:
    subject_id: str
    session_id: str
    decision_index: int
    action_turn_index: int
    outcome_turn_index: int
    decision_id: str
    phase_id: str
    probe_surface_family: str
    current_observation: str
    observation_ref: str
    source_trajectory_sha256: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("subject_id", self.subject_id),
            ("session_id", self.session_id),
            ("decision_id", self.decision_id),
            ("phase_id", self.phase_id),
            ("probe_surface_family", self.probe_surface_family),
            ("current_observation", self.current_observation),
            ("observation_ref", self.observation_ref),
        ):
            _require_text(value, field_name)
        if not 0 <= self.decision_index < _DECISION_SESSIONS:
            raise ValueError("decision index is outside [0, 8)")
        expected_action_turn = _ONBOARDING_SESSIONS + self.decision_index * 2
        if self.action_turn_index != expected_action_turn:
            raise ValueError("P4.1 action turn lineage drifted")
        if self.outcome_turn_index != self.action_turn_index + 1:
            raise ValueError("P4.1 outcome must settle on the next turn")
        _require_sha256(
            self.source_trajectory_sha256,
            "source_trajectory_sha256",
        )

    def to_forecast_request(self, *, subject_scope: str) -> PreferenceActionForecastRequest:
        return PreferenceActionForecastRequest(
            decision_id=self.decision_id,
            interlocutor_id=_INTERLOCUTOR_ID,
            current_observation=self.current_observation,
            observation_ref=self.observation_ref,
            candidate_action_ids=_ACTION_IDS,
            outcome_ids=_OUTCOME_IDS,
            turn_index=self.action_turn_index,
            session_scope=subject_scope,
        )

    def to_sut_payload(self) -> dict[str, object]:
        return {
            "schema_version": "relationship-p4-canary-decision-session.v1",
            "session_id": self.session_id,
            "decision_index": self.decision_index,
            "action_turn_index": self.action_turn_index,
            "decision_id": self.decision_id,
            "phase_id": self.phase_id,
            "probe_surface_family": self.probe_surface_family,
            "current_observation": self.current_observation,
            "observation_ref": self.observation_ref,
            "candidate_action_ids": list(_ACTION_IDS),
            "typed_outcome_ids": list(_OUTCOME_IDS),
        }


@dataclass(frozen=True)
class P4LongitudinalCanarySubject:
    subject_id: str
    subject_scope: str
    onboarding_sessions: tuple[P4CanaryOnboardingSession, ...]
    decision_sessions: tuple[P4CanaryDecisionSession, ...]

    def __post_init__(self) -> None:
        _require_text(self.subject_id, "subject_id")
        _require_sha256(self.subject_scope, "subject_scope")
        if len(self.onboarding_sessions) != _ONBOARDING_SESSIONS:
            raise ValueError("P4.1 subject requires four onboarding sessions")
        if len(self.decision_sessions) != _DECISION_SESSIONS:
            raise ValueError("P4.1 subject requires eight decision sessions")
        if any(item.subject_id != self.subject_id for item in self.onboarding_sessions):
            raise ValueError("onboarding subject lineage mismatch")
        if any(item.subject_id != self.subject_id for item in self.decision_sessions):
            raise ValueError("decision subject lineage mismatch")

    def to_sut_payload(self) -> dict[str, object]:
        return {
            "schema_version": "relationship-p4-canary-subject.v1",
            "subject_scope": self.subject_scope,
            "onboarding_sessions": [item.to_sut_payload() for item in self.onboarding_sessions],
            "decision_sessions": [item.to_sut_payload() for item in self.decision_sessions],
        }


@dataclass(frozen=True)
class RelationshipP4LongitudinalCanaryView:
    contract: RelationshipP4LongitudinalCanaryContract
    subjects: tuple[P4LongitudinalCanarySubject, ...]

    def __post_init__(self) -> None:
        if len(self.subjects) != len(self.contract.subject_specs):
            raise ValueError("P4.1 public subject count does not match protocol")
        _assert_no_public_truth_leakage(self)

    @property
    def public_plan_sha256(self) -> str:
        return sha256_json(
            {
                "protocol_sha256": self.contract.protocol_sha256,
                "subjects": [subject.to_sut_payload() for subject in self.subjects],
            }
        )


@dataclass(frozen=True)
class P4CanaryEvaluatorSession:
    session_id: str
    scene_id: str
    preferred_action_id: str
    environment_seed: int

    def __post_init__(self) -> None:
        _require_text(self.session_id, "session_id")
        _require_text(self.scene_id, "scene_id")
        if self.preferred_action_id not in _ACTION_IDS[:-1]:
            raise ValueError("evaluator preferred action must be a non-noop action")
        if self.environment_seed < 0:
            raise ValueError("environment_seed must be non-negative")


@dataclass(frozen=True)
class P4LongitudinalCanaryEvaluatorBundle:
    protocol_sha256: str
    source_dataset_fingerprint: str
    sessions: tuple[P4CanaryEvaluatorSession, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.protocol_sha256, "protocol_sha256")
        _require_sha256(
            self.source_dataset_fingerprint,
            "source_dataset_fingerprint",
        )
        expected_count = _DEVELOPMENT_SUBJECT_COUNT * _DECISION_SESSIONS
        if len(self.sessions) != expected_count:
            raise ValueError("P4.1 evaluator bundle session count drifted")
        session_ids = tuple(item.session_id for item in self.sessions)
        if len(set(session_ids)) != len(session_ids):
            raise ValueError("P4.1 evaluator session ids must be unique")

    def session(self, session_id: str) -> P4CanaryEvaluatorSession:
        for item in self.sessions:
            if item.session_id == session_id:
                return item
        raise KeyError(session_id)


@dataclass(frozen=True)
class RelationshipP4LabActiveAuthorization:
    authorization_id: str
    protocol_sha256: str
    source_dataset_fingerprint: str
    scope: str
    allowed_policy_artifact_id: str
    allowed_policy_artifact_version: int
    maximum_subjects: int
    maximum_decision_sessions_per_subject: int
    lab_typed_action_active_authorized: bool = True
    environment_consumer_only: bool = True
    expression_authorized: bool = False
    production_authorized: bool = False
    evaluation_feedback_to_learning: bool = False
    oracle_action_authorized: bool = False
    schema_version: str = P4_LONGITUDINAL_CANARY_AUTHORIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_sha256(self.authorization_id, "authorization_id")
        _require_sha256(self.protocol_sha256, "protocol_sha256")
        _require_sha256(
            self.source_dataset_fingerprint,
            "source_dataset_fingerprint",
        )
        if self.schema_version != P4_LONGITUDINAL_CANARY_AUTHORIZATION_SCHEMA_VERSION:
            raise ValueError("P4.1 Lab authorization schema drifted")
        if self.scope != P4_LONGITUDINAL_CANARY_SCOPE:
            raise ValueError("P4.1 Lab authorization scope drifted")
        _require_text(
            self.allowed_policy_artifact_id,
            "allowed_policy_artifact_id",
        )
        if self.allowed_policy_artifact_version != 1:
            raise ValueError("P4.1 allowed gate artifact version drifted")
        if self.maximum_subjects != _DEVELOPMENT_SUBJECT_COUNT:
            raise ValueError("P4.1 Lab authorization subject cap drifted")
        if self.maximum_decision_sessions_per_subject != _DECISION_SESSIONS:
            raise ValueError("P4.1 Lab authorization session cap drifted")
        required_flags = (
            self.lab_typed_action_active_authorized,
            self.environment_consumer_only,
            not self.expression_authorized,
            not self.production_authorized,
            not self.evaluation_feedback_to_learning,
            not self.oracle_action_authorized,
        )
        if not all(required_flags):
            raise ValueError("P4.1 Lab authorization firewall is open")
        if self.authorization_id != sha256_json(self._unsigned_payload()):
            raise ValueError("P4.1 Lab authorization content hash mismatch")

    def _unsigned_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_sha256": self.protocol_sha256,
            "source_dataset_fingerprint": self.source_dataset_fingerprint,
            "scope": self.scope,
            "allowed_policy_artifact_id": self.allowed_policy_artifact_id,
            "allowed_policy_artifact_version": self.allowed_policy_artifact_version,
            "maximum_subjects": self.maximum_subjects,
            "maximum_decision_sessions_per_subject": (self.maximum_decision_sessions_per_subject),
            "lab_typed_action_active_authorized": (self.lab_typed_action_active_authorized),
            "environment_consumer_only": self.environment_consumer_only,
            "expression_authorized": self.expression_authorized,
            "production_authorized": self.production_authorized,
            "evaluation_feedback_to_learning": self.evaluation_feedback_to_learning,
            "oracle_action_authorized": self.oracle_action_authorized,
        }

    def to_payload(self) -> dict[str, object]:
        return {"authorization_id": self.authorization_id, **self._unsigned_payload()}


@dataclass(frozen=True)
class P4LongitudinalCanaryPreparation:
    protocol_sha256: str
    public_plan_sha256: str
    lab_authorization_id: str
    required_arms: tuple[str, ...]
    arm_statuses: tuple[tuple[str, str], ...]
    development_subject_count: int
    formal_minimum_independent_subjects: int
    formal_minimum_context_tokens: int
    model_output_count: int
    formal_evidence_authorized: bool
    correction_redaction_owner_status: str
    claim_boundary: str
    schema_version: str = P4_LONGITUDINAL_CANARY_PREPARATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != P4_LONGITUDINAL_CANARY_PREPARATION_SCHEMA_VERSION:
            raise ValueError("P4.1 preparation schema version mismatch")
        for field_name, digest in (
            ("protocol_sha256", self.protocol_sha256),
            ("public_plan_sha256", self.public_plan_sha256),
            ("lab_authorization_id", self.lab_authorization_id),
        ):
            _require_sha256(digest, field_name)
        if self.required_arms != _REQUIRED_ARMS:
            raise ValueError("P4.1 preparation arm set drifted")
        if tuple(name for name, _ in self.arm_statuses) != self.required_arms:
            raise ValueError("P4.1 preparation status arm order drifted")
        if self.development_subject_count != _DEVELOPMENT_SUBJECT_COUNT:
            raise ValueError("P4.1 preparation subject count drifted")
        if self.formal_minimum_independent_subjects != _FORMAL_MINIMUM_SUBJECTS:
            raise ValueError("P4.1 preparation formal subject floor drifted")
        if self.formal_minimum_context_tokens != _FORMAL_MINIMUM_CONTEXT_TOKENS:
            raise ValueError("P4.1 preparation context floor drifted")
        if self.model_output_count != 0 or self.formal_evidence_authorized:
            raise ValueError("P4.1 zero-output preparation cannot authorize evidence")
        if self.correction_redaction_owner_status != ("implemented_p4_2_engineering_drill_available"):
            raise ValueError("P4.1 correction/redaction owner status drifted")
        _require_text(self.claim_boundary, "claim_boundary")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_sha256": self.protocol_sha256,
            "public_plan_sha256": self.public_plan_sha256,
            "lab_authorization_id": self.lab_authorization_id,
            "required_arms": list(self.required_arms),
            "arm_statuses": [{"arm": arm, "status": status} for arm, status in self.arm_statuses],
            "development_subject_count": self.development_subject_count,
            "formal_minimum_independent_subjects": (self.formal_minimum_independent_subjects),
            "formal_minimum_context_tokens": self.formal_minimum_context_tokens,
            "model_output_count": self.model_output_count,
            "formal_evidence_authorized": self.formal_evidence_authorized,
            "correction_redaction_owner_status": (self.correction_redaction_owner_status),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class P4PreferenceMutationDrillReport:
    protocol_sha256: str
    subject_scope: str
    correction_command_sha256: str
    corrected_evidence_sha256: str
    reader_observed_evidence_sha256: str
    redaction_command_sha256: str
    correction_invalidated_forecast_count: int
    redaction_invalidated_forecast_count: int
    correction_persisted_after_restart: bool
    redaction_content_absent_after_restart: bool
    redaction_tombstone_enforced: bool
    process_restart_count: int
    model_output_count: int = 0
    evaluator_truth_used: bool = False
    formal_evidence_authorized: bool = False
    schema_version: str = P4_PREFERENCE_MUTATION_DRILL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != P4_PREFERENCE_MUTATION_DRILL_SCHEMA_VERSION:
            raise ValueError("P4.2 mutation drill schema version mismatch")
        for field_name, digest in (
            ("protocol_sha256", self.protocol_sha256),
            ("subject_scope", self.subject_scope),
            ("correction_command_sha256", self.correction_command_sha256),
            ("corrected_evidence_sha256", self.corrected_evidence_sha256),
            (
                "reader_observed_evidence_sha256",
                self.reader_observed_evidence_sha256,
            ),
            ("redaction_command_sha256", self.redaction_command_sha256),
        ):
            _require_sha256(digest, field_name)
        for field_name, count in (
            (
                "correction_invalidated_forecast_count",
                self.correction_invalidated_forecast_count,
            ),
            (
                "redaction_invalidated_forecast_count",
                self.redaction_invalidated_forecast_count,
            ),
        ):
            if count < 1:
                raise ValueError(f"{field_name} must be >= 1")
        if self.process_restart_count < _ONBOARDING_SESSIONS + 3:
            raise ValueError("P4.2 mutation drill restart count is too small")
        if self.corrected_evidence_sha256 != self.reader_observed_evidence_sha256:
            raise ValueError("P4.2 forecast reader did not observe corrected evidence")
        if not all(
            (
                self.correction_persisted_after_restart,
                self.redaction_content_absent_after_restart,
                self.redaction_tombstone_enforced,
            )
        ):
            raise ValueError("P4.2 mutation drill did not close every invariant")
        if self.model_output_count != 0 or self.evaluator_truth_used or self.formal_evidence_authorized:
            raise ValueError("P4.2 engineering drill crossed its claim boundary")

    @property
    def passed(self) -> bool:
        return True

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_sha256": self.protocol_sha256,
            "subject_scope": self.subject_scope,
            "correction_command_sha256": self.correction_command_sha256,
            "corrected_evidence_sha256": self.corrected_evidence_sha256,
            "reader_observed_evidence_sha256": (self.reader_observed_evidence_sha256),
            "redaction_command_sha256": self.redaction_command_sha256,
            "correction_invalidated_forecast_count": (self.correction_invalidated_forecast_count),
            "redaction_invalidated_forecast_count": (self.redaction_invalidated_forecast_count),
            "correction_persisted_after_restart": (self.correction_persisted_after_restart),
            "redaction_content_absent_after_restart": (self.redaction_content_absent_after_restart),
            "redaction_tombstone_enforced": self.redaction_tombstone_enforced,
            "process_restart_count": self.process_restart_count,
            "model_output_count": self.model_output_count,
            "evaluator_truth_used": self.evaluator_truth_used,
            "formal_evidence_authorized": self.formal_evidence_authorized,
            "passed": self.passed,
            "claim_boundary": (
                "P4.2 engineering-only owner mutation drill; no Qwen, evaluator "
                "truth, expression, production ACTIVE, or formal effect claim."
            ),
        }


@dataclass(frozen=True)
class P4CanaryArmPreActionRecord:
    """Arm-neutral, outcome-free record frozen before environment settlement."""

    protocol_sha256: str
    public_plan_sha256: str
    arm: P4LongitudinalCanaryArm
    subject_scope: str
    session_id: str
    phase_id: str
    public_context_tokens: int
    latency_ms: float
    response_sha256: str
    decision: PreActionRelationshipDecision
    schema_version: str = P4_LONGITUDINAL_CANARY_ARM_PREACTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != P4_LONGITUDINAL_CANARY_ARM_PREACTION_SCHEMA_VERSION:
            raise ValueError("P4.1 arm pre-action schema version mismatch")
        for field_name, digest in (
            ("protocol_sha256", self.protocol_sha256),
            ("public_plan_sha256", self.public_plan_sha256),
            ("subject_scope", self.subject_scope),
            ("response_sha256", self.response_sha256),
        ):
            _require_sha256(digest, field_name)
        for field_name, value in (
            ("session_id", self.session_id),
            ("phase_id", self.phase_id),
        ):
            _require_text(value, field_name)
        if isinstance(self.public_context_tokens, bool) or not isinstance(
            self.public_context_tokens,
            int,
        ):
            raise ValueError("public_context_tokens must be an integer")
        if self.public_context_tokens < 0:
            raise ValueError("public_context_tokens must be non-negative")
        if (
            isinstance(self.latency_ms, bool)
            or not isinstance(self.latency_ms, (int, float))
            or not math.isfinite(self.latency_ms)
            or self.latency_ms < 0.0
        ):
            raise ValueError("latency_ms must be finite and non-negative")
        if self.decision.chosen_action_id.value not in _ACTION_IDS:
            raise ValueError("pre-action record chose an action outside the surface")

    @property
    def record_sha256(self) -> str:
        return sha256_json(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_sha256": self.protocol_sha256,
            "public_plan_sha256": self.public_plan_sha256,
            "arm": self.arm.value,
            "subject_scope": self.subject_scope,
            "session_id": self.session_id,
            "phase_id": self.phase_id,
            "public_context_tokens": self.public_context_tokens,
            "latency_ms": float(self.latency_ms),
            "response_sha256": self.response_sha256,
            "decision": self.decision.to_payload(),
        }


@dataclass(frozen=True)
class P4CanarySessionTrace:
    subject_scope: str
    session_id: str
    phase_id: str
    decision_id: str
    forecast_id: str
    recommended_action_id: str
    gate_action: str
    exposed_action_id: str
    temporal_status: str
    observed_outcome_id: str
    preferred_action_match: bool
    credit_value: float
    credit_applied_to_gate: bool
    owner_persistence_sha256: str
    gate_checkpoint_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.subject_scope, "subject_scope")
        for field_name, value in (
            ("session_id", self.session_id),
            ("phase_id", self.phase_id),
            ("decision_id", self.decision_id),
            ("forecast_id", self.forecast_id),
            ("recommended_action_id", self.recommended_action_id),
            ("gate_action", self.gate_action),
            ("exposed_action_id", self.exposed_action_id),
            ("temporal_status", self.temporal_status),
            ("observed_outcome_id", self.observed_outcome_id),
        ):
            _require_text(value, field_name)
        if self.recommended_action_id not in _ACTION_IDS:
            raise ValueError("trace recommended action is outside surface")
        if self.exposed_action_id not in _ACTION_IDS:
            raise ValueError("trace exposed action is outside surface")
        if self.observed_outcome_id not in _OUTCOME_IDS:
            raise ValueError("trace outcome is outside surface")
        if self.temporal_status != TemporalActionAdvisoryStatus.APPLIED.value:
            raise ValueError("P4.1 mechanism trace requires Lab APPLIED status")
        if not math.isfinite(self.credit_value) or not -1.0 <= self.credit_value <= 1.0:
            raise ValueError("trace credit_value must be finite and in [-1, 1]")
        _require_sha256(
            self.owner_persistence_sha256,
            "owner_persistence_sha256",
        )
        _require_sha256(
            self.gate_checkpoint_sha256,
            "gate_checkpoint_sha256",
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "subject_scope": self.subject_scope,
            "session_id": self.session_id,
            "phase_id": self.phase_id,
            "decision_id": self.decision_id,
            "forecast_id": self.forecast_id,
            "recommended_action_id": self.recommended_action_id,
            "gate_action": self.gate_action,
            "exposed_action_id": self.exposed_action_id,
            "temporal_status": self.temporal_status,
            "observed_outcome_id": self.observed_outcome_id,
            "preferred_action_match": self.preferred_action_match,
            "credit_value": self.credit_value,
            "credit_applied_to_gate": self.credit_applied_to_gate,
            "owner_persistence_sha256": self.owner_persistence_sha256,
            "gate_checkpoint_sha256": self.gate_checkpoint_sha256,
        }


@dataclass(frozen=True)
class P4CanaryGateAudit:
    """Outcome-free gate state plus the later exact PE-credit update."""

    subject_scope: str
    session_id: str
    decision_id: str
    forecast_id: str
    gate_mode: RelationshipActionGateMode
    gate_action: str
    steer_probability: float
    features: tuple[float, ...]
    pre_update_weights: tuple[float, ...]
    pre_update_bias: float
    pre_update_count: int
    pre_update_state_sha256: str
    credit_record_id: str
    credit_level: str
    credit_track: str
    credit_source_event: str
    credit_environment_outcome_id: str
    credit_value: float
    credit_applied_to_gate: bool
    post_update_weights: tuple[float, ...]
    post_update_bias: float
    post_update_count: int
    post_update_state_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.subject_scope, "gate audit subject_scope")
        for field_name, value in (
            ("session_id", self.session_id),
            ("decision_id", self.decision_id),
            ("forecast_id", self.forecast_id),
            ("gate_action", self.gate_action),
            ("credit_record_id", self.credit_record_id),
            ("credit_source_event", self.credit_source_event),
            (
                "credit_environment_outcome_id",
                self.credit_environment_outcome_id,
            ),
        ):
            _require_text(value, field_name)
        if self.gate_mode is RelationshipActionGateMode.ORACLE:
            raise ValueError("P4.1 gate audit cannot contain oracle mode")
        if not 0.0 <= self.steer_probability <= 1.0:
            raise ValueError("P4.1 gate audit probability must be in [0, 1]")
        if len(self.features) != 5 or any(
            not math.isfinite(value) or not -1.0 <= value <= 1.0
            for value in self.features
        ):
            raise ValueError("P4.1 gate audit feature shape drift")
        if len(self.pre_update_weights) != 5 or len(self.post_update_weights) != 5:
            raise ValueError("P4.1 gate audit parameter shape drift")
        if any(
            not math.isfinite(value)
            for value in (
                *self.pre_update_weights,
                self.pre_update_bias,
                *self.post_update_weights,
                self.post_update_bias,
                self.credit_value,
            )
        ):
            raise ValueError("P4.1 gate audit contains non-finite values")
        if not -1.0 <= self.credit_value <= 1.0:
            raise ValueError("P4.1 gate audit credit must be in [-1, 1]")
        if self.credit_level != RELATIONSHIP_ACTION_CREDIT_LEVEL:
            raise ValueError("P4.1 gate audit credit level drift")
        if self.credit_track != Track.SELF.value:
            raise ValueError("P4.1 gate audit credit track drift")
        if not self.credit_source_event.startswith("social_pe:social-pe:"):
            raise ValueError("P4.1 gate audit bypassed the social PE owner")
        _require_sha256(self.pre_update_state_sha256, "pre-update gate state")
        _require_sha256(self.post_update_state_sha256, "post-update gate state")
        if self.credit_applied_to_gate:
            if self.gate_mode is not RelationshipActionGateMode.LEARNED:
                raise ValueError("P4.1 only learned gate may apply credit")
            if self.post_update_count != self.pre_update_count + 1:
                raise ValueError("P4.1 applied gate update count drift")
        elif (
            self.post_update_count != self.pre_update_count
            or self.post_update_weights != self.pre_update_weights
            or self.post_update_bias != self.pre_update_bias
            or self.post_update_state_sha256 != self.pre_update_state_sha256
        ):
            raise ValueError("P4.1 no-credit gate changed parameter state")

    @property
    def parameter_changed(self) -> bool:
        return self.post_update_state_sha256 != self.pre_update_state_sha256

    def to_payload(self) -> dict[str, object]:
        return {
            "subject_scope": self.subject_scope,
            "session_id": self.session_id,
            "decision_id": self.decision_id,
            "forecast_id": self.forecast_id,
            "gate_mode": self.gate_mode.value,
            "gate_action": self.gate_action,
            "steer_probability": self.steer_probability,
            "features": list(self.features),
            "pre_update_weights": list(self.pre_update_weights),
            "pre_update_bias": self.pre_update_bias,
            "pre_update_count": self.pre_update_count,
            "pre_update_state_sha256": self.pre_update_state_sha256,
            "credit_record_id": self.credit_record_id,
            "credit_level": self.credit_level,
            "credit_track": self.credit_track,
            "credit_source_event": self.credit_source_event,
            "credit_environment_outcome_id": (
                self.credit_environment_outcome_id
            ),
            "credit_value": self.credit_value,
            "credit_applied_to_gate": self.credit_applied_to_gate,
            "post_update_weights": list(self.post_update_weights),
            "post_update_bias": self.post_update_bias,
            "post_update_count": self.post_update_count,
            "post_update_state_sha256": self.post_update_state_sha256,
            "parameter_changed": self.parameter_changed,
        }


@dataclass(frozen=True)
class P4CanaryMechanismRun:
    """Arm-neutral execution of one subject through the real owner loop."""

    subject_scope: str
    gate_mode: RelationshipActionGateMode
    credit_applied_to_gate: bool
    traces: tuple[P4CanarySessionTrace, ...]
    condition_readouts: tuple[RelationshipConditionReadout | None, ...]
    gate_audits: tuple[P4CanaryGateAudit, ...]
    positive_outcome_count: int
    preferred_action_match_count: int
    reversal_opportunity_count: int
    reversal_match_count: int
    gate_update_count: int
    process_restart_count: int

    def __post_init__(self) -> None:
        _require_sha256(self.subject_scope, "subject_scope")
        if self.gate_mode is RelationshipActionGateMode.ORACLE:
            raise ValueError("P4.1 mechanism run cannot use evaluator oracle mode")
        if self.credit_applied_to_gate and self.gate_mode is not RelationshipActionGateMode.LEARNED:
            raise ValueError("P4.1 credit can update only a learned gate")
        if len(self.traces) != _DECISION_SESSIONS:
            raise ValueError("P4.1 mechanism run requires eight traces")
        if len(self.condition_readouts) != len(self.traces):
            raise ValueError("P4.1 condition readout count must match traces")
        if len(self.gate_audits) != len(self.traces):
            raise ValueError("P4.1 gate audit count must match traces")
        for value in (
            self.positive_outcome_count,
            self.preferred_action_match_count,
            self.reversal_opportunity_count,
            self.reversal_match_count,
            self.gate_update_count,
            self.process_restart_count,
        ):
            if value < 0:
                raise ValueError("P4.1 mechanism counts must be non-negative")
        if self.credit_applied_to_gate != all(
            trace.credit_applied_to_gate for trace in self.traces
        ):
            raise ValueError("P4.1 mechanism credit application drift")
        if not self.credit_applied_to_gate and self.gate_update_count != 0:
            raise ValueError("P4.1 no-learning mechanism changed gate state")
        if any(
            audit.subject_scope != self.subject_scope
            or audit.gate_mode is not self.gate_mode
            or audit.credit_applied_to_gate != self.credit_applied_to_gate
            or audit.session_id != trace.session_id
            or audit.decision_id != trace.decision_id
            or audit.forecast_id != trace.forecast_id
            or audit.credit_value != trace.credit_value
            for audit, trace in zip(self.gate_audits, self.traces, strict=True)
        ):
            raise ValueError("P4.1 gate audit/trace lineage drift")


@dataclass(frozen=True)
class P4CanaryArmRun:
    arm: P4LongitudinalCanaryArm
    subject_scope: str
    traces: tuple[P4CanarySessionTrace, ...]
    positive_outcome_count: int
    preferred_action_match_count: int
    reversal_opportunity_count: int
    reversal_match_count: int
    gate_update_count: int
    process_restart_count: int

    def __post_init__(self) -> None:
        if self.arm not in {
            P4LongitudinalCanaryArm.VOLVENCE_CLOSED_LOOP,
            P4LongitudinalCanaryArm.VOLVENCE_TYPED_NOOP_CONTROL,
        }:
            raise ValueError("only built-in typed arms may produce a mechanism run")
        _require_sha256(self.subject_scope, "subject_scope")
        if len(self.traces) != _DECISION_SESSIONS:
            raise ValueError("P4.1 arm run requires eight traces")
        for value in (
            self.positive_outcome_count,
            self.preferred_action_match_count,
            self.reversal_opportunity_count,
            self.reversal_match_count,
            self.gate_update_count,
            self.process_restart_count,
        ):
            if value < 0:
                raise ValueError("P4.1 run counts must be non-negative")

    def to_payload(self) -> dict[str, object]:
        return {
            "arm": self.arm.value,
            "subject_scope": self.subject_scope,
            "traces": [trace.to_payload() for trace in self.traces],
            "positive_outcome_count": self.positive_outcome_count,
            "preferred_action_match_count": self.preferred_action_match_count,
            "reversal_opportunity_count": self.reversal_opportunity_count,
            "reversal_match_count": self.reversal_match_count,
            "gate_update_count": self.gate_update_count,
            "process_restart_count": self.process_restart_count,
        }


@dataclass(frozen=True)
class P4CanaryArmSummary:
    arm: P4LongitudinalCanaryArm
    completed_subject_count: int
    typed_outcome_count: int
    positive_outcome_count: int
    preferred_action_match_count: int
    reversal_opportunity_count: int
    reversal_match_count: int
    gate_update_count: int

    def __post_init__(self) -> None:
        if self.arm not in {
            P4LongitudinalCanaryArm.VOLVENCE_CLOSED_LOOP,
            P4LongitudinalCanaryArm.VOLVENCE_TYPED_NOOP_CONTROL,
        }:
            raise ValueError("P4.1 summary only accepts built-in typed arms")
        if self.completed_subject_count != _DEVELOPMENT_SUBJECT_COUNT:
            raise ValueError("P4.1 summary subject count drifted")
        expected_outcomes = _DEVELOPMENT_SUBJECT_COUNT * _DECISION_SESSIONS
        if self.typed_outcome_count != expected_outcomes:
            raise ValueError("P4.1 summary typed outcome count drifted")
        for value in (
            self.positive_outcome_count,
            self.preferred_action_match_count,
            self.reversal_opportunity_count,
            self.reversal_match_count,
            self.gate_update_count,
        ):
            if value < 0:
                raise ValueError("P4.1 summary counts must be non-negative")

    def to_payload(self) -> dict[str, object]:
        return {
            "arm": self.arm.value,
            "completed_subject_count": self.completed_subject_count,
            "typed_outcome_count": self.typed_outcome_count,
            "positive_outcome_count": self.positive_outcome_count,
            "preferred_action_match_count": self.preferred_action_match_count,
            "reversal_opportunity_count": self.reversal_opportunity_count,
            "reversal_match_count": self.reversal_match_count,
            "gate_update_count": self.gate_update_count,
        }


@dataclass(frozen=True)
class RelationshipP4LongitudinalCanaryReport:
    protocol_sha256: str
    public_plan_sha256: str
    lab_authorization_id: str
    runs: tuple[P4CanaryArmRun, ...]
    arm_summaries: tuple[P4CanaryArmSummary, ...]
    required_arm_statuses: tuple[tuple[str, str], ...]
    model_output_count: int
    expression_output_count: int
    formal_evidence_authorized: bool
    verdict: str
    claim_boundary: str
    schema_version: str = P4_LONGITUDINAL_CANARY_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != P4_LONGITUDINAL_CANARY_REPORT_SCHEMA_VERSION:
            raise ValueError("P4.1 report schema version mismatch")
        for field_name, digest in (
            ("protocol_sha256", self.protocol_sha256),
            ("public_plan_sha256", self.public_plan_sha256),
            ("lab_authorization_id", self.lab_authorization_id),
        ):
            _require_sha256(digest, field_name)
        expected_runs = _DEVELOPMENT_SUBJECT_COUNT * 2
        if len(self.runs) != expected_runs:
            raise ValueError("P4.1 development report run count drifted")
        if tuple(item.arm for item in self.arm_summaries) != (
            P4LongitudinalCanaryArm.VOLVENCE_CLOSED_LOOP,
            P4LongitudinalCanaryArm.VOLVENCE_TYPED_NOOP_CONTROL,
        ):
            raise ValueError("P4.1 report summary order drifted")
        if tuple(name for name, _ in self.required_arm_statuses) != _REQUIRED_ARMS:
            raise ValueError("P4.1 report arm order drifted")
        if self.model_output_count != 0 or self.expression_output_count != 0:
            raise ValueError("P4.1 mechanism report cannot claim model/expression output")
        if self.formal_evidence_authorized:
            raise ValueError("P4.1 development report cannot authorize formal evidence")
        if self.verdict != "engineering_mechanism_ready_formal_effect_not_run":
            raise ValueError("P4.1 development verdict drifted")
        _require_text(self.claim_boundary, "claim_boundary")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_sha256": self.protocol_sha256,
            "public_plan_sha256": self.public_plan_sha256,
            "lab_authorization_id": self.lab_authorization_id,
            "runs": [run.to_payload() for run in self.runs],
            "arm_summaries": [item.to_payload() for item in self.arm_summaries],
            "required_arm_statuses": [{"arm": arm, "status": status} for arm, status in self.required_arm_statuses],
            "model_output_count": self.model_output_count,
            "expression_output_count": self.expression_output_count,
            "formal_evidence_authorized": self.formal_evidence_authorized,
            "verdict": self.verdict,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class _P4CanaryOwnerEvidence:
    subject_id: str
    event_id: str
    source_turn: int
    observation_summary: str
    action_id: str
    observed_outcome_id: str
    reaction_summary: str
    observation_ref: str

    def to_owner_evidence(self) -> PreferenceActionOutcomeEvidence:
        return PreferenceActionOutcomeEvidence(
            evidence_id=self.event_id,
            interlocutor_id=_INTERLOCUTOR_ID,
            observation_summary=self.observation_summary,
            action_id=self.action_id,
            observed_outcome_id=self.observed_outcome_id,
            reaction_summary=self.reaction_summary,
            source_turn=self.source_turn,
            evidence_refs=(self.observation_ref,),
        )


class _P4CanaryEvidenceProposalRuntime(SemanticProposalRuntime):
    def __init__(self, evidence: _P4CanaryOwnerEvidence) -> None:
        self._evidence = evidence
        self.runtime_id = f"relationship-p4-canary-owner:{evidence.event_id}"

    def propose(
        self,
        *,
        target_slot: str,
        user_input: str | None,
        substrate_snapshot: object | None,
        memory_snapshot: object | None,
        previous_snapshot: object | None,
        turn_index: int,
    ) -> SemanticProposalBatch:
        del substrate_snapshot, memory_snapshot, previous_snapshot
        if target_slot != _PREFERENCE_SLOT:
            raise ValueError("P4.1 owner runtime only serves preference_about_other")
        if turn_index != self._evidence.source_turn:
            raise ValueError("P4.1 owner evidence turn lineage mismatch")
        if user_input != self._evidence.observation_summary:
            raise ValueError("P4.1 owner input differs from typed evidence")
        return SemanticProposalBatch(
            proposals=(
                SemanticProposal(
                    proposal_id=self._evidence.event_id,
                    target_slot=_PREFERENCE_SLOT,
                    operation=SemanticProposalOperation.OBSERVE,
                    summary=self._evidence.observation_summary,
                    detail=self._evidence.reaction_summary,
                    confidence=0.90,
                    evidence=self._evidence.observation_ref,
                    control_signal=0.0,
                ),
            ),
            runtime_id=self.runtime_id,
            schema_version=1,
            description="One typed P4.1 relationship outcome proposed to its owner.",
        )


class _P4MutationDrillForecastRuntime:
    """Capture the exact corrected evidence seen by the bounded reader."""

    runtime_id = "relationship-p4.2-mutation-drill-reader.v1"

    def __init__(self, *, target_evidence_id: str) -> None:
        self._target_evidence_id = target_evidence_id
        self._delegate = BoundedRelationshipPreferenceForecastRuntime()
        self.reader_observed_evidence_sha256: str | None = None

    def propose(
        self,
        *,
        request: PreferenceActionForecastRequest,
        records: tuple[OtherMindRecord, ...],
        action_outcomes: tuple[PreferenceActionOutcomeEvidence, ...],
    ) -> PreferenceActionForecastProposal | None:
        matching = tuple(item for item in action_outcomes if item.evidence_id == self._target_evidence_id)
        if len(matching) != 1:
            raise RuntimeError("P4.2 forecast reader did not receive exactly one correction target")
        self.reader_observed_evidence_sha256 = preference_action_outcome_evidence_sha256(matching[0])
        return self._delegate.propose(
            request=request,
            records=records,
            action_outcomes=action_outcomes,
        )


def relationship_p4_longitudinal_canary_protocol_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1] / "lab_protocols" / "relationship_p4_longitudinal_canary_v1.json"


def load_relationship_p4_longitudinal_canary_contract(
    protocol_path: pathlib.Path | None = None,
) -> RelationshipP4LongitudinalCanaryContract:
    raw = _load_protocol_raw(protocol_path)
    source = _require_mapping(raw["source"], "source")
    story = _require_mapping(raw["story"], "story")
    fixture = _require_mapping(raw["development_fixture"], "development_fixture")
    formal = _require_mapping(raw["formal_pilot"], "formal_pilot")
    baselines = _require_mapping(raw["baseline_contracts"], "baseline_contracts")
    authorization = _require_mapping(
        raw["lab_active_authorization"],
        "lab_active_authorization",
    )
    firewall = _require_mapping(raw["firewall"], "firewall")

    _validate_protocol_sections(
        source=source,
        story=story,
        fixture=fixture,
        formal=formal,
        baselines=baselines,
        authorization=authorization,
        firewall=firewall,
    )
    raw_subjects = fixture["subjects"]
    if not isinstance(raw_subjects, list):
        raise ValueError("development_fixture.subjects must be an array")
    subject_specs = tuple(_parse_subject_spec(item) for item in raw_subjects)
    return RelationshipP4LongitudinalCanaryContract(
        protocol_sha256=sha256_json(raw),
        source_package_name=_require_text(source["package_name"], "package_name"),
        source_dataset_fingerprint=_require_text(
            source["dataset_fingerprint"],
            "dataset_fingerprint",
        ),
        evidence_role=_require_text(source["evidence_role"], "evidence_role"),
        story_title=_require_text(story["title"], "story.title"),
        story_plain_language_claim=_require_text(
            story["plain_language_claim"],
            "story.plain_language_claim",
        ),
        subject_specs=subject_specs,
        phase_ids=_require_text_tuple(fixture["phase_ids"], "phase_ids"),
        reactive_seed_namespace=_require_text(
            fixture["reactive_seed_namespace"],
            "reactive_seed_namespace",
        ),
        required_arms=_require_text_tuple(formal["required_arms"], "required_arms"),
        formal_minimum_independent_subjects=_require_int(
            formal["minimum_independent_subjects"],
            "minimum_independent_subjects",
        ),
        formal_minimum_context_tokens=_require_int(
            formal["minimum_public_context_tokens_at_final_decision"],
            "minimum_public_context_tokens_at_final_decision",
        ),
        authorization_scope=_require_text(authorization["scope"], "scope"),
        allowed_policy_artifact_id=_require_text(
            authorization["allowed_policy_artifact_id"],
            "allowed_policy_artifact_id",
        ),
        allowed_policy_artifact_version=_require_int(
            authorization["allowed_policy_artifact_version"],
            "allowed_policy_artifact_version",
        ),
        claim_boundary=_require_text(raw["claim_boundary"], "claim_boundary"),
        schema_version=_require_text(raw["schema_version"], "schema_version"),
    )


def load_relationship_p4_longitudinal_canary_view(
    protocol_path: pathlib.Path | None = None,
) -> RelationshipP4LongitudinalCanaryView:
    contract = load_relationship_p4_longitudinal_canary_contract(protocol_path)
    dataset = load_relationship_transfer_dataset(
        package_name=contract.source_package_name,
    )
    _validate_dataset_lineage(contract, dataset)
    observations_by_hash = _observations_by_hash(dataset)
    subjects = tuple(_public_subject(contract, spec, observations_by_hash) for spec in contract.subject_specs)
    return RelationshipP4LongitudinalCanaryView(
        contract=contract,
        subjects=subjects,
    )


def load_relationship_p4_longitudinal_canary_evaluator_bundle(
    protocol_path: pathlib.Path | None = None,
) -> P4LongitudinalCanaryEvaluatorBundle:
    view = load_relationship_p4_longitudinal_canary_view(protocol_path)
    dataset = load_relationship_transfer_dataset(
        package_name=view.contract.source_package_name,
    )
    _validate_dataset_lineage(view.contract, dataset)
    observations_by_hash = _observations_by_hash(dataset)
    sessions: list[P4CanaryEvaluatorSession] = []
    for subject, spec in zip(
        view.subjects,
        view.contract.subject_specs,
        strict=True,
    ):
        for public_session, trajectory_hash in zip(
            subject.decision_sessions,
            spec.decision_source_trajectory_sha256,
            strict=True,
        ):
            observation = observations_by_hash[trajectory_hash]
            dynamic = dataset.dynamic_for_scene(observation.scene_id)
            sessions.append(
                P4CanaryEvaluatorSession(
                    session_id=public_session.session_id,
                    scene_id=observation.scene_id,
                    preferred_action_id=dynamic.preferred_action.value,
                    environment_seed=_environment_seed(
                        view.contract.reactive_seed_namespace,
                        public_session.session_id,
                    ),
                )
            )
    return P4LongitudinalCanaryEvaluatorBundle(
        protocol_sha256=view.contract.protocol_sha256,
        source_dataset_fingerprint=view.contract.source_dataset_fingerprint,
        sessions=tuple(sessions),
    )


def relationship_p4_lab_active_authorization(
    contract: RelationshipP4LongitudinalCanaryContract,
) -> RelationshipP4LabActiveAuthorization:
    unsigned = {
        "schema_version": P4_LONGITUDINAL_CANARY_AUTHORIZATION_SCHEMA_VERSION,
        "protocol_sha256": contract.protocol_sha256,
        "source_dataset_fingerprint": contract.source_dataset_fingerprint,
        "scope": contract.authorization_scope,
        "allowed_policy_artifact_id": contract.allowed_policy_artifact_id,
        "allowed_policy_artifact_version": contract.allowed_policy_artifact_version,
        "maximum_subjects": _DEVELOPMENT_SUBJECT_COUNT,
        "maximum_decision_sessions_per_subject": _DECISION_SESSIONS,
        "lab_typed_action_active_authorized": True,
        "environment_consumer_only": True,
        "expression_authorized": False,
        "production_authorized": False,
        "evaluation_feedback_to_learning": False,
        "oracle_action_authorized": False,
    }
    return RelationshipP4LabActiveAuthorization(
        authorization_id=sha256_json(unsigned),
        protocol_sha256=contract.protocol_sha256,
        source_dataset_fingerprint=contract.source_dataset_fingerprint,
        scope=contract.authorization_scope,
        allowed_policy_artifact_id=contract.allowed_policy_artifact_id,
        allowed_policy_artifact_version=contract.allowed_policy_artifact_version,
        maximum_subjects=_DEVELOPMENT_SUBJECT_COUNT,
        maximum_decision_sessions_per_subject=_DECISION_SESSIONS,
    )


def authorize_relationship_p4_lab_action_advisory(
    decision: RelationshipActionGateDecision,
    *,
    authorization: RelationshipP4LabActiveAuthorization,
) -> TemporalActionAdvisoryProposal:
    """Authorize one non-oracle gate decision for reactive-environment use only."""

    if decision.evaluator_only or decision.mode is RelationshipActionGateMode.ORACLE:
        raise ValueError("P4.1 Lab ACTIVE cannot authorize evaluator/oracle decisions")
    if decision.artifact_id != authorization.allowed_policy_artifact_id:
        raise ValueError("P4.1 gate artifact id is outside Lab authorization")
    if decision.artifact_version != authorization.allowed_policy_artifact_version:
        raise ValueError("P4.1 gate artifact version is outside Lab authorization")
    if authorization.production_authorized or authorization.expression_authorized:
        raise ValueError("P4.1 Lab authorization cannot reach product expression")
    advisory = temporal_action_advisory_from_gate_decision(decision)
    return replace(
        advisory,
        active_authorized=True,
        evidence_refs=(
            *advisory.evidence_refs,
            f"lab-authorization:{authorization.authorization_id}",
        ),
        rationale_codes=(
            *advisory.rationale_codes,
            "scope:offline-reactive-environment-only",
            "expression:forbidden",
            "production:forbidden",
        ),
    )


def prepare_relationship_p4_longitudinal_canary(
    protocol_path: pathlib.Path | None = None,
) -> P4LongitudinalCanaryPreparation:
    view = load_relationship_p4_longitudinal_canary_view(protocol_path)
    authorization = relationship_p4_lab_active_authorization(view.contract)
    return P4LongitudinalCanaryPreparation(
        protocol_sha256=view.contract.protocol_sha256,
        public_plan_sha256=view.public_plan_sha256,
        lab_authorization_id=authorization.authorization_id,
        required_arms=view.contract.required_arms,
        arm_statuses=(
            (
                P4LongitudinalCanaryArm.QWEN_STEELMAN_FULL_HISTORY.value,
                "blocked_by_p1k_p1m_zero_output_rule",
            ),
            (
                P4LongitudinalCanaryArm.QWEN_STEELMAN_SELECTIVE_RAG.value,
                "blocked_by_p1k_p1m_zero_output_rule",
            ),
            (
                P4LongitudinalCanaryArm.VOLVENCE_CLOSED_LOOP.value,
                "lab_mechanism_ready",
            ),
            (
                P4LongitudinalCanaryArm.VOLVENCE_TYPED_NOOP_CONTROL.value,
                "lab_mechanism_ready",
            ),
        ),
        development_subject_count=len(view.subjects),
        formal_minimum_independent_subjects=(view.contract.formal_minimum_independent_subjects),
        formal_minimum_context_tokens=view.contract.formal_minimum_context_tokens,
        model_output_count=0,
        formal_evidence_authorized=False,
        correction_redaction_owner_status=("implemented_p4_2_engineering_drill_available"),
        claim_boundary=view.contract.claim_boundary,
    )


async def run_relationship_p4_preference_mutation_drill(
    protocol_path: pathlib.Path | None = None,
) -> P4PreferenceMutationDrillReport:
    """Exercise correction, redaction, restart, and anti-resurrection in P4.2."""

    view = load_relationship_p4_longitudinal_canary_view(protocol_path)
    subject = view.subjects[0]
    persistence_snapshot = None
    process_restart_count = 0
    for onboarding in subject.onboarding_sessions:
        store = SocialRecordStore()
        if persistence_snapshot is not None:
            store.hydrate_from_persistence(persistence_snapshot)
            process_restart_count += 1
        evidence = _P4CanaryOwnerEvidence(
            subject_id=subject.subject_id,
            event_id=onboarding.event_id,
            source_turn=onboarding.session_index,
            observation_summary=onboarding.observation_summary,
            action_id=onboarding.action_id,
            observed_outcome_id=onboarding.observed_outcome_id,
            reaction_summary=onboarding.reaction_summary,
            observation_ref=onboarding.observation_ref,
        )
        snapshot = (
            await PreferenceAboutOtherModule(
                proposal_runtime=_P4CanaryEvidenceProposalRuntime(evidence),
                user_input=evidence.observation_summary,
                turn_index=evidence.source_turn,
                wiring_level=WiringLevel.SHADOW,
                record_store=store,
                action_outcome_evidence=evidence.to_owner_evidence(),
            ).process({})
        ).value
        if not isinstance(snapshot, PreferenceAboutOtherSnapshot):
            raise TypeError("P4.2 onboarding owner published unexpected snapshot")
        persistence_snapshot = store.export_persistence_snapshot()
    if persistence_snapshot is None:
        raise RuntimeError("P4.2 mutation drill has no onboarding state")

    pending_store = SocialRecordStore()
    pending_store.hydrate_from_persistence(persistence_snapshot)
    process_restart_count += 1
    initial_request = subject.decision_sessions[0].to_forecast_request(
        subject_scope=subject.subject_scope,
    )
    initial_snapshot = (
        await PreferenceAboutOtherModule(
            turn_index=initial_request.turn_index,
            wiring_level=WiringLevel.SHADOW,
            record_store=pending_store,
            action_forecast_runtime=BoundedRelationshipPreferenceForecastRuntime(),
            action_forecast_request=initial_request,
        ).process({})
    ).value
    if len(initial_snapshot.action_forecasts) != 1:
        raise RuntimeError("P4.2 mutation drill requires one pending forecast")

    correction_store = SocialRecordStore()
    correction_store.hydrate_from_persistence(pending_store.export_persistence_snapshot())
    process_restart_count += 1
    correction_target = subject.onboarding_sessions[0].to_owner_evidence()
    corrected_outcome_id = next(
        outcome_id for outcome_id in _OUTCOME_IDS if outcome_id != correction_target.observed_outcome_id
    )
    corrected_evidence = replace(
        correction_target,
        observation_summary="P4.2 user-corrected relationship observation.",
        observed_outcome_id=corrected_outcome_id,
        reaction_summary="P4.2 user-corrected relationship reaction.",
        evidence_refs=("p4.2:user-correction:20",),
    )
    correction = PreferenceActionOutcomeMutation(
        mutation_id="p4.2:correction:subject-01:event-00",
        target_evidence_id=correction_target.evidence_id,
        expected_evidence_sha256=preference_action_outcome_evidence_sha256(correction_target),
        operation=PreferenceActionOutcomeMutationOperation.CORRECT,
        requested_turn=20,
        evidence_refs=("p4.2:console-command:20",),
        replacement=corrected_evidence,
    )
    correction_reader = _P4MutationDrillForecastRuntime(target_evidence_id=correction_target.evidence_id)
    correction_request = PreferenceActionForecastRequest(
        decision_id="p4.2:decision:after-correction",
        interlocutor_id=_INTERLOCUTOR_ID,
        current_observation="P4.2 typed recovery probe after correction.",
        observation_ref="p4.2:probe:20",
        candidate_action_ids=_ACTION_IDS,
        outcome_ids=_OUTCOME_IDS,
        turn_index=20,
        session_scope=subject.subject_scope,
    )
    correction_snapshot = (
        await PreferenceAboutOtherModule(
            turn_index=20,
            wiring_level=WiringLevel.SHADOW,
            record_store=correction_store,
            action_outcome_mutation=correction,
            action_forecast_runtime=correction_reader,
            action_forecast_request=correction_request,
        ).process({})
    ).value
    correction_receipt = correction_snapshot.action_outcome_mutation_receipts[-1]
    if correction_reader.reader_observed_evidence_sha256 is None:
        raise RuntimeError("P4.2 correction did not reach the forecast reader")

    corrected_store = SocialRecordStore()
    corrected_store.hydrate_from_persistence(correction_store.export_persistence_snapshot())
    process_restart_count += 1
    correction_persisted = any(item == corrected_evidence for item in corrected_store.preference_action_outcomes)

    redaction_target = subject.onboarding_sessions[1].to_owner_evidence()
    redaction = PreferenceActionOutcomeMutation(
        mutation_id="p4.2:redaction:subject-01:event-01",
        target_evidence_id=redaction_target.evidence_id,
        expected_evidence_sha256=preference_action_outcome_evidence_sha256(redaction_target),
        operation=PreferenceActionOutcomeMutationOperation.REDACT,
        requested_turn=21,
        evidence_refs=("p4.2:console-command:21",),
    )
    redaction_snapshot = (
        await PreferenceAboutOtherModule(
            turn_index=21,
            wiring_level=WiringLevel.SHADOW,
            record_store=corrected_store,
            action_outcome_mutation=redaction,
        ).process({})
    ).value
    redaction_receipt = redaction_snapshot.action_outcome_mutation_receipts[-1]
    redacted_persistence = corrected_store.export_persistence_snapshot()
    serialized_persistence = json.dumps(
        redacted_persistence.payload,
        ensure_ascii=False,
        sort_keys=True,
    )

    redacted_store = SocialRecordStore()
    redacted_store.hydrate_from_persistence(redacted_persistence)
    process_restart_count += 1
    redaction_absent = (
        redaction_target.observation_summary not in serialized_persistence
        and redaction_target.reaction_summary not in serialized_persistence
        and all(item.evidence_id != redaction_target.evidence_id for item in redacted_store.preference_action_outcomes)
        and all(item.record_id != redaction_target.evidence_id for item in redacted_store.tom_records(_PREFERENCE_SLOT))
    )

    stale_evidence = _P4CanaryOwnerEvidence(
        subject_id=subject.subject_id,
        event_id=redaction_target.evidence_id,
        source_turn=redaction_target.source_turn,
        observation_summary=redaction_target.observation_summary,
        action_id=redaction_target.action_id,
        observed_outcome_id=redaction_target.observed_outcome_id,
        reaction_summary=redaction_target.reaction_summary,
        observation_ref=redaction_target.evidence_refs[0],
    )
    stale_owner = PreferenceAboutOtherModule(
        proposal_runtime=_P4CanaryEvidenceProposalRuntime(stale_evidence),
        user_input=stale_evidence.observation_summary,
        turn_index=stale_evidence.source_turn,
        wiring_level=WiringLevel.SHADOW,
        record_store=redacted_store,
        action_outcome_evidence=stale_evidence.to_owner_evidence(),
    )
    try:
        await stale_owner.process({})
    except ValueError as exc:
        if "cannot be reintroduced" not in str(exc):
            raise RuntimeError("P4.2 tombstone failed with an unexpected error") from exc
        tombstone_enforced = True
    else:
        raise RuntimeError("P4.2 tombstone allowed redacted evidence to return")

    return P4PreferenceMutationDrillReport(
        protocol_sha256=view.contract.protocol_sha256,
        subject_scope=subject.subject_scope,
        correction_command_sha256=(preference_action_outcome_mutation_sha256(correction)),
        corrected_evidence_sha256=preference_action_outcome_evidence_sha256(corrected_evidence),
        reader_observed_evidence_sha256=(correction_reader.reader_observed_evidence_sha256),
        redaction_command_sha256=(preference_action_outcome_mutation_sha256(redaction)),
        correction_invalidated_forecast_count=len(correction_receipt.invalidated_forecast_ids),
        redaction_invalidated_forecast_count=len(redaction_receipt.invalidated_forecast_ids),
        correction_persisted_after_restart=correction_persisted,
        redaction_content_absent_after_restart=redaction_absent,
        redaction_tombstone_enforced=tombstone_enforced,
        process_restart_count=process_restart_count,
    )


async def run_relationship_p4_longitudinal_canary_development(
    protocol_path: pathlib.Path | None = None,
) -> RelationshipP4LongitudinalCanaryReport:
    """Run the two seen fixtures through closed-loop and typed-noop arms."""

    view = load_relationship_p4_longitudinal_canary_view(protocol_path)
    evaluator = load_relationship_p4_longitudinal_canary_evaluator_bundle(protocol_path)
    authorization = relationship_p4_lab_active_authorization(view.contract)
    dataset = load_relationship_transfer_dataset(
        package_name=view.contract.source_package_name,
    )
    environment = ReactiveRelationshipEnvironment(dataset)
    runs: list[P4CanaryArmRun] = []
    for subject in view.subjects:
        for arm in (
            P4LongitudinalCanaryArm.VOLVENCE_CLOSED_LOOP,
            P4LongitudinalCanaryArm.VOLVENCE_TYPED_NOOP_CONTROL,
        ):
            runs.append(
                await _run_subject_arm(
                    subject=subject,
                    evaluator=evaluator,
                    authorization=authorization,
                    environment=environment,
                    forecast_runtime=BoundedRelationshipPreferenceForecastRuntime(),
                    arm=arm,
                )
            )
    frozen_runs = tuple(runs)
    return RelationshipP4LongitudinalCanaryReport(
        protocol_sha256=view.contract.protocol_sha256,
        public_plan_sha256=view.public_plan_sha256,
        lab_authorization_id=authorization.authorization_id,
        runs=frozen_runs,
        arm_summaries=tuple(
            _summarize_arm_runs(frozen_runs, arm)
            for arm in (
                P4LongitudinalCanaryArm.VOLVENCE_CLOSED_LOOP,
                P4LongitudinalCanaryArm.VOLVENCE_TYPED_NOOP_CONTROL,
            )
        ),
        required_arm_statuses=(
            (
                P4LongitudinalCanaryArm.QWEN_STEELMAN_FULL_HISTORY.value,
                "not_run_blocked_by_p1k_p1m",
            ),
            (
                P4LongitudinalCanaryArm.QWEN_STEELMAN_SELECTIVE_RAG.value,
                "not_run_blocked_by_p1k_p1m",
            ),
            (
                P4LongitudinalCanaryArm.VOLVENCE_CLOSED_LOOP.value,
                "engineering_fixture_complete",
            ),
            (
                P4LongitudinalCanaryArm.VOLVENCE_TYPED_NOOP_CONTROL.value,
                "engineering_fixture_complete",
            ),
        ),
        model_output_count=0,
        expression_output_count=0,
        formal_evidence_authorized=False,
        verdict="engineering_mechanism_ready_formal_effect_not_run",
        claim_boundary=view.contract.claim_boundary,
    )


async def _run_subject_arm(
    *,
    subject: P4LongitudinalCanarySubject,
    evaluator: P4LongitudinalCanaryEvaluatorBundle,
    authorization: RelationshipP4LabActiveAuthorization,
    environment: ReactiveRelationshipEnvironment,
    forecast_runtime: PreferenceActionForecastRuntime,
    arm: P4LongitudinalCanaryArm,
) -> P4CanaryArmRun:
    mode = (
        RelationshipActionGateMode.LEARNED
        if arm is P4LongitudinalCanaryArm.VOLVENCE_CLOSED_LOOP
        else RelationshipActionGateMode.NOOP
    )
    mechanism = await run_relationship_p4_subject_mechanism(
        subject=subject,
        evaluator=evaluator,
        authorization=authorization,
        environment=environment,
        forecast_runtime=forecast_runtime,
        gate_mode=mode,
        apply_credit_to_gate=(
            arm is P4LongitudinalCanaryArm.VOLVENCE_CLOSED_LOOP
        ),
    )
    return P4CanaryArmRun(
        arm=arm,
        subject_scope=mechanism.subject_scope,
        traces=mechanism.traces,
        positive_outcome_count=mechanism.positive_outcome_count,
        preferred_action_match_count=mechanism.preferred_action_match_count,
        reversal_opportunity_count=mechanism.reversal_opportunity_count,
        reversal_match_count=mechanism.reversal_match_count,
        gate_update_count=mechanism.gate_update_count,
        process_restart_count=mechanism.process_restart_count,
    )


async def run_relationship_p4_subject_mechanism(
    *,
    subject: P4LongitudinalCanarySubject,
    evaluator: P4LongitudinalCanaryEvaluatorBundle,
    authorization: RelationshipP4LabActiveAuthorization,
    environment: ReactiveRelationshipEnvironment,
    forecast_runtime: PreferenceActionForecastRuntime,
    gate_mode: RelationshipActionGateMode,
    apply_credit_to_gate: bool,
) -> P4CanaryMechanismRun:
    """Execute one matched subject arm without assigning a comparison label."""

    if gate_mode is RelationshipActionGateMode.ORACLE:
        raise ValueError("P4.1 mechanism runner rejects evaluator oracle mode")
    if apply_credit_to_gate and gate_mode is not RelationshipActionGateMode.LEARNED:
        raise ValueError("P4.1 credit requires learned gate mode")
    persistence_snapshot = None
    restart_count = 0
    for onboarding in subject.onboarding_sessions:
        store = SocialRecordStore()
        if persistence_snapshot is not None:
            store.hydrate_from_persistence(persistence_snapshot)
            restart_count += 1
        evidence = _P4CanaryOwnerEvidence(
            subject_id=subject.subject_id,
            event_id=onboarding.event_id,
            source_turn=onboarding.session_index,
            observation_summary=onboarding.observation_summary,
            action_id=onboarding.action_id,
            observed_outcome_id=onboarding.observed_outcome_id,
            reaction_summary=onboarding.reaction_summary,
            observation_ref=onboarding.observation_ref,
        )
        owner = PreferenceAboutOtherModule(
            proposal_runtime=_P4CanaryEvidenceProposalRuntime(evidence),
            user_input=evidence.observation_summary,
            turn_index=evidence.source_turn,
            wiring_level=WiringLevel.SHADOW,
            record_store=store,
            action_outcome_evidence=evidence.to_owner_evidence(),
        )
        snapshot = (await owner.process({})).value
        if not isinstance(snapshot, PreferenceAboutOtherSnapshot):
            raise TypeError("P4.1 preference owner published unexpected snapshot")
        persistence_snapshot = store.export_persistence_snapshot()
    if persistence_snapshot is None:
        raise RuntimeError("P4.1 subject has no onboarding state")

    gate_checkpoint: RelationshipActionGateCheckpoint | None = None
    traces: list[P4CanarySessionTrace] = []
    condition_readouts: list[RelationshipConditionReadout | None] = []
    gate_audits: list[P4CanaryGateAudit] = []
    expected_actions: list[str] = []
    for public_session in subject.decision_sessions:
        store = SocialRecordStore()
        store.hydrate_from_persistence(persistence_snapshot)
        gate = RelationshipActionGate(checkpoint=gate_checkpoint)
        restart_count += 1
        pre_update_checkpoint = gate.export_checkpoint()
        pre_update_weights, pre_update_bias = gate.parameter_state

        forecast_owner = PreferenceAboutOtherModule(
            turn_index=public_session.action_turn_index,
            wiring_level=WiringLevel.SHADOW,
            record_store=store,
            action_forecast_runtime=forecast_runtime,
            action_forecast_request=public_session.to_forecast_request(
                subject_scope=subject.subject_scope,
            ),
        )
        forecast_snapshot = (await forecast_owner.process({})).value
        if not isinstance(forecast_snapshot, PreferenceAboutOtherSnapshot):
            raise TypeError("P4.1 forecast owner published unexpected snapshot")
        forecasts = tuple(
            item for item in forecast_snapshot.action_forecasts if item.decision_id == public_session.decision_id
        )
        if len(forecasts) != 1:
            raise RuntimeError("P4.1 owner must publish exactly one current forecast")
        forecast = forecasts[0]
        decision = gate.decide(forecast, mode=gate_mode)
        advisory = authorize_relationship_p4_lab_action_advisory(
            decision,
            authorization=authorization,
        )
        temporal_snapshot = await _apply_lab_advisory(advisory)
        exposed_action_id = temporal_snapshot.value.active_abstract_action
        if exposed_action_id != decision.selected_action_id:
            raise RuntimeError("P4.1 temporal owner did not expose the gate action")

        evaluator_session = evaluator.session(public_session.session_id)
        expected_actions.append(evaluator_session.preferred_action_id)
        reactive_outcome = environment.settle(
            scene_id=evaluator_session.scene_id,
            decision_id=public_session.decision_id,
            action=RelationshipAction(exposed_action_id),
            seed=evaluator_session.environment_seed,
        )
        external_evidence = DialogueExternalOutcomeEvidence(
            evidence_id=(
                f"p4-canary-environment:{public_session.decision_id}:{reactive_outcome.environment_evidence_ref}"
            ),
            turn_index=public_session.outcome_turn_index,
            kind=reactive_outcome.typed_outcome,
            source=DialogueExternalOutcomeEvidenceSource.ENVIRONMENT,
            confidence=1.0,
            evidence_ref=reactive_outcome.environment_evidence_ref,
            description="Offline reactive-environment relationship outcome.",
            session_scope=subject.subject_scope,
            action_turn_index=public_session.action_turn_index,
            forecast_id=forecast.forecast_id,
            decision_id=forecast.decision_id,
            action_id=exposed_action_id,
        )
        external_owner = DialogueExternalOutcomeModule(wiring_level=WiringLevel.ACTIVE)
        external_owner.set_turn_index(public_session.outcome_turn_index)
        external_owner.append_evidence(external_evidence)
        external_snapshot = await external_owner.process({})

        outcome_record_id = f"p4-canary-owner-outcome:{subject.subject_scope}:{public_session.decision_index}"
        owner_evidence = _P4CanaryOwnerEvidence(
            subject_id=subject.subject_id,
            event_id=outcome_record_id,
            source_turn=public_session.outcome_turn_index,
            observation_summary=public_session.current_observation,
            action_id=exposed_action_id,
            observed_outcome_id=reactive_outcome.typed_outcome.value,
            reaction_summary=reactive_outcome.rendered_user_reaction,
            observation_ref=reactive_outcome.environment_evidence_ref,
        )
        settlement_owner = PreferenceAboutOtherModule(
            proposal_runtime=_P4CanaryEvidenceProposalRuntime(owner_evidence),
            user_input=owner_evidence.observation_summary,
            turn_index=owner_evidence.source_turn,
            wiring_level=WiringLevel.SHADOW,
            record_store=store,
            action_outcome_evidence=owner_evidence.to_owner_evidence(),
        )
        settled_snapshot = await settlement_owner.process({"dialogue_external_outcome": external_snapshot})
        social_pe_snapshot = await SocialPredictionErrorModule(wiring_level=WiringLevel.ACTIVE).process(
            {"preference_about_other": settled_snapshot}
        )
        credits = derive_preference_action_forecast_credit_records(
            settlements=settled_snapshot.value.forecast_settlements,
            social_errors=social_pe_snapshot.value.errors,
            settled_at_turn=public_session.outcome_turn_index,
            timestamp_ms=public_session.outcome_turn_index * 1000,
        )
        if len(credits) != 1:
            raise RuntimeError("P4.1 settlement must derive exactly one PE credit")
        credit_applied = apply_credit_to_gate
        if credit_applied:
            gate.observe_credit(credits[0])

        persistence_snapshot = store.export_persistence_snapshot()
        gate_checkpoint = gate.export_checkpoint()
        post_update_weights, post_update_bias = gate.parameter_state
        condition_readouts.append(forecast.condition_readout)
        gate_audits.append(
            P4CanaryGateAudit(
                subject_scope=subject.subject_scope,
                session_id=public_session.session_id,
                decision_id=public_session.decision_id,
                forecast_id=forecast.forecast_id,
                gate_mode=gate_mode,
                gate_action=decision.gate_action.value,
                steer_probability=decision.steer_probability,
                features=decision.features,
                pre_update_weights=pre_update_weights,
                pre_update_bias=pre_update_bias,
                pre_update_count=pre_update_checkpoint.update_count,
                pre_update_state_sha256=(
                    pre_update_checkpoint.checkpoint_sha256
                ),
                credit_record_id=credits[0].record_id,
                credit_level=credits[0].level,
                credit_track=credits[0].track.value,
                credit_source_event=credits[0].source_event,
                credit_environment_outcome_id=(
                    credits[0].environment_outcome_id
                ),
                credit_value=credits[0].credit_value,
                credit_applied_to_gate=credit_applied,
                post_update_weights=post_update_weights,
                post_update_bias=post_update_bias,
                post_update_count=gate_checkpoint.update_count,
                post_update_state_sha256=gate_checkpoint.checkpoint_sha256,
            )
        )
        traces.append(
            P4CanarySessionTrace(
                subject_scope=subject.subject_scope,
                session_id=public_session.session_id,
                phase_id=public_session.phase_id,
                decision_id=public_session.decision_id,
                forecast_id=forecast.forecast_id,
                recommended_action_id=forecast.recommended_action_id,
                gate_action=decision.gate_action.value,
                exposed_action_id=exposed_action_id,
                temporal_status=temporal_snapshot.value.action_advisory_status.value,
                observed_outcome_id=reactive_outcome.typed_outcome.value,
                preferred_action_match=(exposed_action_id == evaluator_session.preferred_action_id),
                credit_value=credits[0].credit_value,
                credit_applied_to_gate=credit_applied,
                owner_persistence_sha256=sha256_json(persistence_snapshot.payload),
                gate_checkpoint_sha256=gate_checkpoint.checkpoint_sha256,
            )
        )

    reversal_indices = tuple(
        index for index in range(1, len(expected_actions)) if expected_actions[index] != expected_actions[index - 1]
    )
    reversal_matches = sum(traces[index].preferred_action_match for index in reversal_indices)
    return P4CanaryMechanismRun(
        subject_scope=subject.subject_scope,
        traces=tuple(traces),
        gate_mode=gate_mode,
        credit_applied_to_gate=apply_credit_to_gate,
        condition_readouts=tuple(condition_readouts),
        gate_audits=tuple(gate_audits),
        positive_outcome_count=sum(trace.observed_outcome_id in _POSITIVE_OUTCOMES for trace in traces),
        preferred_action_match_count=sum(trace.preferred_action_match for trace in traces),
        reversal_opportunity_count=len(reversal_indices),
        reversal_match_count=reversal_matches,
        gate_update_count=(gate_checkpoint.update_count if gate_checkpoint is not None else 0),
        process_restart_count=restart_count,
    )


def _summarize_arm_runs(
    runs: tuple[P4CanaryArmRun, ...],
    arm: P4LongitudinalCanaryArm,
) -> P4CanaryArmSummary:
    matching = tuple(run for run in runs if run.arm is arm)
    return P4CanaryArmSummary(
        arm=arm,
        completed_subject_count=len(matching),
        typed_outcome_count=sum(len(run.traces) for run in matching),
        positive_outcome_count=sum(run.positive_outcome_count for run in matching),
        preferred_action_match_count=sum(run.preferred_action_match_count for run in matching),
        reversal_opportunity_count=sum(run.reversal_opportunity_count for run in matching),
        reversal_match_count=sum(run.reversal_match_count for run in matching),
        gate_update_count=sum(run.gate_update_count for run in matching),
    )


async def _apply_lab_advisory(advisory: TemporalActionAdvisoryProposal):
    module = TrackTemporalModule(
        track=Track.SELF,
        policy=PlaceholderTemporalPolicy(),
        wiring_level=WiringLevel.ACTIVE,
        action_advisory=advisory,
        action_advisory_level=WiringLevel.ACTIVE,
    )
    snapshot = await module.process_standalone(
        substrate_snapshot=SubstrateSnapshot(
            model_id="p4-longitudinal-canary-frozen-placeholder",
            is_frozen=True,
            surface_kind=SurfaceKind.PLACEHOLDER,
            token_logits=(),
            feature_surface=(),
            residual_activations=(),
            residual_sequence=(),
            unavailable_fields=(),
            description="Offline P4.1 typed-action exposure substrate.",
        )
    )
    if snapshot.value.action_advisory_status is not TemporalActionAdvisoryStatus.APPLIED:
        raise RuntimeError("P4.1 Lab advisory was not applied")
    return snapshot


def write_relationship_p4_canary_artifact(
    output_path: pathlib.Path,
    payload: Mapping[str, object],
) -> str:
    """Write one create-only canonical JSON artifact and return its digest."""

    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    file_text = f"{serialized}\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(file_text)
    return hashlib.sha256(file_text.encode("utf-8")).hexdigest()


def render_relationship_p4_canary_markdown(
    report: RelationshipP4LongitudinalCanaryReport,
) -> str:
    """Render the outcome-level canary without raw dialogue or hidden truth."""

    summaries = {item.arm: item for item in report.arm_summaries}
    closed_loop = summaries[P4LongitudinalCanaryArm.VOLVENCE_CLOSED_LOOP]
    noop = summaries[P4LongitudinalCanaryArm.VOLVENCE_TYPED_NOOP_CONTROL]
    return "\n".join(
        (
            "# P4.1 长程关系 canary（工程 fixture）",
            "",
            f"判词：`{report.verdict}`",
            "",
            "| 臂 | 正向 typed outcome | 命中事后 evaluator 动作 | 命中需求反转 | gate 更新 |",
            "|---|---:|---:|---:|---:|",
            (
                "| Volvence closed-loop | "
                f"{closed_loop.positive_outcome_count}/{closed_loop.typed_outcome_count} | "
                f"{closed_loop.preferred_action_match_count}/{closed_loop.typed_outcome_count} | "
                f"{closed_loop.reversal_match_count}/{closed_loop.reversal_opportunity_count} | "
                f"{closed_loop.gate_update_count} |"
            ),
            (
                "| Typed no-op control | "
                f"{noop.positive_outcome_count}/{noop.typed_outcome_count} | "
                f"{noop.preferred_action_match_count}/{noop.typed_outcome_count} | "
                f"{noop.reversal_match_count}/{noop.reversal_opportunity_count} | "
                f"{noop.gate_update_count} |"
            ),
            "| Qwen full-history steelman | 未运行 | 未运行 | 未运行 | 不适用 |",
            "| Qwen selective-RAG steelman | 未运行 | 未运行 | 未运行 | 不适用 |",
            "",
            (
                "这说明 typed 因果闭环和 no-op 对照已经会真实分叉；它也清楚暴露了默认 "
                "reader/gate 目前只有 "
                f"{closed_loop.preferred_action_match_count}/"
                f"{closed_loop.typed_outcome_count} 动作命中、"
                f"{closed_loop.reversal_match_count}/"
                f"{closed_loop.reversal_opportunity_count} 反转命中。"
            ),
            "",
            "边界：使用的是已见 v3 工程 fixture，独立 subject=0，Qwen output=0，"
            "expression output=0；不能据此声称 Volvence advantage、真人效果、"
            "production ACTIVE 或四能力成立。",
            "P4.2 preference-action correction/redaction owner 与独立恢复 drill 已实现；"
            "它只证明纠删机械传播，不改变本报告的 formal/effect 边界。",
            "",
        )
    )


def _load_protocol_raw(
    protocol_path: pathlib.Path | None,
) -> dict[str, Any]:
    path = pathlib.Path(protocol_path or relationship_p4_longitudinal_canary_protocol_path())
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("P4.1 protocol must contain a JSON object")
    _require_exact_keys(
        raw,
        {
            "schema_version",
            "source",
            "story",
            "development_fixture",
            "formal_pilot",
            "baseline_contracts",
            "lab_active_authorization",
            "firewall",
            "claim_boundary",
        },
        source="P4.1 protocol",
    )
    return raw


def _validate_protocol_sections(
    *,
    source: Mapping[str, Any],
    story: Mapping[str, Any],
    fixture: Mapping[str, Any],
    formal: Mapping[str, Any],
    baselines: Mapping[str, Any],
    authorization: Mapping[str, Any],
    firewall: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        source,
        {
            "package_name",
            "dataset_fingerprint",
            "evidence_role",
            "p1k_p1m_model_run_status",
        },
        source="P4.1 source",
    )
    if source["p1k_p1m_model_run_status"] != ("blocked_until_instrument_sequence_closes"):
        raise ValueError("P4.1 model-run prerequisite drifted")
    _require_exact_keys(
        story,
        {
            "title",
            "plain_language_claim",
            "abstract_conditions_visible_to_evaluator_only",
        },
        source="P4.1 story",
    )
    if story["abstract_conditions_visible_to_evaluator_only"] is not True:
        raise ValueError("P4.1 abstract-condition firewall drifted")
    _require_exact_keys(
        fixture,
        {
            "subject_count",
            "onboarding_sessions_per_subject",
            "decision_sessions_per_subject",
            "process_restart_between_sessions",
            "incremental_owner_state_only",
            "raw_history_replayed_to_volvence",
            "reactive_seed_namespace",
            "subjects",
            "phase_ids",
        },
        source="P4.1 development fixture",
    )
    expected_fixture = {
        "subject_count": _DEVELOPMENT_SUBJECT_COUNT,
        "onboarding_sessions_per_subject": _ONBOARDING_SESSIONS,
        "decision_sessions_per_subject": _DECISION_SESSIONS,
        "process_restart_between_sessions": True,
        "incremental_owner_state_only": True,
        "raw_history_replayed_to_volvence": False,
    }
    for field_name, expected in expected_fixture.items():
        if fixture[field_name] != expected:
            raise ValueError(f"P4.1 fixture {field_name} drifted")
    _require_exact_keys(
        formal,
        {
            "minimum_independent_subjects",
            "decision_sessions_per_subject",
            "minimum_public_context_tokens_at_final_decision",
            "same_frozen_substrate",
            "same_generation_config",
            "per_arm_reactive_trajectory",
            "process_restart_between_sessions",
            "required_arms",
            "primary_measure",
            "secondary_measures",
            "promotion_rule",
            "current_status",
        },
        source="P4.1 formal pilot",
    )
    expected_formal = {
        "decision_sessions_per_subject": _DECISION_SESSIONS,
        "same_frozen_substrate": True,
        "same_generation_config": True,
        "per_arm_reactive_trajectory": True,
        "process_restart_between_sessions": True,
        "current_status": "not_executed",
    }
    for field_name, expected in expected_formal.items():
        if formal[field_name] != expected:
            raise ValueError(f"P4.1 formal {field_name} drifted")
    _require_text(formal["primary_measure"], "primary_measure")
    _require_text(formal["promotion_rule"], "promotion_rule")
    _require_text_tuple(formal["secondary_measures"], "secondary_measures")
    _require_exact_keys(
        baselines,
        set(_REQUIRED_ARMS),
        source="P4.1 baseline contracts",
    )
    expected_baselines = {
        P4LongitudinalCanaryArm.QWEN_STEELMAN_FULL_HISTORY.value: {
            "memory_surface": "all_public_per_arm_history",
            "prompt_policy": "frozen_before_first_output",
            "pre_action_record_schema": (P4_LONGITUDINAL_CANARY_ARM_PREACTION_SCHEMA_VERSION),
            "typed_action_required_before_environment": True,
        },
        P4LongitudinalCanaryArm.QWEN_STEELMAN_SELECTIVE_RAG.value: {
            "memory_surface": "top_k_public_per_arm_history",
            "retrieval_policy": "frozen_semantic_retriever",
            "prompt_policy": "frozen_before_first_output",
            "pre_action_record_schema": (P4_LONGITUDINAL_CANARY_ARM_PREACTION_SCHEMA_VERSION),
            "typed_action_required_before_environment": True,
        },
        P4LongitudinalCanaryArm.VOLVENCE_CLOSED_LOOP.value: {
            "memory_surface": "hydrated_preference_owner_snapshot",
            "learning_surface": ("exact_settlement_to_social_pe_to_dedicated_credit"),
            "pre_action_record_schema": (P4_LONGITUDINAL_CANARY_ARM_PREACTION_SCHEMA_VERSION),
            "typed_action_required_before_environment": True,
        },
        P4LongitudinalCanaryArm.VOLVENCE_TYPED_NOOP_CONTROL.value: {
            "memory_surface": "hydrated_preference_owner_snapshot",
            "learning_surface": "credit_computed_but_not_applied_to_gate",
            "pre_action_record_schema": (P4_LONGITUDINAL_CANARY_ARM_PREACTION_SCHEMA_VERSION),
            "typed_action_required_before_environment": True,
        },
    }
    for arm_name, raw_contract in baselines.items():
        contract = _require_mapping(raw_contract, f"baseline_contracts.{arm_name}")
        if dict(contract) != expected_baselines[arm_name]:
            raise ValueError(f"P4.1 arm {arm_name} contract drifted")
    _require_exact_keys(
        authorization,
        {
            "scope",
            "allowed_policy_artifact_id",
            "allowed_policy_artifact_version",
            "lab_typed_action_active_authorized",
            "environment_consumer_only",
            "expression_authorized",
            "production_authorized",
            "evaluation_feedback_to_learning",
            "oracle_action_authorized",
        },
        source="P4.1 Lab ACTIVE authorization",
    )
    required_authorization = {
        "lab_typed_action_active_authorized": True,
        "environment_consumer_only": True,
        "expression_authorized": False,
        "production_authorized": False,
        "evaluation_feedback_to_learning": False,
        "oracle_action_authorized": False,
    }
    for field_name, expected in required_authorization.items():
        if authorization[field_name] is not expected:
            raise ValueError(f"P4.1 authorization {field_name} drifted")
    required_firewall = {
        "relationship_transfer_v4_loaded": False,
        "formal_hidden_test_opened": False,
        "new_qwen_output_allowed": False,
        "evaluator_truth_visible_to_sut": False,
        "judge_or_score_used_as_learning_signal": False,
        "lab_active_reachable_from_product": False,
        "preference_action_correction_redaction_owner_implemented": True,
        "raw_dialogue_written_to_report": False,
    }
    _require_exact_keys(
        firewall,
        set(required_firewall),
        source="P4.1 firewall",
    )
    if dict(firewall) != required_firewall:
        raise ValueError("P4.1 firewall is open")


def _parse_subject_spec(raw: object) -> P4LongitudinalCanarySubjectSpec:
    value = _require_mapping(raw, "development_fixture.subject")
    _require_exact_keys(
        value,
        {
            "subject_id",
            "onboarding_source_trajectory_sha256",
            "decision_source_trajectory_sha256",
        },
        source="P4.1 fixture subject",
    )
    return P4LongitudinalCanarySubjectSpec(
        subject_id=_require_text(value["subject_id"], "subject_id"),
        onboarding_source_trajectory_sha256=_require_text(
            value["onboarding_source_trajectory_sha256"],
            "onboarding_source_trajectory_sha256",
        ),
        decision_source_trajectory_sha256=_require_text_tuple(
            value["decision_source_trajectory_sha256"],
            "decision_source_trajectory_sha256",
        ),
    )


def _validate_dataset_lineage(
    contract: RelationshipP4LongitudinalCanaryContract,
    dataset: RelationshipTransferDataset,
) -> None:
    if dataset.package_name != contract.source_package_name:
        raise ValueError("P4.1 source package lineage mismatch")
    if dataset.dataset_fingerprint != contract.source_dataset_fingerprint:
        raise ValueError("P4.1 source dataset fingerprint mismatch")


def _observations_by_hash(
    dataset: RelationshipTransferDataset,
) -> dict[str, RelationshipObservation]:
    result = {item.trajectory_sha256: item for item in dataset.observations}
    if len(result) != len(dataset.observations):
        raise ValueError("P4.1 source trajectories must have unique hashes")
    return result


def _public_subject(
    contract: RelationshipP4LongitudinalCanaryContract,
    spec: P4LongitudinalCanarySubjectSpec,
    observations_by_hash: Mapping[str, RelationshipObservation],
) -> P4LongitudinalCanarySubject:
    try:
        onboarding_observation = observations_by_hash[spec.onboarding_source_trajectory_sha256]
    except KeyError as exc:
        raise ValueError("P4.1 onboarding trajectory is absent from source") from exc
    if len(onboarding_observation.histories) != _ONBOARDING_SESSIONS:
        raise ValueError("P4.1 onboarding source must carry four public histories")
    subject_scope = sha256_json(
        {
            "protocol_sha256": contract.protocol_sha256,
            "subject_id": spec.subject_id,
        }
    )
    onboarding = tuple(
        P4CanaryOnboardingSession(
            subject_id=spec.subject_id,
            session_id=f"{spec.subject_id}:onboarding:{index}",
            session_index=index,
            event_id=f"p4-canary:{subject_scope}:onboarding:{index}",
            observation_summary=history.user_utterance,
            action_id=history.assistant_action.value,
            observed_outcome_id=history.typed_outcome.value,
            reaction_summary=history.user_reaction,
            observation_ref=(f"p4-canary-public:{contract.protocol_sha256}:onboarding:{subject_scope}:{index}"),
        )
        for index, history in enumerate(onboarding_observation.histories)
    )
    decisions: list[P4CanaryDecisionSession] = []
    for index, trajectory_hash in enumerate(spec.decision_source_trajectory_sha256):
        try:
            observation = observations_by_hash[trajectory_hash]
        except KeyError as exc:
            raise ValueError("P4.1 decision trajectory is absent from source") from exc
        action_turn = _ONBOARDING_SESSIONS + index * 2
        session_id = f"{spec.subject_id}:decision:{index}"
        decisions.append(
            P4CanaryDecisionSession(
                subject_id=spec.subject_id,
                session_id=session_id,
                decision_index=index,
                action_turn_index=action_turn,
                outcome_turn_index=action_turn + 1,
                decision_id=f"{session_id}:pre-action",
                phase_id=contract.phase_ids[index],
                probe_surface_family=observation.probe_surface_family,
                current_observation=observation.current_input,
                observation_ref=(f"p4-canary-public:{contract.protocol_sha256}:decision:{subject_scope}:{index}"),
                source_trajectory_sha256=trajectory_hash,
            )
        )
    return P4LongitudinalCanarySubject(
        subject_id=spec.subject_id,
        subject_scope=subject_scope,
        onboarding_sessions=onboarding,
        decision_sessions=tuple(decisions),
    )


def _assert_no_public_truth_leakage(
    view: RelationshipP4LongitudinalCanaryView,
) -> None:
    payload = {
        "protocol_sha256": view.contract.protocol_sha256,
        "subjects": [subject.to_sut_payload() for subject in view.subjects],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    leaked = sorted(key for key in _FORBIDDEN_PUBLIC_KEYS if f'"{key}"' in serialized)
    if leaked:
        raise ValueError(f"P4.1 public view leaks evaluator keys: {leaked!r}")


def _environment_seed(seed_namespace: str, session_id: str) -> int:
    digest = hashlib.sha256(f"{seed_namespace}:{session_id}:reactive-environment".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _require_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} keys must be strings")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    source: str,
) -> None:
    missing = sorted(expected.difference(value))
    extra = sorted(set(value).difference(expected))
    if missing or extra:
        raise ValueError(f"{source} fields drifted; missing={missing}, extra={extra}")


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_text_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty array")
    items = tuple(_require_text(item, field_name) for item in value)
    return items


def _require_unique_texts(values: Sequence[str], field_name: str) -> None:
    if any(not item.strip() for item in values):
        raise ValueError(f"{field_name} must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique strings")


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_sha256(value: object, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase sha256 digest")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare or run the P4 longitudinal relationship canary.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run-development", action="store_true")
    mode.add_argument("--run-mutation-drill", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args()
    if args.prepare:
        if args.format != "json":
            parser.error("--prepare only supports --format json")
        payload = prepare_relationship_p4_longitudinal_canary().to_payload()
        rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    elif args.run_mutation_drill:
        if args.format != "json":
            parser.error("--run-mutation-drill only supports --format json")
        report = asyncio.run(run_relationship_p4_preference_mutation_drill())
        payload = report.to_payload()
        rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    else:
        report = asyncio.run(run_relationship_p4_longitudinal_canary_development())
        payload = report.to_payload()
        rendered = (
            render_relationship_p4_canary_markdown(report)
            if args.format == "markdown"
            else json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
        )
    if args.output is None:
        print(rendered)
    else:
        if args.format != "json":
            parser.error("--output currently requires --format json")
        digest = write_relationship_p4_canary_artifact(args.output, payload)
        print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "P4_LONGITUDINAL_CANARY_AUTHORIZATION_SCHEMA_VERSION",
    "P4_LONGITUDINAL_CANARY_ARM_PREACTION_SCHEMA_VERSION",
    "P4_LONGITUDINAL_CANARY_PREPARATION_SCHEMA_VERSION",
    "P4_LONGITUDINAL_CANARY_REPORT_SCHEMA_VERSION",
    "P4_LONGITUDINAL_CANARY_SCHEMA_VERSION",
    "P4_LONGITUDINAL_CANARY_SCOPE",
    "P4_PREFERENCE_MUTATION_DRILL_SCHEMA_VERSION",
    "P4CanaryArmRun",
    "P4CanaryArmPreActionRecord",
    "P4CanaryArmSummary",
    "P4CanaryDecisionSession",
    "P4CanaryEvaluatorSession",
    "P4CanaryOnboardingSession",
    "P4CanarySessionTrace",
    "P4LongitudinalCanaryArm",
    "P4LongitudinalCanaryEvaluatorBundle",
    "P4LongitudinalCanaryPreparation",
    "P4LongitudinalCanarySubject",
    "P4LongitudinalCanarySubjectSpec",
    "P4PreferenceMutationDrillReport",
    "RelationshipP4LabActiveAuthorization",
    "RelationshipP4LongitudinalCanaryContract",
    "RelationshipP4LongitudinalCanaryReport",
    "RelationshipP4LongitudinalCanaryView",
    "authorize_relationship_p4_lab_action_advisory",
    "load_relationship_p4_longitudinal_canary_contract",
    "load_relationship_p4_longitudinal_canary_evaluator_bundle",
    "load_relationship_p4_longitudinal_canary_view",
    "main",
    "prepare_relationship_p4_longitudinal_canary",
    "render_relationship_p4_canary_markdown",
    "relationship_p4_lab_active_authorization",
    "relationship_p4_longitudinal_canary_protocol_path",
    "run_relationship_p4_longitudinal_canary_development",
    "run_relationship_p4_preference_mutation_drill",
    "write_relationship_p4_canary_artifact",
]
