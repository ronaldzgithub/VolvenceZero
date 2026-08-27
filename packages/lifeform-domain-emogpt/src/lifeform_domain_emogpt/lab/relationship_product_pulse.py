"""Frozen two-phase relationship-action pulse for product-shaped Lab runs.

The pre-action phase can only consume public typed observation data, an
owner-published persistence snapshot, a gate checkpoint, and an injected
forecast collaborator.  It freezes the owner forecast and gate decision before
an outcome exists, then applies the selected *typed* action only to the
offline-environment temporal surface.  It never authorizes expression or a
production runtime and it deliberately accepts only a placeholder substrate.

The settlement phase consumes a separately published typed external outcome,
performs the exact owner join, lifts the owner settlement to social PE, derives
the dedicated PE credit, and optionally applies that credit to the pending
gate decision.  Evaluator truth, judge scores, and preferred-action labels are
not fields of either public input contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import InitVar, dataclass, field, replace
from enum import Enum

from volvence_zero.credit import (
    CreditRecord,
    RelationshipActionCommonBaselineCredit,
    derive_preference_action_common_baseline_credit_records,
    derive_preference_action_forecast_credit_records,
)
from volvence_zero.dialogue_external_outcome import DialogueExternalOutcomeModule
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeSnapshot,
)
from volvence_zero.memory import Track
from volvence_zero.owner_hydration import OwnerPersistenceSnapshot
from volvence_zero.runtime import Snapshot, WiringLevel
from volvence_zero.semantic_state import (
    SemanticProposal,
    SemanticProposalBatch,
    SemanticProposalOperation,
    SemanticProposalRuntime,
)
from volvence_zero.social import (
    PreferenceAboutOtherModule,
    PreferenceActionForecastRequest,
    PreferenceActionForecastRuntime,
    SocialPredictionErrorModule,
    SocialRecordStore,
    replay_preference_action_forecast_publication_persistence,
    replay_preference_action_forecast_settlement_persistence,
    settle_preference_action_forecast,
    social_record_store_persistence_sha256,
)
from volvence_zero.social_cognition import (
    PreferenceAboutOtherSnapshot,
    PreferenceActionForecast,
    PreferenceActionForecastSettlement,
    PreferenceActionOutcomeEvidence,
    SocialPredictionErrorSnapshot,
    SocialPredictionKind,
    SocialScopeKind,
    preference_action_forecast_to_payload,
)
from volvence_zero.substrate import SubstrateSnapshot, SurfaceKind
from volvence_zero.temporal import PlaceholderTemporalPolicy, TrackTemporalModule
from volvence_zero.temporal_types import (
    TemporalAbstractionSnapshot,
    TemporalActionAdvisoryProposal,
    TemporalActionAdvisoryStatus,
)

from lifeform_domain_emogpt.relationship_action_contracts import (
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
    RelationshipAction,
)
from lifeform_domain_emogpt.relationship_action_gate import (
    RelationshipActionGate,
    RelationshipActionGateBatchDisposition,
    RelationshipActionGateCheckpoint,
    RelationshipActionGateDecision,
    RelationshipActionGateForcedExposure,
    RelationshipActionGateFrozenDecision,
    RelationshipActionGateFrozenPolicy,
    RelationshipActionGateMode,
    RelationshipActionGateUpdate,
    RelationshipGateAction,
    temporal_action_advisory_from_gate_decision,
)
from lifeform_domain_emogpt.relationship_action_gate_v2 import (
    RelationshipActionGateV2,
    RelationshipActionGateV2Artifact,
    RelationshipActionGateV2ArtifactKind,
    RelationshipActionGateV2AssignmentReceipt,
    RelationshipActionGateV2AssignmentRole,
    RelationshipActionGateV2BatchReceipt,
    RelationshipActionGateV2CreditBatch,
    RelationshipActionGateV2FederatedAssignmentScheduleArtifact,
    RelationshipActionGateV2FederatedCreditBatch,
    RelationshipActionGateV2FederatedMatchedTransitions,
    RelationshipActionGateV2FederatedTransition,
    RelationshipActionGateV2ForcedExposure,
    RelationshipActionGateV2FrozenDecision,
    RelationshipActionGateV2FrozenPolicy,
    RelationshipActionGateV2OnlineExposure,
    RelationshipActionGateV2OnlineSession,
    RelationshipActionGateV2OnlineTransition,
    commit_relationship_action_gate_v2_federated_matched_transitions,
    temporal_action_advisory_from_gate_v2_decision,
    temporal_action_advisory_from_gate_v2_online_exposure,
)


RELATIONSHIP_PRODUCT_PULSE_SCHEMA_VERSION = "relationship-product-pulse.v1"
RELATIONSHIP_PRODUCT_FROZEN_PULSE_SCHEMA_VERSION = (
    "relationship-product-frozen-pulse.v1"
)
RELATIONSHIP_PRODUCT_EXECUTOR_SCHEMA_VERSION = (
    "relationship-product-executor-receipt.v1"
)
RELATIONSHIP_PRODUCT_FORCED_COLLECTION_SCHEMA_VERSION = (
    "relationship-product-forced-collection.v1"
)
RELATIONSHIP_PRODUCT_FORCED_COLLECTION_SCHEDULE_SCHEMA_VERSION = (
    "relationship-product-forced-collection-schedule.v1"
)
_RELATIONSHIP_PRODUCT_FORCED_COLLECTION_SCHEDULE_ENTRY_SCHEMA_VERSION = (
    "relationship-product-forced-collection-schedule-entry.v1"
)
_PREFERENCE_SLOT = "preference_about_other"
_ACTION_IDS = tuple(action.value for action in RELATIONSHIP_ACTIONS)
_OUTCOME_IDS = tuple(outcome.value for outcome in RELATIONSHIP_OUTCOMES)
_EXECUTOR_COMMAND_PREFIX = "relationship-product-executor-command-sha256:"
_EXECUTOR_RECEIPT_PREFIX = "relationship-product-executor-receipt-sha256:"
_FORCED_COLLECTION_COMMAND_PREFIX = (
    "relationship-product-forced-collection-command-sha256:"
)
_FORCED_COLLECTION_AUTHORIZATION_PREFIX = (
    "relationship-product-forced-collection-authorization-sha256:"
)
_FORCED_COLLECTION_SCHEDULE_ENTRY_PREFIX = (
    "relationship-product-forced-collection-schedule-entry-sha256:"
)
_FORCED_COLLECTION_SCHEDULE_PREFIX = (
    "relationship-product-forced-collection-schedule-sha256:"
)
_FORCED_COLLECTION_RECEIPT_PREFIX = (
    "relationship-product-forced-collection-receipt-sha256:"
)
RELATIONSHIP_PRODUCT_V2_FROZEN_PULSE_SCHEMA_VERSION = (
    "relationship-product-frozen-pulse.v2"
)
RELATIONSHIP_PRODUCT_V2_CONDENSED_THETA0_AUTHORIZATION_SCHEMA_VERSION = (
    "relationship-product-condensed-theta0-authorization.v2"
)
RELATIONSHIP_PRODUCT_V2_EXECUTOR_SCHEMA_VERSION = (
    "relationship-product-executor-receipt.v2"
)
RELATIONSHIP_PRODUCT_V2_ONLINE_PULSE_SCHEMA_VERSION = (
    "relationship-product-online-pulse.v2"
)
RELATIONSHIP_PRODUCT_V2_ONLINE_EXECUTOR_SCHEMA_VERSION = (
    "relationship-product-online-executor-receipt.v2"
)
RELATIONSHIP_PRODUCT_V2_FORCED_COLLECTION_SCHEMA_VERSION = (
    "relationship-product-forced-collection.v2"
)
RELATIONSHIP_PRODUCT_V2_GATE_TRANSITION_SCHEMA_VERSION = (
    "relationship-product-gate-transition.v2"
)
RELATIONSHIP_PRODUCT_V2_COLLECTED_BATCH_SCHEMA_VERSION = (
    "relationship-product-collected-credit-batch.v2"
)
RELATIONSHIP_PRODUCT_V2_COLLECTION_SEGMENT_SCHEMA_VERSION = (
    "relationship-product-collection-segment.v2"
)
RELATIONSHIP_PRODUCT_V2_SEGMENTED_COLLECTED_BATCH_SCHEMA_VERSION = (
    "relationship-product-segmented-collected-credit-batch.v2"
)
RELATIONSHIP_PRODUCT_V2_SEGMENTED_GATE_TRANSITION_SCHEMA_VERSION = (
    "relationship-product-segmented-gate-transition.v2"
)
RELATIONSHIP_PRODUCT_V2_SEGMENTED_MATCHED_TRANSITIONS_SCHEMA_VERSION = (
    "relationship-product-segmented-matched-gate-transitions.v2"
)
RELATIONSHIP_PRODUCT_V2_FEDERATED_COLLECTED_BATCH_SCHEMA_VERSION = (
    "relationship-product-federated-collected-credit-batch.v2"
)
RELATIONSHIP_PRODUCT_V2_FEDERATED_MATCHED_TRANSITIONS_SCHEMA_VERSION = (
    "relationship-product-federated-matched-gate-transitions.v2"
)
RELATIONSHIP_PRODUCT_V2_MATCHED_TRANSITIONS_SCHEMA_VERSION = (
    "relationship-product-matched-gate-transitions.v2"
)
_V2_FROZEN_AUTHORIZATION_PREFIX = (
    "relationship-product-v2-frozen-authorization-sha256:"
)
_V2_CONDENSED_THETA0_AUTHORIZATION_PREFIX = (
    "relationship-product-v2-condensed-theta0-authorization-sha256:"
)
_V2_FORCED_AUTHORIZATION_PREFIX = (
    "relationship-product-v2-forced-authorization-sha256:"
)
_V2_EXECUTOR_COMMAND_PREFIX = "relationship-product-v2-executor-command-sha256:"
_V2_EXECUTOR_RECEIPT_PREFIX = "relationship-product-v2-executor-receipt-sha256:"
_V2_ONLINE_AUTHORIZATION_PREFIX = (
    "relationship-product-v2-online-authorization-sha256:"
)
_V2_ONLINE_EXECUTOR_COMMAND_PREFIX = (
    "relationship-product-v2-online-executor-command-sha256:"
)
_V2_ONLINE_EXECUTOR_RECEIPT_PREFIX = (
    "relationship-product-v2-online-executor-receipt-sha256:"
)
_V2_FORCED_COMMAND_PREFIX = "relationship-product-v2-forced-command-sha256:"
_V2_FORCED_RECEIPT_PREFIX = "relationship-product-v2-forced-receipt-sha256:"
_V2_COLLECTED_BATCH_PREFIX = "relationship-product-v2-collected-batch-sha256:"
_V2_COLLECTION_SEGMENT_PREFIX = (
    "relationship-product-v2-collection-segment-sha256:"
)
_V2_SEGMENTED_COLLECTED_BATCH_PREFIX = (
    "relationship-product-v2-segmented-collected-batch-sha256:"
)
_V2_SEGMENTED_GATE_TRANSITION_PREFIX = (
    "relationship-product-v2-segmented-gate-transition-sha256:"
)
_V2_SEGMENTED_MATCHED_TRANSITIONS_PREFIX = (
    "relationship-product-v2-segmented-matched-transitions-sha256:"
)
_V2_FEDERATED_COLLECTED_BATCH_PREFIX = (
    "relationship-product-v2-federated-collected-batch-sha256:"
)
_V2_FEDERATED_MATCHED_TRANSITIONS_PREFIX = (
    "relationship-product-v2-federated-matched-transitions-sha256:"
)
_V2_GATE_TRANSITION_PREFIX = "relationship-product-v2-gate-transition-sha256:"
_V2_MATCHED_TRANSITIONS_PREFIX = (
    "relationship-product-v2-matched-transitions-sha256:"
)


@dataclass(frozen=True)
class RelationshipProductOnboardingInput:
    """One public, already-observed onboarding experience."""

    session_id: str
    session_index: int
    turn_index: int
    public_observation: str
    action_id: str
    observed_outcome_id: str
    reaction_summary: str
    evidence_ref: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("session_id", self.session_id),
            ("public_observation", self.public_observation),
            ("action_id", self.action_id),
            ("observed_outcome_id", self.observed_outcome_id),
            ("reaction_summary", self.reaction_summary),
            ("evidence_ref", self.evidence_ref),
        ):
            _require_text(value, field_name)
        for field_name, value in (
            ("session_index", self.session_index),
            ("turn_index", self.turn_index),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.action_id not in _ACTION_IDS:
            raise ValueError("onboarding action is outside the relationship surface")
        if self.observed_outcome_id not in _OUTCOME_IDS:
            raise ValueError(
                "onboarding outcome is outside the relationship surface"
            )

    @property
    def evidence_id(self) -> str:
        body = {
            "session_id": self.session_id,
            "session_index": self.session_index,
            "turn_index": self.turn_index,
            "public_observation": self.public_observation,
            "action_id": self.action_id,
            "observed_outcome_id": self.observed_outcome_id,
            "reaction_summary": self.reaction_summary,
            "evidence_ref": self.evidence_ref,
        }
        encoded = json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"relationship-product-onboarding:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class RelationshipProductOnboardingSnapshot:
    """Frozen owner publication and resumable state after onboarding."""

    onboarding: RelationshipProductOnboardingInput
    preference_snapshot: Snapshot[PreferenceAboutOtherSnapshot]
    owner_persistence_snapshot: OwnerPersistenceSnapshot
    schema_version: str = RELATIONSHIP_PRODUCT_PULSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_PRODUCT_PULSE_SCHEMA_VERSION:
            raise ValueError("relationship product onboarding schema mismatch")
        if not isinstance(
            self.preference_snapshot.value,
            PreferenceAboutOtherSnapshot,
        ):
            raise TypeError(
                "onboarding preference snapshot published an unexpected value"
            )
        matching = tuple(
            item
            for item in self.preference_snapshot.value.action_outcome_evidence
            if item.evidence_id == self.onboarding.evidence_id
        )
        if len(matching) != 1:
            raise ValueError(
                "onboarding owner snapshot must contain exactly one current evidence"
            )


@dataclass(frozen=True)
class RelationshipProductPulseAuthorization:
    """Narrow authorization for typed offline-environment action exposure."""

    authorization_id: str
    allowed_policy_artifact_id: str
    allowed_policy_artifact_version: int
    environment_consumer_only: bool = True
    expression_authorized: bool = False
    production_authorized: bool = False
    oracle_action_authorized: bool = False

    def __post_init__(self) -> None:
        _require_text(self.authorization_id, "authorization_id")
        _require_text(
            self.allowed_policy_artifact_id,
            "allowed_policy_artifact_id",
        )
        if (
            isinstance(self.allowed_policy_artifact_version, bool)
            or not isinstance(self.allowed_policy_artifact_version, int)
            or self.allowed_policy_artifact_version < 1
        ):
            raise ValueError("allowed_policy_artifact_version must be >= 1")
        if not self.environment_consumer_only:
            raise ValueError(
                "relationship product pulse authorization must be "
                "environment-consumer-only"
            )
        if (
            self.expression_authorized
            or self.production_authorized
            or self.oracle_action_authorized
        ):
            raise ValueError(
                "relationship product pulse authorization firewall is open"
            )


@dataclass(frozen=True)
class RelationshipProductFrozenPulseAuthorization:
    """Pin one exact immutable gate policy for an offline evaluation pulse."""

    pulse_authorization: RelationshipProductPulseAuthorization
    allowed_frozen_policy_id: str
    allowed_checkpoint_content_sha256: str
    schema_version: str = RELATIONSHIP_PRODUCT_FROZEN_PULSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(
            self.pulse_authorization,
            RelationshipProductPulseAuthorization,
        ):
            raise TypeError(
                "pulse_authorization must be RelationshipProductPulseAuthorization"
            )
        _require_text(self.allowed_frozen_policy_id, "allowed_frozen_policy_id")
        _require_sha256(
            self.allowed_checkpoint_content_sha256,
            "allowed_checkpoint_content_sha256",
        )
        if self.schema_version != RELATIONSHIP_PRODUCT_FROZEN_PULSE_SCHEMA_VERSION:
            raise ValueError("relationship product frozen authorization schema mismatch")

    @property
    def authorization_id(self) -> str:
        return self.pulse_authorization.authorization_id

    def validate_policy(
        self,
        policy: RelationshipActionGateFrozenPolicy,
    ) -> None:
        if not isinstance(policy, RelationshipActionGateFrozenPolicy):
            raise TypeError("policy must be RelationshipActionGateFrozenPolicy")
        if policy.theta0_artifact is None:
            raise ValueError("frozen product pulse requires an explicit theta0 policy")
        if policy.policy_id != self.allowed_frozen_policy_id:
            raise ValueError("frozen policy id is outside pulse authorization")
        if (
            policy.checkpoint.content_sha256
            != self.allowed_checkpoint_content_sha256
        ):
            raise ValueError("frozen checkpoint is outside pulse authorization")
        if (
            policy.artifact.artifact_id
            != self.pulse_authorization.allowed_policy_artifact_id
        ):
            raise ValueError("frozen policy artifact id is outside pulse authorization")
        if (
            policy.artifact.artifact_version
            != self.pulse_authorization.allowed_policy_artifact_version
        ):
            raise ValueError(
                "frozen policy artifact version is outside pulse authorization"
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "pulse_authorization": _pulse_authorization_to_payload(
                self.pulse_authorization
            ),
            "allowed_frozen_policy_id": self.allowed_frozen_policy_id,
            "allowed_checkpoint_content_sha256": (
                self.allowed_checkpoint_content_sha256
            ),
        }


class RelationshipProductExecutorDisposition(str, Enum):
    """The only treatment bit after a frozen intervention candidate exists."""

    APPLY_CANDIDATE = "apply_candidate"
    FORCE_STRICT_NOOP = "force_strict_noop"


class RelationshipProductExecutorStatus(str, Enum):
    APPLIED_CANDIDATE = "applied_candidate"
    GATE_NOOP = "gate_noop"
    STRICT_NOOP = "strict_noop"


class RelationshipProductForcedActionRole(str, Enum):
    """Frozen symbolic action role for matched collection."""

    OWNER_RECOMMENDATION = "owner_recommendation"
    NEUTRAL_NOOP = "neutral_noop"


@dataclass(frozen=True)
class RelationshipProductForcedCollectionScheduleEntry:
    """One symbolic forced action frozen before its forecast is published."""

    decision_id: str
    sequence_index: int
    forced_action_role: RelationshipProductForcedActionRole
    schema_version: str = (
        _RELATIONSHIP_PRODUCT_FORCED_COLLECTION_SCHEDULE_ENTRY_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        _require_text(self.decision_id, "decision_id")
        if (
            isinstance(self.sequence_index, bool)
            or not isinstance(self.sequence_index, int)
            or self.sequence_index < 0
        ):
            raise ValueError("sequence_index must be a non-negative integer")
        if not isinstance(
            self.forced_action_role,
            RelationshipProductForcedActionRole,
        ):
            raise TypeError(
                "forced_action_role must be RelationshipProductForcedActionRole"
            )
        if (
            self.schema_version
            != _RELATIONSHIP_PRODUCT_FORCED_COLLECTION_SCHEDULE_ENTRY_SCHEMA_VERSION
        ):
            raise ValueError("forced collection schedule entry schema mismatch")

    @property
    def entry_id(self) -> str:
        return (
            f"{_FORCED_COLLECTION_SCHEDULE_ENTRY_PREFIX}"
            f"{_canonical_sha256(self._core_payload())}"
        )

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "sequence_index": self.sequence_index,
            "forced_action_role": self.forced_action_role.value,
        }

    def to_payload(self) -> dict[str, object]:
        return {"entry_id": self.entry_id, **self._core_payload()}


@dataclass(frozen=True)
class RelationshipProductForcedCollectionScheduleArtifact:
    """Complete immutable schedule whose identity covers every entry."""

    entries: tuple[RelationshipProductForcedCollectionScheduleEntry, ...]
    schema_version: str = (
        RELATIONSHIP_PRODUCT_FORCED_COLLECTION_SCHEDULE_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or not self.entries:
            raise ValueError("forced collection schedule entries must be non-empty")
        if not all(
            isinstance(item, RelationshipProductForcedCollectionScheduleEntry)
            for item in self.entries
        ):
            raise TypeError(
                "forced collection schedule entries have an unexpected type"
            )
        if (
            self.schema_version
            != RELATIONSHIP_PRODUCT_FORCED_COLLECTION_SCHEDULE_SCHEMA_VERSION
        ):
            raise ValueError("forced collection schedule artifact schema mismatch")
        decision_ids = tuple(item.decision_id for item in self.entries)
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError(
                "forced collection schedule decision ids must be unique"
            )
        sequence_indices = tuple(item.sequence_index for item in self.entries)
        if sequence_indices != tuple(range(len(self.entries))):
            raise ValueError(
                "forced collection schedule sequence indices must be contiguous and "
                "ordered from zero"
            )

    @property
    def artifact_id(self) -> str:
        return (
            f"{_FORCED_COLLECTION_SCHEDULE_PREFIX}"
            f"{_canonical_sha256(self._core_payload())}"
        )

    def entry_for_decision(
        self,
        decision_id: str,
    ) -> RelationshipProductForcedCollectionScheduleEntry:
        _require_text(decision_id, "decision_id")
        matching = tuple(
            item for item in self.entries if item.decision_id == decision_id
        )
        if len(matching) != 1:
            raise ValueError(
                "forced collection schedule must contain exactly one decision entry"
            )
        return matching[0]

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "entries": [item.to_payload() for item in self.entries],
        }

    def to_payload(self) -> dict[str, object]:
        return {"artifact_id": self.artifact_id, **self._core_payload()}


@dataclass(frozen=True)
class RelationshipProductForcedCollectionAuthorization:
    """One frozen schedule entry authorized for a specific decision."""

    frozen_pulse_authorization: RelationshipProductFrozenPulseAuthorization
    schedule_artifact: RelationshipProductForcedCollectionScheduleArtifact
    decision_id: str
    schema_version: str = RELATIONSHIP_PRODUCT_FORCED_COLLECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(
            self.frozen_pulse_authorization,
            RelationshipProductFrozenPulseAuthorization,
        ):
            raise TypeError(
                "frozen_pulse_authorization must be "
                "RelationshipProductFrozenPulseAuthorization"
            )
        if not isinstance(
            self.schedule_artifact,
            RelationshipProductForcedCollectionScheduleArtifact,
        ):
            raise TypeError(
                "schedule_artifact must be "
                "RelationshipProductForcedCollectionScheduleArtifact"
            )
        _require_text(self.decision_id, "decision_id")
        if (
            self.schema_version
            != RELATIONSHIP_PRODUCT_FORCED_COLLECTION_SCHEMA_VERSION
        ):
            raise ValueError(
                "relationship product forced collection authorization schema mismatch"
            )
        self.schedule_artifact.entry_for_decision(self.decision_id)

    @property
    def schedule_entry_id(self) -> str:
        return self.schedule_entry.entry_id

    @property
    def schedule_entry(self) -> RelationshipProductForcedCollectionScheduleEntry:
        return self.schedule_artifact.entry_for_decision(self.decision_id)

    @property
    def forced_action_schedule_artifact_id(self) -> str:
        return self.schedule_artifact.artifact_id

    @property
    def sequence_index(self) -> int:
        return self.schedule_entry.sequence_index

    @property
    def forced_action_role(self) -> RelationshipProductForcedActionRole:
        return self.schedule_entry.forced_action_role

    @property
    def authorization_id(self) -> str:
        return (
            f"{_FORCED_COLLECTION_AUTHORIZATION_PREFIX}"
            f"{_canonical_sha256(self._core_payload())}"
        )

    def validate_policy(
        self,
        frozen_policy: RelationshipActionGateFrozenPolicy,
    ) -> None:
        self.frozen_pulse_authorization.validate_policy(frozen_policy)

    def validate_decision_id(self, decision_id: str) -> None:
        _require_text(decision_id, "decision_id")
        if decision_id != self.decision_id:
            raise ValueError(
                "forced collection decision is outside schedule authorization"
            )

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "frozen_pulse_authorization": self.frozen_pulse_authorization.to_payload(),
            "forced_action_schedule_artifact_id": (
                self.forced_action_schedule_artifact_id
            ),
            "schedule_entry_id": self.schedule_entry_id,
            "schedule_entry": self.schedule_entry.to_payload(),
        }

    def to_payload(self) -> dict[str, object]:
        return {"authorization_id": self.authorization_id, **self._core_payload()}


@dataclass(frozen=True)
class RelationshipProductForcedCollectionCommand:
    """Arm-independent command for one cold-theta0 collection exposure."""

    frozen_policy: RelationshipActionGateFrozenPolicy
    forced_exposure: RelationshipActionGateForcedExposure
    authorization: RelationshipProductForcedCollectionAuthorization
    owner_prestate_sha256: str
    schema_version: str = RELATIONSHIP_PRODUCT_FORCED_COLLECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.frozen_policy, RelationshipActionGateFrozenPolicy):
            raise TypeError("frozen_policy must be RelationshipActionGateFrozenPolicy")
        if not isinstance(
            self.forced_exposure,
            RelationshipActionGateForcedExposure,
        ):
            raise TypeError(
                "forced_exposure must be RelationshipActionGateForcedExposure"
            )
        if not isinstance(
            self.authorization,
            RelationshipProductForcedCollectionAuthorization,
        ):
            raise TypeError(
                "authorization must be "
                "RelationshipProductForcedCollectionAuthorization"
            )
        _require_sha256(self.owner_prestate_sha256, "owner_prestate_sha256")
        if (
            self.schema_version
            != RELATIONSHIP_PRODUCT_FORCED_COLLECTION_SCHEMA_VERSION
        ):
            raise ValueError(
                "relationship product forced collection command schema mismatch"
            )
        self.authorization.validate_policy(self.frozen_policy)
        self.authorization.validate_decision_id(self.forecast.decision_id)
        if self.frozen_policy.checkpoint.update_count != 0:
            raise ValueError("forced collection requires a cold theta0 policy")
        if self.frozen_policy.theta0_artifact is None:
            raise ValueError("forced collection requires an explicit theta0 artifact")
        expected_decision = self.frozen_policy.decide(self.forecast)
        if self.forced_exposure.frozen_decision != expected_decision:
            raise ValueError(
                "forced collection exposure differs from cold-policy replay"
            )
        if self.forced_exposure.sequence_index != self.sequence_index:
            raise ValueError("forced collection exposure sequence drifted")
        if self.forced_exposure.forced_action_id != self.forced_action_id:
            raise ValueError(
                "forced collection concrete action differs from scheduled role"
            )
        if self.forced_exposure.theta0_artifact_id != self.theta0_artifact_id:
            raise ValueError("forced collection theta0 lineage mismatch")

    @property
    def command_id(self) -> str:
        return (
            f"{_FORCED_COLLECTION_COMMAND_PREFIX}"
            f"{_canonical_sha256(self._core_payload())}"
        )

    @property
    def theta0_artifact_id(self) -> str:
        theta0 = self.frozen_policy.theta0_artifact
        if theta0 is None:  # Closed by __post_init__; retained for type narrowing.
            raise RuntimeError("forced collection command lost theta0 lineage")
        return theta0.artifact_id

    @property
    def frozen_decision(self) -> RelationshipActionGateFrozenDecision:
        return self.forced_exposure.frozen_decision

    @property
    def forecast(self) -> PreferenceActionForecast:
        return self.forced_exposure.forecast

    @property
    def forced_action_schedule_artifact_id(self) -> str:
        return self.authorization.forced_action_schedule_artifact_id

    @property
    def sequence_index(self) -> int:
        return self.authorization.sequence_index

    @property
    def forced_action_role(self) -> RelationshipProductForcedActionRole:
        return self.authorization.forced_action_role

    @property
    def gate_selected_action_id(self) -> str:
        return self.frozen_decision.decision.selected_action_id

    @property
    def forced_action_id(self) -> str:
        return _forced_collection_action_id(
            forecast=self.forecast,
            role=self.forced_action_role,
        )

    @property
    def gate_would_noop(self) -> bool:
        return self.frozen_decision.decision.gate_action is RelationshipGateAction.NOOP

    @property
    def forced_override(self) -> bool:
        return self.forced_action_id != self.gate_selected_action_id

    def _core_payload(self) -> dict[str, object]:
        policy = self.frozen_policy
        return {
            "schema_version": self.schema_version,
            "frozen_policy": {
                "policy_id": policy.policy_id,
                "artifact_id": policy.artifact.artifact_id,
                "artifact_version": policy.artifact.artifact_version,
                "checkpoint_content_sha256": policy.checkpoint.content_sha256,
                "checkpoint_update_count": policy.checkpoint.update_count,
                "theta0_artifact_id": self.theta0_artifact_id,
            },
            "forced_exposure": self.forced_exposure.to_payload(),
            "authorization": self.authorization.to_payload(),
            "owner_prestate_sha256": self.owner_prestate_sha256,
        }

    def to_payload(self, *, include_command_id: bool = True) -> dict[str, object]:
        payload = self._core_payload()
        if include_command_id:
            return {"command_id": self.command_id, **payload}
        return payload


@dataclass(frozen=True)
class RelationshipProductExecutorCommand:
    """Exact frozen-policy command whose sole treatment field is disposition."""

    forecast: PreferenceActionForecast
    frozen_policy: RelationshipActionGateFrozenPolicy
    frozen_decision: RelationshipActionGateFrozenDecision
    authorization: RelationshipProductFrozenPulseAuthorization
    owner_prestate_sha256: str
    executor_disposition: RelationshipProductExecutorDisposition
    schema_version: str = RELATIONSHIP_PRODUCT_EXECUTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.forecast, PreferenceActionForecast):
            raise TypeError("forecast must be PreferenceActionForecast")
        if not isinstance(self.frozen_policy, RelationshipActionGateFrozenPolicy):
            raise TypeError("frozen_policy must be RelationshipActionGateFrozenPolicy")
        if not isinstance(
            self.frozen_decision,
            RelationshipActionGateFrozenDecision,
        ):
            raise TypeError(
                "frozen_decision must be RelationshipActionGateFrozenDecision"
            )
        if not isinstance(
            self.authorization,
            RelationshipProductFrozenPulseAuthorization,
        ):
            raise TypeError(
                "authorization must be RelationshipProductFrozenPulseAuthorization"
            )
        _require_sha256(self.owner_prestate_sha256, "owner_prestate_sha256")
        if not isinstance(
            self.executor_disposition,
            RelationshipProductExecutorDisposition,
        ):
            raise TypeError(
                "executor_disposition must be RelationshipProductExecutorDisposition"
            )
        if self.schema_version != RELATIONSHIP_PRODUCT_EXECUTOR_SCHEMA_VERSION:
            raise ValueError("relationship product executor command schema mismatch")
        self.authorization.validate_policy(self.frozen_policy)
        expected = self.frozen_policy.decide(self.forecast)
        if self.frozen_decision != expected:
            raise ValueError(
                "frozen decision differs from exact frozen-policy replay"
            )

    @property
    def command_id(self) -> str:
        return f"{_EXECUTOR_COMMAND_PREFIX}{_canonical_sha256(self._core_payload())}"

    @property
    def theta0_artifact_id(self) -> str:
        theta0 = self.frozen_policy.theta0_artifact
        if theta0 is None:  # Closed by __post_init__; retained for type narrowing.
            raise RuntimeError("executor command lost theta0 artifact lineage")
        return theta0.artifact_id

    @property
    def candidate_action_id(self) -> str:
        return self.frozen_decision.decision.selected_action_id

    @property
    def non_noop_opportunity(self) -> bool:
        return (
            self.frozen_decision.decision.gate_action
            is RelationshipGateAction.STEER
            and self.candidate_action_id
            != RelationshipAction.NEUTRAL_NOOP.value
        )

    def _core_payload(self) -> dict[str, object]:
        policy = self.frozen_policy
        return {
            "schema_version": self.schema_version,
            "forecast": preference_action_forecast_to_payload(self.forecast),
            "forecast_sha256": _canonical_sha256(
                preference_action_forecast_to_payload(self.forecast)
            ),
            "frozen_policy": {
                "policy_id": policy.policy_id,
                "artifact_id": policy.artifact.artifact_id,
                "artifact_version": policy.artifact.artifact_version,
                "checkpoint_content_sha256": policy.checkpoint.content_sha256,
                "checkpoint_update_count": policy.checkpoint.update_count,
                "theta0_artifact_id": self.theta0_artifact_id,
                "transition_batch_id": (
                    policy.transition_batch.batch_id
                    if policy.transition_batch is not None
                    else None
                ),
                "transition_receipt_id": (
                    policy.transition_receipt.receipt_id
                    if policy.transition_receipt is not None
                    else None
                ),
            },
            "frozen_decision": self.frozen_decision.to_payload(),
            "authorization": self.authorization.to_payload(),
            "owner_prestate_sha256": self.owner_prestate_sha256,
            "executor_disposition": self.executor_disposition.value,
        }

    def to_payload(self, *, include_command_id: bool = True) -> dict[str, object]:
        payload = self._core_payload()
        if include_command_id:
            return {"command_id": self.command_id, **payload}
        return payload


@dataclass(frozen=True)
class RelationshipProductTemporalDelivery:
    """Minimal temporal owner projection committed by an executor receipt."""

    slot_name: str
    owner: str
    version: int
    timestamp_ms: int
    active_abstract_action: str
    controller_params_hash: str
    action_family_version: int
    action_advisory_id: str
    action_advisory_status: TemporalActionAdvisoryStatus

    def __post_init__(self) -> None:
        if self.slot_name != "self_temporal" or self.owner != "SelfTemporalModule":
            raise ValueError("executor temporal delivery owner envelope drifted")
        if self.version != 1:
            raise ValueError("executor temporal delivery version must be one")
        if (
            isinstance(self.timestamp_ms, bool)
            or not isinstance(self.timestamp_ms, int)
            or self.timestamp_ms < 0
        ):
            raise ValueError("executor temporal delivery timestamp is invalid")
        for field_name, value in (
            ("active_abstract_action", self.active_abstract_action),
            ("controller_params_hash", self.controller_params_hash),
            ("action_advisory_id", self.action_advisory_id),
        ):
            _require_text(value, field_name)
        if (
            isinstance(self.action_family_version, bool)
            or not isinstance(self.action_family_version, int)
            or self.action_family_version < 0
        ):
            raise ValueError("executor temporal action_family_version is invalid")
        if self.action_advisory_status is not TemporalActionAdvisoryStatus.APPLIED:
            raise ValueError("executor temporal delivery was not applied")

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Snapshot[TemporalAbstractionSnapshot],
        *,
        delivered_advisory: TemporalActionAdvisoryProposal,
    ) -> "RelationshipProductTemporalDelivery":
        if not isinstance(snapshot, Snapshot):
            raise TypeError("temporal_snapshot must be a Snapshot")
        temporal = snapshot.value
        if not isinstance(temporal, TemporalAbstractionSnapshot):
            raise TypeError(
                "temporal_snapshot must publish TemporalAbstractionSnapshot"
            )
        if temporal.action_advisory != delivered_advisory:
            raise ValueError("executor temporal advisory payload drifted")
        if temporal.active_abstract_action != delivered_advisory.action_id:
            raise ValueError("executor temporal active action drifted")
        return cls(
            slot_name=snapshot.slot_name,
            owner=snapshot.owner,
            version=snapshot.version,
            timestamp_ms=snapshot.timestamp_ms,
            active_abstract_action=temporal.active_abstract_action,
            controller_params_hash=temporal.controller_params_hash,
            action_family_version=temporal.action_family_version,
            action_advisory_id=delivered_advisory.advisory_id,
            action_advisory_status=temporal.action_advisory_status,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "slot_name": self.slot_name,
            "owner": self.owner,
            "version": self.version,
            "timestamp_ms": self.timestamp_ms,
            "active_abstract_action": self.active_abstract_action,
            "controller_params_hash": self.controller_params_hash,
            "action_family_version": self.action_family_version,
            "action_advisory_id": self.action_advisory_id,
            "action_advisory_status": self.action_advisory_status.value,
        }


@dataclass(frozen=True)
class RelationshipProductExecutorReceipt:
    """Content-addressed proof of candidate preservation and actual delivery."""

    command: RelationshipProductExecutorCommand
    candidate_advisory: TemporalActionAdvisoryProposal
    delivered_advisory: TemporalActionAdvisoryProposal
    temporal_delivery: RelationshipProductTemporalDelivery
    schema_version: str = RELATIONSHIP_PRODUCT_EXECUTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.command, RelationshipProductExecutorCommand):
            raise TypeError("command must be RelationshipProductExecutorCommand")
        if not isinstance(self.candidate_advisory, TemporalActionAdvisoryProposal):
            raise TypeError(
                "candidate_advisory must be TemporalActionAdvisoryProposal"
            )
        if not isinstance(self.delivered_advisory, TemporalActionAdvisoryProposal):
            raise TypeError(
                "delivered_advisory must be TemporalActionAdvisoryProposal"
            )
        if self.schema_version != RELATIONSHIP_PRODUCT_EXECUTOR_SCHEMA_VERSION:
            raise ValueError("relationship product executor receipt schema mismatch")
        expected_candidate = authorize_relationship_product_pulse_advisory(
            self.command.frozen_decision.decision,
            authorization=self.command.authorization.pulse_authorization,
        )
        if self.candidate_advisory != expected_candidate:
            raise ValueError("executor candidate advisory drifted from gate decision")
        expected_delivered = _delivered_advisory_for_executor_command(
            command=self.command,
            candidate_advisory=expected_candidate,
        )
        if self.delivered_advisory != expected_delivered:
            raise ValueError("executor delivered advisory drifted from disposition")
        if not isinstance(
            self.temporal_delivery,
            RelationshipProductTemporalDelivery,
        ):
            raise TypeError(
                "temporal_delivery must be RelationshipProductTemporalDelivery"
            )
        if (
            self.temporal_delivery.action_advisory_id
            != self.delivered_advisory.advisory_id
            or self.temporal_delivery.active_abstract_action
            != self.delivered_action_id
        ):
            raise ValueError("executor temporal delivery lineage drifted")

    @property
    def receipt_id(self) -> str:
        return f"{_EXECUTOR_RECEIPT_PREFIX}{_canonical_sha256(self._core_payload())}"

    @property
    def executor_apply_bit(self) -> bool:
        return (
            self.command.executor_disposition
            is RelationshipProductExecutorDisposition.APPLY_CANDIDATE
        )

    @property
    def executor_status(self) -> RelationshipProductExecutorStatus:
        if not self.executor_apply_bit:
            return RelationshipProductExecutorStatus.STRICT_NOOP
        if self.command.non_noop_opportunity:
            return RelationshipProductExecutorStatus.APPLIED_CANDIDATE
        return RelationshipProductExecutorStatus.GATE_NOOP

    @property
    def candidate_applied(self) -> bool:
        return self.executor_apply_bit

    @property
    def strict_noop_substituted(self) -> bool:
        return not self.executor_apply_bit

    @property
    def delivered_action_id(self) -> str:
        return self.delivered_advisory.action_id

    @property
    def action_diverged(self) -> bool:
        return self.delivered_action_id != self.command.candidate_action_id

    def _core_payload(self) -> dict[str, object]:
        checkpoint = self.command.frozen_policy.checkpoint
        return {
            "schema_version": self.schema_version,
            "command": self.command.to_payload(),
            "authorization_id": self.command.authorization.authorization_id,
            "frozen_policy_id": self.command.frozen_policy.policy_id,
            "theta0_artifact_id": self.command.theta0_artifact_id,
            "owner_prestate_sha256": self.command.owner_prestate_sha256,
            "checkpoint_content_sha256_before": checkpoint.content_sha256,
            "checkpoint_content_sha256_after": checkpoint.content_sha256,
            "policy_update_count_before": checkpoint.update_count,
            "policy_update_count_after": checkpoint.update_count,
            "evaluation_gate_update_delta": 0,
            "pending_decision_count_before": len(checkpoint.pending_decisions),
            "pending_decision_count_after": len(checkpoint.pending_decisions),
            "forecast_sha256": _canonical_sha256(
                preference_action_forecast_to_payload(self.command.forecast)
            ),
            "frozen_decision": self.command.frozen_decision.to_payload(),
            "gate_selected_action_id": self.command.candidate_action_id,
            "intervention_candidate_action_id": self.command.candidate_action_id,
            "candidate_advisory": _temporal_advisory_to_payload(
                self.candidate_advisory
            ),
            "executor_disposition": self.command.executor_disposition.value,
            "executor_apply_bit": self.executor_apply_bit,
            "executor_status": self.executor_status.value,
            "candidate_non_noop": self.command.non_noop_opportunity,
            "candidate_applied": self.candidate_applied,
            "strict_noop_substituted": self.strict_noop_substituted,
            "delivered_advisory": _temporal_advisory_to_payload(
                self.delivered_advisory
            ),
            "delivered_action_id": self.delivered_action_id,
            "executed_non_noop": (
                self.delivered_action_id
                != RelationshipAction.NEUTRAL_NOOP.value
            ),
            "action_diverged": self.action_diverged,
            "temporal_projection": self.temporal_delivery.to_payload(),
            "evaluator_or_judge_feedback_received": False,
        }

    def to_payload(self, *, include_receipt_id: bool = True) -> dict[str, object]:
        payload = self._core_payload()
        if include_receipt_id:
            return {"receipt_id": self.receipt_id, **payload}
        return payload


@dataclass(frozen=True)
class RelationshipProductForcedCollectionReceipt:
    """Proof that a scheduled collection action was actually delivered."""

    command: RelationshipProductForcedCollectionCommand
    candidate_advisory: TemporalActionAdvisoryProposal
    delivered_advisory: TemporalActionAdvisoryProposal
    temporal_delivery: RelationshipProductTemporalDelivery
    schema_version: str = RELATIONSHIP_PRODUCT_FORCED_COLLECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(
            self.command,
            RelationshipProductForcedCollectionCommand,
        ):
            raise TypeError(
                "command must be RelationshipProductForcedCollectionCommand"
            )
        if not isinstance(self.candidate_advisory, TemporalActionAdvisoryProposal):
            raise TypeError(
                "candidate_advisory must be TemporalActionAdvisoryProposal"
            )
        if not isinstance(self.delivered_advisory, TemporalActionAdvisoryProposal):
            raise TypeError(
                "delivered_advisory must be TemporalActionAdvisoryProposal"
            )
        if not isinstance(
            self.temporal_delivery,
            RelationshipProductTemporalDelivery,
        ):
            raise TypeError(
                "temporal_delivery must be RelationshipProductTemporalDelivery"
            )
        if (
            self.schema_version
            != RELATIONSHIP_PRODUCT_FORCED_COLLECTION_SCHEMA_VERSION
        ):
            raise ValueError(
                "relationship product forced collection receipt schema mismatch"
            )
        expected_candidate = authorize_relationship_product_pulse_advisory(
            self.command.frozen_decision.decision,
            authorization=(
                self.command.authorization.frozen_pulse_authorization
                .pulse_authorization
            ),
        )
        if self.candidate_advisory != expected_candidate:
            raise ValueError(
                "forced collection candidate advisory drifted from gate decision"
            )
        expected_delivered = _delivered_advisory_for_forced_collection_command(
            command=self.command,
            candidate_advisory=expected_candidate,
        )
        if self.delivered_advisory != expected_delivered:
            raise ValueError(
                "forced collection delivered advisory drifted from schedule"
            )
        if (
            self.temporal_delivery.action_advisory_id
            != self.delivered_advisory.advisory_id
            or self.temporal_delivery.active_abstract_action
            != self.delivered_action_id
        ):
            raise ValueError("forced collection temporal delivery lineage drifted")

    @property
    def receipt_id(self) -> str:
        return (
            f"{_FORCED_COLLECTION_RECEIPT_PREFIX}"
            f"{_canonical_sha256(self._core_payload())}"
        )

    @property
    def delivered_action_id(self) -> str:
        return self.delivered_advisory.action_id

    @property
    def gate_would_noop(self) -> bool:
        return self.command.gate_would_noop

    @property
    def forced_override(self) -> bool:
        return self.command.forced_override

    @property
    def checkpoint_content_sha256(self) -> str:
        return self.command.frozen_policy.checkpoint.content_sha256

    @property
    def policy_update_count(self) -> int:
        return self.command.frozen_policy.checkpoint.update_count

    @property
    def pending_decision_count(self) -> int:
        return len(self.command.frozen_policy.checkpoint.pending_decisions)

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "command": self.command.to_payload(),
            "candidate_advisory": _temporal_advisory_to_payload(
                self.candidate_advisory
            ),
            "delivered_advisory": _temporal_advisory_to_payload(
                self.delivered_advisory
            ),
            "temporal_projection": self.temporal_delivery.to_payload(),
        }

    def to_payload(self, *, include_receipt_id: bool = True) -> dict[str, object]:
        payload = self._core_payload()
        if include_receipt_id:
            return {"receipt_id": self.receipt_id, **payload}
        return payload


@dataclass(frozen=True)
class RelationshipProductPreActionRequest:
    """Outcome-free public request for one relationship decision pulse."""

    session_id: str
    forecast_request: PreferenceActionForecastRequest
    outcome_turn_index: int

    def __post_init__(self) -> None:
        _require_text(self.session_id, "session_id")
        if not isinstance(
            self.forecast_request,
            PreferenceActionForecastRequest,
        ):
            raise TypeError(
                "forecast_request must be a PreferenceActionForecastRequest"
            )
        if not self.forecast_request.session_scope:
            raise ValueError("forecast_request.session_scope must be non-empty")
        if self.forecast_request.candidate_action_ids != _ACTION_IDS:
            raise ValueError(
                "forecast_request must expose the canonical relationship "
                "action surface"
            )
        if self.forecast_request.outcome_ids != _OUTCOME_IDS:
            raise ValueError(
                "forecast_request must expose the canonical relationship "
                "outcome surface"
            )
        if (
            isinstance(self.outcome_turn_index, bool)
            or not isinstance(self.outcome_turn_index, int)
            or self.outcome_turn_index
            != self.forecast_request.turn_index + 1
        ):
            raise ValueError(
                "outcome_turn_index must be exactly one turn after preaction"
            )


@dataclass(frozen=True)
class RelationshipProductPreActionSnapshot:
    """Frozen owner/gate/temporal exchange published before any outcome."""

    request: RelationshipProductPreActionRequest
    authorization_id: str
    preference_snapshot: Snapshot[PreferenceAboutOtherSnapshot]
    forecast: PreferenceActionForecast
    gate_decision: RelationshipActionGateDecision
    temporal_snapshot: Snapshot[TemporalAbstractionSnapshot]
    owner_persistence_snapshot: OwnerPersistenceSnapshot
    gate_checkpoint_before: RelationshipActionGateCheckpoint
    gate_checkpoint_after: RelationshipActionGateCheckpoint
    schema_version: str = RELATIONSHIP_PRODUCT_PULSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_PRODUCT_PULSE_SCHEMA_VERSION:
            raise ValueError("relationship product preaction schema mismatch")
        _require_text(self.authorization_id, "authorization_id")
        if not isinstance(
            self.preference_snapshot.value,
            PreferenceAboutOtherSnapshot,
        ):
            raise TypeError(
                "preaction preference snapshot published an unexpected value"
            )
        if self.forecast.decision_id != self.request.forecast_request.decision_id:
            raise ValueError("preaction forecast decision lineage mismatch")
        if self.forecast.session_scope != self.request.forecast_request.session_scope:
            raise ValueError("preaction forecast session lineage mismatch")
        if self.gate_decision.forecast_id != self.forecast.forecast_id:
            raise ValueError("preaction gate forecast lineage mismatch")
        temporal = self.temporal_snapshot.value
        if not isinstance(temporal, TemporalAbstractionSnapshot):
            raise TypeError(
                "temporal_snapshot must publish TemporalAbstractionSnapshot"
            )
        if temporal.action_advisory_status is not TemporalActionAdvisoryStatus.APPLIED:
            raise ValueError("preaction typed action was not applied")
        if temporal.active_abstract_action != self.gate_decision.selected_action_id:
            raise ValueError("preaction temporal action lineage mismatch")
        if self.gate_checkpoint_before.artifact_id != self.gate_decision.artifact_id:
            raise ValueError("preaction gate artifact lineage mismatch")
        if self.gate_checkpoint_after.update_count != self.gate_decision.update_count:
            raise ValueError("preaction gate update-count lineage mismatch")
        matching = tuple(
            item
            for item in self.preference_snapshot.value.action_forecasts
            if item.forecast_id == self.forecast.forecast_id
        )
        if matching != (self.forecast,):
            raise ValueError(
                "preaction owner snapshot must contain exactly the frozen forecast"
            )
        pending = tuple(
            item
            for item in self.gate_checkpoint_after.pending_decisions
            if item.forecast_id == self.forecast.forecast_id
        )
        expected_pending = (
            (self.gate_decision,)
            if self.gate_decision.mode is RelationshipActionGateMode.LEARNED
            else ()
        )
        if pending != expected_pending:
            raise ValueError("preaction pending gate decision lineage mismatch")


@dataclass(frozen=True)
class RelationshipProductFrozenPreActionSnapshot:
    """Frozen-policy preaction whose actual action is owned by executor receipt."""

    request: RelationshipProductPreActionRequest
    preference_snapshot: Snapshot[PreferenceAboutOtherSnapshot]
    forecast: PreferenceActionForecast
    execution_receipt: RelationshipProductExecutorReceipt
    owner_persistence_snapshot: OwnerPersistenceSnapshot
    schema_version: str = RELATIONSHIP_PRODUCT_FROZEN_PULSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_PRODUCT_FROZEN_PULSE_SCHEMA_VERSION:
            raise ValueError("relationship product frozen preaction schema mismatch")
        if not isinstance(
            self.preference_snapshot.value,
            PreferenceAboutOtherSnapshot,
        ):
            raise TypeError(
                "frozen preaction preference snapshot published unexpected value"
            )
        if self.forecast.decision_id != self.request.forecast_request.decision_id:
            raise ValueError("frozen preaction forecast decision lineage mismatch")
        if self.forecast.session_scope != self.request.forecast_request.session_scope:
            raise ValueError("frozen preaction forecast session lineage mismatch")
        if (
            self.forecast.interlocutor_id
            != self.request.forecast_request.interlocutor_id
            or self.forecast.issued_turn
            != self.request.forecast_request.turn_index
        ):
            raise ValueError("frozen preaction forecast request lineage mismatch")
        if tuple(
            candidate.action_id for candidate in self.forecast.candidate_predictions
        ) != self.request.forecast_request.candidate_action_ids:
            raise ValueError("frozen preaction forecast action surface drifted")
        if any(
            tuple(outcome.outcome_id for outcome in candidate.outcomes)
            != self.request.forecast_request.outcome_ids
            for candidate in self.forecast.candidate_predictions
        ):
            raise ValueError("frozen preaction forecast outcome surface drifted")
        if self.execution_receipt.command.forecast != self.forecast:
            raise ValueError("frozen preaction executor forecast lineage mismatch")
        matching = tuple(
            item
            for item in self.preference_snapshot.value.action_forecasts
            if item.forecast_id == self.forecast.forecast_id
        )
        if matching != (self.forecast,):
            raise ValueError(
                "frozen preaction owner snapshot must contain the frozen forecast"
            )
        _hydrate_validated_frozen_preaction_owner(self)

    @property
    def frozen_policy(self) -> RelationshipActionGateFrozenPolicy:
        return self.execution_receipt.command.frozen_policy

    @property
    def frozen_decision(self) -> RelationshipActionGateFrozenDecision:
        return self.execution_receipt.command.frozen_decision

    @property
    def delivered_action_id(self) -> str:
        """The only action identity a new environment consumer may settle."""

        return self.execution_receipt.delivered_action_id


@dataclass(frozen=True)
class RelationshipProductForcedCollectionPreActionSnapshot:
    """Cold-theta0 collection pulse with one scheduled delivered action."""

    request: RelationshipProductPreActionRequest
    preference_snapshot: Snapshot[PreferenceAboutOtherSnapshot]
    forecast: PreferenceActionForecast
    execution_receipt: RelationshipProductForcedCollectionReceipt
    owner_persistence_snapshot: OwnerPersistenceSnapshot
    schema_version: str = RELATIONSHIP_PRODUCT_FORCED_COLLECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != RELATIONSHIP_PRODUCT_FORCED_COLLECTION_SCHEMA_VERSION
        ):
            raise ValueError(
                "relationship product forced collection preaction schema mismatch"
            )
        if not isinstance(
            self.preference_snapshot.value,
            PreferenceAboutOtherSnapshot,
        ):
            raise TypeError(
                "forced collection preference snapshot published unexpected value"
            )
        if not isinstance(
            self.execution_receipt,
            RelationshipProductForcedCollectionReceipt,
        ):
            raise TypeError(
                "execution_receipt must be RelationshipProductForcedCollectionReceipt"
            )
        if self.forecast.decision_id != self.request.forecast_request.decision_id:
            raise ValueError("forced collection forecast decision lineage mismatch")
        if self.forecast.session_scope != self.request.forecast_request.session_scope:
            raise ValueError("forced collection forecast session lineage mismatch")
        if (
            self.forecast.interlocutor_id
            != self.request.forecast_request.interlocutor_id
            or self.forecast.issued_turn
            != self.request.forecast_request.turn_index
        ):
            raise ValueError("forced collection forecast request lineage mismatch")
        if tuple(
            candidate.action_id for candidate in self.forecast.candidate_predictions
        ) != self.request.forecast_request.candidate_action_ids:
            raise ValueError("forced collection forecast action surface drifted")
        if any(
            tuple(outcome.outcome_id for outcome in candidate.outcomes)
            != self.request.forecast_request.outcome_ids
            for candidate in self.forecast.candidate_predictions
        ):
            raise ValueError("forced collection forecast outcome surface drifted")
        if self.execution_receipt.command.forecast != self.forecast:
            raise ValueError("forced collection executor forecast lineage mismatch")
        matching = tuple(
            item
            for item in self.preference_snapshot.value.action_forecasts
            if item.forecast_id == self.forecast.forecast_id
        )
        if matching != (self.forecast,):
            raise ValueError(
                "forced collection owner snapshot must contain the frozen forecast"
            )
        _hydrate_validated_collection_preaction_owner(self)

    @property
    def frozen_policy(self) -> RelationshipActionGateFrozenPolicy:
        return self.execution_receipt.command.frozen_policy

    @property
    def frozen_decision(self) -> RelationshipActionGateFrozenDecision:
        return self.execution_receipt.command.frozen_decision

    @property
    def forced_exposure(self) -> RelationshipActionGateForcedExposure:
        return self.execution_receipt.command.forced_exposure

    @property
    def delivered_action_id(self) -> str:
        """The only action identity the collection environment may settle."""

        return self.execution_receipt.delivered_action_id


@dataclass(frozen=True)
class RelationshipProductSettlementInput:
    """Post-action typed evidence; contains no evaluator or judge fields."""

    external_outcome: DialogueExternalOutcomeEvidence
    owner_outcome_evidence: PreferenceActionOutcomeEvidence
    credit_timestamp_ms: int
    apply_credit_to_gate: bool

    def __post_init__(self) -> None:
        if not isinstance(
            self.external_outcome,
            DialogueExternalOutcomeEvidence,
        ):
            raise TypeError(
                "external_outcome must be DialogueExternalOutcomeEvidence"
            )
        if not isinstance(
            self.owner_outcome_evidence,
            PreferenceActionOutcomeEvidence,
        ):
            raise TypeError(
                "owner_outcome_evidence must be "
                "PreferenceActionOutcomeEvidence"
            )
        if (
            isinstance(self.credit_timestamp_ms, bool)
            or not isinstance(self.credit_timestamp_ms, int)
            or self.credit_timestamp_ms < 0
        ):
            raise ValueError("credit_timestamp_ms must be a non-negative integer")
        if not isinstance(self.apply_credit_to_gate, bool):
            raise TypeError("apply_credit_to_gate must be a boolean")


@dataclass(frozen=True)
class RelationshipProductSettlementSnapshot:
    """Frozen owner settlement, PE, credit, and next resumable state."""

    preaction: RelationshipProductPreActionSnapshot
    external_outcome_snapshot: Snapshot[DialogueExternalOutcomeSnapshot]
    preference_snapshot: Snapshot[PreferenceAboutOtherSnapshot]
    social_prediction_error_snapshot: Snapshot[SocialPredictionErrorSnapshot]
    settlement: PreferenceActionForecastSettlement
    credit: CreditRecord
    gate_update: RelationshipActionGateUpdate | None
    owner_persistence_snapshot: OwnerPersistenceSnapshot
    gate_checkpoint: RelationshipActionGateCheckpoint
    credit_applied_to_gate: bool
    schema_version: str = RELATIONSHIP_PRODUCT_PULSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_PRODUCT_PULSE_SCHEMA_VERSION:
            raise ValueError("relationship product settlement schema mismatch")
        if not isinstance(
            self.external_outcome_snapshot.value,
            DialogueExternalOutcomeSnapshot,
        ):
            raise TypeError(
                "external outcome snapshot published an unexpected value"
            )
        if self.settlement.forecast_id != self.preaction.forecast.forecast_id:
            raise ValueError("settlement forecast lineage mismatch")
        if self.credit.prediction_id != self.preaction.forecast.forecast_id:
            raise ValueError("settlement credit forecast lineage mismatch")
        if self.credit_applied_to_gate != (self.gate_update is not None):
            raise ValueError("settlement gate-update audit is inconsistent")
        if (
            self.gate_checkpoint.update_count
            != self.preaction.gate_checkpoint_after.update_count
            + int(self.credit_applied_to_gate)
        ):
            raise ValueError("settlement gate checkpoint count drifted")


@dataclass(frozen=True)
class RelationshipProductFrozenSettlementSnapshot:
    """Actual-action PE settlement under an immutable evaluation policy."""

    preaction: RelationshipProductFrozenPreActionSnapshot
    settlement_input: RelationshipProductSettlementInput
    external_outcome_snapshot: Snapshot[DialogueExternalOutcomeSnapshot]
    preference_snapshot: Snapshot[PreferenceAboutOtherSnapshot]
    social_prediction_error_snapshot: Snapshot[SocialPredictionErrorSnapshot]
    settlement: PreferenceActionForecastSettlement
    credit: CreditRecord
    owner_persistence_snapshot: OwnerPersistenceSnapshot
    schema_version: str = RELATIONSHIP_PRODUCT_FROZEN_PULSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_PRODUCT_FROZEN_PULSE_SCHEMA_VERSION:
            raise ValueError("relationship product frozen settlement schema mismatch")
        if not isinstance(self.settlement_input, RelationshipProductSettlementInput):
            raise TypeError(
                "settlement_input must be RelationshipProductSettlementInput"
            )
        if self.settlement_input.apply_credit_to_gate:
            raise ValueError("frozen settlement input cannot apply credit to gate")
        if not isinstance(
            self.external_outcome_snapshot.value,
            DialogueExternalOutcomeSnapshot,
        ):
            raise TypeError(
                "frozen external outcome snapshot published unexpected value"
            )
        if not isinstance(
            self.preference_snapshot.value,
            PreferenceAboutOtherSnapshot,
        ):
            raise TypeError(
                "frozen preference snapshot published unexpected value"
            )
        if not isinstance(
            self.social_prediction_error_snapshot.value,
            SocialPredictionErrorSnapshot,
        ):
            raise TypeError("frozen social PE snapshot published unexpected value")
        _validate_frozen_settlement_owner_chain(self)
        checkpoint = self.preaction.frozen_policy.checkpoint
        if checkpoint.pending_decisions:
            raise ValueError("frozen settlement policy cannot contain pending decisions")

    @property
    def gate_checkpoint(self) -> RelationshipActionGateCheckpoint:
        return self.preaction.frozen_policy.checkpoint

    @property
    def credit_applied_to_gate(self) -> bool:
        return False

    @property
    def evaluation_gate_update_delta(self) -> int:
        return 0


@dataclass(frozen=True)
class RelationshipProductForcedCollectionSettlementSnapshot:
    """PE-credit settlement for one arm-independent forced exposure."""

    preaction: RelationshipProductForcedCollectionPreActionSnapshot
    settlement_input: RelationshipProductSettlementInput
    external_outcome_snapshot: Snapshot[DialogueExternalOutcomeSnapshot]
    preference_snapshot: Snapshot[PreferenceAboutOtherSnapshot]
    social_prediction_error_snapshot: Snapshot[SocialPredictionErrorSnapshot]
    settlement: PreferenceActionForecastSettlement
    credit: CreditRecord
    owner_persistence_snapshot: OwnerPersistenceSnapshot
    schema_version: str = RELATIONSHIP_PRODUCT_FORCED_COLLECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != RELATIONSHIP_PRODUCT_FORCED_COLLECTION_SCHEMA_VERSION
        ):
            raise ValueError(
                "relationship product forced collection settlement schema mismatch"
            )
        if not isinstance(
            self.preaction,
            RelationshipProductForcedCollectionPreActionSnapshot,
        ):
            raise TypeError(
                "preaction must be RelationshipProductForcedCollectionPreActionSnapshot"
            )
        if not isinstance(self.settlement_input, RelationshipProductSettlementInput):
            raise TypeError(
                "settlement_input must be RelationshipProductSettlementInput"
            )
        if self.settlement_input.apply_credit_to_gate:
            raise ValueError(
                "forced collection settlement cannot apply credit to gate"
            )
        if not isinstance(
            self.external_outcome_snapshot.value,
            DialogueExternalOutcomeSnapshot,
        ):
            raise TypeError(
                "forced collection external outcome published unexpected value"
            )
        if not isinstance(
            self.preference_snapshot.value,
            PreferenceAboutOtherSnapshot,
        ):
            raise TypeError(
                "forced collection preference snapshot published unexpected value"
            )
        if not isinstance(
            self.social_prediction_error_snapshot.value,
            SocialPredictionErrorSnapshot,
        ):
            raise TypeError(
                "forced collection social PE snapshot published unexpected value"
            )
        _validate_immutable_settlement_owner_chain(
            self,
            source="forced collection",
        )
        checkpoint = self.preaction.frozen_policy.checkpoint
        if checkpoint.update_count != 0 or checkpoint.pending_decisions:
            raise ValueError(
                "forced collection settlement requires unchanged cold theta0"
            )

    @property
    def gate_checkpoint(self) -> RelationshipActionGateCheckpoint:
        return self.preaction.frozen_policy.checkpoint

    @property
    def forced_exposure(self) -> RelationshipActionGateForcedExposure:
        return self.preaction.forced_exposure

    @property
    def credit_applied_to_gate(self) -> bool:
        return False

    @property
    def collection_gate_update_delta(self) -> int:
        return 0


@dataclass(frozen=True)
class RelationshipProductV2CollectedCreditBatch:
    """Pulse-owned complete collection with actual-delivery provenance."""

    settlements: tuple[
        RelationshipProductV2ForcedCollectionSettlementSnapshot,
        ...,
    ]
    schema_version: str = RELATIONSHIP_PRODUCT_V2_COLLECTED_BATCH_SCHEMA_VERSION
    _gate_batch: RelationshipActionGateV2CreditBatch = field(
        init=False,
        repr=False,
    )
    _integrity_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        gate_batch = self._validate_components()
        object.__setattr__(self, "_gate_batch", gate_batch)
        object.__setattr__(
            self,
            "_integrity_sha256",
            _canonical_sha256(self._core_payload()),
        )

    def _validate_components(self) -> RelationshipActionGateV2CreditBatch:
        if type(self.settlements) is not tuple or not self.settlements:
            raise ValueError("v2 collected settlements must be a non-empty exact tuple")
        if any(
            type(item) is not RelationshipProductV2ForcedCollectionSettlementSnapshot
            for item in self.settlements
        ):
            raise TypeError("v2 collected settlements contain an invalid item type")
        if self.schema_version != RELATIONSHIP_PRODUCT_V2_COLLECTED_BATCH_SCHEMA_VERSION:
            raise ValueError("relationship product v2 collected batch schema mismatch")
        for index, settlement in enumerate(self.settlements):
            _validate_relationship_product_v2_preaction(
                settlement.preaction,
                source=f"v2 collected preaction {index}",
            )
            _validate_relationship_product_v2_settlement(
                settlement,
                source=f"v2 collected settlement {index}",
            )
        gate_batch = RelationshipActionGateV2CreditBatch(
            exposures=tuple(item.forced_exposure for item in self.settlements),
            credits=tuple(item.common_baseline_credit for item in self.settlements),
        )
        for previous, current in zip(
            self.settlements,
            self.settlements[1:],
            strict=False,
        ):
            if (
                previous.owner_persistence_snapshot
                != current.preaction.owner_input_persistence_snapshot
            ):
                raise ValueError(
                    "v2 collected settlements broke owner persistence handoff"
                )
        return gate_batch

    def _assert_integrity(self) -> None:
        if self._validate_components() != self._gate_batch:
            raise ValueError("v2 collected gate batch mutated after construction")
        if _canonical_sha256(self._core_payload()) != self._integrity_sha256:
            raise ValueError("v2 collected provenance mutated after construction")

    @property
    def gate_batch(self) -> RelationshipActionGateV2CreditBatch:
        return self._gate_batch

    @property
    def collection_id(self) -> str:
        self._assert_integrity()
        return f"{_V2_COLLECTED_BATCH_PREFIX}{self._integrity_sha256}"

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "gate_batch": self._gate_batch.to_payload(),
            "settlements": [
                _relationship_product_v2_forced_settlement_provenance_payload(
                    item
                )
                for item in self.settlements
            ],
        }

    def to_payload(self) -> dict[str, object]:
        self._assert_integrity()
        return {
            "collection_id": f"{_V2_COLLECTED_BATCH_PREFIX}{self._integrity_sha256}",
            **self._core_payload(),
        }


@dataclass(frozen=True)
class RelationshipProductV2CollectionSegment:
    """One collection segment with an explicit owner-continuous start."""

    segment_scope_id: str
    segment_start_owner_persistence_snapshot: OwnerPersistenceSnapshot
    settlements: tuple[
        RelationshipProductV2ForcedCollectionSettlementSnapshot,
        ...,
    ]
    schema_version: str = RELATIONSHIP_PRODUCT_V2_COLLECTION_SEGMENT_SCHEMA_VERSION
    _integrity_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self._validate_components()
        object.__setattr__(
            self,
            "_integrity_sha256",
            _canonical_sha256(self._core_payload()),
        )

    def _validate_components(self) -> None:
        _require_text(self.segment_scope_id, "segment_scope_id")
        if (
            type(self.segment_start_owner_persistence_snapshot)
            is not OwnerPersistenceSnapshot
        ):
            raise TypeError(
                "segment_start_owner_persistence_snapshot must be OwnerPersistenceSnapshot"
            )
        if type(self.settlements) is not tuple or not self.settlements:
            raise ValueError("v2 collection segment settlements must be non-empty")
        if any(
            type(item)
            is not RelationshipProductV2ForcedCollectionSettlementSnapshot
            for item in self.settlements
        ):
            raise TypeError("v2 collection segment contains an invalid settlement")
        if self.schema_version != RELATIONSHIP_PRODUCT_V2_COLLECTION_SEGMENT_SCHEMA_VERSION:
            raise ValueError("relationship product v2 collection segment schema mismatch")
        if (
            self.settlements[0].preaction.owner_input_persistence_snapshot
            != self.segment_start_owner_persistence_snapshot
        ):
            raise ValueError("v2 collection segment lost its explicit owner start")
        if any(
            item.preaction.forecast.session_scope != self.segment_scope_id
            for item in self.settlements
        ):
            raise ValueError(
                "v2 collection segment forecast scope differs from segment scope"
            )
        for index, settlement in enumerate(self.settlements):
            _validate_relationship_product_v2_preaction(
                settlement.preaction,
                source=f"v2 collection segment preaction {index}",
            )
            _validate_relationship_product_v2_settlement(
                settlement,
                source=f"v2 collection segment settlement {index}",
            )
        for previous, current in zip(
            self.settlements,
            self.settlements[1:],
            strict=False,
        ):
            if (
                previous.owner_persistence_snapshot
                != current.preaction.owner_input_persistence_snapshot
            ):
                raise ValueError(
                    "v2 collection segment broke owner persistence handoff"
                )
        sequence_indices = tuple(
            item.forced_exposure.sequence_index for item in self.settlements
        )
        first = sequence_indices[0]
        if sequence_indices != tuple(range(first, first + len(sequence_indices))):
            raise ValueError("v2 collection segment sequence is not contiguous")

    def _assert_integrity(self) -> None:
        self._validate_components()
        if _canonical_sha256(self._core_payload()) != self._integrity_sha256:
            raise ValueError("v2 collection segment mutated after construction")

    @property
    def first_sequence_index(self) -> int:
        return self.settlements[0].forced_exposure.sequence_index

    @property
    def last_sequence_index(self) -> int:
        return self.settlements[-1].forced_exposure.sequence_index

    @property
    def segment_start_owner_persistence_sha256(self) -> str:
        return social_record_store_persistence_sha256(
            self.segment_start_owner_persistence_snapshot
        )

    @property
    def segment_id(self) -> str:
        self._assert_integrity()
        return f"{_V2_COLLECTION_SEGMENT_PREFIX}{self._integrity_sha256}"

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "segment_scope_id": self.segment_scope_id,
            "segment_start_owner_persistence_sha256": (
                self.segment_start_owner_persistence_sha256
            ),
            "first_sequence_index": self.first_sequence_index,
            "last_sequence_index": self.last_sequence_index,
            "settlements": [
                _relationship_product_v2_forced_settlement_provenance_payload(
                    item
                )
                for item in self.settlements
            ],
        }

    def to_payload(self) -> dict[str, object]:
        self._assert_integrity()
        return {
            "segment_id": f"{_V2_COLLECTION_SEGMENT_PREFIX}{self._integrity_sha256}",
            **self._core_payload(),
        }


@dataclass(frozen=True)
class RelationshipProductV2SegmentedCollectedCreditBatch:
    """Complete gate batch with explicit owner-segment boundaries."""

    segments: tuple[RelationshipProductV2CollectionSegment, ...]
    schema_version: str = (
        RELATIONSHIP_PRODUCT_V2_SEGMENTED_COLLECTED_BATCH_SCHEMA_VERSION
    )
    _gate_batch: RelationshipActionGateV2CreditBatch = field(
        init=False,
        repr=False,
    )
    _gate_batch_id: str = field(
        init=False,
        repr=False,
        compare=False,
    )
    _integrity_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        gate_batch = self._validate_components()
        object.__setattr__(self, "_gate_batch", gate_batch)
        object.__setattr__(self, "_gate_batch_id", gate_batch.batch_id)
        object.__setattr__(
            self,
            "_integrity_sha256",
            _canonical_sha256(self._core_payload()),
        )

    def _validate_components(self) -> RelationshipActionGateV2CreditBatch:
        if type(self.segments) is not tuple or not self.segments:
            raise ValueError("v2 segmented collection requires non-empty segments")
        if any(
            type(item) is not RelationshipProductV2CollectionSegment
            for item in self.segments
        ):
            raise TypeError("v2 segmented collection contains an invalid segment")
        if (
            self.schema_version
            != RELATIONSHIP_PRODUCT_V2_SEGMENTED_COLLECTED_BATCH_SCHEMA_VERSION
        ):
            raise ValueError(
                "relationship product v2 segmented collection schema mismatch"
            )
        for segment in self.segments:
            segment._assert_integrity()
        scope_ids = tuple(item.segment_scope_id for item in self.segments)
        if len(set(scope_ids)) != len(scope_ids):
            raise ValueError("v2 segmented collection scope ids must be unique")
        expected_first = 0
        for segment in self.segments:
            if segment.first_sequence_index != expected_first:
                raise ValueError(
                    "v2 segmented collection boundaries are not schedule ordered"
                )
            expected_first = segment.last_sequence_index + 1
        settlements = tuple(
            settlement
            for segment in self.segments
            for settlement in segment.settlements
        )
        return RelationshipActionGateV2CreditBatch(
            exposures=tuple(item.forced_exposure for item in settlements),
            credits=tuple(item.common_baseline_credit for item in settlements),
        )

    def _assert_integrity(self) -> None:
        if self._validate_components() != self._gate_batch:
            raise ValueError("v2 segmented gate batch mutated after construction")
        if _canonical_sha256(self._core_payload()) != self._integrity_sha256:
            raise ValueError("v2 segmented collection mutated after construction")

    @property
    def gate_batch(self) -> RelationshipActionGateV2CreditBatch:
        return self._gate_batch

    @property
    def collection_id(self) -> str:
        self._assert_integrity()
        return f"{_V2_SEGMENTED_COLLECTED_BATCH_PREFIX}{self._integrity_sha256}"

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "gate_batch_id": self._gate_batch_id,
            "segments": [item.to_payload() for item in self.segments],
        }

    def to_payload(self) -> dict[str, object]:
        self._assert_integrity()
        return {
            "collection_id": (
                f"{_V2_SEGMENTED_COLLECTED_BATCH_PREFIX}{self._integrity_sha256}"
            ),
            **self._core_payload(),
        }


@dataclass(frozen=True)
class RelationshipProductV2GateTransition:
    """Exact v2 gate batch transition retained for later arm authorization."""

    collected_batch: RelationshipProductV2CollectedCreditBatch
    gate_receipt: RelationshipActionGateV2BatchReceipt
    frozen_policy: RelationshipActionGateV2FrozenPolicy
    schema_version: str = RELATIONSHIP_PRODUCT_V2_GATE_TRANSITION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.collected_batch) is not RelationshipProductV2CollectedCreditBatch:
            raise TypeError(
                "collected_batch must be RelationshipProductV2CollectedCreditBatch"
            )
        if type(self.gate_receipt) is not RelationshipActionGateV2BatchReceipt:
            raise TypeError("gate_receipt must be RelationshipActionGateV2BatchReceipt")
        if type(self.frozen_policy) is not RelationshipActionGateV2FrozenPolicy:
            raise TypeError("frozen_policy must be RelationshipActionGateV2FrozenPolicy")
        if self.schema_version != RELATIONSHIP_PRODUCT_V2_GATE_TRANSITION_SCHEMA_VERSION:
            raise ValueError("relationship product v2 gate transition schema mismatch")
        self.collected_batch._assert_integrity()
        if (
            self.frozen_policy.transition_batch != self.batch
            or self.frozen_policy.transition_receipt != self.gate_receipt
        ):
            raise ValueError("v2 transition lost its full batch/receipt components")
        replayed = RelationshipActionGateV2.from_credit_batch_transition(
            self.frozen_policy.artifact,
            batch=self.batch,
            receipt=self.gate_receipt,
        ).freeze_for_evaluation()
        if replayed != self.frozen_policy:
            raise ValueError("v2 transition differs from exact gate replay")

    @property
    def batch(self) -> RelationshipActionGateV2CreditBatch:
        return self.collected_batch.gate_batch

    @property
    def disposition(self) -> RelationshipActionGateBatchDisposition:
        return self.gate_receipt.disposition

    @property
    def transition_id(self) -> str:
        return f"{_V2_GATE_TRANSITION_PREFIX}{_canonical_sha256(self._core_payload())}"

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "collected_batch_id": self.collected_batch.collection_id,
            "gate_batch_id": self.batch.batch_id,
            "gate_receipt_id": self.gate_receipt.receipt_id,
            "frozen_policy_id": self.frozen_policy.policy_id,
            "disposition": self.disposition.value,
        }

    def to_payload(self) -> dict[str, object]:
        return {"transition_id": self.transition_id, **self._core_payload()}


@dataclass(frozen=True)
class RelationshipProductV2MatchedGateTransitions:
    """APPLY/WITHHOLD transitions mechanically paired on one collection."""

    applied: RelationshipProductV2GateTransition
    withheld: RelationshipProductV2GateTransition
    schema_version: str = RELATIONSHIP_PRODUCT_V2_MATCHED_TRANSITIONS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.applied) is not RelationshipProductV2GateTransition:
            raise TypeError("applied must be RelationshipProductV2GateTransition")
        if type(self.withheld) is not RelationshipProductV2GateTransition:
            raise TypeError("withheld must be RelationshipProductV2GateTransition")
        if self.schema_version != RELATIONSHIP_PRODUCT_V2_MATCHED_TRANSITIONS_SCHEMA_VERSION:
            raise ValueError("relationship product v2 matched transitions schema mismatch")
        if self.applied.disposition is not RelationshipActionGateBatchDisposition.APPLY:
            raise ValueError("matched v2 applied transition must use APPLY")
        if self.withheld.disposition is not RelationshipActionGateBatchDisposition.WITHHOLD:
            raise ValueError("matched v2 withheld transition must use WITHHOLD")
        if self.applied.collected_batch != self.withheld.collected_batch:
            raise ValueError("matched v2 transitions must share one collected batch")
        applied_receipt = self.applied.gate_receipt
        withheld_receipt = self.withheld.gate_receipt
        matched = (
            ("batch_id", applied_receipt.batch_id, withheld_receipt.batch_id),
            ("plan_id", applied_receipt.plan_id, withheld_receipt.plan_id),
            (
                "pre_checkpoint",
                applied_receipt.pre_checkpoint_content_sha256,
                withheld_receipt.pre_checkpoint_content_sha256,
            ),
            (
                "candidate_checkpoint",
                applied_receipt.candidate_checkpoint_content_sha256,
                withheld_receipt.candidate_checkpoint_content_sha256,
            ),
        )
        for field_name, applied_value, withheld_value in matched:
            if applied_value != withheld_value:
                raise ValueError(
                    f"matched v2 transition {field_name} differs across dispositions"
                )
        if self.applied.frozen_policy.artifact != self.withheld.frozen_policy.artifact:
            raise ValueError("matched v2 transitions must share one gate artifact")

    @property
    def transitions_id(self) -> str:
        return f"{_V2_MATCHED_TRANSITIONS_PREFIX}{_canonical_sha256(self._core_payload())}"

    def transition_for(
        self,
        disposition: RelationshipActionGateBatchDisposition,
    ) -> RelationshipProductV2GateTransition:
        if disposition is RelationshipActionGateBatchDisposition.APPLY:
            return self.applied
        if disposition is RelationshipActionGateBatchDisposition.WITHHOLD:
            return self.withheld
        raise TypeError("disposition must be RelationshipActionGateBatchDisposition")

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "collected_batch_id": self.applied.collected_batch.collection_id,
            "applied_transition_id": self.applied.transition_id,
            "withheld_transition_id": self.withheld.transition_id,
        }

    def to_payload(self) -> dict[str, object]:
        return {"transitions_id": self.transitions_id, **self._core_payload()}


@dataclass(frozen=True)
class RelationshipProductV2SegmentedGateTransition:
    """Exact gate transition from one explicit-start segmented collection."""

    collected_batch: RelationshipProductV2SegmentedCollectedCreditBatch
    gate_receipt: RelationshipActionGateV2BatchReceipt
    frozen_policy: RelationshipActionGateV2FrozenPolicy
    schema_version: str = (
        RELATIONSHIP_PRODUCT_V2_SEGMENTED_GATE_TRANSITION_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if (
            type(self.collected_batch)
            is not RelationshipProductV2SegmentedCollectedCreditBatch
        ):
            raise TypeError(
                "collected_batch must be RelationshipProductV2SegmentedCollectedCreditBatch"
            )
        if type(self.gate_receipt) is not RelationshipActionGateV2BatchReceipt:
            raise TypeError("gate_receipt must be RelationshipActionGateV2BatchReceipt")
        if type(self.frozen_policy) is not RelationshipActionGateV2FrozenPolicy:
            raise TypeError("frozen_policy must be RelationshipActionGateV2FrozenPolicy")
        if (
            self.schema_version
            != RELATIONSHIP_PRODUCT_V2_SEGMENTED_GATE_TRANSITION_SCHEMA_VERSION
        ):
            raise ValueError("relationship product v2 segmented transition schema mismatch")
        self.collected_batch._assert_integrity()
        batch = self.collected_batch.gate_batch
        if (
            self.frozen_policy.transition_batch != batch
            or self.frozen_policy.transition_receipt != self.gate_receipt
        ):
            raise ValueError(
                "v2 segmented transition lost its full batch/receipt components"
            )
        replayed = RelationshipActionGateV2.from_credit_batch_transition(
            self.frozen_policy.artifact,
            batch=batch,
            receipt=self.gate_receipt,
        ).freeze_for_evaluation()
        if replayed != self.frozen_policy:
            raise ValueError("v2 segmented transition differs from exact gate replay")

    @property
    def batch(self) -> RelationshipActionGateV2CreditBatch:
        return self.collected_batch.gate_batch

    @property
    def disposition(self) -> RelationshipActionGateBatchDisposition:
        return self.gate_receipt.disposition

    @property
    def transition_id(self) -> str:
        return (
            f"{_V2_SEGMENTED_GATE_TRANSITION_PREFIX}"
            f"{_canonical_sha256(self._core_payload())}"
        )

    def _core_payload(self) -> dict[str, object]:
        collected_batch_id = self.collected_batch.collection_id
        return {
            "schema_version": self.schema_version,
            "segmented_collected_batch_id": collected_batch_id,
            "gate_batch_id": self.collected_batch._gate_batch_id,
            "gate_receipt_id": self.gate_receipt.receipt_id,
            "frozen_policy_id": self.frozen_policy.policy_id,
            "disposition": self.disposition.value,
        }

    def to_payload(self) -> dict[str, object]:
        return {"transition_id": self.transition_id, **self._core_payload()}


@dataclass(frozen=True)
class RelationshipProductV2SegmentedMatchedGateTransitions:
    """APPLY/WITHHOLD pair from exactly one segmented collection."""

    applied: RelationshipProductV2SegmentedGateTransition
    withheld: RelationshipProductV2SegmentedGateTransition
    schema_version: str = (
        RELATIONSHIP_PRODUCT_V2_SEGMENTED_MATCHED_TRANSITIONS_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if type(self.applied) is not RelationshipProductV2SegmentedGateTransition:
            raise TypeError(
                "applied must be RelationshipProductV2SegmentedGateTransition"
            )
        if type(self.withheld) is not RelationshipProductV2SegmentedGateTransition:
            raise TypeError(
                "withheld must be RelationshipProductV2SegmentedGateTransition"
            )
        if (
            self.schema_version
            != RELATIONSHIP_PRODUCT_V2_SEGMENTED_MATCHED_TRANSITIONS_SCHEMA_VERSION
        ):
            raise ValueError(
                "relationship product v2 segmented matched transitions schema mismatch"
            )
        if self.applied.disposition is not RelationshipActionGateBatchDisposition.APPLY:
            raise ValueError("segmented matched applied transition must use APPLY")
        if (
            self.withheld.disposition
            is not RelationshipActionGateBatchDisposition.WITHHOLD
        ):
            raise ValueError("segmented matched withheld transition must use WITHHOLD")
        if self.applied.collected_batch != self.withheld.collected_batch:
            raise ValueError(
                "segmented matched transitions must share one collected batch"
            )
        applied_receipt = self.applied.gate_receipt
        withheld_receipt = self.withheld.gate_receipt
        for field_name, applied_value, withheld_value in (
            ("batch_id", applied_receipt.batch_id, withheld_receipt.batch_id),
            ("plan_id", applied_receipt.plan_id, withheld_receipt.plan_id),
            (
                "pre_checkpoint",
                applied_receipt.pre_checkpoint_content_sha256,
                withheld_receipt.pre_checkpoint_content_sha256,
            ),
            (
                "candidate_checkpoint",
                applied_receipt.candidate_checkpoint_content_sha256,
                withheld_receipt.candidate_checkpoint_content_sha256,
            ),
        ):
            if applied_value != withheld_value:
                raise ValueError(
                    "segmented matched transition "
                    f"{field_name} differs across dispositions"
                )
        if self.applied.frozen_policy.artifact != self.withheld.frozen_policy.artifact:
            raise ValueError(
                "segmented matched transitions must share one gate artifact"
            )

    @property
    def transitions_id(self) -> str:
        return (
            f"{_V2_SEGMENTED_MATCHED_TRANSITIONS_PREFIX}"
            f"{_canonical_sha256(self._core_payload())}"
        )

    def transition_for(
        self,
        disposition: RelationshipActionGateBatchDisposition,
    ) -> RelationshipProductV2SegmentedGateTransition:
        if disposition is RelationshipActionGateBatchDisposition.APPLY:
            return self.applied
        if disposition is RelationshipActionGateBatchDisposition.WITHHOLD:
            return self.withheld
        raise TypeError("disposition must be RelationshipActionGateBatchDisposition")

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "segmented_collected_batch_id": (
                self.applied.collected_batch.collection_id
            ),
            "applied_transition_id": self.applied.transition_id,
            "withheld_transition_id": self.withheld.transition_id,
        }

    def to_payload(self) -> dict[str, object]:
        return {"transitions_id": self.transitions_id, **self._core_payload()}


@dataclass(frozen=True)
class RelationshipProductV2FederatedCollectedCreditBatch:
    """Ordered pulse provenance for one externally frozen gate federation.

    The caller must provide the complete parent schedule artifact.  This type
    proves exact child membership, order, and globally increasing credit time;
    durable proof that the parent was persisted before collection belongs to
    the external campaign/theta owner rather than this in-memory pulse owner.
    """

    federated_schedule_artifact: (
        RelationshipActionGateV2FederatedAssignmentScheduleArtifact
    )
    child_collected_batches: tuple[
        RelationshipProductV2SegmentedCollectedCreditBatch,
        ...,
    ]
    schema_version: str = (
        RELATIONSHIP_PRODUCT_V2_FEDERATED_COLLECTED_BATCH_SCHEMA_VERSION
    )
    _gate_batch: RelationshipActionGateV2FederatedCreditBatch = field(
        init=False,
        repr=False,
    )
    _gate_batch_id: str = field(
        init=False,
        repr=False,
        compare=False,
    )
    _integrity_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        gate_batch = self._validate_components()
        object.__setattr__(self, "_gate_batch", gate_batch)
        object.__setattr__(self, "_gate_batch_id", gate_batch.batch_id)
        object.__setattr__(
            self,
            "_integrity_sha256",
            _canonical_sha256(self._core_payload()),
        )

    def _validate_components(self) -> RelationshipActionGateV2FederatedCreditBatch:
        if (
            type(self.federated_schedule_artifact)
            is not RelationshipActionGateV2FederatedAssignmentScheduleArtifact
        ):
            raise TypeError(
                "federated_schedule_artifact must be an exact v2 federated schedule"
            )
        if (
            type(self.child_collected_batches) is not tuple
            or len(self.child_collected_batches) < 2
        ):
            raise ValueError(
                "v2 pulse federation requires at least two exact child collections"
            )
        if any(
            type(item)
            is not RelationshipProductV2SegmentedCollectedCreditBatch
            for item in self.child_collected_batches
        ):
            raise TypeError(
                "v2 pulse federation child collections have an invalid type"
            )
        if (
            self.schema_version
            != RELATIONSHIP_PRODUCT_V2_FEDERATED_COLLECTED_BATCH_SCHEMA_VERSION
        ):
            raise ValueError(
                "relationship product v2 federated collection schema mismatch"
            )
        for item in self.child_collected_batches:
            item._assert_integrity()
        collection_ids = tuple(
            item.collection_id for item in self.child_collected_batches
        )
        if len(set(collection_ids)) != len(collection_ids):
            raise ValueError(
                "v2 pulse federation child collection ids must be unique"
            )
        segments = tuple(
            segment
            for collection in self.child_collected_batches
            for segment in collection.segments
        )
        segment_ids = tuple(item.segment_id for item in segments)
        if len(set(segment_ids)) != len(segment_ids):
            raise ValueError("v2 pulse federation segment ids must be globally unique")
        segment_scope_ids = tuple(item.segment_scope_id for item in segments)
        if len(set(segment_scope_ids)) != len(segment_scope_ids):
            raise ValueError(
                "v2 pulse federation segment scope ids must be globally unique"
            )
        return RelationshipActionGateV2FederatedCreditBatch(
            federated_schedule_artifact=self.federated_schedule_artifact,
            child_batches=tuple(
                item.gate_batch for item in self.child_collected_batches
            ),
        )

    def _assert_integrity(self) -> None:
        if self._validate_components() != self._gate_batch:
            raise ValueError("v2 pulse federated gate batch mutated after construction")
        if _canonical_sha256(self._core_payload()) != self._integrity_sha256:
            raise ValueError("v2 pulse federated provenance mutated after construction")

    @property
    def gate_batch(self) -> RelationshipActionGateV2FederatedCreditBatch:
        self._assert_integrity()
        return self._gate_batch

    @property
    def collection_id(self) -> str:
        self._assert_integrity()
        return (
            f"{_V2_FEDERATED_COLLECTED_BATCH_PREFIX}{self._integrity_sha256}"
        )

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "federated_schedule_artifact_id": (
                self.federated_schedule_artifact.artifact_id
            ),
            "gate_batch_id": self._gate_batch_id,
            "child_collection_count": len(self.child_collected_batches),
            "credit_count": len(self._gate_batch.credits),
            "child_collections": [
                item.to_payload() for item in self.child_collected_batches
            ],
        }

    def to_payload(self) -> dict[str, object]:
        self._assert_integrity()
        return {
            "collection_id": (
                f"{_V2_FEDERATED_COLLECTED_BATCH_PREFIX}"
                f"{self._integrity_sha256}"
            ),
            **self._core_payload(),
        }

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        federated_schedule_artifact: (
            RelationshipActionGateV2FederatedAssignmentScheduleArtifact
        ),
        full_child_collected_batches: tuple[
            RelationshipProductV2SegmentedCollectedCreditBatch,
            ...,
        ],
    ) -> "RelationshipProductV2FederatedCollectedCreditBatch":
        if type(payload) is not dict:
            raise TypeError("v2 pulse federated collection payload must be an exact dict")
        collection = cls(
            federated_schedule_artifact=federated_schedule_artifact,
            child_collected_batches=full_child_collected_batches,
        )
        expected = collection.to_payload()
        if expected != payload or _canonical_sha256(expected) != _canonical_sha256(
            payload
        ):
            raise ValueError("v2 pulse federated collection payload mismatch")
        return collection


@dataclass(frozen=True)
class RelationshipProductV2FederatedMatchedGateTransitions:
    """Pulse provenance joined to one gate-owned parent APPLY/WITHHOLD pair."""

    collected_batch: RelationshipProductV2FederatedCollectedCreditBatch
    gate_matched_transitions: RelationshipActionGateV2FederatedMatchedTransitions
    schema_version: str = (
        RELATIONSHIP_PRODUCT_V2_FEDERATED_MATCHED_TRANSITIONS_SCHEMA_VERSION
    )
    _integrity_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self._validate_components()
        object.__setattr__(
            self,
            "_integrity_sha256",
            _canonical_sha256(self._core_payload()),
        )

    def _validate_components(self) -> None:
        if (
            type(self.collected_batch)
            is not RelationshipProductV2FederatedCollectedCreditBatch
        ):
            raise TypeError(
                "collected_batch must be RelationshipProductV2FederatedCollectedCreditBatch"
            )
        if (
            type(self.gate_matched_transitions)
            is not RelationshipActionGateV2FederatedMatchedTransitions
        ):
            raise TypeError(
                "gate_matched_transitions must be exact v2 federated matched transitions"
            )
        if (
            self.schema_version
            != RELATIONSHIP_PRODUCT_V2_FEDERATED_MATCHED_TRANSITIONS_SCHEMA_VERSION
        ):
            raise ValueError(
                "relationship product v2 federated matched transitions schema mismatch"
            )
        self.collected_batch._assert_integrity()
        batch = self.collected_batch.gate_batch
        if self.gate_matched_transitions.applied.batch != batch:
            raise ValueError(
                "v2 pulse federated APPLY transition lost the full collected batch"
            )
        if self.gate_matched_transitions.withheld.batch != batch:
            raise ValueError(
                "v2 pulse federated WITHHOLD transition lost the full collected batch"
            )

    def _assert_integrity(self) -> None:
        self._validate_components()
        if _canonical_sha256(self._core_payload()) != self._integrity_sha256:
            raise ValueError("v2 pulse federated transition provenance mutated")

    @property
    def applied(self) -> RelationshipActionGateV2FederatedTransition:
        self._assert_integrity()
        return self.gate_matched_transitions.applied

    @property
    def withheld(self) -> RelationshipActionGateV2FederatedTransition:
        self._assert_integrity()
        return self.gate_matched_transitions.withheld

    @property
    def transitions_id(self) -> str:
        self._assert_integrity()
        return (
            f"{_V2_FEDERATED_MATCHED_TRANSITIONS_PREFIX}"
            f"{self._integrity_sha256}"
        )

    def _core_payload(self) -> dict[str, object]:
        batch = self.gate_matched_transitions.applied.batch
        return {
            "schema_version": self.schema_version,
            "federated_collected_batch_id": self.collected_batch.collection_id,
            "federated_schedule_artifact_id": (
                batch.federated_schedule_artifact.artifact_id
            ),
            "gate_batch_id": batch.batch_id,
            "gate_matched_transitions_id": (
                self.gate_matched_transitions.transitions_id
            ),
            "child_batch_count": len(batch.child_batches),
            "child_transition_count": 0,
            "credit_count": len(batch.credits),
            "applied_transition_id": self.gate_matched_transitions.applied.transition_id,
            "withheld_transition_id": (
                self.gate_matched_transitions.withheld.transition_id
            ),
        }

    def to_payload(self) -> dict[str, object]:
        self._assert_integrity()
        return {
            "transitions_id": (
                f"{_V2_FEDERATED_MATCHED_TRANSITIONS_PREFIX}"
                f"{self._integrity_sha256}"
            ),
            **self._core_payload(),
        }

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        collected_batch: RelationshipProductV2FederatedCollectedCreditBatch,
        gate_matched_transitions: RelationshipActionGateV2FederatedMatchedTransitions,
    ) -> "RelationshipProductV2FederatedMatchedGateTransitions":
        if type(payload) is not dict:
            raise TypeError(
                "v2 pulse federated matched transitions payload must be an exact dict"
            )
        matched = cls(
            collected_batch=collected_batch,
            gate_matched_transitions=gate_matched_transitions,
        )
        expected = matched.to_payload()
        if expected != payload or _canonical_sha256(expected) != _canonical_sha256(
            payload
        ):
            raise ValueError("v2 pulse federated matched transitions payload mismatch")
        return matched


@dataclass(frozen=True)
class RelationshipProductV2FrozenPulseAuthorization:
    """Authorize one exact replayed APPLY/WITHHOLD v2 evaluation policy."""

    pulse_authorization: RelationshipProductPulseAuthorization
    matched_transitions: RelationshipProductV2MatchedGateTransitions
    gate_disposition: RelationshipActionGateBatchDisposition
    schema_version: str = RELATIONSHIP_PRODUCT_V2_FROZEN_PULSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.pulse_authorization) is not RelationshipProductPulseAuthorization:
            raise TypeError("pulse_authorization must be RelationshipProductPulseAuthorization")
        if type(self.matched_transitions) is not RelationshipProductV2MatchedGateTransitions:
            raise TypeError(
                "matched_transitions must be RelationshipProductV2MatchedGateTransitions"
            )
        if type(self.gate_disposition) is not RelationshipActionGateBatchDisposition:
            raise TypeError(
                "gate_disposition must be RelationshipActionGateBatchDisposition"
            )
        if self.schema_version != RELATIONSHIP_PRODUCT_V2_FROZEN_PULSE_SCHEMA_VERSION:
            raise ValueError("relationship product v2 frozen authorization schema mismatch")
        _validate_relationship_product_v2_policy_authorization(
            self.pulse_authorization,
            self.frozen_policy,
        )
        if self.frozen_policy.artifact.artifact_kind is not RelationshipActionGateV2ArtifactKind.LEARNED_THETA0:
            raise ValueError("v2 evaluation requires an explicit learned theta0 artifact")
        if self.disposition is RelationshipActionGateBatchDisposition.APPLY:
            if self.frozen_policy.checkpoint.update_count < 1:
                raise ValueError("v2 APPLY evaluation policy has no applied update")
        elif self.disposition is RelationshipActionGateBatchDisposition.WITHHOLD:
            if self.frozen_policy.checkpoint.update_count != 0:
                raise ValueError("v2 WITHHOLD evaluation policy changed its checkpoint")
        else:  # pragma: no cover - enum is closed, retained as a loud boundary.
            raise TypeError("v2 evaluation transition disposition is invalid")

    @property
    def frozen_policy(self) -> RelationshipActionGateV2FrozenPolicy:
        return self.transition.frozen_policy

    @property
    def transition(self) -> RelationshipProductV2GateTransition:
        return self.matched_transitions.transition_for(self.gate_disposition)

    @property
    def disposition(self) -> RelationshipActionGateBatchDisposition:
        return self.gate_disposition

    @property
    def authorization_id(self) -> str:
        return f"{_V2_FROZEN_AUTHORIZATION_PREFIX}{_canonical_sha256(self._core_payload())}"

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "pulse_authorization": _pulse_authorization_to_payload(
                self.pulse_authorization
            ),
            "matched_transitions_id": self.matched_transitions.transitions_id,
            "gate_disposition": self.gate_disposition.value,
        }

    def to_payload(self) -> dict[str, object]:
        return {"authorization_id": self.authorization_id, **self._core_payload()}


@dataclass(frozen=True)
class RelationshipProductV2CondensedTheta0FrozenPulseAuthorization:
    """Authorize one cold policy condensed from an exact federated APPLY."""

    pulse_authorization: RelationshipProductPulseAuthorization
    learned_theta0_artifact: RelationshipActionGateV2Artifact
    source_federated_matched_transitions: InitVar[
        RelationshipProductV2FederatedMatchedGateTransitions
    ]
    schema_version: str = (
        RELATIONSHIP_PRODUCT_V2_CONDENSED_THETA0_AUTHORIZATION_SCHEMA_VERSION
    )
    _source_pulse_transitions_id: str = field(init=False, repr=False)
    _source_gate_transitions_id: str = field(init=False, repr=False)
    _source_parent_artifact_id: str = field(init=False, repr=False)
    _source_federated_batch_id: str = field(init=False, repr=False)
    _source_apply_receipt_id: str = field(init=False, repr=False)
    _source_apply_transition_id: str = field(init=False, repr=False)
    _source_checkpoint_content_sha256: str = field(init=False, repr=False)
    _frozen_policy: RelationshipActionGateV2FrozenPolicy = field(
        init=False,
        repr=False,
    )

    def __post_init__(
        self,
        source_federated_matched_transitions: (
            RelationshipProductV2FederatedMatchedGateTransitions
        ),
    ) -> None:
        if type(self.pulse_authorization) is not RelationshipProductPulseAuthorization:
            raise TypeError(
                "pulse_authorization must be RelationshipProductPulseAuthorization"
            )
        if type(self.learned_theta0_artifact) is not RelationshipActionGateV2Artifact:
            raise TypeError(
                "learned_theta0_artifact must be RelationshipActionGateV2Artifact"
            )
        if (
            type(source_federated_matched_transitions)
            is not RelationshipProductV2FederatedMatchedGateTransitions
        ):
            raise TypeError(
                "source_federated_matched_transitions must be exact pulse federated transitions"
            )
        if (
            self.schema_version
            != RELATIONSHIP_PRODUCT_V2_CONDENSED_THETA0_AUTHORIZATION_SCHEMA_VERSION
        ):
            raise ValueError(
                "relationship product v2 condensed theta0 authorization schema mismatch"
            )

        source_federated_matched_transitions._assert_integrity()
        applied = source_federated_matched_transitions.applied
        withheld = source_federated_matched_transitions.withheld
        credit_count = len(applied.batch.credits)
        if (
            applied.disposition is not RelationshipActionGateBatchDisposition.APPLY
            or withheld.disposition
            is not RelationshipActionGateBatchDisposition.WITHHOLD
        ):
            raise ValueError(
                "condensed theta0 source requires one exact federated APPLY/WITHHOLD pair"
            )
        if (
            applied.gate_receipt.atomic_commit_count != 1
            or applied.gate_receipt.update_count_delta != credit_count
            or applied.gate_receipt.child_transition_count != 0
            or applied.terminal_checkpoint.update_count != credit_count
            or applied.terminal_checkpoint.informative_update_count < 1
        ):
            raise ValueError(
                "condensed theta0 source APPLY is not one informative atomic parent transition"
            )
        if (
            withheld.gate_receipt.atomic_commit_count != 0
            or withheld.gate_receipt.update_count_delta != 0
            or withheld.gate_receipt.informative_update_count_delta != 0
            or withheld.gate_receipt.child_transition_count != 0
            or withheld.terminal_checkpoint.update_count != 0
            or withheld.terminal_checkpoint.informative_update_count != 0
            or withheld.terminal_checkpoint.processed_credit_ids != ()
        ):
            raise ValueError(
                "condensed theta0 source WITHHOLD changed the parent checkpoint"
            )
        if (
            self.learned_theta0_artifact.artifact_kind
            is not RelationshipActionGateV2ArtifactKind.LEARNED_THETA0
        ):
            raise ValueError(
                "condensed theta0 authorization requires a learned theta0 artifact"
            )
        expected_learned_theta0 = (
            RelationshipActionGateV2Artifact.create_learned_theta0_from_federation(
                parent_artifact=applied.artifact,
                source_batch=applied.batch,
                apply_receipt=applied.gate_receipt,
            )
        )
        if self.learned_theta0_artifact != expected_learned_theta0:
            raise ValueError(
                "learned theta0 differs from the canonical federated condensation"
            )

        frozen_policy = RelationshipActionGateV2(
            artifact=self.learned_theta0_artifact
        ).freeze_for_evaluation()
        if (
            frozen_policy.transition_batch is not None
            or frozen_policy.transition_receipt is not None
            or frozen_policy.checkpoint.update_count != 0
            or frozen_policy.checkpoint.informative_update_count != 0
            or frozen_policy.checkpoint.processed_credit_ids != ()
            or frozen_policy.checkpoint.weights
            != self.learned_theta0_artifact.weights
        ):
            raise ValueError(
                "condensed theta0 evaluation policy is not an exact cold checkpoint"
            )
        _validate_relationship_product_v2_policy_authorization(
            self.pulse_authorization,
            frozen_policy,
        )

        object.__setattr__(
            self,
            "_source_pulse_transitions_id",
            source_federated_matched_transitions.transitions_id,
        )
        object.__setattr__(
            self,
            "_source_gate_transitions_id",
            source_federated_matched_transitions.gate_matched_transitions.transitions_id,
        )
        object.__setattr__(
            self,
            "_source_parent_artifact_id",
            applied.artifact.artifact_id,
        )
        object.__setattr__(
            self,
            "_source_federated_batch_id",
            applied.batch.batch_id,
        )
        object.__setattr__(
            self,
            "_source_apply_receipt_id",
            applied.gate_receipt.receipt_id,
        )
        object.__setattr__(
            self,
            "_source_apply_transition_id",
            applied.transition_id,
        )
        object.__setattr__(
            self,
            "_source_checkpoint_content_sha256",
            applied.terminal_checkpoint.content_sha256,
        )
        object.__setattr__(self, "_frozen_policy", frozen_policy)

    @property
    def frozen_policy(self) -> RelationshipActionGateV2FrozenPolicy:
        return self._frozen_policy

    @property
    def source_disposition(self) -> RelationshipActionGateBatchDisposition:
        return RelationshipActionGateBatchDisposition.APPLY

    @property
    def authorization_id(self) -> str:
        return (
            f"{_V2_CONDENSED_THETA0_AUTHORIZATION_PREFIX}"
            f"{_canonical_sha256(self._core_payload())}"
        )

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "pulse_authorization": _pulse_authorization_to_payload(
                self.pulse_authorization
            ),
            "source_pulse_federated_matched_transitions_id": (
                self._source_pulse_transitions_id
            ),
            "source_gate_federated_matched_transitions_id": (
                self._source_gate_transitions_id
            ),
            "source_parent_artifact_id": self._source_parent_artifact_id,
            "source_federated_credit_batch_id": self._source_federated_batch_id,
            "source_apply_receipt_id": self._source_apply_receipt_id,
            "source_apply_transition_id": self._source_apply_transition_id,
            "source_checkpoint_content_sha256": (
                self._source_checkpoint_content_sha256
            ),
            "source_transition_disposition": self.source_disposition.value,
            "learned_theta0_artifact": self.learned_theta0_artifact.to_payload(),
            "frozen_policy": _relationship_product_v2_policy_payload(
                self.frozen_policy
            ),
            "evaluation_transition_disposition": None,
        }

    def to_payload(self) -> dict[str, object]:
        return {"authorization_id": self.authorization_id, **self._core_payload()}

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        pulse_authorization: RelationshipProductPulseAuthorization,
        learned_theta0_artifact: RelationshipActionGateV2Artifact,
        source_federated_matched_transitions: (
            RelationshipProductV2FederatedMatchedGateTransitions
        ),
    ) -> "RelationshipProductV2CondensedTheta0FrozenPulseAuthorization":
        if type(payload) is not dict:
            raise TypeError(
                "v2 condensed theta0 authorization payload must be an exact dict"
            )
        authorization = cls(
            pulse_authorization=pulse_authorization,
            learned_theta0_artifact=learned_theta0_artifact,
            source_federated_matched_transitions=(
                source_federated_matched_transitions
            ),
        )
        expected = authorization.to_payload()
        if expected != payload or _canonical_sha256(expected) != _canonical_sha256(
            payload
        ):
            raise ValueError("v2 condensed theta0 authorization payload mismatch")
        return authorization


@dataclass(frozen=True)
class RelationshipProductV2OnlinePulseAuthorization:
    """Prebind one online APPLY/WITHHOLD arm to an exact cold theta0."""

    theta0_authorization: RelationshipProductV2CondensedTheta0FrozenPulseAuthorization
    gate_disposition: RelationshipActionGateBatchDisposition
    owner_session_scope: str
    schema_version: str = RELATIONSHIP_PRODUCT_V2_ONLINE_PULSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.theta0_authorization)
            is not RelationshipProductV2CondensedTheta0FrozenPulseAuthorization
        ):
            raise TypeError(
                "theta0_authorization must be an exact condensed theta0 authorization"
            )
        if type(self.gate_disposition) is not RelationshipActionGateBatchDisposition:
            raise TypeError(
                "gate_disposition must be RelationshipActionGateBatchDisposition"
            )
        _require_text(self.owner_session_scope, "owner_session_scope")
        if self.schema_version != RELATIONSHIP_PRODUCT_V2_ONLINE_PULSE_SCHEMA_VERSION:
            raise ValueError("relationship product v2 online authorization schema mismatch")
        if (
            self.learned_theta0_artifact.artifact_kind
            is not RelationshipActionGateV2ArtifactKind.LEARNED_THETA0
        ):
            raise ValueError("online pulse requires an exact learned theta0 artifact")
        cold = self.theta0_authorization.frozen_policy.checkpoint
        if (
            cold.update_count != 0
            or cold.informative_update_count != 0
            or cold.processed_credit_ids
            or cold.weights != self.learned_theta0_artifact.weights
        ):
            raise ValueError("online pulse must begin at the exact cold learned theta0")

    @property
    def pulse_authorization(self) -> RelationshipProductPulseAuthorization:
        return self.theta0_authorization.pulse_authorization

    @property
    def learned_theta0_artifact(self) -> RelationshipActionGateV2Artifact:
        return self.theta0_authorization.learned_theta0_artifact

    @property
    def cold_initial_chain_id(self) -> str:
        return RelationshipActionGateV2OnlineSession(
            artifact=self.learned_theta0_artifact,
            disposition=self.gate_disposition,
        ).current_chain_id

    @property
    def authorization_id(self) -> str:
        return (
            f"{_V2_ONLINE_AUTHORIZATION_PREFIX}"
            f"{_canonical_sha256(self._core_payload())}"
        )

    def validate_session(
        self,
        session: RelationshipActionGateV2OnlineSession,
    ) -> None:
        if type(session) is not RelationshipActionGateV2OnlineSession:
            raise TypeError("session must be RelationshipActionGateV2OnlineSession")
        if session.artifact != self.learned_theta0_artifact:
            raise ValueError("online session learned theta0 differs from authorization")
        if session.disposition is not self.gate_disposition:
            raise ValueError("online session disposition differs from authorization")

    def validate_request(self, request: RelationshipProductPreActionRequest) -> None:
        if type(request) is not RelationshipProductPreActionRequest:
            raise TypeError("request must be RelationshipProductPreActionRequest")
        if request.forecast_request.session_scope != self.owner_session_scope:
            raise ValueError("online request owner scope differs from authorization")

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "theta0_authorization": self.theta0_authorization.to_payload(),
            "gate_disposition": self.gate_disposition.value,
            "owner_session_scope": self.owner_session_scope,
            "cold_initial_chain_id": self.cold_initial_chain_id,
            "executor_disposition": (
                RelationshipProductExecutorDisposition.APPLY_CANDIDATE.value
            ),
            "evaluation_or_judge_feedback_received": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {"authorization_id": self.authorization_id, **self._core_payload()}


@dataclass(frozen=True)
class RelationshipProductV2ForcedCollectionAuthorization:
    """Authorize one full v2 assignment receipt under a cold policy."""

    pulse_authorization: RelationshipProductPulseAuthorization
    frozen_policy: RelationshipActionGateV2FrozenPolicy
    assignment: RelationshipActionGateV2AssignmentReceipt
    schema_version: str = RELATIONSHIP_PRODUCT_V2_FORCED_COLLECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.pulse_authorization) is not RelationshipProductPulseAuthorization:
            raise TypeError("pulse_authorization must be RelationshipProductPulseAuthorization")
        if type(self.frozen_policy) is not RelationshipActionGateV2FrozenPolicy:
            raise TypeError("frozen_policy must be RelationshipActionGateV2FrozenPolicy")
        if type(self.assignment) is not RelationshipActionGateV2AssignmentReceipt:
            raise TypeError("assignment must be RelationshipActionGateV2AssignmentReceipt")
        if self.schema_version != RELATIONSHIP_PRODUCT_V2_FORCED_COLLECTION_SCHEMA_VERSION:
            raise ValueError("relationship product v2 forced authorization schema mismatch")
        _validate_relationship_product_v2_policy_authorization(
            self.pulse_authorization,
            self.frozen_policy,
        )
        if (
            self.frozen_policy.transition_batch is not None
            or self.frozen_policy.transition_receipt is not None
            or self.frozen_policy.checkpoint.update_count != 0
        ):
            raise ValueError("v2 forced collection requires a cold no-transition policy")

    @property
    def authorization_id(self) -> str:
        return f"{_V2_FORCED_AUTHORIZATION_PREFIX}{_canonical_sha256(self._core_payload())}"

    @property
    def sequence_index(self) -> int:
        return self.assignment.sequence_index

    @property
    def assignment_role(self) -> RelationshipActionGateV2AssignmentRole:
        return self.assignment.assignment_role

    def validate_decision_id(self, decision_id: str) -> None:
        _require_text(decision_id, "decision_id")
        if decision_id != self.assignment.decision_id:
            raise ValueError("v2 forced decision is outside assignment authorization")

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "pulse_authorization": _pulse_authorization_to_payload(
                self.pulse_authorization
            ),
            "frozen_policy": _relationship_product_v2_policy_payload(
                self.frozen_policy
            ),
            "assignment": self.assignment.to_payload(),
        }

    def to_payload(self) -> dict[str, object]:
        return {"authorization_id": self.authorization_id, **self._core_payload()}


@dataclass(frozen=True)
class RelationshipProductV2ExecutorCommand:
    """Evaluation command with immutable policy authorization and executor bit."""

    forecast: PreferenceActionForecast
    frozen_decision: RelationshipActionGateV2FrozenDecision
    authorization: (
        RelationshipProductV2FrozenPulseAuthorization
        | RelationshipProductV2CondensedTheta0FrozenPulseAuthorization
    )
    owner_prestate_sha256: str
    executor_disposition: RelationshipProductExecutorDisposition
    schema_version: str = RELATIONSHIP_PRODUCT_V2_EXECUTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.forecast) is not PreferenceActionForecast:
            raise TypeError("forecast must be PreferenceActionForecast")
        if type(self.frozen_decision) is not RelationshipActionGateV2FrozenDecision:
            raise TypeError("frozen_decision must be RelationshipActionGateV2FrozenDecision")
        if type(self.authorization) not in (
            RelationshipProductV2FrozenPulseAuthorization,
            RelationshipProductV2CondensedTheta0FrozenPulseAuthorization,
        ):
            raise TypeError("authorization must be an exact v2 evaluation authorization")
        _require_sha256(self.owner_prestate_sha256, "owner_prestate_sha256")
        if type(self.executor_disposition) is not RelationshipProductExecutorDisposition:
            raise TypeError("executor_disposition must be RelationshipProductExecutorDisposition")
        if self.schema_version != RELATIONSHIP_PRODUCT_V2_EXECUTOR_SCHEMA_VERSION:
            raise ValueError("relationship product v2 executor command schema mismatch")
        if self.frozen_policy.decide(self.forecast) != self.frozen_decision:
            raise ValueError("v2 executor decision differs from exact policy replay")

    @property
    def frozen_policy(self) -> RelationshipActionGateV2FrozenPolicy:
        return self.authorization.frozen_policy

    @property
    def candidate_action_id(self) -> str:
        return self.frozen_decision.decision.selected_action_id

    @property
    def non_noop_opportunity(self) -> bool:
        return (
            self.frozen_decision.decision.gate_action is RelationshipGateAction.STEER
            and self.candidate_action_id != RelationshipAction.NEUTRAL_NOOP.value
        )

    @property
    def command_id(self) -> str:
        return f"{_V2_EXECUTOR_COMMAND_PREFIX}{_canonical_sha256(self._core_payload())}"

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "forecast": preference_action_forecast_to_payload(self.forecast),
            "frozen_decision": self.frozen_decision.to_payload(),
            "authorization": self.authorization.to_payload(),
            "owner_prestate_sha256": self.owner_prestate_sha256,
            "executor_disposition": self.executor_disposition.value,
        }

    def to_payload(self, *, include_command_id: bool = True) -> dict[str, object]:
        payload = self._core_payload()
        return {"command_id": self.command_id, **payload} if include_command_id else payload


@dataclass(frozen=True)
class RelationshipProductV2OnlineExecutorCommand:
    """One natural online action under a prebound learning disposition."""

    online_exposure: RelationshipActionGateV2OnlineExposure
    authorization: RelationshipProductV2OnlinePulseAuthorization
    owner_prestate_sha256: str
    schema_version: str = RELATIONSHIP_PRODUCT_V2_ONLINE_EXECUTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.online_exposure) is not RelationshipActionGateV2OnlineExposure:
            raise TypeError("online_exposure must be RelationshipActionGateV2OnlineExposure")
        if type(self.authorization) is not RelationshipProductV2OnlinePulseAuthorization:
            raise TypeError("authorization must be RelationshipProductV2OnlinePulseAuthorization")
        _require_sha256(self.owner_prestate_sha256, "owner_prestate_sha256")
        if self.schema_version != RELATIONSHIP_PRODUCT_V2_ONLINE_EXECUTOR_SCHEMA_VERSION:
            raise ValueError("relationship product v2 online executor schema mismatch")
        decision = self.online_exposure.frozen_decision.decision
        if decision.artifact_id != self.authorization.learned_theta0_artifact.artifact_id:
            raise ValueError("online executor decision artifact differs from authorization")
        if self.forecast.session_scope != self.authorization.owner_session_scope:
            raise ValueError("online executor owner scope differs from authorization")
        if self.online_exposure.delivered_action_id != decision.selected_action_id:
            raise ValueError("online executor must preserve the exact learned gate action")

    @property
    def forecast(self) -> PreferenceActionForecast:
        return self.online_exposure.forecast

    @property
    def delivered_action_id(self) -> str:
        return self.online_exposure.delivered_action_id

    @property
    def command_id(self) -> str:
        return (
            f"{_V2_ONLINE_EXECUTOR_COMMAND_PREFIX}"
            f"{_canonical_sha256(self._core_payload())}"
        )

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "online_exposure": self.online_exposure.to_payload(),
            "authorization": self.authorization.to_payload(),
            "owner_prestate_sha256": self.owner_prestate_sha256,
            "executor_disposition": (
                RelationshipProductExecutorDisposition.APPLY_CANDIDATE.value
            ),
        }

    def to_payload(self, *, include_command_id: bool = True) -> dict[str, object]:
        payload = self._core_payload()
        return {"command_id": self.command_id, **payload} if include_command_id else payload


@dataclass(frozen=True)
class RelationshipProductV2ForcedCollectionCommand:
    """Forced command whose actual action exists only in the gate exposure."""

    forced_exposure: RelationshipActionGateV2ForcedExposure
    authorization: RelationshipProductV2ForcedCollectionAuthorization
    owner_prestate_sha256: str
    schema_version: str = RELATIONSHIP_PRODUCT_V2_FORCED_COLLECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.forced_exposure) is not RelationshipActionGateV2ForcedExposure:
            raise TypeError("forced_exposure must be RelationshipActionGateV2ForcedExposure")
        if type(self.authorization) is not RelationshipProductV2ForcedCollectionAuthorization:
            raise TypeError("authorization must be RelationshipProductV2ForcedCollectionAuthorization")
        _require_sha256(self.owner_prestate_sha256, "owner_prestate_sha256")
        if self.schema_version != RELATIONSHIP_PRODUCT_V2_FORCED_COLLECTION_SCHEMA_VERSION:
            raise ValueError("relationship product v2 forced command schema mismatch")
        if self.forced_exposure.assignment != self.authorization.assignment:
            raise ValueError("v2 forced command assignment receipt drifted")
        self.authorization.validate_decision_id(self.forecast.decision_id)
        if self.frozen_policy.decide(self.forecast) != self.frozen_decision:
            raise ValueError("v2 forced decision differs from exact cold-policy replay")

    @property
    def frozen_policy(self) -> RelationshipActionGateV2FrozenPolicy:
        return self.authorization.frozen_policy

    @property
    def forecast(self) -> PreferenceActionForecast:
        return self.forced_exposure.forecast

    @property
    def frozen_decision(self) -> RelationshipActionGateV2FrozenDecision:
        return self.forced_exposure.frozen_decision

    @property
    def delivered_action_id(self) -> str:
        return self.forced_exposure.delivered_action_id

    @property
    def command_id(self) -> str:
        return f"{_V2_FORCED_COMMAND_PREFIX}{_canonical_sha256(self._core_payload())}"

    @property
    def gate_would_noop(self) -> bool:
        return self.frozen_decision.decision.gate_action is RelationshipGateAction.NOOP

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "forced_exposure": self.forced_exposure.to_payload(),
            "authorization": self.authorization.to_payload(),
            "owner_prestate_sha256": self.owner_prestate_sha256,
        }

    def to_payload(self, *, include_command_id: bool = True) -> dict[str, object]:
        payload = self._core_payload()
        return {"command_id": self.command_id, **payload} if include_command_id else payload


@dataclass(frozen=True)
class RelationshipProductV2ExecutorReceipt:
    """Proof that one v2 evaluation candidate or strict noop was delivered."""

    command: RelationshipProductV2ExecutorCommand
    candidate_advisory: TemporalActionAdvisoryProposal
    delivered_advisory: TemporalActionAdvisoryProposal
    temporal_delivery: RelationshipProductTemporalDelivery
    schema_version: str = RELATIONSHIP_PRODUCT_V2_EXECUTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.command) is not RelationshipProductV2ExecutorCommand:
            raise TypeError("command must be RelationshipProductV2ExecutorCommand")
        _validate_relationship_product_v2_delivery_types(
            self.candidate_advisory,
            self.delivered_advisory,
            self.temporal_delivery,
        )
        if self.schema_version != RELATIONSHIP_PRODUCT_V2_EXECUTOR_SCHEMA_VERSION:
            raise ValueError("relationship product v2 executor receipt schema mismatch")
        expected_candidate = _authorize_relationship_product_v2_pulse_advisory(
            frozen_policy=self.command.frozen_policy,
            frozen_decision=self.command.frozen_decision,
            forecast=self.command.forecast,
            pulse_authorization=self.command.authorization.pulse_authorization,
        )
        if self.candidate_advisory != expected_candidate:
            raise ValueError("v2 executor candidate advisory drifted")
        expected_delivered = _delivered_advisory_for_v2_executor_command(
            command=self.command,
            candidate_advisory=expected_candidate,
        )
        if self.delivered_advisory != expected_delivered:
            raise ValueError("v2 executor delivered advisory drifted")
        _validate_relationship_product_v2_temporal_delivery(
            self.temporal_delivery,
            self.delivered_advisory,
        )

    @property
    def receipt_id(self) -> str:
        return f"{_V2_EXECUTOR_RECEIPT_PREFIX}{_canonical_sha256(self._core_payload())}"

    @property
    def executor_apply_bit(self) -> bool:
        return self.command.executor_disposition is RelationshipProductExecutorDisposition.APPLY_CANDIDATE

    @property
    def executor_status(self) -> RelationshipProductExecutorStatus:
        if not self.executor_apply_bit:
            return RelationshipProductExecutorStatus.STRICT_NOOP
        if self.command.non_noop_opportunity:
            return RelationshipProductExecutorStatus.APPLIED_CANDIDATE
        return RelationshipProductExecutorStatus.GATE_NOOP

    @property
    def delivered_action_id(self) -> str:
        return self.delivered_advisory.action_id

    @property
    def action_diverged(self) -> bool:
        return self.delivered_action_id != self.command.candidate_action_id

    def _core_payload(self) -> dict[str, object]:
        gate_transition_disposition = (
            self.command.authorization.disposition.value
            if type(self.command.authorization)
            is RelationshipProductV2FrozenPulseAuthorization
            else None
        )
        return {
            "schema_version": self.schema_version,
            "command": self.command.to_payload(),
            "gate_transition_disposition": gate_transition_disposition,
            "executor_apply_bit": self.executor_apply_bit,
            "executor_status": self.executor_status.value,
            "candidate_non_noop": self.command.non_noop_opportunity,
            "candidate_advisory": _temporal_advisory_to_payload(self.candidate_advisory),
            "delivered_advisory": _temporal_advisory_to_payload(self.delivered_advisory),
            "delivered_action_id": self.delivered_action_id,
            "executed_non_noop": self.delivered_action_id != RelationshipAction.NEUTRAL_NOOP.value,
            "action_diverged": self.action_diverged,
            "temporal_projection": self.temporal_delivery.to_payload(),
            "evaluation_gate_update_delta": 0,
            "evaluator_or_judge_feedback_received": False,
        }

    def to_payload(self, *, include_receipt_id: bool = True) -> dict[str, object]:
        payload = self._core_payload()
        return {"receipt_id": self.receipt_id, **payload} if include_receipt_id else payload


@dataclass(frozen=True)
class RelationshipProductV2OnlineExecutorReceipt:
    """Logical proof of one exact online action before any source outcome."""

    command: RelationshipProductV2OnlineExecutorCommand
    authorized_advisory: TemporalActionAdvisoryProposal
    temporal_delivery: RelationshipProductTemporalDelivery
    schema_version: str = RELATIONSHIP_PRODUCT_V2_ONLINE_EXECUTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.command) is not RelationshipProductV2OnlineExecutorCommand:
            raise TypeError("command must be RelationshipProductV2OnlineExecutorCommand")
        if type(self.authorized_advisory) is not TemporalActionAdvisoryProposal:
            raise TypeError("authorized_advisory must be TemporalActionAdvisoryProposal")
        if type(self.temporal_delivery) is not RelationshipProductTemporalDelivery:
            raise TypeError("temporal_delivery must be RelationshipProductTemporalDelivery")
        if self.schema_version != RELATIONSHIP_PRODUCT_V2_ONLINE_EXECUTOR_SCHEMA_VERSION:
            raise ValueError("relationship product v2 online executor receipt schema mismatch")
        advisory = self.authorized_advisory
        exposure = self.command.online_exposure
        decision = exposure.frozen_decision.decision
        expected = (
            ("decision", advisory.decision_id, decision.decision_id),
            ("prediction", advisory.prediction_id, decision.forecast_id),
            ("action", advisory.action_id, exposure.delivered_action_id),
            ("artifact", advisory.policy_artifact_id, decision.artifact_id),
            ("artifact version", advisory.policy_artifact_version, 2),
        )
        for field_name, observed, wanted in expected:
            if observed != wanted:
                raise ValueError(f"online executor advisory {field_name} drifted")
        if advisory.active_authorized is not True or advisory.evaluator_only is not False:
            raise ValueError("online executor advisory authorization boundary drifted")
        required_refs = {
            exposure.parent_chain_id,
            exposure.exposure_id,
            f"lab-authorization:{self.command.authorization.pulse_authorization.authorization_id}",
            f"lab-online-authorization:{self.command.authorization.authorization_id}",
        }
        if not required_refs.issubset(advisory.evidence_refs):
            raise ValueError("online executor advisory lineage references are incomplete")
        required_rationale = {
            "scope:offline-reactive-environment-only",
            "expression:forbidden",
            "production:forbidden",
            "executor-disposition:apply-exact-online-candidate",
        }
        if not required_rationale.issubset(advisory.rationale_codes):
            raise ValueError("online executor advisory scope boundary drifted")
        _validate_relationship_product_v2_temporal_delivery(
            self.temporal_delivery,
            advisory,
        )

    @property
    def receipt_id(self) -> str:
        return (
            f"{_V2_ONLINE_EXECUTOR_RECEIPT_PREFIX}"
            f"{_canonical_sha256(self._core_payload())}"
        )

    @property
    def delivered_action_id(self) -> str:
        return self.authorized_advisory.action_id

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "command": self.command.to_payload(),
            "gate_disposition": self.command.authorization.gate_disposition.value,
            "authorized_advisory": _temporal_advisory_to_payload(
                self.authorized_advisory
            ),
            "delivered_action_id": self.delivered_action_id,
            "temporal_projection": self.temporal_delivery.to_payload(),
            "evaluation_gate_update_delta_before_outcome": 0,
            "evaluator_or_judge_feedback_received": False,
        }

    def to_payload(self, *, include_receipt_id: bool = True) -> dict[str, object]:
        payload = self._core_payload()
        return {"receipt_id": self.receipt_id, **payload} if include_receipt_id else payload


@dataclass(frozen=True)
class RelationshipProductV2ForcedCollectionReceipt:
    """Proof that one assignment-derived v2 collection action was delivered."""

    command: RelationshipProductV2ForcedCollectionCommand
    candidate_advisory: TemporalActionAdvisoryProposal
    delivered_advisory: TemporalActionAdvisoryProposal
    temporal_delivery: RelationshipProductTemporalDelivery
    schema_version: str = RELATIONSHIP_PRODUCT_V2_FORCED_COLLECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.command) is not RelationshipProductV2ForcedCollectionCommand:
            raise TypeError("command must be RelationshipProductV2ForcedCollectionCommand")
        _validate_relationship_product_v2_delivery_types(
            self.candidate_advisory,
            self.delivered_advisory,
            self.temporal_delivery,
        )
        if self.schema_version != RELATIONSHIP_PRODUCT_V2_FORCED_COLLECTION_SCHEMA_VERSION:
            raise ValueError("relationship product v2 forced receipt schema mismatch")
        expected_candidate = _authorize_relationship_product_v2_pulse_advisory(
            frozen_policy=self.command.frozen_policy,
            frozen_decision=self.command.frozen_decision,
            forecast=self.command.forecast,
            pulse_authorization=self.command.authorization.pulse_authorization,
        )
        if self.candidate_advisory != expected_candidate:
            raise ValueError("v2 forced candidate advisory drifted")
        expected_delivered = _delivered_advisory_for_v2_forced_command(
            command=self.command,
            candidate_advisory=expected_candidate,
        )
        if self.delivered_advisory != expected_delivered:
            raise ValueError("v2 forced delivered advisory drifted")
        _validate_relationship_product_v2_temporal_delivery(
            self.temporal_delivery,
            self.delivered_advisory,
        )

    @property
    def receipt_id(self) -> str:
        return f"{_V2_FORCED_RECEIPT_PREFIX}{_canonical_sha256(self._core_payload())}"

    @property
    def delivered_action_id(self) -> str:
        return self.delivered_advisory.action_id

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "command": self.command.to_payload(),
            "candidate_advisory": _temporal_advisory_to_payload(self.candidate_advisory),
            "delivered_advisory": _temporal_advisory_to_payload(self.delivered_advisory),
            "delivered_action_id": self.delivered_action_id,
            "executed_non_noop": self.delivered_action_id != RelationshipAction.NEUTRAL_NOOP.value,
            "temporal_projection": self.temporal_delivery.to_payload(),
            "collection_gate_update_delta": 0,
            "evaluator_or_judge_feedback_received": False,
        }

    def to_payload(self, *, include_receipt_id: bool = True) -> dict[str, object]:
        payload = self._core_payload()
        return {"receipt_id": self.receipt_id, **payload} if include_receipt_id else payload


@dataclass(frozen=True)
class RelationshipProductV2FrozenPreActionSnapshot:
    request: RelationshipProductPreActionRequest
    owner_input_persistence_snapshot: OwnerPersistenceSnapshot
    preference_snapshot: Snapshot[PreferenceAboutOtherSnapshot]
    forecast: PreferenceActionForecast
    execution_receipt: RelationshipProductV2ExecutorReceipt
    owner_persistence_snapshot: OwnerPersistenceSnapshot
    schema_version: str = RELATIONSHIP_PRODUCT_V2_FROZEN_PULSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.execution_receipt) is not RelationshipProductV2ExecutorReceipt:
            raise TypeError("execution_receipt must be RelationshipProductV2ExecutorReceipt")
        if self.schema_version != RELATIONSHIP_PRODUCT_V2_FROZEN_PULSE_SCHEMA_VERSION:
            raise ValueError("relationship product v2 frozen preaction schema mismatch")
        _validate_relationship_product_v2_preaction(self, source="v2 frozen preaction")

    @property
    def frozen_policy(self) -> RelationshipActionGateV2FrozenPolicy:
        return self.execution_receipt.command.frozen_policy

    @property
    def frozen_decision(self) -> RelationshipActionGateV2FrozenDecision:
        return self.execution_receipt.command.frozen_decision

    @property
    def delivered_action_id(self) -> str:
        return self.execution_receipt.delivered_action_id


@dataclass(frozen=True)
class RelationshipProductV2OnlinePreActionSnapshot:
    """Logical pre-action barrier retaining one live-session pending exposure."""

    request: RelationshipProductPreActionRequest
    owner_input_persistence_snapshot: OwnerPersistenceSnapshot
    preference_snapshot: Snapshot[PreferenceAboutOtherSnapshot]
    forecast: PreferenceActionForecast
    execution_receipt: RelationshipProductV2OnlineExecutorReceipt
    owner_persistence_snapshot: OwnerPersistenceSnapshot
    schema_version: str = RELATIONSHIP_PRODUCT_V2_ONLINE_PULSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.execution_receipt) is not RelationshipProductV2OnlineExecutorReceipt:
            raise TypeError(
                "execution_receipt must be RelationshipProductV2OnlineExecutorReceipt"
            )
        if self.schema_version != RELATIONSHIP_PRODUCT_V2_ONLINE_PULSE_SCHEMA_VERSION:
            raise ValueError("relationship product v2 online preaction schema mismatch")
        _validate_relationship_product_v2_preaction(
            self,
            source="v2 online preaction",
        )
        self.authorization.validate_request(self.request)
        if self.online_exposure.forecast != self.forecast:
            raise ValueError("v2 online preaction exposure forecast lineage mismatch")

    @property
    def authorization(self) -> RelationshipProductV2OnlinePulseAuthorization:
        return self.execution_receipt.command.authorization

    @property
    def online_exposure(self) -> RelationshipActionGateV2OnlineExposure:
        return self.execution_receipt.command.online_exposure

    @property
    def delivered_action_id(self) -> str:
        return self.execution_receipt.delivered_action_id

    @property
    def parent_chain_id(self) -> str:
        return self.online_exposure.parent_chain_id

    @property
    def gate_transition_count_before(self) -> int:
        return self.online_exposure.sequence_index

    @property
    def gate_checkpoint_content_sha256_before(self) -> str:
        return self.online_exposure.frozen_decision.checkpoint_content_sha256


@dataclass(frozen=True)
class RelationshipProductV2ForcedCollectionPreActionSnapshot:
    request: RelationshipProductPreActionRequest
    owner_input_persistence_snapshot: OwnerPersistenceSnapshot
    preference_snapshot: Snapshot[PreferenceAboutOtherSnapshot]
    forecast: PreferenceActionForecast
    execution_receipt: RelationshipProductV2ForcedCollectionReceipt
    owner_persistence_snapshot: OwnerPersistenceSnapshot
    schema_version: str = RELATIONSHIP_PRODUCT_V2_FORCED_COLLECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.execution_receipt) is not RelationshipProductV2ForcedCollectionReceipt:
            raise TypeError("execution_receipt must be RelationshipProductV2ForcedCollectionReceipt")
        if self.schema_version != RELATIONSHIP_PRODUCT_V2_FORCED_COLLECTION_SCHEMA_VERSION:
            raise ValueError("relationship product v2 forced preaction schema mismatch")
        _validate_relationship_product_v2_preaction(self, source="v2 forced preaction")

    @property
    def frozen_policy(self) -> RelationshipActionGateV2FrozenPolicy:
        return self.execution_receipt.command.frozen_policy

    @property
    def frozen_decision(self) -> RelationshipActionGateV2FrozenDecision:
        return self.execution_receipt.command.frozen_decision

    @property
    def forced_exposure(self) -> RelationshipActionGateV2ForcedExposure:
        return self.execution_receipt.command.forced_exposure

    @property
    def delivered_action_id(self) -> str:
        return self.execution_receipt.delivered_action_id


@dataclass(frozen=True)
class RelationshipProductV2FrozenSettlementSnapshot:
    preaction: RelationshipProductV2FrozenPreActionSnapshot
    settlement_input: RelationshipProductSettlementInput
    external_outcome_snapshot: Snapshot[DialogueExternalOutcomeSnapshot]
    preference_snapshot: Snapshot[PreferenceAboutOtherSnapshot]
    social_prediction_error_snapshot: Snapshot[SocialPredictionErrorSnapshot]
    settlement: PreferenceActionForecastSettlement
    credit: CreditRecord
    common_baseline_credit: RelationshipActionCommonBaselineCredit
    owner_persistence_snapshot: OwnerPersistenceSnapshot
    schema_version: str = RELATIONSHIP_PRODUCT_V2_FROZEN_PULSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.preaction) is not RelationshipProductV2FrozenPreActionSnapshot:
            raise TypeError("preaction must be RelationshipProductV2FrozenPreActionSnapshot")
        if self.schema_version != RELATIONSHIP_PRODUCT_V2_FROZEN_PULSE_SCHEMA_VERSION:
            raise ValueError("relationship product v2 frozen settlement schema mismatch")
        _validate_relationship_product_v2_settlement(self, source="v2 frozen")

    @property
    def credit_applied_to_gate(self) -> bool:
        return False

    @property
    def evaluation_gate_update_delta(self) -> int:
        return 0


@dataclass(frozen=True)
class RelationshipProductV2OnlineSettlementSnapshot:
    """One owner-derived PE credit and its exact online gate transition."""

    preaction: RelationshipProductV2OnlinePreActionSnapshot
    settlement_input: RelationshipProductSettlementInput
    external_outcome_snapshot: Snapshot[DialogueExternalOutcomeSnapshot]
    preference_snapshot: Snapshot[PreferenceAboutOtherSnapshot]
    social_prediction_error_snapshot: Snapshot[SocialPredictionErrorSnapshot]
    settlement: PreferenceActionForecastSettlement
    credit: CreditRecord
    common_baseline_credit: RelationshipActionCommonBaselineCredit
    owner_persistence_snapshot: OwnerPersistenceSnapshot
    gate_transition: RelationshipActionGateV2OnlineTransition
    schema_version: str = RELATIONSHIP_PRODUCT_V2_ONLINE_PULSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.preaction) is not RelationshipProductV2OnlinePreActionSnapshot:
            raise TypeError("preaction must be RelationshipProductV2OnlinePreActionSnapshot")
        if type(self.gate_transition) is not RelationshipActionGateV2OnlineTransition:
            raise TypeError("gate_transition must be RelationshipActionGateV2OnlineTransition")
        if self.schema_version != RELATIONSHIP_PRODUCT_V2_ONLINE_PULSE_SCHEMA_VERSION:
            raise ValueError("relationship product v2 online settlement schema mismatch")
        _validate_relationship_product_v2_online_owner_evidence_projection(
            self.settlement_input
        )
        _validate_relationship_product_v2_settlement(self, source="v2 online")
        transition = self.gate_transition
        if transition.plan.exposure != self.preaction.online_exposure:
            raise ValueError("v2 online transition exposure differs from preaction")
        if transition.plan.credit != self.common_baseline_credit:
            raise ValueError("v2 online transition credit differs from owner common credit")
        if transition.receipt.disposition is not self.preaction.authorization.gate_disposition:
            raise ValueError("v2 online transition disposition differs from authorization")
        if transition.receipt.parent_chain_id != self.preaction.parent_chain_id:
            raise ValueError("v2 online transition parent chain differs from preaction")
        if transition.receipt.sequence_index != self.preaction.gate_transition_count_before:
            raise ValueError("v2 online transition sequence differs from preaction")
        if transition.receipt.pre_checkpoint_content_sha256 != (
            self.preaction.gate_checkpoint_content_sha256_before
        ):
            raise ValueError("v2 online transition pre-checkpoint differs from preaction")
        if transition.receipt.post_checkpoint_content_sha256 != (
            transition.terminal_checkpoint.content_sha256
        ):
            raise ValueError("v2 online transition terminal checkpoint drifted")

    @property
    def credit_applied_to_gate(self) -> bool:
        return self.gate_transition.receipt.credit_applied_to_gate

    @property
    def evaluation_gate_update_delta(self) -> int:
        return self.gate_transition.receipt.update_count_delta

    @property
    def gate_transition_count_after(self) -> int:
        return self.gate_transition_count_before + 1

    @property
    def gate_transition_count_before(self) -> int:
        return self.preaction.gate_transition_count_before

    @property
    def terminal_checkpoint_content_sha256(self) -> str:
        return self.gate_transition.terminal_checkpoint.content_sha256


@dataclass(frozen=True)
class RelationshipProductV2ForcedCollectionSettlementSnapshot:
    preaction: RelationshipProductV2ForcedCollectionPreActionSnapshot
    settlement_input: RelationshipProductSettlementInput
    external_outcome_snapshot: Snapshot[DialogueExternalOutcomeSnapshot]
    preference_snapshot: Snapshot[PreferenceAboutOtherSnapshot]
    social_prediction_error_snapshot: Snapshot[SocialPredictionErrorSnapshot]
    settlement: PreferenceActionForecastSettlement
    credit: CreditRecord
    common_baseline_credit: RelationshipActionCommonBaselineCredit
    owner_persistence_snapshot: OwnerPersistenceSnapshot
    schema_version: str = RELATIONSHIP_PRODUCT_V2_FORCED_COLLECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.preaction) is not RelationshipProductV2ForcedCollectionPreActionSnapshot:
            raise TypeError("preaction must be RelationshipProductV2ForcedCollectionPreActionSnapshot")
        if self.schema_version != RELATIONSHIP_PRODUCT_V2_FORCED_COLLECTION_SCHEMA_VERSION:
            raise ValueError("relationship product v2 forced settlement schema mismatch")
        _validate_relationship_product_v2_settlement(self, source="v2 forced")
        if (
            self.preaction.frozen_policy.transition_batch is not None
            or self.preaction.frozen_policy.checkpoint.update_count != 0
        ):
            raise ValueError("v2 forced settlement requires unchanged cold policy")

    @property
    def forced_exposure(self) -> RelationshipActionGateV2ForcedExposure:
        return self.preaction.forced_exposure

    @property
    def credit_applied_to_gate(self) -> bool:
        return False

    @property
    def collection_gate_update_delta(self) -> int:
        return 0


def _relationship_product_v2_forced_settlement_provenance_payload(
    settlement: RelationshipProductV2ForcedCollectionSettlementSnapshot,
) -> dict[str, object]:
    if type(settlement) is not RelationshipProductV2ForcedCollectionSettlementSnapshot:
        raise TypeError(
            "settlement must be RelationshipProductV2ForcedCollectionSettlementSnapshot"
        )
    preaction = settlement.preaction
    common_credit = settlement.common_baseline_credit
    return {
        "sequence_index": preaction.execution_receipt.command.authorization.sequence_index,
        "assignment_receipt_id": preaction.forced_exposure.assignment_receipt_id,
        "forced_exposure_id": preaction.forced_exposure.exposure_id,
        "forced_receipt_id": preaction.execution_receipt.receipt_id,
        "delivered_action_id": preaction.delivered_action_id,
        "settlement_id": settlement.settlement.settlement_id,
        "external_evidence_sha256": common_credit.external_evidence_sha256,
        "settlement_sha256": common_credit.settlement_sha256,
        "social_prediction_error_sha256": (
            common_credit.social_prediction_error_sha256
        ),
        "parent_action_credit_sha256": common_credit.parent_action_credit_sha256,
        "common_baseline_credit_id": common_credit.record_id,
        "owner_input_persistence_sha256": social_record_store_persistence_sha256(
            preaction.owner_input_persistence_snapshot
        ),
        "owner_preaction_persistence_sha256": (
            social_record_store_persistence_sha256(
                preaction.owner_persistence_snapshot
            )
        ),
        "owner_postsettlement_persistence_sha256": (
            social_record_store_persistence_sha256(
                settlement.owner_persistence_snapshot
            )
        ),
    }


def _relationship_product_v2_policy_payload(
    policy: RelationshipActionGateV2FrozenPolicy,
) -> dict[str, object]:
    if type(policy) is not RelationshipActionGateV2FrozenPolicy:
        raise TypeError("policy must be RelationshipActionGateV2FrozenPolicy")
    return {
        "policy_id": policy.policy_id,
        "artifact": policy.artifact.to_payload(),
        "checkpoint": policy.checkpoint.to_payload(),
        "transition_batch": (
            None if policy.transition_batch is None else policy.transition_batch.to_payload()
        ),
        "transition_receipt": (
            None if policy.transition_receipt is None else policy.transition_receipt.to_payload()
        ),
    }


def _validate_relationship_product_v2_policy_authorization(
    authorization: RelationshipProductPulseAuthorization,
    policy: RelationshipActionGateV2FrozenPolicy,
) -> None:
    if type(authorization) is not RelationshipProductPulseAuthorization:
        raise TypeError("authorization must be RelationshipProductPulseAuthorization")
    if type(policy) is not RelationshipActionGateV2FrozenPolicy:
        raise TypeError("policy must be RelationshipActionGateV2FrozenPolicy")
    if authorization.allowed_policy_artifact_version != 2:
        raise ValueError("relationship product v2 authorization requires artifact version 2")
    if authorization.allowed_policy_artifact_id != policy.artifact.artifact_id:
        raise ValueError("v2 policy artifact is outside pulse authorization")


def _validate_relationship_product_v2_delivery_types(
    candidate_advisory: TemporalActionAdvisoryProposal,
    delivered_advisory: TemporalActionAdvisoryProposal,
    temporal_delivery: RelationshipProductTemporalDelivery,
) -> None:
    if type(candidate_advisory) is not TemporalActionAdvisoryProposal:
        raise TypeError("candidate_advisory must be TemporalActionAdvisoryProposal")
    if type(delivered_advisory) is not TemporalActionAdvisoryProposal:
        raise TypeError("delivered_advisory must be TemporalActionAdvisoryProposal")
    if type(temporal_delivery) is not RelationshipProductTemporalDelivery:
        raise TypeError("temporal_delivery must be RelationshipProductTemporalDelivery")


def _validate_relationship_product_v2_temporal_delivery(
    temporal_delivery: RelationshipProductTemporalDelivery,
    delivered_advisory: TemporalActionAdvisoryProposal,
) -> None:
    if (
        temporal_delivery.action_advisory_id != delivered_advisory.advisory_id
        or temporal_delivery.active_abstract_action != delivered_advisory.action_id
    ):
        raise ValueError("relationship product v2 temporal delivery lineage drifted")


def _validate_relationship_product_v2_preaction(
    preaction: (
        RelationshipProductV2FrozenPreActionSnapshot
        | RelationshipProductV2OnlinePreActionSnapshot
        | RelationshipProductV2ForcedCollectionPreActionSnapshot
    ),
    *,
    source: str,
) -> None:
    if type(preaction.request) is not RelationshipProductPreActionRequest:
        raise TypeError(f"{source} request has an invalid type")
    if type(preaction.forecast) is not PreferenceActionForecast:
        raise TypeError(f"{source} forecast has an invalid type")
    if type(preaction.owner_input_persistence_snapshot) is not OwnerPersistenceSnapshot:
        raise TypeError(f"{source} owner input persistence has an invalid type")
    if not isinstance(preaction.preference_snapshot.value, PreferenceAboutOtherSnapshot):
        raise TypeError(f"{source} preference snapshot published unexpected value")
    request = preaction.request.forecast_request
    forecast = preaction.forecast
    expected = (
        ("decision", forecast.decision_id, request.decision_id),
        ("session", forecast.session_scope, request.session_scope),
        ("interlocutor", forecast.interlocutor_id, request.interlocutor_id),
        ("turn", forecast.issued_turn, request.turn_index),
    )
    for field_name, observed, wanted in expected:
        if observed != wanted:
            raise ValueError(f"{source} forecast {field_name} lineage mismatch")
    if tuple(item.action_id for item in forecast.candidate_predictions) != request.candidate_action_ids:
        raise ValueError(f"{source} forecast action surface drifted")
    if any(
        tuple(outcome.outcome_id for outcome in candidate.outcomes) != request.outcome_ids
        for candidate in forecast.candidate_predictions
    ):
        raise ValueError(f"{source} forecast outcome surface drifted")
    if preaction.execution_receipt.command.forecast != forecast:
        raise ValueError(f"{source} executor forecast lineage mismatch")
    matching = tuple(
        item
        for item in preaction.preference_snapshot.value.action_forecasts
        if item.forecast_id == forecast.forecast_id
    )
    if matching != (forecast,):
        raise ValueError(f"{source} owner snapshot must contain the exact forecast")
    expected_persistence = replay_preference_action_forecast_publication_persistence(
        before=preaction.owner_input_persistence_snapshot,
        forecast=forecast,
    )
    if expected_persistence != preaction.owner_persistence_snapshot:
        raise ValueError(
            f"{source} persistence is not the exact owner forecast publication"
        )
    _hydrate_validated_immutable_preaction_owner(preaction, source=source)


def _validate_relationship_product_v2_settlement(
    snapshot: (
        RelationshipProductV2FrozenSettlementSnapshot
        | RelationshipProductV2OnlineSettlementSnapshot
        | RelationshipProductV2ForcedCollectionSettlementSnapshot
    ),
    *,
    source: str,
) -> None:
    if type(snapshot.settlement_input) is not RelationshipProductSettlementInput:
        raise TypeError("settlement_input must be RelationshipProductSettlementInput")
    if snapshot.settlement_input.apply_credit_to_gate:
        raise ValueError(f"{source} settlement cannot apply credit online")
    if snapshot.settlement_input.external_outcome.confidence != 1.0:
        raise ValueError(f"{source} common-baseline credit requires confidence 1.0")
    if type(snapshot.common_baseline_credit) is not RelationshipActionCommonBaselineCredit:
        raise TypeError("common_baseline_credit has an invalid type")
    _validate_immutable_settlement_owner_chain(snapshot, source=source)
    expected_error_id = f"social-pe:{snapshot.settlement.settlement_id}"
    matching_errors = tuple(
        item
        for item in snapshot.social_prediction_error_snapshot.value.errors
        if item.error_id == expected_error_id
    )
    if len(matching_errors) != 1:
        raise ValueError(f"{source} requires one exact social PE")
    expected = derive_preference_action_common_baseline_credit_records(
        forecasts=(snapshot.preaction.forecast,),
        external_evidence=(snapshot.settlement_input.external_outcome,),
        settlements=(snapshot.settlement,),
        social_errors=matching_errors,
        settled_at_turn=snapshot.settlement.observed_turn,
        timestamp_ms=snapshot.settlement_input.credit_timestamp_ms,
    )
    if expected != (snapshot.common_baseline_credit,):
        raise ValueError(f"{source} common-baseline credit differs from exact owner replay")


@dataclass(frozen=True)
class _RelationshipProductOwnerSettlement:
    external_outcome_snapshot: Snapshot[DialogueExternalOutcomeSnapshot]
    preference_snapshot: Snapshot[PreferenceAboutOtherSnapshot]
    social_prediction_error_snapshot: Snapshot[SocialPredictionErrorSnapshot]
    settlement: PreferenceActionForecastSettlement
    credit: CreditRecord


def _validate_frozen_settlement_owner_chain(
    snapshot: RelationshipProductFrozenSettlementSnapshot,
) -> None:
    _validate_immutable_settlement_owner_chain(snapshot, source="frozen")


def _validate_immutable_settlement_owner_chain(
    snapshot: (
        RelationshipProductFrozenSettlementSnapshot
        | RelationshipProductForcedCollectionSettlementSnapshot
        | RelationshipProductV2FrozenSettlementSnapshot
        | RelationshipProductV2OnlineSettlementSnapshot
        | RelationshipProductV2ForcedCollectionSettlementSnapshot
    ),
    *,
    source: str,
) -> None:
    preaction = snapshot.preaction
    request = preaction.request
    settlement = snapshot.settlement
    settlement_input = snapshot.settlement_input
    _hydrate_validated_immutable_preaction_owner(
        preaction,
        source=f"{source} preaction",
    )
    _validate_immutable_environment_settlement_input(
        settlement_input,
        source=source,
    )
    _validate_settlement_lineage_values(
        request=request,
        forecast=preaction.forecast,
        actual_action_id=preaction.delivered_action_id,
        settlement_input=settlement_input,
    )
    _validate_owner_snapshot_envelope(
        snapshot.external_outcome_snapshot,
        slot_name="dialogue_external_outcome",
        owner="DialogueExternalOutcomeModule",
        source="frozen external outcome",
    )
    external_value = snapshot.external_outcome_snapshot.value
    if external_value.turn_index != request.outcome_turn_index:
        raise ValueError("frozen external outcome snapshot turn drifted")
    if len(external_value.entries) != 1:
        raise ValueError("frozen settlement requires one external outcome entry")
    external = external_value.entries[0]
    external_expected = (
        ("evidence_id", external.evidence_id, settlement.source_evidence_id),
        ("turn_index", external.turn_index, request.outcome_turn_index),
        (
            "action_turn_index",
            external.action_turn_index,
            request.forecast_request.turn_index,
        ),
        (
            "session_scope",
            external.session_scope,
            request.forecast_request.session_scope,
        ),
        ("forecast_id", external.forecast_id, preaction.forecast.forecast_id),
        (
            "decision_id",
            external.decision_id,
            request.forecast_request.decision_id,
        ),
        ("action_id", external.action_id, preaction.delivered_action_id),
        ("outcome_id", external.kind.value, settlement.observed_outcome_id),
        ("confidence", external.confidence, settlement.evidence_confidence),
    )
    for field_name, observed, wanted in external_expected:
        if observed != wanted:
            raise ValueError(f"frozen external outcome {field_name} mismatch")
    if external != settlement_input.external_outcome:
        raise ValueError(
            "frozen external outcome does not match exact settlement input"
        )

    expected_settlement = settle_preference_action_forecast(
        forecast=preaction.forecast,
        evidence=settlement_input.external_outcome,
    )
    if settlement != expected_settlement:
        raise ValueError(
            "frozen settlement differs from preference owner exact derivation"
        )

    settlement_expected = (
        ("forecast_id", settlement.forecast_id, preaction.forecast.forecast_id),
        (
            "decision_id",
            settlement.decision_id,
            request.forecast_request.decision_id,
        ),
        (
            "session_scope",
            settlement.session_scope,
            request.forecast_request.session_scope,
        ),
        (
            "interlocutor_id",
            settlement.interlocutor_id,
            request.forecast_request.interlocutor_id,
        ),
        ("action_id", settlement.action_id, preaction.delivered_action_id),
        ("forecast_issued_turn", settlement.forecast_issued_turn, preaction.forecast.issued_turn),
        ("observed_turn", settlement.observed_turn, request.outcome_turn_index),
    )
    for field_name, observed, wanted in settlement_expected:
        if observed != wanted:
            raise ValueError(f"frozen settlement {field_name} mismatch")

    _validate_owner_snapshot_envelope(
        snapshot.preference_snapshot,
        slot_name=PreferenceAboutOtherModule.slot_name,
        owner=PreferenceAboutOtherModule.owner,
        source="frozen preference",
    )
    current_settlements = tuple(
        item
        for item in snapshot.preference_snapshot.value.forecast_settlements
        if item.forecast_id == settlement.forecast_id
        and item.observed_turn == settlement.observed_turn
    )
    if current_settlements != (settlement,):
        raise ValueError("frozen preference owner settlement payload drifted")
    current_outcomes = tuple(
        item
        for item in snapshot.preference_snapshot.value.action_outcome_evidence
        if item.evidence_id
        == settlement_input.owner_outcome_evidence.evidence_id
    )
    if current_outcomes != (settlement_input.owner_outcome_evidence,):
        raise ValueError("frozen preference owner outcome evidence drifted")

    _validate_owner_snapshot_envelope(
        snapshot.social_prediction_error_snapshot,
        slot_name="social_prediction_error",
        owner="SocialPredictionErrorModule",
        source="frozen social PE",
    )
    expected_error_id = f"social-pe:{settlement.settlement_id}"
    matching_errors = tuple(
        error
        for error in snapshot.social_prediction_error_snapshot.value.errors
        if error.error_id == expected_error_id
    )
    if len(matching_errors) != 1:
        raise ValueError("frozen settlement requires exactly one matching social PE")
    social_error = matching_errors[0]
    expected_social_error = (
        ("prediction_id", social_error.prediction_id, settlement.forecast_id),
        (
            "kind",
            social_error.kind,
            SocialPredictionKind.PREFERENCE_ABOUT_OTHER,
        ),
        ("outcome", social_error.outcome, settlement.outcome),
        ("owner", social_error.owner, PreferenceAboutOtherModule.owner),
        ("scope_kind", social_error.scope_kind, SocialScopeKind.INTERLOCUTOR),
        ("scope_id", social_error.scope_id, settlement.interlocutor_id),
        (
            "evidence",
            social_error.evidence,
            (
                f"forecast_settlement:{settlement.settlement_id}",
                f"external_outcome:{settlement.source_evidence_id}",
                f"action:{settlement.action_id}",
                f"observed_outcome:{settlement.observed_outcome_id}",
                f"predicted_probability={settlement.predicted_probability:.12f}",
                f"negative_log_likelihood={settlement.negative_log_likelihood:.12f}",
                "signed_utility_prediction_error="
                f"{settlement.signed_utility_prediction_error:.12f}",
            ),
        ),
    )
    for field_name, observed, wanted in expected_social_error:
        if observed != wanted:
            raise ValueError(f"frozen settlement social PE {field_name} mismatch")
    if social_error.magnitude != settlement.magnitude:
        raise ValueError("frozen settlement social PE magnitude mismatch")

    if snapshot.credit.timestamp_ms != settlement_input.credit_timestamp_ms:
        raise ValueError("frozen settlement credit timestamp mismatch")
    expected_credits = derive_preference_action_forecast_credit_records(
        settlements=(settlement,),
        social_errors=(social_error,),
        settled_at_turn=settlement.observed_turn,
        timestamp_ms=settlement_input.credit_timestamp_ms,
    )
    if expected_credits != (snapshot.credit,):
        raise ValueError("frozen settlement PE-derived credit payload drifted")

    expected_post_persistence = (
        replay_preference_action_forecast_settlement_persistence(
            before=preaction.owner_persistence_snapshot,
            forecast=preaction.forecast,
            external_evidence=settlement_input.external_outcome,
            owner_outcome_evidence=settlement_input.owner_outcome_evidence,
        )
    )
    if snapshot.owner_persistence_snapshot != expected_post_persistence:
        raise ValueError(
            "frozen settlement persistence is not the exact owner transition"
        )

    persisted = _hydrate_validated_preference_owner_persistence(
        preference_snapshot=snapshot.preference_snapshot,
        owner_persistence_snapshot=snapshot.owner_persistence_snapshot,
        source="frozen settlement",
    )
    persisted_forecasts = tuple(
        item
        for item in persisted.preference_action_forecasts
        if item.forecast_id == preaction.forecast.forecast_id
    )
    if persisted_forecasts:
        raise ValueError(
            "frozen settlement persistence retained the settled forecast"
        )
    persisted_settlements = tuple(
        item
        for item in persisted.preference_forecast_settlements
        if item.forecast_id == preaction.forecast.forecast_id
    )
    if persisted_settlements != (settlement,):
        raise ValueError(
            "frozen settlement persistence lost the exact settlement"
        )
    persisted_outcomes = tuple(
        item
        for item in persisted.preference_action_outcomes
        if item.evidence_id
        == settlement_input.owner_outcome_evidence.evidence_id
    )
    if persisted_outcomes != (settlement_input.owner_outcome_evidence,):
        raise ValueError(
            "frozen settlement persistence lost the exact owner outcome evidence"
        )


def _validate_owner_snapshot_envelope(
    snapshot: Snapshot[object],
    *,
    slot_name: str,
    owner: str,
    source: str,
) -> None:
    if (
        snapshot.slot_name != slot_name
        or snapshot.owner != owner
        or snapshot.version != 1
        or isinstance(snapshot.timestamp_ms, bool)
        or not isinstance(snapshot.timestamp_ms, int)
        or snapshot.timestamp_ms < 0
    ):
        raise ValueError(f"{source} snapshot envelope drifted")


def _validate_frozen_environment_settlement_input(
    settlement_input: RelationshipProductSettlementInput,
) -> None:
    _validate_immutable_environment_settlement_input(
        settlement_input,
        source="frozen",
    )


def _validate_immutable_environment_settlement_input(
    settlement_input: RelationshipProductSettlementInput,
    *,
    source: str,
) -> None:
    if not isinstance(settlement_input, RelationshipProductSettlementInput):
        raise TypeError(
            "settlement_input must be RelationshipProductSettlementInput"
        )
    if (
        settlement_input.external_outcome.source
        is not DialogueExternalOutcomeEvidenceSource.ENVIRONMENT
    ):
        raise ValueError(
            f"{source} product settlement requires ENVIRONMENT external evidence"
        )


def _validate_relationship_product_v2_online_owner_evidence_projection(
    settlement_input: RelationshipProductSettlementInput,
) -> None:
    if type(settlement_input) is not RelationshipProductSettlementInput:
        raise TypeError("settlement_input must be RelationshipProductSettlementInput")
    external = settlement_input.external_outcome
    owner_evidence = settlement_input.owner_outcome_evidence
    if (
        owner_evidence.evidence_id != external.evidence_id
        or owner_evidence.reaction_summary != external.description
        or owner_evidence.evidence_refs != (external.evidence_ref,)
    ):
        raise ValueError(
            "v2 online owner evidence must be the exact environment reaction projection"
        )


def _hydrate_validated_preference_owner_persistence(
    *,
    preference_snapshot: Snapshot[PreferenceAboutOtherSnapshot],
    owner_persistence_snapshot: OwnerPersistenceSnapshot,
    source: str,
) -> SocialRecordStore:
    """Bind durable public owner fields without reading persistence payload."""

    if not isinstance(preference_snapshot.value, PreferenceAboutOtherSnapshot):
        raise TypeError(f"{source} preference snapshot value is invalid")
    if not isinstance(owner_persistence_snapshot, OwnerPersistenceSnapshot):
        raise TypeError(f"{source} owner persistence snapshot is invalid")
    store = SocialRecordStore()
    store.hydrate_from_persistence(owner_persistence_snapshot)
    if store.export_persistence_snapshot() != owner_persistence_snapshot:
        raise ValueError(f"{source} owner persistence is not canonical")
    preference = preference_snapshot.value
    stable_fields = (
        (
            "records",
            store.tom_records(_PREFERENCE_SLOT),
            preference.records,
        ),
        (
            "action_outcome_evidence",
            store.preference_action_outcomes,
            preference.action_outcome_evidence,
        ),
        (
            "action_forecasts",
            store.preference_action_forecasts,
            preference.action_forecasts,
        ),
        (
            "forecast_settlements",
            store.preference_forecast_settlements,
            preference.forecast_settlements,
        ),
        (
            "action_outcome_mutation_receipts",
            store.preference_action_outcome_mutation_receipts,
            preference.action_outcome_mutation_receipts,
        ),
    )
    for field_name, persisted, published in stable_fields:
        if persisted != published:
            raise ValueError(
                f"{source} owner persistence {field_name} drifted from snapshot"
            )
    return store


def _hydrate_validated_frozen_preaction_owner(
    preaction: RelationshipProductFrozenPreActionSnapshot,
) -> SocialRecordStore:
    return _hydrate_validated_immutable_preaction_owner(
        preaction,
        source="frozen preaction",
    )


def _hydrate_validated_collection_preaction_owner(
    preaction: RelationshipProductForcedCollectionPreActionSnapshot,
) -> SocialRecordStore:
    return _hydrate_validated_immutable_preaction_owner(
        preaction,
        source="forced collection preaction",
    )


def _hydrate_validated_immutable_preaction_owner(
    preaction: (
        RelationshipProductFrozenPreActionSnapshot
        | RelationshipProductForcedCollectionPreActionSnapshot
        | RelationshipProductV2FrozenPreActionSnapshot
        | RelationshipProductV2OnlinePreActionSnapshot
        | RelationshipProductV2ForcedCollectionPreActionSnapshot
    ),
    *,
    source: str,
) -> SocialRecordStore:
    _validate_owner_snapshot_envelope(
        preaction.preference_snapshot,
        slot_name=PreferenceAboutOtherModule.slot_name,
        owner=PreferenceAboutOtherModule.owner,
        source=f"{source} preference",
    )
    persisted = _hydrate_validated_preference_owner_persistence(
        preference_snapshot=preaction.preference_snapshot,
        owner_persistence_snapshot=preaction.owner_persistence_snapshot,
        source=source,
    )
    persistence_sha256 = social_record_store_persistence_sha256(
        preaction.owner_persistence_snapshot
    )
    if persistence_sha256 != preaction.execution_receipt.command.owner_prestate_sha256:
        raise ValueError(
            f"{source} owner state hash drifted from executor command"
        )
    persisted_current = tuple(
        item
        for item in persisted.preference_action_forecasts
        if item.forecast_id == preaction.forecast.forecast_id
    )
    if persisted_current != (preaction.forecast,):
        raise ValueError(
            f"{source} persistence must contain the exact forecast"
        )
    return persisted


async def append_relationship_product_onboarding(
    *,
    owner_persistence_snapshot: OwnerPersistenceSnapshot | None,
    onboarding: RelationshipProductOnboardingInput,
) -> RelationshipProductOnboardingSnapshot:
    """Append one public onboarding experience through its unique owner."""

    store = SocialRecordStore()
    if owner_persistence_snapshot is not None:
        store.hydrate_from_persistence(owner_persistence_snapshot)
    evidence = PreferenceActionOutcomeEvidence(
        evidence_id=onboarding.evidence_id,
        interlocutor_id="primary",
        observation_summary=onboarding.public_observation,
        action_id=onboarding.action_id,
        observed_outcome_id=onboarding.observed_outcome_id,
        reaction_summary=onboarding.reaction_summary,
        source_turn=onboarding.turn_index,
        evidence_refs=(onboarding.evidence_ref,),
    )
    owner_snapshot = await PreferenceAboutOtherModule(
        proposal_runtime=_OutcomeEvidenceProposalRuntime(
            evidence=evidence,
            semantic_evidence_ref=onboarding.evidence_ref,
        ),
        user_input=onboarding.public_observation,
        turn_index=onboarding.turn_index,
        wiring_level=WiringLevel.SHADOW,
        record_store=store,
        action_outcome_evidence=evidence,
    ).process({})
    return RelationshipProductOnboardingSnapshot(
        onboarding=onboarding,
        preference_snapshot=owner_snapshot,
        owner_persistence_snapshot=store.export_persistence_snapshot(),
    )


def authorize_relationship_product_pulse_advisory(
    decision: RelationshipActionGateDecision,
    *,
    authorization: RelationshipProductPulseAuthorization,
) -> TemporalActionAdvisoryProposal:
    """Authorize a non-oracle typed decision for an offline environment."""

    if decision.evaluator_only or decision.mode is RelationshipActionGateMode.ORACLE:
        raise ValueError(
            "relationship product pulse cannot authorize evaluator/oracle decisions"
        )
    if decision.artifact_id != authorization.allowed_policy_artifact_id:
        raise ValueError("gate artifact id is outside pulse authorization")
    if (
        decision.artifact_version
        != authorization.allowed_policy_artifact_version
    ):
        raise ValueError("gate artifact version is outside pulse authorization")
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


async def prepare_relationship_product_preaction(
    *,
    request: RelationshipProductPreActionRequest,
    owner_persistence_snapshot: OwnerPersistenceSnapshot,
    gate_checkpoint: RelationshipActionGateCheckpoint | None,
    forecast_runtime: PreferenceActionForecastRuntime,
    gate_mode: RelationshipActionGateMode,
    authorization: RelationshipProductPulseAuthorization,
    substrate_snapshot: SubstrateSnapshot,
) -> RelationshipProductPreActionSnapshot:
    """Freeze owner forecast and expose one typed action before settlement."""

    if gate_mode is RelationshipActionGateMode.ORACLE:
        raise ValueError("relationship product preaction rejects oracle mode")
    if not isinstance(gate_mode, RelationshipActionGateMode):
        raise TypeError("gate_mode must be RelationshipActionGateMode")
    _validate_placeholder_substrate(substrate_snapshot)

    store = SocialRecordStore()
    store.hydrate_from_persistence(owner_persistence_snapshot)
    gate = RelationshipActionGate(checkpoint=gate_checkpoint)
    checkpoint_before = gate.export_checkpoint()
    owner_snapshot, forecast = await _publish_relationship_product_forecast(
        request=request,
        store=store,
        forecast_runtime=forecast_runtime,
    )
    decision = gate.decide(forecast, mode=gate_mode)
    advisory = authorize_relationship_product_pulse_advisory(
        decision,
        authorization=authorization,
    )
    temporal_snapshot = await _apply_typed_environment_advisory(
        advisory,
        substrate_snapshot=substrate_snapshot,
    )
    return RelationshipProductPreActionSnapshot(
        request=request,
        authorization_id=authorization.authorization_id,
        preference_snapshot=owner_snapshot,
        forecast=forecast,
        gate_decision=decision,
        temporal_snapshot=temporal_snapshot,
        owner_persistence_snapshot=store.export_persistence_snapshot(),
        gate_checkpoint_before=checkpoint_before,
        gate_checkpoint_after=gate.export_checkpoint(),
    )


async def prepare_relationship_product_frozen_preaction(
    *,
    request: RelationshipProductPreActionRequest,
    owner_persistence_snapshot: OwnerPersistenceSnapshot,
    forecast_runtime: PreferenceActionForecastRuntime,
    frozen_policy: RelationshipActionGateFrozenPolicy,
    executor_disposition: RelationshipProductExecutorDisposition,
    authorization: RelationshipProductFrozenPulseAuthorization,
    substrate_snapshot: SubstrateSnapshot,
) -> RelationshipProductFrozenPreActionSnapshot:
    """Publish one frozen-policy candidate and its executor-only treatment."""

    _validate_placeholder_substrate(substrate_snapshot)
    if not isinstance(frozen_policy, RelationshipActionGateFrozenPolicy):
        raise TypeError("frozen_policy must be RelationshipActionGateFrozenPolicy")
    if not isinstance(
        executor_disposition,
        RelationshipProductExecutorDisposition,
    ):
        raise TypeError(
            "executor_disposition must be RelationshipProductExecutorDisposition"
        )
    if not isinstance(
        authorization,
        RelationshipProductFrozenPulseAuthorization,
    ):
        raise TypeError(
            "authorization must be RelationshipProductFrozenPulseAuthorization"
        )
    authorization.validate_policy(frozen_policy)
    checkpoint_before = frozen_policy.checkpoint
    store = SocialRecordStore()
    store.hydrate_from_persistence(owner_persistence_snapshot)
    owner_snapshot, forecast = await _publish_relationship_product_forecast(
        request=request,
        store=store,
        forecast_runtime=forecast_runtime,
    )
    owner_persistence = store.export_persistence_snapshot()
    frozen_decision = frozen_policy.decide(forecast)
    command = RelationshipProductExecutorCommand(
        forecast=forecast,
        frozen_policy=frozen_policy,
        frozen_decision=frozen_decision,
        authorization=authorization,
        owner_prestate_sha256=social_record_store_persistence_sha256(
            owner_persistence
        ),
        executor_disposition=executor_disposition,
    )
    execution_receipt = await _execute_relationship_product_executor_command(
        command=command,
        substrate_snapshot=substrate_snapshot,
    )
    if frozen_policy.checkpoint != checkpoint_before:
        raise RuntimeError("frozen product preaction changed the gate checkpoint")
    return RelationshipProductFrozenPreActionSnapshot(
        request=request,
        preference_snapshot=owner_snapshot,
        forecast=forecast,
        execution_receipt=execution_receipt,
        owner_persistence_snapshot=owner_persistence,
    )


async def prepare_relationship_product_forced_collection_preaction(
    *,
    request: RelationshipProductPreActionRequest,
    owner_persistence_snapshot: OwnerPersistenceSnapshot,
    forecast_runtime: PreferenceActionForecastRuntime,
    frozen_policy: RelationshipActionGateFrozenPolicy,
    authorization: RelationshipProductForcedCollectionAuthorization,
    substrate_snapshot: SubstrateSnapshot,
) -> RelationshipProductForcedCollectionPreActionSnapshot:
    """Deliver one frozen schedule action while keeping theta0 unchanged."""

    _validate_placeholder_substrate(substrate_snapshot)
    if not isinstance(frozen_policy, RelationshipActionGateFrozenPolicy):
        raise TypeError("frozen_policy must be RelationshipActionGateFrozenPolicy")
    if not isinstance(
        authorization,
        RelationshipProductForcedCollectionAuthorization,
    ):
        raise TypeError(
            "authorization must be "
            "RelationshipProductForcedCollectionAuthorization"
        )
    authorization.validate_policy(frozen_policy)
    authorization.validate_decision_id(request.forecast_request.decision_id)
    theta0 = frozen_policy.theta0_artifact
    if theta0 is None or frozen_policy.checkpoint.update_count != 0:
        raise ValueError("forced collection requires a cold theta0 policy")

    store = SocialRecordStore()
    store.hydrate_from_persistence(owner_persistence_snapshot)
    owner_snapshot, forecast = await _publish_relationship_product_forecast(
        request=request,
        store=store,
        forecast_runtime=forecast_runtime,
    )
    owner_persistence = store.export_persistence_snapshot()
    gate = RelationshipActionGate.from_theta0(
        theta0,
        checkpoint=frozen_policy.checkpoint,
        random_seed=frozen_policy.random_seed,
    )
    checkpoint_before = gate.export_checkpoint()
    forced_exposure = gate.record_forced_exposure(
        forecast,
        forced_action_id=_forced_collection_action_id(
            forecast=forecast,
            role=authorization.forced_action_role,
        ),
        sequence_index=authorization.sequence_index,
    )
    if gate.export_checkpoint() != checkpoint_before:
        raise RuntimeError("forced collection changed the cold theta0 checkpoint")
    command = RelationshipProductForcedCollectionCommand(
        frozen_policy=frozen_policy,
        forced_exposure=forced_exposure,
        authorization=authorization,
        owner_prestate_sha256=social_record_store_persistence_sha256(
            owner_persistence
        ),
    )
    execution_receipt = (
        await _execute_relationship_product_forced_collection_command(
            command=command,
            substrate_snapshot=substrate_snapshot,
        )
    )
    if frozen_policy.checkpoint != checkpoint_before:
        raise RuntimeError("forced collection mutated its frozen policy")
    return RelationshipProductForcedCollectionPreActionSnapshot(
        request=request,
        preference_snapshot=owner_snapshot,
        forecast=forecast,
        execution_receipt=execution_receipt,
        owner_persistence_snapshot=owner_persistence,
    )


async def settle_relationship_product_pulse(
    *,
    preaction: RelationshipProductPreActionSnapshot,
    settlement_input: RelationshipProductSettlementInput,
) -> RelationshipProductSettlementSnapshot:
    """Settle one frozen preaction through owner PE and dedicated credit."""

    _validate_settlement_lineage(preaction, settlement_input)
    if (
        settlement_input.apply_credit_to_gate
        and preaction.gate_decision.mode is not RelationshipActionGateMode.LEARNED
    ):
        raise ValueError("gate credit can only be applied to a learned decision")

    store = SocialRecordStore()
    store.hydrate_from_persistence(preaction.owner_persistence_snapshot)
    gate = RelationshipActionGate(checkpoint=preaction.gate_checkpoint_after)
    owner_settlement = await _settle_relationship_product_owner_chain(
        request=preaction.request,
        forecast=preaction.forecast,
        store=store,
        settlement_input=settlement_input,
    )
    gate_update = (
        gate.observe_credit(owner_settlement.credit)
        if settlement_input.apply_credit_to_gate
        else None
    )
    return RelationshipProductSettlementSnapshot(
        preaction=preaction,
        external_outcome_snapshot=owner_settlement.external_outcome_snapshot,
        preference_snapshot=owner_settlement.preference_snapshot,
        social_prediction_error_snapshot=(
            owner_settlement.social_prediction_error_snapshot
        ),
        settlement=owner_settlement.settlement,
        credit=owner_settlement.credit,
        gate_update=gate_update,
        owner_persistence_snapshot=store.export_persistence_snapshot(),
        gate_checkpoint=gate.export_checkpoint(),
        credit_applied_to_gate=settlement_input.apply_credit_to_gate,
    )


async def settle_relationship_product_frozen_pulse(
    *,
    preaction: RelationshipProductFrozenPreActionSnapshot,
    settlement_input: RelationshipProductSettlementInput,
) -> RelationshipProductFrozenSettlementSnapshot:
    """Settle the executor-delivered action without mutating the frozen gate."""

    if not isinstance(preaction, RelationshipProductFrozenPreActionSnapshot):
        raise TypeError(
            "preaction must be RelationshipProductFrozenPreActionSnapshot"
        )
    if settlement_input.apply_credit_to_gate:
        raise ValueError("frozen product settlement cannot apply credit to gate")
    _validate_frozen_environment_settlement_input(settlement_input)
    _validate_settlement_lineage_values(
        request=preaction.request,
        forecast=preaction.forecast,
        actual_action_id=preaction.delivered_action_id,
        settlement_input=settlement_input,
    )
    checkpoint_before = preaction.frozen_policy.checkpoint
    store = _hydrate_validated_frozen_preaction_owner(preaction)
    owner_settlement = await _settle_relationship_product_owner_chain(
        request=preaction.request,
        forecast=preaction.forecast,
        store=store,
        settlement_input=settlement_input,
    )
    if preaction.frozen_policy.checkpoint != checkpoint_before:
        raise RuntimeError("frozen product settlement changed the gate checkpoint")
    return RelationshipProductFrozenSettlementSnapshot(
        preaction=preaction,
        settlement_input=settlement_input,
        external_outcome_snapshot=owner_settlement.external_outcome_snapshot,
        preference_snapshot=owner_settlement.preference_snapshot,
        social_prediction_error_snapshot=(
            owner_settlement.social_prediction_error_snapshot
        ),
        settlement=owner_settlement.settlement,
        credit=owner_settlement.credit,
        owner_persistence_snapshot=store.export_persistence_snapshot(),
    )


async def settle_relationship_product_forced_collection(
    *,
    preaction: RelationshipProductForcedCollectionPreActionSnapshot,
    settlement_input: RelationshipProductSettlementInput,
) -> RelationshipProductForcedCollectionSettlementSnapshot:
    """Settle one forced action to PE-credit without applying it online."""

    if not isinstance(
        preaction,
        RelationshipProductForcedCollectionPreActionSnapshot,
    ):
        raise TypeError(
            "preaction must be RelationshipProductForcedCollectionPreActionSnapshot"
        )
    if settlement_input.apply_credit_to_gate:
        raise ValueError("forced collection cannot apply credit to gate")
    _validate_immutable_environment_settlement_input(
        settlement_input,
        source="forced collection",
    )
    _validate_settlement_lineage_values(
        request=preaction.request,
        forecast=preaction.forecast,
        actual_action_id=preaction.delivered_action_id,
        settlement_input=settlement_input,
    )
    checkpoint_before = preaction.frozen_policy.checkpoint
    store = _hydrate_validated_collection_preaction_owner(preaction)
    owner_settlement = await _settle_relationship_product_owner_chain(
        request=preaction.request,
        forecast=preaction.forecast,
        store=store,
        settlement_input=settlement_input,
    )
    if preaction.frozen_policy.checkpoint != checkpoint_before:
        raise RuntimeError("forced collection settlement changed theta0")
    return RelationshipProductForcedCollectionSettlementSnapshot(
        preaction=preaction,
        settlement_input=settlement_input,
        external_outcome_snapshot=owner_settlement.external_outcome_snapshot,
        preference_snapshot=owner_settlement.preference_snapshot,
        social_prediction_error_snapshot=(
            owner_settlement.social_prediction_error_snapshot
        ),
        settlement=owner_settlement.settlement,
        credit=owner_settlement.credit,
        owner_persistence_snapshot=store.export_persistence_snapshot(),
    )


def _authorize_relationship_product_v2_pulse_advisory(
    *,
    frozen_policy: RelationshipActionGateV2FrozenPolicy,
    frozen_decision: RelationshipActionGateV2FrozenDecision,
    forecast: PreferenceActionForecast,
    pulse_authorization: RelationshipProductPulseAuthorization,
) -> TemporalActionAdvisoryProposal:
    """Authorize one replayed v2 advisory for the typed offline environment."""

    _validate_relationship_product_v2_policy_authorization(
        pulse_authorization,
        frozen_policy,
    )
    advisory = temporal_action_advisory_from_gate_v2_decision(
        frozen_decision,
        frozen_policy=frozen_policy,
        forecast=forecast,
    )
    return replace(
        advisory,
        active_authorized=True,
        evidence_refs=(
            *advisory.evidence_refs,
            f"lab-authorization:{pulse_authorization.authorization_id}",
        ),
        rationale_codes=(
            *advisory.rationale_codes,
            "scope:offline-reactive-environment-only",
            "expression:forbidden",
            "production:forbidden",
        ),
    )


def _authorize_relationship_product_v2_online_pulse_advisory(
    *,
    exposure: RelationshipActionGateV2OnlineExposure,
    session: RelationshipActionGateV2OnlineSession,
    authorization: RelationshipProductV2OnlinePulseAuthorization,
) -> TemporalActionAdvisoryProposal:
    """Authorize only the active session's exact pending online action."""

    if type(exposure) is not RelationshipActionGateV2OnlineExposure:
        raise TypeError("exposure must be RelationshipActionGateV2OnlineExposure")
    if type(authorization) is not RelationshipProductV2OnlinePulseAuthorization:
        raise TypeError("authorization must be RelationshipProductV2OnlinePulseAuthorization")
    authorization.validate_session(session)
    advisory = temporal_action_advisory_from_gate_v2_online_exposure(
        exposure,
        session=session,
    )
    return replace(
        advisory,
        active_authorized=True,
        evidence_refs=(
            *advisory.evidence_refs,
            f"lab-authorization:{authorization.pulse_authorization.authorization_id}",
            f"lab-online-authorization:{authorization.authorization_id}",
        ),
        rationale_codes=(
            *advisory.rationale_codes,
            "executor-disposition:apply-exact-online-candidate",
            "scope:offline-reactive-environment-only",
            "expression:forbidden",
            "production:forbidden",
        ),
    )


async def prepare_relationship_product_v2_online_preaction(
    *,
    request: RelationshipProductPreActionRequest,
    owner_persistence_snapshot: OwnerPersistenceSnapshot,
    forecast_runtime: PreferenceActionForecastRuntime,
    online_session: RelationshipActionGateV2OnlineSession,
    authorization: RelationshipProductV2OnlinePulseAuthorization,
    substrate_snapshot: SubstrateSnapshot,
    temporal_delivery_timestamp_ms: int | None = None,
) -> RelationshipProductV2OnlinePreActionSnapshot:
    """Close one logical online pre-action barrier without opening the source."""

    _validate_placeholder_substrate(substrate_snapshot)
    if type(authorization) is not RelationshipProductV2OnlinePulseAuthorization:
        raise TypeError("authorization must be RelationshipProductV2OnlinePulseAuthorization")
    authorization.validate_session(online_session)
    authorization.validate_request(request)
    if (
        online_session.pending_exposure is not None
        or online_session.pending_plan is not None
    ):
        raise ValueError("online session already has an unresolved preaction")
    if (
        temporal_delivery_timestamp_ms is not None
        and type(temporal_delivery_timestamp_ms) is not int
    ):
        raise TypeError("temporal_delivery_timestamp_ms must be an int or None")
    if (
        temporal_delivery_timestamp_ms is not None
        and temporal_delivery_timestamp_ms < 0
    ):
        raise ValueError("temporal_delivery_timestamp_ms must be non-negative")

    parent_chain_id = online_session.current_chain_id
    transition_count = online_session.transition_count
    checkpoint_before = online_session.export_checkpoint()
    store = SocialRecordStore()
    store.hydrate_from_persistence(owner_persistence_snapshot)
    owner_snapshot, forecast = await _publish_relationship_product_forecast(
        request=request,
        store=store,
        forecast_runtime=forecast_runtime,
    )
    if (
        online_session.current_chain_id != parent_chain_id
        or online_session.transition_count != transition_count
        or online_session.export_checkpoint() != checkpoint_before
        or online_session.pending_exposure is not None
        or online_session.pending_plan is not None
    ):
        raise RuntimeError("online session changed while the forecast owner was publishing")

    owner_persistence = store.export_persistence_snapshot()
    decision = online_session.decide(forecast)
    exposure = online_session.record_exposure(
        forecast,
        delivered_action_id=decision.decision.selected_action_id,
    )
    if (
        exposure.parent_chain_id != parent_chain_id
        or exposure.sequence_index != transition_count
        or exposure.frozen_decision.checkpoint_content_sha256
        != checkpoint_before.content_sha256
    ):
        raise RuntimeError("online exposure did not preserve the captured completed prefix")
    command = RelationshipProductV2OnlineExecutorCommand(
        online_exposure=exposure,
        authorization=authorization,
        owner_prestate_sha256=social_record_store_persistence_sha256(
            owner_persistence
        ),
    )
    execution_receipt = await _execute_relationship_product_v2_online_command(
        command=command,
        online_session=online_session,
        substrate_snapshot=substrate_snapshot,
        temporal_delivery_timestamp_ms=temporal_delivery_timestamp_ms,
    )
    return RelationshipProductV2OnlinePreActionSnapshot(
        request=request,
        owner_input_persistence_snapshot=owner_persistence_snapshot,
        preference_snapshot=owner_snapshot,
        forecast=forecast,
        execution_receipt=execution_receipt,
        owner_persistence_snapshot=owner_persistence,
    )


async def prepare_relationship_product_v2_frozen_preaction(
    *,
    request: RelationshipProductPreActionRequest,
    owner_persistence_snapshot: OwnerPersistenceSnapshot,
    forecast_runtime: PreferenceActionForecastRuntime,
    executor_disposition: RelationshipProductExecutorDisposition,
    authorization: (
        RelationshipProductV2FrozenPulseAuthorization
        | RelationshipProductV2CondensedTheta0FrozenPulseAuthorization
    ),
    substrate_snapshot: SubstrateSnapshot,
) -> RelationshipProductV2FrozenPreActionSnapshot:
    """Publish and deliver one policy-authorized v2 evaluation action."""

    _validate_placeholder_substrate(substrate_snapshot)
    if type(authorization) not in (
        RelationshipProductV2FrozenPulseAuthorization,
        RelationshipProductV2CondensedTheta0FrozenPulseAuthorization,
    ):
        raise TypeError("authorization must be an exact v2 evaluation authorization")
    if type(executor_disposition) is not RelationshipProductExecutorDisposition:
        raise TypeError("executor_disposition must be RelationshipProductExecutorDisposition")
    store = SocialRecordStore()
    store.hydrate_from_persistence(owner_persistence_snapshot)
    owner_snapshot, forecast = await _publish_relationship_product_forecast(
        request=request,
        store=store,
        forecast_runtime=forecast_runtime,
    )
    owner_persistence = store.export_persistence_snapshot()
    frozen_decision = authorization.frozen_policy.decide(forecast)
    command = RelationshipProductV2ExecutorCommand(
        forecast=forecast,
        frozen_decision=frozen_decision,
        authorization=authorization,
        owner_prestate_sha256=social_record_store_persistence_sha256(
            owner_persistence
        ),
        executor_disposition=executor_disposition,
    )
    execution_receipt = await _execute_relationship_product_v2_executor_command(
        command=command,
        substrate_snapshot=substrate_snapshot,
    )
    return RelationshipProductV2FrozenPreActionSnapshot(
        request=request,
        owner_input_persistence_snapshot=owner_persistence_snapshot,
        preference_snapshot=owner_snapshot,
        forecast=forecast,
        execution_receipt=execution_receipt,
        owner_persistence_snapshot=owner_persistence,
    )


async def prepare_relationship_product_v2_forced_collection_preaction(
    *,
    request: RelationshipProductPreActionRequest,
    owner_persistence_snapshot: OwnerPersistenceSnapshot,
    forecast_runtime: PreferenceActionForecastRuntime,
    authorization: RelationshipProductV2ForcedCollectionAuthorization,
    substrate_snapshot: SubstrateSnapshot,
    temporal_delivery_timestamp_ms: int | None = None,
) -> RelationshipProductV2ForcedCollectionPreActionSnapshot:
    """Publish and deliver one action derived from a full v2 assignment receipt."""

    _validate_placeholder_substrate(substrate_snapshot)
    if (
        temporal_delivery_timestamp_ms is not None
        and type(temporal_delivery_timestamp_ms) is not int
    ):
        raise TypeError("temporal_delivery_timestamp_ms must be an int or None")
    if (
        temporal_delivery_timestamp_ms is not None
        and temporal_delivery_timestamp_ms < 0
    ):
        raise ValueError("temporal_delivery_timestamp_ms must be non-negative")
    if type(authorization) is not RelationshipProductV2ForcedCollectionAuthorization:
        raise TypeError("authorization must be RelationshipProductV2ForcedCollectionAuthorization")
    authorization.validate_decision_id(request.forecast_request.decision_id)
    store = SocialRecordStore()
    store.hydrate_from_persistence(owner_persistence_snapshot)
    owner_snapshot, forecast = await _publish_relationship_product_forecast(
        request=request,
        store=store,
        forecast_runtime=forecast_runtime,
    )
    owner_persistence = store.export_persistence_snapshot()
    delivered_action_id = (
        forecast.recommended_action_id
        if authorization.assignment_role is RelationshipActionGateV2AssignmentRole.CANDIDATE
        else RelationshipAction.NEUTRAL_NOOP.value
    )
    gate = RelationshipActionGateV2(
        artifact=authorization.frozen_policy.artifact,
        checkpoint=authorization.frozen_policy.checkpoint,
    )
    checkpoint_before = gate.export_checkpoint()
    forced_exposure = gate.record_forced_exposure(
        forecast,
        assignment=authorization.assignment,
        delivered_action_id=delivered_action_id,
    )
    if gate.export_checkpoint() != checkpoint_before:
        raise RuntimeError("v2 forced exposure changed the cold gate")
    command = RelationshipProductV2ForcedCollectionCommand(
        forced_exposure=forced_exposure,
        authorization=authorization,
        owner_prestate_sha256=social_record_store_persistence_sha256(
            owner_persistence
        ),
    )
    execution_receipt = await _execute_relationship_product_v2_forced_command(
        command=command,
        substrate_snapshot=substrate_snapshot,
        temporal_delivery_timestamp_ms=temporal_delivery_timestamp_ms,
    )
    return RelationshipProductV2ForcedCollectionPreActionSnapshot(
        request=request,
        owner_input_persistence_snapshot=owner_persistence_snapshot,
        preference_snapshot=owner_snapshot,
        forecast=forecast,
        execution_receipt=execution_receipt,
        owner_persistence_snapshot=owner_persistence,
    )


async def settle_relationship_product_v2_online_pulse(
    *,
    preaction: RelationshipProductV2OnlinePreActionSnapshot,
    settlement_input: RelationshipProductSettlementInput,
    online_session: RelationshipActionGateV2OnlineSession,
) -> RelationshipProductV2OnlineSettlementSnapshot:
    """Settle one actual outcome and commit the prebound online disposition."""

    if type(preaction) is not RelationshipProductV2OnlinePreActionSnapshot:
        raise TypeError("preaction must be RelationshipProductV2OnlinePreActionSnapshot")
    preaction.authorization.validate_session(online_session)
    if online_session.pending_exposure != preaction.online_exposure:
        raise ValueError("online settlement does not match the active pending preaction")
    if online_session.pending_plan is not None:
        raise ValueError("online settlement found an externally sealed pending credit plan")
    owner_settlement, owner_persistence = (
        await _settle_relationship_product_v2_online_owner_chain(
            preaction=preaction,
            settlement_input=settlement_input,
            online_session=online_session,
        )
    )
    common_credit = _derive_relationship_product_v2_common_credit(
        preaction=preaction,
        settlement_input=settlement_input,
        owner_settlement=owner_settlement,
    )
    plan = online_session.plan_credit(preaction.online_exposure, common_credit)
    transition = online_session.commit_credit(plan)
    if online_session.current_chain_id == preaction.parent_chain_id:
        raise RuntimeError("online settlement did not append its transition to the chain")
    if (
        online_session.pending_exposure is not None
        or online_session.pending_plan is not None
    ):
        raise RuntimeError("online settlement left unexpected pending gate state")
    if online_session.transition_count != preaction.gate_transition_count_before + 1:
        raise RuntimeError("online settlement transition count did not advance exactly once")
    if online_session.export_checkpoint() != transition.terminal_checkpoint:
        raise RuntimeError("online settlement terminal checkpoint differs from live session")
    return RelationshipProductV2OnlineSettlementSnapshot(
        preaction=preaction,
        settlement_input=settlement_input,
        external_outcome_snapshot=owner_settlement.external_outcome_snapshot,
        preference_snapshot=owner_settlement.preference_snapshot,
        social_prediction_error_snapshot=(
            owner_settlement.social_prediction_error_snapshot
        ),
        settlement=owner_settlement.settlement,
        credit=owner_settlement.credit,
        common_baseline_credit=common_credit,
        owner_persistence_snapshot=owner_persistence,
        gate_transition=transition,
    )


async def settle_relationship_product_v2_frozen_pulse(
    *,
    preaction: RelationshipProductV2FrozenPreActionSnapshot,
    settlement_input: RelationshipProductSettlementInput,
) -> RelationshipProductV2FrozenSettlementSnapshot:
    """Settle the actual v2 evaluation action without online gate mutation."""

    if type(preaction) is not RelationshipProductV2FrozenPreActionSnapshot:
        raise TypeError("preaction must be RelationshipProductV2FrozenPreActionSnapshot")
    owner_settlement, owner_persistence = await _settle_relationship_product_v2_owner_chain(
        preaction=preaction,
        settlement_input=settlement_input,
        source="v2 frozen",
    )
    common_credit = _derive_relationship_product_v2_common_credit(
        preaction=preaction,
        settlement_input=settlement_input,
        owner_settlement=owner_settlement,
    )
    return RelationshipProductV2FrozenSettlementSnapshot(
        preaction=preaction,
        settlement_input=settlement_input,
        external_outcome_snapshot=owner_settlement.external_outcome_snapshot,
        preference_snapshot=owner_settlement.preference_snapshot,
        social_prediction_error_snapshot=owner_settlement.social_prediction_error_snapshot,
        settlement=owner_settlement.settlement,
        credit=owner_settlement.credit,
        common_baseline_credit=common_credit,
        owner_persistence_snapshot=owner_persistence,
    )


async def settle_relationship_product_v2_forced_collection(
    *,
    preaction: RelationshipProductV2ForcedCollectionPreActionSnapshot,
    settlement_input: RelationshipProductSettlementInput,
) -> RelationshipProductV2ForcedCollectionSettlementSnapshot:
    """Settle one actual forced action and publish its full common credit."""

    if type(preaction) is not RelationshipProductV2ForcedCollectionPreActionSnapshot:
        raise TypeError("preaction must be RelationshipProductV2ForcedCollectionPreActionSnapshot")
    owner_settlement, owner_persistence = await _settle_relationship_product_v2_owner_chain(
        preaction=preaction,
        settlement_input=settlement_input,
        source="v2 forced",
    )
    common_credit = _derive_relationship_product_v2_common_credit(
        preaction=preaction,
        settlement_input=settlement_input,
        owner_settlement=owner_settlement,
    )
    return RelationshipProductV2ForcedCollectionSettlementSnapshot(
        preaction=preaction,
        settlement_input=settlement_input,
        external_outcome_snapshot=owner_settlement.external_outcome_snapshot,
        preference_snapshot=owner_settlement.preference_snapshot,
        social_prediction_error_snapshot=owner_settlement.social_prediction_error_snapshot,
        settlement=owner_settlement.settlement,
        credit=owner_settlement.credit,
        common_baseline_credit=common_credit,
        owner_persistence_snapshot=owner_persistence,
    )


def build_relationship_product_v2_collected_credit_batch(
    settlements: tuple[RelationshipProductV2ForcedCollectionSettlementSnapshot, ...],
) -> RelationshipProductV2CollectedCreditBatch:
    """Retain complete actual-delivery settlements as pulse provenance."""

    return RelationshipProductV2CollectedCreditBatch(settlements=settlements)


def build_relationship_product_v2_segmented_collected_credit_batch(
    segments: tuple[RelationshipProductV2CollectionSegment, ...],
) -> RelationshipProductV2SegmentedCollectedCreditBatch:
    """Retain explicit owner segments under one complete gate schedule."""

    return RelationshipProductV2SegmentedCollectedCreditBatch(segments=segments)


def build_relationship_product_v2_federated_collected_credit_batch(
    *,
    federated_schedule_artifact: (
        RelationshipActionGateV2FederatedAssignmentScheduleArtifact
    ),
    child_collected_batches: tuple[
        RelationshipProductV2SegmentedCollectedCreditBatch,
        ...,
    ],
) -> RelationshipProductV2FederatedCollectedCreditBatch:
    """Join complete pulse children to an externally frozen parent order."""

    return RelationshipProductV2FederatedCollectedCreditBatch(
        federated_schedule_artifact=federated_schedule_artifact,
        child_collected_batches=child_collected_batches,
    )


def commit_relationship_product_v2_matched_gate_transitions(
    *,
    artifact: RelationshipActionGateV2Artifact,
    collected_batch: RelationshipProductV2CollectedCreditBatch,
) -> RelationshipProductV2MatchedGateTransitions:
    """Derive matched APPLY/WITHHOLD transitions from one pulse collection."""

    if type(artifact) is not RelationshipActionGateV2Artifact:
        raise TypeError("artifact must be RelationshipActionGateV2Artifact")
    if type(collected_batch) is not RelationshipProductV2CollectedCreditBatch:
        raise TypeError(
            "collected_batch must be RelationshipProductV2CollectedCreditBatch"
        )
    batch = collected_batch.gate_batch

    def _transition(
        disposition: RelationshipActionGateBatchDisposition,
    ) -> RelationshipProductV2GateTransition:
        gate = RelationshipActionGateV2(artifact=artifact)
        receipt = gate.commit_credit_batch(
            gate.plan_credit_batch(batch),
            disposition=disposition,
        )
        return RelationshipProductV2GateTransition(
            collected_batch=collected_batch,
            gate_receipt=receipt,
            frozen_policy=gate.freeze_for_evaluation(),
        )

    return RelationshipProductV2MatchedGateTransitions(
        applied=_transition(RelationshipActionGateBatchDisposition.APPLY),
        withheld=_transition(RelationshipActionGateBatchDisposition.WITHHOLD),
    )


def commit_relationship_product_v2_segmented_matched_gate_transitions(
    *,
    artifact: RelationshipActionGateV2Artifact,
    collected_batch: RelationshipProductV2SegmentedCollectedCreditBatch,
) -> RelationshipProductV2SegmentedMatchedGateTransitions:
    """Derive an add-only transition pair from one segmented collection."""

    if type(artifact) is not RelationshipActionGateV2Artifact:
        raise TypeError("artifact must be RelationshipActionGateV2Artifact")
    if (
        type(collected_batch)
        is not RelationshipProductV2SegmentedCollectedCreditBatch
    ):
        raise TypeError(
            "collected_batch must be RelationshipProductV2SegmentedCollectedCreditBatch"
        )
    batch = collected_batch.gate_batch

    def _transition(
        disposition: RelationshipActionGateBatchDisposition,
    ) -> RelationshipProductV2SegmentedGateTransition:
        gate = RelationshipActionGateV2(artifact=artifact)
        receipt = gate.commit_credit_batch(
            gate.plan_credit_batch(batch),
            disposition=disposition,
        )
        return RelationshipProductV2SegmentedGateTransition(
            collected_batch=collected_batch,
            gate_receipt=receipt,
            frozen_policy=gate.freeze_for_evaluation(),
        )

    return RelationshipProductV2SegmentedMatchedGateTransitions(
        applied=_transition(RelationshipActionGateBatchDisposition.APPLY),
        withheld=_transition(RelationshipActionGateBatchDisposition.WITHHOLD),
    )


def commit_relationship_product_v2_federated_matched_gate_transitions(
    *,
    artifact: RelationshipActionGateV2Artifact,
    collected_batch: RelationshipProductV2FederatedCollectedCreditBatch,
) -> RelationshipProductV2FederatedMatchedGateTransitions:
    """Commit one gate-owned parent pair while retaining pulse provenance."""

    if type(artifact) is not RelationshipActionGateV2Artifact:
        raise TypeError("artifact must be RelationshipActionGateV2Artifact")
    if (
        type(collected_batch)
        is not RelationshipProductV2FederatedCollectedCreditBatch
    ):
        raise TypeError(
            "collected_batch must be RelationshipProductV2FederatedCollectedCreditBatch"
        )
    collected_batch._assert_integrity()
    return RelationshipProductV2FederatedMatchedGateTransitions(
        collected_batch=collected_batch,
        gate_matched_transitions=(
            commit_relationship_action_gate_v2_federated_matched_transitions(
                artifact=artifact,
                batch=collected_batch.gate_batch,
            )
        ),
    )


async def _publish_relationship_product_forecast(
    *,
    request: RelationshipProductPreActionRequest,
    store: SocialRecordStore,
    forecast_runtime: PreferenceActionForecastRuntime,
) -> tuple[
    Snapshot[PreferenceAboutOtherSnapshot],
    PreferenceActionForecast,
]:
    owner_snapshot = await PreferenceAboutOtherModule(
        turn_index=request.forecast_request.turn_index,
        wiring_level=WiringLevel.SHADOW,
        record_store=store,
        action_forecast_runtime=forecast_runtime,
        action_forecast_request=request.forecast_request,
    ).process({})
    if not isinstance(owner_snapshot.value, PreferenceAboutOtherSnapshot):
        raise TypeError("preaction preference owner published unexpected snapshot")
    forecasts = tuple(
        forecast
        for forecast in owner_snapshot.value.action_forecasts
        if forecast.decision_id == request.forecast_request.decision_id
    )
    if len(forecasts) != 1:
        raise RuntimeError(
            "preaction preference owner must publish exactly one current forecast"
        )
    return owner_snapshot, forecasts[0]


async def _settle_relationship_product_owner_chain(
    *,
    request: RelationshipProductPreActionRequest,
    forecast: PreferenceActionForecast,
    store: SocialRecordStore,
    settlement_input: RelationshipProductSettlementInput,
) -> _RelationshipProductOwnerSettlement:
    # ``SocialRecordStore`` deliberately excludes one-turn ToM pending state
    # from its cross-session persistence contract. Preaction and settlement
    # are two phases of the same session, so the owner rebuilds that fast state
    # at the original action turn before receiving the actual outcome.
    replayed_preaction_owner = await PreferenceAboutOtherModule(
        turn_index=request.forecast_request.turn_index,
        wiring_level=WiringLevel.SHADOW,
        record_store=store,
    ).process({})
    if not isinstance(
        replayed_preaction_owner.value,
        PreferenceAboutOtherSnapshot,
    ):
        raise TypeError("preaction owner fast-state replay published unexpected snapshot")

    external_owner = DialogueExternalOutcomeModule(wiring_level=WiringLevel.ACTIVE)
    external_owner.set_turn_index(request.outcome_turn_index)
    external_owner.append_evidence(settlement_input.external_outcome)
    external_snapshot = await external_owner.process({})

    evidence = settlement_input.owner_outcome_evidence
    settlement_owner = PreferenceAboutOtherModule(
        proposal_runtime=_OutcomeEvidenceProposalRuntime(
            evidence=evidence,
            semantic_evidence_ref=settlement_input.external_outcome.evidence_ref,
        ),
        user_input=evidence.observation_summary,
        turn_index=evidence.source_turn,
        wiring_level=WiringLevel.SHADOW,
        record_store=store,
        action_outcome_evidence=evidence,
    )
    settled_snapshot = await settlement_owner.process(
        {"dialogue_external_outcome": external_snapshot}
    )
    if not isinstance(settled_snapshot.value, PreferenceAboutOtherSnapshot):
        raise TypeError("settlement preference owner published unexpected snapshot")
    current_settlements = tuple(
        settlement
        for settlement in settled_snapshot.value.forecast_settlements
        if settlement.forecast_id == forecast.forecast_id
        and settlement.observed_turn == request.outcome_turn_index
    )
    if len(current_settlements) != 1:
        raise RuntimeError(
            "relationship product pulse requires exactly one current settlement"
        )

    social_pe_snapshot = await SocialPredictionErrorModule(
        wiring_level=WiringLevel.ACTIVE
    ).process({"preference_about_other": settled_snapshot})
    if not isinstance(
        social_pe_snapshot.value,
        SocialPredictionErrorSnapshot,
    ):
        raise TypeError("social PE owner published unexpected snapshot")
    credits = derive_preference_action_forecast_credit_records(
        settlements=settled_snapshot.value.forecast_settlements,
        social_errors=social_pe_snapshot.value.errors,
        settled_at_turn=request.outcome_turn_index,
        timestamp_ms=settlement_input.credit_timestamp_ms,
    )
    matching_credits = tuple(
        credit for credit in credits if credit.prediction_id == forecast.forecast_id
    )
    if len(matching_credits) != 1:
        raise RuntimeError(
            "relationship product pulse must derive exactly one current PE credit"
        )
    return _RelationshipProductOwnerSettlement(
        external_outcome_snapshot=external_snapshot,
        preference_snapshot=settled_snapshot,
        social_prediction_error_snapshot=social_pe_snapshot,
        settlement=current_settlements[0],
        credit=matching_credits[0],
    )


async def _settle_relationship_product_v2_owner_chain(
    *,
    preaction: (
        RelationshipProductV2FrozenPreActionSnapshot
        | RelationshipProductV2ForcedCollectionPreActionSnapshot
    ),
    settlement_input: RelationshipProductSettlementInput,
    source: str,
) -> tuple[_RelationshipProductOwnerSettlement, OwnerPersistenceSnapshot]:
    if type(settlement_input) is not RelationshipProductSettlementInput:
        raise TypeError("settlement_input must be RelationshipProductSettlementInput")
    if settlement_input.apply_credit_to_gate:
        raise ValueError(f"{source} cannot apply credit online")
    _validate_immutable_environment_settlement_input(
        settlement_input,
        source=source,
    )
    if settlement_input.external_outcome.confidence != 1.0:
        raise ValueError(f"{source} common-baseline credit requires confidence 1.0")
    _validate_settlement_lineage_values(
        request=preaction.request,
        forecast=preaction.forecast,
        actual_action_id=preaction.delivered_action_id,
        settlement_input=settlement_input,
    )
    checkpoint_before = preaction.frozen_policy.checkpoint
    store = _hydrate_validated_immutable_preaction_owner(
        preaction,
        source=f"{source} preaction",
    )
    owner_settlement = await _settle_relationship_product_owner_chain(
        request=preaction.request,
        forecast=preaction.forecast,
        store=store,
        settlement_input=settlement_input,
    )
    if preaction.frozen_policy.checkpoint != checkpoint_before:
        raise RuntimeError(f"{source} changed the frozen v2 gate")
    return owner_settlement, store.export_persistence_snapshot()


async def _settle_relationship_product_v2_online_owner_chain(
    *,
    preaction: RelationshipProductV2OnlinePreActionSnapshot,
    settlement_input: RelationshipProductSettlementInput,
    online_session: RelationshipActionGateV2OnlineSession,
) -> tuple[_RelationshipProductOwnerSettlement, OwnerPersistenceSnapshot]:
    if type(settlement_input) is not RelationshipProductSettlementInput:
        raise TypeError("settlement_input must be RelationshipProductSettlementInput")
    if settlement_input.apply_credit_to_gate:
        raise ValueError(
            "v2 online settlement disposition is prebound; per-outcome apply bit is forbidden"
        )
    _validate_immutable_environment_settlement_input(
        settlement_input,
        source="v2 online",
    )
    if settlement_input.external_outcome.confidence != 1.0:
        raise ValueError("v2 online common-baseline credit requires confidence 1.0")
    _validate_relationship_product_v2_online_owner_evidence_projection(
        settlement_input
    )
    _validate_settlement_lineage_values(
        request=preaction.request,
        forecast=preaction.forecast,
        actual_action_id=preaction.delivered_action_id,
        settlement_input=settlement_input,
    )
    preaction.authorization.validate_session(online_session)
    exposure = preaction.online_exposure
    if online_session.pending_exposure != exposure:
        raise ValueError("v2 online owner settlement requires the exact pending exposure")
    if online_session.pending_plan is not None:
        raise ValueError("v2 online owner settlement found an externally sealed plan")
    if (
        online_session.current_chain_id != preaction.parent_chain_id
        or online_session.transition_count
        != preaction.gate_transition_count_before
        or online_session.export_checkpoint().content_sha256
        != preaction.gate_checkpoint_content_sha256_before
    ):
        raise ValueError("v2 online session completed prefix differs from preaction")

    store = _hydrate_validated_immutable_preaction_owner(
        preaction,
        source="v2 online preaction",
    )
    owner_settlement = await _settle_relationship_product_owner_chain(
        request=preaction.request,
        forecast=preaction.forecast,
        store=store,
        settlement_input=settlement_input,
    )
    preaction.authorization.validate_session(online_session)
    if (
        online_session.pending_exposure != exposure
        or online_session.pending_plan is not None
        or online_session.current_chain_id != preaction.parent_chain_id
        or online_session.transition_count
        != preaction.gate_transition_count_before
        or online_session.export_checkpoint().content_sha256
        != preaction.gate_checkpoint_content_sha256_before
    ):
        raise RuntimeError("v2 online session changed while owner settlement was publishing")
    return owner_settlement, store.export_persistence_snapshot()


def _derive_relationship_product_v2_common_credit(
    *,
    preaction: (
        RelationshipProductV2FrozenPreActionSnapshot
        | RelationshipProductV2OnlinePreActionSnapshot
        | RelationshipProductV2ForcedCollectionPreActionSnapshot
    ),
    settlement_input: RelationshipProductSettlementInput,
    owner_settlement: _RelationshipProductOwnerSettlement,
) -> RelationshipActionCommonBaselineCredit:
    expected_error_id = f"social-pe:{owner_settlement.settlement.settlement_id}"
    matching_errors = tuple(
        item
        for item in owner_settlement.social_prediction_error_snapshot.value.errors
        if item.error_id == expected_error_id
    )
    if len(matching_errors) != 1:
        raise RuntimeError("v2 settlement must publish exactly one current social PE")
    records = derive_preference_action_common_baseline_credit_records(
        forecasts=(preaction.forecast,),
        external_evidence=(settlement_input.external_outcome,),
        settlements=(owner_settlement.settlement,),
        social_errors=matching_errors,
        settled_at_turn=owner_settlement.settlement.observed_turn,
        timestamp_ms=settlement_input.credit_timestamp_ms,
    )
    if len(records) != 1:
        raise RuntimeError("v2 settlement must derive exactly one common-baseline credit")
    return records[0]


class _OutcomeEvidenceProposalRuntime(SemanticProposalRuntime):
    def __init__(
        self,
        *,
        evidence: PreferenceActionOutcomeEvidence,
        semantic_evidence_ref: str,
    ) -> None:
        self._evidence = evidence
        self._semantic_evidence_ref = semantic_evidence_ref
        self.runtime_id = (
            f"relationship-product-pulse-owner:{evidence.evidence_id}"
        )

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
            raise ValueError(
                "relationship product outcome runtime only serves "
                "preference_about_other"
            )
        if turn_index != self._evidence.source_turn:
            raise ValueError("relationship product outcome turn lineage mismatch")
        if user_input != self._evidence.observation_summary:
            raise ValueError("relationship product outcome input drifted")
        return SemanticProposalBatch(
            proposals=(
                SemanticProposal(
                    proposal_id=self._evidence.evidence_id,
                    target_slot=_PREFERENCE_SLOT,
                    operation=SemanticProposalOperation.OBSERVE,
                    summary=self._evidence.observation_summary,
                    detail=self._evidence.reaction_summary,
                    confidence=0.90,
                    evidence=self._semantic_evidence_ref,
                    control_signal=0.0,
                ),
            ),
            runtime_id=self.runtime_id,
            schema_version=1,
            description=(
                "One typed relationship product outcome proposed to its owner."
            ),
        )


async def _execute_relationship_product_executor_command(
    *,
    command: RelationshipProductExecutorCommand,
    substrate_snapshot: SubstrateSnapshot,
) -> RelationshipProductExecutorReceipt:
    if not isinstance(command, RelationshipProductExecutorCommand):
        raise TypeError("command must be RelationshipProductExecutorCommand")
    _validate_placeholder_substrate(substrate_snapshot)
    candidate_advisory = authorize_relationship_product_pulse_advisory(
        command.frozen_decision.decision,
        authorization=command.authorization.pulse_authorization,
    )
    delivered_advisory = _delivered_advisory_for_executor_command(
        command=command,
        candidate_advisory=candidate_advisory,
    )
    temporal_snapshot = await _apply_typed_environment_advisory(
        delivered_advisory,
        substrate_snapshot=substrate_snapshot,
    )
    temporal_delivery = RelationshipProductTemporalDelivery.from_snapshot(
        temporal_snapshot,
        delivered_advisory=delivered_advisory,
    )
    return RelationshipProductExecutorReceipt(
        command=command,
        candidate_advisory=candidate_advisory,
        delivered_advisory=delivered_advisory,
        temporal_delivery=temporal_delivery,
    )


async def _execute_relationship_product_forced_collection_command(
    *,
    command: RelationshipProductForcedCollectionCommand,
    substrate_snapshot: SubstrateSnapshot,
) -> RelationshipProductForcedCollectionReceipt:
    if not isinstance(command, RelationshipProductForcedCollectionCommand):
        raise TypeError(
            "command must be RelationshipProductForcedCollectionCommand"
        )
    _validate_placeholder_substrate(substrate_snapshot)
    candidate_advisory = authorize_relationship_product_pulse_advisory(
        command.frozen_decision.decision,
        authorization=(
            command.authorization.frozen_pulse_authorization.pulse_authorization
        ),
    )
    delivered_advisory = _delivered_advisory_for_forced_collection_command(
        command=command,
        candidate_advisory=candidate_advisory,
    )
    temporal_snapshot = await _apply_typed_environment_advisory(
        delivered_advisory,
        substrate_snapshot=substrate_snapshot,
    )
    temporal_delivery = RelationshipProductTemporalDelivery.from_snapshot(
        temporal_snapshot,
        delivered_advisory=delivered_advisory,
    )
    return RelationshipProductForcedCollectionReceipt(
        command=command,
        candidate_advisory=candidate_advisory,
        delivered_advisory=delivered_advisory,
        temporal_delivery=temporal_delivery,
    )


async def _execute_relationship_product_v2_executor_command(
    *,
    command: RelationshipProductV2ExecutorCommand,
    substrate_snapshot: SubstrateSnapshot,
) -> RelationshipProductV2ExecutorReceipt:
    if type(command) is not RelationshipProductV2ExecutorCommand:
        raise TypeError("command must be RelationshipProductV2ExecutorCommand")
    _validate_placeholder_substrate(substrate_snapshot)
    candidate_advisory = _authorize_relationship_product_v2_pulse_advisory(
        frozen_policy=command.frozen_policy,
        frozen_decision=command.frozen_decision,
        forecast=command.forecast,
        pulse_authorization=command.authorization.pulse_authorization,
    )
    delivered_advisory = _delivered_advisory_for_v2_executor_command(
        command=command,
        candidate_advisory=candidate_advisory,
    )
    temporal_snapshot = await _apply_typed_environment_advisory(
        delivered_advisory,
        substrate_snapshot=substrate_snapshot,
    )
    return RelationshipProductV2ExecutorReceipt(
        command=command,
        candidate_advisory=candidate_advisory,
        delivered_advisory=delivered_advisory,
        temporal_delivery=RelationshipProductTemporalDelivery.from_snapshot(
            temporal_snapshot,
            delivered_advisory=delivered_advisory,
        ),
    )


async def _execute_relationship_product_v2_online_command(
    *,
    command: RelationshipProductV2OnlineExecutorCommand,
    online_session: RelationshipActionGateV2OnlineSession,
    substrate_snapshot: SubstrateSnapshot,
    temporal_delivery_timestamp_ms: int | None,
) -> RelationshipProductV2OnlineExecutorReceipt:
    if type(command) is not RelationshipProductV2OnlineExecutorCommand:
        raise TypeError("command must be RelationshipProductV2OnlineExecutorCommand")
    command.authorization.validate_session(online_session)
    if online_session.pending_exposure != command.online_exposure:
        raise ValueError("online executor requires the exact active pending exposure")
    if online_session.pending_plan is not None:
        raise ValueError("online executor found an externally sealed pending credit plan")
    _validate_placeholder_substrate(substrate_snapshot)
    advisory = _authorize_relationship_product_v2_online_pulse_advisory(
        exposure=command.online_exposure,
        session=online_session,
        authorization=command.authorization,
    )
    temporal_snapshot = await _apply_typed_environment_advisory(
        advisory,
        substrate_snapshot=substrate_snapshot,
        publication_timestamp_ms=temporal_delivery_timestamp_ms,
    )
    command.authorization.validate_session(online_session)
    exposure = command.online_exposure
    if (
        online_session.pending_exposure != exposure
        or online_session.pending_plan is not None
        or online_session.current_chain_id != exposure.parent_chain_id
        or online_session.transition_count != exposure.sequence_index
        or online_session.export_checkpoint().content_sha256
        != exposure.frozen_decision.checkpoint_content_sha256
    ):
        raise RuntimeError("online session changed during temporal delivery")
    return RelationshipProductV2OnlineExecutorReceipt(
        command=command,
        authorized_advisory=advisory,
        temporal_delivery=RelationshipProductTemporalDelivery.from_snapshot(
            temporal_snapshot,
            delivered_advisory=advisory,
        ),
    )


async def _execute_relationship_product_v2_forced_command(
    *,
    command: RelationshipProductV2ForcedCollectionCommand,
    substrate_snapshot: SubstrateSnapshot,
    temporal_delivery_timestamp_ms: int | None,
) -> RelationshipProductV2ForcedCollectionReceipt:
    if type(command) is not RelationshipProductV2ForcedCollectionCommand:
        raise TypeError("command must be RelationshipProductV2ForcedCollectionCommand")
    _validate_placeholder_substrate(substrate_snapshot)
    candidate_advisory = _authorize_relationship_product_v2_pulse_advisory(
        frozen_policy=command.frozen_policy,
        frozen_decision=command.frozen_decision,
        forecast=command.forecast,
        pulse_authorization=command.authorization.pulse_authorization,
    )
    delivered_advisory = _delivered_advisory_for_v2_forced_command(
        command=command,
        candidate_advisory=candidate_advisory,
    )
    temporal_snapshot = await _apply_typed_environment_advisory(
        delivered_advisory,
        substrate_snapshot=substrate_snapshot,
        publication_timestamp_ms=temporal_delivery_timestamp_ms,
    )
    return RelationshipProductV2ForcedCollectionReceipt(
        command=command,
        candidate_advisory=candidate_advisory,
        delivered_advisory=delivered_advisory,
        temporal_delivery=RelationshipProductTemporalDelivery.from_snapshot(
            temporal_snapshot,
            delivered_advisory=delivered_advisory,
        ),
    )


def _delivered_advisory_for_v2_executor_command(
    *,
    command: RelationshipProductV2ExecutorCommand,
    candidate_advisory: TemporalActionAdvisoryProposal,
) -> TemporalActionAdvisoryProposal:
    if command.executor_disposition is RelationshipProductExecutorDisposition.APPLY_CANDIDATE:
        return candidate_advisory
    command_digest = hashlib.sha256(command.command_id.encode("utf-8")).hexdigest()
    return replace(
        candidate_advisory,
        advisory_id=f"relationship-product-v2-strict-noop-advisory:{command_digest}",
        action_id=RelationshipAction.NEUTRAL_NOOP.value,
        evidence_refs=(
            *candidate_advisory.evidence_refs,
            f"v2-executor-command:{command.command_id}",
        ),
        rationale_codes=(
            *candidate_advisory.rationale_codes,
            "executor-disposition:force-strict-noop",
        ),
    )


def _delivered_advisory_for_v2_forced_command(
    *,
    command: RelationshipProductV2ForcedCollectionCommand,
    candidate_advisory: TemporalActionAdvisoryProposal,
) -> TemporalActionAdvisoryProposal:
    command_digest = hashlib.sha256(command.command_id.encode("utf-8")).hexdigest()
    return replace(
        candidate_advisory,
        advisory_id=f"relationship-product-v2-forced-advisory:{command_digest}",
        action_id=command.delivered_action_id,
        evidence_refs=(
            *candidate_advisory.evidence_refs,
            f"v2-forced-command:{command.command_id}",
            f"v2-assignment:{command.authorization.assignment.assignment_id}",
        ),
        rationale_codes=(
            *candidate_advisory.rationale_codes,
            f"forced-assignment-role:{command.authorization.assignment_role.value}",
            "collection-only:no-online-gate-update",
        ),
    )


def _forced_collection_action_id(
    *,
    forecast: PreferenceActionForecast,
    role: RelationshipProductForcedActionRole,
) -> str:
    if not isinstance(forecast, PreferenceActionForecast):
        raise TypeError("forecast must be PreferenceActionForecast")
    if role is RelationshipProductForcedActionRole.OWNER_RECOMMENDATION:
        return forecast.recommended_action_id
    if role is RelationshipProductForcedActionRole.NEUTRAL_NOOP:
        return RelationshipAction.NEUTRAL_NOOP.value
    raise TypeError("role must be RelationshipProductForcedActionRole")


def _delivered_advisory_for_forced_collection_command(
    *,
    command: RelationshipProductForcedCollectionCommand,
    candidate_advisory: TemporalActionAdvisoryProposal,
) -> TemporalActionAdvisoryProposal:
    command_digest = hashlib.sha256(command.command_id.encode("utf-8")).hexdigest()
    return replace(
        candidate_advisory,
        advisory_id=(
            "relationship-product-forced-collection-advisory:"
            f"{command_digest}"
        ),
        action_id=command.forced_action_id,
        evidence_refs=(
            *candidate_advisory.evidence_refs,
            f"forced-collection-command:{command.command_id}",
            (
                "forced-action-schedule-entry:"
                f"{command.authorization.schedule_entry_id}"
            ),
        ),
        rationale_codes=(
            *candidate_advisory.rationale_codes,
            f"forced-action-role:{command.forced_action_role.value}",
            "collection-only:no-online-gate-update",
        ),
    )


def _delivered_advisory_for_executor_command(
    *,
    command: RelationshipProductExecutorCommand,
    candidate_advisory: TemporalActionAdvisoryProposal,
) -> TemporalActionAdvisoryProposal:
    if (
        command.executor_disposition
        is RelationshipProductExecutorDisposition.APPLY_CANDIDATE
    ):
        return candidate_advisory
    command_digest = hashlib.sha256(command.command_id.encode("utf-8")).hexdigest()
    return replace(
        candidate_advisory,
        advisory_id=f"relationship-product-strict-noop-advisory:{command_digest}",
        action_id=RelationshipAction.NEUTRAL_NOOP.value,
        evidence_refs=(
            *candidate_advisory.evidence_refs,
            f"executor-command:{command.command_id}",
        ),
        rationale_codes=(
            *candidate_advisory.rationale_codes,
            "executor-disposition:force-strict-noop",
        ),
    )


async def _apply_typed_environment_advisory(
    advisory: TemporalActionAdvisoryProposal,
    *,
    substrate_snapshot: SubstrateSnapshot,
    publication_timestamp_ms: int | None = None,
) -> Snapshot[TemporalAbstractionSnapshot]:
    module = TrackTemporalModule(
        track=Track.SELF,
        policy=PlaceholderTemporalPolicy(),
        wiring_level=WiringLevel.ACTIVE,
        action_advisory=advisory,
        action_advisory_level=WiringLevel.ACTIVE,
    )
    snapshot = await module.process_standalone(
        substrate_snapshot=substrate_snapshot,
        publication_timestamp_ms=publication_timestamp_ms,
    )
    if snapshot.value.action_advisory_status is not TemporalActionAdvisoryStatus.APPLIED:
        raise RuntimeError("relationship product typed advisory was not applied")
    return snapshot


def _validate_placeholder_substrate(snapshot: SubstrateSnapshot) -> None:
    if not isinstance(snapshot, SubstrateSnapshot):
        raise TypeError("substrate_snapshot must be a SubstrateSnapshot")
    if not snapshot.is_frozen:
        raise ValueError("relationship product pulse requires a frozen substrate")
    if snapshot.surface_kind is not SurfaceKind.PLACEHOLDER:
        raise ValueError(
            "relationship product pulse is typed simulation only and requires "
            "a placeholder substrate"
        )


def _validate_settlement_lineage(
    preaction: RelationshipProductPreActionSnapshot,
    settlement_input: RelationshipProductSettlementInput,
) -> None:
    _validate_settlement_lineage_values(
        request=preaction.request,
        forecast=preaction.forecast,
        actual_action_id=preaction.gate_decision.selected_action_id,
        settlement_input=settlement_input,
    )


def _validate_settlement_lineage_values(
    *,
    request: RelationshipProductPreActionRequest,
    forecast: PreferenceActionForecast,
    actual_action_id: str,
    settlement_input: RelationshipProductSettlementInput,
) -> None:
    external = settlement_input.external_outcome
    owner_evidence = settlement_input.owner_outcome_evidence
    expected = (
        ("external outcome turn", external.turn_index, request.outcome_turn_index),
        (
            "external action turn",
            external.action_turn_index,
            request.forecast_request.turn_index,
        ),
        (
            "external session",
            external.session_scope,
            request.forecast_request.session_scope,
        ),
        ("external forecast", external.forecast_id, forecast.forecast_id),
        (
            "external decision",
            external.decision_id,
            request.forecast_request.decision_id,
        ),
        (
            "external action",
            external.action_id,
            actual_action_id,
        ),
        ("owner outcome turn", owner_evidence.source_turn, request.outcome_turn_index),
        (
            "owner interlocutor",
            owner_evidence.interlocutor_id,
            request.forecast_request.interlocutor_id,
        ),
        (
            "owner observation",
            owner_evidence.observation_summary,
            request.forecast_request.current_observation,
        ),
        (
            "owner action",
            owner_evidence.action_id,
            actual_action_id,
        ),
        (
            "owner typed outcome",
            owner_evidence.observed_outcome_id,
            external.kind.value,
        ),
    )
    for field_name, observed, wanted in expected:
        if observed != wanted:
            raise ValueError(f"{field_name} lineage mismatch")
    if external.evidence_ref not in owner_evidence.evidence_refs:
        raise ValueError("owner outcome does not cite external evidence")


def _pulse_authorization_to_payload(
    authorization: RelationshipProductPulseAuthorization,
) -> dict[str, object]:
    return {
        "authorization_id": authorization.authorization_id,
        "allowed_policy_artifact_id": authorization.allowed_policy_artifact_id,
        "allowed_policy_artifact_version": (
            authorization.allowed_policy_artifact_version
        ),
        "environment_consumer_only": authorization.environment_consumer_only,
        "expression_authorized": authorization.expression_authorized,
        "production_authorized": authorization.production_authorized,
        "oracle_action_authorized": authorization.oracle_action_authorized,
    }


def _temporal_advisory_to_payload(
    advisory: TemporalActionAdvisoryProposal,
) -> dict[str, object]:
    return {
        "advisory_id": advisory.advisory_id,
        "decision_id": advisory.decision_id,
        "prediction_id": advisory.prediction_id,
        "action_id": advisory.action_id,
        "confidence": advisory.confidence,
        "policy_artifact_id": advisory.policy_artifact_id,
        "policy_artifact_version": advisory.policy_artifact_version,
        "evidence_refs": list(advisory.evidence_refs),
        "rationale_codes": list(advisory.rationale_codes),
        "evaluator_only": advisory.evaluator_only,
        "active_authorized": advisory.active_authorized,
    }


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: str, field_name: str) -> None:
    _require_text(value, field_name)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a canonical lowercase SHA-256")


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


__all__ = [
    "RELATIONSHIP_PRODUCT_EXECUTOR_SCHEMA_VERSION",
    "RELATIONSHIP_PRODUCT_FORCED_COLLECTION_SCHEMA_VERSION",
    "RELATIONSHIP_PRODUCT_FORCED_COLLECTION_SCHEDULE_SCHEMA_VERSION",
    "RELATIONSHIP_PRODUCT_FROZEN_PULSE_SCHEMA_VERSION",
    "RELATIONSHIP_PRODUCT_PULSE_SCHEMA_VERSION",
    "RELATIONSHIP_PRODUCT_V2_EXECUTOR_SCHEMA_VERSION",
    "RELATIONSHIP_PRODUCT_V2_CONDENSED_THETA0_AUTHORIZATION_SCHEMA_VERSION",
    "RELATIONSHIP_PRODUCT_V2_COLLECTED_BATCH_SCHEMA_VERSION",
    "RELATIONSHIP_PRODUCT_V2_COLLECTION_SEGMENT_SCHEMA_VERSION",
    "RELATIONSHIP_PRODUCT_V2_FEDERATED_COLLECTED_BATCH_SCHEMA_VERSION",
    "RELATIONSHIP_PRODUCT_V2_FEDERATED_MATCHED_TRANSITIONS_SCHEMA_VERSION",
    "RELATIONSHIP_PRODUCT_V2_FORCED_COLLECTION_SCHEMA_VERSION",
    "RELATIONSHIP_PRODUCT_V2_FROZEN_PULSE_SCHEMA_VERSION",
    "RELATIONSHIP_PRODUCT_V2_GATE_TRANSITION_SCHEMA_VERSION",
    "RELATIONSHIP_PRODUCT_V2_MATCHED_TRANSITIONS_SCHEMA_VERSION",
    "RELATIONSHIP_PRODUCT_V2_SEGMENTED_COLLECTED_BATCH_SCHEMA_VERSION",
    "RELATIONSHIP_PRODUCT_V2_SEGMENTED_GATE_TRANSITION_SCHEMA_VERSION",
    "RELATIONSHIP_PRODUCT_V2_SEGMENTED_MATCHED_TRANSITIONS_SCHEMA_VERSION",
    "RelationshipProductExecutorCommand",
    "RelationshipProductExecutorDisposition",
    "RelationshipProductExecutorReceipt",
    "RelationshipProductExecutorStatus",
    "RelationshipProductForcedActionRole",
    "RelationshipProductForcedCollectionAuthorization",
    "RelationshipProductForcedCollectionCommand",
    "RelationshipProductForcedCollectionPreActionSnapshot",
    "RelationshipProductForcedCollectionReceipt",
    "RelationshipProductForcedCollectionScheduleArtifact",
    "RelationshipProductForcedCollectionScheduleEntry",
    "RelationshipProductForcedCollectionSettlementSnapshot",
    "RelationshipProductFrozenPreActionSnapshot",
    "RelationshipProductFrozenPulseAuthorization",
    "RelationshipProductFrozenSettlementSnapshot",
    "RelationshipProductOnboardingInput",
    "RelationshipProductOnboardingSnapshot",
    "RelationshipProductPreActionRequest",
    "RelationshipProductPreActionSnapshot",
    "RelationshipProductPulseAuthorization",
    "RelationshipProductSettlementInput",
    "RelationshipProductSettlementSnapshot",
    "RelationshipProductTemporalDelivery",
    "RelationshipProductV2CollectedCreditBatch",
    "RelationshipProductV2CollectionSegment",
    "RelationshipProductV2CondensedTheta0FrozenPulseAuthorization",
    "RelationshipProductV2ExecutorCommand",
    "RelationshipProductV2ExecutorReceipt",
    "RelationshipProductV2FederatedCollectedCreditBatch",
    "RelationshipProductV2FederatedMatchedGateTransitions",
    "RelationshipProductV2ForcedCollectionAuthorization",
    "RelationshipProductV2ForcedCollectionCommand",
    "RelationshipProductV2ForcedCollectionPreActionSnapshot",
    "RelationshipProductV2ForcedCollectionReceipt",
    "RelationshipProductV2ForcedCollectionSettlementSnapshot",
    "RelationshipProductV2FrozenPreActionSnapshot",
    "RelationshipProductV2FrozenPulseAuthorization",
    "RelationshipProductV2FrozenSettlementSnapshot",
    "RelationshipProductV2GateTransition",
    "RelationshipProductV2MatchedGateTransitions",
    "RelationshipProductV2OnlineExecutorCommand",
    "RelationshipProductV2OnlineExecutorReceipt",
    "RelationshipProductV2OnlinePreActionSnapshot",
    "RelationshipProductV2OnlinePulseAuthorization",
    "RelationshipProductV2OnlineSettlementSnapshot",
    "RelationshipProductV2SegmentedCollectedCreditBatch",
    "RelationshipProductV2SegmentedGateTransition",
    "RelationshipProductV2SegmentedMatchedGateTransitions",
    "append_relationship_product_onboarding",
    "authorize_relationship_product_pulse_advisory",
    "build_relationship_product_v2_collected_credit_batch",
    "build_relationship_product_v2_federated_collected_credit_batch",
    "build_relationship_product_v2_segmented_collected_credit_batch",
    "commit_relationship_product_v2_federated_matched_gate_transitions",
    "commit_relationship_product_v2_matched_gate_transitions",
    "commit_relationship_product_v2_segmented_matched_gate_transitions",
    "prepare_relationship_product_forced_collection_preaction",
    "prepare_relationship_product_frozen_preaction",
    "prepare_relationship_product_preaction",
    "prepare_relationship_product_v2_forced_collection_preaction",
    "prepare_relationship_product_v2_frozen_preaction",
    "prepare_relationship_product_v2_online_preaction",
    "settle_relationship_product_frozen_pulse",
    "settle_relationship_product_forced_collection",
    "settle_relationship_product_pulse",
    "settle_relationship_product_v2_forced_collection",
    "settle_relationship_product_v2_frozen_pulse",
    "settle_relationship_product_v2_online_pulse",
]
